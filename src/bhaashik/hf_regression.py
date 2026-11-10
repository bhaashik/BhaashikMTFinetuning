"""Optional local HF-only behavior probes for original/adapted OneNMT checkpoints.

No architecture conversion, remote code execution, vocabulary resizing or training.
"""
from collections import Counter, defaultdict
import math
import time
from pathlib import Path
from .data import file_hash, normalized, read_jsonl, write_json
from .evaluation import metrics
from .preflight import AssetError, FairseqCodec, checkpoint_layout, coverage, load_probes, preflight, protect_outputs


def output_signals(source, prediction, token_ids, codec, max_new_tokens=None):
    words = prediction.split()
    trigrams = Counter(tuple(words[i:i+3]) for i in range(max(len(words)-2, 0)))
    return {
        "empty": not prediction.strip(),
        "unknown_token": codec.special["unk"] in token_ids,
        "replacement_character": "\ufffd" in prediction,
        "source_copy": normalized(source) == normalized(prediction),
        "repeated_trigram": bool(trigrams) and max(trigrams.values()) >= 3,
        "generation_limit_reached": bool(token_ids) and max_new_tokens is not None and len(token_ids)-1 >= max_new_tokens and token_ids[-1] != codec.special["eos"],
    }


def generate_probes(assets, rows, *, device="cpu", precision="fp32", max_new_tokens=64, batch_size=1):
    import torch
    from transformers import GenerationConfig, MBartForConditionalGeneration, LogitsProcessor, LogitsProcessorList
    if device not in ("cpu", "cuda") or precision not in ("fp32", "fp16"):
        raise AssetError("Use cpu/cuda and fp32/fp16")
    if precision == "fp16" and device != "cuda":
        raise AssetError("fp16 probes require CUDA")
    if device == "cuda" and not torch.cuda.is_available():
        raise AssetError("CUDA was requested but is unavailable")
    if max_new_tokens < 1 or batch_size < 1:
        raise AssetError("max_new_tokens and batch_size must be positive")
    codec = FairseqCodec(assets)
    if codec.config.get("model_type") != "mbart":
        raise AssetError("HF-only regression runner expects the supplied MBart-format checkpoint")
    # Avoid pickle-backed loading even when a directory happens to contain .bin/.pt files.
    layout = checkpoint_layout(codec)
    if layout["issues"]:
        raise AssetError("Checkpoint layout issues: " + "; ".join(layout["issues"]))
    if not layout["weight_layout_checked"]:
        raise AssetError("HF regression requires local safetensors weights")
    cov, _ = coverage(codec, rows)
    if cov["source_token_limit"] is None:
        raise AssetError("HF regression requires config.max_position_embeddings")
    if any(g["source_context_overflows"] for g in cov["groups"].values()):
        raise AssetError("Source probes exceed the model context; shorten probes explicitly")
    if max_new_tokens + 1 > cov["source_token_limit"]:
        raise AssetError("Decoder start plus max_new_tokens exceeds positional context")
    model = MBartForConditionalGeneration.from_pretrained(
        str(assets), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float16 if precision == "fp16" else torch.float32)
    model.to(device).eval()
    # A fresh config prevents candidate-specific saved decoding defaults from
    # masquerading as tokenizer drift or forgetting.
    generation = GenerationConfig(
        decoder_start_token_id=codec.special["eos"], bos_token_id=codec.special["bos"],
        eos_token_id=codec.special["eos"], pad_token_id=codec.special["pad"],
        forced_bos_token_id=None, forced_eos_token_id=None,
        do_sample=False, num_beams=1, num_return_sequences=1,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
        min_length=0, min_new_tokens=0, max_new_tokens=max_new_tokens,
        remove_invalid_values=False, renormalize_logits=False)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Check raw next-token scores before generation processors introduce -inf masks.
    class FiniteScores(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                raise AssetError("Nonfinite model logits during generation")
            return scores
    outputs = []
    started = time.time()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start+batch_size]
                sequences = [codec.encode_source(r["source"], r["src_lang"], r["tgt_lang"]) for r in batch]
                length = max(map(len, sequences))
                ids = torch.full((len(batch), length), codec.special["pad"], dtype=torch.long, device=device)
                attention = torch.zeros_like(ids)
                for i, sequence in enumerate(sequences):
                    ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
                    attention[i, :len(sequence)] = 1
                generated = model.generate(
                    input_ids=ids, attention_mask=attention,
                    generation_config=generation,
                    logits_processor=LogitsProcessorList([FiniteScores()]))
                for row, tokens in zip(batch, generated.tolist()):
                    prediction = codec.decode(tokens)
                    outputs.append({**row, "prediction": prediction,
                                    "signals": output_signals(row["source"], prediction, tokens, codec, max_new_tokens)})
        metadata = {"assets": codec.fingerprint(), "seconds": time.time()-started,
                    "device": device, "precision": precision, "decoding": {
                        "num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens,
                        "decoder_start_token_id": codec.special["eos"], "forced_bos_token_id": None,
                        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
                    "torch": torch.__version__, "transformers": __import__('transformers').__version__}
        return outputs, metadata
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def compare_outputs(baseline, candidate, max_retention_drop=2.0, reference=None):
    if not math.isfinite(max_retention_drop) or max_retention_drop < 0:
        raise AssetError("max_retention_drop must be finite and nonnegative")
    old, new = {r["id"]: r for r in baseline}, {r["id"]: r for r in candidate}
    if len(old) != len(baseline) or len(new) != len(candidate) or set(old) != set(new) or not old:
        raise AssetError("Baseline and candidate must have unique, identical nonempty ID sets")
    for idx in old:
        if any(old[idx].get(k) != new[idx].get(k) for k in ("source", "target", "src_lang", "tgt_lang", "group", "document_id", "domain")):
            raise AssetError(f"Probe mismatch: {idx}")
    blocks, reviews, groups = [], [], defaultdict(list)
    for idx, row in old.items():
        groups[(row.get("group", "unspecified"), row.get("src_lang"), row.get("tgt_lang"))].append(idx)
    result = {"groups": {}, "thresholds": {"max_retention_chrfpp_drop": max_retention_drop},
              "reference_parity": "unverified; no independently produced reference predictions supplied"}
    for (group, src, tgt), indices in groups.items():
        refs = [old[i]["target"] for i in indices]
        a = metrics([old[i]["prediction"] for i in indices], refs)
        b = metrics([new[i]["prediction"] for i in indices], refs)
        delta = b["chrfpp"] - a["chrfpp"]
        counts = {side: {key: sum(bool(data[i].get("signals", {}).get(key)) for i in indices)
                         for key in ("empty", "unknown_token", "replacement_character", "source_copy", "repeated_trigram", "generation_limit_reached")}
                  for side, data in (("baseline", old), ("candidate", new))}
        label = f"{group}|{src}->{tgt}"
        result["groups"][label] = {"n": len(indices), "baseline": a, "candidate": b,
                "chrfpp_delta": delta, "changed_outputs": sum(old[i]["prediction"] != new[i]["prediction"] for i in indices),
                "triage_counts": counts}
        if group == "retention" and delta < -max_retention_drop:
            blocks.append(f"{label}: chrF++ decline exceeds the configured retention budget")
        for signal in ("empty", "unknown_token", "replacement_character", "repeated_trigram", "generation_limit_reached"):
            if counts["candidate"][signal] > counts["baseline"][signal]:
                reviews.append(f"{label}: increased {signal} outputs; inspect manually")
        if counts["baseline"]["empty"]:
            reviews.append(f"{label}: baseline contains empty outputs; do not assume HF conversion works")
    if not any(k[0] == "retention" for k in groups):
        reviews.append("No retention group: forgetting in old languages has not been assessed")
    if reference is not None:
        gold = {r["id"]: r for r in reference}
        if len(gold) != len(reference) or set(gold) != set(old):
            raise AssetError("Reference predictions must match all probe IDs uniquely")
        for idx in old:
            if any(old[idx].get(k) != gold[idx].get(k) for k in ("source", "target")):
                raise AssetError(f"Independent reference content mismatch: {idx}")
        mismatches = [i for i in old if old[i]["prediction"] != gold[i]["prediction"]]
        result["reference_parity"] = {"exact_output_matches": len(old)-len(mismatches), "n": len(old),
                                      "mismatch_examples": mismatches[:20],
                                      "scope": "decoded text only; not logits or architecture equivalence"}
        if mismatches:
            blocks.append("Baseline differs from supplied reference outputs; verify identical assets, prefixes and decoding before adaptation")
    else:
        reviews.append("Independent reference parity is unverified; two equally incorrect HF runs can agree")
    result["blocking_findings"], result["review_findings"] = blocks, reviews
    result["status"] = "blocked" if blocks else ("review_required" if reviews else "probe_checks_passed")
    result["limits"] = "Small-sample chrF++ and output heuristics are triage, not proof of language support, safety from forgetting, linguistic quality or architecture equivalence. Changed outputs can be legitimate improvements."
    return result


def hf_regression(baseline_assets, candidate_assets, corpus, output, src=None, tgt=None,
                  reference=None, max_retention_drop=2.0, device="cpu", precision="fp32",
                  max_new_tokens=64, batch_size=1):
    protect_outputs([baseline_assets, candidate_assets], [corpus, reference], [output])
    if not math.isfinite(max_retention_drop) or max_retention_drop < 0:
        raise AssetError("max_retention_drop must be finite and nonnegative")
    rows = load_probes(corpus, src, tgt)
    result = {"schema_version": 1, "kind": "onenmt_hf_regression", "corpus_sha256": file_hash(corpus)}
    structure = preflight(baseline_assets, corpus, None, src, tgt, candidate_assets=candidate_assets)
    result["structural_preflight"] = structure
    if structure["blocking_findings"]:
        result["status"] = "blocked"
        result["blocking_findings"] = structure["blocking_findings"]
        result["review_findings"] = structure["review_findings"]
        write_json(output, result)
        return result
    for label, assets in (("baseline", baseline_assets), ("candidate", candidate_assets)):
        try:
            outputs, metadata = generate_probes(assets, rows, device=device, precision=precision,
                                  max_new_tokens=max_new_tokens, batch_size=batch_size)
            result[label] = {"generation": metadata, "outputs": outputs}
        except (AssetError, OSError, RuntimeError, ValueError) as error:
            result["status"] = "blocked"
            result["blocking_findings"] = [f"{label} generation failed: {error}"]
            write_json(output, result)
            return result
    comparison = compare_outputs(result["baseline"]["outputs"], result["candidate"]["outputs"],
                  max_retention_drop, read_jsonl(reference, predictions=True) if reference else None)
    result.update(comparison)
    result["review_findings"].extend(structure["review_findings"])
    if result["status"] != "blocked" and result["review_findings"]:
        result["status"] = "review_required"
    if reference:
        result["reference_sha256"] = file_hash(reference)
    write_json(output, result)
    return result
