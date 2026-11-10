"""Read-only OneNMT vocabulary checks. No architecture conversion or asset rewriting."""
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from .data import file_hash, normalized, read_jsonl, write_json

SPECIAL_SYMBOLS = {"bos": "<s>", "pad": "<pad>", "eos": "</s>", "unk": "<unk>"}
EMBEDDING_SUFFIXES = ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight")


class AssetError(ValueError):
    pass


def read_object(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssetError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise AssetError(f"Expected a JSON object: {path}")
    return value


class FairseqCodec:
    """Follow the authors' SPM-piece → dictionary-ID inference convention.

    Source/target IDs are deliberately independent. Never load tokenizer.json.
    """
    def __init__(self, assets):
        import sentencepiece as spm
        self.root = Path(assets)
        self.dictionary = read_object(self.root / "fairseq_dict.json")
        self.special = self.dictionary.get("special", {})
        if not isinstance(self.special, dict):
            raise AssetError("special must be an object of token IDs")
        for name, symbol in SPECIAL_SYMBOLS.items():
            idx = self.special.get(name)
            if type(idx) is not int or idx < 0:
                raise AssetError(f"Invalid special ID: {name}")
        if len(set(self.special[k] for k in SPECIAL_SYMBOLS)) != 4:
            raise AssetError("Special IDs must be distinct")
        for side in ("src", "tgt"):
            mapping = self.dictionary.get(side)
            if not isinstance(mapping, dict) or not mapping:
                raise AssetError(f"Missing {side} dictionary")
            if any(not isinstance(p, str) or not p or type(i) is not int or i < 0 for p, i in mapping.items()):
                raise AssetError(f"Invalid piece or ID in {side} dictionary")
            if len(set(mapping.values())) != len(mapping):
                raise AssetError(f"Duplicate IDs in {side} dictionary")
            for name, symbol in SPECIAL_SYMBOLS.items():
                if mapping.get(symbol) != self.special[name]:
                    raise AssetError(f"{side} {symbol} does not match special.{name}")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.root / "onemtv3b_spm.model"))
        self.inverse = {side: {i: piece for piece, i in self.dictionary[side].items()} for side in ("src", "tgt")}
        self.inverse_target = self.inverse["tgt"]
        self.config = read_object(self.root / "config.json") if (self.root / "config.json").exists() else {}

    @staticmethod
    def tagged(text, src, tgt):
        if not all(isinstance(x, str) and re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", x) for x in (src, tgt)):
            raise AssetError("Language codes must have the form hin_Deva or eng_Latn")
        return f"###{src}-to-{tgt}### {text}"

    def pieces(self, text):
        return self.sp.encode(text, out_type=str)

    def encode(self, text, side="src"):
        return [self.dictionary[side].get(p, self.special["unk"]) for p in self.pieces(text)] + [self.special["eos"]]

    def encode_source(self, text, src, tgt):
        return self.encode(self.tagged(text, src, tgt), "src")

    def decode(self, ids):
        strip = {self.special[k] for k in ("bos", "pad", "eos")}
        pieces = [self.inverse_target.get(int(i), "<unk>") for i in ids if int(i) not in strip]
        return self.sp.decode(pieces).strip()

    def fingerprint(self):
        return {name: file_hash(self.root/name) for name in
                ("fairseq_dict.json", "onemtv3b_spm.model", "config.json", "generation_config.json")
                if (self.root/name).is_file()}


def load_probes(path, src=None, tgt=None):
    rows = read_jsonl(path)
    for row in rows:
        row["src_lang"] = row.get("src_lang", src)
        row["tgt_lang"] = row.get("tgt_lang", tgt)
        row["group"] = row.get("group", "unspecified")
        if row["group"] not in ("retention", "adaptation", "unspecified"):
            raise AssetError("Probe group must be retention, adaptation or unspecified")
        FairseqCodec.tagged(row["source"], row["src_lang"], row["tgt_lang"])
    return rows


def checkpoint_layout(codec):
    """Read tensor shapes only; never deserialize pickled checkpoints or load weights."""
    from safetensors import safe_open, SafetensorError
    config = codec.config
    required = {side: max(codec.dictionary[side].values()) + 1 for side in ("src", "tgt")}
    issues, warnings, shapes = [], [], {}
    if any(codec.special[name] != idx for name, idx in (("bos", 0), ("pad", 1), ("eos", 2), ("unk", 3))):
        issues.append("Special IDs differ from the authors' OneNMT convention (bos=0, pad=1, eos=2, unk=3)")
    if not config:
        warnings.append("No config.json: model dimensions and generation IDs are unverified")
    else:
        for key, expected in (("bos_token_id", codec.special["bos"]), ("pad_token_id", codec.special["pad"]),
                              ("eos_token_id", codec.special["eos"]), ("decoder_start_token_id", codec.special["eos"])):
            if config.get(key) != expected:
                issues.append(f"config.{key}={config.get(key)!r}; expected {expected}")
        vocab = config.get("vocab_size")
        if type(vocab) is not int or vocab < max(required.values()):
            issues.append(f"config.vocab_size={vocab!r} cannot cover required ID ranges {required}")
        if config.get("forced_bos_token_id") is not None:
            warnings.append("config forces a BOS token; reference OneNMT inference uses no forced BOS")
        if config.get("model_type") != "mbart":
            warnings.append("config.model_type is not mbart; the HF regression runner supports MBart only")
    generation_path = codec.root / "generation_config.json"
    if generation_path.exists():
        generation = read_object(generation_path)
        for key, expected in (("decoder_start_token_id", codec.special["eos"]),
                              ("pad_token_id", codec.special["pad"]), ("eos_token_id", codec.special["eos"])):
            if key in generation and generation[key] != expected:
                issues.append(f"generation_config.{key} conflicts with dictionary special IDs")
        if generation.get("forced_bos_token_id") is not None:
            warnings.append("generation_config forces a BOS token; use the explicit OneNMT generation settings")
    # Follow the index, not arbitrary neighboring/adapter safetensors files.
    index_path = codec.root / "model.safetensors.index.json"
    if index_path.exists():
        index = read_object(index_path)
        if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
            raise AssetError("Invalid safetensors index")
        files = []
        if any(not isinstance(filename, str) for filename in index["weight_map"].values()):
            raise AssetError("Invalid shard name")
        for filename in sorted(set(index["weight_map"].values())):
            path = (codec.root / filename).resolve()
            if not path.is_relative_to(codec.root.resolve()) or path.suffix != ".safetensors":
                raise AssetError("Shard path must stay inside the asset directory")
            files.append(path)
    else:
        files = [codec.root / "model.safetensors"] if (codec.root / "model.safetensors").exists() else []
    for path in files:
        try:
            with safe_open(path, framework="np") as handle:
                for key in handle.keys():
                    if key.endswith(EMBEDDING_SUFFIXES):
                        shapes[key] = list(handle.get_slice(key).get_shape())
        except SafetensorError as error:
            raise AssetError(f"Invalid safetensors header in {path.name}: {error}") from error
    for name, shape in shapes.items():
        sides = ("src",) if "encoder.embed_tokens" in name else (("tgt",) if "decoder.embed_tokens" in name or "lm_head" in name else ("src", "tgt"))
        if len(shape) != 2 or shape[0] < max(required[s] for s in sides):
            issues.append(f"{name} shape {shape} cannot cover dictionary IDs")
        if len(shape) == 2 and type(config.get("vocab_size")) is int and shape[0] != config["vocab_size"]:
            issues.append(f"{name} row count disagrees with config.vocab_size")
        if len(shape) == 2 and config.get("d_model") is not None and shape[1] != config["d_model"]:
            issues.append(f"{name} width disagrees with config.d_model")
    if not shapes:
        warnings.append("No embedding tensors inspected: config alone does not verify checkpoint compatibility")
    differing_ids = sum(codec.dictionary["tgt"].get(piece) != idx for piece, idx in codec.dictionary["src"].items())
    if differing_ids and any(name.endswith("shared.weight") for name in shapes):
        warnings.append("Shared embedding tensor with differing source/target dictionary IDs: verify HF weight tying and independent reference outputs")
    return {"required_rows": required, "embedding_shapes": shapes, "weight_layout_checked": bool(shapes),
            "source_target_id_disagreements": differing_ids,
            "issues": issues, "warnings": warnings,
            "architecture": {k: config.get(k) for k in ("model_type", "d_model", "encoder_layers", "decoder_layers", "max_position_embeddings", "vocab_size")},
            "architecture_parity": "unverified; no architecture conversion is performed"}


def coverage(codec, rows, max_source_tokens=None):
    config_limit = codec.config.get("max_position_embeddings")
    if config_limit is not None and (type(config_limit) is not int or config_limit < 1):
        raise AssetError("Invalid config.max_position_embeddings")
    if max_source_tokens is not None and max_source_tokens < 1:
        raise AssetError("max_source_tokens must be positive")
    limits = [n for n in (config_limit, max_source_tokens) if n is not None]
    limit = min(limits) if limits else None
    groups, sequences, prefix_encodings = {}, {}, {}
    for row in rows:
        direction = f"{row['src_lang']}->{row['tgt_lang']}"
        prefix = codec.tagged("", row["src_lang"], row["tgt_lang"]).strip()
        prefix_ids = codec.encode(prefix)
        prefix_encodings[direction] = {"ids": prefix_ids, "unknown_ids": prefix_ids.count(codec.special["unk"])}
        for side, text in (("src", row["source"]), ("tgt", row["target"])):
            pieces, ids = codec.pieces(text), codec.encode(text, side)
            key = f"{row['group']}|{direction}|{side}"
            group = groups.setdefault(key, {"sentences": 0, "pieces": 0, "words": 0, "unknown_ids": 0,
                        "dictionary_misses": 0, "spm_unknown_pieces": 0, "roundtrip_changes": 0,
                        "source_context_overflows": 0, "max_encoded_tokens": 0, "missing_pieces": Counter()})
            group["sentences"] += 1
            group["pieces"] += len(pieces)
            group["words"] += len(text.split())
            group["unknown_ids"] += ids[:-1].count(codec.special["unk"])
            missing = [p for p in pieces if p not in codec.dictionary[side]]
            group["dictionary_misses"] += len(missing)
            group["missing_pieces"].update(missing)
            group["spm_unknown_pieces"] += codec.sp.encode(text, out_type=int).count(codec.sp.unk_id())
            inverse = codec.inverse[side]
            decoded = codec.sp.decode([inverse.get(i, "<unk>") for i in ids[:-1]])
            group["roundtrip_changes"] += normalized(decoded) != normalized(text)
            tagged_ids = codec.encode_source(text, row["src_lang"], row["tgt_lang"]) if side == "src" else ids
            group["max_encoded_tokens"] = max(group["max_encoded_tokens"], len(tagged_ids))
            group["source_context_overflows"] += side == "src" and limit is not None and len(tagged_ids) > limit
            sequences[(row["id"], side)] = {"pieces": pieces, "ids": ids, "tagged_ids": tagged_ids}
    for group in groups.values():
        group["unknown_rate"] = group["unknown_ids"] / max(group["pieces"], 1)
        group["tokens_per_whitespace_word"] = group["pieces"] / max(group["words"], 1)
        group["missing_pieces"] = group["missing_pieces"].most_common(20)
    collisions = []
    by_encoding = defaultdict(list)
    for direction, entry in prefix_encodings.items():
        by_encoding[tuple(entry["ids"])].append(direction)
    for directions in by_encoding.values():
        if len(directions) > 1:
            collisions.append(directions)
    return {"groups": groups, "source_token_limit": limit, "prefix_encodings": prefix_encodings,
            "prefix_collisions": collisions,
            "note": "Whitespace fertility is a descriptive proxy, not a cross-script quality score. Target coverage uses plain reference text; original training target conventions still need verification."}, sequences


def compare_codecs(original, candidate, original_sequences, candidate_sequences):
    mapping = {}
    for side in ("src", "tgt"):
        old, new = original.dictionary[side], candidate.dictionary[side]
        removed = [p for p in old if p not in new]
        remapped = [p for p in old if p in new and old[p] != new[p]]
        added = [p for p in new if p not in old]
        append_only = all(new[p] > max(old.values()) for p in added)
        mapping[side] = {"removed_count": len(removed), "remapped_count": len(remapped), "added_count": len(added),
                         "removed_examples": removed[:20], "remapped_examples": remapped[:20],
                         "added_examples": added[:20], "additions_after_old_max_id": append_only,
                         "old_ids_preserved": not removed and not remapped}
    changed = {"piece_sequences": 0, "id_sequences": 0, "tagged_source_sequences": 0, "examples": []}
    for key, old in original_sequences.items():
        new = candidate_sequences[key]
        piece_change, id_change = old["pieces"] != new["pieces"], old["ids"] != new["ids"]
        tag_change = key[1] == "src" and old["tagged_ids"] != new["tagged_ids"]
        changed["piece_sequences"] += piece_change
        changed["id_sequences"] += id_change
        changed["tagged_source_sequences"] += tag_change
        if (piece_change or id_change or tag_change) and len(changed["examples"]) < 20:
            changed["examples"].append({"id": key[0], "side": key[1], "pieces_changed": piece_change,
                                        "ids_changed": id_change, "tagged_source_changed": tag_change})
    changed["sequence_count"] = len(original_sequences)
    return {"dictionaries": mapping, "segmentation_and_ids": changed,
            "sentencepiece_artifact_changed": original.fingerprint()["onemtv3b_spm.model"] != candidate.fingerprint()["onemtv3b_spm.model"],
            "special_ids_preserved": original.special == candidate.special}


def plan_append(codec, selected, rows):
    """Build an inspectable dictionary proposal in memory; never resize/mutate assets."""
    if not isinstance(selected, dict) or set(selected) - {"src", "tgt"}:
        raise AssetError("Selected tokens must be an object with src/tgt lists")
    planned = {"src": dict(codec.dictionary["src"]), "tgt": dict(codec.dictionary["tgt"]), "special": dict(codec.special)}
    report = {}
    for side in ("src", "tgt"):
        tokens = selected.get(side, [])
        if not isinstance(tokens, list) or any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens):
            raise AssetError("Selected tokens must be lists of nonempty SentencePiece pieces without whitespace")
        if len(set(tokens)) != len(tokens):
            raise AssetError("Selected tokens contain duplicates")
        emitted = Counter()
        for row in rows:
            text = codec.tagged(row["source"], row["src_lang"], row["tgt_lang"]) if side == "src" else row["target"]
            emitted.update(codec.pieces(text))
        additions, existing = [], []
        for token in tokens:
            if token in planned[side]:
                existing.append(token)
                continue
            idx = max(planned[side].values()) + 1
            planned[side][token] = idx
            sp_id = codec.sp.piece_to_id(token)
            active_piece = codec.sp.id_to_piece(sp_id) == token and not codec.sp.is_control(sp_id) and not codec.sp.is_unused(sp_id)
            additions.append({"piece": token, "id": idx, "in_current_spm_inventory": active_piece,
                              "observed_emissions": emitted[token],
                              "dictionary_only_addition_inactive": emitted[token] == 0 and not active_piece})
        report[side] = {"additions": additions, "already_present": existing,
                        "old_max_id": max(codec.dictionary[side].values()), "required_rows_after_append": max(planned[side].values()) + 1}
    return report, planned


def protect_outputs(asset_roots, inputs, outputs):
    roots = [Path(p).resolve() for p in asset_roots if p]
    protected = {Path(p).resolve() for p in inputs if p}
    destinations = [Path(p).resolve() for p in outputs if p]
    if len(set(destinations)) != len(destinations):
        raise AssetError("Report and plan outputs must be distinct")
    for path in destinations:
        if path in protected or any(path == root or path.is_relative_to(root) for root in roots):
            raise AssetError("Outputs must be outside asset directories and must not overwrite input files")


def preflight(assets, corpus, output, src=None, tgt=None, candidate_assets=None,
              selected_tokens=None, plan_output=None, max_source_tokens=None, max_unknown_rate=0.0):
    if not 0 <= max_unknown_rate <= 1:
        raise AssetError("max_unknown_rate must be between 0 and 1")
    if plan_output and not selected_tokens:
        raise AssetError("plan_output requires selected_tokens")
    protect_outputs([assets, candidate_assets], [corpus, selected_tokens], [output, plan_output])
    original = FairseqCodec(assets)
    rows = load_probes(corpus, src, tgt)
    original_coverage, sequences = coverage(original, rows, max_source_tokens)
    original_layout = checkpoint_layout(original)
    blocks = list(original_layout["issues"])
    reviews = list(original_layout["warnings"])
    def repair_state(codec):
        path = codec.root / "repair_manifest.json"
        if not path.exists():
            return None
        state = read_object(path)
        if state.get("requires_embedding_training"):
            reviews.append(f"{codec.root.name}: repair manifest marks new vocabulary rows as requiring training")
        return {"operation": state.get("operation"), "status": state.get("status"),
                "requires_embedding_training": bool(state.get("requires_embedding_training")),
                "manifest_sha256": file_hash(path)}
    result = {"schema_version": 1, "kind": "onenmt_preflight", "assets": original.fingerprint(),
              "corpus_sha256": file_hash(corpus), "probe_rows": len(rows), "original": {
                  "coverage": original_coverage, "checkpoint": original_layout, "repair_state": repair_state(original)},
              "thresholds": {"max_unknown_rate": max_unknown_rate},
              "behavioral_validation": "not_run; structural checks cannot establish translation quality or conversion parity",
              "model_history": "Public inspected code uses Fairseq and an HF MBart port; an earlier OpenNMT origin has not been verified."}
    for label, cov in [("original", original_coverage)]:
        if cov["prefix_collisions"]:
            blocks.append(f"{label}: translation direction prefixes collide")
    if not any(r["group"] == "retention" for r in rows):
        reviews.append("No retention probes: old-language behavior cannot be assessed")
    if selected_tokens:
        selected = read_object(selected_tokens)
        proposal, dictionary = plan_append(original, selected, rows)
        result["append_plan"] = proposal
        if plan_output:
            # A proposal contains only dictionaries, never tokenizer/model replacements.
            write_json(plan_output, dictionary)
        if any(proposal[s]["additions"] for s in proposal):
            reviews.append("Append plan requires embedding/output row initialization and training; dictionary changes alone do not add learned support")
        if any(a["dictionary_only_addition_inactive"] for s in proposal.values() for a in s["additions"]):
            reviews.append("Some proposed tokens are not emitted by the current tokenizer; dictionary-only appends will not activate them")
    target_coverage = original_coverage
    if candidate_assets:
        candidate = FairseqCodec(candidate_assets)
        candidate_cov, candidate_seq = coverage(candidate, rows, max_source_tokens)
        candidate_layout = checkpoint_layout(candidate)
        comparison = compare_codecs(original, candidate, sequences, candidate_seq)
        result["candidate"] = {"assets": candidate.fingerprint(), "coverage": candidate_cov,
                               "checkpoint": candidate_layout, "comparison": comparison, "repair_state": repair_state(candidate)}
        target_coverage = candidate_cov
        blocks.extend("candidate: " + x for x in candidate_layout["issues"])
        reviews.extend("candidate: " + x for x in candidate_layout["warnings"])
        if not comparison["special_ids_preserved"]:
            blocks.append("Candidate changes special IDs: incompatible with direct checkpoint reuse")
        if any(not x["old_ids_preserved"] or not x["additions_after_old_max_id"] for x in comparison["dictionaries"].values()):
            blocks.append("Candidate removes/reorders/reuses old dictionary IDs: direct checkpoint reuse would change embedding semantics")
        if comparison["sentencepiece_artifact_changed"]:
            reviews.append("SentencePiece changed: unchanged probe segmentation cannot establish global equivalence; require behavioral regression tests")
        if comparison["segmentation_and_ids"]["id_sequences"] or comparison["segmentation_and_ids"]["tagged_source_sequences"]:
            reviews.append("Candidate changes encoded probe sequences; compare retention and adaptation outputs before training/release")
        if any(x["added_count"] for x in comparison["dictionaries"].values()):
            reviews.append("New dictionary rows need initialization, trainable parameters and checkpoint roundtrip tests")
        if candidate_cov["prefix_collisions"]:
            blocks.append("candidate: translation direction prefixes collide")
    if any(p["unknown_ids"] for p in target_coverage["prefix_encodings"].values()):
        reviews.append("Direction prefixes contain unknown IDs; a new language route may not be represented distinctly")
    if any(g["unknown_rate"] > max_unknown_rate for g in target_coverage["groups"].values()):
        reviews.append("Probe unknown-token rate exceeds the configured threshold")
    if any(g["source_context_overflows"] for g in target_coverage["groups"].values()):
        blocks.append("Source probes exceed the configured context budget; no truncation has been applied")
    if target_coverage["source_token_limit"] is None:
        reviews.append("Context limit unknown: supply config.json or max_source_tokens")
    result["blocking_findings"] = blocks
    result["review_findings"] = reviews
    result["structural_status"] = "blocked" if blocks else ("review_required" if reviews else "checks_passed")
    result["decision"] = "stop_direct_reuse" if blocks else "behavioral_validation_required"
    result["strategy_guidance"] = {
        "unchanged_vocabulary": "Prefer first if coverage is adequate; adding data can still cause forgetting or poor outputs, so run retained-language probes.",
        "append_only": "Preserve every old ID. Verify that pieces are emitted, allocate model rows, train new rows, and evaluate old/new languages.",
        "replace_or_retrain_spm": "Treat as a separate high-risk ablation. Preserve dictionary IDs and measure segmentation drift; rebuilding dictionaries is incompatible with direct reuse.",
        "limits": "These checks sample the supplied probes. They do not prove language support, corpus quality, target-label correctness or architecture parity."}
    if output is not None:
        write_json(output, result)
    return result
