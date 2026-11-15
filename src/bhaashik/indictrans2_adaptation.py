"""Experimental IndicTrans2 language-conditioned LoRA/full finetuning.

Upstream custom HF model code is opt-in. No unsupported tag is passed directly
to IndicProcessor: an explicit supported processing proxy is recorded instead.
The prefix strategy is an ablation, not native language support.
"""
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

from .data import audit, file_hash, read_jsonl, write_json
from .model_comparisons import INDICTRANS2_LANGS
from .multilingual import resolve_languages


def processing_pair(src, tgt, *, strategy, proxy_src=None, proxy_tgt=None):
    if strategy not in ("native", "text_prefix", "dedicated_token"):
        raise ValueError("IndicTrans2 strategy must be native, text_prefix or dedicated_token")
    if strategy == "native":
        if src not in INDICTRANS2_LANGS or tgt not in INDICTRANS2_LANGS:
            raise ValueError(f"IndicTrans2 does not natively support {src}->{tgt}")
        return src, tgt
    if ((src not in INDICTRANS2_LANGS and proxy_src is None) or
            (tgt not in INDICTRANS2_LANGS and proxy_tgt is None)):
        raise ValueError("Explicit proxy_src/proxy_tgt is required for unsupported language tags")
    ps, pt = proxy_src or src, proxy_tgt or tgt
    if ps not in INDICTRANS2_LANGS or pt not in INDICTRANS2_LANGS:
        raise ValueError("Proxy processing languages must be native IndicTrans2 tags")
    return ps, pt


def _model_id(src, tgt):
    if src == "eng_Latn":
        return "ai4bharat/indictrans2-en-indic-dist-200M"
    if tgt == "eng_Latn":
        return "ai4bharat/indictrans2-indic-en-dist-200M"
    return "ai4bharat/indictrans2-indic-indic-dist-320M"


def prepare_indic_rows(rows, processor, *, strategy, proxy_src=None, proxy_tgt=None):
    """Apply upstream processor using native or explicit proxy tags, per row."""
    inputs, targets, routes = [], [], []
    for row in rows:
        src, tgt = resolve_languages(row)
        ps, pt = processing_pair(src, tgt, strategy=strategy,
                                  proxy_src=proxy_src, proxy_tgt=proxy_tgt)
        source = row["source"]
        if strategy == "text_prefix":
            source = f"<src:{src}><tgt:{tgt}> " + source
        prepared = processor.preprocess_batch([source], src_lang=ps, tgt_lang=pt)
        if len(prepared) != 1:
            raise ValueError("IndicProcessor returned an unexpected batch length")
        if strategy == "dedicated_token":
            prepared[0] = f"<bhaashik_src_{src}> <bhaashik_tgt_{tgt}> " + prepared[0]
        inputs.append(prepared[0])
        # Target references remain CLEAN. The source language marker is learned
        # through source conditioning, not through contamination of references.
        targets.append(row["target"])
        routes.append({"source_lang": src, "target_lang": tgt,
                       "processor_source": ps, "processor_target": pt})
    return inputs, targets, routes


def train_indictrans2(config_path):
    """Train on explicit-tag JSONL using a reviewed upstream IndicTrans2 model.

    When using a target proxy (unsupported target), this is an *experimental*
    supervised decoder adaptation. It is not compatible with default target
    processor postprocessing without an independent language validation.
    """
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not cfg.get("allow_remote_code", False):
        raise ValueError("Explicitly review model code then set allow_remote_code=true")
    mode = cfg.get("mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError("mode must be lora or full")
    strategy = cfg.get("strategy", "native")
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()) and not cfg.get("resume_from_checkpoint"):
        raise FileExistsError("Use an empty output_dir or explicit resume_from_checkpoint")
    audited = audit({"train": cfg["train_file"], "dev": cfg["dev_file"]})
    from IndicTransToolkit.processor import IndicProcessor
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                              Seq2SeqTrainer, set_seed)
    set_seed(cfg.get("seed", 42))
    processor = IndicProcessor(inference=False)
    training = read_jsonl(cfg["train_file"])
    dev = read_jsonl(cfg["dev_file"])
    for row in training + dev:
        resolve_languages(row)
    src, tgt = resolve_languages(training[0])
    selected_model = cfg.get("model") or _model_id(src, tgt)
    inputs, targets, routes = prepare_indic_rows(
        training, processor, strategy=strategy, proxy_src=cfg.get("proxy_src"),
        proxy_tgt=cfg.get("proxy_tgt"))
    dev_inputs, dev_targets, dev_routes = prepare_indic_rows(
        dev, processor, strategy=strategy, proxy_src=cfg.get("proxy_src"),
        proxy_tgt=cfg.get("proxy_tgt"))
    # Do not mix incompatible IndicTrans2 direction-specific checkpoints.
    # One training run may contain many source languages, but each example must
    # resolve to the same en->indic / indic->en / indic->indic checkpoint family.
    families = {_model_id(s, t) for r in training + dev
                for s, t in [resolve_languages(r)]}
    if len(families) != 1:
        raise ValueError("Mixed IndicTrans2 model families need separate training runs")
    if cfg.get("model") and not cfg.get("allow_model_override", False):
        if selected_model != next(iter(families)):
            raise ValueError("Explicit model differs from required direction family; review override")
    tok = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True,
                                        revision=cfg.get("revision"))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        selected_model, trust_remote_code=True, revision=cfg.get("revision"))
    if strategy == "dedicated_token":
        # Additional special tokens need new embeddings that ordinary attention
        # LoRA would freeze. Use full finetuning with the resized checkpoint.
        if mode != "full":
            raise ValueError("IndicTrans2 dedicated_token requires mode=full")
        language_tokens = sorted({f"<bhaashik_{side}_{lang}>"
                                   for route in routes + dev_routes
                                   for side, lang in (("src", route["source_lang"]),
                                                     ("tgt", route["target_lang"]))})
        old_vocab_size = len(tok)
        tok.add_special_tokens({"additional_special_tokens": language_tokens})
        if len(tok) > old_vocab_size:
            model.resize_token_embeddings(len(tok))
    max_length = cfg.get("max_length", 128)
    def pack(texts, labels):
        encoded = tok(texts, text_target=labels, max_length=max_length,
                      truncation=True)
        return Dataset.from_dict(encoded)
    train_ds, dev_ds = pack(inputs, targets), pack(dev_inputs, dev_targets)
    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(
            model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                              r=cfg.get("lora_rank", 8), lora_alpha=16,
                              lora_dropout=.05, target_modules=["q_proj", "v_proj"]))
    precision = cfg.get("precision", "fp16")
    if precision != "fp32" and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires GPU; use fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    args = Seq2SeqTrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs=cfg.get("epochs", 3),
        max_steps=cfg.get("max_steps", -1),
        learning_rate=cfg.get("learning_rate", 2e-4 if mode == "lora" else 2e-5),
        fp16=precision == "fp16", bf16=precision == "bf16",
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", logging_steps=10,
        dataloader_num_workers=0, predict_with_generate=False,
        optim="adamw_torch", seed=cfg.get("seed", 42))
    trainer = Seq2SeqTrainer(
        model=model, args=args, tokenizer=tok,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=8))
    started = time.time()
    result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    dev_metrics = trainer.evaluate()
    if mode == "lora":
        model = model.merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    tok.save_pretrained(output)
    write_json(output / "run_manifest.json", {
        "model": selected_model, "model_revision": cfg.get("revision"),
        "config": cfg, "data": audited, "routes": sorted({(r["source_lang"], r["target_lang"]) for r in routes}),
        "strategy": strategy, "proxy_src": cfg.get("proxy_src"),
        "proxy_tgt": cfg.get("proxy_tgt"), "train": result.metrics, "dev": dev_metrics,
        "seconds": time.time() - started, "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("torch","transformers","peft","datasets")},
        "note": "Text-prefix proxy conditioning is experimental, not native IndicTrans2 support. "
                "Inspect script normalization and generation on human-checked probes."})
    return dev_metrics

def generate_indictrans2(model_id, dev_path, output, *, strategy="native",
                        proxy_src=None, proxy_tgt=None, batch_size=4,
                        max_length=256, max_new_tokens=256, beams=4,
                        allow_remote_code=False):
    """Opt-in inference for native or explicitly adapted IndicTrans2 models.

    A model trained using target-language proxies may require custom
    postprocessing and language expert validation. All outputs retain row tags.
    """
    if not allow_remote_code:
        raise ValueError("Review upstream model implementation and explicitly enable custom code")
    if batch_size < 1 or max_length < 1 or max_new_tokens < 1 or beams < 1:
        raise ValueError("Positive batch/length/beam values required")
    from IndicTransToolkit.processor import IndicProcessor
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from .model_comparisons import _write_predictions
    rows = read_jsonl(dev_path)
    prepared, _, routes = prepare_indic_rows(
        rows, IndicProcessor(inference=True), strategy=strategy,
        proxy_src=proxy_src, proxy_tgt=proxy_tgt)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    predicted = []
    processor = IndicProcessor(inference=True)
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            text_batch = prepared[i:i+batch_size]
            route_batch = routes[i:i+batch_size]
            encoded = tok(text_batch, padding="longest", truncation=True,
                          max_length=max_length, return_tensors="pt",
                          return_attention_mask=True).to(device)
            sequences = model.generate(**encoded, num_beams=beams,
                                       max_new_tokens=max_new_tokens)
            raw = tok.batch_decode(sequences, skip_special_tokens=True)
            for text, route in zip(raw, route_batch):
                if route["processor_target"] != route["target_lang"]:
                    # Do not apply another language's postprocessor silently.
                    predicted.append(text)
                else:
                    predicted.extend(processor.postprocess_batch(
                        [text], lang=route["target_lang"]))
    output = _write_predictions(rows, predicted, output)
    write_json(str(output) + ".manifest.json", {
        "model": str(model_id), "strategy": strategy, "proxy_src": proxy_src,
        "proxy_tgt": proxy_tgt, "n": len(rows), "input_sha256": file_hash(dev_path),
        "output_sha256": file_hash(output),
        "warning": "Proxy-target raw output bypasses target postprocessing and needs expert review."})
    return output
