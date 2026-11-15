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
    if strategy not in ("native", "text_prefix"):
        raise ValueError("IndicTrans2 strategy must be native or text_prefix")
    if strategy == "native":
        if src not in INDICTRANS2_LANGS or tgt not in INDICTRANS2_LANGS:
            raise ValueError(f"IndicTrans2 does not natively support {src}->{tgt}")
        return src, tgt
    ps, pt = proxy_src or src, proxy_tgt or tgt
    if ps not in INDICTRANS2_LANGS or pt not in INDICTRANS2_LANGS:
        raise ValueError("Proxy processing languages must be native IndicTrans2 tags")
    if (src not in INDICTRANS2_LANGS and proxy_src is None or
            tgt not in INDICTRANS2_LANGS and proxy_tgt is None):
        raise ValueError("Explicit proxy_src/proxy_tgt is required for unsupported language tags")
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
