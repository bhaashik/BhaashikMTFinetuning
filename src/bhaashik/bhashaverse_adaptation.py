"""HF MBart-format BhashaVerse/OneNMT supervised adaptation with Fairseq IDs.

Only compatible, locally inspected HF assets are accepted; no generic MBart
tokenizer, remote code or silent vocabulary remapping is used.
"""
import json
from pathlib import Path
import shutil
import time
from importlib.metadata import version

from .data import audit, read_jsonl, write_json
from .multilingual import resolve_languages
from .preflight import AssetError, FairseqCodec, checkpoint_layout


def validate_prefix_routes(codec, rows, allow_unknown=False):
    """Check newly composed directions are representable and distinguishable."""
    routes = {resolve_languages(row) for row in rows}
    signatures = {}
    issues = {}
    for src, tgt in sorted(routes):
        prefix = codec.tagged("", src, tgt).strip()
        ids = codec.encode(prefix, "src")[:-1]
        unknown = sum(x == codec.special["unk"] for x in ids)
        if unknown and not allow_unknown:
            raise AssetError(f"Prefix {src}->{tgt} has {unknown} unknown dictionary pieces; "
                             "use the validated SPM/dictionary/embedding repair pipeline")
        signature = tuple(ids)
        if signature in signatures:
            raise AssetError(f"Direction prefix collision: {src}->{tgt} and {signatures[signature]}")
        signatures[signature] = f"{src}->{tgt}"
        issues[f"{src}->{tgt}"] = {"prefix": prefix, "unknown_ids": unknown,
                                   "pieces": codec.pieces(prefix), "ids": ids}
    return issues


def train_bhashaverse(config_path):
    """Optional full or LoRA finetuning on local HF MBart-compatible assets."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    assets = Path(cfg["model_assets"]).resolve()
    output = Path(cfg["output_dir"]).resolve()
    if assets == output or assets in output.parents or output in assets.parents:
        raise ValueError("Model assets and outputs must be separate directories")
    if output.exists() and any(output.iterdir()) and not cfg.get("resume_from_checkpoint"):
        raise FileExistsError("Use an empty output directory or explicit resume_from_checkpoint")
    mode = cfg.get("mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError("mode must be lora or full")
    if cfg.get("strategy", "text_prefix") != "text_prefix":
        raise ValueError("OneNMT HF adaptation uses direction text_prefix; native inference must be externally validated")
    codec = FairseqCodec(assets)
    layout = checkpoint_layout(codec)
    if codec.config.get("model_type") != "mbart" or not layout["weight_layout_checked"] or layout["issues"]:
        raise AssetError("Require validated, local MBart-compatible HF safetensors: " +
                         "; ".join(layout["issues"]))
    repair = assets / "repair_manifest.json"
    if repair.exists():
        manifest = json.loads(repair.read_text(encoding="utf-8"))
        if manifest.get("requires_embedding_training") and mode == "lora":
            raise ValueError("New dictionary/embedding rows require full training; attention-only LoRA freezes them")
    audit_result = audit({"train": cfg["train_file"], "dev": cfg["dev_file"]})
    train_rows, dev_rows = read_jsonl(cfg["train_file"]), read_jsonl(cfg["dev_file"])
    for row in train_rows + dev_rows:
        resolve_languages(row)
    all_rows = train_rows + dev_rows
    prefix_review = validate_prefix_routes(codec, all_rows,
                                allow_unknown=cfg.get("allow_unknown_prefix", False))
    if cfg.get("allow_unknown_prefix", False):
        raise ValueError("Unknown prefix IDs require dictionary/SPM repair, not a silent override")
    import torch
    from transformers import (MBartForConditionalGeneration, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, set_seed)
    from datasets import Dataset
    set_seed(cfg.get("seed", 42))
    model = MBartForConditionalGeneration.from_pretrained(
        assets, local_files_only=True, use_safetensors=True)
    model.config.forced_bos_token_id = None
    model.config.use_cache = False
    if mode == "lora":
        from peft import TaskType, LoraConfig, get_peft_model
        model = get_peft_model(
            model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                              r=cfg.get("lora_rank", 8), lora_alpha=16, lora_dropout=.05,
                              target_modules=["q_proj", "v_proj"]))
    max_length = cfg.get("max_length", 128)
    def records(rows):
        examples = []
        for row in rows:
            src, tgt = resolve_languages(row)
            input_ids = codec.encode_source(row["source"], src, tgt)
            labels = codec.encode(row["target"], "tgt")
            if len(input_ids) > max_length or len(labels) > max_length:
                raise ValueError(f"Over-length example {row['id']}; inspect or raise max_length explicitly")
            examples.append({"input_ids": input_ids, "attention_mask": [1]*len(input_ids),
                             "labels": labels})
        return Dataset.from_list(examples)
    train_ds, dev_ds = records(train_rows), records(dev_rows)

    def collator(examples):
        longest_src = max(len(x["input_ids"]) for x in examples)
        longest_tgt = max(len(x["labels"]) for x in examples)
        pad = codec.special["pad"]
        return {
            "input_ids": torch.tensor(
                [x["input_ids"] + [pad]*(longest_src-len(x["input_ids"])) for x in examples]),
            "attention_mask": torch.tensor(
                [x["attention_mask"] + [0]*(longest_src-len(x["attention_mask"])) for x in examples]),
            "labels": torch.tensor(
                [x["labels"] + [-100]*(longest_tgt-len(x["labels"])) for x in examples]),
        }
    precision = cfg.get("precision", "fp16")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("Precision must be fp32/fp16/bf16")
    if precision != "fp32" and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires CUDA; select fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16")
    args = Seq2SeqTrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        per_device_eval_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs=cfg.get("epochs", 3), max_steps=cfg.get("max_steps", -1),
        learning_rate=cfg.get("learning_rate", 2e-4 if mode == "lora" else 2e-5),
        fp16=precision == "fp16", bf16=precision == "bf16",
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=10, report_to="none",
        dataloader_num_workers=0, predict_with_generate=False,
        optim="adamw_torch", seed=cfg.get("seed", 42))
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator)
    started = time.time()
    trained = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    dev_result = trainer.evaluate()
    if mode == "lora":
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    # These two files are essential: HF AutoTokenizer cannot recreate the
    # original fairseq piece IDs.
    for filename in ("fairseq_dict.json", "onemtv3b_spm.model"):
        shutil.copy2(assets / filename, output / filename)
    write_json(output / "run_manifest.json", {
        "config": cfg, "data": audit_result, "prefixes": prefix_review,
        "train": trained.metrics, "dev": dev_result,
        "seconds": time.time()-started, "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else "cpu",
        "packages": {p: version(p) for p in ("torch","transformers","datasets","peft","sentencepiece")},
        "warning": "HF MBart compatibility does not prove upstream BhashaVerse reference-output parity; "
                   "validate unchanged-language probes and native inference first."})
    return dev_result


def generate_bhashaverse_dev(assets, dev_path, output, device=None, batch_size=1,
                             max_new_tokens=64):
    """Generate tagged JSONL without a generic AutoTokenizer."""
    from .hf_regression import generate_probes
    import torch
    rows = read_jsonl(dev_path)
    for row in rows:
        src, tgt = resolve_languages(row)
        row["src_lang"], row["tgt_lang"] = src, tgt
    outputs, metadata = generate_probes(
        assets, rows, device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        precision="fp16" if (device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda" else "fp32",
        max_new_tokens=max_new_tokens, batch_size=batch_size)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(str(destination) + ".manifest.json", metadata)
    return destination
