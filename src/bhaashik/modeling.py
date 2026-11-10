"""NLLB HF training, inference and explicit language-token extension."""
import json
import platform
import time
from pathlib import Path
from importlib.metadata import version
from .data import audit, read_jsonl, file_hash, write_json


def require_language(tokenizer, language):
    if language not in tokenizer.additional_special_tokens:
        raise ValueError(f"Unsupported language token {language}; use extend before training")
    token_id = tokenizer.convert_tokens_to_ids(language)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown language token: {language}")
    return token_id


def load(model_id, revision=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if "NllbTokenizer" not in type(tokenizer).__name__:
        raise ValueError("This runner supports NLLB tokenization only; use the upstream IndicTrans toolkit for IndicTrans2")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def extend(model_id, output, language, donor="hin_Deva", revision=None):
    import torch
    if not __import__('re').fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", language):
        raise ValueError("Use a language_Script tag such as raj_Deva; document its linguistic scope")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("extension output must be empty")
    model, tokenizer = load(model_id, revision)
    donor_id = require_language(tokenizer, donor)
    if language in tokenizer.get_vocab():
        raise ValueError(f"{language} already exists; use normal finetuning")
    tokenizer.add_special_tokens({"additional_special_tokens": [language]}, replace_additional_special_tokens=False)
    # Tokenizers can have sparse IDs; pretrained model tables may also be padded.
    # Never shrink an existing model or assume len(tokenizer) is max ID + 1.
    model.resize_token_embeddings(max(model.config.vocab_size, max(tokenizer.get_vocab().values()) + 1))
    new_id = require_language(tokenizer, language)
    with torch.no_grad():
        model.get_input_embeddings().weight[new_id].copy_(model.get_input_embeddings().weight[donor_id])
        model.get_output_embeddings().weight[new_id].copy_(model.get_output_embeddings().weight[donor_id])
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    write_json(output / "bhaashik_extension.json", {"language": language, "donor": donor,
               "base": str(model_id), "revision": revision, "resolved_revision": getattr(model.config, "_commit_hash", None),
               "requires_full_finetuning": True, "note": "New tag initialization is not learned translation support."})


def train(config_path):
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
    config = json.loads(Path(config_path).read_text())
    seed = config.get("seed", 42)
    set_seed(seed)
    mode = config.get("mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError("mode must be lora or full")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not config.get("resume_from_checkpoint"):
        raise ValueError("output_dir is nonempty; use a new run directory or explicit resume_from_checkpoint")
    data_audit = audit({"train": config["train_file"], "dev": config["dev_file"]})
    model, tokenizer = load(config["model"], config.get("revision"))
    if mode == "lora" and Path(config["model"], "bhaashik_extension.json").exists():
        raise ValueError("Extended language embeddings require mode=full; LoRA freezes them")
    tokenizer.src_lang, tokenizer.tgt_lang = config["src_lang"], config["tgt_lang"]
    require_language(tokenizer, tokenizer.src_lang)
    target_id = require_language(tokenizer, tokenizer.tgt_lang)
    model.config.forced_bos_token_id = None
    model.generation_config.forced_bos_token_id = target_id
    model.config.use_cache = False
    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                    r=config.get("lora_rank", 8), lora_alpha=16, lora_dropout=.05,
                    target_modules=["q_proj", "v_proj"]))
    max_length = config.get("max_length", 128)
    def encode(batch):
        return tokenizer(batch["source"], text_target=batch["target"], max_length=max_length, truncation=True)
    def dataset(path):
        ds = Dataset.from_list(read_jsonl(path))
        return ds.map(encode, batched=True, remove_columns=ds.column_names)
    started = time.time()
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if precision != "fp32" and not torch.cuda.is_available():
        raise ValueError("Mixed precision config requires a CUDA GPU; select fp32 on CPU")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16 or fp32")
    args = Seq2SeqTrainingArguments(
        output_dir=str(output / "checkpoints"), seed=seed, data_seed=seed,
        per_device_train_batch_size=config.get("batch_size", 1),
        per_device_eval_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        learning_rate=config.get("learning_rate", 2e-4 if mode == "lora" else 2e-5),
        num_train_epochs=config.get("epochs", 3), max_steps=config.get("max_steps", -1),
        fp16=precision == "fp16", bf16=precision == "bf16",
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2,
        logging_steps=10, report_to="none", dataloader_num_workers=0,
        predict_with_generate=False, optim="adamw_torch")
    trainer = Seq2SeqTrainer(model=model, args=args, tokenizer=tokenizer,
                train_dataset=dataset(config["train_file"]), eval_dataset=dataset(config["dev_file"]),
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8))
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    dev_metrics = trainer.evaluate()
    if mode == "lora":
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    # Carry the marker so subsequent LoRA runs cannot silently freeze new embeddings.
    extension = Path(config["model"], "bhaashik_extension.json")
    if extension.exists():
        (output / extension.name).write_text(extension.read_text())
    write_json(output / "run_manifest.json", {
        "config": config, "data": data_audit, "train": train_result.metrics, "dev": dev_metrics,
        "seconds": time.time() - started, "python": platform.python_version(),
        "packages": {p: version(p) for p in ("torch", "transformers", "peft", "datasets", "accelerate", "sentencepiece", "tokenizers")},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "trainable_parameters": trainable_parameters,
        "total_parameters": sum(p.numel() for p in model.parameters())})
    (output / "README.md").write_text(
        "---\nlibrary_name: transformers\npipeline_tag: translation\n---\n\n"
        "# Bhaashik adapted NLLB\n\nSee run_manifest.json for data hashes, configuration and validation loss. "
        "Inherited model/data license restrictions apply. No test-quality claims are implied.\n", encoding="utf-8")
    return dev_metrics


def translate(model_id, src, tgt, input_path, output, revision=None, batch_size=4, max_length=256, max_new_tokens=256, beams=4):
    import torch
    if min(batch_size, max_length, max_new_tokens, beams) < 1:
        raise ValueError("batch size, lengths and beams must be positive")
    rows = read_jsonl(input_path)
    model, tokenizer = load(model_id, revision)
    tokenizer.src_lang, tokenizer.tgt_lang = src, tgt
    require_language(tokenizer, src)
    target_id = require_language(tokenizer, tgt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    predictions, truncated = [], 0
    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = [r["source"] for r in rows[i:i+batch_size]]
            lengths = tokenizer(batch, truncation=False)["input_ids"]
            truncated += sum(len(ids) > max_length for ids in lengths)
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            tokens = model.generate(**encoded, forced_bos_token_id=target_id,
                                    max_new_tokens=max_new_tokens, num_beams=beams)
            predictions.extend(tokenizer.batch_decode(tokens, skip_special_tokens=True))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions):
            handle.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    write_json(str(output) + ".manifest.json", {"model": str(model_id), "revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None), "src_lang": src, "tgt_lang": tgt,
        "n": len(rows), "truncated_sources": truncated, "input_sha256": file_hash(input_path),
        "output_sha256": file_hash(output), "num_beams": beams, "max_length": max_length,
        "max_new_tokens": max_new_tokens, "seconds": time.time()-started, "device": device})
