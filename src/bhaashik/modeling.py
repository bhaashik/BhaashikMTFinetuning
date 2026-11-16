"""NLLB HF training, inference and explicit language-token extension."""
import json
import platform
import re
import time
from pathlib import Path
from importlib.metadata import version
from .data import audit, read_jsonl, file_hash, write_json
from .multilingual import resolve_languages, condition_sentence


def require_language(tokenizer, language):
    """Resolve registered NLLB language IDs on Transformers 4.x and 5.x."""
    # "additional_special_tokens" disappeared in Transformers 5. The
    # "all_special_tokens" interface includes the NLLB language markers in
    # both releases. Check the vocabulary too: an ordinary subword is not a
    # valid NLLB language tag, and an unknown ID must never be silently used.
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", language):
        raise ValueError(f"Invalid NLLB language tag: {language!r}")
    if language not in tokenizer.all_special_tokens:
        raise ValueError(f"Unsupported language token {language}; use extend before training")
    vocabulary = tokenizer.get_vocab()
    if language not in vocabulary:
        raise ValueError(f"Unknown language token: {language}")
    token_id = tokenizer.convert_tokens_to_ids(language)
    if (not isinstance(token_id, int) or token_id == tokenizer.unk_token_id
            or token_id != vocabulary[language]):
        raise ValueError(f"Unknown language token: {language}")
    return token_id


def add_language_token(tokenizer, language):
    """Register a new language marker while retaining existing NLLB markers."""
    if hasattr(tokenizer, "extra_special_tokens"):
        return tokenizer.add_special_tokens(
            {"extra_special_tokens": [language]}, replace_extra_special_tokens=False
        )
    return tokenizer.add_special_tokens(
        {"additional_special_tokens": [language]}, replace_additional_special_tokens=False
    )


def load(model_id, revision=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if "NllbTokenizer" not in type(tokenizer).__name__:
        raise ValueError("This runner supports NLLB tokenization only; use the upstream IndicTrans toolkit for IndicTrans2")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def extend(model_id, output, language, donor="hin_Deva", revision=None):
    import torch
    if not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", language):
        raise ValueError("Use a language_Script tag such as raj_Deva; document its linguistic scope")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("extension output must be empty")
    model, tokenizer = load(model_id, revision)
    donor_id = require_language(tokenizer, donor)
    if language in tokenizer.get_vocab():
        raise ValueError(f"{language} already exists; use normal finetuning")
    add_language_token(tokenizer, language)
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
    strategy = config.get("conditioning", "native")
    if strategy not in ("native", "text_prefix", "dedicated_token"):
        raise ValueError("conditioning must be native, text_prefix or dedicated_token")
    # A text-prefix ablation may use an explicit supported proxy; it is NOT a
    # native-language checkpoint. Native/dedicated use the actual registered ID.
    def processing_route(src, tgt):
        ps = config.get("proxy_source_lang", src) if strategy == "text_prefix" else src
        pt = config.get("proxy_target_lang", tgt) if strategy == "text_prefix" else tgt
        if strategy == "text_prefix":
            if ps != src and "proxy_source_lang" not in config or (
                    pt != tgt and "proxy_target_lang" not in config):
                raise ValueError("Non-native tags require explicit donor tags")
        require_language(tokenizer, ps)
        require_language(tokenizer, pt)
        return ps, pt
    training_rows = read_jsonl(config["train_file"])
    dev_rows = read_jsonl(config["dev_file"])
    training_routes = {resolve_languages(r, config["src_lang"], config["tgt_lang"])
                       for r in training_rows + dev_rows}
    resolved_routes = {r: processing_route(*r) for r in training_routes}
    tokenizer.src_lang, tokenizer.tgt_lang = resolved_routes[next(iter(sorted(resolved_routes)))]
    model.config.forced_bos_token_id = None
    model.generation_config.forced_bos_token_id = None
    model.config.use_cache = False
    if mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                    r=config.get("lora_rank", 8), lora_alpha=16, lora_dropout=.05,
                    target_modules=["q_proj", "v_proj"]))
    max_length = config.get("max_length", 128)
    def encode(batch):
        records = {"input_ids": [], "attention_mask": [], "labels": []}
        for idx, (source, target) in enumerate(zip(batch["source"], batch["target"])):
            row = {"source": source, "target": target}
            for key in ("source_lang", "target_lang"):
                if key in batch:
                    row[key] = batch[key][idx]
            src, tgt = resolve_languages(row, config["src_lang"], config["tgt_lang"])
            tokenizer.src_lang, tokenizer.tgt_lang = resolved_routes[(src, tgt)]
            source_text = condition_sentence(source, src, tgt, "nllb", strategy)
            encoded = tokenizer(source_text, text_target=target,
                                max_length=max_length, truncation=True)
            for key in records:
                records[key].append(encoded[key])
        return records
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
        "conditioning": strategy, "resolved_routes": {
            f"{src}->{tgt}": {"source": ps, "target": pt}
            for (src, tgt), (ps, pt) in sorted(resolved_routes.items())},
        "trainable_parameters": trainable_parameters,
        "total_parameters": sum(p.numel() for p in model.parameters())})
    (output / "README.md").write_text(
        "---\nlibrary_name: transformers\npipeline_tag: translation\n---\n\n"
        "# Bhaashik adapted NLLB\n\nSee run_manifest.json for data hashes, configuration and validation loss. "
        "Inherited model/data license restrictions apply. No test-quality claims are implied.\n", encoding="utf-8")
    return dev_metrics


def translate(model_id, src, tgt, input_path, output, revision=None, batch_size=4,
              max_length=256, max_new_tokens=256, beams=4,
              conditioning="native", proxy_source_lang=None, proxy_target_lang=None,
              length_penalty=1.0):
    """Run direction-aware NLLB translation; preserve original row metadata."""
    import torch
    from collections import defaultdict
    if min(batch_size, max_length, max_new_tokens, beams) < 1:
        raise ValueError("Batch size, lengths and beams must be positive")
    if conditioning not in ("native", "text_prefix", "dedicated_token"):
        raise ValueError("Unknown conditioning strategy")
    rows = read_jsonl(input_path)
    model, tokenizer = load(model_id, revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    predictions = [None] * len(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        actual_src, actual_tgt = resolve_languages(row, src, tgt)
        ps = (proxy_source_lang or actual_src) if conditioning == "text_prefix" else actual_src
        pt = (proxy_target_lang or actual_tgt) if conditioning == "text_prefix" else actual_tgt
        require_language(tokenizer, ps)
        require_language(tokenizer, pt)
        groups[(actual_src, actual_tgt, ps, pt)].append(i)
    truncated = 0
    started = time.time()
    with torch.inference_mode():
        for (actual_src, actual_tgt, ps, pt), indices in groups.items():
            tokenizer.src_lang, tokenizer.tgt_lang = ps, pt
            target_id = require_language(tokenizer, pt)
            for k in range(0, len(indices), batch_size):
                batch_indices = indices[k:k+batch_size]
                batch = [condition_sentence(rows[i]["source"], actual_src, actual_tgt,
                                            "nllb", conditioning) for i in batch_indices]
                lengths = tokenizer(batch, truncation=False)["input_ids"]
                truncated += sum(len(ids) > max_length for ids in lengths)
                encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=max_length).to(device)
                tokens = model.generate(
                    **encoded, forced_bos_token_id=target_id,
                    max_new_tokens=max_new_tokens, num_beams=beams,
                    length_penalty=length_penalty)
                for i, text in zip(batch_indices, tokenizer.batch_decode(
                        tokens, skip_special_tokens=True)):
                    predictions[i] = text
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions):
            handle.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
    write_json(str(output) + ".manifest.json", {"model": str(model_id),
        "revision": revision, "resolved_revision": getattr(model.config, "_commit_hash", None),
        "src_lang": src, "tgt_lang": tgt, "conditioning": conditioning,
        "proxy_source_lang": proxy_source_lang, "proxy_target_lang": proxy_target_lang,
        "n": len(rows), "directions": len(groups), "truncated_sources": truncated,
        "input_sha256": file_hash(input_path), "output_sha256": file_hash(output),
        "num_beams": beams, "length_penalty": length_penalty, "max_length": max_length,
        "max_new_tokens": max_new_tokens, "seconds": time.time()-started,
        "device": device})
