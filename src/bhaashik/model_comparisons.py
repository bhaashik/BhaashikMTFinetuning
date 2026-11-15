"""Opt-in external MT baselines, always evaluated on the same development IDs.

No downloads, remote code, or model inference occur at import time.
"""
import importlib.util
import json
from pathlib import Path
import random

from .data import file_hash, read_jsonl, write_json
from .language_support import LANGUAGE_BY_TAG


INDICTRANS2_LANGS = {
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva",
    "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva",
    "mal_Mlym", "mar_Deva", "mni_Mtei", "npi_Deva", "ory_Orya",
    "pan_Guru", "san_Deva", "sat_Olck", "snd_Deva", "tam_Taml",
    "tel_Telu", "urd_Arab", "eng_Latn",
}
# The 23 languages explicitly listed on the present Hugging Face model card.
BHASHAVERSE_CARD = {
    "eng_Latn": "eng", "hin_Deva": "hin", "tel_Telu": "tel",
    "tam_Taml": "tam", "mal_Mlym": "mal", "kan_Knda": "kan",
    "ben_Beng": "ben", "guj_Gujr": "guj", "mar_Deva": "mar",
    "pan_Guru": "pan", "urd_Arab": "urd", "asm_Beng": "asm",
    "npi_Deva": "npi", "ory_Orya": "ory", "san_Deva": "san",
    "mai_Deva": "mai", "brx_Deva": "brx", "doi_Deva": "doi",
    "gom_Deva": "gom", "mni_Beng": "mni", "sat_Olck": "sat",
    "kas_Arab": "kas", "snd_Arab": "snd",
}


def select_dev_sample(dev_path, output_path, sample_size=100, seed=42):
    """Create a fixed, reproducible sample of *development* examples."""
    dev_path = Path(dev_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite sampled dev data: {output_path}")
    rows = read_jsonl(dev_path)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    selected = sorted(random.Random(seed).sample(range(len(rows)), min(sample_size, len(rows))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index in selected:
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    write_json(str(output_path) + ".manifest.json",
               {"split": "dev", "source_sha256": file_hash(dev_path),
                "seed": seed, "sample_size": len(selected), "ids": [rows[i]["id"] for i in selected]})
    return output_path


def _write_predictions(rows, predictions, output_path):
    predictions = list(predictions)
    if len(rows) != len(predictions):
        raise ValueError("Translator returned a different number of predictions")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, hypothesis in zip(rows, predictions):
            if not isinstance(hypothesis, str):
                raise TypeError("Each translation must be a string")
            handle.write(json.dumps({**row, "prediction": hypothesis},
                                    ensure_ascii=False) + "\n")
    return output_path


def run_indictrans2_dev(dev_path, output_path, src, tgt, model_id=None,
                        batch_size=4, allow_remote_code=False, beams=5):
    """Opt-in upstream IndicTrans2 with the official IndicProcessor pipeline.

    Requires optional IndicTransToolkit and accepted HF model terms. The model
    uses remote custom code, which callers must explicitly allow.
    """
    if src not in INDICTRANS2_LANGS or tgt not in INDICTRANS2_LANGS or src == tgt:
        raise ValueError(f"IndicTrans2 does not support the requested pair: {src}->{tgt}")
    if not allow_remote_code:
        raise ValueError("Review the upstream model code and set allow_remote_code=True to opt in")
    if batch_size < 1 or beams < 1:
        raise ValueError("batch_size and beams must be positive")
    if model_id is None:
        if src == "eng_Latn":
            model_id = "ai4bharat/indictrans2-en-indic-dist-200M"
        elif tgt == "eng_Latn":
            model_id = "ai4bharat/indictrans2-indic-en-dist-200M"
        else:
            model_id = "ai4bharat/indictrans2-indic-indic-dist-320M"
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    rows = read_jsonl(dev_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True).to(device).eval()
    processor = IndicProcessor(inference=True)
    predictions = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = [row["source"] for row in rows[i:i+batch_size]]
            prepared = processor.preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
            encoded = tokenizer(prepared, padding="longest", truncation=True,
                                return_tensors="pt", return_attention_mask=True).to(device)
            generated = model.generate(**encoded, num_beams=beams, max_length=256)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(processor.postprocess_batch(decoded, lang=tgt))
    result = _write_predictions(rows, predictions, output_path)
    write_json(str(result) + ".manifest.json",
               {"model": model_id, "method": "IndicTransToolkit", "src": src, "tgt": tgt,
                "input_sha256": file_hash(dev_path), "output_sha256": file_hash(result),
                "split": "dev", "n": len(rows), "beams": beams,
                "warning": "Custom model code enabled by explicit user opt-in"})
    return result


def run_bhashaverse_dev(dev_path, output_path, src, tgt, inference_script,
                        batch_size=4):
    """Use upstream *reviewed local* hf_inference.py; no generic MBart tokenizer.

    Only the languages explicitly documented on the Hub card are enabled here.
    Other paper-listed directions require separate checkpoint/code validation.
    """
    if src not in BHASHAVERSE_CARD or tgt not in BHASHAVERSE_CARD:
        raise ValueError(
            f"BhashaVerse Hub card does not verify {src}->{tgt}. "
            "Do not infer availability from the paper's 36-language claim.")
    script = Path(inference_script).resolve()
    if not script.is_file() or script.name != "hf_inference.py":
        raise FileNotFoundError("Provide the reviewed upstream hf_inference.py script path")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    spec = importlib.util.spec_from_file_location("bhashaverse_upstream_inference", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Intentionally executes user-reviewed source.
    if not hasattr(module, "translate_batch"):
        raise ValueError("Upstream inference module lacks translate_batch")
    rows = read_jsonl(dev_path)
    predictions = []
    for index in range(0, len(rows), batch_size):
        batch = [r["source"] for r in rows[index:index+batch_size]]
        translated = module.translate_batch(
            batch, sl=BHASHAVERSE_CARD[src], tl=BHASHAVERSE_CARD[tgt], batch_size=batch_size)
        predictions.extend(translated)
    result = _write_predictions(rows, predictions, output_path)
    write_json(str(result) + ".manifest.json",
               {"model": "ltrciiith/bhashaverse", "method": "upstream hf_inference.py",
                "script_sha256": file_hash(script), "split": "dev",
                "src": src, "tgt": tgt, "n": len(rows),
                "input_sha256": file_hash(dev_path), "output_sha256": file_hash(result)})
    return result
