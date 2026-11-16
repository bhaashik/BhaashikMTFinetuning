"""Four-stage reproducible machine-translation evaluation workflow.

Stages:
  lowest_baseline: untouched model -> full dev and test predictions/reports.
  low_baseline: train -> dev loss + full dev/test predictions/reports.
  decoding_search: compare decoding on dev only, select once, evaluate test.
  language_extension: controlled tokenizer/embedding extension + training/reports.

Outputs are stage-scoped, independently archivable, with hashes and config.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import time

from .data import file_hash, read_jsonl, write_json
from .evaluation import evaluate, export_errors, metrics
from .experiment_workflow import STAGES, resolve_path


def source_model(cfg, *, family, root):
    if family == "nllb":
        return cfg["model_id"]
    if family == "indictrans2":
        from .indictrans2_adaptation import _model_id
        return cfg.get("model_id") or _model_id(
            cfg["source_lang"], cfg["target_lang"])
    if family == "bhashaverse":
        return str(resolve_path(root, cfg["model_assets"]))
    raise ValueError("Unknown model family")


def model_for_stage(root, cfg, paths, *, family):
    """Choose immutable base or already trained model, never guess."""
    stage = cfg["stage"]
    if stage == "lowest_baseline":
        return source_model(cfg, family=family, root=root)
    if stage == "decoding_search":
        selection = cfg.get("search_source_stage", "low_baseline")
        if selection not in ("low_baseline", "language_extension"):
            raise ValueError("search_source_stage must be low_baseline or language_extension")
        name = paths["run"] / selection / "model"
        if not (name / "run_manifest.json").exists():
            raise FileNotFoundError(
                f"Complete trained model missing: {name}. Run that stage first "
                "or restore it from your saved artifact before decoding search.")
        return str(name)
    return str(paths["stage"] / "model")


def check_model_access(cfg, *, family, root):
    """Fail before expensive inference if model-specific prerequisites absent."""
    if family == "indictrans2":
        if not cfg.get("allow_remote_code", False):
            raise ValueError("Review upstream IndicTrans2 custom code and set allow_remote_code=True")
        from .indictrans2_adaptation import processing_pair
        processing_pair(cfg["source_lang"], cfg["target_lang"],
                        strategy=cfg.get("conditioning", "native"),
                        proxy_src=cfg.get("proxy_source_lang"),
                        proxy_tgt=cfg.get("proxy_target_lang"))
    if family == "bhashaverse":
        from .preflight import FairseqCodec, checkpoint_layout
        from .bhashaverse_adaptation import validate_prefix_routes
        from .multilingual import resolve_languages
        assets = resolve_path(root, cfg["model_assets"])
        codec = FairseqCodec(assets)
        layout = checkpoint_layout(codec)
        if not layout["weight_layout_checked"] or layout["issues"]:
            raise ValueError("BhashaVerse must have a validated HF MBart local checkpoint")
    return True


def _generate(model, data, output, cfg, *, family, decoding):
    """Use unchanged model-specific tokenizers/preprocessors."""
    if output.exists():
        # A complete output is only reusable if the data and decoding match.
        manifest_path = Path(str(output) + ".manifest.json")
        if not manifest_path.exists():
            raise FileExistsError(f"Prediction output lacks provenance: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("stage_data_sha256") != file_hash(data)
                or manifest.get("stage_decoding") != decoding
                or manifest.get("stage_model") != str(model)
                or manifest.get("stage_output_sha256") != file_hash(output)):
            raise FileExistsError(f"Predictions have different data/model/decoding: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if family == "nllb":
        from .modeling import translate
        translate(
            model, cfg["source_lang"], cfg["target_lang"], data, output,
            revision=cfg.get("model_revision") if str(model) == cfg.get("model_id") else None,
            batch_size=cfg["batch_size"], max_length=cfg["max_length"],
            max_new_tokens=cfg["max_new_tokens"],
            beams=decoding["beams"], length_penalty=decoding["length_penalty"],
            conditioning=cfg.get("conditioning", "native"),
            proxy_source_lang=cfg.get("proxy_source_lang"),
            proxy_target_lang=cfg.get("proxy_target_lang"),
        )
    elif family == "indictrans2":
        from .indictrans2_adaptation import generate_indictrans2
        generate_indictrans2(
            model, data, output,
            strategy=cfg.get("conditioning", "native"),
            proxy_src=cfg.get("proxy_source_lang"),
            proxy_tgt=cfg.get("proxy_target_lang"),
            batch_size=cfg["batch_size"], max_length=cfg["max_length"],
            max_new_tokens=cfg["max_new_tokens"],
            beams=decoding["beams"],
            length_penalty=decoding["length_penalty"],
            allow_remote_code=cfg.get("allow_remote_code", False),
        )
    elif family == "bhashaverse":
        from .bhashaverse_adaptation import generate_bhashaverse_dev
        generate_bhashaverse_dev(
            model, data, output,
            batch_size=cfg["batch_size"], max_new_tokens=cfg["max_new_tokens"],
            beams=decoding["beams"], length_penalty=decoding["length_penalty"],
        )
    else:
        raise ValueError(family)
    existing_path = Path(str(output) + ".manifest.json")
    manifest = {}
    if existing_path.exists():
        manifest = json.loads(existing_path.read_text(encoding="utf-8"))
    manifest.update({
        "stage_model": str(model), "stage_data_sha256": file_hash(data),
        "stage_decoding": decoding,
        "stage_output_sha256": file_hash(output),
    })
    write_json(existing_path, manifest)
    return output


def evaluate_split(predictions, output_dir, split, *, plots=True, errors=True):
    """Evaluate a whole held-out partition and retain reviewable artifacts."""
    output_dir = Path(output_dir)
    result = evaluate(predictions, output_dir / f"{split}_scores.json")
    if errors:
        export_errors(predictions, output_dir / f"{split}_review.csv")
    return result


def _render_plot(scores, output_dir, title):
    """Always write dependency-free SVG; optionally render an additional PNG."""
    from xml.sax.saxutils import escape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("dev", "test"):
        if split in scores:
            for metric in ("bleu", "chrfpp"):
                rows.append((split, metric, scores[split]["scores"][metric]))
    if not rows:
        return []
    width = 650
    height = 80 + 48 * len(rows)
    shapes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="29" font-size="18" font-family="sans-serif">'
        f'{escape(title)}</text>',
    ]
    for i, (split, metric, value) in enumerate(rows):
        y = 54 + 48 * i
        bar_width = max(0, min(440, 4.4 * float(value)))
        shapes.extend([
            f'<text x="20" y="{y + 18}" font-size="13" font-family="sans-serif">'
            f'{escape(split + " " + metric)}</text>',
            f'<rect x="156" y="{y}" width="{bar_width:.1f}" height="25" fill="#296f91"/>',
            f'<text x="{165 + bar_width:.1f}" y="{y + 19}" font-size="13" '
            f'font-family="sans-serif">{value:.2f}</text>',
        ])
    shapes.append("</svg>")
    image = output_dir / "dev_test_scores.svg"
    image.write_text("\n".join(shapes) + "\n", encoding="utf-8")
    return [image]


def _training_history(model_dir, stage_dir):
    p = Path(model_dir) / "trainer_history.json"
    if not p.is_file():
        return None
    history = json.loads(p.read_text(encoding="utf-8"))
    write_json(stage_dir / "training_history.json", history)
    # A simple CSV suitable for future publication plots.
    import csv
    keys = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    with (stage_dir / "training_loss.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for entry in history:
            w.writerow({key: entry.get(key, "") for key in keys})
    return stage_dir / "training_loss.csv"


def _train(root, cfg, paths, trainer_config, *, family):
    model_dir = paths["stage"] / "model"
    if (model_dir / "run_manifest.json").is_file():
        raise FileExistsError(
            f"This stage is already trained: {model_dir}. "
            "Do not overwrite; choose a new run_label or run the report stage.")
    if not cfg.get("allow_training", False):
        raise ValueError("This stage needs allow_training=True")
    if family == "nllb":
        from .modeling import train
        if cfg["stage"] == "language_extension":
            from .modeling import extend
            if cfg.get("conditioning") != "dedicated_token":
                raise ValueError("NLLB extension requires dedicated_token")
            language = cfg.get("extend_language")
            if not language:
                raise ValueError("Set extend_language to the new source/target tag")
            extended = paths["stage"] / "extended-tokenizer-model"
            if not (extended / "bhaashik_extension.json").exists():
                extend(cfg["model_id"], extended, language,
                       donor=cfg.get("language_donor", "hin_Deva"),
                       revision=cfg.get("model_revision"))
            trainer_config["model"] = str(extended)
            trainer_config["revision"] = None
        return train(trainer_config)
    if family == "indictrans2":
        from .indictrans2_adaptation import train_indictrans2
        return train_indictrans2(trainer_config)
    if family == "bhashaverse":
        if cfg["stage"] == "language_extension":
            # Never rewrite SPM/fairseq IDs in this stage automatically.
            manifest = resolve_path(root, cfg["model_assets"]) / "repair_manifest.json"
            if not manifest.is_file():
                raise ValueError("First prepare a separately reviewed repaired BhashaVerse "
                                 "SPM/dictionary/embedding checkpoint and its repair_manifest.json")
        from .bhashaverse_adaptation import train_bhashaverse
        return train_bhashaverse(trainer_config)
    raise ValueError(family)


def _scores_for_stage(prediction_files, stage_dir):
    scores = {}
    for split, path in prediction_files.items():
        scores[split] = evaluate_split(path, stage_dir, split)
    return scores


def _compare_with_lowest(paths, stage_dir, scores):
    baseline = paths["run"] / "lowest_baseline"
    baseline_scores = {}
    for split in ("dev", "test"):
        p = baseline / f"{split}_scores.json"
        if p.is_file():
            baseline_scores[split] = json.loads(p.read_text(encoding="utf-8"))
    if not baseline_scores:
        return None
    comparison = {
        "baseline": "lowest_baseline",
        "candidate": str(stage_dir.name),
        "splits": {
            part: {
                metric: {
                    "untouched": baseline_scores[part]["scores"][metric],
                    "candidate": scores[part]["scores"][metric],
                    "difference": scores[part]["scores"][metric] - baseline_scores[part]["scores"][metric],
                }
                for metric in ("bleu", "chrfpp")
            }
            for part in ("dev", "test") if part in baseline_scores and part in scores
        },
    }
    write_json(stage_dir / "comparison_to_untouched.json", comparison)
    return comparison


def _search(root, cfg, prepared, paths, *, family):
    stage_dir = paths["stage"]
    model = model_for_stage(root, cfg, paths, family=family)
    grid = cfg["search_grid"]
    results = []
    for item in grid:
        decoding = {"beams": item["beams"], "length_penalty": float(item["length_penalty"])}
        name = f"beam{decoding['beams']}-lp{decoding['length_penalty']:g}"
        path = stage_dir / "candidates" / f"{name}-dev.jsonl"
        _generate(model, prepared["paths"]["dev"], path, cfg,
                  family=family, decoding=decoding)
        score = metrics(
            [r["prediction"] for r in read_jsonl(path, predictions=True)],
            [r["target"] for r in read_jsonl(path, predictions=True)])
        results.append({"candidate": decoding, "dev_scores": score,
                        "predictions": str(path), "prediction_sha256": file_hash(path)})
    criterion = cfg.get("search_metric", "chrfpp")
    if criterion not in ("chrfpp", "bleu"):
        raise ValueError("search_metric must be 'chrfpp' or 'bleu'")
    best = max(results, key=lambda r: r["dev_scores"][criterion])
    # Freeze selection BEFORE touching the test set.
    report = {
        "selected_on": "dev", "metric": criterion,
        "candidates": results, "selected": best["candidate"],
        "note": "Do not inspect or use test performance when choosing a decoding policy.",
    }
    write_json(stage_dir / "decoding_search.json", report)
    selected_dev = stage_dir / "dev_predictions.jsonl"
    if not selected_dev.exists():
        import shutil
        shutil.copyfile(best["predictions"], selected_dev)
        import shutil
        shutil.copyfile(best["predictions"] + ".manifest.json",
                        str(selected_dev) + ".manifest.json")
    selected_test = stage_dir / "test_predictions.jsonl"
    _generate(model, prepared["paths"]["test"], selected_test, cfg,
              family=family, decoding=best["candidate"])
    return {"dev": selected_dev, "test": selected_test}, report


def run_stage(root, cfg, prepared, paths, trainer_cfg_path, *, family):
    """Execute exactly one stage; preserve baseline/evaluation independently of training."""
    stage = cfg["stage"]
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    stage_dir = paths["stage"]
    if (stage_dir / "stage_results.json").is_file():
        raise FileExistsError(f"Stage already complete: {stage_dir}. Change run_label.")
    stage_dir.mkdir(parents=True, exist_ok=True)
    check_model_access(cfg, family=family, root=root)
    started = time.time()
    stage_manifest = {
        "stage": stage, "model_family": family, "status": "started",
        "input_sha256": prepared["data_sha256"],
        "config_path": str(paths["config"]),
        "model_origin": source_model(cfg, family=family, root=root),
        "utc_started": datetime.now(timezone.utc).isoformat(),
    }
    write_json(stage_dir / "stage_manifest.json", stage_manifest)
    try:
        train_result = None
        search_report = None
        if stage in ("low_baseline", "language_extension"):
            trainer_cfg = json.loads(Path(trainer_cfg_path).read_text(encoding="utf-8"))
            train_result = _train(root, cfg, paths, trainer_cfg_path, family=family)
            _training_history(stage_dir / "model", stage_dir)
            model = model_for_stage(root, cfg, paths, family=family)
        elif stage == "decoding_search":
            model = model_for_stage(root, cfg, paths, family=family)
        else:
            model = model_for_stage(root, cfg, paths, family=family)
        if stage == "decoding_search":
            outputs, search_report = _search(root, cfg, prepared, paths, family=family)
            decode = search_report["selected"]
        else:
            decode = {
                "beams": cfg.get("beams", 4 if family != "bhashaverse" else 1),
                "length_penalty": float(cfg.get("length_penalty", 1.0)),
            }
            outputs = {}
            for part in ("dev", "test"):
                dest = stage_dir / f"{part}_predictions.jsonl"
                _generate(model, prepared["paths"][part], dest, cfg,
                          family=family, decoding=decode)
                outputs[part] = dest
        scores = _scores_for_stage(outputs, stage_dir)
        comparison = (_compare_with_lowest(paths, stage_dir, scores)
                      if stage != "lowest_baseline" else None)
        charts = (_render_plot(scores, stage_dir, f"{family} · {stage}")
                  if cfg.get("result_plots", True) else [])
        results = {
            "status": "complete", "stage": stage, "family": family,
            "data_sha256": prepared["data_sha256"],
            "model": str(model), "decoding": decode,
            "train_result": train_result,
            "dev_scores": scores["dev"]["scores"],
            "test_scores": scores["test"]["scores"],
            "predictions": {key: str(path) for key, path in outputs.items()},
            "output_hashes": {key: file_hash(path) for key, path in outputs.items()},
            "comparison_to_untouched": comparison,
            "decoding_search": search_report,
            "plots": [str(c) for c in charts],
            "seconds": time.time() - started,
            "note": ("All dev/test scores are descriptive for baseline stages. "
                     "Only decoding_search ranks candidates on dev; test is evaluated "
                     "once after selection. Corpus quality requires human review."),
        }
        write_json(stage_dir / "stage_results.json", results)
        stage_manifest["status"] = "complete"
        stage_manifest["utc_finished"] = datetime.now(timezone.utc).isoformat()
        write_json(stage_dir / "stage_manifest.json", stage_manifest)
        return results
    except BaseException as error:
        stage_manifest["status"] = "incomplete"
        stage_manifest["error"] = str(error)
        write_json(stage_dir / "stage_manifest.json", stage_manifest)
        raise
