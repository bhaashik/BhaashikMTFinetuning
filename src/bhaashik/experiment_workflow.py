"""Common experiment configuration and data preparation for model notebooks.

Only the first configuration cell should contain editable experiment choices.
Downstream cells use the returned paths; no global flags are required.
"""
from pathlib import Path
import json
import re

from .data import audit, file_hash, read_jsonl, write_json
from .data_splitting import split_parallel_jsonl
from .language_support import LANGUAGE_BY_TAG
from .multilingual import require_routes
from .text_conversion import convert_aligned_text_files, convert_parallel_tsv

VALID_MODES = ("aligned_text", "tsv", "jsonl", "pre_split", "example")
STAGES = ("lowest_baseline", "low_baseline", "decoding_search", "language_extension")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def runtime_root():
    """Absolute checkout path for Kaggle, Colab or current local repo."""
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/BhaashikMTFinetuning")
    if Path("/content").is_dir():
        return Path("/content/BhaashikMTFinetuning")
    return Path.cwd().resolve()


def resolve_path(root, value):
    if value is None:
        raise ValueError("A required input path is not configured")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _require_string(cfg, key):
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set a nonempty value for {key!r} in the settings cell")
    return value


def check_settings(cfg, *, model_family):
    """Validate top-level settings without downloading models or touching files."""
    if not isinstance(cfg, dict):
        raise TypeError("EXPERIMENT must be a dictionary")
    if model_family not in ("nllb", "indictrans2", "bhashaverse"):
        raise ValueError(f"Unknown model family: {model_family}")
    mode = cfg.get("data_mode", "aligned_text")
    if mode not in VALID_MODES:
        raise ValueError(f"data_mode must be one of {VALID_MODES}")
    stage = cfg.get("stage", "lowest_baseline")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if "result_plots" in cfg and not isinstance(cfg["result_plots"], bool):
        raise ValueError("result_plots must be True or False")
    if "search_grid" in cfg:
        grid = cfg["search_grid"]
        if not isinstance(grid, (tuple, list)) or not grid:
            raise ValueError("search_grid must contain at least one candidate")
        for item in grid:
            if (not isinstance(item, dict)
                    or isinstance(item.get("beams"), bool)
                    or not isinstance(item.get("beams"), int)
                    or item["beams"] < 1
                    or not isinstance(item.get("length_penalty"), (float, int))
                    or item["length_penalty"] <= 0):
                raise ValueError("Each decoding-search candidate requires positive beams and length_penalty")
    if stage == "language_extension":
        if cfg.get("finetune_mode") != "full":
            raise ValueError("Language extension requires full finetuning to train new embeddings")
        if model_family == "nllb":
            if cfg.get("conditioning") != "dedicated_token" or not cfg.get("extend_language"):
                raise ValueError("NLLB extension needs dedicated_token and extend_language")
            if cfg["extend_language"] not in (cfg["source_lang"], cfg["target_lang"]):
                raise ValueError("extend_language must match the corpus source or target tag")
        if model_family == "indictrans2" and cfg.get("conditioning") != "dedicated_token":
            raise ValueError("IndicTrans2 extension needs dedicated_token")
        if model_family == "bhashaverse" and cfg.get("conditioning") != "text_prefix":
            raise ValueError("BhashaVerse requires original or explicitly repaired prefix assets")
    if stage in ("low_baseline", "language_extension") and cfg.get("allow_training") is not True:
        raise ValueError("This stage requires allow_training=True and a fresh GPU session")
    source = _require_string(cfg, "source_lang")
    target = _require_string(cfg, "target_lang")
    from .multilingual import resolve_languages
    resolve_languages({}, source, target)
    for name in ("dataset_id", "run_label"):
        value = _require_string(cfg, name)
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(f"{name} should contain letters, digits, dots, dashes and underscores only")
    for name in ("split_seed", "train_seed"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    # Older notebooks explicitly set reporting flags. New staged notebooks
    # use one 'stage' switch instead; those legacy flags are optional.
    for name in ("validate_data", "reject_within_split_duplicates", "data_report",
                 "pretrain_baseline", "posttrain_test",
                 "publication_report", "save_archive", "include_checkpoints", "allow_training"):
        if name in cfg and not isinstance(cfg[name], bool):
            raise ValueError(f"{name} must be set to True or False")
    if "stage" not in cfg and cfg.get("publication_report") and not cfg.get("posttrain_test"):
        raise ValueError("publication_report=True requires posttrain_test=True")
    if cfg.get("finetune_mode") not in ("lora", "full"):
        raise ValueError("finetune_mode must be lora or full")
    if cfg.get("precision") not in ("fp16", "bf16", "fp32"):
        raise ValueError("precision must be fp16, bf16 or fp32")
    if not isinstance(cfg.get("batch_size"), int) or cfg["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(cfg.get("gradient_accumulation_steps"), int) or cfg["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    for key in ("epochs", "learning_rate"):
        value = cfg.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key} must be positive")
    if model_family == "nllb":
        strategy = cfg.get("conditioning", "native")
        if strategy not in ("native", "text_prefix", "dedicated_token"):
            raise ValueError("Unknown NLLB conditioning")
        if strategy == "native":
            for tag in (source, target):
                lang = LANGUAGE_BY_TAG.get(tag)
                if lang is not None and not lang.nllb_native:
                    raise ValueError(f"{lang.name} has no native NLLB tag; choose an explicit extension experiment")
        if strategy == "dedicated_token" and cfg["finetune_mode"] != "full":
            raise ValueError("New NLLB language embeddings require full finetuning")
        if strategy == "text_prefix":
            for key in ("proxy_source_lang", "proxy_target_lang"):
                if cfg.get(key) is None:
                    raise ValueError(f"Explicit {key} required for text-prefix experiments")
    if mode == "example" and (source, target) != ("bho_Deva", "hin_Deva"):
        raise ValueError("Repository example corpus is Bhojpuri to Hindi only; set its tags or use your own data")
    return cfg


def prepare_parallel_data(root, cfg, *, model_family, print_summary=True):
    """Prepare shared train/dev/test paths from one configured source format.

    If validation is disabled, only basic schema checks and split integrity
    checks run. Training functions still call audit() on train/dev.
    """
    check_settings(cfg, model_family=model_family)
    root = Path(root).resolve()
    pair = f"{cfg['source_lang'].split('_')[0]}-{cfg['target_lang'].split('_')[0]}"
    mode = cfg["data_mode"]
    run_data_root = root / "data" / "local" / pair / cfg["dataset_id"]
    if mode == "example":
        data_dir = root / "data" / "examples"
        converted = None
        split_manifest = None
    elif mode == "pre_split":
        data_dir = resolve_path(root, cfg.get("split_directory"))
        converted = None
        split_manifest = None
    else:
        if mode == "aligned_text":
            converted = run_data_root / "parallel.jsonl"
            convert_aligned_text_files(
                resolve_path(root, cfg.get("source_text")),
                resolve_path(root, cfg.get("target_text")),
                converted, source_lang=cfg["source_lang"], target_lang=cfg["target_lang"],
                id_prefix=f"{pair}-{cfg['dataset_id']}",
            )
        elif mode == "tsv":
            converted = run_data_root / "parallel.jsonl"
            convert_parallel_tsv(
                resolve_path(root, cfg.get("tsv_file")),
                converted, source_lang=cfg["source_lang"], target_lang=cfg["target_lang"],
                has_header=cfg.get("tsv_has_header", False),
                id_prefix=f"{pair}-{cfg['dataset_id']}",
            )
        else:
            converted = resolve_path(root, cfg.get("parallel_jsonl"))
        split_id = "-".join(str(round(float(r)*1000)) for r in cfg.get("split_ratios", (.8, .1, .1)))
        data_dir = run_data_root / f"splits-s{cfg['split_seed']}-{split_id}"
        split_manifest = split_parallel_jsonl(
            converted, data_dir,
            seed=cfg["split_seed"], ratios=cfg.get("split_ratios", (.8, .1, .1)),
            source_lang=cfg["source_lang"], target_lang=cfg["target_lang"],
            group_by_document=cfg.get("group_by_document", True),
            validate=cfg.get("validate_data", False),
            reject_within_split_duplicates=cfg.get("reject_within_split_duplicates", False),
        )
    paths = {key: data_dir / f"{key}.jsonl" for key in ("train", "dev", "test")}
    # ALWAYS prevent cross-split leakage, even in quick/pilot mode.
    # Corpus-native duplicate pairs *within* a split are permitted by default.
    split_audit = audit(
        paths,
        reject_within_split_duplicates=cfg.get("reject_within_split_duplicates", False),
        enforce_document_disjoint=cfg.get("group_by_document", True),
    )
    counts = {}
    for split, path in paths.items():
        rows, routes = require_routes(path, cfg["source_lang"], cfg["target_lang"])
        counts[split] = len(rows)
        if len(routes) != 1 or (cfg["source_lang"], cfg["target_lang"]) not in routes:
            raise ValueError(f"{split} contains routes other than the selected language pair: {routes}")
    if print_summary:
        print(f"Prepared {pair} ({mode}), seed {cfg['split_seed']}: "
              + ", ".join(f"{key}={n}" for key, n in counts.items()))
        print(f"Data directory: {data_dir}")
        print("Cross-split leakage check: ON (mandatory)")
        print("Repeated pairs inside a split:",
              "REJECT" if cfg.get("reject_within_split_duplicates", False) else "ALLOW")
        print("Extended corpus diagnostics:",
              "enabled" if cfg.get("validate_data", False) else "OFF (quick experiment)")
    return {
        "pair": pair, "data_dir": data_dir, "paths": paths,
        "counts": counts, "split_audit": split_audit, "converted_jsonl": converted,
        "split_manifest": split_manifest,
        "data_sha256": {part: file_hash(path) for part, path in paths.items()},
    }


def experiment_paths(root, cfg, *, family, prepared=None):
    """Single naming convention for all three notebooks."""
    pair = prepared["pair"] if prepared else f"{cfg['source_lang'].split('_')[0]}-{cfg['target_lang'].split('_')[0]}"
    slug = f"{family}-{pair}-{cfg['run_label']}-s{cfg['train_seed']}"
    root = Path(root).resolve()
    stage = cfg.get("stage", "lowest_baseline")
    return {
        "run": root / "runs" / slug,
        "stage": root / "runs" / slug / stage,
        "report": root / "runs" / f"{slug}-report",
        "config": root / "configs" / f"{slug}-{stage}.json",
        "archive": root.parent / f"{slug}-{stage}-artifacts.zip",
    }


def training_parameters(cfg, prepared, paths, *, model_family):
    """Create a complete, saved trainer config from the one settings cell."""
    common = {
        "train_file": str(prepared["paths"]["train"]),
        "dev_file": str(prepared["paths"]["dev"]),
        "output_dir": str(paths.get("stage", paths["run"]) / "model"),
        "mode": cfg["finetune_mode"], "seed": cfg["train_seed"],
        "group_by_document": cfg.get("group_by_document", True),
        "reject_within_split_duplicates": cfg.get("reject_within_split_duplicates", False),
        "batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "max_length": cfg["max_length"], "max_steps": cfg["max_steps"],
        "epochs": cfg["epochs"], "precision": cfg["precision"],
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
    }
    if model_family == "nllb":
        common.update({
            "model": cfg["model_id"], "revision": cfg.get("model_revision"),
            "src_lang": cfg["source_lang"], "tgt_lang": cfg["target_lang"],
            "conditioning": cfg.get("conditioning", "native"),
            "lora_rank": cfg.get("lora_rank", 8),
            "learning_rate": cfg["learning_rate"],
        })
        if cfg.get("conditioning") == "text_prefix":
            common["proxy_source_lang"] = cfg["proxy_source_lang"]
            common["proxy_target_lang"] = cfg["proxy_target_lang"]
    elif model_family == "indictrans2":
        common.update({
            "model": cfg.get("model_id"), "strategy": cfg.get("conditioning", "native"),
            "proxy_src": cfg.get("proxy_source_lang"),
            "proxy_tgt": cfg.get("proxy_target_lang"),
            "allow_remote_code": cfg.get("allow_remote_code", False),
            "learning_rate": cfg["learning_rate"], "lora_rank": cfg.get("lora_rank", 8),
        })
    elif model_family == "bhashaverse":
        common.update({
            "model_assets": str(resolve_path(paths["run"].parents[1], cfg["model_assets"])),
            "strategy": "text_prefix",
            "learning_rate": cfg["learning_rate"], "lora_rank": cfg.get("lora_rank", 8),
        })
    else:
        raise ValueError(model_family)
    return common


def write_experiment_config(cfg, prepared, paths, trainer_cfg):
    """Write immutable config and environment-independent data manifest."""
    config_path = paths["config"]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": cfg, "trainer": trainer_cfg,
        "data_sha256": prepared["data_sha256"],
        "data_counts": prepared["counts"],
        "split_manifest": prepared["split_manifest"],
    }
    # Convert Path objects supplied by callers to strings for JSON.
    serializable = json.loads(json.dumps(record, default=str))
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        # Review-only flags can change AFTER training without creating a new
        # checkpoint path; preserve the exact original training configuration.
        # Trainer settings and data hashes must NEVER silently change.
        stable_keys = ("trainer", "data_sha256", "data_counts", "split_manifest")
        immutable_experiment_keys = (
            "run_label", "dataset_id", "split_seed", "train_seed",
            "data_mode", "source_lang", "target_lang", "stage",
        )
        changed_identity = any(
            old.get("experiment", {}).get(key) != cfg.get(key)
            for key in immutable_experiment_keys
        )
        if changed_identity or any(old.get(key) != serializable[key]
                                   for key in stable_keys):
            raise FileExistsError(
                f"Saved training settings/data differ; choose a new run_label: {config_path}")
        if old != serializable:
            print("Review-only settings changed; original training config preserved.")
    else:
        write_json(config_path, serializable)
    # Keep a trainer-only config next to it for existing train(config_path).
    train_path = config_path.with_name(config_path.stem + "-train.json")
    if train_path.exists():
        if json.loads(train_path.read_text(encoding="utf-8")) != trainer_cfg:
            raise FileExistsError(f"Trainer settings differ; choose a new run_label: {train_path}")
    else:
        write_json(train_path, trainer_cfg)
    return train_path


def report_data_if_enabled(cfg, prepared, report_dir):
    if not cfg.get("data_report", False):
        print("Data plots/reports OFF. Set data_report=True for publication preparation.")
        return None
    from .pretrain_review import inspect_splits, plot_review
    report = inspect_splits(prepared["paths"], report_dir)
    for issue in report["issues"]:
        print(issue["severity"], issue["check"], issue["message"])
    for path in plot_review(report_dir):
        print("Figure:", path)
    return report


def save_optional_archive(root, cfg, paths, *, baseline=None, report=None):
    if not cfg.get("save_archive", True):
        print("Artifact ZIP OFF. Set save_archive=True to package this stage.")
        return None
    from .artifacts import archive_experiment
    destination = paths["archive"]
    return archive_experiment(
        root, paths.get("stage", paths["run"]), destination,
        config_path=paths["config"],
        baseline_dir=baseline if baseline and Path(baseline).is_dir() else None,
        report_dir=report if report and Path(report).is_dir() else None,
        include_checkpoints=cfg.get("include_checkpoints", False),
    )
