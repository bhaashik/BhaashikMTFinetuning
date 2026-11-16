"""Tests for single-cell experiments, path prep, TSV conversion and configurations."""
import json
from pathlib import Path

import pytest

from bhaashik.experiment_workflow import (
    check_settings, prepare_parallel_data, experiment_paths,
    training_parameters, write_experiment_config, report_data_if_enabled,
    resolve_path,
)
from bhaashik.text_conversion import convert_parallel_tsv
from bhaashik.data import read_jsonl


def _cfg(**changes):
    cfg = {
        "data_mode": "aligned_text", "source_lang": "bho_Deva",
        "target_lang": "hin_Deva", "dataset_id": "toy-v1",
        "source_text": "input/bho.txt", "target_text": "input/hin.txt",
        "split_seed": 42, "train_seed": 42, "split_ratios": (.8, .1, .1),
        "group_by_document": True,
        "validate_data": False, "data_report": False,
        "pretrain_baseline": False, "posttrain_test": False,
        "publication_report": False, "save_archive": False,
        "include_checkpoints": False, "allow_training": True,
        "finetune_mode": "lora", "precision": "fp32",
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "learning_rate": .0002, "epochs": 1, "max_steps": 1,
        "max_length": 64, "run_label": "pilot1",
        "model_id": "facebook/nllb-200-distilled-600M",
        "conditioning": "native",
        "model_revision": None,
    }
    return {**cfg, **changes}


def _input_pair(root, n=40):
    folder = root / "input"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bho.txt").write_text(
        "".join(f"भोजपुरी वाक्य {i}\n" for i in range(n)),
        encoding="utf-8")
    (folder / "hin.txt").write_text(
        "".join(f"हिंदी अनुवाद {i}\n" for i in range(n)),
        encoding="utf-8")


def test_quick_real_data_preparation_and_reuse(tmp_path):
    _input_pair(tmp_path)
    cfg = _cfg()
    check_settings(cfg, model_family="nllb")
    prepared = prepare_parallel_data(tmp_path, cfg, model_family="nllb")
    assert sum(prepared["counts"].values()) == 40
    assert prepared["counts"]["train"] > 0
    assert prepared["counts"]["dev"] > 0
    assert prepared["counts"]["test"] > 0
    assert prepared["converted_jsonl"].exists()
    assert (prepared["data_dir"] / "split_manifest.json").exists()
    assert all(row["source_lang"] == "bho_Deva"
               for row in read_jsonl(prepared["paths"]["train"]))
    assert prepare_parallel_data(tmp_path, cfg, model_family="nllb")[
        "data_sha256"] == prepared["data_sha256"]
    paths = experiment_paths(tmp_path, cfg, family="nllb", prepared=prepared)
    trainer = training_parameters(cfg, prepared, paths, model_family="nllb")
    saved = write_experiment_config(cfg, prepared, paths, trainer)
    assert saved.name.endswith("-train.json")
    assert json.loads(paths["config"].read_text())["experiment"]["data_mode"] == "aligned_text"
    assert trainer["conditioning"] == "native"
    assert trainer["max_steps"] == 1
    assert report_data_if_enabled(cfg, prepared, paths["report"]) is None
    assert write_experiment_config(cfg, prepared, paths, trainer) == saved


def test_one_cell_change_controls_split_and_model_run(tmp_path):
    _input_pair(tmp_path, 42)
    a = _cfg()
    b = _cfg(split_seed=13, train_seed=13, run_label="pilot2")
    pa = prepare_parallel_data(tmp_path, a, model_family="nllb")
    pb = prepare_parallel_data(tmp_path, b, model_family="nllb")
    assert pa["data_dir"] != pb["data_dir"]
    assert experiment_paths(tmp_path, a, family="nllb", prepared=pa)["run"] != (
        experiment_paths(tmp_path, b, family="nllb", prepared=pb)["run"])
    paths = experiment_paths(tmp_path, a, family="nllb", prepared=pa)
    trainer = training_parameters(a, pa, paths, model_family="nllb")
    write_experiment_config(a, pa, paths, trainer)
    changed_train = _cfg(epochs=2)
    new_trainer = training_parameters(changed_train, pa, paths, model_family="nllb")
    with pytest.raises(FileExistsError, match="Saved training settings/data differ"):
        write_experiment_config(changed_train, pa, paths, new_trainer)
    # Only optional, post-training reporting flags can change in-place.
    reporting = _cfg(allow_training=False, posttrain_test=True,
                     publication_report=True)
    assert write_experiment_config(reporting, pa, paths, trainer) == (
        paths["config"].with_name(paths["config"].stem + "-train.json"))


def test_one_file_tsv_mode_roundtrip(tmp_path):
    input_file = tmp_path / "input" / "pairs.tsv"
    input_file.parent.mkdir(parents=True)
    input_file.write_text("source\ttarget\n" + "".join(
        f"मैथिली वाक्य {i}\tहिंदी वाक्य {i}\n" for i in range(33)),
        encoding="utf-8")
    cfg = _cfg(data_mode="tsv", source_lang="mai_Deva",
               target_lang="hin_Deva", tsv_file="input/pairs.tsv",
               tsv_has_header=True, dataset_id="tsv-v1")
    prepared = prepare_parallel_data(tmp_path, cfg, model_family="nllb")
    assert sum(prepared["counts"].values()) == 33
    rows = read_jsonl(prepared["converted_jsonl"])
    assert rows[0]["source"].startswith("मैथिली")
    assert rows[0]["source_lang"] == "mai_Deva"
    assert prepare_parallel_data(tmp_path, cfg, model_family="nllb")["counts"] == prepared["counts"]
    with pytest.raises(FileExistsError):
        prepare_parallel_data(tmp_path, _cfg(**{
            **cfg, "tsv_has_header": False}), model_family="nllb")


def test_tsv_rejects_blank_or_extra_tab_without_output(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("one\tहिंदी\nsecond\t\n")
    with pytest.raises(ValueError, match="exactly two nonempty columns"):
        convert_parallel_tsv(path, tmp_path / "bad.jsonl",
                             source_lang="bho_Deva", target_lang="hin_Deva")
    assert not (tmp_path / "bad.jsonl").exists()
    path.write_text("one\tहिंदी\tother\n")
    with pytest.raises(ValueError):
        convert_parallel_tsv(path, tmp_path / "bad.jsonl",
                             source_lang="bho_Deva", target_lang="hin_Deva")


def test_config_validation_blocks_unsupported_tags_and_inconsistent_flags():
    with pytest.raises(ValueError, match="no native NLLB"):
        check_settings(_cfg(source_lang="bra_Deva"), model_family="nllb")
    with pytest.raises(ValueError, match="requires posttrain_test"):
        check_settings(_cfg(publication_report=True), model_family="nllb")
    with pytest.raises(ValueError, match="must be"):
        check_settings(_cfg(split_seed=-1), model_family="nllb")
    with pytest.raises(ValueError, match="full finetuning"):
        check_settings(_cfg(conditioning="dedicated_token"), model_family="nllb")
    assert check_settings(_cfg(data_mode="example"), model_family="nllb")


def test_example_data_mode(tmp_path):
    # Uses a temporary repo-like layout, no model downloads.
    sample = Path(__file__).resolve().parents[1] / "data" / "examples"
    path = tmp_path / "data" / "examples"
    path.mkdir(parents=True)
    for part in ("train", "dev", "test"):
        (path / f"{part}.jsonl").write_text(
            (sample / f"{part}.jsonl").read_text(encoding="utf-8"),
            encoding="utf-8")
    cfg = _cfg(data_mode="example")
    result = prepare_parallel_data(tmp_path, cfg, model_family="nllb")
    assert result["counts"] == {"train": 2, "dev": 2, "test": 2}
    with pytest.raises(ValueError, match="Bhojpuri"):
        check_settings(_cfg(data_mode="example", source_lang="mai_Deva"), model_family="nllb")
