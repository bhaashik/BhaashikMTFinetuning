"""Offline tests of four-stage MT workflows with mocked model inference."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from bhaashik.data import file_hash, read_jsonl, write_json
from bhaashik.experiment_workflow import check_settings, experiment_paths
from bhaashik.staged_experiments import run_stage


def fixture(tmp_path, stage="lowest_baseline", **changes):
    files = {}
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for part in ("train", "dev", "test"):
        path = data_dir / f"{part}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for i in range(4):
                out.write(json.dumps({
                    "id": f"{part}{i}", "source": f"sentence {part}{i}",
                    "target": f"अनुवाद {part}{i}", "source_lang": "bho_Deva",
                    "target_lang": "hin_Deva"}, ensure_ascii=False) + "\n")
        files[part] = path
    config = {
        "stage": stage, "run_label": "a", "dataset_id": "d",
        "source_lang": "bho_Deva", "target_lang": "hin_Deva",
        "data_mode": "pre_split", "split_directory": str(data_dir),
        "split_seed": 42, "train_seed": 42, "finetune_mode": "full",
        "conditioning": "native", "allow_training": False,
        "model_id": "dummy-nllb", "batch_size": 1, "max_length": 32,
        "max_new_tokens": 12, "beams": 4, "length_penalty": 1.0,
        "result_plots": True, "save_archive": True,
        "include_checkpoints": False, "precision": "fp32",
        "gradient_accumulation_steps": 1, "epochs": 1,
        "learning_rate": 1e-5, "max_steps": -1,
        "search_metric": "chrfpp",
        "search_grid": [{"beams": 1, "length_penalty": 1.0},
                        {"beams": 4, "length_penalty": 1.0}],
    }
    config.update(changes)
    paths = experiment_paths(tmp_path, config, family="nllb")
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    write_json(paths["config"], config)
    train_config = tmp_path / "train.json"
    write_json(train_config, {})
    return config, {
        "paths": files, "data_sha256": {part: file_hash(path)
                                       for part, path in files.items()},
    }, paths, train_config


def fake_generation(model, data, output, config, *, family, decoding):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dest:
        for item in read_jsonl(data):
            predicted = item["target"] if decoding["beams"] == 4 else "गलत"
            dest.write(json.dumps({**item, "prediction": predicted},
                                  ensure_ascii=False) + "\n")
    write_json(str(output) + ".manifest.json", {
        "stage_data_sha256": file_hash(data),
        "stage_decoding": decoding, "stage_model": str(model),
        "stage_output_sha256": file_hash(output),
    })
    return output


def test_untouched_model_baseline_evaluates_both_splits_and_archives(
        tmp_path, monkeypatch):
    import bhaashik.staged_experiments as stages
    monkeypatch.setattr(stages, "check_model_access", lambda *args, **kwargs: True)
    monkeypatch.setattr(stages, "_generate", fake_generation)
    cfg, data, paths, train_cfg = fixture(tmp_path)
    result = run_stage(tmp_path, cfg, data, paths, train_cfg, family="nllb")
    assert result["status"] == "complete"
    assert set(result["predictions"]) == {"dev", "test"}
    assert result["dev_scores"]["chrfpp"] == pytest.approx(100)
    assert result["test_scores"]["chrfpp"] == pytest.approx(100)
    assert paths["archive"].is_file()
    assert (paths["stage"] / "dev_test_scores.svg").is_file()
    assert (paths["stage"] / "dev_review.csv").is_file()
    with ZipFile(paths["archive"]) as zipfile:
        names = zipfile.namelist()
        assert any(x.endswith("test_predictions.jsonl") for x in names)
        assert any(x.endswith("dev_scores.json") for x in names)
        assert any(x.endswith("stage_results.json") for x in names)


def test_incomplete_baseline_is_still_archived(tmp_path, monkeypatch):
    import bhaashik.staged_experiments as stages
    monkeypatch.setattr(stages, "check_model_access", lambda *args, **kwargs: True)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic error")
    monkeypatch.setattr(stages, "_generate", fail)
    cfg, data, paths, train_cfg = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic error"):
        run_stage(tmp_path, cfg, data, paths, train_cfg, family="nllb")
    assert paths["archive"].exists()
    assert json.loads((paths["stage"] / "stage_manifest.json").read_text())["status"] == "incomplete"


def test_search_tunes_only_on_dev_before_one_test_pass(tmp_path, monkeypatch):
    import bhaashik.staged_experiments as stages
    cfg, data, paths, train_cfg = fixture(tmp_path, stage="decoding_search")
    trained = paths["run"] / "low_baseline" / "model"
    trained.mkdir(parents=True)
    write_json(trained / "run_manifest.json", {"mock": True})
    monkeypatch.setattr(stages, "check_model_access", lambda *args, **kwargs: True)
    calls = []
    def tracked(model, split, output, cfg, *, family, decoding):
        calls.append((split.name, decoding["beams"]))
        return fake_generation(model, split, output, cfg, family=family, decoding=decoding)
    monkeypatch.setattr(stages, "_generate", tracked)
    result = run_stage(tmp_path, cfg, data, paths, train_cfg, family="nllb")
    assert calls == [("dev.jsonl", 1), ("dev.jsonl", 4), ("test.jsonl", 4)]
    assert result["decoding_search"]["selected"]["beams"] == 4
    assert paths["archive"].exists()


def test_training_stage_preserves_loss_history_and_full_splits(tmp_path, monkeypatch):
    import bhaashik.staged_experiments as stages
    cfg, data, paths, train_cfg = fixture(tmp_path, "low_baseline", allow_training=True)
    monkeypatch.setattr(stages, "check_model_access", lambda *args, **kwargs: True)
    monkeypatch.setattr(stages, "_generate", fake_generation)
    def train_mock(root, cfg, paths, trainer_config, *, family):
        model = paths["stage"] / "model"
        model.mkdir(parents=True)
        write_json(model / "run_manifest.json", {"dev_loss": 2.1})
        write_json(model / "trainer_history.json",
                   [{"step": 1, "loss": 2.9}, {"step": 2, "eval_loss": 2.1}])
        return {"eval_loss": 2.1}
    monkeypatch.setattr(stages, "_train", train_mock)
    result = run_stage(tmp_path, cfg, data, paths, train_cfg, family="nllb")
    assert result["train_result"]["eval_loss"] == 2.1
    assert (paths["stage"] / "training_loss.csv").exists()
    assert (paths["stage"] / "training_loss.svg").exists()
    assert result["test_scores"]["chrfpp"] == pytest.approx(100)


def test_stage_validation_rejects_bad_transitions(tmp_path):
    cfg, *_ = fixture(tmp_path)
    assert check_settings(cfg, model_family="nllb")
    with pytest.raises(ValueError, match="allow_training"):
        check_settings({**cfg, "stage": "low_baseline"}, model_family="nllb")
    with pytest.raises(ValueError, match="dedicated_token"):
        check_settings({**cfg, "stage": "language_extension",
                        "allow_training": True}, model_family="nllb")
    with pytest.raises(ValueError, match="search_grid"):
        check_settings({**cfg, "search_grid": []}, model_family="nllb")
