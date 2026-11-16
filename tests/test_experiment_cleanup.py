"""Safety contract for preview-only and confirmed stage-scoped cleanup."""
from pathlib import Path

import pytest

from bhaashik.experiment_cleanup import cleanup_stage
from bhaashik.experiment_workflow import experiment_paths


def _experiment(root, *, stage="low_baseline"):
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    cfg = {
        "run_label": "mistake1", "dataset_id": "d",
        "source_lang": "bho_Deva", "target_lang": "hin_Deva",
        "train_seed": 42, "stage": stage,
    }
    paths = experiment_paths(root, cfg, family="nllb")
    paths["stage"].mkdir(parents=True)
    (paths["stage"] / "stage_manifest.json").write_text("partial experiment")
    (paths["stage"] / "checkpoint.bin").write_bytes(b"generated")
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    paths["config"].write_text("wrong settings")
    train_config = paths["config"].with_name(paths["config"].stem + "-train.json")
    train_config.write_text("wrong train settings")
    paths["archive"].write_bytes(b"generated zip")
    protected = root / "data" / "local" / "bho-hin" / "parallel.jsonl"
    protected.parent.mkdir(parents=True)
    protected.write_text("original corpus")
    other = paths["run"] / "lowest_baseline" / "stage_results.json"
    other.parent.mkdir()
    other.write_text("good baseline")
    return cfg, paths, protected, other


def test_cleanup_preview_and_confirmed_stage_deletion_only(tmp_path):
    cfg, paths, protected, other = _experiment(tmp_path)
    initial = cleanup_stage(tmp_path, cfg, family="nllb", action="preview")
    assert not initial["deleted"]
    assert initial["file_count"] >= 4
    assert paths["stage"].exists() and paths["archive"].exists()
    with pytest.raises(ValueError, match="Deletion not authorized"):
        cleanup_stage(tmp_path, cfg, family="nllb", action="delete",
                      confirmation="DELETE something-else")
    completed = cleanup_stage(
        tmp_path, cfg, family="nllb", action="delete",
        confirmation=initial["confirmation_required"],
    )
    assert completed["deleted"]
    assert not paths["stage"].exists()
    assert not paths["archive"].exists()
    assert not paths["config"].exists()
    assert protected.read_text() == "original corpus"
    assert other.read_text() == "good baseline"
    assert (tmp_path / "pyproject.toml").exists()


def test_can_cleanup_other_stage_without_deleting_selected_stage(tmp_path):
    cfg, paths, protected, good = _experiment(tmp_path)
    preview = cleanup_stage(tmp_path, cfg, family="nllb", stage="lowest_baseline")
    assert any("lowest_baseline" in p for p in preview["existing_paths"])
    assert paths["stage"].is_dir()
    assert good.is_file()


def test_cleanup_refuses_bad_targets_and_unsafe_paths(tmp_path):
    cfg, paths, *_ = _experiment(tmp_path)
    with pytest.raises(ValueError, match="target stage"):
        cleanup_stage(tmp_path, cfg, family="nllb", stage="../../data")
    with pytest.raises(ValueError, match="Unsafe cleanup"):
        cleanup_stage(tmp_path, {**cfg, "run_label": "../data"},
                      family="nllb")
    with pytest.raises(ValueError, match="Unsafe cleanup"):
        cleanup_stage(tmp_path, {**cfg, "source_lang": "../test"},
                      family="nllb")
    with pytest.raises(ValueError, match="Cleanup action"):
        cleanup_stage(tmp_path, cfg, family="nllb", action="overwrite")


def test_cleanup_refuses_symlinked_stage(tmp_path):
    cfg, paths, *_ = _experiment(tmp_path)
    external = tmp_path.parent / (tmp_path.name + "-outside")
    external.mkdir()
    (external / "do-not-delete.txt").write_text("protected")
    link = paths["stage"] / "link"
    link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        cleanup_stage(tmp_path, cfg, family="nllb", action="preview")
    assert (external / "do-not-delete.txt").exists()
