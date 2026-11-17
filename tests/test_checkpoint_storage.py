"""Checkpoint location, retention, backup verification and safety tests."""
import os
from pathlib import Path
import shutil

import pytest

from bhaashik.checkpoint_storage import (
    backup_checkpoint, backup_preflight, checkpoint_directory,
    checkpoint_training_options, cleanup_scratch_checkpoints,
)


def test_scratch_checkpoints_are_separate_and_model_only(tmp_path):
    model = tmp_path / "repo" / "runs" / "nllb-bho-hin-pilot-s42" / "low_baseline" / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_limit": 1, "checkpoint_model_only": True,
           "checkpoint_min_free_gib": 0}
    options = checkpoint_training_options(cfg, model)
    assert options["output_dir"].startswith(str(tmp_path / "scratch"))
    assert options["save_only_model"] is True
    assert options["save_total_limit"] == 1
    assert options["load_best_model_at_end"] is True
    assert options["save_strategy"] == "epoch"
    assert options["metric_for_best_model"] == "eval_loss"
    assert not Path(options["output_dir"]).is_relative_to(model)
    assert checkpoint_directory(model, cfg) == Path(options["output_dir"])
    with pytest.raises(FileExistsError, match="Old checkpoint"):
        (Path(options["output_dir"]) / "existing.bin").write_bytes(b"x")
        checkpoint_training_options(cfg, model)


def test_low_space_preflight_and_exact_resume_guard(tmp_path):
    model = tmp_path / "output" / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_min_free_gib": 999, "checkpoint_model_only": True}
    with pytest.raises(OSError, match="Insufficient checkpoint space"):
        checkpoint_training_options(cfg, model)
    with pytest.raises(ValueError, match="Exact resume"):
        checkpoint_training_options(
            {**cfg, "resume_from_checkpoint": "some-checkpoint",
             "checkpoint_min_free_gib": 0}, model)
    with pytest.raises(ValueError, match="checkpoint_limit"):
        checkpoint_training_options(
            {**cfg, "checkpoint_min_free_gib": 0, "checkpoint_limit": 0}, model)


def test_mounted_directory_backups_are_verified_and_preserved(tmp_path):
    model = tmp_path / "model"
    scratch = tmp_path / "scratch"
    backup = tmp_path / "drive-mounted"
    backup.mkdir()
    cfg = {"checkpoint_storage": "scratch",
           "checkpoint_scratch_root": str(scratch),
           "checkpoint_min_free_gib": 0,
           "checkpoint_backup": "mounted_directory",
           "checkpoint_backup_directory": str(backup),
           "checkpoint_cleanup_on_success": True}
    path = Path(checkpoint_training_options(cfg, model)["output_dir"])
    checkpoint = path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"saved model data")
    (checkpoint / "trainer_state.json").write_text("{}")
    backup_preflight(cfg)
    saved = Path(backup_checkpoint(checkpoint, cfg))
    assert (saved / "model.safetensors").read_bytes() == b"saved model data"
    assert (saved / "trainer_state.json").read_text() == "{}"
    with pytest.raises(FileExistsError, match="Backup destination exists"):
        backup_checkpoint(checkpoint, cfg)
    assert cleanup_scratch_checkpoints(cfg, model) is True
    assert not path.exists()
    assert (saved / "model.safetensors").exists()


def test_no_backup_does_not_require_rclone(tmp_path):
    model = tmp_path / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_min_free_gib": 0,
           "checkpoint_backup": "none"}
    opts = checkpoint_training_options(cfg, model)
    assert backup_preflight(cfg) is None
    assert Path(opts["output_dir"]).is_dir()
    assert backup_checkpoint(Path(opts["output_dir"]), cfg) is None


def test_missing_mount_and_no_cleanup_of_custom_storage(tmp_path):
    model = tmp_path / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_min_free_gib": 0,
           "checkpoint_backup": "mounted_directory",
           "checkpoint_backup_directory": str(tmp_path / "not-mounted")}
    with pytest.raises(FileNotFoundError, match="unavailable"):
        backup_preflight(cfg)
    opts = checkpoint_training_options(cfg, model)
    (Path(opts["output_dir"]) / "keep.bin").write_bytes(b"keep")
    assert cleanup_scratch_checkpoints(cfg, model) is False
    assert (Path(opts["output_dir"]) / "keep.bin").exists()


def test_rclone_configuration_fails_fast_without_remote(tmp_path, monkeypatch):
    model = tmp_path / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_min_free_gib": 0,
           "checkpoint_backup": "rclone"}
    with pytest.raises(ValueError, match="checkpoint_rclone_remote"):
        checkpoint_training_options(cfg, model)
    monkeypatch.setattr(shutil, "which", lambda command: None)
    with pytest.raises(RuntimeError, match="rclone executable"):
        backup_preflight({**cfg, "checkpoint_rclone_remote": "drive:checkpoints"})


def test_pinned_huggingface_training_arguments_accept_compact_checkpoint_options(tmp_path):
    from transformers import Seq2SeqTrainingArguments

    model_dir = tmp_path / "run" / "model"
    cfg = {"checkpoint_storage": "custom",
           "checkpoint_custom_root": str(tmp_path / "scratch"),
           "checkpoint_min_free_gib": 0,
           "checkpoint_model_only": True, "checkpoint_limit": 1}
    options = checkpoint_training_options(cfg, model_dir)
    args = Seq2SeqTrainingArguments(
        **options, eval_strategy="epoch",
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        report_to="none", use_cpu=True,
    )
    assert args.save_only_model is True
    assert args.save_total_limit == 1
    assert args.load_best_model_at_end is True
    assert str(args.output_dir).startswith(str(tmp_path / "scratch"))
