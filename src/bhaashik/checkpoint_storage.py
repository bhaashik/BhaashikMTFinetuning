"""Disk-aware Hugging Face Trainer checkpoints with optional Drive backup.

Kaggle: train checkpoints under /kaggle/tmp, preserving /kaggle/working
for the exported model/reports. The scratch directory is ephemeral.
Optional rclone backups use a Kaggle Secret containing rclone.conf; secrets
are never written to the tracked repository or saved notebook outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .data import write_json


STORAGE_MODES = ("auto", "working", "scratch", "custom")
BACKUP_MODES = ("none", "rclone", "mounted_directory")


def _scratch_root():
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/tmp/bhaashik-checkpoints")
    if Path("/content").is_dir():
        return Path("/content/bhaashik-checkpoints")
    return Path(tempfile.gettempdir()) / "bhaashik-checkpoints"


def checkpoint_directory(model_output, config):
    """Choose a stable, separate Trainer checkpoint directory."""
    model_output = Path(model_output).resolve()
    mode = config.get("checkpoint_storage", "auto")
    if mode not in STORAGE_MODES:
        raise ValueError(f"checkpoint_storage must be one of {STORAGE_MODES}")
    if mode == "working":
        return model_output / "checkpoints"
    base = (config.get("checkpoint_scratch_root") if mode in ("auto", "scratch")
            else config.get("checkpoint_custom_root"))
    if mode == "custom" and not base:
        raise ValueError("Set checkpoint_custom_root for custom checkpoint storage")
    base = Path(base).expanduser().resolve() if base else _scratch_root().resolve()
    # Include a reproducible hash to prevent unrelated runs colliding in scratch.
    slug = model_output.parent.parent.name + "-" + model_output.parent.name
    key = hashlib.sha256(str(model_output).encode("utf-8")).hexdigest()[:12]
    target = base / f"{slug}-{key}" / "checkpoints"
    if target == model_output or target.is_relative_to(model_output):
        raise ValueError("Checkpoint destination overlaps the exported model")
    return target


def checkpoint_training_options(config, model_output):
    """Validate settings and preflight free storage BEFORE allocating a GPU model."""
    path = checkpoint_directory(model_output, config)
    resume = config.get("resume_from_checkpoint")
    model_only = config.get("checkpoint_model_only", True)
    if not isinstance(model_only, bool):
        raise ValueError("checkpoint_model_only must be True or False")
    if resume and model_only:
        raise ValueError(
            "Exact resume needs optimizer state: set checkpoint_model_only=False")
    limit = config.get("checkpoint_limit", 1)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("checkpoint_limit must be a positive integer")
    backup = config.get("checkpoint_backup", "none")
    if backup not in BACKUP_MODES:
        raise ValueError(f"checkpoint_backup must be one of {BACKUP_MODES}")
    if backup == "rclone" and not config.get("checkpoint_rclone_remote"):
        raise ValueError("Set checkpoint_rclone_remote (e.g. 'drive:Bhaashik/checkpoints')")
    if backup == "mounted_directory" and not config.get("checkpoint_backup_directory"):
        raise ValueError("Set checkpoint_backup_directory for mounted_directory backups")
    free_gib = config.get("checkpoint_min_free_gib", 5)
    if (not isinstance(free_gib, (int, float)) or isinstance(free_gib, bool)
            or not 0 <= free_gib < 1000):
        raise ValueError("checkpoint_min_free_gib must be a nonnegative number")
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < free_gib * (1024 ** 3):
        raise OSError(
            f"Insufficient checkpoint space at {path}: "
            f"{free / 2**30:.1f} GiB free, minimum {free_gib:.1f} GiB. "
            "Select a larger scratch/custom filesystem or lower the threshold "
            "only after verifying checkpoint size.")
    if not resume and any(path.iterdir()):
        raise FileExistsError(
            f"Old checkpoint data already exists: {path}; choose a new run_label "
            "or explicitly restore a full resumable checkpoint.")
    return {
        "output_dir": str(path),
        "save_strategy": "epoch",
        "save_total_limit": limit,
        "save_only_model": model_only,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


def _private_rclone_environment(config):
    """Obtain a configured remote, optionally from one Kaggle Secret."""
    if not shutil.which("rclone"):
        raise RuntimeError(
            "rclone executable is missing. Install rclone in the notebook "
            "before training, or select checkpoint_backup='none'.")
    env = os.environ.copy()
    secret_name = config.get("checkpoint_rclone_secret")
    if secret_name:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError as exc:
            raise RuntimeError(
                "Kaggle Secrets are unavailable. Configure rclone locally "
                "and leave checkpoint_rclone_secret empty.") from exc
        secret_value = UserSecretsClient().get_secret(secret_name)
        if not secret_value or "[drive]" not in secret_value and "[Drive]" not in secret_value:
            # Custom remote names are accepted: only require INI-style content.
            if not secret_value or "[" not in secret_value or "]" not in secret_value:
                raise ValueError("The Kaggle secret must contain a complete rclone.conf")
        private_root = _scratch_root().parent / "bhaashik-private"
        private_root.mkdir(parents=True, exist_ok=True)
        private_config = private_root / "rclone.conf"
        # Do not expose the secret in a command line or notebook output.
        private_config.write_text(secret_value, encoding="utf-8")
        private_config.chmod(0o600)
        env["RCLONE_CONFIG"] = str(private_config)
    remote = config["checkpoint_rclone_remote"]
    if not isinstance(remote, str) or ":" not in remote:
        raise ValueError("checkpoint_rclone_remote must use 'remote:path' syntax")
    configured = subprocess.run(
        ["rclone", "listremotes"], env=env, check=True, capture_output=True, text=True)
    registered = remote.split(":", 1)[0] + ":"
    if registered not in configured.stdout.splitlines():
        raise ValueError(f"Rclone remote {registered} is not configured")
    return env


def backup_preflight(config):
    """Fail fast on unavailable external backup before hours of training."""
    mode = config.get("checkpoint_backup", "none")
    if mode == "none":
        return None
    if mode == "rclone":
        return _private_rclone_environment(config)
    if mode == "mounted_directory":
        directory = Path(config["checkpoint_backup_directory"]).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Backup mount is unavailable: {directory}. In Colab mount Drive first.")
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"Backup mount is not writable: {directory}")
        return None
    raise ValueError(f"Unrecognized checkpoint backup: {mode}")


def backup_checkpoint(checkpoint, config, env=None):
    """Copy and verify a completed checkpoint; never delete its local source."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Completed checkpoint missing: {checkpoint}")
    mode = config.get("checkpoint_backup", "none")
    if mode == "none":
        return None
    if mode == "rclone":
        base = config["checkpoint_rclone_remote"].rstrip("/")
        target = base + "/" + checkpoint.parent.parent.name + "/" + checkpoint.name
        environment = env or _private_rclone_environment(config)
        # copy (not sync) never deletes remote data; check verifies the contents.
        subprocess.run(
            ["rclone", "copy", str(checkpoint), target, "--checksum"],
            env=environment, check=True)
        subprocess.run(
            ["rclone", "check", str(checkpoint), target, "--one-way"],
            env=environment, check=True)
        return target
    if mode == "mounted_directory":
        base = Path(config["checkpoint_backup_directory"]).expanduser().resolve()
        target = base / checkpoint.parent.parent.name / checkpoint.name
        if target.exists():
            raise FileExistsError(f"Backup destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint, target)
        for source in checkpoint.rglob("*"):
            if source.is_file():
                counterpart = target / source.relative_to(checkpoint)
                if _sha256(source) != _sha256(counterpart):
                    raise OSError(f"Checkpoint backup failed verification: {source}")
        return str(target)
    raise ValueError(f"Unrecognized backup mode: {mode}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_backup_callback(config):
    """Lazily construct HF Trainer callback for every completed checkpoint."""
    backup_preflight(config)
    from transformers import TrainerCallback

    class BackupOnSave(TrainerCallback):
        def __init__(self):
            self.environment = (backup_preflight(config)
                                if config.get("checkpoint_backup") != "none" else None)

        def on_save(self, args, state, control, **kwargs):
            if config.get("checkpoint_backup", "none") == "none":
                return control
            name = f"checkpoint-{state.global_step}"
            directory = Path(args.output_dir) / name
            target = backup_checkpoint(directory, config, env=self.environment)
            print(f"Verified external checkpoint backup: {target}")
            return control

    return BackupOnSave()


def cleanup_scratch_checkpoints(config, model_output):
    """Optional post-export cleanup; NEVER deletes persistent/custom/Drive paths."""
    if not config.get("checkpoint_cleanup_on_success", True):
        return False
    if config.get("checkpoint_storage", "auto") not in ("auto", "scratch"):
        return False
    directory = checkpoint_directory(model_output, config)
    root = Path(config.get("checkpoint_scratch_root") or _scratch_root()).resolve()
    if not directory.is_relative_to(root) or directory == root or directory.is_symlink():
        raise ValueError("Refusing to delete checkpoints outside the scratch root")
    if directory.is_dir():
        shutil.rmtree(directory)
        # Only our own per-run scratch directory; leave shared scratch untouched.
        if directory.parent.is_dir() and not any(directory.parent.iterdir()):
            directory.parent.rmdir()
        return True
    return False


def checkpoint_storage_report(config, model_output):
    path = checkpoint_directory(model_output, config)
    total, used, free = shutil.disk_usage(path if path.exists() else path.parent)
    return {
        "checkpoint_dir": str(path),
        "storage": config.get("checkpoint_storage", "auto"),
        "save_only_model": config.get("checkpoint_model_only", True),
        "checkpoint_limit": config.get("checkpoint_limit", 1),
        "backup": config.get("checkpoint_backup", "none"),
        "free_gib": round(free / 2**30, 2),
    }
