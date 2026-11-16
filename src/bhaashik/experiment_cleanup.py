"""Preview-first cleanup for one explicitly named generated experiment stage.

Never deletes source corpora, split manifests, shared reports, original model
assets, repository files or unrelated run stages. No wildcard deletion.
"""
from pathlib import Path
import shutil

from .experiment_workflow import STAGES, experiment_paths


def cleanup_stage(root, cfg, *, family, stage=None, action="preview",
                  confirmation=""):
    """Preview or delete a specific generated stage and its ZIP/config.

    action='preview' NEVER modifies files.
    action='delete' requires the printed exact confirmation phrase.
    """
    if action not in ("preview", "delete"):
        raise ValueError("Cleanup action must be 'preview' or 'delete'")
    selected = stage or cfg.get("stage")
    if selected not in STAGES:
        raise ValueError(f"Cleanup target stage must be one of {STAGES}")
    root = Path(root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Refusing cleanup outside a recognized repository root")
    local = {**cfg, "stage": selected}
    paths = experiment_paths(root, local, family=family)
    run = paths["run"]
    stage_path = paths["stage"]
    config_path = paths["config"]
    archive_path = paths["archive"]
    trainer_path = config_path.with_name(config_path.stem + "-train.json")
    expected_paths = [stage_path, config_path, trainer_path, archive_path]
    if (run.parent.resolve() != (root / "runs").resolve()
            or stage_path.parent.resolve() != run.resolve()
            or config_path.parent.resolve() != (root / "configs").resolve()
            or archive_path.parent.resolve() != root.parent.resolve()
            or stage_path.name != selected):
        raise ValueError("Cleanup paths do not match the generated experiment layout")
    for item in expected_paths:
        if item.is_symlink():
            raise ValueError(f"Refusing symlinked cleanup target: {item}")
        if item.is_dir() and any(child.is_symlink() for child in item.rglob("*")):
            raise ValueError(f"Refusing symlink inside cleanup target: {item}")
    found = [p for p in expected_paths if p.exists()]
    count = sum(1 for p in found if p.is_file())
    count += sum(1 for p in found if p.is_dir() for child in p.rglob("*")
                 if child.is_file())
    total_bytes = sum(p.stat().st_size for p in found if p.is_file())
    total_bytes += sum(child.stat().st_size
                       for p in found if p.is_dir() for child in p.rglob("*")
                       if child.is_file())
    phrase = f"DELETE {run.name}/{selected}"
    plan = {
        "action": action, "stage": selected, "existing_paths": [str(p) for p in found],
        "file_count": count, "bytes": total_bytes,
        "confirmation_required": phrase,
        "protected": ["data/", "data/local/", "data/examples/",
                      "other experiment stages", "shared corpus reports",
                      "original downloaded model assets", "git repository"],
        "deleted": False,
    }
    print(f"Cleanup target: {run.name}/{selected}")
    print(f"Generated files: {count}; size: {total_bytes / 2**20:.1f} MiB")
    for item in found:
        print("  ", item)
    if action == "preview":
        print("DRY RUN ONLY. To delete, set cleanup_action='delete' and")
        print(f"cleanup_confirmation={phrase!r} in the settings cell.")
        return plan
    if confirmation != phrase:
        raise ValueError(f"Deletion not authorized. Required exact phrase: {phrase!r}")
    for item in found:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    plan["deleted"] = True
    print("Deleted only the selected generated stage and its config/archive.")
    return plan
