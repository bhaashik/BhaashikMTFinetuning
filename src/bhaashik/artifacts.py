"""Archive only selected MT experiment artifacts, not every run in the workspace."""
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_STORED


def archive_experiment(root, run_dir, destination, baseline_dir=None,
                       config_path=None, include_checkpoints=False, report_dir=None):
    """Save one run, its baseline, and its config in a quick ZIP_STORED archive.

    By default Trainer optimizer/checkpoint snapshots are excluded; the exported
    model, tokenizer and evaluation files under run_dir are retained. Set
    include_checkpoints=True to preserve trainer state for exact resumption.\n    Pass report_dir to include pre-training diagnostics and development comparisons.
    """
    root = Path(root).resolve()

    def located(path):
        path = Path(path)
        full = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not full.is_relative_to(root):
            raise ValueError(f"Refusing to archive files outside repository: {path}")
        return full

    run = located(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {run}")

    baseline = located(baseline_dir) if baseline_dir is not None else None
    config = located(config_path) if config_path is not None else None

    paths = [run]
    if baseline is not None and baseline != run and baseline.is_dir():
        paths.append(baseline)
    reports = located(report_dir) if report_dir is not None else None
    if reports is not None:
        if not reports.is_dir():
            raise FileNotFoundError(f"Review reports do not exist: {reports}")
        if reports not in paths:
            paths.append(reports)
    if config is not None and not config.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config}")

    destination = Path(destination).resolve()
    if destination.suffix.lower() != ".zip":
        raise ValueError("Destination must have a .zip extension")
    if destination.exists():
        raise FileExistsError(f"Archive already exists: {destination}")

    # Build the file list before writing to avoid partial ZIPs on input errors.
    files = set()
    for folder in paths:
        for file in folder.rglob("*"):
            if not file.is_file() or file.is_symlink():
                continue
            if not file.resolve().is_relative_to(root):
                raise ValueError(f"File points outside repository: {file}")
            if not include_checkpoints and any(
                    folder.name == "checkpoints"
                    for folder in file.parents if folder == run or run in folder.parents):
                continue
            files.add(file)
    if config is not None:
        if config.is_symlink():
            raise ValueError(f"Configuration is a symbolic link: {config}")
        files.add(config)
    if not files:
        raise ValueError(f"No experiment files found in {run}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # ZIP_STORED duplicates model bytes while archiving. Avoid filling
    # /kaggle/working with an archive that cannot fit.
    archive_bytes = sum(file.stat().st_size for file in files)
    available = shutil.disk_usage(destination.parent).free
    reserve = 256 * 1024 * 1024
    if archive_bytes + reserve > available:
        raise OSError(
            f"Not enough free disk for stage ZIP at {destination.parent}: "
            f"need approximately {(archive_bytes + reserve) / 2**30:.2f} GiB "
            f"(including safety reserve), have {available / 2**30:.2f} GiB. "
            "The stage files are intact. Free space, disable save_archive, "
            "or move exported results to persistent storage.")
    try:
        with ZipFile(destination, mode="x", compression=ZIP_STORED, allowZip64=True) as archive:
            for file in sorted(files):
                archive.write(file, arcname=file.relative_to(root).as_posix())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination
