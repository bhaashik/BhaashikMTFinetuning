"""Fast, selective archive tests using tiny synthetic run files."""
from zipfile import ZipFile
import pytest
from bhaashik.artifacts import archive_experiment


def test_only_current_run_and_baseline(tmp_path):
    root = tmp_path / "checkout"
    run = root / "runs" / "bho-hin-lora-s42"
    base = root / "runs" / "bho-hin-baseline"
    other = root / "runs" / "mai-hin-lora-s42"
    checkpoint = run / "checkpoints" / "checkpoint-2"
    config = root / "configs" / "bho-hin-lora-s42.json"
    for path in (run, base, other, checkpoint, config.parent):
        path.mkdir(parents=True, exist_ok=True)
    (run / "model.safetensors").write_bytes(b"MODEL")
    (run / "predictions.jsonl").write_text("prediction")
    (base / "predictions.jsonl").write_text("baseline")
    (other / "model.safetensors").write_bytes(b"UNRELATED")
    (checkpoint / "optimizer.pt").write_bytes(b"CHECKPOINT")
    config.write_text("{}")

    archive = tmp_path / "experiment.zip"
    result = archive_experiment(root, run, archive, base, config)
    assert result == archive
    with ZipFile(archive) as z:
        names = set(z.namelist())
        assert "runs/bho-hin-lora-s42/model.safetensors" in names
        assert "runs/bho-hin-lora-s42/predictions.jsonl" in names
        assert "runs/bho-hin-baseline/predictions.jsonl" in names
        assert "configs/bho-hin-lora-s42.json" in names
        assert not any("checkpoint" in name for name in names)
        assert not any("mai-hin" in name for name in names)

    with pytest.raises(FileExistsError):
        archive_experiment(root, run, archive, base, config)

    archive_with_checkpoints = tmp_path / "resume.zip"
    archive_experiment(root, run, archive_with_checkpoints, base, config,
                       include_checkpoints=True)
    with ZipFile(archive_with_checkpoints) as z:
        assert "runs/bho-hin-lora-s42/checkpoints/checkpoint-2/optimizer.pt" in z.namelist()


def test_rejects_paths_outside_repo(tmp_path):
    root = tmp_path / "repo"
    run = root / "runs" / "example"
    run.mkdir(parents=True)
    (run / "README.md").write_text("model")
    with pytest.raises(ValueError, match="outside repository"):
        archive_experiment(root, tmp_path / "other", tmp_path / "archive.zip")
