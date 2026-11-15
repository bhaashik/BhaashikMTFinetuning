"""Seeded single-file splitting tests; no model downloads required."""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from bhaashik.data import audit, file_hash, read_jsonl
from bhaashik.data_splitting import split_parallel_jsonl


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _dataset(count=60):
    rows = []
    for i in range(count):
        rows.append({
            "id": f"item-{i:03d}",
            "source": f"Language sample {i}",
            "target": f"अनुवाद {i}",
            "source_lang": "bra_Deva" if i % 2 else "mai_Deva",
            "target_lang": "hin_Deva",
            "document_id": f"document-{i // 2}",
            "domain": "synthetic",
        })
    return rows


def test_reproducible_three_way_split_and_manifest(tmp_path):
    source = _write(tmp_path / "input" / "parallel.jsonl", _dataset())
    out = tmp_path / "out" / "seed42"
    manifest = split_parallel_jsonl(source, out, seed=42, ratios=(.8, .1, .1),
                                    validate=True)
    assert manifest["seed"] == 42
    assert manifest["counts"]["train"] > manifest["counts"]["dev"] > 0
    assert manifest["counts"]["test"] > 0
    assert sum(manifest["counts"].values()) == 60
    assert manifest["validation_performed_at_creation"]
    assert audit({name: out / f"{name}.jsonl"
                  for name in ("train", "dev", "test")})["counts"] == manifest["counts"]
    assert manifest == split_parallel_jsonl(source, out, seed=42,
                                            ratios=(.8, .1, .1), validate=True)
    all_ids = []
    documents = {}
    for part in ("train", "dev", "test"):
        for row in read_jsonl(out / f"{part}.jsonl"):
            all_ids.append(row["id"])
            if row["document_id"] in documents:
                assert documents[row["document_id"]] == part
            documents[row["document_id"]] = part
    assert len(all_ids) == len(set(all_ids)) == 60
    assert manifest["input_sha256"] == file_hash(source)
    assert manifest["output_sha256"]["train"] == file_hash(out / "train.jsonl")


def test_seed_changes_assignments_and_reuse_guard(tmp_path):
    source = _write(tmp_path / "parallel.jsonl", _dataset(90))
    a = tmp_path / "seed1"
    b = tmp_path / "seed2"
    split_parallel_jsonl(source, a, seed=1)
    split_parallel_jsonl(source, b, seed=2)
    ids_a = {r["id"] for r in read_jsonl(a / "train.jsonl")}
    ids_b = {r["id"] for r in read_jsonl(b / "train.jsonl")}
    assert ids_a != ids_b
    with pytest.raises(FileExistsError, match="different"):
        split_parallel_jsonl(source, a, seed=2)
    (a / "dev.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        split_parallel_jsonl(source, a, seed=1)


def test_single_file_language_defaults_and_minimum_schema(tmp_path):
    source = _write(tmp_path / "parallel.jsonl", [
        {"id": str(i), "source": f"Source {i}", "target": f"लक्ष्य {i}"}
        for i in range(12)
    ])
    out = tmp_path / "split"
    report = split_parallel_jsonl(source, out, seed=9,
                                  ratios=(.5, .25, .25),
                                  source_lang="mwr_Deva", target_lang="hin_Deva")
    assert not report["validation_performed_at_creation"]
    rows = [r for name in ("train", "dev", "test")
            for r in read_jsonl(out / f"{name}.jsonl")]
    assert all(r["source_lang"] == "mwr_Deva" and
               r["target_lang"] == "hin_Deva" for r in rows)
    assert "source_lang" not in read_jsonl(source)[0]
    validated = split_parallel_jsonl(source, out, seed=9, ratios=(.5, .25, .25),
                                     source_lang="mwr_Deva",
                                     target_lang="hin_Deva", validate=True)
    assert validated == report


def test_repeated_source_and_document_stay_together(tmp_path):
    rows = _dataset(30)
    rows[2]["source"] = rows[20]["source"]
    rows[2]["document_id"] = "doc-2"
    rows[20]["document_id"] = "doc-20"
    source = _write(tmp_path / "parallel.jsonl", rows)
    out = tmp_path / "splits"
    manifest = split_parallel_jsonl(source, out, seed=17,
                                    ratios=(.7, .2, .1))
    assert sum(manifest["counts"].values()) == 30
    labels = {}
    for part in ("train", "dev", "test"):
        for row in read_jsonl(out / f"{part}.jsonl"):
            labels[row["id"]] = part
    assert labels["item-002"] == labels["item-020"]
    for i in range(0, 30, 2):
        if i not in (2, 20):
            assert labels[f"item-{i:03d}"] == labels[f"item-{i+1:03d}"]


def test_refuse_too_few_groups_and_wrong_ratios_without_writing(tmp_path):
    file = _write(tmp_path / "parallel.jsonl", [
        {"id": str(i), "source": "same source", "target": f"target {i}"}
        for i in range(12)
    ])
    with pytest.raises(ValueError, match="independent"):
        split_parallel_jsonl(file, tmp_path / "no", validate=False)
    assert not (tmp_path / "no").exists()
    for ratios in ((.8, .2, .2), (.8, -.1, .3), (.5, .5, 0), (1, 0, 0)):
        with pytest.raises(ValueError):
            split_parallel_jsonl(file, tmp_path / "bad", ratios=ratios)
    with pytest.raises(ValueError, match="both language"):
        split_parallel_jsonl(file, tmp_path / "bad", source_lang="bra_Deva")
    assert not (tmp_path / "bad").exists()


def test_document_grouping_optional(tmp_path):
    rows = _dataset(60)
    source = _write(tmp_path / "data.jsonl", rows)
    report = split_parallel_jsonl(source, tmp_path / "ungrouped",
                                  seed=8, group_by_document=False)
    assert report["group_count"] == 60
    assert sum(report["counts"].values()) == 60
    # No claims of document independence when group_by_document=False.


def test_non_destructive_and_input_conflict(tmp_path):
    original = tmp_path / "parallel.jsonl"
    _write(original, _dataset(20))
    before = file_hash(original)
    with pytest.raises(ValueError, match="not contain"):
        split_parallel_jsonl(original, tmp_path)
    assert file_hash(original) == before
    assert not (tmp_path / "train.jsonl").exists()
