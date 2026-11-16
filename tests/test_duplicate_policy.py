"""Repetitions may be corpus-native; leakage and split mistakes never are."""
import json

import pytest

from bhaashik.data import audit, read_jsonl, write_json
from bhaashik.data_splitting import split_parallel_jsonl, verify_split_integrity


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _pair(i, source=None, target=None):
    return {"id": f"pair-{i}", "source": source or f"sentence {i}",
            "target": target or f"अनुवाद {i}"}


def test_repeated_pairs_within_train_are_allowed_and_reported(tmp_path):
    train = _write(tmp_path / "train.jsonl", [
        _pair(1, "same sentence", "एक"),
        _pair(2, "same sentence", "एक"),
        _pair(3, "train-only", "दो"),
    ])
    dev = _write(tmp_path / "dev.jsonl", [_pair(4, "dev-only", "तीन")])
    test = _write(tmp_path / "test.jsonl", [_pair(5, "test-only", "चार")])
    paths = {"train": train, "dev": dev, "test": test}
    result = audit(paths)
    assert result["within_split_duplicate_pairs"] == {
        "train": 1, "dev": 0, "test": 0}
    with pytest.raises(ValueError, match="duplicate parallel pairs inside train"):
        audit(paths, reject_within_split_duplicates=True)


@pytest.mark.parametrize("where", ["dev", "test"])
def test_cross_split_same_pair_fails_even_with_repetition_allowed(tmp_path, where):
    train = _write(tmp_path / "train.jsonl", [
        _pair(1, "same sentence", "एक"),
        _pair(2, "same sentence", "एक"),
    ])
    other = _write(tmp_path / (where + ".jsonl"), [
        _pair(3, "same sentence", "एक"),
    ])
    with pytest.raises(ValueError, match="leakage between train and " + where):
        audit({"train": train, where: other})


def test_source_with_different_target_across_splits_is_still_blocked(tmp_path):
    train = _write(tmp_path / "train.jsonl", [_pair(1, "shared", "एक")])
    test = _write(tmp_path / "test.jsonl", [_pair(2, "shared", "दो")])
    with pytest.raises(ValueError, match="source leakage"):
        audit({"train": train, "test": test})


def test_split_preserves_each_corpus_record_even_with_repeated_pairs(tmp_path):
    original = [_pair(i) for i in range(30)]
    original[2]["source"] = original[20]["source"]
    original[2]["target"] = original[20]["target"]
    original[3]["source"] = original[20]["source"]
    original[3]["target"] = original[20]["target"]
    input_path = _write(tmp_path / "parallel.jsonl", original)
    folder = tmp_path / "split"
    m = split_parallel_jsonl(input_path, folder, validate=False)
    assert m["split_integrity"]["partition_complete"] is True
    assert m["split_integrity"]["rows_verified"] == len(original)
    paths = {part: folder / f"{part}.jsonl" for part in ("train", "dev", "test")}
    result = audit(paths)
    assert sum(result["within_split_duplicate_pairs"].values()) >= 2
    assert sum(m["counts"].values()) == len(original)
    m2 = split_parallel_jsonl(input_path, folder, validate=False)
    assert m == m2
    with pytest.raises(ValueError, match="duplicate parallel pairs"):
        split_parallel_jsonl(input_path, tmp_path / "strict",
                             reject_within_split_duplicates=True)


def test_detects_accidentally_duplicated_and_missing_split_records(tmp_path):
    rows = [_pair(i) for i in range(12)]
    original = _write(tmp_path / "input.jsonl", rows)
    folder = tmp_path / "split"
    split_parallel_jsonl(original, folder)
    paths = {s: folder / (s + ".jsonl") for s in ("train", "dev", "test")}
    first = read_jsonl(paths["train"])[0]
    with paths["train"].open("a", encoding="utf-8") as out:
        out.write(json.dumps({**first, "id": "accidental-extra"}) + "\n")
    with pytest.raises(ValueError, match="Split integrity failure"):
        verify_split_integrity(rows, paths)
    with pytest.raises(ValueError, match="Saved split differs"):
        split_parallel_jsonl(original, folder)
