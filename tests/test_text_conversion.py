"""Aligned UTF-8 sentence file conversion tests (no model downloads)."""
import json
from pathlib import Path

import pytest

from bhaashik.data import read_jsonl, file_hash
from bhaashik.text_conversion import convert_aligned_text_files
from bhaashik.data_splitting import split_parallel_jsonl


def test_aligned_text_to_tagged_jsonl_and_seeded_splits(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("\ufeff" + "\n".join(f"मैं वाक्य {i}" for i in range(40)) + "\n",
                      encoding="utf-8")
    target.write_text("\n".join(f"अलग अनुवाद {i}" for i in range(40)) + "\n",
                      encoding="utf-8")
    original_sha = (file_hash(source), file_hash(target))
    out = tmp_path / "work" / "parallel.jsonl"
    result = convert_aligned_text_files(source, target, out,
                                        source_lang="bra_Deva", target_lang="hin_Deva",
                                        domain="fiction", provenance="test-fixture")
    assert result["sentence_pairs"] == 40
    assert (Path(str(out) + ".manifest.json")).is_file()
    assert result == convert_aligned_text_files(source, target, out,
                                                source_lang="bra_Deva", target_lang="hin_Deva",
                                                domain="fiction", provenance="test-fixture")
    rows = read_jsonl(out)
    assert rows[0]["source"] == "मैं वाक्य 0"
    assert rows[0]["id"] == "bra-hin-00000001"
    assert rows[0]["source_lang"] == "bra_Deva"
    assert rows[0]["target_lang"] == "hin_Deva"
    assert rows[0]["domain"] == "fiction"
    assert rows[1]["source"] == "मैं वाक्य 1"
    splits = split_parallel_jsonl(out, tmp_path / "split", seed=23,
                                  ratios=(0.8, 0.1, 0.1), validate=True)
    assert sum(splits["counts"].values()) == 40
    assert all(value > 0 for value in splits["counts"].values())
    assert (file_hash(source), file_hash(target)) == original_sha


@pytest.mark.parametrize("source_text,target_text,matched", [
    ("one\ntwo\n", "एक\n", "line counts differ"),
    ("one\n\ntwo\n", "एक\nदो\nतीन\n", "Blank source"),
    ("one\ntwo\n", "एक\n\n", "Blank source"),
    ("", "", "empty"),
])
def test_reject_misaligned_or_blank_files(tmp_path, source_text, target_text, matched):
    source, target = tmp_path / "a.txt", tmp_path / "b.txt"
    source.write_text(source_text)
    target.write_text(target_text)
    with pytest.raises(ValueError, match=matched):
        convert_aligned_text_files(source, target, tmp_path / "out.jsonl",
                                   source_lang="bho_Deva", target_lang="hin_Deva")
    assert not (tmp_path / "out.jsonl").exists()
    assert not (tmp_path / "out.jsonl.manifest.json").exists()


def test_refuses_overwrite_and_changed_inputs(tmp_path):
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    source.write_text("one\ntwo\nthree\n")
    target.write_text("एक\nदो\nतीन\n")
    out = tmp_path / "parallel.jsonl"
    convert_aligned_text_files(source, target, out,
                               source_lang="awa_Deva", target_lang="hin_Deva")
    source.write_text("one changed\ntwo\nthree\n")
    with pytest.raises(FileExistsError, match="different"):
        convert_aligned_text_files(source, target, out,
                                   source_lang="awa_Deva", target_lang="hin_Deva")
    source.write_text("one\ntwo\nthree\n")
    out.write_text("tamper!")
    with pytest.raises(ValueError, match="differs"):
        convert_aligned_text_files(source, target, out,
                                   source_lang="awa_Deva", target_lang="hin_Deva")


def test_rejects_same_file_and_invalid_tags(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("hello")
    with pytest.raises(ValueError, match="distinct"):
        convert_aligned_text_files(source, source, tmp_path / "out.jsonl",
                                   source_lang="mai_Deva", target_lang="hin_Deva")
    with pytest.raises(ValueError, match="invalid source_lang"):
        convert_aligned_text_files(source, tmp_path / "missing.txt", tmp_path / "out.jsonl",
                                   source_lang="maithili", target_lang="hin_Deva")
