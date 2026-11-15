"""Offline tests for corpus diagnostics and aligned dev comparisons."""
import json
from pathlib import Path
import pytest

from bhaashik.pretrain_review import inspect_splits, compare_dev, plot_review
from bhaashik.model_comparisons import (
    select_dev_sample, _write_predictions, run_indictrans2_dev, run_bhashaverse_dev,
)
from bhaashik.language_support import language_options, require_native_nllb


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _row(key, source, target):
    return {"id": key, "source": source, "target": target,
            "document_id": "doc-" + key, "domain": "fiction",
            "license": "CC0", "provenance": "toy-fixture"}


@pytest.fixture
def splits(tmp_path):
    rows = {
        "train": [_row("1", "हम किताब पढ़त बानी", "मैं किताब पढ़ रहा हूँ"),
                  _row("2", "तुम कहाँ जात बाड़", "तुम कहाँ जा रहे हो")],
        "dev": [_row("3", "आज मौसम बढ़िया बा", "आज मौसम अच्छा है"),
                _row("4", "किताब एहाँ रख द", "किताब यहाँ रख दो")],
        "test": [_row("5", "काल्ह स्कूल बन्द बा", "कल स्कूल बन्द है"),
                 _row("6", "पानी ले आव", "पानी ले आओ")],
    }
    return {k: _write(tmp_path / (k + ".jsonl"), value) for k, value in rows.items()}


def test_inspect_splits_generates_reports_without_training(splits, tmp_path):
    report = inspect_splits(splits, tmp_path / "review")
    assert report["ready_for_training"]
    assert report["audit"]["counts"] == {"train": 2, "dev": 2, "test": 2}
    assert report["splits"]["train"]["rows_with_license"] == 2
    assert report["splits"]["train"]["script_letters"]["source"]["Devanagari"] > 0
    assert (tmp_path / "review" / "sentence_lengths.csv").is_file()
    assert (tmp_path / "review" / "corpus_diagnostics.json").is_file()


def test_blocks_cross_split_leakage(tmp_path):
    row = _row("same", "hello", "नमस्ते")
    paths = {name: _write(tmp_path / (name + ".jsonl"), [row])
             for name in ("train", "dev", "test")}
    result = inspect_splits(paths, tmp_path / "review")
    assert not result["ready_for_training"]
    assert any(i["severity"] == "block" for i in result["issues"])


def test_frozen_dev_sampling_and_id_aligned_scoring(splits, tmp_path):
    sample = select_dev_sample(splits["dev"], tmp_path / "sample.jsonl", sample_size=2, seed=3)
    with pytest.raises(FileExistsError):
        select_dev_sample(splits["dev"], sample, sample_size=2)
    rows = [json.loads(s) for s in sample.read_text().splitlines()]
    first = _write(tmp_path / "nllb.jsonl", [{**r, "prediction": r["target"]} for r in rows])
    second = _write(tmp_path / "alt.jsonl", [{**r, "prediction": r["source"]} for r in rows])
    result = compare_dev(sample, {"NLLB": first, "Other": second}, tmp_path / "review")
    assert result["n"] == 2
    assert len(result["models"]) == 2
    assert result["models"][0]["chrfpp"] >= result["models"][1]["chrfpp"]
    assert (tmp_path / "review" / "model_comparison.csv").is_file()
    wrong = [{**r, "prediction": r["target"]} for r in rows]
    wrong[0]["target"] = "WRONG"
    bad = _write(tmp_path / "wrong.jsonl", wrong)
    with pytest.raises(ValueError, match="target differs"):
        compare_dev(sample, {"bad": bad}, tmp_path / "review")


def test_unsupported_languages_fail_before_model_download(splits, tmp_path):
    supported = {row["name"]: row for row in language_options()}
    assert supported["Awadhi"]["nllb"] == "native"
    assert supported["Nepali"]["indictrans2"] == "native"
    assert supported["Braj"]["nllb"] != "native"
    assert supported["Marwari"]["indictrans2"] == "unsupported"
    with pytest.raises(ValueError, match="no native NLLB"):
        require_native_nllb("bra_Deva", "hin_Deva")
    with pytest.raises(ValueError, match="does not support"):
        run_indictrans2_dev(splits["dev"], tmp_path / "it2.jsonl", "bho_Deva", "hin_Deva")
    with pytest.raises(ValueError, match="does not verify"):
        run_bhashaverse_dev(splits["dev"], tmp_path / "bhasha.jsonl",
                            "bho_Deva", "hin_Deva", tmp_path / "hf_inference.py")


def test_plot_review_if_matplotlib_available(splits, tmp_path):
    pytest.importorskip("matplotlib")
    folder = tmp_path / "review"
    inspect_splits(splits, folder)
    figures = plot_review(folder)
    assert len(figures) == 5
    assert all(fig.stat().st_size > 0 for fig in figures)


def test_output_write_refuses_overwrite(splits, tmp_path):
    rows = [json.loads(s) for s in splits["dev"].read_text().splitlines()]
    output = _write_predictions(rows, ["x", "y"], tmp_path / "pred.jsonl")
    assert output.exists()
    with pytest.raises(FileExistsError):
        _write_predictions(rows, ["x", "y"], output)
