"""Offline tests of explicit row tags and model-specific conditioning routes."""
import json
from pathlib import Path

import pytest

from bhaashik.data import read_jsonl
from bhaashik.multilingual import (
    require_routes, resolve_languages, inspect_routes, write_tagged_copy,
    condition_sentence, language_prefix,
)
from bhaashik.indictrans2_adaptation import processing_pair, prepare_indic_rows
from bhaashik.bhashaverse_adaptation import validate_prefix_routes
from bhaashik.preflight import AssetError


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_migration_preserves_text_and_tagging(tmp_path):
    original = _write(tmp_path / "legacy.jsonl", [
        {"id": "1", "source": "ब्रज भाषा", "target": "ब्रज भाषा", "domain": "oral"},
        {"id": "2", "source": "शब्द", "target": "शब्द", "document_id": "doc2"},
    ])
    destination = tmp_path / "tagged.jsonl"
    write_tagged_copy(original, destination, "bra_Deva", "hin_Deva")
    assert original.read_text() != destination.read_text()
    old, new = read_jsonl(original), read_jsonl(destination)
    assert [r["source"] for r in old] == [r["source"] for r in new]
    assert [r["target"] for r in old] == [r["target"] for r in new]
    assert all(r["source_lang"] == "bra_Deva" and r["target_lang"] == "hin_Deva" for r in new)
    assert require_routes(destination)[1] == {("bra_Deva", "hin_Deva"): 2}
    assert inspect_routes({"train": destination}, tmp_path / "routes.json")["splits"]["train"]["n"] == 2
    with pytest.raises(FileExistsError):
        write_tagged_copy(original, destination, "bra_Deva", "hin_Deva")


def test_missing_or_invalid_tag_pair_rejected(tmp_path):
    path = _write(tmp_path / "bad.jsonl", [
        {"id": "x", "source": "हेलो", "target": "hello", "source_lang": "bra_Deva"}])
    with pytest.raises(ValueError, match="target_lang"):
        read_jsonl(path)
    path = _write(tmp_path / "bad2.jsonl", [
        {"id": "x", "source": "हेलो", "target": "hello", "source_lang": "bra",
         "target_lang": "eng_Latn"}])
    with pytest.raises(ValueError, match="source_lang"):
        read_jsonl(path)
    assert resolve_languages({"source": "A", "target": "B"}, "eng_Latn", "hin_Deva") == (
        "eng_Latn", "hin_Deva")


def test_mixed_route_not_silently_converted(tmp_path):
    rows = [{"id": "1", "source": "x", "target": "a", "source_lang": "bra_Deva", "target_lang": "hin_Deva"},
            {"id": "2", "source": "y", "target": "b", "source_lang": "mwr_Deva", "target_lang": "hin_Deva"}]
    path = _write(tmp_path / "mixed.jsonl", rows)
    assert len(require_routes(path)[1]) == 2
    with pytest.raises(ValueError, match="one language direction"):
        require_routes(path, homogeneous=True)
    with pytest.raises(ValueError, match="conflict"):
        write_tagged_copy(path, tmp_path / "wrong.jsonl", "bra_Deva", "hin_Deva")


def test_model_conditioning_is_not_embedded_in_reference():
    text = "some text"
    assert condition_sentence(text, "bra_Deva", "hin_Deva", "nllb") == text
    assert condition_sentence(text, "bra_Deva", "hin_Deva", "nllb", "text_prefix").startswith(
        "<bra_Deva> <hin_Deva>")
    assert language_prefix("bra_Deva", "hin_Deva", "bhashaverse") == (
        "###bra_Deva-to-hin_Deva### ")
    assert language_prefix("bra_Deva", "hin_Deva", "indictrans2") == "bra_Deva hin_Deva "


class _Processor:
    def preprocess_batch(self, sentences, src_lang, tgt_lang):
        return [f"{src_lang} {tgt_lang} {text}" for text in sentences]


def test_it2_requires_native_or_explicit_proxies():
    with pytest.raises(ValueError, match="does not natively support"):
        processing_pair("bra_Deva", "hin_Deva", strategy="native")
    with pytest.raises(ValueError, match="Explicit proxy"):
        processing_pair("bra_Deva", "hin_Deva", strategy="text_prefix")
    assert processing_pair("bra_Deva", "hin_Deva", strategy="text_prefix",
                           proxy_src="hin_Deva") == ("hin_Deva", "hin_Deva")
    rows = [{"id": "1", "source": "ब्रज", "target": "हिन्दी",
             "source_lang": "bra_Deva", "target_lang": "hin_Deva"}]
    sources, targets, routes = prepare_indic_rows(
        rows, _Processor(), strategy="dedicated_token", proxy_src="hin_Deva")
    assert sources[0].startswith("<bhaashik_src_bra_Deva> <bhaashik_tgt_hin_Deva>")
    assert targets == ["हिन्दी"]
    assert routes[0]["processor_source"] == "hin_Deva"


class _Codec:
    special = {"unk": 3}
    def tagged(self, text, src, tgt):
        return f"###{src}-to-{tgt}### {text}"
    def encode(self, text, side):
        if "bho_Deva" in text:
            return [10, 11, 2]
        if "bra_Deva" in text:
            return [12, 13, 2]
        return [10, 11, 2]
    def pieces(self, text):
        return [text]


def test_bhashaverse_prefix_collision_detection():
    rows = [{"id": "1", "source": "a", "target": "b",
             "source_lang": "bra_Deva", "target_lang": "hin_Deva"}]
    assert "bra_Deva->hin_Deva" in validate_prefix_routes(_Codec(), rows)
    rows.append({"id": "2", "source": "a", "target": "b",
                 "source_lang": "mai_Deva", "target_lang": "hin_Deva"})
    assert len(validate_prefix_routes(_Codec(), rows)) == 2
    rows.append({"id": "3", "source": "a", "target": "b",
                 "source_lang": "hin_Deva", "target_lang": "eng_Latn"})
    with pytest.raises(AssetError, match="collision"):
        validate_prefix_routes(_Codec(), rows)
