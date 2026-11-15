"""Common multilingual data contract and explicit, auditable conditioning.

Raw corpus text is never modified on disk. Model-specific prefixes belong to
the preprocessing pipeline, and unknown tags never silently become Hindi.
"""
from collections import Counter
from pathlib import Path
import json
import re
from .data import read_jsonl, file_hash, write_json

TAG = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def resolve_languages(row, default_source=None, default_target=None):
    """Return a validated route; optional defaults support legacy parallel data."""
    source = row.get("source_lang", default_source)
    target = row.get("target_lang", default_target)
    for role, value in (("source", source), ("target", target)):
        if not isinstance(value, str) or not TAG.fullmatch(value):
            raise ValueError(f"Missing or invalid {role}_lang: {value!r}")
    return source, target


def require_routes(path, source=None, target=None, homogeneous=False):
    rows = read_jsonl(path)
    routes = Counter(resolve_languages(r, source, target) for r in rows)
    if homogeneous and len(routes) != 1:
        raise ValueError(f"Expected one language direction, found {len(routes)}: {dict(routes)}")
    return rows, dict(routes)


def language_prefix(src, tgt, model, strategy="native"):
    """Text-prefix format is an experiment, not a native supported tag."""
    if not TAG.fullmatch(src) or not TAG.fullmatch(tgt):
        raise ValueError("Tags must be three-letter language plus script")
    if model == "indictrans2":
        if strategy not in ("native", "text_prefix"):
            raise ValueError("IndicTrans2 supports native or text_prefix only in this module")
        return f"{src} {tgt} "
    if model == "bhashaverse":
        if strategy not in ("native", "text_prefix"):
            raise ValueError("BhashaVerse supports native or text_prefix only in this module")
        return f"###{src}-to-{tgt}### "
    if model == "nllb":
        if strategy not in ("native", "text_prefix", "dedicated_token"):
            raise ValueError("Unknown NLLB conditioning strategy")
        return f"<{src}> <{tgt}> " if strategy == "text_prefix" else ""
    raise ValueError(f"Unknown model family: {model}")


def condition_sentence(text, src, tgt, model, strategy="native"):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Source text must be nonempty")
    if model == "nllb" and strategy == "native":
        return text
    if model == "nllb" and strategy == "dedicated_token":
        return text  # NLLB tokenizer injects registered language markers.
    return language_prefix(src, tgt, model, strategy) + text


def inspect_routes(paths, output, source=None, target=None):
    """Record language coverage/hash and reject label leakage or bad labels."""
    result = {"schema_version": 1, "splits": {}, "all_routes": {}}
    for split, path in paths.items():
        rows, counts = require_routes(path, source, target)
        result["splits"][split] = {
            "n": len(rows), "sha256": file_hash(path),
            "routes": {f"{src}->{tgt}": n for (src, tgt), n in sorted(counts.items())},
            "explicit_tagged": sum("source_lang" in r and "target_lang" in r for r in rows),
        }
        for route, n in counts.items():
            key = f"{route[0]}->{route[1]}"
            result["all_routes"][key] = result["all_routes"].get(key, 0) + n
    write_json(output, result)
    return result


def write_tagged_copy(source_path, destination, src, tgt, overwrite=False):
    """Migrate a legacy single-direction JSONL to explicit tags, without overwrites."""
    if not TAG.fullmatch(src) or not TAG.fullmatch(tgt):
        raise ValueError("Invalid language tags")
    dest = Path(destination)
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)
    rows = read_jsonl(source_path)
    for r in rows:
        if ("source_lang" in r and r["source_lang"] != src or
                "target_lang" in r and r["target_lang"] != tgt):
            raise ValueError(f"Existing tags conflict for row {r['id']}")
        r["source_lang"], r["target_lang"] = src, tgt
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"source_sha256": file_hash(source_path),
                "destination_sha256": file_hash(dest), "src_lang": src,
                "tgt_lang": tgt, "n": len(rows)}
    write_json(str(dest) + ".manifest.json", manifest)
    return dest
