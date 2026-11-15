"""Convert two aligned UTF-8 sentence files to a tagged parallel JSONL file.

Each line of source_file must correspond to the same-numbered line in
target_file. Conversion never modifies the original inputs, and does not
guess a missing sentence or silently discard blank lines.
"""
from itertools import zip_longest
import json
from pathlib import Path
import re
import tempfile

from .data import file_hash, write_json
from .multilingual import resolve_languages


_MISSING = object()


def convert_aligned_text_files(source_file, target_file, output_jsonl, *,
                               source_lang, target_lang, id_prefix=None,
                               domain=None, provenance=None, license=None):
    """Convert aligned text pair to tagged JSONL with a reproducible manifest.

    Lines are preserved verbatim except for their line endings and an initial
    UTF-8 BOM, if present. Blank lines and differing line counts are rejected.
    Existing output + matching manifest are verified and reused, never overwritten.
    For initial testing, no expensive linguistic/corpus validation is required.
    """
    src = Path(source_file).resolve()
    tgt = Path(target_file).resolve()
    output = Path(output_jsonl).resolve()

    resolve_languages({}, source_lang, target_lang)
    if src == tgt:
        raise ValueError("Source and target must be two distinct aligned text files")
    if output in (src, tgt):
        raise ValueError("Output JSONL must not overwrite either input text file")
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Output must end in .jsonl")
    for path in (src, tgt):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input text file: {path}")
    if id_prefix is None:
        id_prefix = f"{source_lang.split('_')[0]}-{target_lang.split('_')[0]}"
    if not isinstance(id_prefix, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", id_prefix):
        raise ValueError("id_prefix must contain only letters, digits, '.', '-' or '_'")
    metadata = {"domain": domain, "provenance": provenance, "license": license}
    for name, value in metadata.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string when provided")
    expected = {
        "schema_version": 1,
        "format": "aligned_utf8_text_to_jsonl",
        "source_sha256": file_hash(src),
        "target_sha256": file_hash(tgt),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "id_prefix": id_prefix,
        "metadata": metadata,
    }
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        if not output.is_file() or not manifest_path.is_file():
            raise FileExistsError(
                "Conversion output or manifest is incomplete; use a new output path")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise FileExistsError(
                "Converted file was created from different inputs or settings; "
                "choose a new output filename")
        if file_hash(output) != manifest.get("output_sha256"):
            raise ValueError("Converted JSONL differs from its saved manifest")
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{output.stem}.conversion-", dir=output.parent) as directory:
        staging = Path(directory)
        converted = staging / "parallel.jsonl"
        count = 0
        # utf-8-sig ignores only an optional initial BOM.
        with src.open("r", encoding="utf-8-sig") as src_handle, \
             tgt.open("r", encoding="utf-8-sig") as tgt_handle, \
             converted.open("w", encoding="utf-8") as out_handle:
            for line_number, (src_line, tgt_line) in enumerate(
                    zip_longest(src_handle, tgt_handle, fillvalue=_MISSING), start=1):
                if src_line is _MISSING or tgt_line is _MISSING:
                    raise ValueError(
                        f"Source and target line counts differ at line {line_number}")
                source = src_line.rstrip("\r\n")
                target = tgt_line.rstrip("\r\n")
                if not source.strip() or not target.strip():
                    raise ValueError(
                        f"Blank source or target sentence at aligned line {line_number}; "
                        "fix the text files instead of silently skipping rows")
                row = {
                    "id": f"{id_prefix}-{line_number:08d}",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source": source,
                    "target": target,
                }
                row.update({key: value for key, value in metadata.items()
                            if value is not None})
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("Both parallel text files are empty")
        # Detect input updates made while conversion was reading them.
        if expected["source_sha256"] != file_hash(src) or \
                expected["target_sha256"] != file_hash(tgt):
            raise ValueError("Input text files changed during conversion")
        manifest = {
            **expected,
            "source_file": str(src),
            "target_file": str(tgt),
            "output_file": str(output),
            "sentence_pairs": count,
            "output_sha256": file_hash(converted),
            "note": ("Sentence alignment is positional only: the converter cannot "
                     "verify that corresponding sentences are actual translations. "
                     "No document IDs can be inferred from plain sentence files."),
        }
        write_json(staging / "conversion_manifest.json", manifest)
        converted.rename(output)
        (staging / "conversion_manifest.json").rename(manifest_path)
    return manifest
