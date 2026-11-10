"""Strict parallel-data loading and conservative split leakage checks."""
import hashlib
import json
import unicodedata
from pathlib import Path


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_jsonl(path, predictions=False):
    rows, ids = [], set()
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: each row must be a JSON object")
            for field in ["id", "source", "target"] + (["prediction"] if predictions else []):
                if field == "prediction" and row.get(field) == "":
                    continue  # Empty outputs are model failures that must be scored.
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_no}: nonempty string required: {field}")
            if row["id"] in ids:
                raise ValueError(f"{path}:{line_no}: duplicate id {row['id']}")
            for field in ("document_id", "domain"):
                if field in row and (not isinstance(row[field], str) or not row[field].strip()):
                    raise ValueError(f"{path}:{line_no}: invalid {field}")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(paths):
    seen = {key: {} for key in ("id", "source", "pair", "document")}
    counts = {}
    for split, path in paths.items():
        rows = read_jsonl(path)
        counts[split] = len(rows)
        for row in rows:
            keys = {"id": row["id"], "source": normalized(row["source"]),
                    "pair": (normalized(row["source"]), normalized(row["target"]))}
            if row.get("document_id"):
                keys["document"] = row["document_id"]
            for kind, value in keys.items():
                previous = seen[kind].get(value)
                if previous is not None and previous != split:
                    raise ValueError(f"{kind} leakage between {previous} and {split}: {row['id']}")
                seen[kind][value] = split
        pairs = {(normalized(r["source"]), normalized(r["target"])) for r in rows}
        if len(pairs) != len(rows):
            raise ValueError(f"duplicate parallel pairs inside {split}")
    return {"counts": counts, "sha256": {k: file_hash(v) for k, v in paths.items()}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
