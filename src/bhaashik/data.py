"""Strict parallel-data loading and conservative split leakage checks."""
import hashlib
import json
import re
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
            if "source_lang" in row or "target_lang" in row:
                for field in ("source_lang", "target_lang"):
                    tag = row.get(field)
                    if not isinstance(tag, str) or not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", tag):
                        raise ValueError(f"{path}:{line_no}: invalid or missing {field} language_Script tag")
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


def audit(paths, *, reject_within_split_duplicates=False,
          enforce_document_disjoint=True):
    """Always block cross-split leakage; optionally reject corpus repetitions.

    Repeated parallel pairs already present *within one split* are allowed by
    default. The same normalized source, exact pair, or row ID may never
    cross train/dev/test. Document IDs are also disjoint by default, unless
    group_by_document=False has been explicitly selected.
    """
    if not isinstance(reject_within_split_duplicates, bool) or not isinstance(enforce_document_disjoint, bool):
        raise TypeError("Duplicate and document policy flags must be booleans")
    seen = {key: {} for key in ("id", "source", "pair", "document")}
    counts, repeated_pairs = {}, {}
    for split, path in paths.items():
        rows = read_jsonl(path)
        counts[split] = len(rows)
        local_pairs = set()
        repeats = 0
        for row in rows:
            pair = (normalized(row["source"]), normalized(row["target"]))
            repeats += pair in local_pairs
            local_pairs.add(pair)
            keys = {"id": row["id"], "source": pair[0], "pair": pair}
            if enforce_document_disjoint and row.get("document_id"):
                keys["document"] = row["document_id"]
            for kind, value in keys.items():
                previous = seen[kind].get(value)
                if previous is not None and previous != split:
                    raise ValueError(
                        f"{kind} leakage between {previous} and {split}: {row['id']}")
                seen[kind][value] = split
        repeated_pairs[split] = repeats
        if reject_within_split_duplicates and repeats:
            raise ValueError(f"duplicate parallel pairs inside {split}: {repeats} repeated rows")
    return {
        "counts": counts,
        "within_split_duplicate_pairs": repeated_pairs,
        "sha256": {key: file_hash(path) for key, path in paths.items()},
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
