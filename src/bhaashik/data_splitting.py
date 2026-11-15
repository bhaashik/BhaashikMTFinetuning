"""Seeded, non-destructive train/dev/test splitting of a single parallel JSONL.

Rows sharing a normalized source sentence stay together; by default, all rows
of one document stay together as well. This prevents common exact leakage.
Optional deep validation can be turned off for early smoke tests.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from .data import audit, file_hash, normalized, read_jsonl, write_json
from .multilingual import resolve_languages


SPLITS = ("train", "dev", "test")


def _ratios(values):
    if len(values) != 3:
        raise ValueError("Exactly three ratios (train, dev, test) are required")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ratios must be numeric") from exc
    if not all(math.isfinite(v) and 0 < v < 1 for v in result):
        raise ValueError("Each split ratio must be finite and strictly between zero and one")
    if not math.isclose(sum(result), 1.0, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Split ratios must sum to one")
    return result


def _components(rows, group_by_document):
    """Union rows sharing a document ID or normalized source text."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen_source, seen_document = {}, {}
    for i, row in enumerate(rows):
        key = normalized(row["source"]).casefold()
        if key in seen_source:
            union(i, seen_source[key])
        else:
            seen_source[key] = i
        if group_by_document and row.get("document_id"):
            key = row["document_id"]
            if key in seen_document:
                union(i, seen_document[key])
            else:
                seen_document[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def _assign(groups, ratios, seed):
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # Stable: seeded ties affect allocation.
    n = sum(map(len, groups))
    targets = [ratio * n for ratio in ratios]
    assigned = [[] for _ in SPLITS]
    counts = [0] * len(SPLITS)
    for i, group in enumerate(groups):
        remaining = len(groups) - i
        empty = [j for j, value in enumerate(assigned) if not value]
        candidates = empty if remaining <= len(empty) else range(len(SPLITS))
        # Prioritize the largest unmet allocation; never leave a split empty.
        choice = max(candidates, key=lambda j: (targets[j] - counts[j], -j))
        assigned[choice].extend(group)
        counts[choice] += len(group)
    return assigned


def split_parallel_jsonl(input_path, output_dir, *, seed=42,
                         ratios=(0.8, 0.1, 0.1),
                         source_lang=None, target_lang=None,
                         group_by_document=True, validate=False):
    """Produce train.jsonl/dev.jsonl/test.jsonl and a SHA-256 split manifest.

    Input is never overwritten. Re-running with the same input/seed/ratios
    reuses a matching, hash-verified split; changed inputs or options fail.
    'validate' performs the more expensive exact corpus audit; basic schema,
    output integrity and grouping checks always apply.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(group_by_document, bool) or not isinstance(validate, bool):
        raise ValueError("group_by_document and validate must be booleans")
    fractions = _ratios(ratios)
    if (source_lang is None) != (target_lang is None):
        raise ValueError("Supply both language defaults or neither")
    if source_lang is not None:
        resolve_languages({}, source_lang, target_lang)
    input_path = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if input_path.is_relative_to(destination) or destination == input_path.parent:
        raise ValueError("Output must not contain/replace the input corpus")
    rows = read_jsonl(input_path)  # Basic schema is always checked.
    for row in rows:
        if source_lang is not None:
            if "source_lang" in row and (
                    row["source_lang"] != source_lang or
                    row["target_lang"] != target_lang):
                raise ValueError(f"Language defaults conflict with existing tags: {row['id']}")
            row.setdefault("source_lang", source_lang)
            row.setdefault("target_lang", target_lang)
    groups = _components(rows, group_by_document)
    if len(groups) < 3:
        raise ValueError(
            "Need at least three independent document/source groups to make "
            "nonempty train, dev, and test splits without exact leakage")
    expected = {
        "schema_version": 1,
        "input_sha256": file_hash(input_path),
        "seed": seed,
        "ratios": list(fractions),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "group_by_document": group_by_document,
    }
    paths = {part: destination / f"{part}.jsonl" for part in SPLITS}
    manifest_path = destination / "split_manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Output exists without a complete manifest: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise FileExistsError("Existing split was made with different data/options")
        if any(not path.is_file() or file_hash(path) !=
               manifest.get("output_sha256", {}).get(part)
               for part, path in paths.items()):
            raise ValueError("Saved split differs from its manifest; refusing reuse")
        if validate:
            audit(paths)
        return manifest

    allocated = _assign(groups, fractions, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        split_paths = {}
        distribution = {}
        output_hashes = {}
        counts = {}
        for part, indices in zip(SPLITS, allocated):
            path = staging / f"{part}.jsonl"
            counts[part] = len(indices)
            directions = Counter()
            with path.open("w", encoding="utf-8") as handle:
                for i in indices:
                    row = rows[i]
                    pair = (row.get("source_lang", "unspecified"),
                            row.get("target_lang", "unspecified"))
                    directions[f"{pair[0]}->{pair[1]}"] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_paths[part] = path
            distribution[part] = dict(sorted(directions.items()))
            output_hashes[part] = file_hash(path)
        if validate:
            audit(split_paths)
        manifest = {
            **expected,
            "input_path": str(input_path),
            "counts": counts,
            "actual_ratios": {name: counts[name] / len(rows) for name in SPLITS},
            "group_count": len(groups),
            "largest_group": max(map(len, groups)),
            "routes_by_split": distribution,
            "output_sha256": output_hashes,
            "validation_performed_at_creation": validate,
            "note": ("Groups sharing normalized source text or document_id are "
                     "never split. Exact split ratios may vary due to group sizes. "
                     "Heuristic/semantic leakage and linguistic quality need later review."),
        }
        write_json(staging / "split_manifest.json", manifest)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
