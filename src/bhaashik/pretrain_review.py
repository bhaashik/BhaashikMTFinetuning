"""Pre-training data diagnostics and safe, aligned development-set model comparison."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import statistics
import unicodedata

from .data import audit, file_hash, normalized, read_jsonl, write_json
from .evaluation import metrics, triage


def _script_counts(text):
    counts = Counter()
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            counts["Devanagari"] += 1
        elif 0x0041 <= cp <= 0x007A and char.isalpha():
            counts["Latin"] += 1
        elif char.isalpha():
            counts["other_letters"] += 1
    return dict(counts)


def _stats(values):
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {"min": ordered[0],
            "median": statistics.median(ordered),
            "mean": round(statistics.mean(ordered), 3),
            "p95": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * .95))],
            "max": ordered[-1]}


def _shingles(text, n=3):
    words = normalized(text).casefold().split()
    if len(words) < n:
        return set(words)
    return set(zip(*(words[i:] for i in range(n))))


def _near_duplicate_pairs(rows, threshold=.85, limit=15, max_candidates=150000):
    """Bounded, token-trigram candidate screen; not semantic duplicate detection."""
    buckets = defaultdict(list)
    features = []
    output = []
    candidate_checks = 0
    for row in rows:
        words = normalized(row["source"]).casefold().split()
        shingles = _shingles(row["source"])
        signature = (words[0] if words else "", min(12, len(words) // 5))
        indexes = buckets[signature]
        for previous in indexes[-400:]:
            candidate_checks += 1
            if candidate_checks > max_candidates:
                break
            earlier = features[previous]
            if row["id"] == earlier["id"]:
                continue
            a, b = shingles, earlier["shingles"]
            similarity = len(a & b) / len(a | b) if a or b else 0
            if similarity >= threshold:
                output.append({"id_a": earlier["id"], "id_b": row["id"],
                               "similarity": round(similarity, 3)})
                if len(output) >= limit:
                    break
        if len(output) >= limit or candidate_checks > max_candidates:
            break
        features.append({"id": row["id"], "shingles": shingles})
        indexes.append(len(features) - 1)
    return output, candidate_checks > max_candidates


def inspect_splits(paths, output_dir, near_duplicate_limit=15):
    """Audit strict train/dev/test splits and write diagnostics, including warnings."""
    paths = {key: Path(value) for key, value in paths.items()}
    if set(paths) != {"train", "dev", "test"}:
        raise ValueError("Must supply train, dev and test paths")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {part: read_jsonl(path) for part, path in paths.items()}
    issues = []
    try:
        core = audit(paths)
    except ValueError as error:
        core = {"counts": {key: len(value) for key, value in rows.items()},
                "sha256": {key: file_hash(path) for key, path in paths.items()}}
        issues.append({"severity": "block", "check": "split_audit", "message": str(error)})
    splits = {}
    length_rows = []
    for name, subset in rows.items():
        src_chars = [len(row["source"]) for row in subset]
        tgt_chars = [len(row["target"]) for row in subset]
        src_tokens = [len(row["source"].split()) for row in subset]
        tgt_tokens = [len(row["target"].split()) for row in subset]
        ratios = [t / max(s, 1) for s, t in zip(src_tokens, tgt_tokens)]
        scripts = {"source": Counter(), "target": Counter()}
        identical = 0
        unusual = []
        for row in subset:
            scripts["source"].update(_script_counts(row["source"]))
            scripts["target"].update(_script_counts(row["target"]))
            if normalized(row["source"]) == normalized(row["target"]):
                identical += 1
            if any(unicodedata.category(c) in {"Cc", "Cf"} for field in ("source", "target")
                   for c in row[field]):
                unusual.append(row["id"])
            length_rows.append({"split": name, "id": row["id"],
                                "source_tokens": len(row["source"].split()),
                                "target_tokens": len(row["target"].split()),
                                "token_ratio": round(len(row["target"].split()) /
                                                     max(len(row["source"].split()), 1), 3)})
        samples = [row for row in subset if row["source"].strip() and row["target"].strip()]
        near_dupes, truncated = _near_duplicate_pairs(samples, limit=near_duplicate_limit)
        splits[name] = {
            "n": len(subset), "source_chars": _stats(src_chars),
            "target_chars": _stats(tgt_chars),
            "source_words": _stats(src_tokens), "target_words": _stats(tgt_tokens),
            "target_to_source_word_ratio": _stats(ratios),
            "identical_source_target": identical,
            "script_letters": {k: dict(v) for k, v in scripts.items()},
            "domain_counts": dict(Counter(row.get("domain", "unspecified") for row in subset)),
            "document_count": len({row.get("document_id") for row in subset if row.get("document_id")}),
            "rows_with_document_id": sum(bool(row.get("document_id")) for row in subset),
            "rows_with_provenance": sum(bool(row.get("provenance")) for row in subset),
            "rows_with_license": sum(bool(row.get("license")) for row in subset),
            "control_character_examples": unusual[:10],
            "near_duplicate_candidates": near_dupes,
            "near_duplicate_scan_truncated": truncated,
        }
        if identical:
            issues.append({"severity": "review", "check": "identical_pair",
                           "message": f"{name}: {identical} source=target pairs; may be valid (names/numbers)"})
        if unusual:
            issues.append({"severity": "review", "check": "control_characters",
                           "message": f"{name}: {len(unusual)} rows contain control/format characters"})
        if len(subset) < 100:
            issues.append({"severity": "review", "check": "small_split",
                           "message": f"{name} contains only {len(subset)} sentences; scores will be unstable"})
    report = {"schema_version": 1, "audit": core, "splits": splits, "issues": issues,
              "ready_for_training": not any(item["severity"] == "block" for item in issues),
              "note": "Ready means no detected structural blockers; it does not certify linguistic quality, licensing, or semantic split independence."}
    write_json(output_dir / "corpus_diagnostics.json", report)
    with (output_dir / "sentence_lengths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["split", "id", "source_tokens", "target_tokens", "token_ratio"])
        writer.writeheader()
        writer.writerows(length_rows)
    return report


def compare_dev(reference_path, predictions_by_model, output_dir):
    """Compare models on the EXACT same dev IDs/references, never on test data."""
    gold = read_jsonl(reference_path)
    expected = {row["id"]: row for row in gold}
    if not predictions_by_model:
        raise ValueError("Supply at least one dev prediction file")
    rows = []
    flags = []
    for model_name, prediction_path in predictions_by_model.items():
        candidates = read_jsonl(prediction_path, predictions=True)
        actual = {r["id"]: r for r in candidates}
        if actual.keys() != expected.keys():
            raise ValueError(f"{model_name}: dev IDs differ; cannot compare unequal sets")
        ordered = []
        for row in gold:
            candidate = actual[row["id"]]
            for field in ("source", "target", "document_id", "domain"):
                if candidate.get(field) != row.get(field):
                    raise ValueError(f"{model_name}: {field} differs on {row['id']}")
            ordered.append(candidate)
        result = metrics([r["prediction"] for r in ordered], [r["target"] for r in ordered])
        flag_counts = Counter()
        for row in ordered:
            for flag in filter(None, triage(row).split(";")):
                flag_counts[flag] += 1
        rows.append({"model": model_name, "n": len(ordered),
                     "bleu": round(result["bleu"], 4),
                     "chrfpp": round(result["chrfpp"], 4),
                     "prediction_sha256": file_hash(prediction_path),
                     "metric_signatures": result["signatures"],
                     "review_flags": dict(flag_counts)})
        flags.extend({"model": model_name, "flag": flag, "n": count}
                     for flag, count in flag_counts.items())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "split": "dev", "reference_sha256": file_hash(reference_path),
              "n": len(gold), "models": rows,
              "note": "Scores are descriptive; use paired document bootstrap, seed repeats and blind human assessment for claims."}
    write_json(output_dir / "model_comparison.json", report)
    with (output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["model", "n", "bleu", "chrfpp", "prediction_sha256"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    return report


def plot_review(report_dir):
    """Write compact PNG figures; matplotlib is only imported when requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report_dir = Path(report_dir)
    outputs = []
    corpus_path = report_dir / "corpus_diagnostics.json"
    if corpus_path.exists():
        report = json.loads(corpus_path.read_text(encoding="utf-8"))
        names = list(report["splits"])
        counts = [report["splits"][name]["n"] for name in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, counts)
        ax.set(ylabel="Sentence pairs", title="Train / development / test sizes")
        fig.tight_layout()
        path = report_dir / "split_sizes.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([[report["splits"][name]["source_words"]["median"],
                     report["splits"][name]["target_words"]["median"]] for name in names],
                   tick_labels=names)
        ax.set(ylabel="Median sentence length (words)", title="Source / target median lengths per split")
        fig.tight_layout()
        path = report_dir / "length_summary.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    comparison_path = report_dir / "model_comparison.json"
    if comparison_path.exists():
        report = json.loads(comparison_path.read_text(encoding="utf-8"))
        models = report["models"]
        fig, ax = plt.subplots(figsize=(max(6, len(models)*2), 4))
        x = list(range(len(models)))
        ax.bar([i-.18 for i in x], [r["chrfpp"] for r in models], width=.36, label="chrF++")
        ax.bar([i+.18 for i in x], [r["bleu"] for r in models], width=.36, label="BLEU")
        ax.set_xticks(x, [r["model"] for r in models], rotation=20, ha="right")
        ax.set(ylabel="Corpus score", title="Development-set model comparison")
        ax.legend()
        fig.tight_layout()
        path = report_dir / "dev_model_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs
