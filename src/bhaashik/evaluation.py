"""Corpus metrics, aligned paired bootstrap, and human review sheets."""
import csv
import re
import random
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from .data import read_jsonl, file_hash, write_json


def metrics(hypotheses, references):
    bleu, chrf = BLEU(tokenize="13a"), CHRF(word_order=2)
    return {"bleu": bleu.corpus_score(hypotheses, [references]).score,
            "chrfpp": chrf.corpus_score(hypotheses, [references]).score,
            "signatures": {"bleu": str(bleu.get_signature()), "chrfpp": str(chrf.get_signature())}}


def evaluate(path, output, baseline=None, samples=1000, seed=42):
    rows = read_jsonl(path, predictions=True)
    hypotheses, references = [r["prediction"] for r in rows], [r["target"] for r in rows]
    result = {"schema_version": 1, "n": len(rows), "input_sha256": file_hash(path),
              "scores": metrics(hypotheses, references), "domains": {}}
    for domain in sorted({r.get("domain", "unspecified") for r in rows}):
        subset = [r for r in rows if r.get("domain", "unspecified") == domain]
        result["domains"][domain] = {"n": len(subset), "scores": metrics(
            [r["prediction"] for r in subset], [r["target"] for r in subset])}
    if baseline:
        if samples < 1:
            raise ValueError("bootstrap samples must be positive")
        base = {r["id"]: r for r in read_jsonl(baseline, predictions=True)}
        if set(base) != {r["id"] for r in rows}:
            raise ValueError("baseline IDs must match the candidate exactly")
        for row in rows:
            if any(row.get(k) != base[row["id"]].get(k) for k in ("source", "target", "document_id", "domain")):
                raise ValueError(f"baseline content/metadata mismatch: {row['id']}")
        base_hyp = [base[r["id"]]["prediction"] for r in rows]
        base_scores = metrics(base_hyp, references)
        has_docs = [bool(r.get("document_id")) for r in rows]
        if any(has_docs) and not all(has_docs):
            raise ValueError("paired bootstrap requires document_id on all rows or none")
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(row.get("document_id", row["id"]), []).append(i)
        units = list(groups.values())
        if len(units) < 2:
            raise ValueError("paired bootstrap requires at least two independent units")
        rng, deltas = np.random.default_rng(seed), {"bleu": [], "chrfpp": []}
        for _ in range(samples):
            indices = [i for unit in rng.integers(0, len(units), len(units)) for i in units[unit]]
            refs = [references[i] for i in indices]
            a = metrics([hypotheses[i] for i in indices], refs)
            b = metrics([base_hyp[i] for i in indices], refs)
            for key in deltas:
                deltas[key].append(a[key] - b[key])
        result["paired_bootstrap"] = {
            "baseline_sha256": file_hash(baseline), "baseline_scores": base_scores,
            "samples": samples, "seed": seed, "unit": "document" if all(has_docs) else "sentence",
            "units": len(units), "delta_candidate_minus_baseline": {
                k: {"point": result["scores"][k] - base_scores[k],
                    "ci95": np.percentile(v, [2.5, 97.5]).tolist()} for k, v in deltas.items()}}
    write_json(output, result)
    return result


def triage(row):
    """Heuristics flag review candidates, never assert linguistic errors."""
    flags = []
    prediction, source = row["prediction"], row["source"]
    if not prediction.strip():
        flags.append("empty")
    ratio = len(prediction) / max(len(row["target"]), 1)
    if ratio < .5 or ratio > 2:
        flags.append("length_outlier")
    if prediction.strip() == source.strip():
        flags.append("source_copy")
    def numerals(text):
        import unicodedata
        return re.findall(r"\d+", "".join(str(unicodedata.digit(c)) if c.isdecimal() else c for c in text))
    if sorted(numerals(source)) != sorted(numerals(prediction)):
        flags.append("numeral_mismatch")
    return ";".join(flags)


def export_errors(path, output, seed=42):
    from pathlib import Path
    rows = read_jsonl(path, predictions=True)
    random.Random(seed).shuffle(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fields = ["blind_id", "id", "source", "target", "prediction", "review_flags",
              "annotator", "adequacy_1_5", "fluency_1_5", "error_category", "severity", "span", "notes"]
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"blind_id": f"item-{i+1:05d}", **{k: row[k] for k in ("id", "source", "target", "prediction")},
                             "review_flags": triage(row)})
