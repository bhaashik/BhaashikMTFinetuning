import argparse
import json
from .data import audit
from .evaluation import evaluate, export_errors


def main():
    parser = argparse.ArgumentParser(prog="bhaashik", description="Hindi-belt neural MT research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("audit", help="Reject duplicate and leaking splits")
    for split in ("train", "dev", "test"):
        p.add_argument("--" + split, required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p = commands.add_parser("extend", help="Initialize a new language tag; requires subsequent full training")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--donor", default="hin_Deva")
    p.add_argument("--revision")
    p = commands.add_parser("translate")
    for key in ("model", "src", "tgt", "input", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--revision")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--beams", type=int, default=4)
    p = commands.add_parser("evaluate")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("errors")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("preflight", help="Read-only OneNMT dictionary/tokenizer/checkpoint compatibility checks")
    p.add_argument("--assets", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--src")
    p.add_argument("--tgt")
    p.add_argument("--candidate-assets")
    p.add_argument("--selected-tokens")
    p.add_argument("--plan-output")
    p.add_argument("--max-source-tokens", type=int)
    p.add_argument("--max-unknown-rate", type=float, default=0.0)
    p.add_argument("--strict", action="store_true", help="Exit 2 for pending review as well as blocking findings")
    p = commands.add_parser("hf-regression", help="HF-only original/adapted OneNMT output regression probes")
    p.add_argument("--baseline-assets", required=True)
    p.add_argument("--candidate-assets", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--src")
    p.add_argument("--tgt")
    p.add_argument("--reference", help="Independent reference JSONL; identical probes and generation settings required")
    p.add_argument("--max-retention-drop", type=float, default=2.0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--strict", action="store_true")
    for name, help_text in (
        ("generation", "Fix dictionary-backed generation IDs; preserve architecture and weights"),
        ("dictionary", "Restore IDs using a trusted dictionary/checkpoint pair"),
        ("append", "Append dictionary entries without changing existing IDs"),
        ("spm-extend", "Append SPM USER_DEFINED pieces without retraining old pieces"),
        ("spm-train", "Train a separate replacement SentencePiece model"),
        ("spm-replace", "Install a supplied replacement SPM; keep dictionary IDs"),
        ("resize", "Initialize added vocabulary rows in standard HF MBart"),
    ):
        p = commands.add_parser("repair-" + name, help=help_text)
        p.add_argument("--assets", required=True)
        p.add_argument("--output", required=True, help="New candidate directory; existing destinations are refused")
        if name == "dictionary":
            p.add_argument("--dictionary", required=True, help="Proposal to compare with trusted --assets; its weights are not imported")
        if name == "append":
            choice = p.add_mutually_exclusive_group(required=True)
            choice.add_argument("--tokens")
            choice.add_argument("--from-spm", action="store_true")
            p.add_argument("--max-additions", type=int, default=10000)
        if name == "spm-extend":
            p.add_argument("--tokens", required=True)
        if name == "spm-train":
            p.add_argument("--text", required=True, help="UTF-8 text, one sentence per line; include retained-language data")
            p.add_argument("--vocab-size", type=int, required=True)
            p.add_argument("--model-type", choices=["unigram", "bpe", "char"], required=True)
        if name == "spm-replace":
            p.add_argument("--spm-model", required=True)
        if name == "resize":
            p.add_argument("--reference-assets", required=True, help="Immutable original dictionary/SPM for old IDs and initialization donors")
            p.add_argument("--initialization", choices=["subword-mean", "mean", "random"], default="subword-mean")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--max-new-rows", type=int, default=10000)
    a = parser.parse_args()
    if a.command.startswith("repair-"):
        import sys
        try:
            from .repairs.generation import repair_generation
            from .repairs.dictionary import repair_append, repair_dictionary
            from .repairs.sentencepiece import repair_spm_extend, repair_spm_replace, repair_spm_train
            if a.command == "repair-generation":
                result = repair_generation(a.assets, a.output)
            elif a.command == "repair-dictionary":
                result = repair_dictionary(a.assets, a.dictionary, a.output)
            elif a.command == "repair-append":
                result = repair_append(a.assets, a.output, a.tokens, a.from_spm, a.max_additions)
            elif a.command == "repair-spm-extend":
                result = repair_spm_extend(a.assets, a.tokens, a.output)
            elif a.command == "repair-spm-train":
                result = repair_spm_train(a.assets, a.text, a.output, a.vocab_size, a.model_type)
            elif a.command == "repair-spm-replace":
                result = repair_spm_replace(a.assets, a.spm_model, a.output)
            else:
                from .repairs.embeddings import repair_resize
                result = repair_resize(a.assets, a.reference_assets, a.output, a.initialization, a.seed, a.max_new_rows)
        except (ValueError, OSError, RuntimeError) as error:
            print(json.dumps({"status": "failed", "operation": a.command, "error": str(error)}), file=sys.stderr)
            raise SystemExit(2)
        except ImportError as error:
            print(f"Missing repair dependency: {error}. Install .[repairs]; resize also requires .[train].", file=sys.stderr)
            raise SystemExit(2)
        print(json.dumps({"status": result["status"], "operation": a.command, "output": a.output,
                          "manifest": str(__import__('pathlib').Path(a.output) / 'repair_manifest.json')}, indent=2))
        return
    elif a.command in ("preflight", "hf-regression"):
        from .preflight import preflight, protect_outputs
        from .data import write_json
        import sys
        roots = [a.assets, a.candidate_assets] if a.command == "preflight" else [a.baseline_assets, a.candidate_assets]
        inputs = [a.corpus, a.selected_tokens] if a.command == "preflight" else [a.corpus, a.reference]
        outputs = [a.report, a.plan_output] if a.command == "preflight" else [a.report]
        # Validate destinations separately so an error report cannot overwrite source assets.
        try:
            protect_outputs(roots, inputs, outputs)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2)
        try:
            if a.command == "preflight":
                result = preflight(a.assets, a.corpus, a.report, a.src, a.tgt, a.candidate_assets,
                                   a.selected_tokens, a.plan_output, a.max_source_tokens, a.max_unknown_rate)
            else:
                from .hf_regression import hf_regression
                result = hf_regression(a.baseline_assets, a.candidate_assets, a.corpus, a.report,
                                       a.src, a.tgt, a.reference, a.max_retention_drop,
                                       a.device, a.precision, a.max_new_tokens, a.batch_size)
        except (ValueError, OSError, RuntimeError) as error:
            result = {"schema_version": 1, "kind": a.command, "status": "blocked", "blocking_findings": [str(error)]}
            write_json(a.report, result)
        except ImportError as error:
            print(f"Missing optional dependency: {error}. Install .[preflight] or .[train] for HF generation.", file=sys.stderr)
            raise SystemExit(2)
        status = result.get("structural_status", result.get("status"))
        print(json.dumps({"status": status, "report": a.report,
                          "blocking_findings": result.get("blocking_findings", []),
                          "review_findings": result.get("review_findings", [])}, indent=2))
        raise SystemExit(2 if status == "blocked" or (a.strict and status == "review_required") else 0)
    elif a.command == "audit":
        print(json.dumps(audit({s: getattr(a, s) for s in ("train", "dev", "test")}), indent=2))
    elif a.command == "evaluate":
        print(json.dumps(evaluate(a.input, a.output, a.baseline, a.samples, a.seed), indent=2))
    elif a.command == "errors":
        export_errors(a.input, a.output, a.seed)
    else:
        from .modeling import train, extend, translate
        if a.command == "train":
            print(train(a.config))
        elif a.command == "extend":
            extend(a.model, a.output, a.language, a.donor, a.revision)
        else:
            translate(a.model, a.src, a.tgt, a.input, a.output, a.revision,
                      a.batch_size, a.max_length, a.max_new_tokens, a.beams)


if __name__ == "__main__":
    main()
