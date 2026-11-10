# OneNMT vocabulary and HF behavior preflight

This module assesses the risks of adapting BhashaVerse/OneNMT with unchanged vocabulary, append-only vocabulary, or a replacement SentencePiece model. It performs **no architecture conversion, training, embedding resizing, tokenizer retraining or original-asset rewriting**. All generation uses the supplied HF MBart implementation. The existing `bhaashik train` command remains NLLB-only.

The inspected authors' [HF inference code](https://github.com/ltrc/bhashaverse/blob/main/hf_model/hf_inference.py) uses SentencePiece **piece strings**, separate source/target Fairseq dictionaries, a textual direction prefix, and EOS as decoder start. `tokenizer.json` and HF tokenizer compliance files are intentionally ignored. The authors' public code establishes Fairseq and an HF port; it does not verify an earlier OpenNMT origin. Parameter/context claims differ between their documents, so these checks inspect local config/tensor shapes rather than adopting those claims as constants.

## Prepare immutable assets and representative probes

For structural checks, install `pip install -e '.[preflight]'`. Place these files in a local asset directory:

- `fairseq_dict.json` containing `src`, `tgt` piece-to-integer-ID dictionaries and `special` IDs (`bos`, `pad`, `eos`, `unk`).
- `onemtv3b_spm.model`.
- `config.json` and optionally `generation_config.json`.
- Optionally `model.safetensors`, or `model.safetensors.index.json` and all referenced shards, to inspect actual embedding/output tensor shapes. Only headers are inspected; the structural checker does not load billions of parameters. Without weights, row compatibility remains unverified.

Pin a Hub commit and keep the original directory unchanged. Put candidate tokenizer/dictionary/checkpoint assets in a separate directory. Model downloads are not automatic. No credentials are required by the checker. For actual HF generation, install `pip install -e '.[train]'` and supply local safetensors weights; pickle-backed `.pt`/`.bin` weights and remote custom code are not loaded.

A probe JSONL row:

```json
{"id":"old-en-hi-001","source":"The book is here.","target":"किताब यहाँ है।","src_lang":"eng_Latn","tgt_lang":"hin_Deva","group":"retention","domain":"general","document_id":"doc-001"}
```

Use `group: retention` for existing languages/domains and `group: adaptation` for new languages **or new data for existing languages**. Keep independent old-domain probes when adapting an already supported language: unchanged vocabularies can still suffer forgetting. One file can contain multiple directions; supply row language codes or common `--src`/`--tgt` defaults. Only `retention`, `adaptation` and `unspecified` groups are accepted. IDs must be unique. References must be nonempty strings.

The files under `data/preflight/` are illustrative software fixtures, not validated linguistic gold or a sufficient regression benchmark. Replace them with native-speaker-reviewed sentences covering names, numbers, spelling variants, agreement, register and multiple domains. Retain a final test set outside probe-driven tuning.

## 1. Check original assets and current vocabulary coverage

```bash
bhaashik preflight \
  --assets data/local/onenmt-original \
  --corpus data/preflight/probes.jsonl \
  --report runs/preflight/original.json
```

The report contains asset hashes, per-group/direction/side unknown rates, dictionary misses, SentencePiece unknown pieces, tokens per whitespace word, roundtrip differences, encoded direction prefixes, collisions, and source context overflows. Prefixes consume context too. No truncation is applied. `--max-source-tokens` can impose a smaller budget; the config's positional limit remains an upper bound.

Whitespace fertility is descriptive, not a script-independent quality measure. SentencePiece normalization may legitimately change roundtrip text. Source coverage includes separately reported direction prefixes; target coverage measures plain reference text, not a verified training-label format. Verify the original target preprocessing before implementing finetuning.

The dictionary checker rejects duplicate JSON keys, duplicate/negative/noninteger IDs and mismatched special symbols. IDs need not be contiguous: required rows are **max ID + 1**, not dictionary length. Tensor checks inspect available shared/encoder/decoder/output matrices and compare rows/widths against config. They do not establish full architecture equivalence or that all checkpoint tensors are correct.

## 2. Assess append-only proposals

```bash
bhaashik preflight \
  --assets data/local/onenmt-original \
  --corpus data/local/probes.jsonl \
  --selected-tokens data/preflight/selected-tokens.example.json \
  --plan-output runs/preflight/proposed-fairseq-dict.json \
  --report runs/preflight/append-plan.json
```

The selected-token format is `{"src":["▁piece"],"tgt":["▁piece"]}`. The optional proposal appends IDs **after the original maximum**, preserving every existing mapping separately on each side. Existing pieces retain their IDs. It is a dictionary proposal, not a ready-to-use checkpoint. Outputs must be outside asset directories and cannot overwrite input files.

For each requested piece the report records whether it already exists, its proposed ID, whether it is an active SentencePiece piece, and observed emissions in the probes. **Adding a word to the dictionary does not make SentencePiece emit that word as a piece.** An inactive addition has no effect without a separately designed tokenizer extension. For existing SPM pieces missing from the Fairseq dictionary, an append may remove an unknown mapping, but corresponding model rows still need initialization and training.

New input/output rows must exist in the checkpoint and receive gradients. Attention-only LoRA does not update frozen new embedding rows. A future training adapter could update selected rows; this preflight does not implement that optimizer behavior. Do not assume that passing an ID-preservation check makes the new rows useful.

## 3. Compare a candidate replacement, append or reordered dictionary

```bash
bhaashik preflight \
  --assets data/local/onenmt-original \
  --candidate-assets data/local/onenmt-candidate \
  --corpus data/local/probes.jsonl \
  --report runs/preflight/candidate.json --strict
```

The report distinguishes:

| Finding | Meaning / action |
|---|---|
| Existing piece removed, remapped or an old ID hole reused | Block direct checkpoint reuse. Embedding semantics no longer agree. |
| Append preserves IDs but model rows are too small | Block until the separately prepared candidate checkpoint covers the IDs. |
| New dictionary entries never emitted | Vocabulary growth alone will not help. |
| Same IDs, changed SentencePiece/normalization | Measure changed segmentation and encoded sequences; run behavior probes. |
| Same tokenizer and IDs | Lower structural risk; forgetting/poor outputs remain possible after finetuning. |
| Direction prefixes collide | Block: the model receives indistinguishable direction instructions. |
| Unknown direction-prefix pieces | Review language routing before training. Distinct encodings alone do not establish learned support. |

Replacing/retraining SentencePiece does not always reorder dictionary IDs. The report separates those events. Even with stable IDs, new segmentation changes the model's input distribution. An unchanged probe result does not prove global equivalence; changing the tokenizer artifact is explicitly flagged. The module compares a candidate you provide; it does not train a new SentencePiece model or rewrite tokenization rules.

Structural status is `blocked`, `review_required`, or `checks_passed`. The latter means the structural checks passed **on these inputs**, not that the model is safe to train or that translation works. `decision` remains `behavioral_validation_required` unless direct reuse is blocked. CLI exit code 2 means blocked; `--strict` also returns 2 for review findings. Exit 0 is not a scientific go/no-go decision. `--max-unknown-rate` changes the review threshold (default zero), not the tokenizer.

## 4. Check HF behavior before and after adaptation

```bash
bhaashik hf-regression \
  --baseline-assets data/local/onenmt-original \
  --candidate-assets data/local/onenmt-adapted \
  --corpus data/local/probes.jsonl \
  --report runs/preflight/behavior.json \
  --device cuda --precision fp16 --batch-size 1 --max-new-tokens 64 \
  --max-retention-drop 2.0
```

For tokenizer-only changes, copy the same HF checkpoint into the candidate directory alongside its candidate assets to isolate tokenization effects. For new-language or new-domain training, compare untouched vs adapted checkpoints with their respective assets. Models load sequentially, not simultaneously. CPU/fp32 is the default; CUDA/fp16 is an explicit option. Neither peak GPU memory nor full-scale runtime has been measured for OneNMT here.

Generation uses source prefix → SPM pieces → source dictionary IDs, and target dictionary IDs → pieces → decoded output. It uses EOS as decoder start, no forced BOS, deterministic greedy decoding and neutral repetition constraints, shared across both runs. It fails on nonfinite generation logits, insufficient context, or incompatible inspected tensor dimensions. This is a controlled diagnostic profile, **not the authors' published beam-search profile**. Full evaluation can use other declared, matched decoding settings.

The behavioral command also runs the structural preflight and stops on blocking findings before loading model weights. It builds a fresh, shared generation profile rather than inheriting candidate-specific decoding defaults. Results include outputs, package versions, settings, elapsed times and asset hashes. Hashes cover dictionary/tokenizer/config files, not multi-gigabyte weights; preserve immutable checkpoint revisions and weight checksums separately. The checker reports chrF++/BLEU and changed-output counts separately for every retention/adaptation direction. Retention losses cannot be hidden by gains on a new language. The default 2-point chrF++ retention budget is a configurable triage policy, not a significance test or universal acceptable loss.

Heuristics flag empty outputs, unknown tokens, replacement characters, repeated trigrams, generation-limit hits and source copies. These are review signals, not automatic judgments of linguistic quality; source copying can be legitimate for related languages. A changed output is not inherently a regression. Inspect retained and new-language outputs with qualified speakers.

### Independent HF reference replay

Optionally add `--reference data/local/trusted-reference.jsonl`, containing the **same IDs, source texts, references and `prediction`** recorded independently from the authors' HF wrapper or a trusted HF run with exactly the same prefixes/assets/decoding profile. The original wrapper's default beam size and repetition constraints differ: do not compare defaults against this greedy profile. A reference mismatch blocks proceeding until its cause is understood.

Without independent reference outputs, reference parity remains unverified: two equally wrong HF runs can agree. Exact decoded-text replay does not prove logits equivalence, correct original conversion, corpus quality or new-language support. No Fairseq/OpenNMT runtime is required or invoked. The checker does not assert an unverified model-port history.

If an actual checkpoint cannot pass these checks, investigate its assets or upstream HF conversion; do not spend this project on an architecture reimplementation. Pin the working HF model/assets/software versions once verified.

## Apply justified changes separately

Use the [candidate repair modules](repairs.md) for generation-ID fixes, trusted dictionary restoration, append-only IDs, SPM extension/retraining/replacement, and HF vocabulary-row initialization. Each operation writes a new candidate; rerun this preflight and the behavior probes afterward. Repair manifests marking new rows as requiring training are reported as pending review. A successful mutation does not clear failed checks.

## Recommended decision sequence

1. Start with unchanged tokenizer/dictionaries. Inspect prefixes, coverage and independent HF reference outputs.
2. Run a small adaptation pilot with new-language/new-domain **and retained-language** probes. Review both behavior and metric changes.
3. If coverage is inadequate, investigate a small append-only candidate and repeat the same checks, including trained new rows and save/reload tests in the eventual training implementation.
4. Treat SPM replacement as a separate high-risk ablation; block direct reuse when dictionary IDs change. Keep the original assets and benchmark condition intact.
5. Use existing evaluation/bootstrap/error-analysis tools for final research comparisons with NLLB and IndicTrans2. Keep each model's native preprocessing, independent test data and declared language-support status.

Validation in this repository uses synthetic assets and a tiny random **standard HF MBart** model, including actual generation and a deliberately corrupted NaN checkpoint. It demonstrates checker mechanics, not OneNMT conversion parity, pretrained quality, GPU fit or successful finetuning. Full OneNMT inspection still requires its actual accessible assets and expert-reviewed probes.
