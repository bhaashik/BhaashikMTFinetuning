# Four stages of translation-corpus evaluation

The objective is to **document the usefulness, quality and novelty of the
fine-tuning corpus**, not merely maximize one automated metric. Use the same
parallel source/target data, split seed and test references for comparable
experiments. The three notebooks share the same `stage` setting and
`bhaashik.staged_experiments` runner.

## The four stages

| Stage in settings | What it actually does | Development/eval split | Test split | Training? |
|---|---|---|---|---|
| `lowest_baseline` (DEFAULT) | Untouched pretrained model, standard decoding, scores, plots, reviewable outputs | **Full** model-generated translations + BLEU/chrF++ | **Full** model-generated translations + BLEU/chrF++ | No |
| `low_baseline` | Full real fine-tuning run, then the exact same inference and reporting as Stage 1 | Loss measured after every training epoch; after training, full translations and scores | Full translations and scores | Yes |
| `decoding_search` | Try beam size/length-penalty candidates on development data, select the winner | All candidates evaluated on dev; selected settings and candidate scores retained | **Only the selected** settings evaluated on test | No new fine-tuning |
| `language_extension` | Controlled vocabulary/tag initialization or prior repaired checkpoint, then train | Training loss + full translations and metrics | Full translations and metrics | Yes, with trained new embeddings |

### Stage 1 — lowest baseline (recommended starting point)

The default `EXPERIMENT["stage"]` is `"lowest_baseline"`. For NLLB,
specify your two aligned UTF-8 sentence files, set native `source_lang`
and `target_lang`, then run all cells. It creates a reproducible train/dev/
test split and **evaluates the entire dev and test sets without training**.

Each stage directory includes:

- `dev_predictions.jsonl` and `test_predictions.jsonl` (original source,
  target/reference, sentence ID, language tags, model prediction).
- `dev_scores.json` and `test_scores.json` (corpus BLEU/chrF++,
  score signatures, per-domain and per-direction scores).
- `dev_review.csv` and `test_review.csv` (shuffled rows and heuristic
  flags for later human linguistic review).
- `dev_test_scores.svg` (figure for dev-versus-test performance).
- `stage_manifest.json`, `stage_results.json`, model/data/output hashes,
  experiment configuration and the independent stage archive.
- Input alignment, original files, and split-manifest provenance are
  preserved. The ZIP does not need a newly trained checkpoint.

The dev/test performance difference is **descriptive only** in Stage 1: it
helps characterize the two samples and the pretrained model. It is **not
used for choosing training hyperparameters**.

### Stage 2 — low baseline (real fine-tuning)

Change only:

```python
"stage": "low_baseline",
"allow_training": True,
```

Use `"finetune_mode": "full"` for full-model parameter finetuning, or
`"lora"` if GPU memory is insufficient; these methods should be reported
as *different experimental conditions*. Use `"max_steps": -1` for the
full requested number of training epochs. `max_steps=1` is **only a
software smoke test**, not a complete experiment.

Stage 2 records per-epoch development/evaluation loss and the best
checkpoint, saves `trainer_history.json`, `training_loss.csv` and
`training_loss.svg`, and then evaluates **all dev and test sentences**
using the same fixed decoding settings as Stage 1. If Stage 1 was run using
the same `run_label`, its dev/test scores are compared in
`comparison_to_untouched.json`. Training and evaluation results are
separately archived even if later reporting encounters an error.

**Difference between loss and inference:** computing dev loss performs
teacher-forced forward passes, not expensive autoregressive search.
Full dev/test translations are generated once **after** training.

### Stage 3 — decoding search, only if needed

Change:

```python
"stage": "decoding_search",
"allow_training": False,
"search_source_stage": "low_baseline",
"search_metric": "chrfpp",
"search_grid": [
    {"beams": 1, "length_penalty": 1.0},
    {"beams": 4, "length_penalty": 1.0},
    {"beams": 6, "length_penalty": 1.0},
],
```

The model must be present at
`runs/MODEL-PAIR-RUN_LABEL-sSEED/low_baseline/model/`
from Stage 2 (or restored from an archive). The runner writes all
development predictions/scores for each candidate to
`candidates/`, chooses the best setting according to the selected
**development** metric, and writes `decoding_search.json` before it
ever runs generation on the test set. The test set is generated and
evaluated **once**, with the selected setting.

The current search covers reproducible **autoregressive decoding
hyperparameters** (beam width and length penalty), not reinforcement
learning or gradient-based hyperparameter search. Changing the number
of training steps, model parameters or datasets is a new experiment.

### Stage 4 — model/language extension

New-language experiments must keep original model vocabulary IDs stable.
They are not needed for the current native NLLB experiment.

- **NLLB:** set `stage="language_extension"`,
  `conditioning="dedicated_token"`, `extend_language` to the new
  actual corpus language tag, `finetune_mode="full"`, and
  `allow_training=True`. The existing extension function initializes a
  new language token using a documented donor token and saves an
  independent tokenizer/model candidate. Training must then update the
  new embeddings. This does *not* confer native pretrained support.
- **IndicTrans2:** `conditioning="dedicated_token"`, full
  finetuning, and explicit approved native processor proxies where
  needed. The processor, tokenizer and decoder handling must be
  reviewed before accepting new-language outputs.
- **BhashaVerse:** do not replace or reorder fairseq dictionary IDs.
  First use the separate `02_onenmt_preflight_repairs.ipynb` to prepare
  a reversible, reviewed SPM/dictionary/embedding candidate and
  `repair_manifest.json`. Stage 4 refuses unreviewed assets.
  Independent upstream-output parity remains a prerequisite.

## Stage-independent archive structure

Every stage is saved **separately**:

```text
/kaggle/working/BhaashikMTFinetuning/
  runs/nllb-bho-hin-corpus-study-1-s42/
    lowest_baseline/
      stage_manifest.json
      stage_results.json
      dev_predictions.jsonl
      test_predictions.jsonl
      dev_scores.json
      test_scores.json
      dev_review.csv
      test_review.csv
      dev_test_scores.svg
    low_baseline/
      model/                 # Selected trained model/tokenizer.
      training_history.json
      training_loss.csv
      training_loss.svg
      ...                   # Full predictions, metrics and plot.
    decoding_search/
      candidates/            # Development-only candidate predictions.
      decoding_search.json
      ...                   # Winner's dev/test predictions and report.
    language_extension/
      extended-tokenizer-model/  # NLLB only.
      model/
      ...                   # Full predictions, metrics and losses.
```

Each stage has a corresponding config under `configs/` and a ZIP in
`/kaggle/working/` named `MODEL-PAIR-RUN_LABEL-sSEED-STAGE-artifacts.zip`.
The archive is attempted automatically after each stage, **even if
inference/training fails**; an incomplete run is labelled
`"status": "incomplete"` in `stage_manifest.json`. Saved ZIPs
will never silently overwrite previous archives. The original model
is not copied into Stage 1 results: its Hub ID/revision or local
asset path and input/output hashes are recorded.

`include_checkpoints=False` excludes nested Trainer optimizer snapshots
but retains an exported selected finetuned model, its tokenizer,
losses and results. Use `include_checkpoints=True` to preserve
optimizer state for exact resumption, at much greater storage cost.

## Scientific reporting recommendations

The contribution of a low-resource corpus is not synonymous with BLEU.
In each stage preserve examples where semantic accuracy, morphology,
dialect features, code switching, script, transliteration, numerals,
named entities, register and lexical choice differ. Include provenance,
sampling criteria, license, annotation conventions, sentence alignment
accuracy and speaker/domain balance in the eventual corpus release or
paper. Manual blind expert review can reveal important phenomena missed
by aggregate BLEU and chrF++.

**Test-set contamination warning:** the user's planned Stage 1/2 includes
examining the full test scores. That is acceptable for *descriptive corpus
research*, but if those test findings influence subsequent Stage 3 model
selection, the same test set is no longer a fully blind final benchmark.
For a claim about optimized generalization, reserve another untouched
external or final blind test sample. Stage 3 itself always selects decoding
parameters exclusively on dev.

## Notes specific to each model

NLLB: works with native language tags and pinned Transformers/PEFT.
IndicTrans2: requires separate installation of official IndicTransToolkit
and a deliberate `allow_remote_code=True` opt-in after reviewing source.
BhashaVerse: requires reviewed local MBart-compatible safetensors,
original SentencePiece/fairseq dictionary and upstream generation parity.
Neither paper-level language coverage nor an extra prefix proves support.

To run in Kaggle/Colab, see [the practical notebook guide](notebook-platforms.md).
For input conversion, see [data preparation](data.md) and for old
tokenizer repairs see [multilingual adaptation](multilingual-adaptation.md).
