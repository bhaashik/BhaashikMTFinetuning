# Experiment settings: NLLB, IndicTrans2 and BhashaVerse

## Quick start: one real NLLB experiment

Open the updated `notebooks/01_nllb_research.ipynb`. **Edit only the first
Python settings cell**, then run all cells in order in a fresh Kaggle/Colab
session. The default is a real (not toy) native NLLB experiment:

```python
"data_mode": "aligned_text",       # Your two aligned source/target .txt files.
"source_lang": "bho_Deva",
"target_lang": "hin_Deva",
"source_text": "/kaggle/input/YOUR_DATASET/bho.txt",
"target_text": "/kaggle/input/YOUR_DATASET/hin.txt",
"split_seed": 42,
"split_ratios": (0.80, 0.10, 0.10),
"conditioning": "native",         # No new language token or proxy.
"finetune_mode": "lora",
"epochs": 3,
"max_steps": -1,                  # -1 uses full epochs; 1 is an install/train smoke test.
"validate_data": False,
"data_report": False,
"pretrain_baseline": False,
"posttrain_test": False,
"publication_report": False,
"allow_training": True,
"save_archive": True,
```

You only need to replace the input paths and, if appropriate, the language
pair. Models are **not** downloaded until the model setup/baseline/training
stage. Training includes **development loss at the end of each epoch** and
restores the checkpoint with the lowest development loss. This is important:
`pretrain_baseline=False` does *not* turn off development validation loss.
It only skips slow inference from the untouched model. Similarly,
`publication_report=False` skips extra BLEU/chrF++ computations and
visualization, but does not affect model training or checkpoint selection.

If you already have a single tab-separated plain-text file rather than two
aligned files, use `data_mode="tsv"` and set `tsv_file` (two columns
`source<TAB>target`, optional header controlled by `tsv_has_header`).
JSONL and already split JSONL also remain supported via
`data_mode="jsonl"` / `data_mode="pre_split"`.

## The required notebook cells

1. **Settings**: the only editable experiment cell; no expensive operations.
2. **Environment**: clone/update the checkout and install dependency stack.
   This is a required cell; it uses `setup_mode="train"` for GPU training.
3. **Data preparation**: resolve input paths, convert if necessary, split,
   record data hashes, display counts, write a fixed experiment configuration.
4. **Optional dev baseline**: skipped with `pretrain_baseline=False`.
5. **Training**: real LoRA/full finetuning when `allow_training=True`.
   The selected checkpoint is saved under `runs/`.
6. **Optional final test**: skipped with `posttrain_test=False`. It can
   generate held-out predictions after model selection. Set
   `publication_report=True` for corpus scores and review sheets.
7. **Archive**: `save_archive=True` packages the exported model and config.
   Use `include_checkpoints=True` for resuming exact optimizer state.

The name of each model notebook, rather than its flags, selects the model
family. All three use shared `bhaashik.experiment_workflow` functions for
paths, conversion, split manifests and reports.

## Settings explained

| Setting | Purpose | Relation to other settings |
|---|---|---|
| `data_mode` | Input format: aligned_text, tsv, jsonl, pre_split, example | The corresponding file/directory field is read |
| `source_lang`, `target_lang` | Language/script tags | Native tags must be supported by the selected checkpoint; no silent substitutions |
| `source_text`, `target_text` | Two aligned UTF-8 files | Used only with aligned_text; line counts and blank lines checked |
| `tsv_file`, `tsv_has_header` | One tab-separated text file | Used only with tsv; exactly two nonempty columns |
| `parallel_jsonl` | Already converted single JSONL file | Used only with jsonl |
| `split_directory` | Directory containing train/dev/test JSONL | Used only with pre_split |
| `dataset_id` | Stable name for a version of the corpus | Use a new ID when files change; changing the model should not change it |
| `split_seed` | Random seed for the *data partition* | Keep the same across model comparisons |
| `split_ratios` | Train/dev/test proportions | Three positive fractions totaling 1 |
| `group_by_document` | Group existing document IDs | Prevents same-document leakage when IDs exist; normalized-source duplicates always grouped |
| `validate_data` | Full exact audit of partitions | False only skips optional full checks; trainer still audits train/dev |
| `data_report` | Corpus plots, anomalies and warnings | Extra publication preparation; unrelated to train/development loss |
| `model_id` | Base NLLB/IT2 Hugging Face identifier | NLLB default: facebook/nllb-200-distilled-600M; IT2 can choose automatic direction-specific model |
| `model_assets` | BhashaVerse local checkpoint and original tokenizer/dictionary | Only used in BhashaVerse notebook |
| `conditioning` | Native NLLB/IT2 tags, text-prefix or special-token extension | Keep native for first real run; unsupported tags need an explicit research design |
| `finetune_mode` | lora or full | Full has much greater memory cost; new language embedding rows need full training |
| `train_seed` | Model parameter/random training seed | Unlike split_seed, vary this for model stability checks |
| `epochs` | Full traversals of training set | `max_steps=-1` uses all requested epochs |
| `max_steps` | Override total optimization steps | 1 only for a smoke test; -1 for a real run |
| `batch_size` | Examples per GPU training step | Increase only if memory permits |
| `gradient_accumulation_steps` | Accumulate gradients over N steps | Effective batch ~= batch_size × accumulation × GPUs |
| `learning_rate` | Update step size | Default 2e-4 LoRA; reduce when changing to full finetuning |
| `max_length` | Maximum tokenized source/target length | Longer sequences cost more memory; NLLB truncates when exceeded |
| `precision` | fp16/bf16/fp32 | Use fp16 on T4; bf16 only when GPU supports it |
| `gradient_checkpointing` | Reduce activation memory by recomputing | Slower but useful for modest GPU memory |
| `allow_training` | Explicit training ON/OFF | NLLB default True; IT2/BhashaVerse default False until extra assets vetted |
| `pretrain_baseline` | Untouched model translation on a dev sample | Extra inference; not needed for per-epoch evaluation loss |
| `posttrain_test` | Generate held-out test translations after training | Separate from development loss; default False |
| `publication_report` | BLEU, chrF++, qualitative sheets | Requires posttrain_test=True, not needed for training |
| `test_baseline` | Additional untouched-model test inference | NLLB-only optional comparison; expensive |
| `run_label` | Name one configuration/results run | Change when training settings change; outputs never silently overwritten |
| `save_archive` | Save model+config ZIP | Recommended True in ephemeral Kaggle sessions |
| `include_checkpoints` | Keep optimizer/scheduler checkpoint history in ZIP | Default False; True adds large files needed for exact resume |
| `setup_mode` | Runtime packages and GPU check | Use train for NLLB/IT2 GPU; train,repairs for BhashaVerse |
| `refresh_repo` | Update the checkout by git pull --ff-only | False for a stable session; True when you want the latest checked-in code |

The model notebooks may expose additional special-purpose fields, such as
`extend_language`, `language_donor`, `proxy_source_lang`,
`proxy_target_lang`, `check_assets`, and `allow_remote_code`.
They are disabled by default and only apply to experiments explicitly adding
unsupported languages, adapting processing proxies, or using trusted remote
model code. Do **not** enable them for native-language initial training.

## How to switch experiment type by changing one or two lines

**Native real-data run:** defaults above; edit paths and tags once.
No manual edits in later cells.

**One-minute mechanics test:** change `data_mode="example"`,
`max_steps=1` and make sure `source_lang=bho_Deva` and
`target_lang=hin_Deva`. The example has only two sentences per split;
its scores are meaningless. If you want a real-data pilot with one step,
leave `data_mode="aligned_text"` and change **only** `max_steps=1`.

**Publication data audit:** set `validate_data=True` and
`data_report=True`. The report code is in the library, not spread
across notebook cells.

**Post-training publication evaluation:** keep the same
`run_label`, `train_seed`, dataset and model/trainer settings; change just:

```python
"allow_training": False,
"posttrain_test": True,
"publication_report": True,
```

You can restart the session, reimport the same notebook and run all its
cells in order **provided the earlier exported model is still present** under
its original `runs/RUN_NAME/` path. On a fresh Kaggle runtime, first restore
the model from your saved version or archive (and restore/recreate the same
input data and split manifest). Otherwise `posttrain_test=True` will
correctly report that the trained checkpoint is missing. The shared setup
preserves the original training configuration and verifies the same data,
while skipping the trainer and generating final test/report files.
It creates a separate
`*-publication.zip` instead of overwriting the original training ZIP.
Do not change the model, training hyperparameters or `run_label` in this
reporting-only pass.

**More seeds:** keep `split_seed` fixed but change `train_seed` and
`run_label` for each independent model run. Compare on the same held-out
dev/test IDs; freeze the test split before hyperparameter selection.

**Exact training resumption:** set `include_checkpoints=True` *before*
archiving and preserve the checkpoint folder. The current notebook is not a
general automatic resume manager; advanced users can pass
`resume_from_checkpoint` in the trainer config.

## What is evaluated in a real finetuning experiment?

Three activities are often confused:

1. **Development loss (default ON)**: Trainer computes held-out dev
   cross-entropy/loss after every training epoch, uses it to select the
   best model, and evaluates once after training. This costs model forward
   passes on dev, but **does not run autoregressive beam-search** or
   calculate corpus BLEU/chrF++ each epoch.
2. **Optional untouched-model dev baseline (default OFF)**: translation
   and BLEU/chrF++ of the original model on a fixed sample; additional GPU
   inference for reporting.
3. **Optional final test/publication evaluation (default OFF)**: after model
   selection, translate the held-out test set and optionally compute BLEU,
   chrF++, per-language scores, source-copy/error flags and review sheets.

The data audit controls are separate from these: disabling expensive
corpus diagnostics does **not** disable development loss. The model trainer
still checks exact train/dev leakage and rejects duplicate parallel pairs.
A `validate_data=False` run is a **pilot**, not a publication-quality
data validation; run the full audit before reporting a result.

## Common configuration and runtime errors

- `NameError: REVIEW_DIR`/`require_routes`: old notebooks depended on
  earlier optional cells. Reimport the new one-cell-configuration notebook;
  its required data-preparation cell creates all paths and variables.
- Data path incorrect: print actual Kaggle input paths in the platform guide.
  Change the single `source_text`/`target_text` setting, not library code.
- Invalid language: the native NLLB path deliberately refuses unsupported
  tags rather than quietly using a Hindi proxy.
- Model out of memory: try `batch_size=1`, `max_length=128`,
  `gradient_checkpointing=True`; choosing a smaller model may help.
- Need fast initial testing: set `max_steps=1` without reducing your corpus.
- Repeated output error: create a new `run_label` for a new experiment or
  deliberately restore/resume an existing run; do not delete good checkpoints.
- Dataset file was replaced: change `dataset_id`, preserving the old split
  manifest for reproducible comparisons.

See [platform-specific paths and UI](notebook-platforms.md), [parallel
data contract](data.md), and [new-language experiments](multilingual-adaptation.md).
