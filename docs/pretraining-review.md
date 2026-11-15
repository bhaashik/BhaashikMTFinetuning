# Pre-training diagnostics and seven-language baseline comparisons

The original `notebooks/01_nllb_research.ipynb` performs corpus review and an
**unfinetuned development-set** comparison before approving real-data training.

## Coverage and research scope

| Language | Project corpus tag | NLLB-200 native | IndicTrans2 native | BhashaVerse HF checkpoint |
| --- | --- | --- | --- | --- |
| Bhojpuri | `bho_Deva` | yes | no | paper claim; not in card's 23-language list |
| Maithili | `mai_Deva` | yes | yes | explicitly listed |
| Magahi | `mag_Deva` | yes | no | paper claim; not in card's 23-language list |
| Awadhi | `awa_Deva` | yes | no | paper claim; not in card's 23-language list |
| Braj | `bra_Deva` (local label) | no | no | paper claim; not in card's 23-language list |
| Nepali | `npi_Deva` | yes | yes | explicitly listed |
| Marwari | `mwr_Deva` (local label) | no | no | not independently confirmed |

A project's local corpus label is **not** a checkpoint's native language tag.
Do not substitute Hindi, Marathi or another tag to make an unsupported model
run; a deliberately documented tag-transfer experiment is different from a
native checkpoint baseline. A research paper's coverage claim is not a
guarantee that the particular published checkpoint + code supports a direction.

Sources:
- [NLLB tokenizer inventory](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/nllb/tokenization_nllb.py)
- [IndicTrans2 distilled Indic–Indic](https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M)
- [BhashaVerse / OneNMT HF card and mandatory inference script](https://huggingface.co/ltrciiith/bhashaverse)

## Running the notebook

1. Clone in the existing location:
   `/kaggle/working/BhaashikMTFinetuning` on Kaggle, and
   `/content/BhaashikMTFinetuning` on Colab. No new directory conventions.
2. With `SETUP_MODE='base'`, run corpus inspection without a GPU. Supply
   `data/local/{PAIR}/train.jsonl`, `dev.jsonl`, `test.jsonl`, and
   set `USE_EXAMPLE_DATA=False` for actual data.
3. Inspect `runs/{PAIR}-pretrain-review/corpus_diagnostics.json`,
   `sentence_lengths.csv`, and charts, especially split leakage, missing
   licensing/provenance, unusually long/short sentences and near-duplicates.
   The near-duplicate heuristic is bounded, approximate and not exhaustive.
4. Enable a GPU, start a fresh kernel, set `SETUP_MODE='train'` and rerun the
   notebook. It will sample **dev**, not test. The NLLB base checkpoint produces
   dev predictions when its native tags exist.
5. Optional comparators are **disabled by default**. For IndicTrans2, install
   the upstream `IndicTransToolkit` in a compatible dedicated environment;
   review trusted upstream model code, accept any gated model license, and only
   then set `RUN_INDICTRANS2=True` and
   `ALLOW_INDIC_REMOTE_CODE=True`. The notebook uses the upstream
   `IndicProcessor` pre/post-processing. For non-English indic directions,
   the default is the 320M Indic–Indic distilled checkpoint; English directions
   use the appropriate 200M distilled checkpoint.
6. For BhashaVerse, download/review its **upstream** `hf_inference.py` along
   with required assets, and set `BHASHAVERSE_INFERENCE_SCRIPT` and
   `RUN_BHASHAVERSE=True`. Generic `AutoTokenizer` or MBart IDs are
   **incorrect** for this model. Only languages documented in the HF card are
   enabled by this adapter. Alternative unsupported directions need separate,
   verified inference support, not silent substitution.
7. Running two models can need different dependencies or more RAM than Kaggle
   provides. You may run each adapter in its own notebook/session, preserve
   its predictions JSONL, and supply them in the primary notebook using
   `EXTERNAL_DEV_PREDICTIONS = {'Name': Path('...')}`. The comparison rejects
   mismatching dev IDs, references and metadata.
8. Inspect `model_comparison.json`, `model_comparison.csv`,
   `dev_side_by_side.csv` and `dev_model_comparison.png`. The sheet is a
   review aid, **not a properly blinded human-annotation protocol**.
   chrF++ and sacreBLEU are descriptive; report metric signatures,
   direction, sample size, data/model hashes and limitations. Small samples
   do not support quality claims.
9. Only after review, set `APPROVE_REAL_TRAINING=True`. Test translations
   are generated **after** training; freeze test data and never use test scores
   to choose training hyperparameters.
10. Final run ZIP includes the pretraining reports alongside the selected
    exported model, predictions, baseline and config. Intermediate Trainer
    checkpoints are excluded unless `INCLUDE_CHECKPOINTS=True`, which is
    necessary for exact resume.

## Reproducibility and limits

These baseline adapters are *optional* and have not been benchmarked or
certified on live Kaggle/Colab GPUs. They may require separate environment
setup; BhashaVerse model is approximately 1.1B parameters, and the optional
IndicTrans2 checkpoints download additional weights. The original NLLB
workflow remains the default. Repeated training seeds and blind speaker review
are needed for research conclusions. See [research protocol](research.md),
[data contract](data.md), and [Kaggle guide](kaggle.md).
