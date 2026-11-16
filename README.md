# Bhaashik · Hindi-belt neural machine translation

A reproducible research toolkit for evaluating and adapting encoder–decoder
translation models (NLLB, IndicTrans2 and BhashaVerse/OneNMT) on low-resource
parallel corpora. The principal focus is **corpus novelty, linguistic quality
and provenance**, not merely automated score improvements. It supports
Bhojpuri, Maithili and Magahi, with exploratory experiments for Awadhi, Braj,
Nepali and Marwari.

## Four experiment stages

| Setting | Purpose | Default? |
|---|---|---|
| `stage="lowest_baseline"` | Untouched pretrained model; full dev/test translations, BLEU/chrF++, plots, review sheets and an independent ZIP | **Yes** |
| `stage="low_baseline"` | Real finetuning with per-epoch dev loss; afterwards full dev/test inference and comparison to Stage 1 | No |
| `stage="decoding_search"` | Try autoregressive decoding settings on dev; select once, then test | No |
| `stage="language_extension"` | Cautious tokenizer/tag extension with full embedding training | No |

**Archiving does not depend on training.** Every stage creates its own
manifest, prediction files, scores and ZIP. A partial or interrupted stage
is explicitly marked incomplete. Stages share the same data and split seed.

Read the **[four-stage guide](docs/four-stage-workflow.md)** for the exact
outputs, scientific interpretation and test-set caveats.

## Start with a notebook

Use the notebook for your model family:

- [NLLB notebook](notebooks/01_nllb_research.ipynb)
- [IndicTrans2 notebook](notebooks/03_indictrans2_adaptation.ipynb)
- [BhashaVerse/OneNMT notebook](notebooks/04_bhashaverse_adaptation.ipynb)
- [Original OneNMT preflight and repair notebook](notebooks/02_onenmt_preflight_repairs.ipynb)

**Edit just the one `EXPERIMENT` settings cell near the top** of a fresh
notebook, then execute cells in order. The notebook automatically clones the
GitHub repository if it is absent. Default Stage 1 requires real
source/target data paths and a selected inference model, but no finetuning.
The NLLB notebook defaults to native `bho_Deva → hin_Deva` language tags.

For Stage 2, change `stage` to `"low_baseline"` and
`allow_training` to `True`. Set `max_steps=-1` to complete the
chosen number of epochs. `finetune_mode="full"` updates all model
parameters; change to `"lora"` on smaller GPUs.

For a very small mechanics test, change `data_mode="example"`
(Bhojpuri→Hindi fixture). It is **not** suitable for scientific scores.

## Supported input data

- Two aligned UTF-8 text files, one source sentence and reference translation
  per corresponding line (`data_mode="aligned_text"`).
- A two-column tab-separated text file (`data_mode="tsv"`).
- One parallel JSONL file (`data_mode="jsonl"`).
- Existing `train.jsonl`, `dev.jsonl`, `test.jsonl`
  (`data_mode="pre_split"`).

Conversion never modifies the original files. A seeded splitter keeps
identical normalized source sentences and available document IDs together.
Data and output hashes, generated IDs and language tags are preserved.

## Practical usage and documentation

- [Kaggle/Colab operations](docs/notebook-platforms.md): clone location,
  attached datasets, GPU/Internet controls, Drive mounting, session outputs
  and saving ZIPs.
- [Full experiment settings](docs/experiment-guide.md): meaning and
  interactions of important fields. The **four-stage guide supersedes**
  older separate-baseline/reporting switches.
- [Input and split format](docs/data.md)
- [Multilingual tag adaptation](docs/multilingual-adaptation.md)
- [Pretraining corpus review](docs/pretraining-review.md)
- [Kaggle legacy guide](docs/kaggle.md), [Colab legacy guide](docs/colab.md),
  [Titan V guide](docs/titan-v.md)

Model-specific requirements: IndicTrans2 needs the reviewed official
IndicTransToolkit and explicit opt-in to custom model code. BhashaVerse
needs a verified HF MBart-compatible local checkpoint and original
SentencePiece/fairseq dictionary mapping, with upstream output-parity
checks. Adding a language tag alone does **not** establish new-language
translation ability.

Tests run using small synthetic checkpoints and mocked inference on CPU;
published-model GPU runtime, real data quality and linguistic evaluation
must be independently checked before research claims.

## Code layout

```text
BhaashikMTFinetuning/
  pyproject.toml
  README.md
  AGENTS.md
  notebooks/
  src/bhaashik/
  tests/
  docs/
  scripts/
  data/examples/
```
