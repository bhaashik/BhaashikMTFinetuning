# Bhaashik · Hindi-belt neural machine translation

A research toolkit for adapting encoder–decoder MT models to **Bhojpuri, Maithili, and Magahi**, with Hindi and English as bridge languages. Designed for Colab/Kaggle GPUs and standard Hugging Face checkpoints. This is an experiment framework, **not a claim of new state-of-the-art results**.

## Start

**[Quickstart](QUICKSTART.md)** · **[Colab guide](docs/colab.md)** · **[Kaggle guide](docs/kaggle.md)** · **[Titan V guide](docs/titan-v.md)**

[Open the NLLB notebook in Colab](https://colab.research.google.com/github/bhaashik/BhaashikMTFinetuning/blob/main/notebooks/01_nllb_research.ipynb) · [OneNMT checks/repairs notebook](notebooks/02_onenmt_preflight_repairs.ipynb)

Clone the repository first. For GPU training, install the platform-appropriate CUDA PyTorch before the project extras; see the platform guides. Tests run on CPU without downloading research weights.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,repairs,dev]'
bhaashik audit --train data/examples/train.jsonl --dev data/examples/dev.jsonl --test data/examples/test.jsonl
pytest
```

Use [the notebook](notebooks/01_nllb_research.ipynb) on Colab or Kaggle for GPU training. Run commands from the repository root. Set Internet on in Kaggle; accept any upstream model access terms yourself. No token is required for public ungated models. Never put credentials in notebooks. Python 3.10–3.13 is the intended toolchain; CI covers 3.11 and 3.13. Python 3.13 selects SentencePiece 0.2.1 wheels; older supported Python versions retain 0.2.0.

```bash
# Replace the tiny illustrative data before research training.
bhaashik train --config configs/nllb-bho-hin.json
bhaashik translate --model runs/bho-hin --src bho_Deva --tgt hin_Deva \
  --input data/examples/test.jsonl --output runs/bho-hin/predictions.jsonl
bhaashik evaluate --input runs/bho-hin/predictions.jsonl --output runs/bho-hin/evaluation.json
bhaashik errors --input runs/bho-hin/predictions.jsonl --output runs/bho-hin/annotations.csv
```

If an older Colab notebook tries to compile SentencePiece on Python 3.13, refresh the checkout and use the updated setup cell. Do not use a catch-all `sudo apt-get` fallback. See [Python 3.13 troubleshooting](docs/colab.md#python-313-and-the-ver-02-installation-failure).

## Model strategy

| Model | Bhojpuri | Maithili | Magahi | Role |
|---|---|---|---|---|
| `facebook/nllb-200-distilled-600M` | `bho_Deva` | `mai_Deva` | `mag_Deva` | Implemented native HF baseline + LoRA/full finetuning |
| IndicTrans2 distilled 200M / Indic–Indic 320M | Not listed | `mai_Deva` | Not listed | Regional comparator; use upstream toolkit, not this NLLB runner |
| BhashaVerse / OneNMT | Not in inspected language map | Listed | Not in inspected language map | HF vocabulary/behavior preflight; training backend pending |
| Extended NLLB | Already supported | Already supported | Already supported | New language tags for genuinely unsupported varieties |

Coverage: [NLLB inventory](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/nllb/tokenization_nllb.py), [IndicTrans2 upstream](https://github.com/AI4Bharat/IndicTrans2). These are established research baselines, not an exhaustive ranking of current models. Benchmark before making SOTA claims. NLLB weights are **CC-BY-NC 4.0** ([license](https://github.com/facebookresearch/fairseq/blob/nllb/LICENSE.model.md)); public weights do not imply unrestricted open-source licensing. The toolkit is MIT; weights and datasets keep their own terms.

## OneNMT compatibility before adaptation

Run the [preflight module](docs/preflight.md) before changing BhashaVerse vocabulary or adding new-language/new-domain training data:

```bash
pip install -e '.[preflight]'
bhaashik preflight --assets data/local/onenmt-original \
  --corpus data/preflight/probes.jsonl --report runs/preflight/original.json
# Compare a separately prepared candidate tokenizer/dictionary/checkpoint:
bhaashik preflight --assets data/local/onenmt-original \
  --candidate-assets data/local/onenmt-candidate \
  --corpus data/local/probes.jsonl --report runs/preflight/candidate.json --strict
```

It checks preserved IDs, segmentation/coverage, emitted vs inactive token proposals, model row counts, and direction/context handling. `bhaashik hf-regression` optionally runs local HF MBart checkpoints against retained-language and adaptation probes, with metric-drop gates, output triage and independent reference replay. No architecture conversion is performed. Passing structural checks does not establish translation quality, and original assets remain unchanged. See the guide for append-plan and GPU regression commands. The separate [candidate repair commands](docs/repairs.md) can fix generation IDs, restore trusted dictionary IDs, append entries, extend/retrain/replace SPM, and initialize model rows. Each writes a new candidate with a change manifest; OneNMT training remains a separate step.

## Research workflow

1. Collect licensed bitext with stable IDs, provenance, domain and document IDs. Review language identity with native speakers. Do not treat Hindi-normalized text as gold for another language.
2. Freeze document-disjoint train/dev/test splits; run `audit` before training. The runner also checks train/dev leakage. Keep test data out of tuning and synthetic data generation.
3. Generate untouched NLLB predictions, then compare matched-budget LoRA and full finetuning. Use seeds 13, 42, 87; report each run and mean/dispersion.
4. Evaluate both directions separately for each language and domain. Report sacreBLEU, chrF++ and signatures, paired bootstrap deltas, sample counts, runtime, data hashes and checkpoint revisions.
5. Blind and double-annotate outputs using the exported sheet. Report agreement before adjudication; audit omissions, hallucinations, Hindi drift, agreement, names, numerals, honorifics and dialectal variation.

[Research protocol](docs/research.md) · [Data contract](docs/data.md) · [Language extension](docs/extension.md) · [OneNMT preflight](docs/preflight.md) · [Candidate repairs](docs/repairs.md) · [Limitations](docs/limitations.md)

Build and check the static portal locally:

```bash
# In the activated environment: pip install -e '.[dev]'
python scripts/build_site.py
python -m http.server 8000 --directory build/site
# Optional browser checks: pip install -e '.[web]'; playwright install chromium
# In a second terminal: python scripts/check_site.py
```

The `docs/` directory contains the sources for a static research portal. GitHub Actions tests the toolkit and deploys it to Pages after successful tests on `main`. The portal accepts real evaluation JSON locally and contains no fabricated scores. Pages must use GitHub Actions as its source.

## Scope

Implemented: NLLB training/inference, LoRA merged HF export, full finetuning, language-token initialization, data auditing, corpus metrics, paired bootstrap, heuristic triage, human annotation export, read-only OneNMT vocabulary/HF behavior preflight, and separate candidate repair operations. OneNMT/IndicTrans2 training, automatic corpus acquisition, backtranslation, COMET and annotation agreement computation are documented follow-up experiments, not implemented features. No model weights or research results ship with this repository.
