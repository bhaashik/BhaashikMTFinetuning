# Quickstart

Bhaashik provides NLLB finetuning/evaluation and separate OneNMT HF compatibility/repair tools. It does not yet provide a OneNMT or IndicTrans2 training backend.

Choose your environment:

- **[Colab](docs/colab.md):** open the GPU notebook, supply data and preserve artifacts in Drive.
- **[Kaggle](docs/kaggle.md):** import the notebook, enable Internet/GPU, mount input data and save outputs.
- **[Titan V desktop](docs/titan-v.md):** create an isolated CUDA-enabled environment and start with LoRA.
- **[Step-by-step workflow](docs/quickstart.md):** install, smoke-test, prepare a run, evaluate and inspect errors.

[Open the NLLB notebook in Colab](https://colab.research.google.com/github/bhaashik/BhaashikMTFinetuning/blob/main/notebooks/01_nllb_research.ipynb)

```bash
git clone https://github.com/bhaashik/BhaashikMTFinetuning.git
cd BhaashikMTFinetuning
python3 -m venv .venv
source .venv/bin/activate
# Install GPU PyTorch first using the platform guide, then:
python -m pip install -e '.[train,repairs,dev]'
python -m pytest -q
bhaashik audit --train data/examples/train.jsonl \
  --dev data/examples/dev.jsonl --test data/examples/test.jsonl
```

Tests use tiny random local models; they download no research weights and require no GPU. The six illustrative parallel sentences are not a research dataset. Real training requires licensed, verified bitext and model download access. NLLB weights are CC-BY-NC 4.0.

For OneNMT, begin with [preflight](docs/preflight.md), then apply only justified [separate repairs](docs/repairs.md). Repairs create new candidates and do not establish learned language support. Preserve the original model/dictionary/tokenizer together.
