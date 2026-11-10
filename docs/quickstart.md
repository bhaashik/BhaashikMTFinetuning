# Quickstart: from checkout to research artifacts

Choose [Colab](colab.md), [Kaggle](kaggle.md), or a [Titan V desktop](titan-v.md) for platform-specific installation and storage. This guide runs all commands from the repository root. Python 3.10–3.13 is the intended range; local validation covers Python 3.12 / CPU PyTorch 2.5.1 and Python 3.13 / CPU PyTorch 2.6.0; CI covers Python 3.11 and 3.13. GPU workflows still require a real hardware smoke test.

## 1. Install and verify mechanics

Clone the repository, activate your platform's Python environment, and install its CUDA-enabled PyTorch as described in the platform guide. Then:

```bash
python -m pip install -e '.[train,repairs,dev]'
python -m pytest -q
bhaashik audit --train data/examples/train.jsonl \
  --dev data/examples/dev.jsonl --test data/examples/test.jsonl
```

The tests exercise real training, generation and HF save/reload using tiny random local models. They do not download large checkpoints, benchmark translation quality, or establish that your GPU has enough memory. `.[preflight]` alone suffices for read-only vocabulary checks; `.[repairs]` adds SPM mutation support. HF regression/embedding resizing need `.[train]`.

## 2. Prepare real data

Use three UTF-8 JSONL files under ignored `data/local/bho-hin/`: `train.jsonl`, `dev.jsonl`, `test.jsonl`. Each line needs nonempty string `id`, `source` and `target`. Include document/domain/provenance/license metadata for research. See the [data contract](data.md).

```json
{"id":"corpus-doc17-003","source":"...","target":"...","document_id":"doc17","domain":"news","provenance":"corpus@revision","license":"license-URL"}
```

Use distinct documents and IDs across splits. Run:

```bash
bhaashik audit --train data/local/bho-hin/train.jsonl \
  --dev data/local/bho-hin/dev.jsonl --test data/local/bho-hin/test.jsonl
```

Keep test data outside hyperparameter selection, synthetic-data generation and tokenizer training. The files in `data/examples/` are fixtures for a mechanics demonstration, not native-speaker-validated research gold.

## 3. Make a run configuration

```bash
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('configs/nllb-bho-hin.json').read_text())
config.update(train_file='data/local/bho-hin/train.jsonl',
              dev_file='data/local/bho-hin/dev.jsonl',
              output_dir='runs/bho-hin-lora-s42', seed=42)
# Pin config['revision'] to an immutable Hub commit for final paper runs.
Path('configs/local-bho-hin.json').write_text(json.dumps(config, indent=2))
PY
```

For Maithili/Magahi, start from their checked-in configs and matching data. NLLB tags are `bho_Deva`, `mai_Deva`, `mag_Deva`, `hin_Deva` and `eng_Latn`. For reverse translation, swap the data's source/target texts and config tags. One direction per run. Use fresh output directories for each condition/seed.

Default GPU settings: LoRA, FP16, batch 1, accumulation 16, max length 128, gradient checkpointing. They are a starting budget, not a fit guarantee. Use FP32 for a small CPU experiment, not for a practical full 600M-model training run on a slow CPU.

## 4. Generate the untouched baseline and train

```bash
bhaashik translate --model facebook/nllb-200-distilled-600M \
  --src bho_Deva --tgt hin_Deva --input data/local/bho-hin/test.jsonl \
  --output runs/bho-hin-baseline/predictions.jsonl \
  --batch-size 1 --max-length 128 --max-new-tokens 128 --beams 4
bhaashik train --config configs/local-bho-hin.json
bhaashik translate --model runs/bho-hin-lora-s42 \
  --src bho_Deva --tgt hin_Deva --input data/local/bho-hin/test.jsonl \
  --output runs/bho-hin-lora-s42/predictions.jsonl \
  --batch-size 1 --max-length 128 --max-new-tokens 128 --beams 4
```

Add `--revision IMMUTABLE_HUB_COMMIT` to baseline generation and put the same revision in training config for final comparisons. Public ungated model downloads need Internet access, not a token pasted into code. Respect upstream licenses/access terms. If your network blocks the Hub/CDN, use supported network settings or transfer complete model assets; do not disable TLS verification.

Training exports a merged/full HF model and tokenizer in the run root. LoRA consumers do not need a PEFT wrapper. `run_manifest.json` records data hashes, configuration, package versions, device, trainable parameters and dev loss. Checkpoints under `checkpoints/` support resuming; preserve those as well as the merged export if a run may be interrupted.

## 5. Evaluate and export human review

```bash
bhaashik evaluate --input runs/bho-hin-lora-s42/predictions.jsonl \
  --baseline runs/bho-hin-baseline/predictions.jsonl \
  --output runs/bho-hin-lora-s42/evaluation.json --samples 1000 --seed 42
bhaashik errors --input runs/bho-hin-lora-s42/predictions.jsonl \
  --output runs/bho-hin-lora-s42/annotations.csv
```

Candidate/baseline IDs, source texts, references and metadata must align. Report chrF++, BLEU signatures, per-domain results, paired intervals and training-seed variation separately. Use native-speaker review for Hindi drift, omissions, entities and register. A tiny fixture-sized score is not a paper result. See the [research protocol](research.md).

## 6. Preserve, resume and repeat

Store model/tokenizer, training config, checkpoints, manifests, baseline/candidate predictions, evaluation JSON and annotation files in persistent storage. Keep licensed data/provenance separately. The repository ignores local data and runs: Git push does not preserve your training outputs. The notebook creates its ZIP outside the archived directory to avoid archiving itself.

To resume an interrupted run, set `resume_from_checkpoint` in its config to the actual `runs/.../checkpoints/checkpoint-N` directory and keep the same data/hyperparameters. Use the same model source (the original base checkpoint), not a partial output folder as the base. A nonempty output directory is otherwise refused. For a fresh experiment, use a fresh output directory instead.

## OneNMT comparison branch

The NLLB `train`, `translate` and `extend` commands do not accept OneNMT/IndicTrans2. OneNMT currently has local HF [preflight](preflight.md), [behavior regression](preflight.md#4-check-hf-behavior-before-and-after-adaptation) and [separate repairs](repairs.md). These can be used on every platform, subject to RAM/disk/GPU availability. Do not put OneNMT through AutoTokenizer or the NLLB runner. No architecture conversion is performed.

## Build the portal

```bash
python scripts/build_site.py
python -m http.server 8000 --directory build/site
```

The generated site contains rendered guides, experiment config downloads and local evaluation-report import. Pushing to `main` triggers the repository's test and Pages workflow when GitHub Actions/Pages permissions allow it. A successful local build does not establish deployment success.
