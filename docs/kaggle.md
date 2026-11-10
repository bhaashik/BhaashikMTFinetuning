# Run on Kaggle Notebooks

## Import and enable the runtime

1. Open [Kaggle Code](https://www.kaggle.com/code) and create a notebook.
2. Import `notebooks/01_nllb_research.ipynb` from the GitHub repository, or download that file and upload it through the notebook import dialog.
3. Enable an available GPU accelerator and Internet in notebook settings. Account verification, GPU quota and platform policy may be prerequisites.
4. Run the setup cell before importing Torch/Transformers. It clones into `/kaggle/working/BhaashikMTFinetuning` and preserves the provided PyTorch/CUDA runtime.

If packages were already imported before installation changed them, restart the kernel and rerun setup. Check actual accelerator execution:

```python
import torch
assert torch.cuda.is_available(), 'Enable a GPU accelerator'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
assert torch.isfinite(a @ a).all().item()
del a
```

A notebook assigned multiple GPUs still uses one GPU with the documented default workflow. Do not assume two 16 GB devices behave like one 32 GB device. Distributed training is outside the supplied notebook workflow. The current Kaggle image has not been validated here; test the assigned environment before a full experiment.

## Supply immutable input data

Upload your licensed train/dev/test JSONL splits as a private/public Kaggle dataset as appropriate, then attach it with Add Input. `/kaggle/input/` is read-only; copy into the checkout's ignored local-data directory:

```python
from pathlib import Path
import shutil
src = Path('/kaggle/input/YOUR-DATASET-SLUG/bho-hin')
dst = ROOT / 'data/local/bho-hin'
assert src.is_dir(), 'Inspect the attached dataset folder layout'
assert not dst.exists(), 'Inspect existing data or choose a new folder'
shutil.copytree(src, dst)
```

Set notebook `USE_EXAMPLE_DATA=False`, choose the matching language direction and a fresh run name/seed. Run the split audit. Retain provenance, corpus license and document IDs. See the [quickstart](quickstart.md) and [data contract](data.md).

If Internet is disabled, cloning, pip installation and Hub downloads will fail unless you have already attached complete compatible code/package/model artifacts. Enable supported Internet access or prepare an offline dataset deliberately; the project does not automatically assemble an offline wheelhouse.

## Training and storage

Begin with LoRA, batch 1, length 128, FP16 and gradient checkpointing. Free GPU hours, device choice and session length are not guaranteed. Stop after a short pilot if it does not fit or produces invalid outputs. Full finetuning and OneNMT CPU embedding repair may exceed available GPU/host memory.

Write results under `/kaggle/working/`, not `/kaggle/input/`. The final notebook archive is created outside the repository's `runs/` tree, under `/kaggle/working/`, so it does not include itself. It contains all current runs; keep disk/output-size limits in mind.

Use Save Version / Save & Run All as appropriate for your workflow, and confirm that desired artifacts appear in the saved notebook's Output files. Download important artifacts or publish a separately reviewed model/data artifact when appropriate; an interactive session's live files alone are not durable storage. Do not publish licensed/private data accidentally when sharing notebook outputs. Preserve the training config and checkpoint state for resumability, not just the merged export.

To resume, attach the prior checkpoint/output dataset, copy the needed checkpoint/config into a writable location, then set `resume_from_checkpoint` to that actual path while retaining the same model source, data and hyperparameters. See [resume guidance](quickstart.md#6-preserve-resume-and-repeat).

## OneNMT branch and diagnostics

Install `.[train,repairs]`, supply immutable original assets and reviewed probes, and follow the [preflight](preflight.md) / [separate repair](repairs.md) guides. Snapshot operations require a new output directory per stage and can copy gigabytes of weights. They do not implement OneNMT training or automatically make new tokens useful.

Use the [100-step pilot](titan-v.md#measure-a-pilot-instead-of-guessing-training-time) with platform-appropriate paths to measure runtime. This release has not been benchmarked on a hosted Kaggle GPU; CPU integration tests demonstrate software mechanics, not Kaggle hardware fit or translation quality.

Setup supports Python 3.10–3.13. The shared notebook installer requires wheels for SentencePiece/Tokenizers and shows original pip failures. Python 3.13 selects SentencePiece 0.2.1; older supported versions keep 0.2.0. It does not attempt sudo/toolchain installation. Refresh an old checkout deliberately and restart the kernel if model libraries were already imported.
