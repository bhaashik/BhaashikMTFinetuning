# Run on Google Colab

## Open and select a GPU

[Open the NLLB research notebook](https://colab.research.google.com/github/bhaashik/BhaashikMTFinetuning/blob/main/notebooks/01_nllb_research.ipynb). Select a GPU hardware accelerator in Runtime settings before running the setup cell. Free GPU allocation and session duration vary; access is not guaranteed. Check the assigned device, not the notebook's metadata.

The notebook clones into `/content/BhaashikMTFinetuning` and installs the project. It preserves the runtime's CUDA PyTorch rather than automatically replacing it. Dependencies are pinned for the HF training stack; the exact Colab runtime has not been validated here. Run setup before importing Torch/Transformers. If installation changes already-imported packages, restart the session and rerun setup before proceeding.

Check actual GPU execution:

```python
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
a = torch.randn(128, 128, device='cuda', dtype=torch.float16)
assert torch.isfinite(a @ a).all().item()
print('FP16 GPU matmul succeeded')
del a
```

If `no kernel image` or binary/torchvision import errors occur, use a fresh runtime and ensure Torch/torchvision/torchaudio versions are compatible. Do not layer repeated unverified package downgrades into the same session. The [Titan V guide](titan-v.md) gives an isolated pinned reference stack for local systems; it is not necessary to replace a working Colab GPU stack.

## Choose data and experiment

Run notebook cells in order. `USE_EXAMPLE_DATA=True` demonstrates mechanics using two examples per split. For research, set it to `False`, select `SRC`, `TGT`, `SEED` and `MODE`, and provide matching licensed data under `data/local/{PAIR}/`.

To use Drive data without overwriting files:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import shutil
src = Path('/content/drive/MyDrive/bhaashik-data/bho-hin')
dst = ROOT / 'data/local/bho-hin'
assert not dst.exists(), 'Choose a new data folder or inspect the existing one'
shutil.copytree(src, dst)
```

The source folder needs `train.jsonl`, `dev.jsonl`, `test.jsonl` following the [data contract](data.md). Keep test material out of tokenizer training and adaptation. Audit splits before downloading/training the large model. Set an immutable Hub revision for paper runs and record the Git commit with `git rev-parse HEAD`.

Start with NLLB-600M LoRA, batch 1, length 128, FP16 and gradient checkpointing. BF16 is only appropriate on supported GPUs; a T4/Titan V does not support it. Full finetuning may exceed the assigned GPU's memory. Shorten sequences or choose a larger GPU when necessary; increasing gradient accumulation does not make a batch of one sentence use less activation memory.

## Preserve outputs before disconnecting

Colab's `/content` disk is ephemeral. The final notebook cell creates an artifact archive under `/content/`, outside `runs/`. Copy it to Drive or download it:

```python
from pathlib import Path
import shutil
archive_path = Path(archive)
persistent = Path('/content/drive/MyDrive/bhaashik-artifacts')
persistent.mkdir(parents=True, exist_ok=True)
destination = persistent / archive_path.name
assert not destination.exists(), 'Use a new artifact name rather than overwriting an earlier run'
shutil.copy2(archive_path, destination)
```

The archive includes all current `runs/` entries, so avoid accumulating unrelated large runs in one session. Drive/archive quotas and transfer time matter. For interruption protection, copy completed checkpoint directories/configs to persistent storage during training as well; the final archive cannot preserve an interrupted run that never reaches its export cell. Restore a checkpoint and set `resume_from_checkpoint` as described in the [quickstart](quickstart.md#6-preserve-resume-and-repeat).

## OneNMT checks and repairs

Install `.[train,repairs]` in the checkout. Supply a complete immutable local asset folder and separate candidate destinations. Run the [preflight](preflight.md) and [repair](repairs.md) commands through notebook shell cells or `subprocess.run([...], check=True)`. Check free disk first: candidate snapshots copy model weights, and CPU save/reload verification can need substantial host RAM. OneNMT training is not implemented in this project.

## Timing and limitations

No hosted Colab GPU timing or model quality result has been measured for this release. Installation/model downloads and notebook quotas affect wall time. Measure a short pilot on the assigned device, then extrapolate optimizer steps; do not assume the same runtime across free sessions. See the [Titan V guide](titan-v.md#measure-a-pilot-instead-of-guessing-training-time) for a 100-step config example that also works here.
