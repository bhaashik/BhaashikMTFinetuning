# Run on an NVIDIA Titan V desktop

A Titan V has 12 GB VRAM and Volta compute capability **7.0**. Start with NLLB-600M LoRA, batch 1, length 128 and gradient checkpointing. FP16 is supported; **BF16 and FlashAttention 2 are not suitable for this GPU**. Actual model fit must be measured. Full finetuning may exceed 12 GB.

## Create an isolated reference environment

Use a Linux desktop with an NVIDIA driver compatible with CUDA 11.8. Check `nvidia-smi` first. The CUDA version printed by `nvidia-smi` describes the driver's capability, not necessarily the PyTorch wheel's CUDA runtime. The wheel supplies runtime libraries; a separate system CUDA toolkit is usually unnecessary for this project's prebuilt packages.

Recommended host resources are 32 GB RAM and enough free SSD space for model downloads, optimizer checkpoints and candidate copies. Allow at least 30 GB for an initial NLLB workflow and more for OneNMT repair branches; every copied checkpoint can consume several additional GB. These are planning budgets, not measured peak requirements.

```bash
git clone https://github.com/bhaashik/BhaashikMTFinetuning.git
cd BhaashikMTFinetuning
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e '.[train,repairs,dev]'
```

The pinned CUDA 11.8 / PyTorch 2.5.1 combination is an intentional starting point for Volta. Do not assume a newest CUDA/PyTorch wheel still includes `sm_70`. CPU tests here use PyTorch 2.5.1; this exact GPU wheel/driver/Titan V combination has not been executed here. Verify it below. Torchvision/torchaudio are not needed in this isolated environment; do not install incompatible optional builds. No FlashAttention, quantization library or custom CUDA extension is required.

## Verify the actual GPU kernels

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Check the NVIDIA driver and CUDA-enabled PyTorch wheel'
print('Torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
print('Device:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('Compiled arch list:', torch.cuda.get_arch_list())
if torch.cuda.get_device_capability(0) == (7, 0):
    print('Volta detected: use FP16 or FP32, not BF16')
a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
b = a @ a
assert torch.isfinite(b).all().item()
torch.cuda.synchronize()
print('FP16 matmul passed; free GiB:', round(torch.cuda.mem_get_info()[0] / 2**30, 2))
PY
python -m pytest -q
```

An unavailable CUDA device or `no kernel image is available` is a setup blocker, not a reason to change model architecture. Recheck driver/wheel compatibility and compiled architecture support. Preserve TLS/checksum verification during installation.

## Configure and train

Follow the [quickstart](quickstart.md) with your own `data/local/` splits. The checked-in `configs/nllb-*-hin.json` files already use LoRA, FP16, batch 1, accumulation 16, length 128 and gradient checkpointing. The Bhojpuri config points at illustrative fixtures; update it before real experiments. Use separate run directories for each language, direction and seed.

Monitor memory in another terminal with `nvidia-smi -l 2`. Close unrelated GPU processes you own when appropriate. If a batch of one does not fit, shorten the configured sequence length, reduce inference beams/output length, or use a larger GPU. Gradient accumulation increases the effective batch but does not reduce per-sentence activation memory. FP16 overflow should be investigated; FP32 can be a diagnostic comparison but costs more memory. Do not switch to BF16 on Titan V.

## Measure a pilot instead of guessing training time

After supplying real train/dev data and pinning the model revision, create a separate pilot config:

```bash
python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('configs/local-bho-hin.json').read_text())
c['max_steps'] = 100
c['output_dir'] = 'runs/bho-hin-pilot-100'
Path('configs/local-pilot.json').write_text(json.dumps(c, indent=2))
PY
bhaashik train --config configs/local-pilot.json
```

Inspect `run_manifest.json` for `train.train_runtime`, training loss, evaluation loss and the selected config. The pilot must finish with finite losses and reloadable artifacts before scaling. `max_steps` counts optimizer steps, not microbatches; gradient accumulation is already accounted for. For one GPU, estimate roughly:

```text
steps ≈ epochs × ceil(training_pairs / (batch_size × gradient_accumulation_steps))
training_time ≈ pilot_train_runtime × full_run_steps / pilot_steps
```

This is a planning estimate; epoch boundaries, shorter/longer sequences, evaluation, saving, warmup and I/O change wall time. The tiny fixture corpus is unsuitable for estimating real-corpus speed. Record actual elapsed time and peak memory in final experiments. No Titan V training-time or translation-quality benchmark is supplied.

## Persist and resume

Local disk persists across Python sessions, but still back up important artifacts. Keep the model/tokenizer export, optimizer checkpoints, config, data provenance/hashes, predictions and evaluation together. Resume using `resume_from_checkpoint` as in the [quickstart](quickstart.md#6-preserve-resume-and-repeat). Do not overwrite an earlier run directory to start a different condition.

## OneNMT checks and repair on this desktop

Read-only [preflight](preflight.md) runs primarily on CPU and needs no GPU weights loaded. [HF behavior probes](preflight.md#4-check-hf-behavior-before-and-after-adaptation) can use `--device cuda --precision fp16 --batch-size 1`. Original and candidate models load sequentially. Whether the full OneNMT checkpoint fits 12 GB has not been measured here.

[Embedding resize repairs](repairs.md#initialize-model-rows-separately) run on CPU and perform save/reload verification. Budget host RAM for two model instances plus embedding copies; 32 GB is a sensible starting allocation, not a guarantee. Source/target dictionary consistency and HF loading checks may refuse a candidate. A config patch does not repair arbitrary corrupt weights. Initialized rows still need a compatible training workflow, which this project has not yet implemented for OneNMT.
