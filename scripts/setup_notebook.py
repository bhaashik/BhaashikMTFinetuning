"""Wheel-only installation and fresh-process import/GPU checks for notebook setup."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys


def validate_python(current=None):
    current = tuple((current or sys.version_info)[:2])
    if not (3, 10) <= current < (3, 14):
        raise RuntimeError(
            f'Python {current[0]}.{current[1]} is outside the supported 3.10–3.13 range. '
            'Choose a compatible notebook runtime/kernel; installing compiler tools is not the fix.')
    return current


def verify_stack(require_cuda=False):
    import sentencepiece
    import torch
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer
    from peft import LoraConfig
    from datasets import Dataset
    print('Import smoke test passed: Transformers Trainer, PEFT, Datasets, SentencePiece', flush=True)
    for package in ('bhaashik-mt', 'torch', 'transformers', 'peft', 'datasets', 'sentencepiece', 'tokenizers'):
        print(f'{package}: {version(package)}', flush=True)
    print('CUDA wheel runtime:', torch.version.cuda, flush=True)
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('Installation/imports passed, but no CUDA GPU is available. Select a GPU accelerator and rerun setup.')
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
        tensor = torch.randn(128, 128, device='cuda', dtype=torch.float16)
        if not torch.isfinite(tensor @ tensor).all().item():
            raise RuntimeError('GPU FP16 matrix multiplication produced nonfinite values')
        torch.cuda.synchronize()
        del tensor
        print('FP16 GPU execution passed; free GiB:', round(torch.cuda.mem_get_info()[0]/2**30, 2), flush=True)


def setup(root, extras='train', require_cuda=False):
    current = validate_python()
    root = Path(root).resolve()
    if not (root/'pyproject.toml').is_file():
        raise RuntimeError('Run setup from the repository checkout')
    if extras not in ('train', 'train,repairs'):
        raise ValueError('Choose train or train,repairs extras')
    print('Interpreter:', sys.executable, f'Python {current[0]}.{current[1]}', flush=True)
    print('Installing wheel-backed native dependencies; original pip output follows.', flush=True)
    # Fail fast for a missing native wheel. Never assume that an arbitrary pip error
    # is a SentencePiece compiler problem, and never silently relax dependency pins.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=sentencepiece,tokenizers',
                    '-e', f'.[{extras}]'], cwd=root, check=True)
    verify_stack(require_cuda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--extras', choices=['train', 'train,repairs'], default='train')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    try:
        setup(args.root, args.extras, args.require_cuda)
    except subprocess.CalledProcessError as error:
        print(f'Installation failed with exit code {error.returncode}; inspect the pip error above. '
              'Fix that specific error before rerunning.', file=sys.stderr)
        return error.returncode or 1
    except (RuntimeError, ImportError, ValueError) as error:
        print(f'Setup stopped: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
