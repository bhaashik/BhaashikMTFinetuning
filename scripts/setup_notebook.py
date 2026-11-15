"""Kaggle/Colab setup with a lightweight CPU mode and PyTorch-only ML imports."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys


EXTRAS = ('base', 'preflight', 'train', 'train,repairs')


def validate_python(current=None):
    current = tuple((current or sys.version_info)[:2])
    if not (3, 10) <= current < (3, 14):
        raise RuntimeError(
            f'Python {current[0]}.{current[1]} is outside the supported 3.10-3.13 range. '
            'Choose a compatible notebook runtime/kernel; installing compiler tools is not the fix.')
    return current


def configure_backends():
    """Keep Transformers from importing preinstalled Kaggle TensorFlow/Flax.

    This must run *before* any Transformers import. It intentionally does not
    uninstall TensorFlow or modify its Protobuf runtime.
    """
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'


def verify_stack(require_cuda=False, extras='train'):
    import bhaashik
    if extras == 'base':
        import numpy
        import sacrebleu
        print('CPU data/audit stack passed: Bhaashik, NumPy, sacreBLEU', flush=True)
        for package in ('bhaashik-mt', 'numpy', 'sacrebleu'):
            print(f'{package}: {version(package)}', flush=True)
        return
    import sentencepiece
    if extras == 'preflight':
        import safetensors
        print('CPU preflight stack passed: Bhaashik, SentencePiece, Safetensors', flush=True)
        for package in ('bhaashik-mt', 'sentencepiece', 'safetensors'):
            print(f'{package}: {version(package)}', flush=True)
        return

    import torch
    from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer
    from peft import LoraConfig
    from datasets import Dataset
    print('Import smoke test passed: Transformers Trainer, PEFT, Datasets, SentencePiece', flush=True)
    for package in ('bhaashik-mt', 'torch', 'transformers', 'peft', 'datasets',
                    'sentencepiece', 'tokenizers'):
        print(f'{package}: {version(package)}', flush=True)
    print('CUDA wheel runtime:', torch.version.cuda, flush=True)
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('Installation/imports passed, but no CUDA GPU is available. '
                               'Select a GPU accelerator and rerun setup.')
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
        tensor = torch.randn(128, 128, device='cuda', dtype=torch.float16)
        if not torch.isfinite(tensor @ tensor).all().item():
            raise RuntimeError('GPU FP16 matrix multiplication produced nonfinite values')
        torch.cuda.synchronize()
        del tensor
        print('FP16 GPU execution passed; free GiB:',
              round(torch.cuda.mem_get_info()[0]/2**30, 2), flush=True)


def setup(root, extras='train', require_cuda=False):
    current = validate_python()
    root = Path(root).resolve()
    if not (root / 'pyproject.toml').is_file():
        raise RuntimeError('Run setup from the repository checkout')
    if extras not in EXTRAS:
        raise ValueError(f'Choose one of: {", ".join(EXTRAS)}')
    if require_cuda and extras not in ('train', 'train,repairs'):
        raise ValueError('--require-cuda only applies to the train extras')
    configure_backends()
    print('Interpreter:', sys.executable, f'Python {current[0]}.{current[1]}', flush=True)
    print('Install mode:', extras, '(TensorFlow/Flax disabled for this PyTorch project)', flush=True)
    print('Installing wheel-backed dependencies; original pip output follows.', flush=True)
    spec = '.' if extras == 'base' else f'.[{extras}]'
    # An old, optional TorchAO installed by Kaggle can make newer PEFT refuse
    # to construct LoRA adapters. This project does not use TorchAO; remove
    # only the incompatible preinstalled versions, and only for training.
    if extras in ('train', 'train,repairs'):
        try:
            torchao_version = version('torchao')
        except PackageNotFoundError:
            torchao_version = None
        if torchao_version:
            # Version prefix only (e.g. 0.10.0+cu130); no optional
            # version-parsing dependencies required in this installer.
            fields = torchao_version.split('.')
            try:
                incompatible = (int(fields[0]), int(fields[1])) < (0, 16)
            except (ValueError, IndexError):
                incompatible = False
            if incompatible:
                print(f'Removing optional incompatible torchao {torchao_version} before PEFT setup',
                      flush=True)
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
                               cwd=root, check=True)
    # Do not replace the runtime's Torch/CUDA installation proactively.
    # Do not constrain Protobuf here: Kaggle's TensorFlow and cloud SDKs may
    # require Protobuf 6+, and this project's PyTorch code does not need a pin.
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    '--only-binary=sentencepiece,tokenizers', '-e', spec],
                   cwd=root, check=True)
    # Editable installs add a .pth file to site-packages. This running process
    # started before pip wrote that file, so it may not yet be able to import
    # bhaashik. Verify in a new interpreter, which processes the new .pth file.
    check = [sys.executable, str(Path(__file__).resolve()), '--verify-only',
             '--extras', extras]
    if require_cuda:
        check.append('--require-cuda')
    subprocess.run(check, cwd=root, env=os.environ.copy(), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--extras', choices=EXTRAS, default='train',
                        help='base: CPU data/audit; preflight: CPU tokenizer checks; train: HF/PEFT; train,repairs: full extras')
    parser.add_argument('--require-cuda', action='store_true',
                        help='Fail unless GPU is available and FP16 matmul works (train modes only)')
    parser.add_argument('--verify-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.verify_only:
            validate_python()
            if args.require_cuda and args.extras not in ('train', 'train,repairs'):
                raise ValueError('--require-cuda only applies to training extras')
            configure_backends()
            verify_stack(require_cuda=args.require_cuda, extras=args.extras)
        else:
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
