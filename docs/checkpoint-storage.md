# Disk-safe model checkpoint storage (Kaggle / Colab)

## Why training filled Kaggle storage

Hugging Face `Trainer` used to write intermediate epoch checkpoints under
the final model directory in `/kaggle/working`, with two checkpoints
retained. **Full finetuning** saves a large model and, by default,
optimizer/scheduler/RNG state, which may occupy several times the model
weight size. The exported selected model and optional ZIP also use space.
Dataset size does not determine checkpoint size; the model/optimizer does.

Current Kaggle documentation specifies **up to 20 GB of automatically
saved output in `/kaggle/working`**. Kaggle staff describe roughly
**60 GiB of additional, nonpersistent scratch space** under
`/kaggle/tmp`, but free capacity varies by runtime. Never assume an
exact free-space figure. Check it inside the actual notebook:

```python
import shutil
from pathlib import Path
for folder in ("/kaggle/working", "/kaggle/tmp"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(folder)
    print(folder, "free:", round(free / 2**30, 2), "GiB",
          "total:", round(total / 2**30, 2), "GiB")
```

Kaggle source: https://www.kaggle.com/docs/notebooks (20 GB
auto-saved output and additional transient scratch).
Kaggle staff discussion:
https://www.kaggle.com/product-feedback/372506
(approximately 60 GiB of scratch; no persistence guarantee).

## Recommended initial full-finetuning options

In the **first EXPERIMENT settings cell**, no other cells need editing:

```python
"stage": "low_baseline",
"allow_training": True,
"finetune_mode": "full",
"checkpoint_storage": "auto",
"checkpoint_limit": 1,
"checkpoint_model_only": True,
"checkpoint_min_free_gib": 5,
"checkpoint_cleanup_on_success": True,
"checkpoint_backup": "none",
```

This is a space-efficient experiment:
- Trainer saves to `/kaggle/tmp/bhaashik-checkpoints/` on Kaggle
  instead of the 20 GB working output. Colab uses
  `/content/bhaashik-checkpoints/`.
- `checkpoint_limit=1` retains the minimum permitted history;
  because best-checkpoint selection is enabled, **two checkpoint
  directories may temporarily remain** when the best is not the latest.
- `checkpoint_model_only=True` saves weights and Trainer state without
  the optimizer/scheduler/RNG payload. Best-model selection still uses
  per-epoch dev loss. **Such a checkpoint cannot resume training
  exactly after a session restart.**
- After a successful export of the selected model to the stage
  `model/` directory, scratch checkpoints are cleaned automatically.
  On failure, existing scratch files remain until you intentionally
  clean them, or the temporary Kaggle session ends.
- `checkpoint_min_free_gib` provides an *initial* free-space check.
  It does not guarantee that a later save fits: checkpoint size and
  free space change over time.

**Important:** Before creating an uncompressed stage ZIP, the library now
checks that the destination has room for the whole ZIP plus a 256 MiB reserve.
If not, it raises an explicit error and leaves the stage files intact rather
than filling the disk. This is a conservative check, not a guarantee of
capacity under concurrent disk usage.

**Important:** Stage-result ZIPs and exported final model still consume
`/kaggle/working` output. If that filesystem is nearly full, even
external checkpoints cannot make the final export fit. Keep the saved
archive outside the working output when necessary, or download/copy the
model and clean obsolete run stages. Do not delete data from active training
without first verifying its backup.

## Option 1: Kaggle → Google Drive automatically, via rclone

Kaggle does **not** provide Colab's interactive
`google.colab.drive.mount`. Use `rclone` with a previously authorized
Google Drive remote. Since you already configured rclone locally on
Ubuntu, you can transfer a dedicated `rclone.conf` through **Kaggle Secrets**
instead of signing in during every runtime.

1. On a trusted Ubuntu computer, create or use a Google Drive rclone
   remote. Its config is typically in
   `~/.config/rclone/rclone.conf`. Check `rclone listremotes`.
   For example a remote named `drive:`.
2. In Kaggle, open Notebook **Add-ons → Secrets** (the menu layout
   may change). Add a Secret such as `RCLONE_CONF` whose *value* is the
   complete rclone configuration. Enable that Secret's access for
   the notebook. **Treat this as a credential**—never print it, paste
   it into the notebook, or commit it to GitHub.
3. Install `rclone` in the runtime **before the training cell**,
   for example in a separate one-time setup cell:
   `!command -v rclone || (curl -fsSL https://rclone.org/install.sh | sudo bash)`
   Install software only from a source you trust. The current notebook's
   regular package installer does not install rclone.
4. Change **only** the first settings cell:

```python
"checkpoint_backup": "rclone",
"checkpoint_rclone_secret": "RCLONE_CONF",
"checkpoint_rclone_remote": "drive:BhaashikMT/checkpoints",
"checkpoint_storage": "auto",
"checkpoint_model_only": False,
```

`checkpoint_model_only=False` is **recommended if you need to resume**
the training after a Kaggle interruption: Drive backup now contains
optimizer/scheduler state, but uploads can be substantially larger. For
the smallest research backup without exact resume, keep
`checkpoint_model_only=True`.

The code checks for `rclone` and the named remote **before training**.
Whenever Trainer finishes writing a checkpoint, it runs `rclone copy`,
then `rclone check --one-way`. This verifies the backup before reporting
success. It **never deletes the local checkpoint during the upload**;
the local retention policy remains controlled by Trainer.
A failed upload raises an error rather than silently claiming the
checkpoint was backed up. The Drive destination includes a unique
run/stage folder and checkpoint step to avoid collisions.

Remember: **the full checkpoint is written to scratch first**, then
uploaded. Drive does not eliminate local save-space requirements.
External upload speed, quota, network and credentials can also fail.
When `checkpoint_model_only=False`, the very first checkpoint can be
large; check scratch free space before training.

If you use an existing rclone remote with a different name from `drive`,
change the prefix in `checkpoint_rclone_remote` accordingly.

## Option 2: Colab mounted Google Drive

In Colab, mount Drive in an earlier cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then configure:

```python
"checkpoint_backup": "mounted_directory",
"checkpoint_backup_directory": "/content/drive/MyDrive/BhaashikMT/checkpoints",
"checkpoint_storage": "auto",
"checkpoint_model_only": False,
```

Create the backup directory before training. The callback copies a
completed local checkpoint to Drive and checks each copied file's SHA-256.
Colab Drive mounts can be slow or unstable for very large models, so
the local scratch-first + verified-copy design is preferred to directly
writing Trainer checkpoints into a mounted Drive.

## Option 3: explicit alternative filesystem

If your environment has another persistent disk with sufficient capacity,
choose:

```python
"checkpoint_storage": "custom",
"checkpoint_custom_root": "/path/to/large/storage",
```

Do **not** point this at `/kaggle/input` (read-only). A custom path
must already be suitable for writing large model files. Automatic scratch
cleanup applies only to `auto` or `scratch`, never arbitrary custom
locations.

## Resume and reliability

| Option | Space | Resume after interruption? |
|---|---|---|
| Scratch + model-only + no Drive | Smallest; ephemeral | **No** exact resume |
| Scratch + model-only + Drive | Small and backed up | Weights are preserved; not optimizer/scheduler state |
| Scratch + full checkpoint + Drive | Largest local save and upload | **Yes**, after restoring full checkpoint, pointing `resume_from_checkpoint` at it, and retaining the same training configuration |
| Exported final model + stage ZIP | Space under working | Can run inference/retrain, but not exact optimizer-state continuation |

The repository does **not** yet include automatic restoration of a
specific Google Drive checkpoint. To resume exactly, restore the verified
full checkpoint in a new runtime and set the appropriate trainer-only
`resume_from_checkpoint` path; do not pretend an exported model-only
checkpoint can resume optimizer state.

**For your immediate NLLB full finetuning:** scratch + model-only
checkpoints and one retained history entry is the lowest-friction fix.
Choose Drive backup if resumption across Kaggle sessions matters.
