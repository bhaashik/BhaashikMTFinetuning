# Everyday Kaggle and Colab workflow (BhaashikMTFinetuning)

This guide is for researchers using notebooks without a terminal. You can
copy Python snippets into notebook cells. The three model notebooks have the
**same editable experiment settings cell near the top**. Edit this one cell,
then run the notebook **in order**. A fresh session/reimport is recommended
for each independent experiment.

## 1. Most useful paths

| Purpose | Kaggle path | Google Colab path |
|---|---|---|
| Temporary working directory | \`/kaggle/working/\` | \`/content/\` |
| Cloned repository | \`/kaggle/working/BhaashikMTFinetuning/\` | \`/content/BhaashikMTFinetuning/\` |
| Kaggle datasets mounted as read-only inputs | \`/kaggle/input/DATASET_SLUG/\` | Not available; use a Colab upload or Drive |
| User's mounted Google Drive | Not mounted by default | \`/content/drive/MyDrive/\` |
| Intermediate parallel JSONL | \`/kaggle/working/BhaashikMTFinetuning/data/local/PAIR/DATASET_ID/parallel.jsonl\` | \`/content/BhaashikMTFinetuning/data/local/PAIR/DATASET_ID/parallel.jsonl\` |
| Deterministic splits | \`/kaggle/working/BhaashikMTFinetuning/data/local/PAIR/DATASET_ID/splits-sSEED-800-100-100/\` | \`/content/BhaashikMTFinetuning/data/local/PAIR/DATASET_ID/splits-sSEED-800-100-100/\` |
| Exported checkpoints and model | \`/kaggle/working/BhaashikMTFinetuning/runs/RUN_NAME/\` | \`/content/BhaashikMTFinetuning/runs/RUN_NAME/\` |
| Experiment settings/manifest | \`/kaggle/working/BhaashikMTFinetuning/configs/RUN_NAME.json\` | \`/content/BhaashikMTFinetuning/configs/RUN_NAME.json\` |
| Portable exported ZIP | \`/kaggle/working/RUN_NAME-artifacts.zip\` | \`/content/RUN_NAME-artifacts.zip\` |

Examples: \`PAIR=bho-hin\` for Bhojpuri–Hindi; \`DATASET_ID=corpus-v1\`.
Your actual **dataset slug** is shown by Kaggle when you attach a dataset, and
may differ from its title. Use the file browser or the listing below rather
than guessing it.

## 2. Kaggle notebook: start and locate data

1. Open a new Kaggle Notebook, or import the updated
   \`notebooks/01_nllb_research.ipynb\` from GitHub.
2. In the notebook right-hand **Settings/Session options** area choose
   **Accelerator → GPU** (for NLLB, a T4 GPU is a common starting option).
   The exact panel/wording may change. Turn **Internet** on while cloning
   GitHub and downloading Hugging Face models; note that the toggle may
   depend on account verification or platform policy.
3. On the right-hand **Input** (sometimes **Add Input** or **Add Data**) panel,
   attach your uploaded Kaggle dataset containing two UTF-8 sentence files,
   for example \`bho.txt\` and \`hin.txt\`. Kaggle mounts datasets under
   \`/kaggle/input/\`; **do not write output there**.
4. Find the actual paths from a Python cell:

\`\`\`python
from pathlib import Path
for path in sorted(Path("/kaggle/input").rglob("*")):
    if path.is_file():
        print(path)
\`\`\`

5. In the **first experiment settings cell** choose the paths displayed.
   Typical case:

\`\`\`python
"data_mode": "aligned_text",
"source_lang": "bho_Deva",
"target_lang": "hin_Deva",
"source_text": "/kaggle/input/my-parallel-corpus/bho.txt",
"target_text": "/kaggle/input/my-parallel-corpus/hin.txt",
"split_seed": 42,
"validate_data": False,
"pretrain_baseline": False,
"posttrain_test": False,
"publication_report": False,
"allow_training": True,
\`\`\`

6. Run all cells top to bottom (**Run All**, or the triangular **Run** icon
   beside each cell). The first setup cell clones the repo when absent and
   installs dependencies without a terminal. A new session normally has no
   previous \`/kaggle/working/BhaashikMTFinetuning/\` folder.
   \`refresh_repo=True\` runs \`git pull --ff-only\` when a checkout exists.
7. If you change the input data, use a new \`dataset_id\`. If you change
   model/training settings, use a new \`run_label\`. For valid model comparisons,
   **keep \`dataset_id\`, \`split_seed\` and split ratios unchanged**.
8. The dataset converter, splitter, setup, and Trainer are sequential steps.
   When \`validate_data=False\`, optional expensive diagnostics are skipped,
   but basic format/group checks and the Trainer's own train/dev audit remain.

### Saving Kaggle outputs correctly

\`/kaggle/working/\` is runtime storage; do **not** assume files will be
available in a future fresh session. The model and ZIP are stored there.
To retain them, use the notebook's **Save Version** workflow and verify
that output files are included, or download the ZIP from the file/output
browser. Some interfaces offer **Quick Save** and **Save & Run All**:
Quick Save should not launch a second expensive training run, whereas
Save & Run All can execute the whole notebook again. Check the selected
options before saving. Large artifacts may exceed version/storage limits.
You can also make the output into a separate Kaggle Dataset for reuse.

Inspect outputs:

\`\`\`python
from pathlib import Path
for p in sorted(Path("/kaggle/working").glob("*artifacts.zip")):
    print(p, round(p.stat().st_size / 2**20, 1), "MiB")
\`\`\`

If your checkpoint requires exact resumption, use
\`include_checkpoints=True\` before the final archiving cell; it makes a
larger ZIP. Without that option the exported model is suitable for
inference and further tuning, but optimizer/scheduler history is not
preserved for *exact* training resumption.

## 3. Google Colab: start, mount Drive and locate data

1. Open or import the notebook in Colab. Choose **Runtime → Change
   runtime type → Hardware accelerator → T4 GPU** (or another available
   GPU). GPU availability and menu wording vary by plan/runtime.
2. If your parallel text files are on Google Drive, first insert a
   separate cell **before the notebook's setup/data cells**:

\`\`\`python
from google.colab import drive
drive.mount("/content/drive")
\`\`\`

3. Follow the authorization prompt. The typical personal Drive path is
   \`/content/drive/MyDrive/\`, for example
   \`/content/drive/MyDrive/Bhaashik/data/bho.txt\`. Find the actual
   names with:

\`\`\`python
from pathlib import Path
folder = Path("/content/drive/MyDrive/Bhaashik/data")
for file in folder.glob("*"):
    print(file)
\`\`\`

4. Set the notebook's \`source_text\` and \`target_text\` to those absolute
   paths. For a single tab-separated file choose
   \`data_mode="tsv"\` and \`tsv_file=".../parallel.tsv"\`.
5. Run the setup and subsequent cells. The repo is cloned at
   \`/content/BhaashikMTFinetuning/\`, while intermediate files and
   saved models stay under \`/content/\` unless you copy them elsewhere.
6. After training, persist the final ZIP in Drive:

\`\`\`python
import shutil
from pathlib import Path
dest = Path("/content/drive/MyDrive/Bhaashik/archives")
dest.mkdir(parents=True, exist_ok=True)
for archive in Path("/content").glob("*artifacts.zip"):
    saved = dest / archive.name
    if saved.exists():
        raise FileExistsError(f"Archive already saved: {saved}")
    shutil.copy2(archive, saved)
    print("Saved:", saved)
\`\`\`

Google Drive mounted I/O can be slower and may time out; prefer working on
\`/content/\` for training and **copying outputs to Drive afterwards**.
Grant Drive access only in notebooks you trust.

## 4. What happens inside the repository?

For data mode \`"aligned_text"\`:
- Two aligned plain-text files become tagged \`parallel.jsonl\` with one
  UTF-8 source sentence and reference per row.
- The reproducible splitter creates **train/dev/test** partitions from the
  supplied seed/ratios. Identical normalized source sentences and optional
  document IDs stay together. Existing complete outputs are reused.
- The trainer checks train/dev data and uses **development loss each epoch**
  for checkpoint selection. It does **not** generate translations over
  the whole dev set or calculate BLEU/chrF++ at each epoch.
- The model is exported to \`runs/RUN_NAME/\`; a ZIP can preserve it.
- Optional final test inference and publication evaluation require explicitly
  enabling flags in the first settings cell.

## 5. Common troubleshooting

- **\`git: cannot change to ...\`**: the runtime is fresh or the repo was
  not cloned. Run the new setup cell; it clones automatically.
- **\`NameError: REVIEW_DIR\`/\`require_routes\`**: old notebooks assumed
  optional cells had already run. Reimport the updated notebook; the shared
  workflow initializes these objects in required cells.
- **\`ModuleNotFoundError\` after pip**: use a fresh GPU runtime, run setup
  before model imports, restart the kernel if you changed installed versions.
- **Model doesn't load/unsupported language**: use native language tokens
  only for confirmed supported directions; Braj/Marwari require a separate
  extension/controlled experiment.
- **Source and target lines differ or a blank appears**: correct the
  aligned text files; the converter never silently guesses.
- **Old output refuses overwrite**: check input hashes; use a new
  \`dataset_id\` for changed data or a new \`run_label\` for a new experiment.
- **CUDA not available**: select GPU in the notebook's session/runtime
  settings **before** running the setup cell.
- **Notebook version was saved but exported model disappeared**: inspect
  the notebook's saved Output files or your Drive archive; do not rely
  on temporary runtime disk being retained.

Related: [experiment settings](experiment-guide.md),
[data format](data.md), [multilingual conditioning](multilingual-adaptation.md).

Reference platform documentation:
- https://www.kaggle.com/page/GPU-tips-and-tricks
- https://research.google.com/colaboratory/faq.html
