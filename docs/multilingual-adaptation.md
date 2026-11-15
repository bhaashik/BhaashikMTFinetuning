# Explicit language tags and model-specific adaptation

## Shared UTF-8 JSONL contract

Keep clean parallel sentence text and identify the language pair **per row**:

```json
{"id":"bra-doc1-001","source_lang":"bra_Deva","target_lang":"hin_Deva","source":"...","target":"...","document_id":"bra-doc1","domain":"oral"}
```

Supported focus-language corpus tags (labels, not a guarantee of checkpoint
support) are `bho_Deva`, `mai_Deva`, `mag_Deva`, `awa_Deva`,
`bra_Deva`, `npi_Deva` and `mwr_Deva`. Hindi is `hin_Deva`,
English is `eng_Latn`. Field names are `source_lang` and
`target_lang`, **not** `src_lang` or `tgt_lang`. The latter are used by
legacy OneNMT *probes*, not by the new parallel-data contract.

When either language field is present, **both are mandatory** and must match
`xxx_Script`. Legacy single-language files without either field remain
supported via the notebook's `SRC` and `TGT` defaults. To make tagged copies
without modifying originals:

```python
from bhaashik.multilingual import write_tagged_copy
for part in ("train", "dev", "test"):
    write_tagged_copy(f"data/local/bra-hin/{part}.jsonl",
                      f"data/local/bra-hin-tagged/{part}.jsonl",
                      "bra_Deva", "hin_Deva")
```

**Only use this migration when the entire original split really has that
direction.** Mixed-language files must be labelled individually. Do not prefix
raw source/reference text before writing the JSONL, and never add tags to
evaluation references. The shared `require_routes`, `inspect_routes`
and corpus audit/visualizations record route counts and file hashes. The
cross-model comparison checks that sentence IDs, text and language fields
match exactly. A method that uses proxy tags must record the proxies and
language-conditioning strategy in its manifest.

## Choose the appropriate notebook

| Notebook | Purpose | Language tagging |
|---|---|---|
| `notebooks/01_nllb_research.ipynb` | Existing NLLB baselines, optional tag extension, multilingual LoRA/full training and evaluation | Native NLLB token, dedicated new tag, or experimental text prefix with explicitly supported proxy tags |
| `notebooks/03_indictrans2_adaptation.ipynb` | IndicTrans2 via reviewed HF model code and IndicTransToolkit; native directions or experimental new languages | Official `IndicProcessor` and native codes, text prefixes using explicit supported processing proxies, or dedicated tokens with **full** finetuning |
| `notebooks/04_bhashaverse_adaptation.ipynb` | BhashaVerse HF MBart-format adaptation; local assets and codec, preflight and checkpoints | Direction prefix `###src-to-tgt###` through the **original** SentencePiece/fairseq dictionary |
| `notebooks/02_onenmt_preflight_repairs.ipynb` | Existing OneNMT checkpoint repair and regression checks | Individual immutable candidate operations; mandatory when original prefix pieces or model IDs need adjustment |

`*-ver2` notebooks remain untouched.

Separate notebooks are intentional: dependencies, model loading, preprocessing,
checkpoint formats and runtime requirements differ. All share corpus diagnostics,
held-out dev comparison (`compare_dev`), review figures, and efficient selective
archive logic. The custom language routing is disabled or gated by default.

## Three conditioning strategies

**Native language tokens / prefixes.** The checkpoint already supports the
language and tokenizer/processor recognizes its tag. This is the baseline
method for all genuinely supported directions.

**Text-prefix conditioning.** Include the actual source and target labels
in the model's input only, while feeding native processing tags into
NLLB/IndicTrans2 as **explicit proxies**. This is a controlled experiment:
predictions are *not* evidence that the original checkpoint supported the new
language. When the target language is unsupported, generation/postprocessing
needs particular scrutiny. Comparisons must include an unmodified model on
the same data/decoding protocol, and native-speaking evaluation.

**Dedicated language tokens.** Register new control tokens with the model
tokenizer, resize input/output embeddings, initialize new rows, then **train
them** (ordinary attention-only LoRA freezes them). For NLLB use the existing
`bhaashik.modeling.extend` operation into a separate folder, then set
`MODE='full'` and train its saved checkpoint, with
`NLLB_CONDITIONING='dedicated_token'`. For IndicTrans2, the optional trainer
can register `<bhaashik_src_bra_Deva>` and
`<bhaashik_tgt_hin_Deva>` special tokens and resize the model; this strategy
also requires full finetuning. For BhashaVerse, new dictionary/SPM IDs must
go through the existing **separate** `repair-spm-extend`,
`repair-append`, `repair-resize` operations first. Never reorder original
fairseq IDs or use `AutoTokenizer` for the saved BhashaVerse checkpoint.
`repair_manifest.json` records whether new embedding training is required,
and attention-only LoRA refuses these candidates.

One cannot obtain true language support by merely prepending text or
registering a token. Actual learning requires appropriately licensed parallel
data and fine-tuning, and new language quality must be measured rather than
inferred from the preprocessing format.

## Practical preflight and evaluation

1. Check licences, provenance, script, exact/near-duplicate leakage,
   domain, row-language label consistency, length ratios and document
   boundaries using `inspect_splits` and `inspect_routes`. Inspect examples
   manually. Freeze train/dev/test hashes.
2. Choose one model and strategy. Use native official support for control
   baselines, and only after verifying actual coverage use tag extension.
   The published BhashaVerse paper and HF model card have differing explicit
   language inventories: do not assume 36 supported inference tags.
3. Freeze the **same dev sample and IDs** for model comparisons. Never choose
   a strategy from test-set metrics. Report chrF++, sacreBLEU with signatures,
   qualitative review sheets, Hindi drift and named-entity/numeral errors,
   per-language score variations and retention on previously supported routes.
4. First pilot with a 1-step training run on tiny data where feasible, then
   train on real corpus with clearly labelled hypotheses and multiple seeds.
   Track model/dataset hashes, model-code revisions, proxy labels and any
   tokenizer/dictionary repair manifest. Check old-language retention, not
   just the target new language.
5. Export a complete model plus required tokenizer assets and manifest.
   Preserve Trainer checkpoints separately (`INCLUDE_CHECKPOINTS=True`)
   when exact resumption is needed. Notebook-only interactive files are
   ephemeral in Kaggle/Colab.

## Installation and important limitations

NLLB/OneNMT share `scripts/setup_notebook.py --extras train` (BhashaVerse
repair may additionally require `train,repairs`). In a fresh notebook kernel,
the installer verifies CUDA and core packages, and selectively cleans up
incompatible old TorchAO. IndicTrans2 additionally needs the separately
installed and reviewed official **IndicTransToolkit** and its model's HF
custom code; it is recommended to run IndicTrans2 in a separate environment
rather than modify an active NLLB/OneNMT kernel.

The IndicTrans2 `native`/proxy trainer, BhashaVerse HF MBart-only trainer and
new NLLB mixed-tag paths are **experimental implementations**. The repository's
synthetic unit tests cannot prove compatibility or training quality on real
published checkpoints. In particular, BhashaVerse HF parity with independent
upstream outputs and effective new-direction generation remain research
preconditions. CPU/GPU memory, model licences and usage limits vary.
