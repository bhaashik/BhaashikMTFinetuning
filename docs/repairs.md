# Separate OneNMT candidate repair operations

Run a [preflight](preflight.md) first, then select an individual operation below. No command interprets a report as permission to apply every possible change. These commands use the existing **standard HF MBart architecture**; they do not implement or convert architectures, invoke OpenNMT/Fairseq, run training, or claim that newly initialized tokens translate correctly.

Install `pip install -e '.[repairs]'` for dictionary/SPM/config operations. `repair-resize` also requires `pip install -e '.[train]'`. All inputs are local; there are no model downloads or credential requirements.

## Independent operations

| Preflight finding / experiment | Command | What it changes |
|---|---|---|
| Wrong decoder start, EOS/PAD/BOS or forced language BOS | `repair-generation` | Known generation IDs/settings in config and generation config |
| A proposed dictionary removes/reorders trusted old IDs | `repair-dictionary` | Restores the trusted original dictionary/checkpoint pair; compares the bad proposal |
| Selected pieces need dictionary entries | `repair-append` | Appends synchronized source/target IDs after the old maximum |
| New selected strings are not emitted by SPM | `repair-spm-extend` | Appends USER_DEFINED SPM pieces, retaining original piece IDs/normalizer |
| Test tokenizer retraining as an ablation | `repair-spm-train` | Trains a separate SPM model with an explicit model type and vocabulary budget |
| Use an externally prepared tokenizer candidate | `repair-spm-replace` | Installs that SPM without reordering model dictionary IDs |
| Dictionary additions need model rows / initialization | `repair-resize` | Grows HF tables if necessary and initializes only newly assigned semantic rows |

Every command writes a **new directory**, with its own `repair_manifest.json`. Existing destinations and overlapping input/output paths are refused. Operations use a temporary sibling directory and promote it only after completion. On failure, only that operation's temporary directory is removed. Original inputs are never edited. The manifest records input/output small-asset hashes, operation details, parent change history and pending checks. Original model cards/license files are retained where present; review/update the card before any release.

Checkpoint files are copied, not hardlinked. This prevents later candidate writes from corrupting the original through a shared inode. It also consumes disk: each full candidate can need another checkpoint-sized allocation, and an atomic operation needs temporary headroom. Keep only experimental branches you need. Weight sizes/filenames are recorded, but multi-gigabyte weights are not hashed on each operation; preserve trusted immutable revisions/checksums separately. Compatibility tokenizer files such as `tokenizer.json` are not copied: these candidates use the Fairseq codec and HF model loader, not the upstream broken AutoTokenizer path.

## Fix generation IDs independently

```bash
bhaashik repair-generation --assets data/local/original \
  --output data/local/generation-fixed
```

Uses the dictionary-backed OneNMT convention BOS=0, PAD=1, EOS=2, UNK=3 and decoder start=EOS. Clears forced BOS, forced EOS and forced decoder IDs for explicit unforced token generation. Retains other decoding settings, including beam and repetition settings. Does not change dictionary IDs, tokenization, weights, layer dimensions or context. An unexpected dictionary-special convention is refused; the command does not guess a new convention or invent architecture config. Rerun reference replay if these settings differ from an earlier recorded generation profile.

## Restore dictionary IDs from a trusted baseline

```bash
bhaashik repair-dictionary --assets data/local/trusted-original \
  --dictionary data/local/broken-proposal/fairseq_dict.json \
  --output data/local/ids-restored
```

`--assets` must be the **trusted original checkpoint/dictionary pair**, not a checkpoint trained under the broken proposal. Output preserves that trusted checkpoint and dictionary. The proposal is used to report corrected/missing entries and discarded extra pieces. Wanted extra pieces can be appended afterward with `repair-append`.

This is a reset to a known pair, not a permutation of already adapted model weights. If weights were trained using reordered IDs, relabelling their dictionary would be unsafe. Such weights are not imported. Recover the trusted original and rerun the adaptation, or investigate that checkpoint separately. Duplicate JSON keys, missing trusted assets and ambiguous dictionary formats are not guessed away.

## Append dictionary entries independently

```bash
bhaashik repair-append --assets data/local/original \
  --tokens data/preflight/selected-tokens.example.json \
  --output data/local/dictionary-appended
```

Selection format: `{"src":["▁newpiece"],"tgt":["anotherpiece"]}`. For this shared-embedding HF MBart path, mutation requires **identical original source and target dictionaries**. Differing dictionaries need upstream validation; automatic reconciliation could change embedding semantics.

The union of selected strings is appended to **both** dictionaries, using the same ID for a piece and distinct IDs for distinct pieces. Every old ID is preserved, including sparse old ID layouts. New IDs start after the old maximum; holes are not reused. `--max-additions` defaults to 10,000 and rejects unexpectedly large proposals. Inactive SPM pieces are recorded as inactive; this command never silently edits the tokenizer.

After a replacement/retrained SPM, append all active missing pieces instead:

```bash
bhaashik repair-append --assets data/local/tokenizer-candidate \
  --from-spm --output data/local/dictionary-synced
```

This excludes control, unknown and unused SPM entries. The model is not resized by this operation. `requires_embedding_training` remains true through later repair operations until a separately validated training workflow resolves it.

## Activate selected pieces without retraining SPM

```bash
bhaashik repair-spm-extend --assets data/local/original \
  --tokens data/preflight/selected-tokens.example.json \
  --output data/local/spm-extended
```

Appends USER_DEFINED entries to the SPM protobuf. Existing native SPM IDs and normalizer are preserved. Existing normal/user-defined pieces are retained; control/unknown/unused entries cannot be repurposed. This does not change the Fairseq dictionaries or embedding rows.

Use actual SentencePiece-style strings and test their emissions. A word-boundary marker `▁` is different from literal whitespace. USER_DEFINED matching can change old-text segmentation wherever those strings occur. Preserving old SPM IDs does **not** imply preserving all old segmentations or translation outputs. Run preflight drift checks and retained-language HF probes.

## Train or install a replacement tokenizer separately

```bash
bhaashik repair-spm-train --assets data/local/original \
  --text data/local/mixed-old-new-text.txt \
  --model-type unigram --vocab-size 64000 \
  --output data/local/retrained-spm
```

Vocabulary size and model type are explicit CLI choices, not automatically inferred fixes. Training text is UTF-8, one sentence per line. Include retained-language/domain text and exclude all evaluation material. Training is single-threaded with input shuffling disabled; exact artifact bytes may still include path/version metadata. Supported named normalization settings and native special IDs are carried forward, and normalizer-byte equality is recorded. Unknown/custom normalizers are refused rather than silently replaced. SentencePiece can produce fewer pieces than requested (`hard_vocab_limit=False`); requested and actual sizes are recorded.

Alternatively:

```bash
bhaashik repair-spm-replace --assets data/local/original \
  --spm-model data/local/external/tokenizer.model \
  --output data/local/replacement-spm
```

The supplied model must be readable and use the `<unk>` piece convention. Native SPM IDs may differ because model IDs still come from the preserved Fairseq dictionary. The `.vocab` text is regenerated so it cannot silently describe the old model. Missing dictionary pieces are reported. Neither operation rebuilds/reorders dictionaries or modifies weights. Use `repair-append --from-spm` followed by `repair-resize` when new pieces need model rows. These are high-risk tokenizer experiments, not required steps for adding every language or dataset.

## Initialize model rows separately

```bash
bhaashik repair-resize --assets data/local/dictionary-appended \
  --reference-assets data/local/original \
  --initialization subword-mean --seed 42 \
  --output data/local/initialized
```

The immutable reference supplies old dictionary IDs and the old SPM for initialization donors. All old mappings/specials must match, and new IDs must form a contiguous append after the old maximum. Source/target dictionaries must be identical for the shared input-table path. There is no `ignore_mismatched_sizes` escape: incomplete/unexpected checkpoint loading, inconsistent dimensions, missing old-vocabulary rows and nonfinite vocabulary tensors cause failure. Serialized embedding aliases that HF would tie must also agree: a mismatch is refused before HF can silently overwrite learned values. Only a stale `config.vocab_size` can be inferred from actual consistent tensor rows; layer/position architecture is not reconstructed.

Initialization choices:

- `subword-mean` (default): mean of known original subpiece rows, excluding specials; fall back to the original used-row mean when no donor is available. Donor IDs/fallbacks are recorded.
- `mean`: mean of the original used-vocabulary rows.
- `random`: seeded, zero-mean values scaled by the original used-row standard deviation.

Tables grow only when required. If newly assigned IDs occupy previously padded model rows, those rows are still explicitly initialized. Unassigned old padding and all trusted old rows are preserved. Tied and untied output heads and `final_logits_bias` are handled; old row/bias equality is checked. A stale copied shard index/weights are removed **only in the candidate**, and the new safetensors checkpoint is saved/reloaded to verify vocabulary tensors. `--max-new-rows` defaults to 10,000 to reject accidental enormous allocations.

This runs on CPU and can require substantial RAM and disk for full weights, preserved embedding copies, and save/reload verification. It is not a lightweight header-only check. Allow room for two model instances during verification; no full OneNMT RAM/runtime benchmark has been measured here.

**Initialized embeddings are not learned language support.** New output rows alter the softmax competition and can be selected before they are useful, even when every old row is unchanged. Attention-only LoRA freezes embeddings and will not train these rows. Follow with a training workflow that explicitly updates them; the current NLLB-only trainer does not train OneNMT. The manifest marker is deliberately not auto-cleared by a successful repair.

## Example staged append experiment

```bash
# 1. Activate pieces only if coverage checks justify it.
bhaashik repair-spm-extend --assets data/local/original \
  --tokens data/local/selected.json --output data/local/stage-1-spm
# 2. Give active pieces model dictionary IDs.
bhaashik repair-append --assets data/local/stage-1-spm \
  --tokens data/local/selected.json --output data/local/stage-2-dict
# 3. Allocate/initialize model rows using the immutable old vocabulary.
bhaashik repair-resize --assets data/local/stage-2-dict \
  --reference-assets data/local/original --output data/local/stage-3-model
# 4. Quantify structural drift; pending training/review is expected.
bhaashik preflight --assets data/local/original \
  --candidate-assets data/local/stage-3-model --corpus data/local/probes.jsonl \
  --report runs/preflight/after-repair.json
# 5. Probe HF behavior, then train new rows and repeat before any release.
bhaashik hf-regression --baseline-assets data/local/original \
  --candidate-assets data/local/stage-3-model --corpus data/local/probes.jsonl \
  --device cuda --precision fp16 --report runs/preflight/after-repair-behavior.json
```

You can skip SPM modification if current segmentation already emits the needed pieces. You can skip dictionary/model changes entirely when unchanged-vocabulary adaptation is adequate. Commands do not clear bad test results or force a passing preflight; intermediate candidates may correctly remain blocked pending subsequent explicit operations.

## Findings that are not mechanical repairs

- **Nonfinite/corrupt weights:** obtain a verified original checkpoint or investigate the failed training run. Do not fill arbitrary weights with zeros/random values.
- **Retention loss, Hindi drift, poor translation:** change training data/budget/regularization and rerun experiments with expert review. A vocabulary/config patch cannot establish quality.
- **Direction prefix collisions:** choose and test explicit distinguishable prefixes, possibly activate justified pieces, then train routing. The module does not invent language identity or claim that a new tag is learned.
- **Context overflow:** shorten representative probes explicitly or document any excluded examples; these commands do not increase positional context or silently truncate data.
- **Missing/incorrect architecture tensors:** recover compatible HF assets or ask upstream. Architecture reimplementation remains outside this project.

Every successful transformation exits 0 with `requires_validation`; refusal/failure exits 2 and does not leave a promoted candidate. Preflight/regression statuses and exit codes remain independent. Tests use synthetic assets and tiny standard HF checkpoints, including tied/untied heads, padding-row initialization, sharded reloads and failure rollback. Actual OneNMT compatibility, training quality and full-scale resource usage still need its real assets and trusted probe outputs.
