# Data contract and provenance

Each UTF-8 JSONL row is one parallel sentence:

```json
{"id":"corpus-doc17-003","source":"...","target":"...","source_lang":"bra_Deva","target_lang":"hin_Deva","document_id":"corpus-doc17","domain":"news","provenance":"corpus@revision","license":"SPDX-or-license-URL"}
```

Required: nonempty string `id`, `source`, `target`. For multilingual or extended-language work also supply **both** `source_lang` and `target_lang` on each JSONL row (e.g. `bra_Deva` and `hin_Deva`). The pair must be complete; incomplete or invalid tags are rejected. Legacy single-direction corpora may omit both and use explicit notebook defaults. Predictions add `prediction` (empty outputs are retained and scored). Domain/document IDs are optional for loading, but research releases should include them, along with provenance and license. Files may contain multiple language directions if **each row is explicitly tagged**. Reverse by swapping source/target, swapping both language fields, and documenting that operation. The original text stays unchanged; model-specific prefixes are added during tokenization. See [multilingual adaptation](multilingual-adaptation.md). Language tags belong in config, not sentence text. Never conflate Magahi with Hindi or Bhojpuri.

## Optional single-file splitting

A language pair may arrive as **one** UTF-8 `parallel.jsonl` instead of
pre-split train/dev/test files. Use the shared function:

```python
from bhaashik.data_splitting import split_parallel_jsonl

manifest = split_parallel_jsonl(
    "data/local/bra-hin/parallel.jsonl",
    "data/local/bra-hin/splits-s42-80-10-10",
    seed=42, ratios=(0.80, 0.10, 0.10),
    source_lang="bra_Deva", target_lang="hin_Deva",
    group_by_document=True, validate=False,
)
print(manifest["counts"], manifest["actual_ratios"])
```

The function writes `train.jsonl`, `dev.jsonl`, `test.jsonl` and
`split_manifest.json` into a **new** output directory. It does not modify
the input; identical runs reuse manifest-verified outputs, and changed
options/data require a distinct output directory. Seed controls the initial
shuffle; changing it generally changes assignments. Grouping keeps any
`document_id` together and also keeps repeated normalized source texts
together, even when documents are distinct. A document/source group is
indivisible, so actual ratios can differ. At least three independent groups
are required. `group_by_document=False` is available only when document-level
independence is not required; identical sources are still grouped.

The `source_lang`/`target_lang` parameters are optional **defaults for an
entire single-direction file**. If rows are already tagged individually with
mixed directions, omit both defaults so their labels are retained. The
splitter does not introduce artificial text prefixes or change gold target
strings.

`validate=False` skips the full `audit` on creation and is meant for
initial mechanics testing, **not research approval**. Schema validity and
nonempty splits are always enforced. Change to `validate=True` for the exact
ID, source, pair and document leakage audit; notebook
`VALIDATE_DATA=True` additionally computes corpus diagnostics and plots.
Model trainers independently enforce train/dev audits during actual training.
Near-duplicates, shared paraphrases and related content can still leak across
groups, so inspect the corpus manually and freeze split hashes for research.
The test split must not influence model selection.

Store private/local data under ignored `data/local/`. The tiny examples are original illustrative fixtures, not a benchmark or a claim of native-speaker gold quality. Have language experts validate all real data.

## Candidate sources (verify terms and coverage before download)

- [FLORES-200](https://github.com/facebookresearch/flores): held-out evaluation for NLLB languages; retain dev/devtest separation. Existing models may have benchmark exposure. Never add these test sentences to finetuning.
- [BPCC / IN22](https://github.com/AI4Bharat/IndicTrans2): training and evaluation resources for scheduled languages including Maithili; do not assume Bhojpuri/Magahi coverage or one license for all components.
- [OPUS](https://opus.nlpl.eu/): inspect actual language pair availability, domain, deduplication and each constituent corpus's license. A catalog listing is not a quality guarantee.
- Community-contributed, consented text: document speaker region, variety, orthographic choices, rights and annotation process.

Acquisition is deliberate and manual in v0.1: the toolkit does not silently download, redistribute or synthesize datasets. Keep a manifest of URLs, versions, hashes, licenses, filtering counts and split membership. Preserve raw text separately; use NFC/whitespace normalization for leak detection, not Hindi spelling replacement. Add near-duplicate and semantic-overlap review to the implemented exact checks. Audit IDs, source texts, parallel pairs and document overlap across all splits before training. Freeze test hashes in the paper's artifact release.
