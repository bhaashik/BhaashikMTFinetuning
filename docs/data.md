# Data contract and provenance

Each UTF-8 JSONL row is one parallel sentence:

```json
{"id":"corpus-doc17-003","source":"...","target":"...","document_id":"corpus-doc17","domain":"news","provenance":"corpus@revision","license":"SPDX-or-license-URL"}
```

Required: nonempty string `id`, `source`, `target`. Predictions add `prediction` (empty outputs are retained and scored). Domain/document IDs are optional for loading, but research releases should include them, along with provenance and license. One language pair and direction per file; reverse by swapping source/target and updating config. Language tags belong in config, not sentence text. Never conflate Magahi with Hindi or Bhojpuri.

Store private/local data under ignored `data/local/`. The tiny examples are original illustrative fixtures, not a benchmark or a claim of native-speaker gold quality. Have language experts validate all real data.

## Candidate sources (verify terms and coverage before download)

- [FLORES-200](https://github.com/facebookresearch/flores): held-out evaluation for NLLB languages; retain dev/devtest separation. Existing models may have benchmark exposure. Never add these test sentences to finetuning.
- [BPCC / IN22](https://github.com/AI4Bharat/IndicTrans2): training and evaluation resources for scheduled languages including Maithili; do not assume Bhojpuri/Magahi coverage or one license for all components.
- [OPUS](https://opus.nlpl.eu/): inspect actual language pair availability, domain, deduplication and each constituent corpus's license. A catalog listing is not a quality guarantee.
- Community-contributed, consented text: document speaker region, variety, orthographic choices, rights and annotation process.

Acquisition is deliberate and manual in v0.1: the toolkit does not silently download, redistribute or synthesize datasets. Keep a manifest of URLs, versions, hashes, licenses, filtering counts and split membership. Preserve raw text separately; use NFC/whitespace normalization for leak detection, not Hindi spelling replacement. Add near-duplicate and semantic-overlap review to the implemented exact checks. Audit IDs, source texts, parallel pairs and document overlap across all splits before training. Freeze test hashes in the paper's artifact release.
