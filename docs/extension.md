# Extending to an unsupported language

Bhojpuri `bho_Deva`, Maithili `mai_Deva` and Magahi `mag_Deva` already exist in NLLB. Do not add duplicate tags for them. IndicTrans2's coverage is different.

For a well-defined, genuinely unsupported language/variety (example tag below is illustrative, not a claim about standardization):

```bash
bhaashik extend --model facebook/nllb-200-distilled-600M \
  --language raj_Deva --donor hin_Deva --output runs/extended-base
```

The command preserves existing special tokens, resizes shared input/output embeddings, copies the donor row, and saves a regular HF model + tokenizer with an extension manifest. SentencePiece subword vocabulary stays intact. This reuses Devanagari pieces; it is not tokenizer retraining. Check unknown-token rate, fertility and orthographic coverage before assuming adequacy for another script.

Use the resulting folder as `model`, the new language tag as `src_lang` or `tgt_lang`, and **`mode: full`** in a training config. LoRA-only training would freeze the new embedding row, so this runner rejects it for marked extended checkpoints. Full training has a larger memory budget; shorten sequences or reduce batch size. Initialization alone does not learn a new language. The new special token is saved with the tokenizer and works with `AutoTokenizer` and `AutoModelForSeq2SeqLM`; always distribute both together. Existing tags must continue to reload unchanged.

A proper extension experiment includes borrowed-Hindi-tag inference, donor vs random initialization, limited-data learning curves, native-speaker error analysis and regression checks on Hindi and supported neighboring languages. The CLI implements donor initialization; random initialization and comparative studies remain research work. IndicTrans2 extension needs its own tokenizer/preprocessing changes and is outside the NLLB runner.

## OneNMT/BhashaVerse candidates

Use the [read-only OneNMT preflight](preflight.md) to assess append-only and replacement-tokenizer proposals, including ID drift and retained-language HF behavior. The NLLB `extend` command above does not apply to OneNMT.

After diagnosis, use the [separate candidate repairs](repairs.md) to change one component at a time. Candidate repair success does not establish learned support; embedding initialization still requires training and retained-language evaluation.
