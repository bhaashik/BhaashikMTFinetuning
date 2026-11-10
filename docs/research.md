# Research protocol

## Questions and baselines

Study whether targeted supervision improves an established non-LLM encoder–decoder MT model for Bhojpuri, Maithili and Magahi without erasing language identity. All three have native NLLB tags; baseline inference without task-specific finetuning is not evidence of an unsupported-language extension. Unseen translation directions and unseen languages are different zero-shot settings.

Run Hindi ↔ each focus language, then English ↔ each where licensed bitext permits. Report each direction separately. First freeze data, checkpoint revisions and decoding; generate untouched NLLB-600M predictions. Compare LoRA with full finetuning at matched data/token budgets. IndicTrans2 distilled models are an additional comparator for supported Maithili directions, using their upstream preprocessing and toolkit. Unsupported IndicTrans2 directions require explicit extension work, not passing unrecognized tags.

## Experimental matrix

- Data sizes: 1k / 5k / 10k / all available verified pairs; nested subsets and identical test sets.
- Seeds: 13, 42, 87. New output directory per seed/condition. Report per-seed scores and mean ± standard deviation.
- Ablations: native baseline → LoRA → full finetuning; later multilingual related-language transfer, filtered backtranslation and new-language embedding initialization. Future ablations need implementation and their own validation.
- Select learning rate, rank and epochs on dev only. Record actual optimizer steps and effective batch tokens. Loss-based selection is implemented; inspect dev chrF++ on shortlisted checkpoints if used for selection, and record that policy.
- Extension study: choose a genuinely unsupported and well-defined variety, compare Hindi-tag transfer to an explicit new tag, then compare random versus donor initialization. A borrowed tag is a proxy baseline, not native support.

## Reporting

Use `bhaashik evaluate --input candidate.jsonl --baseline baseline.jsonl --output evaluation.json`. IDs and source/reference/metadata must align exactly. Report corpus sacreBLEU (13a), chrF++ (word order 2), signatures, domains, sample counts and paired bootstrap deltas with percentile 95% intervals. The command resamples documents if all rows have document IDs, otherwise sentences. This interval concerns the test sample, not training-seed uncertainty. Small samples and few documents give unreliable intervals; report units and do not infer significance from a single run. Unicode-aware chrF++ is the primary lexical metric for these scripts. Tokenization choices affect BLEU; publish them.

Optional COMET is a later external analysis: check evaluator coverage, license, checkpoint/version, and reliability on these languages. Do not claim its score measures dialect fidelity. Add human judgments regardless of automatic scores.

## Human error study

`bhaashik errors` exports a shuffled, model-name-free sheet. It is a single-system export: for a blind multi-system study, pool outputs under a private randomized system key, hide original IDs if they reveal systems, and preserve an adjudication mapping outside the annotator view. Hide heuristic flags in the primary rating pass to avoid bias. Double-annotate a stratified sample across languages, directions, domains and score bands, plus a random sample. Analyze the random sample separately from targeted error cases.

Categories: mistranslation, omission, addition/hallucination, Hindi drift, morphology/agreement, honorific/register, lexical/dialect choice, named entity, numeral, word order, orthography and acceptable variant. Mark spans and minor/major/critical severity. Rate adequacy and fluency 1–5; document rubric anchors and allow multiple errors per sentence via repeated rows. Use qualified speakers with relevant regional knowledge, fair compensation and consent. Compute weighted kappa for ordinal ratings and category agreement before adjudication using an external analysis tool; these calculations are not implemented here. Report annotator counts, coverage, disagreement and adjudication policy.

## Reproducibility and citations

Release configs, manifests, data hashes, splits when rights permit, model cards, decoding settings, predictions and annotation guidelines. Pin Hub revisions to immutable commit SHAs in final runs. Record notebook accelerator and runtime versions; free GPU quotas and available memory vary. Deterministic seeds do not ensure bitwise GPU reproducibility. Report training energy/time estimates when available and negative results.

- NLLB Team (2022), [No Language Left Behind](https://arxiv.org/abs/2207.04672).
- Gala et al. (2023), [IndicTrans2](https://arxiv.org/abs/2305.16307).
- Hu et al. (2021), [LoRA](https://arxiv.org/abs/2106.09685).
- Post (2018), [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/).
- Popović (2017), [chrF++](https://aclanthology.org/W17-4770/).

No numerical quality results are supplied. A paper must establish its own improvements against these baselines and survey newer relevant MT literature before claiming state of the art.
