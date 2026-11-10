# Implementation and validation boundaries

The toolkit implements one NLLB language pair per training run, merged LoRA/full HF exports, language-tag donor initialization, exact split auditing, prediction, sacreBLEU/chrF++, paired bootstrap and annotation export. It does not automatically mine corpora, generate synthetic training pairs, train IndicTrans2, compute COMET, perform language identification, or compute inter-annotator agreement.

GPU notebook settings are starting budgets, not guarantees of fitting every free accelerator. Default LoRA uses batch 1, accumulation 16, length 128 and gradient checkpointing. Try fp16 on a T4; bf16 requires compatible hardware. Full finetuning and extensions may exceed a free notebook's resources. Reduce length/accumulation evaluation batch, or use a paid/larger GPU rather than claim a completed experiment. Downloads and checkpoint writes need disk space; copy finished artifacts to persistent storage before runtime termination.

The toolkit is tested with a tiny randomly initialized local NLLB-compatible checkpoint to exercise real training and reload paths without downloading large weights. This validates mechanics, not translation quality, full-scale GPU memory, or notebook provider availability. Real benchmark runs, language expert review and paper results remain to be produced.

The Pages site is a static portal, not hosted model inference or GPU training. Report imports stay in your browser and are not uploaded to a server. It never fabricates evaluation results.

Model and dataset licenses are separate from this MIT code. NLLB weights are noncommercial CC-BY-NC 4.0. IndicTrans2 code is MIT; verify each selected checkpoint and corpus's own card/terms. Do not redistribute incompatible data or present NLLB as unrestricted commercial software.

The [OneNMT preflight](preflight.md) checks local tokenizer/dictionary/model layouts and optionally compares original/adapted outputs using standard HF MBart only. It does not train OneNMT, modify embeddings, rewrite SentencePiece, verify an alleged OpenNMT-to-Fairseq history, or establish architecture equivalence. Its new integration tests use synthetic assets and a tiny random HF model. Actual OneNMT parity and quality require real assets, independent reference outputs and language-expert review.

The separate [candidate repair operations](repairs.md) can mutate copied OneNMT assets and initialize standard HF MBart vocabulary rows. They preserve the source, record lineage, and require subsequent validation/training. They do not correct arbitrary corrupted model weights, implement architecture conversions, automatically improve translation quality, or provide a OneNMT training backend. Vocabulary mutations require identical source/target dictionaries on this shared-embedding path.
