"""Grow standard HF MBart tables and initialize only newly assigned vocabulary rows."""
from ..preflight import AssetError, FairseqCodec, checkpoint_layout
from .common import candidate_snapshot, require_shared_dictionary, weight_files


def verify_serialized_ties(assets, tie_output):
    """HF must not silently overwrite disagreeing serialized aliases while tying weights."""
    import torch
    from safetensors import safe_open
    suffixes = ('shared.weight', 'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight')
    if tie_output:
        suffixes += ('lm_head.weight',)
    locations = []
    for path in weight_files(assets):
        if path.suffix != '.safetensors':
            continue
        with safe_open(path, framework='pt', device='cpu') as handle:
            locations.extend((path, key) for key in handle.keys() if key.endswith(suffixes))
    if len(locations) < 2:
        return
    first_path, first_key = locations[0]
    for path, key in locations[1:]:
        with safe_open(first_path, framework='pt', device='cpu') as left, safe_open(path, framework='pt', device='cpu') as right:
            a, b = left.get_slice(first_key), right.get_slice(key)
            if a.get_shape() != b.get_shape():
                raise AssetError('Serialized shared-weight alias shapes disagree')
            for start in range(0, a.get_shape()[0], 4096):
                if not torch.equal(a[start:start+4096], b[start:start+4096]):
                    raise AssetError('Serialized shared-weight aliases disagree; HF tying could overwrite learned values. Resolve this with upstream rather than forcing a resize.')


def repair_resize(assets, reference_assets, output, initialization='subword-mean', seed=42, max_new_rows=10000):
    import torch
    from transformers import MBartConfig, MBartForConditionalGeneration
    if initialization not in ('subword-mean', 'mean', 'random'):
        raise AssetError('Initialization must be subword-mean, mean or random')
    if type(max_new_rows) is not int or max_new_rows < 1 or type(seed) is not int or seed < 0:
        raise AssetError('max_new_rows must be positive and seed must be a nonnegative integer')
    candidate, reference = FairseqCodec(assets), FairseqCodec(reference_assets)
    require_shared_dictionary(candidate)
    require_shared_dictionary(reference)
    old, new = reference.dictionary['src'], candidate.dictionary['src']
    if candidate.special != reference.special or any(new.get(p) != i for p,i in old.items()):
        raise AssetError('Resize requires all trusted old dictionary IDs and specials to be preserved')
    added = {p:i for p,i in new.items() if p not in old}
    if any(i <= max(old.values()) for i in added.values()):
        raise AssetError('New dictionary IDs must follow the trusted old maximum; do not reuse ID holes')
    if len(added) > max_new_rows:
        raise AssetError('Too many new semantic rows; increase max_new_rows only after inspecting the proposal')
    layout = checkpoint_layout(candidate)
    if candidate.config.get('model_type') != 'mbart' or not layout['weight_layout_checked']:
        raise AssetError('Resize requires local standard HF MBart safetensors weights')
    # Check actual tensors. Fix only a stale vocab_size, never layer dimensions/position tables.
    shapes = layout['embedding_shapes']
    rows = {shape[0] for shape in shapes.values() if len(shape) == 2}
    widths = {shape[1] for shape in shapes.values() if len(shape) == 2}
    if len(rows) != 1 or len(widths) != 1 or any(len(shape) != 2 for shape in shapes.values()):
        raise AssetError('Embedding/output table shapes disagree; this is not a simple vocabulary resize')
    actual_rows = next(iter(rows))
    if next(iter(widths)) != candidate.config.get('d_model') or actual_rows <= max(old.values()):
        raise AssetError('Checkpoint cannot represent the trusted old vocabulary or hidden width disagrees')
    required_rows = max(actual_rows, max(new.values()) + 1)
    if required_rows - actual_rows > max_new_rows:
        raise AssetError('Required row allocation exceeds max_new_rows; refusing a potentially accidental huge ID')
    if sorted(added.values()) != list(range(max(old.values()) + 1, max(old.values()) + 1 + len(added))):
        raise AssetError('Added IDs must be a contiguous append after the trusted maximum; use repair-append rather than introducing unmapped new output rows')
    config = MBartConfig.from_pretrained(str(assets), local_files_only=True)
    config.vocab_size = actual_rows
    verify_serialized_ties(assets, config.tie_word_embeddings)
    # No ignore_mismatched_sizes: missing/unexpected tensors must not silently initialize.
    model, info = MBartForConditionalGeneration.from_pretrained(
        str(assets), config=config, local_files_only=True, use_safetensors=True,
        torch_dtype='auto', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise AssetError(f'Checkpoint load is incomplete; refusing to manufacture unrelated weights: {info}')
    input_before = model.get_input_embeddings().weight.detach().clone()
    tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    output_before = input_before if tied else model.get_output_embeddings().weight.detach().clone()
    bias_before = model.final_logits_bias.detach().clone()
    if not torch.isfinite(input_before).all() or not torch.isfinite(output_before).all() or not torch.isfinite(bias_before).all():
        raise AssetError('Original vocabulary tensors contain nonfinite values; resizing cannot repair corrupted weights')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.resize_token_embeddings(required_rows)
    trusted_ids = sorted(set(old.values()))
    special_ids = set(reference.special.values())
    donor_log = {}
    # Compute fallback statistics once, not once per added piece.
    input_mean = input_before[trusted_ids].float().mean(dim=0)
    output_mean = output_before[trusted_ids].float().mean(dim=0)
    input_std = input_before[trusted_ids].float().std(unbiased=False).item()
    output_std = output_before[trusted_ids].float().std(unbiased=False).item()
    def vector(piece, before, fallback, std):
        donors = []
        if initialization == 'subword-mean':
            text = piece.replace('▁', ' ').strip()
            donors = [old[p] for p in reference.pieces(text) if p in old and old[p] not in special_ids]
        if initialization == 'random':
            value = torch.randn((before.shape[1],), generator=generator, dtype=torch.float32) * std
        elif donors:
            value = before[donors].float().mean(dim=0)
        else:
            value = fallback
        return value.to(dtype=before.dtype), donors
    with torch.no_grad():
        for piece, idx in added.items():
            value, donors = vector(piece, input_before, input_mean, input_std)
            model.get_input_embeddings().weight[idx].copy_(value)
            if not tied:
                value, _ = vector(piece, output_before, output_mean, output_std)
                model.get_output_embeddings().weight[idx].copy_(value)
            model.final_logits_bias[0, idx] = 0
            donor_log[piece] = {'id': idx, 'donor_ids': donors,
                                'global_mean_fallback': initialization == 'subword-mean' and not donors}
    # Preserve all original rows, including unused padding, except explicitly claimed new IDs.
    added_ids = set(added.values())
    unchanged = [i for i in range(actual_rows) if i not in added_ids]
    if not torch.equal(model.get_input_embeddings().weight[unchanged], input_before[unchanged]) or not torch.equal(model.get_output_embeddings().weight[unchanged], output_before[unchanged]):
        raise AssetError('Internal verification failed: an old vocabulary row changed')
    if not torch.equal(model.final_logits_bias[:, unchanged], bias_before[:, unchanged]):
        raise AssetError('Internal verification failed: old output biases changed')
    with candidate_snapshot(assets, output, 'repair-resize', [reference_assets]) as (stage, manifest):
        # Remove only copied weight files/index: a stale shard index must never point to old rows.
        for path in weight_files(stage):
            path.unlink()
        model.save_pretrained(stage, safe_serialization=True)
        reloaded, reload_info = MBartForConditionalGeneration.from_pretrained(
            str(stage), local_files_only=True, use_safetensors=True, torch_dtype='auto', output_loading_info=True)
        if any(reload_info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise AssetError('Reload verification found missing or mismatched tensors')
        for name in ('get_input_embeddings','get_output_embeddings'):
            expected = getattr(model,name)().weight
            observed = getattr(reloaded,name)().weight
            if not torch.equal(expected, observed):
                raise AssetError('Vocabulary tensors changed during save/reload')
        if not torch.equal(model.final_logits_bias, reloaded.final_logits_bias):
            raise AssetError('Output bias changed during save/reload')
        manifest['changes'] = {'old_model_rows': actual_rows, 'new_model_rows': required_rows,
                               'new_semantic_rows': donor_log, 'initialization': initialization, 'seed': seed,
                               'old_rows_verified_unchanged': True, 'save_reload_verified': True,
                               'input_output_weights_tied': tied, 'reference_assets': reference.fingerprint()}
        manifest['requires_embedding_training'] = manifest['requires_embedding_training'] or bool(added)
        manifest['pending'] = ['Initialized rows are not learned language support. Train them explicitly; attention-only LoRA freezes embeddings.',
                               'Run preflight and HF retention/adaptation regression before using the candidate for research.']
    return manifest
