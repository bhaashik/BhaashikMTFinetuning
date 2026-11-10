"""Repair dictionary-backed generation IDs without touching weights or architecture."""
from ..preflight import AssetError, FairseqCodec, read_object
from ..data import write_json
from .common import candidate_snapshot


def repair_generation(assets, output):
    codec = FairseqCodec(assets)
    expected = {'bos_token_id': codec.special['bos'], 'pad_token_id': codec.special['pad'],
                'eos_token_id': codec.special['eos'], 'decoder_start_token_id': codec.special['eos'],
                'forced_bos_token_id': None, 'forced_eos_token_id': None, 'forced_decoder_ids': None}
    if any(codec.special.get(k) != v for k,v in {'bos': 0, 'pad': 1, 'eos': 2, 'unk': 3}.items()):
        raise AssetError('Unexpected special-ID convention; a config patch cannot repair a remapped dictionary')
    if not codec.config:
        raise AssetError('config.json is required; architecture settings will not be invented')
    with candidate_snapshot(assets, output, 'repair-generation') as (stage, manifest):
        changes = {}
        for name in ('config.json', 'generation_config.json'):
            config = read_object(stage/name) if (stage/name).exists() else {}
            changes[name] = {k: {'old': config.get(k), 'new': v} for k, v in expected.items() if k not in config or config[k] != v}
            config.update(expected)
            write_json(stage/name, config)
        manifest['changes'] = changes
        manifest['preserved'] = 'Weights, dictionary, segmentation, model dimensions, context and other decoding settings are unchanged.'
    return manifest
