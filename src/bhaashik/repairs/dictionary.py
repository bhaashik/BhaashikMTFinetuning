"""Restore a proposed dictionary to trusted IDs; append only into shared MBart vocab."""
import copy
from ..data import file_hash, write_json
from ..preflight import AssetError, FairseqCodec, read_object
from .common import active_spm_pieces, candidate_snapshot, require_shared_dictionary, selected_pieces


def repair_dictionary(assets, proposed_dictionary, output):
    """Always use the trusted checkpoint, never assume reordered trained weights can be relabelled."""
    trusted = FairseqCodec(assets)
    proposed = read_object(proposed_dictionary)
    corrections = {}
    for side in ('src', 'tgt'):
        mapping = proposed.get(side)
        if not isinstance(mapping, dict):
            raise AssetError('Proposed dictionary must contain src and tgt objects')
        original = trusted.dictionary[side]
        corrections[side] = {
            'restored_or_corrected': sum(mapping.get(p) != i for p, i in original.items()),
            'discarded_proposed_additions': [p for p in mapping if p not in original]}
    with candidate_snapshot(assets, output, 'repair-dictionary', [proposed_dictionary]) as (stage, manifest):
        write_json(stage/'fairseq_dict.json', trusted.dictionary)
        manifest['changes'] = {'proposed_dictionary_sha256': file_hash(proposed_dictionary), 'sides': corrections,
                               'specials_restored': trusted.special}
        manifest['policy'] = 'Trusted dictionary AND trusted checkpoint retained. Proposed reordered/trained weights are never imported. Add wanted new pieces in a separate append operation.'
    return manifest


def repair_append(assets, output, tokens=None, from_spm=False, max_additions=10000):
    if bool(tokens) == bool(from_spm):
        raise AssetError('Choose exactly one of tokens or from_spm')
    if type(max_additions) is not int or max_additions < 1:
        raise AssetError('max_additions must be positive')
    codec = FairseqCodec(assets)
    require_shared_dictionary(codec)
    selected = active_spm_pieces(codec) if from_spm else selected_pieces(tokens)
    new = [p for p in selected if p not in codec.dictionary['src']]
    if len(new) > max_additions:
        raise AssetError('Too many additions; inspect the proposal or explicitly increase max_additions')
    dictionary = copy.deepcopy(codec.dictionary)
    next_id = max(dictionary['src'].values()) + 1
    additions = []
    active = set(active_spm_pieces(codec))
    for piece in new:
        for side in ('src', 'tgt'):
            dictionary[side][piece] = next_id
        additions.append({'piece': piece, 'id': next_id, 'active_spm_piece': piece in active})
        next_id += 1
    with candidate_snapshot(assets, output, 'repair-append', [tokens]) as (stage, manifest):
        write_json(stage/'fairseq_dict.json', dictionary)
        manifest['changes'] = {'added': additions, 'old_ids_preserved': True,
                               'src_tgt_synchronized': True, 'already_present': [p for p in selected if p not in new]}
        manifest['requires_embedding_training'] = manifest['requires_embedding_training'] or bool(new)
        manifest['pending'] = ['Run repair-resize with the immutable reference before using added IDs.',
                               'Inactive pieces require a separate SPM extension/replacement; dictionary append alone does not activate them.']
    return manifest
