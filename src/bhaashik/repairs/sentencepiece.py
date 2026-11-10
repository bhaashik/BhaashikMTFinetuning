"""Separate SPM extension, retraining and replacement; dictionary IDs stay unchanged."""
import shutil
from pathlib import Path
from ..data import file_hash
from ..preflight import AssetError, FairseqCodec
from .common import candidate_snapshot, selected_pieces


def repair_spm_extend(assets, tokens, output):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    codec = FairseqCodec(assets)
    selected = selected_pieces(tokens)
    proto = pb.ModelProto()
    proto.ParseFromString((codec.root/'onemtv3b_spm.model').read_bytes())
    inventory = {p.piece: i for i, p in enumerate(proto.pieces)}
    added, existing = [], []
    for piece in selected:
        if piece in inventory:
            if proto.pieces[inventory[piece]].type != pb.ModelProto.SentencePiece.NORMAL and proto.pieces[inventory[piece]].type != pb.ModelProto.SentencePiece.USER_DEFINED:
                raise AssetError('Cannot reactivate control/unknown/unused pieces through append-only SPM extension')
            existing.append(piece)
            continue
        entry = proto.pieces.add()
        entry.piece, entry.score, entry.type = piece, 0.0, pb.ModelProto.SentencePiece.USER_DEFINED
        proto.trainer_spec.user_defined_symbols.append(piece)
        added.append({'piece': piece, 'spm_id': len(proto.pieces)-1})
    proto.trainer_spec.vocab_size = len(proto.pieces)
    serialized = proto.SerializeToString()
    candidate = spm.SentencePieceProcessor(model_proto=serialized)
    if any(candidate.id_to_piece(i) != codec.sp.id_to_piece(i) for i in range(codec.sp.get_piece_size())):
        raise AssetError('Internal error: SPM extension changed an existing native piece ID')
    with candidate_snapshot(assets, output, 'repair-spm-extend', [tokens]) as (stage, manifest):
        (stage/'onemtv3b_spm.model').write_bytes(serialized)
        # A stale human-readable .vocab would misleadingly describe the old tokenizer.
        (stage/'onemtv3b_spm.vocab').write_text(''.join(f'{p.piece}\t{p.score}\n' for p in proto.pieces), encoding='utf-8')
        manifest['changes'] = {'added_user_defined': added, 'already_present': existing,
                               'old_native_spm_ids_preserved': True, 'normalizer_preserved': True,
                               'dictionary_unchanged': True, 'weights_unchanged': True}
        manifest['pending'] = ['Append dictionary entries and initialize/train corresponding model rows separately.',
                               'USER_DEFINED matching may change segmentation of old-language text containing these strings; rerun drift and retention checks.']
    return manifest


def repair_spm_replace(assets, model_path, output):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    codec = FairseqCodec(assets)
    model_path = Path(model_path)
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    if processor.id_to_piece(processor.unk_id()) != '<unk>':
        raise AssetError('Replacement tokenizer must preserve the <unk> piece convention')
    proto = pb.ModelProto();proto.ParseFromString(model_path.read_bytes())
    with candidate_snapshot(assets, output, 'repair-spm-replace', [model_path]) as (stage, manifest):
        shutil.copy2(model_path, stage/'onemtv3b_spm.model')
        (stage/'onemtv3b_spm.vocab').write_text(''.join(f'{p.piece}\t{p.score}\n' for p in proto.pieces), encoding='utf-8')
        missing = {side: [processor.id_to_piece(i) for i in range(processor.get_piece_size())
                         if not processor.is_control(i) and not processor.is_unknown(i) and not processor.is_unused(i)
                         and processor.id_to_piece(i) not in codec.dictionary[side]] for side in ('src', 'tgt')}
        manifest['changes'] = {'replacement_sha256': file_hash(model_path), 'dictionary_unchanged': True,
                               'weights_unchanged': True, 'missing_dictionary_pieces': {s: {'count': len(v), 'examples': v[:30]} for s,v in missing.items()}}
        manifest['pending'] = ['Run preflight to quantify segmentation/normalization drift.',
                               'Append needed active pieces and initialize/train model rows before claiming compatibility.']
    return manifest


def repair_spm_train(assets, text_path, output, vocab_size, model_type='unigram'):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    if type(vocab_size) is not int or vocab_size < 8 or model_type not in ('unigram', 'bpe', 'char'):
        raise AssetError('Choose a positive vocab_size >= 8 and unigram, bpe or char')
    text_path = Path(text_path)
    if not text_path.is_file():
        raise AssetError('Training text must be a nonempty UTF-8 line-oriented file')
    with text_path.open(encoding='utf-8') as handle:
        if not any(line.strip() for line in handle):
            raise AssetError('Training text must be a nonempty UTF-8 line-oriented file')
    codec = FairseqCodec(assets)
    original = pb.ModelProto(); original.ParseFromString((codec.root/'onemtv3b_spm.model').read_bytes())
    normalizer = original.normalizer_spec
    if normalizer.name not in ('identity', 'nmt_nfkc', 'nfkc', 'nmt_nfkc_cf', 'nfkc_cf'):
        raise AssetError('Custom/unknown normalizer: supply an explicitly trained replacement through repair-spm-replace')
    args = dict(input=str(text_path.resolve()), vocab_size=vocab_size, model_type=model_type,
                character_coverage=1.0, hard_vocab_limit=False, num_threads=1,
                shuffle_input_sentence=False, input_sentence_size=0, minloglevel=2,
                normalization_rule_name=normalizer.name, add_dummy_prefix=normalizer.add_dummy_prefix,
                remove_extra_whitespaces=normalizer.remove_extra_whitespaces,
                escape_whitespaces=normalizer.escape_whitespaces,
                bos_id=codec.sp.bos_id(), eos_id=codec.sp.eos_id(), unk_id=codec.sp.unk_id(), pad_id=codec.sp.pad_id())
    for name in ('bos','eos','unk','pad'):
        idx = args[name+'_id']
        if idx >= 0:
            args[name+'_piece'] = codec.sp.id_to_piece(idx)
    with candidate_snapshot(assets, output, 'repair-spm-train', [text_path]) as (stage, manifest):
        spm.SentencePieceTrainer.train(model_prefix=str(stage/'onemtv3b_spm'), **args)
        trained = pb.ModelProto();trained.ParseFromString((stage/'onemtv3b_spm.model').read_bytes())
        manifest['changes'] = {'training_text_sha256': file_hash(text_path), 'requested_vocab_size': vocab_size,
                               'actual_vocab_size': len(trained.pieces), 'model_type': model_type,
                               'num_threads': 1, 'shuffle_input_sentence': False, 'dictionary_unchanged': True,
                               'weights_unchanged': True,
                               'normalizer_bytes_preserved': normalizer.SerializeToString() == trained.normalizer_spec.SerializeToString()}
        manifest['pending'] = ['High-risk ablation: segmentation has been retrained, not model weights.',
                               'Keep old-language text in tokenizer training, append dictionary pieces without reordering, initialize/train rows, and run behavior regression.']
    return manifest
