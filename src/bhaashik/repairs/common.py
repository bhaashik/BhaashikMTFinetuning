"""Transactional candidate snapshots and shared input validation."""
from contextlib import contextmanager
import shutil
import tempfile
from pathlib import Path
from ..data import file_hash, write_json
from ..preflight import AssetError, FairseqCodec, read_object


ASSET_FILES = ('fairseq_dict.json', 'onemtv3b_spm.model', 'onemtv3b_spm.vocab',
               'config.json', 'generation_config.json', 'README.md', 'LICENSE', 'LICENSE.txt',
               'NOTICE', 'CITATION.cff', 'repair_manifest.json')


def weight_files(root):
    root = Path(root)
    index = root/'model.safetensors.index.json'
    if index.exists():
        mapping = read_object(index).get('weight_map')
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(n, str) for n in mapping.values()):
            raise AssetError('Invalid safetensors shard index')
        names = sorted(set(mapping.values()))
        result = [index]
        for name in names:
            path = (root/name).resolve()
            if not path.is_relative_to(root.resolve()) or path.suffix != '.safetensors' or not path.is_file():
                raise AssetError('Missing shard or shard outside the asset directory')
            result.append(path)
        return result
    return [root/'model.safetensors'] if (root/'model.safetensors').is_file() else []


def selected_pieces(path):
    value = read_object(path)
    if set(value) - {'src', 'tgt'}:
        raise AssetError('Token selection must contain src/tgt lists')
    result = []
    for side in ('src', 'tgt'):
        tokens = value.get(side, [])
        if not isinstance(tokens, list) or any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens):
            raise AssetError('Select nonempty SentencePiece strings without literal whitespace')
        if len(set(tokens)) != len(tokens):
            raise AssetError('Duplicate selected tokens')
        for token in tokens:
            if token not in result:
                result.append(token)
    if not result:
        raise AssetError('Token selection is empty')
    return result


def file_inventory(root):
    """Small asset hashes; large weights are recorded by path/size, not read twice."""
    root = Path(root)
    return {
        'asset_sha256': {n: file_hash(root/n) for n in ASSET_FILES if (root/n).is_file() and n != 'repair_manifest.json'},
        'weights': [{'file': str(p.relative_to(root.resolve())), 'bytes': p.stat().st_size} for p in weight_files(root.resolve())],
        'weight_identity_note': 'Weights are copied as files, not hardlinks. Pin source checkpoint revisions/checksums separately; weights are not hashed by this manifest.'}


@contextmanager
def candidate_snapshot(assets, output, operation, extra_inputs=()):
    """Only our temporary directory is removed on failure; originals are never edited."""
    source, destination = Path(assets).resolve(), Path(output).absolute()
    resolved_destination = destination.resolve()
    if not source.is_dir():
        raise AssetError('Asset directory does not exist')
    if destination.exists() or destination.is_symlink():
        raise AssetError('Output must be a new, nonexistent directory')
    inputs = [source] + [Path(p).resolve() for p in extra_inputs if p]
    if any(resolved_destination == p or resolved_destination.is_relative_to(p) or p.is_relative_to(resolved_destination) for p in inputs):
        raise AssetError('Output must not overlap input files or directories')
    # Validate immutable source assets before allocating a snapshot.
    FairseqCodec(source)
    inventory = file_inventory(source)
    parent = read_object(source/'repair_manifest.json') if (source/'repair_manifest.json').is_file() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        for name in ASSET_FILES:
            if (source/name).is_file():
                shutil.copy2(source/name, stage/name)
        for path in weight_files(source):
            relative = path.relative_to(source)
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        manifest = {'schema_version': 1, 'kind': 'onenmt_candidate_change', 'operation': operation,
                    'source': inventory, 'parent_change': parent, 'changes': {},
                    'status': 'requires_validation', 'architecture_reimplemented': False,
                    'requires_embedding_training': bool(parent and parent.get('requires_embedding_training')),
                    'next_checks': ['bhaashik preflight against immutable original assets',
                                    'bhaashik hf-regression on retained and adaptation probes',
                                    'qualified speaker review; train new embeddings when added'],
                    'limits': 'A successful transformation is not evidence of translation quality or learned new-language support.'}
        yield stage, manifest
        FairseqCodec(stage)  # Reject unreadable output dictionaries/SPM before promotion.
        manifest['result'] = file_inventory(stage)
        write_json(stage/'repair_manifest.json', manifest)
        if destination.exists() or destination.is_symlink():
            raise AssetError('Output appeared during the operation; refusing replacement')
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def require_shared_dictionary(codec):
    if codec.dictionary['src'] != codec.dictionary['tgt']:
        raise AssetError('Vocabulary mutation requires identical src/tgt dictionaries for this shared-embedding HF MBart path; differing dictionaries need upstream validation, not automatic reconciliation')


def active_spm_pieces(codec):
    return [codec.sp.id_to_piece(i) for i in range(codec.sp.get_piece_size())
            if not codec.sp.is_control(i) and not codec.sp.is_unknown(i) and not codec.sp.is_unused(i)]
