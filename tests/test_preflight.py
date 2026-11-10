"""Synthetic asset tests exercise mappings, drift and gates without research weights."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import sentencepiece as spm
from safetensors.numpy import save_file
from bhaashik.data import write_json
from bhaashik.preflight import AssetError, FairseqCodec, checkpoint_layout, preflight


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    return path


@pytest.fixture
def assets(tmp_path):
    root=tmp_path/'original';root.mkdir()
    corpus=tmp_path/'spm-training.txt'
    corpus.write_text(('###hin_Deva-to-mai_Deva### ###bho_Deva-to-hin_Deva### ###eng_Latn-to-hin_Deva### '
                       'hello world book new data abcdefghijklmnopqrstuvwxyz '
                       'हम घर जात बानी किताब मौसम नीक बा रउआ कइसन बानी आज दुनिया नमस्ते।\n')*12)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root/'onemtv3b_spm'),
        model_type='char', vocab_size=256, hard_vocab_limit=False,
        bos_id=0, pad_id=1, eos_id=2, unk_id=3, minloglevel=2)
    sp=spm.SentencePieceProcessor(model_file=str(root/'onemtv3b_spm.model'))
    pieces=[sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    # Deliberately reorder IDs relative to SPM and make src/tgt disagree.
    src={piece:i for i,piece in enumerate(pieces[:4]+list(reversed(pieces[4:])))}
    tgt={piece:i for i,piece in enumerate(pieces)}
    write_json(root/'fairseq_dict.json',{'src':src,'tgt':tgt,'special':{'bos':0,'pad':1,'eos':2,'unk':3}})
    write_json(root/'config.json',{'model_type':'mbart','vocab_size':len(pieces),'d_model':8,
        'encoder_layers':1,'decoder_layers':1,'max_position_embeddings':512,
        'bos_token_id':0,'pad_token_id':1,'eos_token_id':2,'decoder_start_token_id':2})
    save_file({'model.shared.weight':np.zeros((len(pieces),8),dtype='float32')},root/'model.safetensors')
    return root


@pytest.fixture
def probes(tmp_path):
    return jsonl(tmp_path/'probes.jsonl',[
        {'id':'retain-1','source':'hello world','target':'हम घर','src_lang':'eng_Latn','tgt_lang':'hin_Deva','group':'retention'},
        {'id':'adapt-1','source':'हम घर जात बानी','target':'हम घर','src_lang':'bho_Deva','tgt_lang':'hin_Deva','group':'adaptation'}])


def test_codec_uses_dictionary_not_spm_and_target_is_independent(assets):
    codec=FairseqCodec(assets)
    text='hello'
    pieces=codec.sp.encode(text,out_type=str)
    assert codec.encode(text)==[codec.dictionary['src'][p] for p in pieces]+[2]
    assert codec.encode(text)!=codec.sp.encode(text,out_type=int)+[2]
    assert codec.encode(text,'tgt')!=codec.encode(text,'src')
    assert codec.decode(codec.encode(text,'tgt'))==text
    assert codec.encode_source(text,'eng_Latn','hin_Deva')==codec.encode('###eng_Latn-to-hin_Deva### '+text)


def test_unchanged_assets_pass_structure_but_not_behavior(assets,probes,tmp_path):
    report=preflight(assets,probes,tmp_path/'report.json',candidate_assets=assets)
    assert report['structural_status']=='review_required'
    assert not report['blocking_findings']
    assert any('Shared embedding' in w for w in report['review_findings'])
    assert report['decision']=='behavioral_validation_required'
    assert report['original']['checkpoint']['weight_layout_checked']
    assert report['candidate']['comparison']['segmentation_and_ids']['id_sequences']==0
    assert all(g['unknown_rate']==0 for g in report['original']['coverage']['groups'].values())


def test_reordering_is_blocked_even_without_probe_occurrences(assets,probes,tmp_path):
    candidate=tmp_path/'candidate';shutil.copytree(assets,candidate)
    mapping=json.loads((candidate/'fairseq_dict.json').read_text())
    keys=[p for p in mapping['src'] if p not in ('<s>','</s>','<pad>','<unk>')][-2:]
    a,b=keys;mapping['src'][a],mapping['src'][b]=mapping['src'][b],mapping['src'][a]
    write_json(candidate/'fairseq_dict.json',mapping)
    report=preflight(assets,probes,tmp_path/'report.json',candidate_assets=candidate)
    assert report['structural_status']=='blocked'
    assert report['candidate']['comparison']['dictionaries']['src']['remapped_count']==2


def test_append_plan_is_read_only_and_exposes_inactive_tokens(assets,probes,tmp_path):
    codec=FairseqCodec(assets); before=codec.fingerprint()
    selected=tmp_path/'selected.json';write_json(selected,{'src':['▁nonexistentpiece'],'tgt':['▁novelword']})
    plan=tmp_path/'proposal.json'
    report=preflight(assets,probes,tmp_path/'report.json',selected_tokens=selected,plan_output=plan)
    proposal=json.loads(plan.read_text())
    for side in ('src','tgt'):
        assert all(proposal[side][piece]==idx for piece,idx in codec.dictionary[side].items())
        additions=report['append_plan'][side]['additions']
        assert additions[0]['id']==max(codec.dictionary[side].values())+1
        assert additions[0]['dictionary_only_addition_inactive']
    assert before==FairseqCodec(assets).fingerprint()
    assert report['structural_status']=='review_required'


def test_append_requires_matching_model_rows(assets,probes,tmp_path):
    candidate=tmp_path/'candidate';shutil.copytree(assets,candidate)
    mapping=json.loads((candidate/'fairseq_dict.json').read_text())
    mapping['src']['▁novel']=max(mapping['src'].values())+1
    write_json(candidate/'fairseq_dict.json',mapping)
    report=preflight(assets,probes,tmp_path/'report.json',candidate_assets=candidate)
    assert report['candidate']['comparison']['dictionaries']['src']['old_ids_preserved']
    assert report['structural_status']=='blocked'
    assert any('cannot cover' in text for text in report['blocking_findings'])


def test_filling_old_id_holes_is_not_append_only(assets,probes,tmp_path):
    mapping=json.loads((assets/'fairseq_dict.json').read_text())
    piece=next(p for p,i in mapping['src'].items() if i==6)
    mapping['src'][piece]=max(mapping['src'].values())+1
    write_json(assets/'fairseq_dict.json',mapping)
    config=json.loads((assets/'config.json').read_text());config['vocab_size']+=1;write_json(assets/'config.json',config)
    save_file({'model.shared.weight':np.zeros((config['vocab_size'],8),dtype='float32')},assets/'model.safetensors')
    candidate=tmp_path/'candidate';shutil.copytree(assets,candidate)
    mapping['src']['novel']=6;write_json(candidate/'fairseq_dict.json',mapping)
    report=preflight(assets,probes,tmp_path/'report.json',candidate_assets=candidate)
    assert report['structural_status']=='blocked'
    assert not report['candidate']['comparison']['dictionaries']['src']['additions_after_old_max_id']


def test_replacement_spm_changes_sequences_even_when_ids_preserved(assets,probes,tmp_path):
    candidate=tmp_path/'candidate';shutil.copytree(assets,candidate)
    corpus=tmp_path/'replacement-corpus.txt'
    corpus.write_text(('hello world hello world हम घर जात बानी ###eng_Latn-to-hin_Deva###\n')*80)
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(candidate/'onemtv3b_spm'),
        model_type='bpe',vocab_size=90,hard_vocab_limit=False,bos_id=0,pad_id=1,eos_id=2,unk_id=3,minloglevel=2)
    report=preflight(assets,probes,tmp_path/'report.json',candidate_assets=candidate)
    comparison=report['candidate']['comparison']
    assert comparison['dictionaries']['src']['old_ids_preserved']
    assert comparison['sentencepiece_artifact_changed']
    assert comparison['segmentation_and_ids']['piece_sequences']>0
    assert report['structural_status']=='review_required'
    assert any(g['unknown_ids'] for g in report['candidate']['coverage']['groups'].values())


def test_context_overflow_is_blocked_without_silent_truncation(assets,probes,tmp_path):
    report=preflight(assets,probes,tmp_path/'report.json',max_source_tokens=5)
    assert report['structural_status']=='blocked'
    assert any('context budget' in x for x in report['blocking_findings'])


def test_output_cannot_overwrite_inputs(assets,probes,tmp_path):
    before=(assets/'fairseq_dict.json').read_bytes()
    with pytest.raises(AssetError,match='outside asset'):
        preflight(assets,probes,assets/'fairseq_dict.json')
    with pytest.raises(AssetError,match='outside asset'):
        preflight(assets,probes,probes)
    assert before==(assets/'fairseq_dict.json').read_bytes()


def test_missing_weights_is_unverified_not_ready(assets,probes,tmp_path):
    (assets/'model.safetensors').unlink()
    report=preflight(assets,probes,tmp_path/'report.json')
    assert report['structural_status']=='review_required'
    assert not report['original']['checkpoint']['weight_layout_checked']


def test_duplicate_ids_and_duplicate_json_keys_rejected(assets):
    path=assets/'fairseq_dict.json';mapping=json.loads(path.read_text())
    pieces=[p for p in mapping['src'] if p not in ('<s>','</s>','<pad>','<unk>')]
    mapping['src'][pieces[0]]=mapping['src'][pieces[1]];write_json(path,mapping)
    with pytest.raises(AssetError,match='Duplicate IDs'):
        FairseqCodec(assets)
    path.write_text('{"src":{},"src":{}}')
    with pytest.raises(AssetError,match='Duplicate JSON key'):
        FairseqCodec(assets)


def test_generation_special_mismatch_is_blocked(assets,probes,tmp_path):
    write_json(assets/'generation_config.json',{'decoder_start_token_id':0})
    report=preflight(assets,probes,tmp_path/'report.json')
    assert report['structural_status']=='blocked'


def test_cli_invalid_assets_writes_blocked_report(tmp_path):
    report=tmp_path/'report.json'
    run=subprocess.run([sys.executable,'-m','bhaashik.cli','preflight','--assets',str(tmp_path/'missing'),
                        '--corpus',str(tmp_path/'missing.jsonl'),'--report',str(report)],capture_output=True,text=True)
    assert run.returncode==2
    assert json.loads(report.read_text())['status']=='blocked'


def test_hf_generation_roundtrip_and_no_architecture_reimplementation(assets,probes,tmp_path):
    import torch
    from transformers import MBartConfig, MBartForConditionalGeneration
    from bhaashik.hf_regression import hf_regression
    torch.set_num_threads(1)
    original_config=json.loads((assets/'config.json').read_text())
    config=MBartConfig(vocab_size=original_config['vocab_size'],d_model=8,
        encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
        encoder_ffn_dim=16,decoder_ffn_dim=16,max_position_embeddings=128,
        bos_token_id=0,pad_token_id=1,eos_token_id=2,decoder_start_token_id=2,
        forced_eos_token_id=None,dropout=0,attention_dropout=0)
    model=MBartForConditionalGeneration(config)
    model.save_pretrained(assets,safe_serialization=True)
    candidate=tmp_path/'candidate';shutil.copytree(assets,candidate)
    result=hf_regression(assets,candidate,probes,tmp_path/'regression.json',max_new_tokens=4)
    assert result['status']=='review_required'  # Reference parity is not established by self-agreement.
    assert len(result['baseline']['outputs'])==2
    assert all(group['changed_outputs']==0 for group in result['groups'].values())
    assert result['baseline']['generation']['decoding']['decoder_start_token_id']==2
    assert result['baseline']['generation']['decoding']['forced_bos_token_id'] is None
    assert (assets/'fairseq_dict.json').read_bytes()==(candidate/'fairseq_dict.json').read_bytes()
    # Golden HF replay checks the reference gate's mechanics, not original Fairseq parity.
    reference=jsonl(tmp_path/'reference.jsonl',result['baseline']['outputs'])
    replay=hf_regression(assets,candidate,probes,tmp_path/'replay.json',reference=reference,max_new_tokens=4)
    assert replay['reference_parity']['exact_output_matches']==2
    # An adapted checkpoint with NaNs must fail even if generation would otherwise pick a token.
    with torch.no_grad():
        model.model.shared.weight.fill_(float('nan'))
    model.save_pretrained(candidate,safe_serialization=True)
    failed=hf_regression(assets,candidate,probes,tmp_path/'failed.json',max_new_tokens=4)
    assert failed['status']=='blocked'
    assert 'Nonfinite' in failed['blocking_findings'][0]


def test_retention_decline_cannot_be_hidden_by_adaptation_gain():
    from bhaashik.hf_regression import compare_outputs
    baseline=[{'id':'old','source':'hello','target':'the correct reference sentence',
        'prediction':'the correct reference sentence','src_lang':'eng_Latn','tgt_lang':'hin_Deva','group':'retention'},
        {'id':'new','source':'new input','target':'another correct reference sentence',
        'prediction':'wrong','src_lang':'bho_Deva','tgt_lang':'hin_Deva','group':'adaptation'}]
    candidate=[{**baseline[0],'prediction':'wrong'},{**baseline[1],'prediction':baseline[1]['target']}]
    result=compare_outputs(baseline,candidate)
    assert result['status']=='blocked'
    assert any('retention' in x for x in result['blocking_findings'])
    assert result['groups']['adaptation|bho_Deva->hin_Deva']['chrfpp_delta']>0
    with pytest.raises(AssetError,match='Probe mismatch'):
        compare_outputs(baseline,[{**candidate[0],'target':'modified gold'},candidate[1]])
    with pytest.raises(AssetError,match='finite'):
        compare_outputs(baseline,candidate,float('nan'))


def test_reference_disagreement_is_a_blocker():
    from bhaashik.hf_regression import compare_outputs
    row={'id':'1','source':'input','target':'reference','prediction':'output','group':'retention'}
    result=compare_outputs([row],[row],reference=[{**row,'prediction':'different original output'}])
    assert result['status']=='blocked'
    assert result['reference_parity']['exact_output_matches']==0


def test_shard_paths_cannot_escape_asset_directory(assets):
    write_json(assets/'model.safetensors.index.json',{'weight_map':{'model.shared.weight':'../outside.safetensors'}})
    with pytest.raises(AssetError,match='stay inside'):
        checkpoint_layout(FairseqCodec(assets))


def test_cli_success_and_strict_review_exit_codes(assets,probes,tmp_path):
    command=[sys.executable,'-m','bhaashik.cli','preflight','--assets',str(assets),
             '--corpus',str(probes),'--report',str(tmp_path/'cli.json')]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads((tmp_path/'cli.json').read_text())['decision']=='behavioral_validation_required'
    strict=subprocess.run(command+['--strict'],capture_output=True,text=True)
    assert strict.returncode==2  # Deliberately different src/tgt IDs on shared embeddings need review.


def test_invalid_safetensors_header_is_diagnosed(assets):
    (assets/'model.safetensors').write_bytes(b'not a valid tensor file')
    with pytest.raises(AssetError,match='Invalid safetensors header'):
        checkpoint_layout(FairseqCodec(assets))


def test_direction_prefix_unknown_collapse_is_blocked(assets,tmp_path):
    path=jsonl(tmp_path/'collision.jsonl',[
        {'id':'a','source':'hello','target':'हम','src_lang':'zzz_Zzzz','tgt_lang':'hin_Deva','group':'retention'},
        {'id':'b','source':'hello','target':'हम','src_lang':'zzz_Yzzz','tgt_lang':'hin_Deva','group':'adaptation'}])
    result=preflight(assets,path,tmp_path/'collision-report.json')
    assert result['original']['coverage']['prefix_collisions']
    assert result['structural_status']=='blocked'
