"""Candidate mutations tested with genuine tiny HF MBart checkpoints, never research weights."""
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
import sentencepiece as spm
import torch
from transformers import MBartConfig, MBartForConditionalGeneration
from safetensors.torch import save_file
from bhaashik.data import file_hash, write_json
from bhaashik.preflight import AssetError, FairseqCodec, checkpoint_layout, preflight
from bhaashik.repairs.generation import repair_generation
from bhaashik.repairs.dictionary import repair_dictionary, repair_append
from bhaashik.repairs.sentencepiece import repair_spm_extend, repair_spm_train, repair_spm_replace
from bhaashik.repairs.embeddings import repair_resize


@pytest.fixture
def original(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(13)
    root=tmp_path/'original';root.mkdir()
    text=tmp_path/'spm.txt'
    text.write_text(('hello world abcdefghijklmnopqrstuvwxyz ###eng_Latn-to-hin_Deva### '
                     '###bho_Deva-to-hin_Deva### हम घर जात बानी मौसम किताब नीक बा\n')*15)
    spm.SentencePieceTrainer.train(input=str(text),model_prefix=str(root/'onemtv3b_spm'),
        model_type='char',vocab_size=256,hard_vocab_limit=False,bos_id=0,pad_id=1,eos_id=2,unk_id=3,minloglevel=2)
    sp=spm.SentencePieceProcessor(model_file=str(root/'onemtv3b_spm.model'))
    pieces=[sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    mapping={p:i for i,p in enumerate(pieces[:4]+list(reversed(pieces[4:])))}
    write_json(root/'fairseq_dict.json',{'src':mapping,'tgt':mapping,'special':{'bos':0,'pad':1,'eos':2,'unk':3}})
    config=MBartConfig(vocab_size=len(mapping)+8,d_model=8,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=16,decoder_ffn_dim=16,
        max_position_embeddings=128,bos_token_id=0,pad_token_id=1,eos_token_id=2,
        decoder_start_token_id=2,forced_eos_token_id=None,dropout=0,attention_dropout=0)
    MBartForConditionalGeneration(config).save_pretrained(root,safe_serialization=True)
    return root


def snapshot(root):
    return {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()}


def select(tmp_path,values):
    path=tmp_path/'tokens.json';write_json(path,{'src':values,'tgt':values});return path


def test_generation_patch_only_changes_known_config_fields(original,tmp_path):
    config=json.loads((original/'config.json').read_text())
    config['decoder_start_token_id']=0;config['forced_bos_token_id']=99
    write_json(original/'config.json',config)
    write_json(original/'generation_config.json',{'decoder_start_token_id':0,'num_beams':5,'repetition_penalty':1.3})
    before=snapshot(original)
    output=tmp_path/'generation-fixed';manifest=repair_generation(original,output)
    after=json.loads((output/'config.json').read_text())
    assert after['decoder_start_token_id']==2 and after['forced_bos_token_id'] is None
    assert all(after[k]==config[k] for k in ('d_model','encoder_layers','decoder_layers','max_position_embeddings','vocab_size'))
    generation=json.loads((output/'generation_config.json').read_text())
    assert generation['num_beams']==5 and generation['repetition_penalty']==1.3
    assert file_hash(output/'model.safetensors')==before['model.safetensors']
    assert snapshot(original)==before
    assert manifest['status']=='requires_validation'


def test_dictionary_repair_uses_trusted_checkpoint_not_reordered_weights(original,tmp_path):
    trusted=FairseqCodec(original)
    proposed=copy.deepcopy(trusted.dictionary)
    keys=[p for p in proposed['src'] if p not in ('<s>','</s>','<pad>','<unk>')]
    a,b=keys[:2]
    proposed['src'][a],proposed['src'][b]=proposed['src'][b],proposed['src'][a]
    del proposed['tgt'][a]
    proposed['tgt']['novel']=100000
    path=tmp_path/'bad-dictionary.json';write_json(path,proposed)
    before=snapshot(original)
    output=tmp_path/'dictionary-fixed';result=repair_dictionary(original,path,output)
    assert FairseqCodec(output).dictionary==trusted.dictionary
    assert result['changes']['sides']['src']['restored_or_corrected']==2
    assert 'novel' in result['changes']['sides']['tgt']['discarded_proposed_additions']
    assert file_hash(output/'model.safetensors')==file_hash(original/'model.safetensors')
    assert snapshot(original)==before


def test_dictionary_append_coordinates_source_target_and_does_not_touch_weights(original,tmp_path):
    path=tmp_path/'tokens.json';write_json(path,{'src':['ΩΩ'],'tgt':['ΨΨ']})
    before=snapshot(original);old=FairseqCodec(original).dictionary['src']
    output=tmp_path/'appended';result=repair_append(original,output,path)
    codec=FairseqCodec(output)
    assert codec.dictionary['src']==codec.dictionary['tgt']
    assert codec.dictionary['src']['ΩΩ']!=codec.dictionary['src']['ΨΨ']
    assert all(codec.dictionary['src'][p]==i for p,i in old.items())
    assert all(x['id']>max(old.values()) for x in result['changes']['added'])
    assert result['requires_embedding_training']
    assert snapshot(original)==before
    assert file_hash(output/'model.safetensors')==file_hash(original/'model.safetensors')


def test_spm_extension_preserves_old_native_ids_and_activates_new_piece(original,tmp_path):
    tokens=select(tmp_path,['ΩΩ'])
    before=snapshot(original);base=FairseqCodec(original)
    output=tmp_path/'spm-extended';result=repair_spm_extend(original,tokens,output)
    codec=FairseqCodec(output)
    assert 'ΩΩ' in codec.pieces('hello ΩΩ world')
    assert all(codec.sp.id_to_piece(i)==base.sp.id_to_piece(i) for i in range(base.sp.get_piece_size()))
    assert codec.dictionary==base.dictionary
    assert result['changes']['normalizer_preserved']
    assert snapshot(original)==before
    assert codec.encode('ΩΩ').count(codec.special['unk'])>0  # Separate dictionary stage is still required.


@pytest.mark.parametrize('tied',[True,False])
def test_extension_append_resize_preserves_old_weights_and_reloads(original,tmp_path,tied):
    if not tied:
        model=MBartForConditionalGeneration.from_pretrained(original)
        model.config.tie_word_embeddings=False
        model.lm_head=torch.nn.Linear(8,model.config.vocab_size,bias=False)
        model.save_pretrained(original,safe_serialization=True)
    before=snapshot(original)
    tokens=select(tmp_path,['ΩΩ'])
    extended=tmp_path/'extended';repair_spm_extend(original,tokens,extended)
    appended=tmp_path/'appended';repair_append(extended,appended,tokens)
    final=tmp_path/'resized';result=repair_resize(appended,original,final,initialization='mean')
    old_model=MBartForConditionalGeneration.from_pretrained(original)
    new_model=MBartForConditionalGeneration.from_pretrained(final)
    codec=FairseqCodec(final);added=codec.dictionary['src']['ΩΩ']
    used=sorted(FairseqCodec(original).dictionary['src'].values())
    assert torch.equal(old_model.get_input_embeddings().weight[used],new_model.get_input_embeddings().weight[used])
    assert torch.equal(old_model.get_output_embeddings().weight[used],new_model.get_output_embeddings().weight[used])
    assert torch.allclose(new_model.get_input_embeddings().weight[added],old_model.get_input_embeddings().weight[used].mean(0))
    assert torch.allclose(new_model.get_output_embeddings().weight[added],old_model.get_output_embeddings().weight[used].mean(0))
    assert result['changes']['old_model_rows']==result['changes']['new_model_rows']  # Semantic additions used previously padded rows.
    assert result['changes']['save_reload_verified']
    assert (new_model.get_input_embeddings().weight.data_ptr()==new_model.get_output_embeddings().weight.data_ptr())==tied
    for name,value in old_model.state_dict().items():
        if not any(term in name for term in ('shared.weight','embed_tokens.weight','lm_head.weight','final_logits_bias')):
            assert torch.equal(value,new_model.state_dict()[name]),name
    assert snapshot(original)==before
    assert not checkpoint_layout(codec)['issues']


def test_resize_grows_tables_and_removes_stale_shard_index(original,tmp_path):
    model=MBartForConditionalGeneration.from_pretrained(original)
    (original/'model.safetensors').unlink()
    model.save_pretrained(original,safe_serialization=True,max_shard_size='1KB')
    assert (original/'model.safetensors.index.json').exists()
    before=snapshot(original)
    tokens=select(tmp_path,[f'newpiece{i}' for i in range(12)])
    appended=tmp_path/'appended';repair_append(original,appended,tokens)
    final=tmp_path/'grown';result=repair_resize(appended,original,final,initialization='random',seed=17)
    assert result['changes']['new_model_rows']>result['changes']['old_model_rows']
    assert not (final/'model.safetensors.index.json').exists()
    assert list(final.glob('*.safetensors'))==[final/'model.safetensors']
    reloaded=MBartForConditionalGeneration.from_pretrained(final)
    assert reloaded.config.vocab_size==max(FairseqCodec(final).dictionary['src'].values())+1
    assert snapshot(original)==before


def test_resize_rejects_missing_model_parameters(original,tmp_path):
    model=MBartForConditionalGeneration.from_pretrained(original)
    state=model.state_dict()
    del state['model.encoder.layers.0.self_attn.q_proj.weight']
    # Clone tied tensors to produce an independently serialized deliberately incomplete checkpoint.
    save_file({k:v.clone().contiguous() for k,v in state.items()},original/'model.safetensors',metadata={'format':'pt'})
    with pytest.raises(AssetError,match='incomplete'):
        repair_resize(original,original,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_retrain_and_replace_keep_original_dictionary_order(original,tmp_path):
    text=tmp_path/'mixed.txt';text.write_text(('hello world hello world हम घर जात बानी ΩΩ\n')*30)
    before=snapshot(original)
    trained=tmp_path/'trained';result=repair_spm_train(original,text,trained,vocab_size=80,model_type='bpe')
    assert FairseqCodec(trained).dictionary==FairseqCodec(original).dictionary
    assert result['changes']['weights_unchanged']
    assert file_hash(trained/'onemtv3b_spm.model')!=file_hash(original/'onemtv3b_spm.model')
    replaced=tmp_path/'replaced';repair_spm_replace(original,trained/'onemtv3b_spm.model',replaced)
    assert FairseqCodec(replaced).pieces('hello world')==FairseqCodec(trained).pieces('hello world')
    appended=tmp_path/'synced';repair_append(replaced,appended,from_spm=True)
    final=tmp_path/'ready-for-training';repair_resize(appended,original,final)
    assert not checkpoint_layout(FairseqCodec(final))['issues']
    assert snapshot(original)==before


def test_repair_failure_rolls_back_only_its_temporary_directory(original,tmp_path):
    text=tmp_path/'too-many-characters.txt';text.write_text('abcdefghijklmnopqrstuvwxyz\n')
    before=snapshot(original)
    with pytest.raises((RuntimeError,ValueError)):
        repair_spm_train(original,text,tmp_path/'failed',vocab_size=8)
    assert not (tmp_path/'failed').exists()
    assert not list(tmp_path.glob('.failed-*'))
    assert snapshot(original)==before


def test_existing_and_overlapping_outputs_are_refused(original,tmp_path):
    before=snapshot(original)
    with pytest.raises(AssetError,match='new, nonexistent'):
        repair_generation(original,original)
    with pytest.raises(AssetError,match='overlap'):
        repair_generation(original,original/'nested')
    output=tmp_path/'existing';output.mkdir();(output/'user.txt').write_text('keep me')
    with pytest.raises(AssetError,match='new, nonexistent'):
        repair_generation(original,output)
    assert (output/'user.txt').read_text()=='keep me'
    assert snapshot(original)==before


def test_shared_mbart_mutations_reject_inconsistent_dictionaries(original,tmp_path):
    mapping=json.loads((original/'fairseq_dict.json').read_text())
    keys=[p for p in mapping['tgt'] if p not in ('<s>','</s>','<pad>','<unk>')]
    a,b=keys[:2];mapping['tgt'][a],mapping['tgt'][b]=mapping['tgt'][b],mapping['tgt'][a]
    write_json(original/'fairseq_dict.json',mapping)
    with pytest.raises(AssetError,match='identical src/tgt'):
        repair_append(original,tmp_path/'no',select(tmp_path,['ΩΩ']))


def test_cli_repair_success_and_non_overwrite_exit(original,tmp_path):
    output=tmp_path/'cli-candidate'
    command=[sys.executable,'-m','bhaashik.cli','repair-generation','--assets',str(original),'--output',str(output)]
    success=subprocess.run(command,capture_output=True,text=True)
    assert success.returncode==0,success.stderr
    assert json.loads(success.stdout)['status']=='requires_validation'
    assert (output/'repair_manifest.json').exists()
    failure=subprocess.run(command,capture_output=True,text=True)
    assert failure.returncode==2
    assert json.loads(failure.stderr)['status']=='failed'


def test_pending_training_marker_survives_later_generation_patch(original,tmp_path):
    tokens=select(tmp_path,['hello'])
    appended=tmp_path/'appended';repair_append(original,appended,tokens)
    resized=tmp_path/'resized';repair_resize(appended,original,resized)
    patched=tmp_path/'patched';repair_generation(resized,patched)
    assert json.loads((patched/'repair_manifest.json').read_text())['requires_embedding_training']
    probes=tmp_path/'probes.jsonl'
    probes.write_text(json.dumps({'id':'retain','source':'hello','target':'हम','src_lang':'eng_Latn','tgt_lang':'hin_Deva','group':'retention'})+'\n')
    result=preflight(patched,probes,tmp_path/'report.json')
    assert result['original']['repair_state']['requires_embedding_training']
    assert any('requiring training' in x for x in result['review_findings'])


def test_resize_is_seeded_and_huge_id_allocation_is_refused(original,tmp_path):
    tokens=select(tmp_path,['hello'])
    appended=tmp_path/'appended';repair_append(original,appended,tokens)
    first,second=tmp_path/'first',tmp_path/'second'
    repair_resize(appended,original,first,initialization='random',seed=19)
    repair_resize(appended,original,second,initialization='random',seed=19)
    a=MBartForConditionalGeneration.from_pretrained(first)
    b=MBartForConditionalGeneration.from_pretrained(second)
    assert torch.equal(a.get_input_embeddings().weight,b.get_input_embeddings().weight)
    dictionary=json.loads((appended/'fairseq_dict.json').read_text())
    dictionary['src']['hello']=1000000000;dictionary['tgt']['hello']=1000000000
    write_json(appended/'fairseq_dict.json',dictionary)
    with pytest.raises(AssetError,match='allocation exceeds'):
        repair_resize(appended,original,tmp_path/'too-large')
    assert not (tmp_path/'too-large').exists()


def test_resize_refuses_disagreeing_serialized_shared_aliases(original,tmp_path):
    model=MBartForConditionalGeneration.from_pretrained(original)
    state={k:v.clone().contiguous() for k,v in model.state_dict().items()}
    state['model.encoder.embed_tokens.weight'][4,0]+=1
    save_file(state,original/'model.safetensors',metadata={'format':'pt'})
    with pytest.raises(AssetError,match='aliases disagree'):
        repair_resize(original,original,tmp_path/'must-not-tie-silently')
    assert not (tmp_path/'must-not-tie-silently').exists()
