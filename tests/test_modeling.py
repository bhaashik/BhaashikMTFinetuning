"""Real offline training and HF reload tests; these are not quality benchmarks."""
import json
import torch
import pytest
import sentencepiece as spm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100Config, M2M100ForConditionalGeneration, NllbTokenizer
from bhaashik.modeling import train, translate, extend, require_language
from bhaashik.data import read_jsonl


@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('model')
    corpus=root/'corpus.txt'
    corpus.write_text('hello world the book is here\nहम घर जात बानी किताब मौसम नीक बा\n'*20)
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(root/'spm'),vocab_size=48,
                                  hard_vocab_limit=False,bos_id=0,pad_id=1,eos_id=2,unk_id=3,
                                  minloglevel=2)
    tokenizer=NllbTokenizer(vocab_file=str(root/'spm.model'),src_lang='bho_Deva',tgt_lang='hin_Deva')
    config=M2M100Config(vocab_size=len(tokenizer),d_model=16,encoder_layers=1,decoder_layers=1,
                        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,
                        decoder_ffn_dim=32,max_position_embeddings=64,pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.eos_token_id)
    model=M2M100ForConditionalGeneration(config)
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


@pytest.mark.parametrize('mode',['lora','full'])
def test_train_export_reload_and_generate(tiny_model,tmp_path,mode):
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    output=tmp_path/'trained'
    cfg={'model':str(tiny_model),'src_lang':'bho_Deva','tgt_lang':'hin_Deva',
         'train_file':str(root/'data/examples/train.jsonl'),'dev_file':str(root/'data/examples/dev.jsonl'),
         'output_dir':str(output),'mode':mode,'max_steps':1,'max_length':24,'precision':'fp32',
         'batch_size':1,'gradient_accumulation_steps':1,'gradient_checkpointing':True}
    config=tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    result=train(config)
    assert result['eval_loss'] > 0
    model=AutoModelForSeq2SeqLM.from_pretrained(output)
    assert not any('lora_' in name for name,_ in model.named_parameters())
    tokenizer=AutoTokenizer.from_pretrained(output)
    assert require_language(tokenizer,'bho_Deva') != tokenizer.unk_token_id
    assert (output/'run_manifest.json').exists()
    translate(output,'bho_Deva','hin_Deva',root/'data/examples/test.jsonl',tmp_path/'pred.jsonl',
              max_length=24,max_new_tokens=4,beams=1)
    assert len(read_jsonl(tmp_path/'pred.jsonl',predictions=True))==2
    with pytest.raises(ValueError,match='nonempty'):
        train(config)


def test_extension_preserves_tags_and_shared_embeddings(tiny_model,tmp_path):
    output=tmp_path/'extended'
    original=AutoTokenizer.from_pretrained(tiny_model)
    extend(tiny_model,output,'raj_Deva')
    tokenizer=AutoTokenizer.from_pretrained(output)
    model=AutoModelForSeq2SeqLM.from_pretrained(output)
    assert len(tokenizer)==len(original)+1
    for tag in ('bho_Deva','mai_Deva','mag_Deva','hin_Deva'):
        assert require_language(tokenizer,tag)==require_language(original,tag)
    new,donor=require_language(tokenizer,'raj_Deva'),require_language(tokenizer,'hin_Deva')
    assert torch.equal(model.get_input_embeddings().weight[new],model.get_input_embeddings().weight[donor])
    tokenizer.src_lang='raj_Deva'
    tokenizer.tgt_lang='hin_Deva'
    assert new in tokenizer('हम घर')['input_ids']
    tokenizer.tgt_lang='raj_Deva'
    assert new in tokenizer(text_target='हम घर')['input_ids']
    with pytest.raises(ValueError,match='already exists'):
        extend(tiny_model,tmp_path/'bad','mag_Deva')


def test_extended_tag_training_updates_embedding(tiny_model,tmp_path):
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    extended=tmp_path/'extended'
    extend(tiny_model,extended,'raj_Deva')
    tokenizer=AutoTokenizer.from_pretrained(extended)
    tag=require_language(tokenizer,'raj_Deva')
    before=AutoModelForSeq2SeqLM.from_pretrained(extended).get_input_embeddings().weight[tag].detach().clone()
    # These fixture texts test mechanics only, not the identity/quality of Rajasthani data.
    config={'model':str(extended),'src_lang':'bho_Deva','tgt_lang':'raj_Deva',
            'train_file':str(root/'data/examples/train.jsonl'),'dev_file':str(root/'data/examples/dev.jsonl'),
            'output_dir':str(tmp_path/'trained'),'mode':'lora','max_steps':1,'max_length':24,
            'precision':'fp32','batch_size':1,'gradient_accumulation_steps':1}
    path=tmp_path/'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='require mode=full'):
        train(path)
    config['mode']='full'
    path.write_text(json.dumps(config))
    train(path)
    model=AutoModelForSeq2SeqLM.from_pretrained(tmp_path/'trained')
    assert not torch.equal(before,model.get_input_embeddings().weight[tag])
    assert (tmp_path/'trained/bhaashik_extension.json').exists()
