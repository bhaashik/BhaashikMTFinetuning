import csv
import json
from pathlib import Path
import pytest
from bhaashik.data import audit, read_jsonl
from bhaashik.evaluation import evaluate, export_errors, triage


def save(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return path


def rows():
    return [{'id':str(i),'source':f'source {i}','target':f'the correct reference sentence number {i}',
             'prediction':f'the correct reference sentence number {i}','document_id':f'doc{i}',
             'domain':'news'} for i in range(4)]


def test_metrics_and_paired_alignment(tmp_path):
    candidate = rows()
    baseline = [{**r,'prediction':'wrong translation'} for r in reversed(candidate)]
    a,b = save(tmp_path/'a.jsonl',candidate),save(tmp_path/'b.jsonl',baseline)
    out = evaluate(a,tmp_path/'evaluation.json',b,samples=20)
    assert out['scores']['bleu'] == pytest.approx(100)
    assert out['scores']['chrfpp'] == pytest.approx(100)
    assert 'version:2.4.3' in out['scores']['signatures']['bleu']
    assert out['paired_bootstrap']['unit']=='document'
    assert out['paired_bootstrap']['delta_candidate_minus_baseline']['chrfpp']['ci95'][0] > 0
    baseline[0]['target']='different reference'
    save(b, baseline)
    with pytest.raises(ValueError,match='mismatch'):
        evaluate(a,tmp_path/'bad.json',b,samples=2)


def test_empty_predictions_are_scored(tmp_path):
    path=save(tmp_path/'empty.jsonl',[{**r,'prediction':''} for r in rows()])
    assert evaluate(path,tmp_path/'out.json')['scores']['chrfpp']==0


def test_split_leakage_and_unicode(tmp_path):
    a=save(tmp_path/'train.jsonl',[{'id':'a','source':'café','target':'x','document_id':'doc'}])
    b=save(tmp_path/'dev.jsonl',[{'id':'b','source':'cafe\u0301','target':'y'}])
    with pytest.raises(ValueError,match='source leakage'):
        audit({'train':a,'dev':b})
    save(b,[{'id':'b','source':'different','target':'y','document_id':'doc'}])
    with pytest.raises(ValueError,match='document leakage'):
        audit({'train':a,'dev':b})
    save(b,[{'id':'b','source':'different','target':'y','document_id':'other'}])
    assert audit({'train':a,'dev':b})['counts']=={'train':1,'dev':1}


def test_duplicate_ids_and_empty_file(tmp_path):
    path=save(tmp_path/'duplicate.jsonl',[rows()[0],rows()[0]])
    with pytest.raises(ValueError,match='duplicate id'):
        read_jsonl(path)
    path.write_text('')
    with pytest.raises(ValueError,match='empty dataset'):
        read_jsonl(path)


def test_review_export_and_numerals(tmp_path):
    path=save(tmp_path/'predictions.jsonl',rows())
    export_errors(path,tmp_path/'annotations.csv')
    with open(tmp_path/'annotations.csv') as f:
        sheet=list(csv.DictReader(f))
    assert len(sheet)==4 and {r['id'] for r in sheet}=={'0','1','2','3'}
    assert all(r['annotator']=='' and r['error_category']=='' for r in sheet)
    assert 'numeral_mismatch' not in triage({'source':'१२ लोग','prediction':'12 लोग','target':'12 लोग'})
    assert 'numeral_mismatch' in triage({'source':'१२ लोग','prediction':'13 लोग','target':'12 लोग'})


def test_mixed_bootstrap_units_rejected(tmp_path):
    a=rows()
    del a[0]['document_id']
    p=save(tmp_path/'p.jsonl',a)
    with pytest.raises(ValueError,match='all rows or none'):
        evaluate(p,tmp_path/'out.json',p,samples=2)
