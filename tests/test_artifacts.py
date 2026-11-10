import ast
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
        self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        if tag in ('a','script','link'):
            self.links.append(attrs.get('href',attrs.get('src','')))


def test_built_site_links_and_anchors(tmp_path):
    spec=importlib.util.spec_from_file_location('build_site',ROOT/'scripts/build_site.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    site=module.build(tmp_path/'site')
    for page in site.glob('*.html'):
        parser=Links(); parser.feed(page.read_text())
        for href in parser.links:
            parts=urlparse(href)
            if parts.scheme or parts.netloc: continue
            target=site/unquote(parts.path) if parts.path else page
            assert target.is_file(), (page,href)
            if parts.fragment and target.suffix=='.html':
                other=Links();other.feed(target.read_text())
                assert parts.fragment in other.ids,(page,href)


def test_notebook_cells_are_valid_python():
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        nb=json.loads(path.read_text())
        assert nb['nbformat']==4
        code=[c for c in nb['cells'] if c['cell_type']=='code']
        assert len(code)>=7
        for cell in code:
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[]


def test_example_data_audit():
    from bhaashik.data import audit
    result=audit({s:ROOT/f'data/examples/{s}.jsonl' for s in ('train','dev','test')})
    assert result['counts']=={'train':2,'dev':2,'test':2}
