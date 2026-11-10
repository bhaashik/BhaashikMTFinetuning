"""Browser smoke checks. Build first; serves the artifact on an ephemeral local port."""
import functools
import http.server
import json
import os
import tempfile
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright
from bhaashik.data import read_jsonl
from bhaashik.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT/'build/site'))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}'
    screenshots = ROOT/'build/browser'
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as directory, sync_playwright() as p:
            scratch = Path(directory)
            options = {'headless': True}
            if os.environ.get('BHAASHIK_CHROMIUM'):
                options['executable_path'] = os.environ['BHAASHIK_CHROMIUM']
            browser = p.chromium.launch(**options)
            page = browser.new_page(viewport={'width':1440, 'height':1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url, wait_until='networkidle')
            page.screenshot(path=str(screenshots/'desktop.png'), full_page=True)
            page.select_option('#language', 'mag')
            page.select_option('#bridge', 'eng')
            page.select_option('#direction', 'from')
            page.select_option('#seed', '13')
            with page.expect_download() as download:
                page.get_by_role('button', name='Download configuration').click()
            config = json.loads(Path(download.value.path()).read_text())
            assert config['src_lang'] == 'eng_Latn' and config['tgt_lang'] == 'mag_Deva'
            assert config['seed'] == 13 and 'eng_Latn' in page.locator('#eval-command').inner_text()
            rows = read_jsonl(ROOT/'data/examples/test.jsonl')
            predictions = scratch/'predictions.jsonl'
            predictions.write_text(''.join(json.dumps({**r, 'prediction':r['target']})+'\n' for r in rows))
            report_path = scratch/'evaluation.json'
            report = evaluate(predictions, report_path)
            page.locator('#report-file').set_input_files(str(report_path))
            page.wait_for_function("document.getElementById('chrf-score').textContent === '100.00'")
            assert page.locator('#sample-count').inner_text() == '2'
            report['scores']['signatures']['bleu'] = '<img src=x onerror=alert(1)>'
            unsafe = scratch/'unsafe.json'
            unsafe.write_text(json.dumps(report))
            page.locator('#report-file').set_input_files(str(unsafe))
            page.wait_for_function("document.getElementById('report-state').textContent === 'unsafe.json'")
            assert page.locator('#report-metadata img').count() == 0
            page.locator('#report-file').set_input_files({
                'name':'bad.json', 'mimeType':'application/json', 'buffer':b'{"scores":{}}'})
            page.wait_for_function("document.getElementById('report-state').textContent === 'Invalid report'")
            assert page.locator('#chrf-score').inner_text() == '—'
            page.goto(url+'/research.html')
            assert page.get_by_role('heading', name='Research protocol', exact=True).count() == 1
            page.set_viewport_size({'width':390, 'height':844})
            page.goto(url, wait_until='networkidle')
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            page.screenshot(path=str(screenshots/'mobile.png'), full_page=True)
            assert not errors, errors
            browser.close()
            print('Passed: configuration download, direction/seed selection, report import/rejection, safe text rendering, documentation and mobile layout.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
