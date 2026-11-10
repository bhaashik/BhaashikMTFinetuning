"""Build a dependency-light Pages artifact; documentation and portal share styling."""
from pathlib import Path
import argparse
import html
import re
import shutil
import markdown
from urllib.parse import urlsplit, urlunsplit


def build(destination):
    root = Path(__file__).resolve().parents[1]
    source = root / 'docs'
    target = Path(destination).resolve()
    if target == source or source in target.parents or target == root:
        raise ValueError('Build output must be outside docs/ and cannot be the repo root')
    target.mkdir(parents=True, exist_ok=True)
    for name in ('index.html', 'styles.css', 'app.js'):
        shutil.copyfile(source / name, target / name)
    for path in source.glob('*.md'):
        content = markdown.markdown(path.read_text(encoding='utf-8'), extensions=['fenced_code', 'tables', 'toc'])
        def local_guide_link(match):
            parts = urlsplit(html.unescape(match.group(1)))
            if not parts.scheme and not parts.netloc and parts.path.endswith('.md'):
                parts = parts._replace(path=parts.path[:-3] + '.html')
                return 'href="' + html.escape(urlunsplit(parts), quote=True) + '"'
            return match.group(0)
        content = re.sub(r'href="([^"]+)"', local_guide_link, content)
        title = path.read_text(encoding='utf-8').splitlines()[0].lstrip('# ')
        page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)} · Bhaashik</title>
<link rel="stylesheet" href="styles.css"></head><body class="guide-page"><header class="guide-header">
<a class="brand" href="index.html"><span class="brand-mark">भा</span><span>Bhaashik</span></a>
<a class="text-link" href="index.html#protocol">← Research workspace</a></header><article class="guide-content">{content}</article>
<footer><a href="index.html">Bhaashik · Neural MT Lab</a><span>Research toolkit · No model quality claims</span></footer></body></html>'''
        (target / (path.stem + '.html')).write_text(page, encoding='utf-8')
    (target / '.nojekyll').touch()
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='build/site')
    args = parser.parse_args()
    print(build(args.output))
