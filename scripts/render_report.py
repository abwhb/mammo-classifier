"""Render REPORT.md to reports/REPORT.pdf via markdown→HTML→Chromium headless.

Usage:
    uv run --with markdown --with pygments --with playwright --no-project \
        python scripts/render_report.py

Why not pandoc? Pandoc PDF output needs a TeX engine (basictex / mactex), which
is ~80 MB and slow to install on a fresh machine. Chromium-headless ships with
the dev tooling we already use for end-to-end testing, so we reuse it.
"""

from __future__ import annotations

import base64
import mimetypes
import pathlib
import re
import sys

import markdown
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
HTML_OUT = ROOT / "reports" / "REPORT.html"
PDF_OUT = ROOT / "reports" / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 0.8in 0.7in; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif; font-size: 10.5pt;
       color: #1a1a1a; line-height: 1.55; max-width: 100%; margin: 0; }
h1 { font-size: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 4px;
     margin-top: 1.4em; page-break-after: avoid; }
h2 { font-size: 1.25em; margin-top: 1.2em; page-break-after: avoid; }
h3 { font-size: 1.08em; margin-top: 1em; page-break-after: avoid; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 0.92em;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #f4f4f6; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
       font-size: 0.92em; background: #f4f4f6; padding: 1px 4px; border-radius: 3px; }
pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em; background: #f6f6f8; padding: 10px; border-radius: 6px;
      overflow-x: auto; line-height: 1.3; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #b00; padding: 0.5em 1em; color: #555;
             margin: 1em 0; background: #fff4f4; }
a { color: #06c; text-decoration: none; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
img { max-width: 70%; display: block; margin: 1em auto; }
.title-block { text-align: center; border-bottom: 2px solid #333;
               padding-bottom: 1em; margin-bottom: 1.5em; }
.title-block .title { font-size: 1.8em; font-weight: 700; margin: 0; }
.title-block .subtitle { font-size: 1.1em; color: #555; margin-top: 0.3em; }
.title-block .meta { color: #777; margin-top: 0.6em; font-size: 0.95em; }
""".strip()


def inline_local_images(html: str, base: pathlib.Path) -> str:
    """Embed `<img src="local/path">` as `data:` URIs so the PDF is self-contained."""
    def replace(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (base / src).resolve()
        if not path.exists():
            return match.group(0)
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f'<img src="data:{mime};base64,{b64}"'

    return re.sub(r'<img\s+[^>]*?src="([^"]+)"', replace, html)


def strip_yaml_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Pull out the YAML-style frontmatter (between two `---` lines) so it
    doesn't get rendered as a paragraph of raw text."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}, text
    front: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            front[k.strip()] = v.strip().strip('"')
    return front, text[m.end():]


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")
    raw = SOURCE.read_text()
    meta, body_md = strip_yaml_frontmatter(raw)
    body_html = markdown.markdown(
        body_md, extensions=["fenced_code", "tables", "codehilite", "attr_list"]
    )

    body_html = inline_local_images(body_html, ROOT)
    title = meta.get("title", "Technical Report")
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", "")
    date = meta.get("date", "")
    title_block = f"""
    <div class="title-block">
      <p class="title">{title}</p>
      {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
      <p class="meta">{author} · {date}</p>
    </div>
    """.strip()

    html = (
        f"<html><head><meta charset='utf-8'><title>{title}</title>"
        f"<style>{CSS}</style></head><body>{title_block}{body_html}</body></html>"
    )
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html)
    print(f"wrote {HTML_OUT}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{HTML_OUT}")
        page.pdf(
            path=str(PDF_OUT),
            format="A4",
            margin={"top": "0.7in", "bottom": "0.7in", "left": "0.6in", "right": "0.6in"},
            print_background=True,
        )
        browser.close()
    print(f"wrote {PDF_OUT}  ({PDF_OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
