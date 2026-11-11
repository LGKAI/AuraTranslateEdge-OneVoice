# -*- coding: utf-8 -*-
"""Render a Markdown report (with relative images such as docs/pipeline.svg) to HTML, then to PDF with headless Edge / Chrome.
  python tools/md_to_pdf.py docs/aibox8550_report.md docs/aibox8550_report.pdf"""
import os, subprocess, sys, time
import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 20pt; border-bottom: 2px solid #2b5797; padding-bottom: 4pt; }
h2 { font-size: 15pt; color: #2b5797; margin-top: 18pt; border-bottom: 1px solid #ccd; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #b8bec8; padding: 3pt 5pt; vertical-align: top; }
th { background: #e8eef7; text-align: left; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; background: #f2f3f5; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f2f3f5; border: 1px solid #d5d8de; border-radius: 3pt; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #ccd; margin: 12pt 0; }
blockquote { margin: 6pt 0; padding: 2pt 10pt; border-left: 3px solid #2b5797; background: #f6f8fb; }
a { color: #2b5797; text-decoration: none; }
"""


def browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(p): return p
    sys.exit("no Edge / Chrome found")


def main():
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); html_path = os.path.splitext(dst)[0] + ".html"
    body = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables", "fenced_code", "sane_lists"])
    open(html_path, "w", encoding="utf-8").write(f'<!doctype html><html><head><meta charset="utf-8"><title>{os.path.basename(src)}</title><style>{CSS}</style></head><body>{body}</body></html>')
    if os.path.exists(dst): os.remove(dst)
    p = subprocess.Popen([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={dst}", "file:///" + html_path.replace("\\", "/")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        time.sleep(1)
        if os.path.exists(dst) and os.path.getsize(dst) > 0: time.sleep(2); break
    p.terminate()
    print(dst, os.path.getsize(dst) if os.path.exists(dst) else "FAILED")


if __name__ == "__main__": main()
