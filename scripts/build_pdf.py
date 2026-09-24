"""Build report/MIDSEM_REPORT.pdf from report/MIDSEM_REPORT.md (print-styled HTML -> headless Chromium).

  python scripts/build_pdf.py [--chrome /path/to/chrome]
Needs: pip install markdown; any Chromium/Chrome binary.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess

import markdown

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPORT = os.path.join(ROOT, "report")

# the handful of LaTeX expressions in the report, as typeset HTML
MATH = {
    r"$$ w(e) = \max_t \log\big(1 + \mathrm{ReLU}(\alpha\,\langle h_t,\; P\,E_e\rangle)\big) $$":
        '<div class="eq"><i>w</i>(<i>e</i>) = max<sub><i>t</i></sub> log( 1 + ReLU( α · ⟨ <i>h</i><sub><i>t</i></sub> , '
        '<i>P</i> <i>E</i><sub><i>e</i></sub> ⟩ ) )</div>',
    r"$w'(e) = w(e)\cdot\sigma(\mathrm{MLP}([s_e, \text{linked}_e, \text{dense}_e, \text{pop}_e, z_e]))$":
        "<i>w′</i>(<i>e</i>) = <i>w</i>(<i>e</i>) · σ( MLP( [<i>s<sub>e</sub></i>, linked<sub><i>e</i></sub>, "
        "dense<sub><i>e</i></sub>, pop<sub><i>e</i></sub>, <i>z<sub>e</sub></i>] ) )",
}
INLINE = {r"$h_t$": "<i>h<sub>t</sub></i>", r"$t$": "<i>t</i>", r"$E_e$": "<i>E<sub>e</sub></i>",
          r"$P$": "<i>P</i>", r"$\alpha$": "α", r"$s_e$": "<i>s<sub>e</sub></i>", r"$e$": "<i>e</i>",
          r"$z_e$": "<i>z<sub>e</sub></i>"}

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm;
  @bottom-center { content: counter(page); } }
body { font: 10.2pt/1.5 "DejaVu Serif", Georgia, serif; color: #1a1d1b; }
h1 { font: 700 18pt/1.2 "DejaVu Sans", sans-serif; margin: 0 0 2pt; color: #0f3d35; }
h2 { font: 700 13.5pt/1.25 "DejaVu Sans", sans-serif; color: #0f3d35; margin: 18pt 0 6pt;
     border-bottom: 1.2pt solid #0f6b5c; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font: 700 11pt/1.3 "DejaVu Sans", sans-serif; margin: 12pt 0 4pt; page-break-after: avoid; }
h4 { font: 700 10.4pt/1.3 "DejaVu Sans", sans-serif; margin: 10pt 0 3pt; color: #33403a; page-break-after: avoid; }
p, li { text-align: justify; hyphens: auto; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin: 1.5pt 0; }
code { font: 8.6pt "DejaVu Sans Mono", monospace; background: #eef2ef; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f2f5f3; border: 0.6pt solid #d4dcd7; padding: 6pt 8pt; font-size: 8.4pt;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt; font: 8.3pt/1.35 "DejaVu Sans", sans-serif;
        page-break-inside: avoid; }
th { background: #e3ede9; text-align: left; font-weight: 700; }
th, td { border: 0.5pt solid #c3cec8; padding: 2.6pt 4.5pt; vertical-align: top; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
blockquote { margin: 8pt 0; padding: 7pt 10pt; background: #eef5f2; border-left: 3pt solid #0f6b5c; }
blockquote p { margin: 0 0 3pt; }
blockquote ol { margin: 2pt 0 0; }
blockquote li, td, th { text-align: left; }
.titleblock + hr { display: none; }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; }
hr { border: 0; border-top: 0.6pt solid #c3cec8; margin: 10pt 0; }
.eq { text-align: center; font-size: 11pt; margin: 6pt 0 8pt; }
.titleblock { border-bottom: 1.5pt solid #0f6b5c; padding-bottom: 8pt; margin-bottom: 8pt; }
.titleblock .meta { font: 9pt "DejaVu Sans", sans-serif; color: #56625c; margin-top: 4pt; }
em { font-style: italic; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--src", default=os.path.join(REPORT, "MIDSEM_REPORT.md"))
    ap.add_argument("--out", default=os.path.join(REPORT, "MIDSEM_REPORT.pdf"))
    a = ap.parse_args()
    md = open(a.src, encoding="utf-8").read()
    for k, v in {**MATH, **INLINE}.items():
        md = md.replace(k, v)
    # python-markdown (unlike GitHub) needs a blank line before a list that follows a paragraph
    out, prev = [], ""
    item = re.compile(r"^(> )?(\d+\.|[*-]) ")
    for line in md.split("\n"):
        if item.match(line) and prev.strip() not in ("", ">") and not item.match(prev) \
                and not prev.startswith(("   ", "  ", "> ")) or (item.match(line) and prev.startswith("> ") and not item.match(prev) and not prev.startswith(">    ")):
            out.append(">" if line.startswith("> ") else "")
        out.append(line)
        prev = line
    md = "\n".join(out)
    leftover = re.findall(r"\$[^$\n]{1,80}\$", md)
    assert not leftover, f"unconverted math: {leftover}"
    # title block: first H1 + H2 + byline line become a styled header
    lines = md.split("\n")
    head, rest = lines[:3], lines[3:]
    title = markdown.markdown("\n".join(head))
    body = markdown.markdown("\n".join(rest).lstrip().removeprefix("---"), extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace('src="results/', f'src="file://{os.path.abspath(REPORT)}/results/')
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>DyVo mid-sem report</title>"
            f"<style>{CSS}</style></head><body><div class='titleblock'>{title}"
            f"<div class='meta'>CS6101 course project · mid-semester evaluation · code, data and results: "
            f"github.com/infinity2147/cs6101_midsem (branch claude/dyvo-dynamic-vocabularies-sparse-3qvicl)</div>"
            f"</div>{body}</body></html>")
    tmp = os.path.join(REPORT, "_print.html")
    open(tmp, "w", encoding="utf-8").write(html)
    chrome = a.chrome or shutil.which("chromium") or shutil.which("google-chrome") or \
        next(iter(sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))), None)
    subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(a.out)}", f"file://{os.path.abspath(tmp)}"],
                   check=True, stderr=subprocess.DEVNULL)
    os.remove(tmp)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
