"""Build index.html from post.md. One source, two formats.

Usage:  python3 build.py          (requires: pip install markdown)

- Markdown stays readable on GitHub as-is.
- The HTML is fully self-contained: CSS inline, figures embedded as base64.
- LaTeX spans ($...$ / $$...$$) render via lightweight inline substitution
  for the handful of expressions this post uses (no MathJax dependency).
"""
import base64
import pathlib
import re

import markdown  # pip install markdown

ROOT = pathlib.Path(__file__).parent
SRC = (ROOT / "post.md").read_text(encoding="utf-8")

# --- title / subtitle ---------------------------------------------------------
title = re.search(r"^# (.+)$", SRC, re.M).group(1)

# --- render the small fixed set of TeX used by the post -----------------------
TEX = {
    r"\sigma\sqrt{2\ln t}": "σ√(2 ln t)",
    r"N \;\approx\; \frac{\sigma^2}{\Delta^2}\,\log\frac{B}{\delta}":
        "N ≈ (σ²/Δ²) · log(B/δ)",
    r"\big|\,\text{true score} - \text{validation score}\,\big| \;\lesssim\; "
    r"\sqrt{\frac{\ell(a)\ln 2}{2n}}":
        "| true score − validation score |  ≲  √( ℓ(a)·ln2 / 2n )",
}
INLINE = {
    r"\mathcal{N}(0, \sigma^2)": "𝒩(0, σ²)", r"\sigma^2/\Delta^2 \cdot \log B": "σ²/Δ²·log B",
    r"\varepsilon^{-k}": "ε⁻ᵏ", r"\varepsilon": "ε", r"\sqrt{\ell/n}": "√(ℓ/n)",
    r"\gamma < 1": "γ < 1", r"\gamma": "γ", r"\sigma = \Delta": "σ = Δ", r"\Delta": "Δ",
    r"\sigma_b\sqrt{2\ln m}": "σ_b√(2 ln m)", r"\alpha_t \sim \sigma_b\sqrt{\ln m_t}":
    "α_t ~ σ_b√(ln m_t)", r"\sigma\sqrt{2\ln t}": "σ√(2 ln t)", r"\ln t": "ln t",
    r"O(k^2)": "O(k²)", r"10^{-40}": "10⁻⁴⁰", r"10^{-2}": "10⁻²", r"2^L": "2^L",
    r"\ell(a)": "ℓ(a)", r"\ell": "ℓ", r"\log B": "log B", r"\alpha_t": "α_t", r"\alpha": "α",
    r"\sigma_b": "σ_b", r"\sigma": "σ", r"\delta": "δ", r"d\ell/dt": "dℓ/dt",
    r"a^*": "a*", r"m_t": "m_t", r"k^2": "k²",
}


def detex(s: str) -> str:
    for k, v in sorted(INLINE.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    return s


def block_math(m):
    body = m.group(1).strip()
    return f'\n<p class="math">{TEX.get(body, detex(body))}</p>\n'


SRC = re.sub(r"\$\$(.+?)\$\$", block_math, SRC, flags=re.S)
SRC = re.sub(r"\$(.+?)\$", lambda m: f"<em class=\"m\">{detex(m.group(1))}</em>", SRC)

# --- markdown -> html ----------------------------------------------------------
body = markdown.markdown(SRC, extensions=["fenced_code", "tables", "attr_list", "smarty"])

# --- embed figures as data URIs so index.html is a single file ------------------
def embed(m):
    p = ROOT / m.group(1)
    if not p.exists():
        return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:image/png;base64,{b64}"'


body = re.sub(r'src="(assets/[^"]+\.png)"', embed, body)

CSS = """
:root{--ivory:#FAF9F5;--slate:#141413;--clay:#D97757;--olive:#788C5D;--sky:#6A8CAF;
--g150:#F0EEE6;--g300:#D1CFC5;--g500:#87867F;--g700:#3D3D3A;
--serif:ui-serif,Georgia,"Times New Roman",serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ivory);color:var(--g700);font:16.5px/1.7 var(--sans);padding:64px 24px 120px}
main{max-width:720px;margin:0 auto}
.eyebrow{font:11px var(--mono);text-transform:uppercase;letter-spacing:.09em;color:var(--g500);margin-bottom:14px}
h1{font:500 36px/1.2 var(--serif);color:var(--slate);letter-spacing:-.012em;margin-bottom:10px}
h1+p em{font:400 17px/1.55 var(--serif);color:var(--g500)}
h2{font:500 24px/1.3 var(--serif);color:var(--slate);margin:46px 0 14px}
p{margin:0 0 16px}
a{color:var(--sky);text-underline-offset:2px}
strong{color:var(--slate)}
img{max-width:100%;border:1.5px solid var(--g300);border-radius:12px;margin:10px 0 6px;background:#fff}
img+em,p>img~em{display:block;font:13px var(--mono);color:var(--g500)}
code{font:14px var(--mono);background:var(--g150);padding:1px 5px;border-radius:4px}
pre{background:var(--g150);border:1px solid var(--g300);border-radius:10px;padding:16px 18px;
font:13.5px/1.55 var(--mono);overflow-x:auto;margin:0 0 18px}
pre code{background:none;padding:0}
.math{font:18px var(--serif);color:var(--slate);text-align:center;margin:22px 0;letter-spacing:.01em}
em.m{font-style:normal;font-family:var(--serif)}
blockquote{border-left:3px solid var(--clay);padding:4px 18px;color:var(--slate);font-family:var(--serif)}
hr{border:none;border-top:1.5px solid var(--g300);margin:46px 0}
ul{margin:0 0 16px 22px}
li{margin-bottom:8px}
footer{margin-top:60px;border-top:1.5px solid var(--g300);padding-top:16px;
font:12px var(--mono);color:var(--g500)}
@media(max-width:760px){body{padding:36px 16px 80px}h1{font-size:29px}}
"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><main>
<div class="eyebrow">Learning Beyond Gradients · companion blog · June 2026</div>
{body}
<footer>Built from post.md · figures from make_figures.py (seeded) · single-file HTML, images embedded</footer>
</main></body></html>
"""

(ROOT / "index.html").write_text(html, encoding="utf-8")
print(f"index.html written ({len(html)//1024} KB)")
