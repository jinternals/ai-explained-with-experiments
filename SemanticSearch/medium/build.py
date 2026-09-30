"""Build the Medium version of the post from the lab's results.

    python3 medium/build.py
    node ~/.claude/skills/medium-post/scripts/render-figures.mjs medium/figures medium/images

Writes medium/semantic-search-medium.md and one HTML fragment per figure in medium/figures/.
Every number comes from out/example.json, out/benchmark.json and out/length.json.
Standard library only.
"""
import html
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

e = json.loads((OUT / "example.json").read_text())
b = json.loads((OUT / "benchmark.json").read_text())
l = json.loads((OUT / "length.json").read_text())
D = b["datasets"]
L = l["datasets"]
NAMES = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA"}


def f(value, digits=3):
    """Round half up, like Java's printf and the web page."""
    q = Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:,f}".replace("-", "−")


def signed(value, digits=3):
    q = Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{q:+,f}".replace("-", "−")


def pct(value):
    return f"{f(100 * value, 0)}%"


def esc(text):
    return html.escape(str(text))


# ---------------------------------------------------------------- derived values

queries = sum(ds["queries"] for ds in D.values())
groups = {}
for g in ("low", "middle", "high"):
    rows = [ds["by_word_overlap"][g] for ds in D.values()]
    n = sum(r["queries"] for r in rows)
    groups[g] = {
        "queries": n,
        "keyword": sum(r["keyword"] * r["queries"] for r in rows) / n,
        "meaning": sum(r["meaning"] * r["queries"] for r in rows) / n,
        "better": sum(r["meaning_better"] for r in rows),
        "worse": sum(r["meaning_worse"] for r in rows),
    }
hnsw_same = sum(ds["hnsw_vs_exact"]["same_top10"] for ds in D.values())
metrics_same = sum(ds["metrics_agree"]["same_top10"] for ds in D.values())
gain = {k: D[k]["exact"]["ndcg@10"] - D[k]["keyword"]["ndcg@10"] for k in D}
nf = e["nfcorpus"]
lv = e["levels"]
S = e["sentences"]
P = e["pairs"]

# The worked nDCG example: part 1's question has three labelled papers (2, 1 and 1 points).
labels = sorted((r["relevance"] for r in nf["relevant"]), reverse=True)
ideal = sum(g / math.log2(i + 2) for i, g in enumerate(labels))
top_rel = nf["relevant"][0]
kw_pos, mn_pos = top_rel["keyword_rank"], top_rel["meaning_rank"]


def disc(pos):
    return math.log2(pos + 1)


# ---------------------------------------------------------------- figures

def fig(name, content):
    (FIG / f"{name}.html").write_text(content)


(FIG / "_head.html").write_text(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">')
(FIG / "_style.css").write_text("""
body { font-family: "IBM Plex Sans", system-ui, sans-serif; color: #1d1f29; }
.figure { background: #fff; }
.t { width: 100%; border-collapse: collapse; font-size: 15px; }
.t th, .t td { padding: 9px 10px; border-bottom: 1px solid #e3e4ec; text-align: left; vertical-align: top; }
.t thead th.keep { text-transform: none; letter-spacing: 0; font-size: 13px; }
.t thead th { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #6b6f80; font-weight: 600; border-bottom: 2px solid #c9cbd8; }
.t th.n { text-align: right; }
.t td.n { text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; white-space: nowrap; }
.t tbody th { font-weight: 600; }
.win { color: #1f7a45; font-weight: 700; }
.loss { color: #b3261e; font-weight: 700; }
.hl td, .hl th { background: #f2eefc; }
.muted { color: #6b6f80; }
.cap { font-size: 12px; color: #6b6f80; margin-top: 10px; }
.title { font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: #6b6f80; font-weight: 600; margin: 0 0 12px; }
.chip { display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 12px; border: 1px solid #c9cbd8; border-radius: 4px; padding: 0 6px; margin: 2px 4px 0 0; color: #4a4e60; }
.chip.none { border-style: dashed; color: #9a9db0; }
.bar { position: relative; height: 22px; background: #eef0f5; border-radius: 4px; }
.bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: #b9c8f2; }
.bar.kw i { background: #d3d5df; }
.bar b { position: absolute; left: 8px; top: 2px; font-family: "IBM Plex Mono", monospace; font-size: 13px; font-weight: 600; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.mono { font-family: "IBM Plex Mono", monospace; }
""")


def bar(value, cls="", scale=1.0):
    return f'<div class="bar {cls}"><i style="width:{max(value, 0) / scale * 100:.1f}%"></i><b>{f(value)}</b></div>'


def chips(words):
    return "".join(f'<span class="chip">{esc(w)}</span>' for w in words) or '<span class="chip none">none</span>'


# 01 hero: the five sentences, ranked two ways
order_words = sorted(range(5), key=lambda i: (-len(S[i]["shared_words"]), i))
order_cos = sorted(range(5), key=lambda i: -S[i]["cosine"])


def rank_col(order, title, by_cos):
    rows = ""
    for pos, i in enumerate(order, 1):
        s = S[i]
        rows += (f'<div style="display:grid;grid-template-columns:22px 1fr;gap:8px;padding:9px 0;border-bottom:1px solid #e3e4ec">'
                 f'<b class="mono" style="color:#6d4bd8">{pos}</b><div><div style="font-size:14px">{esc(s["text"])}</div>'
                 + (f'<div style="margin-top:6px">{bar(s["cosine"])}</div>' if by_cos else f'<div>{chips(s["shared_words"])}</div>')
                 + "</div></div>")
    return f'<div><p class="title">{title}</p>{rows}</div>'


fig("01-two-rankings",
    f'<p style="font-size:15px;margin:0 0 14px">Question: <b>“{esc(e["question"])}”</b></p>'
    f'<div class="cols">{rank_col(order_words, "By shared words", False)}{rank_col(order_cos, "By meaning (cosine)", True)}</div>')

# 02 arrows
A = e["arrows"]
fig("02-arrows", f"""
<svg viewBox="0 0 652 300" width="652" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace">
<defs><marker id="p" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6d4bd8"/></marker>
<marker id="q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#3b6fd8"/></marker></defs>
<line x1="70" y1="150" x2="370" y2="150" stroke="#d3d5df" stroke-width="1.5"/><line x1="220" y1="20" x2="220" y2="280" stroke="#d3d5df" stroke-width="1.5"/>
<line x1="220" y1="150" x2="332" y2="66" stroke="#6d4bd8" stroke-width="3.5" marker-end="url(#p)"/>
<line x1="220" y1="150" x2="304" y2="38" stroke="#3b6fd8" stroke-width="2.5" marker-end="url(#q)"/>
<line x1="220" y1="150" x2="136" y2="38" stroke="#3b6fd8" stroke-width="2.5" marker-end="url(#q)"/>
<line x1="220" y1="150" x2="108" y2="234" stroke="#3b6fd8" stroke-width="2.5" marker-end="url(#q)"/>
<g font-size="15" fill="#1d1f29"><text x="340" y="72" font-weight="600">a (4, 3)</text><text x="298" y="28">b (3, 4)</text><text x="88" y="28">c (−3, 4)</text><text x="36" y="256">d (−4, −3)</text></g>
<g font-size="14" fill="#1d1f29"><text x="430" y="86">a, b: {f(A[0]["degrees"], 0)}° apart</text><text x="430" y="106" fill="#6b6f80">similar</text>
<text x="430" y="150">a, c: {f(A[1]["degrees"], 0)}° apart</text><text x="430" y="170" fill="#6b6f80">unrelated</text>
<text x="430" y="214">a, d: {f(A[2]["degrees"], 0)}° apart</text><text x="430" y="234" fill="#6b6f80">opposite</text></g>
</svg>""")

# 03 the four steps, worked
rows = ""
for arrow, prod in zip(A, ["12 and 12", "−12 and 12", "−16 and −9"]):
    v = arrow["vector"]
    cls = "win" if arrow["cosine"] > 0.5 else ("loss" if arrow["cosine"] < 0 else "")
    rows += (f'<tr><th>{"bcd"[A.index(arrow)]} = ({f(v[0], 0)}, {f(v[1], 0)})</th><td class="n">{prod}</td><td class="n">{f(arrow["dot"], 0)}</td>'
             f'<td class="n">5 × 5</td><td class="n {cls}">{f(arrow["cosine"], 2)}</td><td class="n">{f(arrow["degrees"], 0)}°</td></tr>')
fig("03-four-steps", f"""<table class="t"><thead><tr><th class="keep">a = (4, 3) against</th><th class="n">1. Multiply</th><th class="n">2. Add</th>
<th class="n">3. Lengths</th><th class="n">4. Divide</th><th class="n">Angle</th></tr></thead><tbody>{rows}</tbody></table>""")

# 04 formula
fig("04-formula", """
<div style="font-family:'IBM Plex Mono',monospace;font-size:22px;padding:10px 6px">
<div style="display:flex;align-items:center;gap:14px"><b>cosine</b> =
<span style="display:inline-flex;flex-direction:column;text-align:center">
<span style="border-bottom:2px solid #1d1f29;padding:0 10px 6px;color:#6d4bd8">a · b</span>
<span style="padding:6px 10px 0;color:#3b6fd8">length of a × length of b</span></span></div>
<div style="margin-top:22px;font-size:17px"><b style="color:#6d4bd8">a · b</b> = a₁ × b₁ + a₂ × b₂ + … <span class="muted" style="font-size:13px">steps 1 and 2</span></div>
<div style="margin-top:10px;font-size:17px"><b style="color:#3b6fd8">length of a</b> = √( a₁² + a₂² + … ) <span class="muted" style="font-size:13px">step 3</span></div>
</div>""")

# 05 toy: "bank" before and after adjusting, in two sentences (illustration)
TOY = {
    "She sat on the bank of the river": [("bank", (0.55, 0.50), (0.20, 0.88)), ("river", (0.05, 0.95), (0.06, 0.96)), ("water", (0.45, 0.80), (0.40, 0.82))],
    "He opened a savings account at the bank": [("savings", (0.95, 0.08), (0.95, 0.10)), ("account", (0.80, 0.40), (0.82, 0.36)), ("bank", (0.55, 0.50), (0.92, 0.18))],
}


def unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def mean(vs):
    return (sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs))


def toy_panel(title, words, adjusted):
    ox, oy, s = 40, 230, 180
    out = ""
    for name, v0, v1 in words:
        v = v1 if adjusted else v0
        color = "#6d4bd8" if name == "bank" else "#3b6fd8"
        x, y = ox + v[0] * s, oy - v[1] * s
        out += (f'<line x1="{ox}" y1="{oy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{3 if name == "bank" else 2.2}" marker-end="url(#{"p" if name == "bank" else "q"})"/>'
                f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" font-size="12" font-weight="{700 if name == "bank" else 400}">{name}</text>')
    if adjusted:
        m = unit(mean([w[2] for w in words]))
        mx, my = ox + m[0] * s * 0.6, oy - m[1] * s * 0.6
        out += (f'<line x1="{ox}" y1="{oy}" x2="{mx:.1f}" y2="{my:.1f}" stroke="#1d1f29" stroke-width="4.5" marker-end="url(#k)"/>'
                + (f'<text x="{mx + 8:.1f}" y="{my + 14:.1f}" font-size="12" font-weight="700">sentence</text>' if m[1] > m[0]
                 else f'<text x="{mx - 40:.1f}" y="{my - 14:.1f}" font-size="12" font-weight="700">sentence</text>'))
    return (f'<g><text x="{ox - 10}" y="16" font-size="12" fill="#6b6f80">{title}</text>'
            f'<line x1="{ox}" y1="{oy}" x2="{ox + 200}" y2="{oy}" stroke="#d3d5df"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy - 200}" stroke="#d3d5df"/>'
            f'<text x="{ox + 200}" y="{oy + 18}" font-size="11" fill="#6b6f80" text-anchor="end">money →</text>'
            f'<text x="{ox - 12}" y="{oy - 200}" font-size="11" fill="#6b6f80" transform="rotate(-90 {ox - 12} {oy - 200})" text-anchor="end">nature, water →</text>{out}</g>')


def toy_fig(name, sentence):
    words = TOY[sentence]
    fig(name, f"""<p class="title">Toy model with two named directions · “{esc(sentence)}”</p>
<svg viewBox="0 0 652 250" width="652" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Mono, monospace" fill="#1d1f29">
<defs><marker id="p" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#6d4bd8"/></marker>
<marker id="q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#3b6fd8"/></marker>
<marker id="k" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#1d1f29"/></marker></defs>
{toy_panel("Step 2: starting arrows", words, False)}
<g transform="translate(330 0)">{toy_panel("Steps 3–4: adjusted, averaged", words, True)}</g>
</svg>""")


toy_fig("05-toy-river", "She sat on the bank of the river")
toy_fig("06-toy-savings", "He opened a savings account at the bank")
river = unit(mean([w[2] for w in TOY["She sat on the bank of the river"]]))
savings = unit(mean([w[2] for w in TOY["He opened a savings account at the bank"]]))
river0 = unit(mean([w[1] for w in TOY["She sat on the bank of the river"]]))
savings0 = unit(mean([w[1] for w in TOY["He opened a savings account at the bank"]]))
toy_with = river[0] * savings[0] + river[1] * savings[1]
toy_without = river0[0] * savings0[0] + river0[1] * savings0[1]

# 07 the real model at three sizes
W, SE, PA = lv["words"], lv["sentences"], lv["paragraph"]
fig("07-three-sizes", f"""<table class="t"><thead><tr><th>Size</th><th>Compared</th><th class="n">Cosine</th></tr></thead><tbody>
<tr><th rowspan="3">Word</th><td>{W[0]["a"]} · {W[0]["b"]}</td><td class="n">{f(W[0]["cosine"])}</td></tr>
<tr><td>{W[1]["a"]} · {W[1]["b"]}</td><td class="n">{f(W[1]["cosine"])}</td></tr>
<tr><td>{W[2]["a"]} · {W[2]["b"]}</td><td class="n">{f(W[2]["cosine"])}</td></tr>
<tr><th rowspan="3">Sentence</th><td>river bank · savings bank <span class="muted">(share “bank”)</span></td><td class="n loss">{f(SE[0]["cosine"])}</td></tr>
<tr><td>river bank · picnic by the stream <span class="muted">(share nothing)</span></td><td class="n win">{f(SE[1]["cosine"])}</td></tr>
<tr><td>savings bank · salary into an account</td><td class="n win">{f(SE[2]["cosine"])}</td></tr>
<tr><th rowspan="2">Paragraph</th><td>parking question · the parking sentence alone</td><td class="n win">{f(PA["sentences"][2]["cosine"])}</td></tr>
<tr><td>parking question · the whole three-sentence paragraph</td><td class="n loss">{f(PA["whole"])}</td></tr>
</tbody></table>
<p class="cap">all-MiniLM-L6-v2, real 384-number vectors. Sentences: “{esc(SE[0]["a"])}” / “{esc(SE[0]["b"])}” / “{esc(SE[1]["b"])}” / “{esc(SE[2]["b"])}”</p>""")

# 08 the four limits
fig("08-limits", """<table class="t"><thead><tr><th>How much text does the model read?</th><th class="n">Word pieces</th><th>Where it comes from</th></tr></thead><tbody>
<tr><th>What the model can physically take</th><td class="n">512</td><td>Its design has 512 position slots</td></tr>
<tr><th>What its makers set</th><td class="n">256</td><td>Model card and settings file; MTEB scores use this</td></tr>
<tr><th>What it was trained on</th><td class="n">128</td><td>Model card</td></tr>
<tr class="hl"><th>What LangChain4j gives it</th><td class="n">128</td><td>The settings file bundled with the Java library</td></tr>
</tbody></table>""")

# 09 pairs
verdicts = ["Synonyms: close", "Related: fairly close", "Unrelated: far", "Opposites look related",
            "Who bit whom: lost", "“Not” barely counts", "Different orders blur"]
rows = "".join(
    f'<tr><td>{esc(p["a"])}</td><td>{esc(p["b"])}</td><td class="n">{f(p["cosine"])}</td>'
    f'<td class="{"win" if i < 3 else "loss"}">{verdicts[i]}</td></tr>' for i, p in enumerate(P))
fig("09-pairs", f'<table class="t"><thead><tr><th>Text A</th><th>Text B</th><th class="n">Cosine</th><th>What it shows</th></tr></thead><tbody>{rows}</tbody></table>')

# 10 part 1's question, both ways
def lst(items, extra=False):
    out = ""
    for pos, r in enumerate(items, 1):
        rel = r["relevance"] > 0
        note = f'<div class="muted" style="font-size:12px">keyword search: #{r["keyword_rank"]}</div>' if extra and r.get("keyword_rank") and r["keyword_rank"] > 10 else ""
        out += (f'<div style="display:grid;grid-template-columns:20px 1fr;gap:6px;padding:8px 8px;border:1px solid {"#1f7a45" if rel else "#e3e4ec"};'
                f'border-radius:6px;margin-bottom:6px;{"box-shadow:inset 3px 0 0 #1f7a45;" if rel else ""}"><b class="mono muted">{pos}</b>'
                f'<div style="font-size:13px">{esc(r["title"])}{"<div class=win style=font-size:11px>MARKED RELEVANT</div>" if rel else ""}{note}</div></div>')
    return out


fig("10-part1-question", f"""<p style="font-size:15px;margin:0 0 14px">NFCorpus question: <b>“{esc(nf["query"])}”</b></p>
<div class="cols"><div><p class="title">Keyword search (BM25)</p>{lst(nf["keyword_top"])}</div>
<div><p class="title">Meaning search (cosine)</p>{lst(nf["meaning_top"], True)}</div></div>""")

# 11 nDCG worked example
fig("11-ndcg", f"""<table class="t"><thead><tr><th>Position</th><th class="n">1</th><th class="n">2</th><th class="n">3</th><th class="n">4</th><th class="n">5</th><th class="n">10</th></tr></thead><tbody>
<tr><th>Points are divided by</th>{''.join(f'<td class="n">{f(disc(p), 2)}</td>' for p in (1, 2, 3, 4, 5, 10))}</tr></tbody></table>
<table class="t" style="margin-top:18px"><thead><tr><th>“{esc(nf["query"])}”</th><th>Where the 2-point paper landed</th><th class="n">Points</th><th class="n">nDCG@10</th></tr></thead><tbody>
<tr><th>Perfect order (all 3 labelled papers on top)</th><td>#1, #2, #3</td><td class="n">{f(ideal, 2)}</td><td class="n">1.000</td></tr>
<tr><th>Keyword search</th><td>#{kw_pos}</td><td class="n">{f(2 / disc(kw_pos), 2)}</td><td class="n win">{f(nf["ndcg@10"]["keyword"])}</td></tr>
<tr><th>Meaning search</th><td>#{mn_pos}</td><td class="n">{f(2 / disc(mn_pos), 2)}</td><td class="n loss">{f(nf["ndcg@10"]["meaning"])}</td></tr>
</tbody></table>""")

# 12 results
rows = ""
for k in ("nfcorpus", "scifact", "fiqa"):
    ds = D[k]
    kw, mn = ds["keyword"]["ndcg@10"], ds["exact"]["ndcg@10"]
    mvk = ds["meaning_vs_keyword"]
    rows += (f'<tr><th>{NAMES[k]}</th><td class="n">{ds["queries"]:,}</td>'
             f'<td class="n {"win" if kw > mn else ""}">{f(kw)}</td><td class="n {"win" if mn > kw else ""}">{f(mn)}</td>'
             f'<td class="n">{f(ds["keyword"]["recall@100"])} → {f(ds["exact"]["recall@100"])}</td>'
             f'<td class="n"><span class="win">{mvk["better"]}</span> / {mvk["same"]} / <span class="loss">{mvk["worse"]}</span></td></tr>')
fig("12-results", f"""<table class="t"><thead><tr><th>Dataset</th><th class="n">Questions</th><th class="n">nDCG@10 keyword</th>
<th class="n">nDCG@10 meaning</th><th class="n">Recall@100 keyword → meaning</th><th class="n">Meaning better / same / worse</th></tr></thead><tbody>{rows}</tbody></table>
<p class="cap">OpenSearch 2.19.1. Keyword: BM25, default settings. Meaning: all-MiniLM-L6-v2, cosine, every document checked.</p>""")

# 13 word overlap groups
names = {"low": "Under ⅓ of the words", "middle": "⅓ to ⅔", "high": "⅔ or more"}
rows = ""
for g in ("low", "middle", "high"):
    G = groups[g]
    rows += (f'<div style="display:grid;grid-template-columns:150px 1fr 110px;gap:14px;align-items:center;padding:10px 0;border-bottom:1px solid #e3e4ec">'
             f'<div><b style="font-size:14px">{names[g]}</b><div class="muted" style="font-size:12px">{G["queries"]} questions</div></div>'
             f'<div>{bar(G["keyword"], "kw")}<div style="height:5px"></div>{bar(G["meaning"])}</div>'
             f'<div class="mono" style="font-size:13px;text-align:right"><span class="win">{G["better"]}</span> : <span class="loss">{G["worse"]}</span></div></div>')
fig("13-word-overlap", f"""<p class="title">Share of the question's words found in its relevant documents</p>
<div style="display:flex;gap:18px;font-size:12px;color:#6b6f80;margin-bottom:6px"><span><i style="display:inline-block;width:12px;height:12px;background:#d3d5df;border-radius:2px;vertical-align:-1px"></i> keyword nDCG@10</span>
<span><i style="display:inline-block;width:12px;height:12px;background:#b9c8f2;border-radius:2px;vertical-align:-1px"></i> meaning nDCG@10</span><span style="margin-left:auto">meaning won : lost</span></div>{rows}
<p class="cap">All three datasets together, {queries:,} questions.</p>""")

# 14 how much text
rows = ""
for k in ("nfcorpus", "scifact", "fiqa"):
    x = L[k]["ndcg@10"]
    best = max(x, key=x.get)
    cells = "".join(f'<td class="n {"win" if key == best else ""}">{f(x[key])}</td>' for key in ("128", "256", "none"))
    rows += f'<tr><th>{NAMES[k]}</th><td class="n">{pct(L[k]["share_longer_than"]["128"])}</td>{cells}</tr>'
fig("14-length", f"""<table class="t"><thead><tr><th>Dataset</th><th class="n">Longer than 128 pieces</th><th class="n">First 128 (as shipped)</th>
<th class="n">First 256</th><th class="n">Whole document</th></tr></thead><tbody>{rows}</tbody></table>
<p class="cap">nDCG@10 for meaning search, same model and questions. MTEB publishes 0.316, 0.645 and 0.369 for this model, cut at 256.</p>""")

# 15 HNSW
rows = ""
for k in ("nfcorpus", "scifact", "fiqa"):
    ds = D[k]
    rows += (f'<tr><th>{NAMES[k]}</th><td class="n">{ds["documents"]:,}</td><td class="n">{pct(ds["hnsw_vs_exact"]["same_top10_share"])}</td>'
             f'<td class="n">{f(ds["exact"]["ndcg@10"])}</td><td class="n">{f(ds["hnsw"]["ndcg@10"])}</td>'
             f'<td class="n">{f(ds["timing_ms"]["exact"], 1)} ms</td><td class="n win">{f(ds["timing_ms"]["hnsw"], 1)} ms</td></tr>')
fig("15-hnsw", f"""<table class="t"><thead><tr><th>Dataset</th><th class="n">Docs</th><th class="n">Same top 10</th><th class="n">nDCG exact</th>
<th class="n">nDCG HNSW</th><th class="n">Exact search</th><th class="n">HNSW</th></tr></thead><tbody>{rows}</tbody></table>
<p class="cap">Median time per question, top 10, both inside OpenSearch. “Exact” compares the question with every document.</p>""")

# 16 metrics
order = sorted(range(5), key=lambda i: -S[i]["cosine"])
rows = "".join(f'<tr><td>{esc(S[i]["text"])}</td><td class="n">{f(S[i]["cosine"])}</td><td class="n">{f(S[i]["dot"])}</td><td class="n">{f(S[i]["euclidean"])}</td></tr>' for i in order)
fig("16-metrics", f"""<table class="t"><thead><tr><th>Compared with “{esc(e["question"])}”</th><th class="n">Cosine</th><th class="n">Dot product</th><th class="n">Distance</th></tr></thead>
<tbody>{rows}</tbody></table><p class="cap">Higher cosine and dot product mean closer; lower distance means closer. The order is the same all three ways.</p>""")

# ---------------------------------------------------------------- the story

s0, s1 = S[0], S[1]
r384 = e["recipe_384"]
la = e["longer_arrow"]
probe = L["fiqa"]["longest_document_probe"]
alt_rank = "; ".join(f'{pos}. {S[i]["text"]} ({len(S[i]["shared_words"])} shared words)' for pos, i in enumerate(order_words, 1))
alt_cos = "; ".join(f'{pos}. {S[i]["text"]} ({f(S[i]["cosine"])})' for pos, i in enumerate(order_cos, 1))

story = f"""# Semantic Search, Without the Scary Math

## How search by meaning turns words, sentences and paragraphs into arrows of numbers, and when it beats keyword search, tested on {queries:,} real questions.

![The question "{e["question"]}" and five sentences, ranked two ways. By shared words: {alt_rank}. By meaning, with cosine similarity: {alt_cos}.](images/01-two-rankings.png)

*The same five sentences, ranked by shared words and by meaning. The salt sentence shares no words with the question.*

"Eating less salt helps reduce hypertension" is a good answer to "How can I lower my blood pressure?". It doesn't share a single word with it.

The first part of this series was about keyword search, BM25, which scores a document by the words it has in common with the question. By that measure, the salt sentence scores nothing, and "Lower the tyre pressure before driving on sand" is a match on two words.

Semantic search ranks them the other way round. It scored the salt sentence **{f(s0["cosine"])}** out of 1 and the tyre sentence {f(S[3]["cosine"])}. This post shows where those numbers come from, using nothing harder than multiplication, and then tests on {queries:,} real questions when searching by meaning beats searching by words.

Every number here was printed by a Java lab you can run yourself; the link is at the end.

---

## Meaning as a direction

Linguists noticed in the 1950s that words with similar meanings turn up in similar company. J. R. Firth's version: "You shall know a word by the company it keeps".

Record that company as a list of numbers and you can draw it as an arrow. With two numbers, (4, 3) means 4 steps right and 3 steps up.

![Four arrows from one point. a = (4, 3). b = (3, 4) is {f(A[0]["degrees"], 0)} degrees from a: similar. c = (−3, 4) is {f(A[1]["degrees"], 0)} degrees from a: unrelated. d = (−4, −3) is {f(A[2]["degrees"], 0)} degrees from a: opposite.](images/02-arrows.png)

*Similarity is the angle between two arrows. Their length doesn't matter.*

Semantic search does exactly this, with 384 numbers per text instead of 2. Texts that mean similar things get arrows that point in similar directions.

---

## Cosine similarity in four steps

With 384 numbers there's nothing to draw, so the angle has to come from the numbers alone. **Cosine similarity** does that. It turns the angle into a score: 1 for the same direction, 0 for a right angle, −1 for opposite directions.

For a = (4, 3) and b = (3, 4):

- **1. Multiply** the numbers in matching positions: 4 × 3 = 12, and 3 × 4 = 12.
- **2. Add** the results: 12 + 12 = {f(A[0]["dot"], 0)}. This total is called the **dot product**.
- **3. Measure** each arrow's length, with Pythagoras: √(4² + 3²) = √25 = {f(A[0]["length_a"], 0)}. b is also {f(A[0]["length_other"], 0)} long.
- **4. Divide** the dot product by both lengths: {f(A[0]["dot"], 0)} ÷ 25 = **{f(A[0]["cosine"], 2)}**.

![The four steps for a = (4, 3) against three arrows. Against b (3, 4): multiply 12 and 12, add {f(A[0]["dot"], 0)}, lengths 5 × 5, cosine {f(A[0]["cosine"], 2)}, {f(A[0]["degrees"], 0)} degrees. Against c (−3, 4): −12 and 12, add {f(A[1]["dot"], 0)}, cosine {f(A[1]["cosine"], 2)}, {f(A[1]["degrees"], 0)} degrees. Against d (−4, −3): −16 and −9, add {f(A[2]["dot"], 0)}, cosine {f(A[2]["cosine"], 2)}, {f(A[2]["degrees"], 0)} degrees.](images/03-four-steps.png)

*Negative numbers do the work. For c, one position agrees and the other disagrees, and they cancel to 0.*

Step 4 is why only direction counts. Take e = (8, 6): it points exactly where a points, but it's twice as long. Its dot product with b is {f(la["dot_with_b"], 0)}, twice a's {f(la["dot_a_with_b"], 0)}. Dividing by e's length of {f(la["length"], 0)} brings the cosine back to {f(la["cosine_with_b"], 2)}, the same as a's. Without that step, long arrows would score high against everything.

Written as a formula, for when you meet it elsewhere:

![Formula: cosine = (a · b) ÷ (length of a × length of b). The dot product a · b = a1 × b1 + a2 × b2 + and so on. The length of a = the square root of (a1² + a2² + and so on).](images/04-formula.png)

*Steps 1 and 2 are the top of the fraction, step 3 the bottom.*

Real embeddings use the same four steps. For the question and the salt sentence, the first multiplication is {f(r384["first_products"][0]["question"], 4)} × {f(r384["first_products"][0]["sentence"], 4)}, and the second is {f(r384["first_products"][1]["question"], 4)} × {f(r384["first_products"][1]["sentence"], 4)}. After 382 more, the total is {f(r384["dot"], 4)}. No single product matters much. The score is hundreds of small agreements added up.

---

## Words, sentences and paragraphs: one recipe

People talk about "word embeddings", "sentence embeddings" and "document embeddings" as if they were three different things. In a model like the one used here, they're the same thing. Whatever text goes in, one arrow of 384 numbers comes out, made by the same four steps:

- **1. Split** the text into word pieces. Common words are one piece; rarer words are split, so "hypertension" becomes {" + ".join('"' + p + '"' for p in lv["pieces"][1]["pieces"] if p not in ("[CLS]", "[SEP]"))}.
- **2. Start**: every piece gets a starting arrow from a fixed table. It's the same arrow in every sentence. This is all a "word embedding" is.
- **3. Adjust**: every piece looks at every other piece in the text and shifts its arrow to fit. The model repeats this six times.
- **4. Average** all the pieces' arrows into one, and scale it to length 1.

To see step 3 happen, here's a toy version with just two directions, and we can give them names: **money** to the right, **nature and water** up.

![Toy model, "She sat on the bank of the river". Left, starting arrows: bank points halfway between money and nature, river and water point up toward nature. Right, after adjusting: bank has swung up toward nature, and the black averaged arrow for the sentence points toward nature.](images/05-toy-river.png)

*Next to "river" and "water", "bank" swings toward nature. An illustration, not real model output.*

![Toy model, "He opened a savings account at the bank". Left, starting arrows: bank points halfway, savings and account point toward money. Right, after adjusting: bank has swung toward money, and the black averaged arrow for the sentence points toward money.](images/06-toy-savings.png)

*Same starting arrow for "bank", opposite result.*

In the toy, the two sentences end up with a cosine of {f(toy_with, 2)}. If you skipped step 3 and averaged the starting arrows, they'd score {f(toy_without, 2)}, because the shared "bank" would drag them together.

One way to picture step 3: every word is at a meeting. It arrives with its own opinion, listens to everyone else, and shifts its view to fit the conversation. The text's arrow is the room's average view. A **single word** is a meeting of one, so "bank" stays in the middle. A **paragraph** is a bigger meeting: more voices averaged, so if it covers several topics, its arrow sits between them and points clearly at none.

### The real model, measured

The same effects, on the real model's 384 numbers:

![Real cosines from all-MiniLM-L6-v2. Words: bank and river {f(W[0]["cosine"])}, bank and money {f(W[1]["cosine"])}, river and money {f(W[2]["cosine"])}. Sentences: river bank and savings bank {f(SE[0]["cosine"])}; river bank and picnic by the stream {f(SE[1]["cosine"])}; savings bank and salary into an account {f(SE[2]["cosine"])}. Paragraph: parking question against the parking sentence {f(PA["sentences"][2]["cosine"])}, against the whole paragraph {f(PA["whole"])}.](images/07-three-sizes.png)

*The two "bank" sentences share a word and score lower than the river bank and the stream, which share none.*

### Three things people get wrong

**"A sentence's arrow is the average of its words' arrows."** The words are adjusted by each other first, and only then averaged. If you embed each word on its own and average them, the river-bank and savings-bank sentences score **{f(SE[0]["cosine_of_word_averages"])}**, nearly identical. The model's real sentence arrows score **{f(SE[0]["cosine"])}**.

**"A longer text gives a richer arrow."** It gives a more blended one. Every text ends up as 384 numbers however long it is. The parking question scores {f(PA["sentences"][2]["cosine"])} against the parking sentence alone and {f(PA["whole"])} against the paragraph that contains it. That's why search systems split long documents into chunks and embed each chunk separately.

**"Word, sentence and document embeddings are different kinds of thing."** Same model, same 384 numbers, same space. That's what lets a seven-word question be compared directly with a 200-word abstract, which is all semantic search does. Arrows from two *different* models can't be compared at all, though.

---

## Meet the model

Everything here uses one small, free model, **all-MiniLM-L6-v2**, from the Sentence Transformers project. The name says what it is: a MiniLM, Microsoft's compressed version of Google's BERT, with 6 layers (the six rounds of step 3). The Sentence Transformers team trained it on about 1.17 billion pairs of texts that belong together, such as a Stack Exchange question and its accepted answer, or a paper's title and its abstract. Training pulled each pair's arrows together and pushed unrelated texts apart.

It outputs 384 numbers per text, knows 30,522 word pieces, lowercases everything, has about 23 million weights, and runs on a laptop CPU in {f(D["nfcorpus"]["timing_ms"]["embed_question"], 0)} milliseconds per question.

One detail needs a table of its own, because there are four different answers to "how much text does it read?":

![How much text does the model read, in word pieces. What the model can physically take: 512. What its makers set: 256. What it was trained on: 128. What LangChain4j, the Java library that runs it, gives it: 128.](images/08-limits.png)

*The last row applies to every search result in this post. More on it below.*

128 word pieces is roughly the first 100 words.

---

## What the arrows capture, and what they miss

![Cosines for pairs. {"; ".join(f'{p["a"]} and {p["b"]}: {f(p["cosine"])}, {verdicts[i].lower()}' for i, p in enumerate(P))}.](images/09-pairs.png)

*Green rows behave as hoped. Red rows are where meaning search goes wrong.*

The arrows measure "about the same thing", which is close to "means the same thing" but not equal to it. "Hot" and "cold" turn up in the same kinds of sentences, so they sit fairly close. "The dog bit the man" and "The man bit the dog" score {f(P[4]["cosine"])}. A 2024 study, [NevIR](https://arxiv.org/abs/2305.07614), found that most retrieval models do no better than chance at telling a document from its negated version. And to the model, order 4471 is just "an order number", while the person searching wants that one order.

---

## Part 1's question, searched by meaning

Part 1 took apart one keyword search on **NFCorpus**, a public test set of {nf["documents"]:,} medical research papers with real questions. Here's the same question, searched both ways:

![The NFCorpus question "{nf["query"]}". Keyword search top 5: {"; ".join(r["title"].rstrip(".") for r in nf["keyword_top"])}. Meaning search top 5: {"; ".join(r["title"].rstrip(".") for r in nf["meaning_top"])}. The paper marked relevant is #2 by keyword and #4 by meaning.](images/10-part1-question.png)

*Meaning search found two papers on aspartame and the brain that keyword search ranked #{nf["meaning_top"][0]["keyword_rank"]} and #{nf["meaning_top"][1]["keyword_rank"]}.*

Keyword search wants "artificial" and "sweeteners", and gets a paper on urinary tract tumours for it. Meaning search puts two papers on aspartame and the brain first: aspartame is an artificial sweetener, and the brain is what neurobiology studies. Neither title shares a word with the question.

The test set doesn't mark them relevant, though, so they earn nothing in the score. Here is how that score is calculated.

### How the score works: nDCG@10

Every result in this post is scored with **nDCG@10**, a number from 0 to 1 for how good the first 10 results are.

- **@10**: only the first 10 results count.
- **Gain**: each relevant result earns points. NFCorpus grades them: a very relevant paper is worth 2 points, a somewhat relevant one 1.
- **Discounted**: points shrink the lower a result appears.
- **Normalised**: the total is divided by the score of a perfect top 10, so the result runs from 0 to 1.

![How points are discounted by position: position 1 divides by 1, position 2 by {f(disc(2), 2)}, 3 by 2, 4 by {f(disc(4), 2)}, 5 by {f(disc(5), 2)}, 10 by {f(disc(10), 2)}. Worked example for "{nf["query"]}": a perfect order scores {f(ideal, 2)} points. Keyword search put the 2-point paper at #{kw_pos} for {f(2 / disc(kw_pos), 2)} points, nDCG@10 {f(nf["ndcg@10"]["keyword"])}. Meaning search put it at #{mn_pos} for {f(2 / disc(mn_pos), 2)} points, nDCG@10 {f(nf["ndcg@10"]["meaning"])}.](images/11-ndcg.png)

*Both searches found the same paper and missed the same two. Only its position differs.*

So by this question's labels, keyword search wins: {f(nf["ndcg@10"]["keyword"])} against {f(nf["ndcg@10"]["meaning"])}. NFCorpus labels come from which papers a NutritionFacts.org article cited, not from someone judging every paper. One question settles nothing, so next, all of them.

---

## {queries:,} questions: keyword against meaning

The same three test sets from the BEIR benchmark as in part 1: **NFCorpus** (medical research), **SciFact** (scientific claims checked against abstracts) and **FiQA** (finance questions answered on a forum). **Recall@100** below is the share of all relevant documents found anywhere in the top 100.

![Results. {" ".join(f'{NAMES[k]}: {D[k]["documents"]:,} documents, {D[k]["queries"]} questions, nDCG@10 keyword {f(D[k]["keyword"]["ndcg@10"])} and meaning {f(D[k]["exact"]["ndcg@10"])}, Recall@100 {f(D[k]["keyword"]["recall@100"])} to {f(D[k]["exact"]["recall@100"])}, meaning better on {D[k]["meaning_vs_keyword"]["better"]} questions, same on {D[k]["meaning_vs_keyword"]["same"]}, worse on {D[k]["meaning_vs_keyword"]["worse"]}.' for k in ("nfcorpus", "scifact", "fiqa"))}](images/12-results.png)

*Green marks the higher score in each row.*

Dataset by dataset:

- **FiQA**: meaning search wins by {f(gain["fiqa"])} nDCG@10, about half as much again. People ask about money in everyday words and get answered in other everyday words.
- **SciFact**: keyword search wins the top 10 by {f(-gain["scifact"])}. The claims use the abstracts' own technical terms, which is where word matching is strongest.
- **NFCorpus**: about even at the top, but meaning search finds more of the relevant papers in its top 100.

Published results show the same split. In the [BEIR paper](https://arxiv.org/abs/2104.08663), the meaning-search model TAS-B scored 0.319 against BM25's 0.325 on NFCorpus, 0.643 against 0.665 on SciFact, and 0.300 against 0.236 on FiQA.

---

## When meaning search wins

The table suggests a simple explanation: meaning search helps when the question and the answer use different words. We can test that. For every question we measured what share of its words appear in its relevant documents, then grouped the questions by that share.

![Questions grouped by the share of their words found in the relevant documents. Under a third: {groups["low"]["queries"]} questions, keyword nDCG@10 {f(groups["low"]["keyword"])}, meaning {f(groups["low"]["meaning"])}, meaning won {groups["low"]["better"]} and lost {groups["low"]["worse"]}. A third to two thirds: {groups["middle"]["queries"]} questions, keyword {f(groups["middle"]["keyword"])}, meaning {f(groups["middle"]["meaning"])}, won {groups["middle"]["better"]}, lost {groups["middle"]["worse"]}. Two thirds or more: {groups["high"]["queries"]} questions, keyword {f(groups["high"]["keyword"])}, meaning {f(groups["high"]["meaning"])}, won {groups["high"]["better"]}, lost {groups["high"]["worse"]}.](images/13-word-overlap.png)

*Few shared words: meaning search wins. Most words shared: keyword search is hard to beat.*

Four questions from the extremes:

- **"{D["nfcorpus"]["biggest_wins"][2]["query"]}"**: keyword search matched "preventing" and put "{D["nfcorpus"]["biggest_wins"][2]["keyword_first"]["title"].rstrip(".")}" first. The relevant paper says "cataract", not "cataracts"; keyword search had it at #{D["nfcorpus"]["biggest_wins"][2]["relevant"][0]["keyword_rank"]}, meaning search at #1.
- **"{D["scifact"]["biggest_wins"][4]["query"]}"**: the evidence, "{D["scifact"]["biggest_wins"][4]["relevant"][0]["title"].rstrip(".")}", talks about blood pressure and glucose tolerance. Keyword search didn't have it in its top 100; meaning search ranked it first.
- **"{D["nfcorpus"]["biggest_losses"][0]["query"]}"**: one rare word. Keyword search found the relevant paper at #1. To the model, kohlrabi is roughly "some plant": it returned a paper on hibiscus and had the relevant one at #{D["nfcorpus"]["biggest_losses"][0]["relevant"][0]["meaning_rank"]}.
- **"{D["scifact"]["biggest_losses"][1]["query"]}"**: PPM1D is a gene. Keyword search ranked both relevant papers first and second. Meaning search drifted to p53, the better-known gene, and missed both in its top 10.

Names, codes, gene symbols and rare terms are keyword search's strength and meaning search's weakness. That's why most production systems run both and merge the two lists. Merging is part 3, [Reciprocal Rank Fusion, Without the Scary Math](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505).

---

## How much of each document the model reads

A model on its own is just a file of numbers. Something has to cut the text into word pieces, feed them in and read the 384 numbers out. In this lab that's **LangChain4j**, a Java library for working with AI models. It ships with its own copy of all-MiniLM-L6-v2 and of the settings file that decides where text is cut.

- **What should happen**: the model can take 512 word pieces, and its makers chose to cut at 256.
- **What happens in LangChain4j**: its settings file cuts every text at 128 pieces, roughly the first 100 words, before the model sees it.
- **How we know**: FiQA's longest document is {probe["tokens"]:,} pieces long. Its arrow and the arrow of just its first 126 pieces have a cosine of {f(probe["shipped_vs_its_first_126_tokens"], 4)}. They're the same arrow.
- **How much it matters**: {pct(L["nfcorpus"]["share_longer_than"]["128"])} of NFCorpus papers, {pct(L["scifact"]["share_longer_than"]["128"])} of SciFact abstracts and {pct(L["fiqa"]["share_longer_than"]["128"])} of FiQA answers are longer than 128 pieces.

We found this while checking why our SciFact score was lower than the published one. So we embedded every document twice more: cut at 256, and not cut at all, letting LangChain4j split long texts into parts and average them.

![Meaning search nDCG@10 by how much of each document the model reads. {" ".join(f'{NAMES[k]} ({pct(L[k]["share_longer_than"]["128"])} of documents longer than 128 pieces): first 128 pieces {f(L[k]["ndcg@10"]["128"])}, first 256 {f(L[k]["ndcg@10"]["256"])}, whole document {f(L[k]["ndcg@10"]["none"])}.' for k in ("nfcorpus", "scifact", "fiqa"))} MTEB publishes 0.316, 0.645 and 0.369 for this model at 256.](images/14-length.png)

*At 256 pieces, all three match the published scores. The model was fine; the 128 cut was the whole gap.*

Reading more isn't automatically better, though. SciFact's abstracts gained {signed(L["scifact"]["ndcg@10"]["none"] - L["scifact"]["ndcg@10"]["128"])} from being read in full. NFCorpus and FiQA did slightly worse read in full than at 256 pieces. A whole long document becomes one average of several parts, the blending from the paragraph example, and the model was trained on at most 128 pieces. We didn't measure which of the two matters more.

So check how many word pieces your library actually gives the model, rather than trusting the model card alone, and test the limit on your own data.

---

## Finding the nearest arrows fast

Comparing the question with every document is called exact search. On FiQA that's {D["fiqa"]["documents"]:,} comparisons of 384 numbers each, for every question.

Vector databases skip most of that with an **HNSW** index (Hierarchical Navigable Small World). Each document is linked to a few of its nearest neighbours, in layers: a sparse top layer with long links, and a bottom layer with every document. A search starts at the top, hops to whichever neighbour is closest to the question, drops a layer when no neighbour is closer, and repeats. It's like crossing a country on motorways first and side streets last. It looks at a small fraction of the documents and can occasionally miss a close one.

![HNSW against exact search. {" ".join(f'{NAMES[k]}, {D[k]["documents"]:,} documents: same top 10 for {pct(D[k]["hnsw_vs_exact"]["same_top10_share"])} of questions, nDCG@10 {f(D[k]["exact"]["ndcg@10"])} exact and {f(D[k]["hnsw"]["ndcg@10"])} HNSW, {f(D[k]["timing_ms"]["exact"], 1)} ms exact and {f(D[k]["timing_ms"]["hnsw"], 1)} ms HNSW.' for k in ("nfcorpus", "scifact", "fiqa"))}](images/15-hnsw.png)

*HNSW stays near 1 ms as the collection grows. Exact search doesn't.*

HNSW gave the same top 10 as exact search for {hnsw_same:,} of the {queries:,} questions, and cost at most {f(D["fiqa"]["exact"]["ndcg@10"] - D["fiqa"]["hnsw"]["ndcg@10"])} nDCG@10. For a few thousand documents, exact search is fine. For millions, it isn't.

---

## Cosine, dot product or distance?

Vector databases make you choose how to compare arrows. OpenSearch offers cosine (`cosinesimil`), dot product (`innerproduct`) and straight-line distance (`l2`).

![The five sentences compared with "{e["question"]}" three ways. {"; ".join(f'{S[i]["text"]} cosine {f(S[i]["cosine"])}, dot product {f(S[i]["dot"])}, distance {f(S[i]["euclidean"])}' for i in order)}. The order is the same all three ways.](images/16-metrics.png)

*Dot product equals cosine here, and distance falls exactly as cosine rises.*

The order never changes, because this model scales every arrow to length 1. Step 4 divides by 1 × 1, so the dot product *is* the cosine. Across all {queries:,} questions, the three gave the same top 10 {metrics_same:,} times out of {queries:,}. With length-1 arrows, use the dot product; it's the cheapest. With other arrows, use cosine, because the dot product favours long arrows.

---

## In code

With LangChain4j, the model runs inside the Java process, with no server and no API key:

```java
EmbeddingModel model = new AllMiniLmL6V2EmbeddingModel();
float[] vector = model.embed("How can I lower my blood pressure?").content().vector(); // 384 numbers
```

In OpenSearch, the vectors sit in a `knn_vector` field next to the text, so one index serves both keyword and meaning search:

```json
PUT /nfcorpus
{{
  "settings": {{ "index": {{ "knn": true }} }},
  "mappings": {{
    "properties": {{
      "title":     {{ "type": "text" }},
      "text":      {{ "type": "text" }},
      "embedding": {{
        "type": "knn_vector", "dimension": 384,
        "method": {{ "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil" }}
      }}
    }}
  }}
}}
```

Then search with the question's vector:

```json
POST /nfcorpus/_search
{{ "size": 10, "query": {{ "knn": {{ "embedding": {{ "vector": [-0.0321, 0.1206, ...], "k": 10 }} }} }} }}
```

---

## Summary

- **An embedding is an arrow of 384 numbers.** Cosine similarity compares directions: multiply, add, divide by the lengths.
- **Words, sentences and paragraphs go through one recipe.** Split into pieces, adjust each piece by its neighbours, average. A longer text gives a more blended arrow, not a richer one.
- **It finds answers written in other words**, and blurs names, numbers, word order and "not".
- **It wins when the answers share few of the question's words**: FiQA {signed(gain["fiqa"])}, SciFact {signed(gain["scifact"])}, and {groups["low"]["better"]} questions won to {groups["low"]["worse"]} lost when few words are shared.
- **Check how much text your library feeds the model.** Ours gave it 128 word pieces. At 256, our scores match the published ones.
- **HNSW is nearly free**: the same top 10 as exact search for {pct(hnsw_same / queries)} of questions, in about 1 ms.

The next part, [Reciprocal Rank Fusion](https://medium.com/jinternals/reciprocal-rank-fusion-without-the-scary-math-68421476c505), merges keyword search and meaning search into one list.

---

## Run it yourself

The lab is Java 21 with OpenSearch in Docker, in [ai-explained-with-experiments](https://github.com/jinternals/ai-explained-with-experiments/tree/main/SemanticSearch) on GitHub. The first run downloads the three test sets. The model is bundled in LangChain4j, so there's nothing else to install and no API key.

```bash
docker compose run --rm --build lab example     # the worked examples
docker compose run --rm --build lab benchmark   # 1,271 questions, 3 datasets
docker compose run --rm --build lab length      # how much of each document the model reads
```

## Sources

- D. Jurafsky and J. H. Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/5.pdf), chapter "Embeddings": the distributional hypothesis and cosine similarity.
- T. Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), 2013 (word2vec).
- N. Reimers and I. Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084), EMNLP 2019.
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), model card.
- N. Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663), NeurIPS 2021 Datasets and Benchmarks.
- [MTEB results for all-MiniLM-L6-v2](https://github.com/embeddings-benchmark/results/tree/main/results/sentence-transformers__all-MiniLM-L6-v2).
- Y. Malkov and D. Yashunin, [HNSW](https://arxiv.org/abs/1603.09320), IEEE TPAMI 2018.
- O. Weller et al., [NevIR: Negation in Neural Information Retrieval](https://arxiv.org/abs/2305.07614), EACL 2024.
- Spring AI, [Understanding Vectors](https://docs.spring.io/spring-ai/reference/api/vectordbs/understand-vectordbs.html).
"""
# Medium cuts alt text at 500 characters, so the longest ones get shorter versions with the same numbers.
short = {k: {"without medication": S[1], "salt and hypertension": S[0], "measuring blood pressure": S[2], "tyre pressure": S[3], "stock market": S[4]}[k]
         for k in ("without medication", "salt and hypertension", "measuring blood pressure", "tyre pressure", "stock market")}
by_cos = sorted(short.items(), key=lambda kv: -kv[1]["cosine"])
ALT = {
    "01-two-rankings.png": f'Five sentences ranked for "{e["question"]}". By shared words: without medication, measuring blood pressure and tyre pressure share 2 words each; salt and hypertension, and the stock market, share none. By meaning (cosine): ' + ", ".join(f'{k} {f(v["cosine"])}' for k, v in by_cos) + ".",
    "10-part1-question.png": f'Top 5 for the NFCorpus question "{nf["query"]}". Keyword search: toxicity of artificial sweeteners; sucrose vs sweetener taste (relevant); sugar substitutes; sweeteners and urinary tract tumours; food dyes and ADHD. Meaning search: aspartame\'s effects on the brain; neurologic effects of aspartame; sugar substitutes; sucrose vs sweetener taste (relevant); cherry phenolics and neurons.',
    "12-results.png": "Results. " + " ".join(
        f'{NAMES[k]}, {D[k]["queries"]} questions: nDCG@10 keyword {f(D[k]["keyword"]["ndcg@10"])}, meaning {f(D[k]["exact"]["ndcg@10"])}; Recall@100 {f(D[k]["keyword"]["recall@100"])} to {f(D[k]["exact"]["recall@100"])}; meaning better, same, worse on {D[k]["meaning_vs_keyword"]["better"]}, {D[k]["meaning_vs_keyword"]["same"]}, {D[k]["meaning_vs_keyword"]["worse"]} questions.'
        for k in ("nfcorpus", "scifact", "fiqa")),
    "16-metrics.png": f'The five sentences against "{e["question"]}" as cosine, dot product and distance: ' + "; ".join(
        f'{k} {f(v["cosine"])}, {f(v["dot"])}, {f(v["euclidean"])}' for k, v in by_cos) + ". The order is the same all three ways.",
}
for file, alt in ALT.items():
    assert len(alt) <= 500, file
    start = story.index(f"](images/{file})")
    open_bracket = story.rindex("![", 0, start)
    story = story[:open_bracket] + f"![{alt}" + story[start:]
(HERE / "semantic-search-medium.md").write_text(story)
print(f"medium/semantic-search-medium.md and {len(list(FIG.glob('[0-9]*.html')))} figures written")
