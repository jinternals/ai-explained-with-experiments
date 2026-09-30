"""Assemble blog/index.html from blog/page/, filling every number from out/*.json.

    python3 blog/build_page.py           # fails if any number is missing
    python3 blog/build_page.py --draft   # marks missing numbers instead of failing

Standard library only. In blog/page/body.html a number is written as a placeholder:

    {{b:datasets.nfcorpus.ndcg@10.rrf.100|3}}   value from out/benchmark.json, 3 decimals
    {{e:documents.0.cross_encoder_score|3}}      value from out/example.json
    {{d:gain_pct.nfcorpus.stemming|0}}           a value derived below, from the JSON
    {{grid:nfcorpus.english}}                     the k1 x b grid as a shaded table

Formats: a number of decimals, "int" (thousands separators), "pct" (0.123 -> 12%),
"signed1" etc. (always show + or −), "text".
"""
import html
from decimal import ROUND_HALF_UP, Decimal
import json
import re
import sys
from pathlib import Path

BLOG = Path(__file__).resolve().parent
OUT = BLOG.parent / "out"
DRAFT = "--draft" in sys.argv


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else {}


sources = {"b": load("benchmark.json"), "e": load("example.json")}


def derived(bench):
    """Values the text needs that are simple arithmetic on the results."""
    d = {"gain": {}, "gain_pct": {}, "spread": {}, "total": {}}
    queries = 0
    for name, ds in bench.get("datasets", {}).items():
        queries += ds["queries"]
        std, eng = ds["analyzers"]["standard"], ds["analyzers"]["english"]
        # What stemming alone adds, at OpenSearch's default k1 and b.
        d["gain"].setdefault(name, {})["stemming"] = eng["default"]["ndcg@10"] - std["default"]["ndcg@10"]
        d["gain_pct"].setdefault(name, {})["stemming"] = 100 * d["gain"][name]["stemming"] / std["default"]["ndcg@10"]
        for analyzer, a in ds["analyzers"].items():
            # What the best k1 and b add over the defaults, and the full range across the grid.
            d["gain"][name]["tuning_" + analyzer] = a["best"]["ndcg@10"] - a["default"]["ndcg@10"]
            values = [cell["ndcg@10"] for cell in a["grid"].values()]
            d["spread"].setdefault(name, {})[analyzer] = {"min": min(values), "max": max(values)}
    if queries:
        d["total"] = {"queries": queries}
        d["max_tuning"] = max(v for name in d["gain"] for k, v in d["gain"][name].items() if k.startswith("tuning_"))
    return d


def grid_table(match):
    """{{grid:<dataset>.<analyzer>}}: the k1 x b grid of nDCG@10 as a shaded HTML table."""
    name, analyzer = match.group(1), match.group(2)
    bench = sources["b"]
    try:
        grid = bench["datasets"][name]["analyzers"][analyzer]["grid"]
        k1s, bs = bench["k1_values"], bench["b_values"]
    except KeyError:
        missing.append(f"grid:{name}.{analyzer}")
        return '<p><mark class="pending">…</mark></p>'
    values = [cell["ndcg@10"] for cell in grid.values()]
    low, high = min(values), max(values)
    default = f"k1={bench['defaults']['k1']},b={bench['defaults']['b']}"
    base = grid[default]["ndcg@10"]
    best = max(grid, key=lambda k: grid[k]["ndcg@10"])
    rows = []
    for k1 in k1s:
        cells = []
        for b in bs:
            key = f"k1={k1},b={b}"
            v = grid[key]["ndcg@10"]
            # Colour by distance from the default: purple above it, red below, scaled to each side's extreme.
            if v >= base:
                side, share = "up", 0 if high == base else (v - base) / (high - base)
            else:
                side, share = "down", 0 if low == base else (base - v) / (base - low)
            marks = (" is-default" if key == default else "") + (" is-best" if key == best else "")
            cells.append(f'<td class="heat {side}{marks}" style="--h:{share:.2f}">{fmt(v, "3")}</td>')
        rows.append(f"<tr><th>{k1}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{b}</th>" for b in bs)
    return (f'<table class="matrix grid"><thead><tr><th>k1 ↓ · b →</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


sources["d"] = derived(sources["b"])
sources["d"]["example_terms"] = len(sources["e"].get("terms", [])) or None
# OpenSearch's explanation splits TF into (k1 + 1), which it calls "boost", and the rest. The post shows them as one number.
sources["d"]["tf_full"] = [t["boost"] * t["tf"] for t in sources["e"].get("terms", [])]
missing = []


def lookup(source, path):
    """Follow a dotted path. Keys may themselves contain dots (k1 "1.2", b "0.75"), so at each level
    the longest run of path parts that is an actual key wins."""
    return _walk(sources[source], path.split("."), path)


def _walk(value, parts, path):
    if not parts:
        if value is None:
            raise KeyError(path)
        return value
    if isinstance(value, list):
        return _walk(value[int(parts[0])], parts[1:], path)
    if isinstance(value, dict):
        for n in range(len(parts), 0, -1):
            key = ".".join(parts[:n])
            if key in value:
                return _walk(value[key], parts[n:], path)
    raise KeyError(path)


def token_list(match):
    """{{tokens:<path>}}: a list of terms from example.json, as chips."""
    try:
        terms = lookup("e", match.group(1))
    except (KeyError, IndexError):
        missing.append("tokens:" + match.group(1))
        return '<li><mark class="pending">…</mark></li>'
    return "".join(f"<li>{html.escape(t)}</li>" for t in terms)


def fmt(value, spec):
    if spec == "text":
        return html.escape(str(value))
    if spec == "int":
        return f"{int(value):,}"
    if spec == "pct":
        return f"{100 * value:.0f}%"
    if spec.startswith("signed"):
        rounded = half_up(value, int(spec[6:] or 3))
        return f"{rounded:+,f}".replace("-", "−")
    return f"{half_up(value, int(spec)):,f}".replace("-", "−")


def half_up(value, digits):
    """Round 0.3525 to 0.353 like Java's printf, not to 0.352 like Python's float formatting."""
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def fill(match):
    source, path, spec = match.group(1), match.group(2), match.group(3) or "3"
    try:
        return fmt(lookup(source, path), spec)
    except (KeyError, IndexError, ValueError, TypeError):
        missing.append(f"{source}:{path}")
        return '<mark class="pending">…</mark>'


page = BLOG / "page"
body = (page / "body.html").read_text()
body = re.sub(r"\{\{grid:(\w+)\.(\w+)\}\}", grid_table, body)
body = re.sub(r"\{\{tokens:([^}]+)\}\}", token_list, body)
body = re.sub(r"\{\{([bed]):([^|}]+)(?:\|([^}]+))?\}\}", fill, body)
if missing and not DRAFT:
    sys.exit("Missing numbers (run the lab first, or use --draft):\n  " + "\n  ".join(sorted(set(missing))))

html_out = (body.replace("{{CSS}}", (page / "base.css").read_text())
                .replace("{{JS}}", (page / "runtime.js").read_text()))
(BLOG / "index.html").write_text(html_out)
note = f", {len(set(missing))} numbers still pending" if missing else ""
print(f"blog/index.html written ({len(html_out) // 1024} KB){note}")
