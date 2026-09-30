"""Assemble blog/index.html from blog/page/, filling every number from out/*.json.

    python3 blog/build_page.py           # fails if any number is missing
    python3 blog/build_page.py --draft   # marks missing numbers instead of failing

Standard library only. In blog/page/body.html a number is written as a placeholder:

    {{b:datasets.nfcorpus.ndcg@10.rrf.100|3}}   value from out/benchmark.json, 3 decimals
    {{e:documents.0.cross_encoder_score|3}}      value from out/example.json
    {{d:gain_pct.nfcorpus.rrf|0}}                a value derived below, from the JSON

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
    d = {"gain": {}, "gain_pct": {}, "best_depth": {}, "total": {}}
    queries = pairs = 0
    for name, ds in bench.get("datasets", {}).items():
        queries += ds["queries"]
        pairs += ds["pairs_scored"]
        for stage, by_depth in ds["ndcg@10"].items():
            before, after = by_depth["0"], by_depth["100"]
            d["gain"].setdefault(name, {})[stage] = after - before
            d["gain_pct"].setdefault(name, {})[stage] = 100 * (after - before) / before
            d["best_depth"].setdefault(name, {})[stage] = max(
                (n for n in by_depth if n != "0"), key=lambda n: by_depth[n])
    if queries:
        d["total"] = {"queries": queries, "pairs": pairs}
    return d


sources["d"] = derived(sources["b"])
missing = []


def lookup(source, path):
    value = sources[source]
    for key in path.split("."):
        if isinstance(value, list):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(path)
    if value is None:
        raise KeyError(path)
    return value


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
body = re.sub(r"\{\{([bed]):([^|}]+)(?:\|([^}]+))?\}\}", fill, (page / "body.html").read_text())
if missing and not DRAFT:
    sys.exit("Missing numbers (run the lab first, or use --draft):\n  " + "\n  ".join(sorted(set(missing))))

html_out = (body.replace("{{CSS}}", (page / "base.css").read_text())
                .replace("{{JS}}", (page / "runtime.js").read_text()))
(BLOG / "index.html").write_text(html_out)
note = f", {len(set(missing))} numbers still pending" if missing else ""
print(f"blog/index.html written ({len(html_out) // 1024} KB){note}")
