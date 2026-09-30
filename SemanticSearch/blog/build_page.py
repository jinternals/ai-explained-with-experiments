"""Assemble blog/index.html from blog/page/, filling every number from out/*.json.

    python3 blog/build_page.py           # fails if any number is missing
    python3 blog/build_page.py --draft   # marks missing numbers instead of failing

Standard library only. In blog/page/body.html a number is written as a placeholder:

    {{b:datasets.fiqa.exact.ndcg@10|3}}          value from out/benchmark.json, 3 decimals
    {{e:sentences.0.cosine|3}}                   value from out/example.json
    {{l:datasets.scifact.ndcg@10.256|3}}         value from out/length.json
    {{d:gain.fiqa|signed3}}                      a value derived below, from the JSON

Formats: a number of decimals, "int" (thousands separators), "pct" (0.123 -> 12%),
"pct1" (0.1234 -> 12.3%), "signed1" etc. (always show + or −), "words" (a list, comma-separated),
"text".
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


sources = {"b": load("benchmark.json"), "e": load("example.json"), "l": load("length.json")}


def derived(bench, example, length):
    """Values the text needs that are simple arithmetic on the results."""
    d = {"gain": {}, "hnsw_loss": {}, "groups": {}, "total": {}}
    datasets = bench.get("datasets", {})
    if datasets:
        queries = sum(ds["queries"] for ds in datasets.values())
        d["total"] = {
            "queries": queries,
            "hnsw_same_top10": sum(ds["hnsw_vs_exact"]["same_top10"] for ds in datasets.values()),
            "metrics_same_top10": sum(ds["metrics_agree"]["same_top10"] for ds in datasets.values()),
            "better": sum(ds["meaning_vs_keyword"]["better"] for ds in datasets.values()),
            "worse": sum(ds["meaning_vs_keyword"]["worse"] for ds in datasets.values()),
        }
        d["total"]["hnsw_same_share"] = d["total"]["hnsw_same_top10"] / queries
        for name, ds in datasets.items():
            d["gain"][name] = ds["exact"]["ndcg@10"] - ds["keyword"]["ndcg@10"]
            d["hnsw_loss"][name] = ds["exact"]["ndcg@10"] - ds["hnsw"]["ndcg@10"]
        # The word-overlap groups, added up over all three datasets.
        for group in ("low", "middle", "high"):
            rows = [ds["by_word_overlap"][group] for ds in datasets.values()]
            n = sum(r["queries"] for r in rows)
            d["groups"][group] = {
                "queries": n,
                "keyword": sum(r["keyword"] * r["queries"] for r in rows) / n,
                "meaning": sum(r["meaning"] * r["queries"] for r in rows) / n,
                "better": sum(r["meaning_better"] for r in rows),
                "worse": sum(r["meaning_worse"] for r in rows),
            }
        # Where HNSW's top 10 wasn't identical to the exact one: how many documents differed, on average.
        changed = sum((1 - ds["hnsw_vs_exact"]["top10_overlap"]) * 10 * ds["queries"] for ds in datasets.values())
        d["avg_docs_changed"] = changed / (queries - d["total"]["hnsw_same_top10"])
        fiqa = datasets["fiqa"]["timing_ms"]
        d["fiqa_exact_vs_hnsw"] = fiqa["exact"] / fiqa["hnsw"]
        d["docs_ratio"] = datasets["fiqa"]["documents"] / datasets["nfcorpus"]["documents"]
        d["exact_time_ratio"] = fiqa["exact"] / datasets["nfcorpus"]["timing_ms"]["exact"]
    averages = [x["cosine_of_word_averages"] for x in example.get("levels", {}).get("sentences", [])]
    if averages:
        d["word_avg_min"], d["word_avg_max"] = min(averages), max(averages)
    for item in example.get("levels", {}).get("pieces", []):
        if item["text"] == "hypertension":
            d["hypertension_pieces"] = [p for p in item["pieces"] if p not in ("[CLS]", "[SEP]")]
    if example.get("nfcorpus"):
        d["example_relevant"] = len(example["nfcorpus"]["relevant"])
    if example.get("sentences"):
        s = example["sentences"][1]
        # For vectors of length 1: distance squared = 2 - 2 x cosine.
        d["dist_squared"] = s["euclidean"] ** 2
        d["two_minus_two_cos"] = 2 - 2 * s["cosine"]
        # OpenSearch reports a cosinesimil match as (1 + cosine) / 2.
        d["opensearch_score"] = (1 + s["cosine"]) / 2
    if length.get("datasets"):
        d["length_gain"] = {name: {k: v - ds["ndcg@10"]["128"] for k, v in ds["ndcg@10"].items()}
                            for name, ds in length["datasets"].items()}
    return d


sources["d"] = derived(sources["b"], sources["e"], sources["l"])
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
    if spec == "raw":
        return str(value)
    if spec == "json":
        return html.escape(json.dumps(value))
    if spec == "chips":
        return "".join(f"<li>{html.escape(w)}</li>" for w in value) if value else '<li class="none">none</li>'
    if spec == "words":
        return html.escape(", ".join(value)) if value else "none"
    if spec == "int":
        return f"{int(value):,}"
    if spec == "pct":
        return f"{half_up(100 * value, 0)}%"
    if spec == "pct1":
        return f"{half_up(100 * value, 1)}%"
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
        # raw and json values sit inside HTML attributes, where a <mark> would break the markup.
        return "" if spec in ("raw", "json") else '<mark class="pending">…</mark>'


page = BLOG / "page"
body = re.sub(r"\{\{([bedl]):([^|}]+)(?:\|([^}]+))?\}\}", fill, (page / "body.html").read_text())
# Sentences in the text that can only be written once the lab has run are marked TO_WRITE.
if "TO_WRITE" in body:
    missing.append("text still marked TO_WRITE")
if missing and not DRAFT:
    sys.exit("Missing numbers (run the lab first, or use --draft):\n  " + "\n  ".join(sorted(set(missing))))

html_out = (body.replace("{{CSS}}", (page / "base.css").read_text())
                .replace("{{JS}}", (page / "runtime.js").read_text()))
(BLOG / "index.html").write_text(html_out)
note = f", {len(set(missing))} numbers still pending" if missing else ""
print(f"blog/index.html written ({len(html_out) // 1024} KB){note}")
