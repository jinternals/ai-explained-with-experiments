"""Step 3: write a person's history as the chat text the LLM reads (the paper's "verbalization").

GenRec turns each member's history into a conversation: a user message with context, profile,
history and a task, and an assistant message with what the member actually did next. This
file does the same for MovieLens, with the paper's context-engineering choices as settings:

    recent    how many of the latest ratings are written out one by one ("retain in full")
    summary   one line that sums up everything older ("summarize or compress")
    genres    which ratings get their genres ("elaborate selectively"): all, liked or none
    drop      star values left out of the written-out part ("omit low-signal events")

Ratings made on the same day share one line, so the date isn't repeated.

    python lab/verbalize.py

Prints an example and token counts for several settings, and writes out/verbalize.json.
"""
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from data import OUT, POSITIVE, load

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
SYSTEM = "You recommend movies."
TASK = "Which movie will this member watch next and rate 4 or 5 stars?"
ARTICLES = ("The", "A", "An", "La", "Le", "Les", "L'", "Il", "El", "Das", "Der", "Die")


@dataclass(frozen=True)
class Style:
    recent: int | None = 50   # None = write out every rating
    summary: bool = True
    genres: str = "liked"     # "all", "liked" or "none"
    drop: tuple = ()          # e.g. (3,) leaves 3-star ratings out of the written-out part


STYLES = {
    "everything":           Style(recent=None, summary=False, genres="all"),
    "everything, no genres": Style(recent=None, summary=False, genres="none"),
    "last 100 + summary":   Style(recent=100),
    "last 50 + summary":    Style(recent=50),
    "last 50, no 3★":       Style(recent=50, drop=(3,)),
    "last 20 + summary":    Style(recent=20),
}
DEFAULT = STYLES["last 50 + summary"]


def title(movie):
    """'Bug's Life, A' -> 'A Bug's Life', keeping any alternative title in brackets."""
    name, alt = movie["title"], ""
    if " (" in name:
        name, alt = name.split(" (", 1)
        alt = " (" + alt
    for article in ARTICLES:
        if name.endswith(", " + article):
            name = article + ("" if article.endswith("'") else " ") + name[: -len(article) - 2]
            break
    return f"{name}{alt} ({movie['year']})"


def day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def month(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%b %Y")


def top_genres(movies, n):
    counts = Counter(g for m in movies for g in m["genres"])
    return ", ".join(g for g, _ in counts.most_common(n)) or "none"


def summarise(d, older):
    liked = [d.movies[e.movie] for e in older if e.rating >= POSITIVE]
    disliked = [d.movies[e.movie] for e in older if e.rating <= 2]
    first, last = month(older[0].ts), month(older[-1].ts)
    when = first if first == last else f"{first} to {last}"
    text = f"Older history ({len(older)} ratings, {when}): liked {len(liked)}"
    if liked:
        text += f", mostly {top_genres(liked, 3)}"
    text += f"; disliked {len(disliked)}"
    if disliked:
        text += f", mostly {top_genres(disliked, 2)}"
    return text + "."


def line(d, e, style):
    m = d.movies[e.movie]
    text = f"{title(m)} {e.rating}★"
    if style.genres == "all" or (style.genres == "liked" and e.rating >= POSITIVE):
        text += f" [{', '.join(m['genres'])}]"
    return text


def messages(d, ex, style=DEFAULT, answer=True):
    """The chat for one example: system, user (context + history + task) and, optionally, the answer."""
    past = d.past(ex)
    cut = 0 if style.recent is None else max(0, len(past) - style.recent)
    older, recent = past[:cut], [e for e in past[cut:] if e.rating not in style.drop]

    u = d.users[ex.user]
    parts = [f"Member: {u['gender']}, {u['age']}, {u['job']}.",
             f"Today: {day(d.history[ex.user][ex.index].ts)}."]
    if older and style.summary:
        parts.append(summarise(d, older))
    parts.append("Ratings, oldest first (1-5 stars):")
    days = {}
    for e in recent:
        days.setdefault(day(e.ts), []).append(line(d, e, style))
    parts += [f"{date}: " + "; ".join(items) for date, items in days.items()]
    parts.append(TASK)

    chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]
    if answer:
        chat.append({"role": "assistant", "content": title(d.movies[d.target(ex)])})
    return chat


def tokenizer():
    from mlx_lm.utils import load_tokenizer
    from huggingface_hub import snapshot_download
    return load_tokenizer(snapshot_download(MODEL))


def prompt_tokens(tok, d, ex, style=DEFAULT):
    """Token ids the model reads before it scores the catalogue (everything up to the answer)."""
    return tok.apply_chat_template(messages(d, ex, style, answer=False), add_generation_prompt=True)


if __name__ == "__main__":
    from data import percentiles

    d = load()
    tok = tokenizer()

    ex = d.test[0]
    chat = messages(d, ex, DEFAULT)
    print("=" * 72)
    print(f"Person {ex.user}, test example, style 'last 50 + summary'")
    print("=" * 72)
    for m in chat:
        print(f"[{m['role']}]\n{m['content']}\n")

    long_ex = max(d.test, key=lambda e: e.index)
    print("=" * 72)
    print(f"Longest history: person {long_ex.user}, {long_ex.index} past ratings. First lines of the user message:")
    print("=" * 72)
    print("\n".join(messages(d, long_ex, DEFAULT)[1]["content"].splitlines()[:5]))

    results = {}
    print(f"\nTokens per test prompt ({len(d.test):,} people, {MODEL.split('/')[-1]} tokenizer)")
    print(f"  {'style':<24}{'p10':>7}{'median':>8}{'p90':>8}{'max':>8}{'mean':>8}")
    for name, style in STYLES.items():
        counts = [len(prompt_tokens(tok, d, e, style)) for e in d.test]
        p = percentiles(counts)
        mean = sum(counts) / len(counts)
        results[name] = asdict(style) | p | {"max": max(counts), "mean": round(mean)}
        print(f"  {name:<24}{p['p10']:>7,}{p['p50']:>8,}{p['p90']:>8,}{max(counts):>8,}{mean:>8,.0f}")

    answers = [len(tok.encode(title(d.movies[d.target(e)]))) for e in d.test]
    results["answer_tokens"] = percentiles(answers) | {"max": max(answers)}
    print(f"\nAnswer (assistant message) tokens: median {results['answer_tokens']['p50']}, max {max(answers)}")

    OUT.mkdir(exist_ok=True)
    (OUT / "verbalize.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Wrote out/verbalize.json")
