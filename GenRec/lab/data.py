"""Step 1: MovieLens 1M, split into "what the person watched before" and "what they watched next".

GenRec learns from members' histories and is scored on the next title they engage with.
MovieLens gives us the same shape of data: 6,040 people rating 3,706 movies over time.

A rating of 4 or 5 stars plays the part of a "high-quality engagement" (the paper's long
plays and thumbs-up). Every person's ratings are sorted by time, and the split is
leave-one-out on those positives:

    test   = the person's last 4-5 star rating
    valid  = the one before it
    train  = every earlier 4-5 star rating

The input for any target is *everything* the person rated before it, low ratings included.
Deciding what to keep from that history is step 3's job (context engineering).

    python lab/data.py

Downloads the data on the first run (6 MB) and writes out/data.json.
"""
import io, json, urllib.request, zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ml-1m"
OUT = ROOT / "out"
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
POSITIVE = 4  # ratings at or above this count as a high-quality engagement

AGES = {1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
JOBS = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student",
        "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
        "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist",
        "self-employed", "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]


@dataclass
class Event:
    movie: int
    rating: int
    ts: int


@dataclass
class Example:
    """Predict history[user][index] from history[user][:index]."""
    user: int
    index: int


@dataclass
class Dataset:
    movies: dict    # movie id -> {"title", "year", "genres"}
    users: dict     # user id -> {"gender", "age", "job"}
    history: dict   # user id -> [Event], oldest first
    catalog: list   # movie ids that have at least one rating, sorted
    train: list     # [Example]
    valid: list
    test: list

    def target(self, ex):
        return self.history[ex.user][ex.index].movie

    def past(self, ex):
        return self.history[ex.user][:ex.index]


def download():
    if (DATA / "ratings.dat").exists():
        return
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA.parent)


def read(name):
    # The .dat files are Latin-1 (film titles such as "Café au Lait") and use "::" as separator.
    return [line.split("::") for line in (DATA / name).read_text(encoding="latin-1").splitlines()]


def load():
    download()
    movies = {}
    for mid, title, genres in read("movies.dat"):
        name, year = title, None
        if title.endswith(")") and title[-6] == "(":
            name, year = title[:-7], int(title[-5:-1])
        movies[int(mid)] = {"title": name, "year": year, "genres": genres.split("|")}

    users = {int(uid): {"gender": "female" if g == "F" else "male", "age": AGES[int(age)], "job": JOBS[int(job)]}
             for uid, g, age, job, _zip in read("users.dat")}

    history = {}
    for uid, mid, rating, ts in read("ratings.dat"):
        history.setdefault(int(uid), []).append(Event(int(mid), int(rating), int(ts)))
    for events in history.values():
        events.sort(key=lambda e: e.ts)  # stable: ratings made in the same second keep file order

    catalog = sorted({e.movie for events in history.values() for e in events})

    train, valid, test = [], [], []
    for uid, events in history.items():
        positives = [i for i, e in enumerate(events) if e.rating >= POSITIVE]
        if len(positives) < 3:
            continue
        test.append(Example(uid, positives[-1]))
        valid.append(Example(uid, positives[-2]))
        train += [Example(uid, i) for i in positives[:-2] if i > 0]  # needs at least one past event
    return Dataset(movies, users, history, catalog, train, valid, test)


def like_counts(d):
    """How many people rated each catalogue movie 4-5 stars, leaving out the validation and test answers."""
    held_out = {(ex.user, ex.index) for ex in d.valid + d.test}
    counts = Counter(e.movie for u, events in d.history.items() for i, e in enumerate(events)
                     if e.rating >= POSITIVE and (u, i) not in held_out)
    return [counts[m] for m in d.catalog]


def training_pool(d, n=40_000, recent=3, seed=0):
    """A fixed, shuffled sample of training examples: each person's `recent` latest ones (closest
    to what the test asks), topped up with random older ones. Training runs and the head probe
    take slices of it, so they all see the same examples in the same order."""
    import random
    by_person = {}
    for ex in d.train:
        by_person.setdefault(ex.user, []).append(ex)
    chosen = [ex for exs in by_person.values() for ex in exs[-recent:]]
    rest = [ex for exs in by_person.values() for ex in exs[:-recent]]
    rng = random.Random(seed)
    chosen += rng.sample(rest, max(0, n - len(chosen)))
    return rng.sample(chosen, min(n, len(chosen)))


def percentiles(values, ps=(10, 50, 90)):
    values = sorted(values)
    return {f"p{p}": values[min(len(values) - 1, len(values) * p // 100)] for p in ps}


if __name__ == "__main__":
    d = load()
    ratings = [e for events in d.history.values() for e in events]
    same_second = sum(1 for events in d.history.values() for a, b in zip(events, events[1:]) if a.ts == b.ts)
    years = sorted(e.ts for e in ratings)

    stats = {
        "people": len(d.history),
        "movies_in_file": len(d.movies),
        "catalog": len(d.catalog),
        "ratings": len(ratings),
        "stars": {str(k): v for k, v in sorted(Counter(e.rating for e in ratings).items())},
        "positive_share": round(sum(e.rating >= POSITIVE for e in ratings) / len(ratings), 3),
        "people_with_3_positives": len(d.test),
        "examples": {"train": len(d.train), "valid": len(d.valid), "test": len(d.test)},
        "ratings_per_person": percentiles([len(v) for v in d.history.values()]),
        "past_ratings_at_test": percentiles([ex.index for ex in d.test]),
        "same_second_share": round(same_second / (len(ratings) - len(d.history)), 3),
        "first_rating": years[0],
        "last_rating": years[-1],
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"{stats['people']:,} people, {stats['catalog']:,} movies rated, {stats['ratings']:,} ratings")
    print("Stars:", ", ".join(f"{k}★ {v:,}" for k, v in stats["stars"].items()),
          f"-> {stats['positive_share']:.1%} are 4-5★")
    print(f"Examples: train {len(d.train):,}, valid {len(d.valid):,}, test {len(d.test):,}")
    print("Ratings per person:", stats["ratings_per_person"])
    print("Past ratings before the test target:", stats["past_ratings_at_test"])
    print(f"Ratings made in the same second as the previous one: {stats['same_second_share']:.1%}")

    ex = d.test[0]
    print(f"\nPerson {ex.user} ({d.users[ex.user]}), last 5 of {ex.index} past ratings:")
    for e in d.past(ex)[-5:]:
        m = d.movies[e.movie]
        print(f"  {e.rating}★  {m['title']} ({m['year']})  {'/'.join(m['genres'])}")
    m = d.movies[d.target(ex)]
    print(f"  -> next 4-5★ movie (the answer): {m['title']} ({m['year']})")
    print("\nWrote out/data.json")
