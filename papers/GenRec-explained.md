# GenRec, explained simply

Plain-language notes on **GenRec: An LLM-Backed Recommendation Ranker at Netflix** (Li et al., 2026). Paper: [GenRec-2608.10257v2.pdf](GenRec-2608.10257v2.pdf), [arXiv 2608.10257](https://arxiv.org/abs/2608.10257). All numbers below are quoted from the paper.

**In one line:** Netflix tried writing each member's viewing history as plain text and letting a language model choose what to recommend. It beat their long-tuned recommender in a live test, while using much less training data.

## The problem

Netflix's current ranker (the system that orders titles on your homepage) runs on thousands of hand-made **features**. A feature is a number someone designed, such as "how many thrillers this person watched in the last 30 days". Over years this system has become large and hard to change. Adding something new, like games or live events, means engineers design new features and change the model.

## The idea

Instead of turning a member's activity into thousands of numbers, **describe it in words**:

> *May 21, 2 PM, TV: played Wednesday, 30 min. May 23, 4 PM, iPad: added Stranger Things to list. May 23, 4 PM, iPad: played Mindhunter, 2 hours…*

Then give that text to an LLM and let it work out the patterns. Engineers no longer design features. Their job becomes deciding **what to put in the text**, which the paper calls moving "from feature engineering to context engineering".

## How it's trained: two phases

| | Phase 1 | Phase 2 (this paper) |
|---|---|---|
| What it does | Takes an open-source LLM and teaches it about Netflix: its titles and how members behave | Teaches that model one job: putting titles in the best order |
| How often | Rarely, because it's expensive | Often, to keep up with new releases and changing tastes |

In Phase 2 each member's history is written as a **chat**. The "user" message holds the context (time, device), the member's profile, their history and a task like "predict what they'll play next". The "assistant" reply is **what the member actually did**. The model learns by trying to predict that reply.

## Choosing what goes into the text

A long history can be thousands of tokens, and every token costs money each time the model runs. So they edit it:

- **Keep** strong signals: long viewing sessions, thumbs-up.
- **Drop** noise: titles played for a few seconds, accidental clicks.
- **Compress** repetition: a binge becomes one line, not 10 episode entries.
- **Add detail** for new titles the model can't know much about yet.
- **Summarise** old history in a sentence; keep recent history detailed.

## How it produces a ranking

The model **doesn't write out titles word by word**, which is how chatbots answer. That would be too slow and could invent titles that don't exist. Instead:

1. The LLM reads the text once and boils it down to a single list of numbers (a **vector**) that sums up "this member, right now".
2. Each Netflix title also has its own learned vector.
3. A small **scoring head** compares the member's vector with every title's vector and gives each title a score. Sort by score and you have the ranking.

This is the same idea as the [Semantic search](../SemanticSearch/) post: compare a query vector with document vectors. Here the "query" is a person's whole viewing history. Because the model can only score titles that are in the catalogue, it **can't recommend titles that don't exist**. And because it reads the text in **one pass** (the paper calls this "prefill-only"), it's cheap enough for Netflix's traffic.

The model also keeps some ordinary language training. This stops it forgetting how to read, so you could later steer it with instructions written in plain language.

## Teaching it what "good" means

If the model only copies what people clicked, it learns bad habits: too much binge-watching, too much video and not enough games, chasing clicks over long-term happiness. So separate **reward models** score each training example, and examples that point to long-term satisfaction count for more during training. They tried full reinforcement learning (GRPO) and it helped a bit more, but it was too costly to train, so they chose this simpler weighting.

## Results

**MRR**, used below, measures roughly *how near the top of the list the title the member actually watched ends up*. Higher is better. See the [note on MRR](#note-what-mrr-means) at the end.

| What they tested | Result |
|---|---|
| Training data | About **40× fewer** labelled examples than the current ranker, and fewer input signals |
| Offline quality | **+1.6%** MRR over the current ranker |
| Live A/B test (10% of Netflix traffic, 4 weeks) | Short-term homepage engagement **+0.115%**, long-term core metric **+0.006%**, both statistically significant |
| Phase 1's contribution | +10–20% MRR compared with an off-the-shelf LLM |
| Phase 2's contribution | +35–50% MRR, growing to about +80% after two weeks as the Phase 1 model gets out of date |
| Shorter text | About 5,000 → 1,700 tokens with almost no loss in quality, so **serving cost dropped to about a third** |

Live-test gains of 0.1% sound tiny, but across hundreds of millions of members they are real. These are also small gains over a system Netflix has tuned for years.

**Scaling** behaves like it does for LLMs in general: more Phase 2 data gave steady gains (20× the data → about 16% better), and bigger models (from about 1B to about 10B parameters) did better under the same training budget.

## Why it matters

The paper argues recommenders are becoming LLM systems:

- **Write good context** instead of designing features.
- **Share one big foundation model** across many products instead of building a custom model for each.
- **Run on LLM serving tools** (GPUs, vLLM, caching) instead of classic recommender systems.

## Limits

- It's tested only on surfaces where recommendations are computed ahead of time ("batch-compute"), not live requests.
- Most numbers are relative or normalised, so you can't see how good the ranker is in absolute terms.
- Netflix calls this a **first step**, not a replacement for its current system.
- LLMs are expensive to run, so it works only because of the cost-cutting tricks above: shorter text, one-pass scoring, smaller models.

## Note: what MRR means

**MRR** stands for **Mean Reciprocal Rank**. It measures how high up the list the right answer appears, on average.

For each test case:

1. Find the **position** of the right answer in the list. In GenRec, the right answer is the title the member actually went on to watch.
2. Take **1 ÷ that position**. This is the "reciprocal rank".

Then **average** these over all test cases. That average is the "mean".

| Position of the right title | Score (1 ÷ position) |
|---|---|
| 1st | 1.00 |
| 2nd | 0.50 |
| 3rd | 0.33 |
| 10th | 0.10 |
| 100th | 0.01 |

**Example.** The model ranks titles for three members:

- **Member A** watched the title it put **1st** → 1/1 = 1.00
- **Member B** watched the title it put **2nd** → 1/2 = 0.50
- **Member C** watched the title it put **5th** → 1/5 = 0.20

**MRR** = (1.00 + 0.50 + 0.20) ÷ 3 = **0.57**. The best possible score is 1.0, which means the right title was always first.

**Why it suits recommendations.** Most of the score comes from the top few positions. Moving the right title from 2nd to 1st adds 0.5. Moving it from 50th to 49th adds almost nothing. That matches how people use Netflix: they look at the first few rows and rarely scroll far.

**What it doesn't tell you.**

- It counts only the **first** right answer. If a member would have enjoyed five titles, MRR ignores the other four. The **nDCG@10** used in the [BM25](../BM25/) post rewards every good result in the top 10.
- Netflix reports MRR only as a **relative change**, for example "+1.6% over the current ranker". It never gives the actual value, so you can't tell how often the right title really ends up first.
