# GenRec on a laptop

A small, working copy of Netflix's LLM-based recommender, built step by step on a MacBook with public movie ratings, to see which of the paper's ideas hold up at laptop scale.

**Paper:** [GenRec: An LLM-Backed Recommendation Ranker at Netflix](../papers/GenRec-2608.10257v2.pdf) (Li et al., 2026, [arXiv 2608.10257](https://arxiv.org/abs/2608.10257)) · **Plain-language summary of the paper:** [papers/GenRec-explained.md](../papers/GenRec-explained.md) · **Data:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) · **Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16) with [MLX](https://github.com/ml-explore/mlx)

> **Status:** steps 1 to 4 are done. In step 5, the first three training attempts didn't learn to rank; the fourth (LP-FT) does. Trained on 40,000 examples, it beats item-kNN, the best classic method, on the untouched test set: MRR 22% higher, with a 95% interval entirely above zero (step 6). Step 7 is still to come. Every number on this page was printed by a script in `lab/` and saved in `out/`. Numbers from the paper say so.

## The short version

A recommender has to put a few titles at the top of a list out of thousands. Netflix's current ranker does this with thousands of hand-made **features**: numbers someone designed, such as "how many thrillers this person watched last month". GenRec replaces them with **text**. It writes a member's history out in words, gives it to a language model, and lets the model work out what matters.

The model doesn't write its answer word by word the way a chatbot does. It reads the history once and turns it into one vector, a list of numbers that sums up "this person, right now". It then compares that vector with a learned vector for every title in the catalogue. Sorting by those scores gives the ranking. One pass, every title scored, and nothing outside the catalogue can come out.

We don't have Netflix's data or its in-house model, so this lab rebuilds the same pieces with what anyone can download:

| In the paper | In this lab |
|---|---|
| Netflix members' viewing history | MovieLens 1M: 6,040 people, 3,706 movies, 1,000,209 ratings from 2000 to 2003 |
| A long play or a thumbs-up | A rating of 4 or 5 stars |
| Netflix's in-house foundation LLM (about 1B to 10B parameters, paper) | Qwen2.5-0.5B-Instruct, an open model with about half a billion parameters |
| Phase 1: teach the LLM about Netflix | Skipped for now (see step 7) |
| Phase 2: post-train for ranking | Steps 4 and 5 |
| An A/B test against the production ranker | An offline test against classic recommenders on 6,035 people |

## Step 1: the data

[`lab/data.py`](lab/data.py) downloads MovieLens 1M (6 MB) and sorts each person's ratings by time.

The question every method has to answer is: **given everything this person rated so far, which movie will they rate 4 or 5 stars next?** 57.5% of all ratings are 4 or 5 stars, so "liked" is common, which is why the next liked movie is the target, not just the next movie.

The split is the standard one for this kind of test, called leave-one-out:

- **Test:** each person's last 4–5★ rating. 6,035 people have at least three, so there are 6,035 test examples.
- **Validation:** the 4–5★ rating before that, used to check training while it runs. Also 6,035.
- **Training:** every earlier 4–5★ rating. That gives 559,906 examples.

The input for any example is everything the person rated before it, low ratings included. A typical person has 93 ratings before their test movie; 10% have fewer than 26 and 10% have more than 396.

**A quirk worth knowing:** 53.2% of ratings were made in the same second as the person's previous rating. People rated movies in bursts on the MovieLens website, so the order within a burst is arbitrary, and the "next" movie is partly chance. This makes the task harder for every method equally.

## Step 2: what to beat

Before training anything, we need to know what a simple method scores. [`lab/baselines.py`](lab/baselines.py) runs four recommenders that need no training:

- **random:** shuffles the catalogue. Any real method has to beat it.
- **popular:** the most-liked movies, the same list for everyone.
- **item-kNN:** "people who liked X also liked Y". Two movies are similar when the same people liked both. A movie's score is its total similarity to the movies this person liked.
- **item-kNN, last N:** the same, using only the person's N most recent likes.

Each method scores all 3,706 movies; [`lab/metrics.py`](lab/metrics.py) finds where the right movie landed and turns that position into three numbers:

- **MRR** (mean reciprocal rank): 1 ÷ the position, averaged over people. 1st place scores 1, 2nd scores 0.5, 10th scores 0.1. This is the paper's main offline measure.
- **nDCG@10:** the same idea, gentler and limited to the top 10. A hit at position p scores 1 ÷ log₂(p + 1), and anything below 10th scores 0.
- **Recall@10:** the share of people whose movie made the top 10 at all.

Movies the person already rated are removed before ranking, because MovieLens allows one rating per person per movie, so they can never be the answer.

Results on the test set (6,035 people, all 3,706 movies ranked):

| Method | MRR | nDCG@10 | Recall@10 | Median position |
|---|---|---|---|---|
| random | 0.0023 | 0.0011 | 0.2% | 1,807 |
| popular | 0.0231 | 0.0222 | 4.6% | 340 |
| item-kNN, all likes | 0.0369 | 0.0375 | 7.2% | 241 |
| item-kNN, last 20 likes | 0.0440 | 0.0462 | 9.0% | 195 |
| **item-kNN, last 5 likes** | **0.0671** | **0.0744** | **13.7%** | **141** |
| item-kNN, last 1 like | 0.0674 | 0.0742 | 13.3% | 162 |

Two things stand out:

- **Recent history matters most.** Using only the last 5 likes almost doubles MRR compared with using all of them. People rated in themed bursts, a run of children's films, then a run of thrillers, so the last few movies say the most about the next one. The paper makes the same bet when it keeps recent history in detail and shortens older history (step 3).
- **The numbers look small because the task is hard.** Every method ranks the full catalogue, as the paper does. Many papers on MovieLens rank the right movie against only 100 random others, which gives much higher scores that can't be compared with these.

**The target for the LLM is item-kNN on the last 5 likes: MRR 0.0671, nDCG@10 0.0744.** It plays the part of Netflix's production ranker here, although it is far simpler.

## Step 3: writing a history as text

The paper calls this **verbalization**: turning a member's history into text the LLM can read. Each example becomes a short chat. The user message holds the context, the person's profile, their history and a question. The assistant message is the answer: the movie they actually liked next. [`lab/verbalize.py`](lab/verbalize.py) builds it. This is a real test example, shortened in the middle:

```
[system]    You recommend movies.
[user]
Member: female, under 18, K-12 student.
Today: 2001-01-06.
Older history (2 ratings, Dec 2000): liked 2, mostly Drama, Comedy, Sci-Fi; disliked 0.
Ratings, oldest first (1-5 stars):
2000-12-31: Titanic (1997) 4★ [Drama, Romance]; Cinderella (1950) 5★ [Animation, Children's, Musical]; Meet Joe Black (1998) 3★; …
2001-01-06: Ponette (1996) 4★ [Drama]; Schindler's List (1993) 5★ [Drama, War]; … A Bug's Life (1998) 5★ [Animation, Children's, Comedy]; Antz (1998) 4★ [Animation, Children's]; … Mulan (1998) 4★ [Animation, Children's]
Which movie will this member watch next and rate 4 or 5 stars?
[assistant] Pocahontas (1995)
```

Writing out everything isn't an option: one person would need 43,190 tokens. The paper calls the work of deciding what goes in **context engineering**, and lists four moves. This lab uses all four:

- **Keep in full:** the 50 most recent ratings are written out one by one.
- **Summarise:** everything older becomes one line. For the person with the longest history, 2,261 ratings become *"liked 1192, mostly Drama, Comedy, Thriller; disliked 334, mostly Comedy, Drama"*.
- **Add detail selectively:** only liked movies get their genres.
- **Compress:** ratings from the same day share one line, so the date is written once.

Titles are also rewritten to read naturally ("A Bug's Life", not MovieLens's "Bug's Life, A").

How long the prompt is, in Qwen tokens, for each setting (6,035 test people):

| Setting | Median | 90% are under | Longest | Mean |
|---|---|---|---|---|
| everything | 1,889 | 7,724 | 43,190 | 3,227 |
| everything, no genres | 1,391 | 5,615 | 32,884 | 2,376 |
| last 100 + summary | 1,606 | 1,939 | 2,739 | 1,365 |
| **last 50 + summary** (used for training) | **946** | **1,063** | **1,478** | **881** |
| last 50, no 3★ ratings | 749 | 965 | 1,367 | 727 |
| last 20 + summary | 466 | 513 | 694 | 468 |

The chosen setting cuts the average prompt from 3,227 tokens to 881, about 27% of the original. The paper reports a similar cut, from about 5,000 tokens to about 1,700, with almost no loss in ranking quality (paper, section 5.4). Whether quality survives the cut here is one of step 7's experiments.

## Step 4: the model

[`lab/model.py`](lab/model.py) follows the paper's "scoring model" (section 4.5) in three parts:

1. **Read:** the LLM reads the prompt once.
2. **Pool:** its hidden state at the prompt's last token is kept as one vector, *h*, with 896 numbers. Because the model reads left to right, this last position has seen the whole history.
3. **Score:** a small **scoring head** compares *h* with a learned vector for each of the 3,706 movies. The score is the cosine similarity (how closely the two vectors point the same way) times a learned scale, plus a learned bias per movie. Both sides are centred first, for a reason explained in step 5.

So one forward pass gives a score for every movie. The paper calls this **prefill-only** inference: the model only does the "read the prompt" part of a normal LLM call, and never generates text. For person 1, 1,047 prompt tokens go in and 3,706 scores come out.

**Where the movie vectors start.** Each movie's vector starts as the *h* the LLM produces for a made-up person whose only rating is that movie, 5 stars. Before any training, a real person then scores close to a movie when the LLM "sees" them like a fan of that movie. That gives a fair off-the-shelf score to compare training against.

**Before training, the LLM is no better than random** (first 300 test people, scored on the cosine alone, with every bias at zero):

| | MRR | nDCG@10 | Recall@10 | Median position |
|---|---|---|---|---|
| LLM, no training | 0.0032 | 0.0000 | 0.0% | 1,517 |

None of the 300 people found their movie in the top 10. Person 1's untrained top 10 shows why: *Titanic (1953), Pandora and the Flying Dutchman, Two or Three Things I Know About Her, Pollyanna, Belle de jour, Pit and the Pendulum, Telling Lies in America, Penny Serenade, Puppet Master 5, Pather Panchali*. Six of the ten start with P or T, and the right answer, *Pocahontas*, is 15th. Before training, the hidden state at that position is mostly about which word the model would write next, not about taste. This is what the paper means when it says off-the-shelf LLMs are "not yet suitable" as recommenders.

**Speed and memory** on an M4 Pro with 48 GB:

- **Ranking:** 108 ms per person (8,064 tokens per second, 2.0 GB of memory). In practice the whole test set took about 15 minutes per model in step 6.
- **Training:** LoRA adds small trainable matrices inside each of the LLM's 24 layers and leaves the original weights frozen. With the scoring head, that is 12.1M trainable numbers out of about 506M. `model.py` times 8 batches of 8 and gets about 2.4 seconds and 24.2 GB per batch. That test uses person 1's examples, whose histories are short. Real training batches, with prompts of about 950 tokens, take about 4 seconds (`out/train_part1.log`), so about half a second per example.

At that speed, all 559,906 training examples would take over three days, so training uses a sample.

## Step 5: training

[`lab/train.py`](lab/train.py) does the paper's Phase-2 post-training. The paper trains two objectives at once (section 4.4), and so does the current version:

- **Ranking:** a cross-entropy loss over the whole catalogue, from the scoring head. In plain terms, the model is pushed to give the movie the person actually liked next a higher score than the other 3,705.
- **Language:** the LLM's ordinary "predict the next word" loss over the chat, history lines and answer included. It keeps the model's language skills, and every history line is another example of which movie comes next.

The total is 0.5 × ranking + 0.5 × language. One forward pass over the full chat gives both: the scoring head reads *h* at the end of the prompt, and since the model only looks backwards, the answer that follows can't leak into *h*.

- **What is trained:** LoRA matrices inside the LLM (learning rate 1e-4) and the scoring head (1e-3, and 1e-2 for its single cosine weight). The rest of the LLM stays frozen.
- **Training data:** a fixed, shuffled pool of 40,000 examples: each person's 3 most recent training examples, the closest match to what the test asks, plus random older ones.
- **Two parts:** part 1 trains on the first 20,000 examples; part 2 continues from part 1's weights on the next 20,000. Scoring after each part answers the paper's data question: does more Phase-2 data keep helping?
- **Checking progress:** every 250 steps, the model ranks the catalogue for 500 validation people, and the best weights are kept.

The baselines on the same 500 validation people (from `baselines.py`):

| | MRR | nDCG@10 | Recall@10 | Median position |
|---|---|---|---|---|
| popular | 0.0323 | 0.0316 | 5.8% | 311 |
| item-kNN, last 5 likes | 0.0759 | 0.0841 | 15.4% | 106 |

Getting the model to learn has taken several attempts. They are recorded here because each one shows something about putting a ranking head on an LLM.

### Attempt 1: ranking only, untouched head

The first run used the head exactly as step 4 describes, with every bias at zero (`out/train_part1_slow-head.json`):

| | MRR | nDCG@10 | Recall@10 | Median position |
|---|---|---|---|---|
| step 0 | 0.0047 | 0.0027 | 0.4% | 1,788 |
| step 250 | 0.0110 | 0.0094 | 2.0% | 638 |

It was improving, but slowly and from far below popularity, and the training loss had barely moved from 8.2, which is what an even guess across 3,706 movies gives. There were two causes.

**The model had to learn popularity from scratch.** With every bias at zero, "popular movies come first" could only be learned through thousands of small updates. Now each movie's bias starts at the log of how many people liked it, counted from training ratings only. The model starts out knowing what plain popularity knows.

**The LLM's vectors all point the same way.** Straight out of the LLM, the cosine between a person's *h* and any movie's vector averages 0.943 and varies by only 0.007 from one movie to the next. Times the scale of 20, that moves scores by about 0.13, against 1.8 for the popularity bias, so the "taste" part of the score could barely tell movies apart. Language models are known to do this: their hidden states share one large common direction. The fix is to **subtract the average direction** before comparing: the average person vector (from 200 training prompts, then learned) is taken off *h*, and the average movie vector off the movie vectors. After centring, the spread across movies is 0.091, 13 times wider.

Centred but untrained, the cosine is still mostly noise, so adding it to popularity makes the ranking worse, and more weight makes it worse still (200 validation people, `out/centring.json`):

| Score | MRR | Median position |
|---|---|---|
| popularity alone | 0.0336 | 324 |
| popularity + 2 × cosine | 0.0253 | 316 |
| popularity + 5 × cosine | 0.0155 | 344 |
| popularity + 20 × cosine | 0.0070 | 614 |

So the cosine weight now starts at 2 instead of 20 and is learned. A projection layer between *h* and the cosine was also removed: with a fast learning rate, its 802,816 numbers added noise faster than signal.

### Attempt 2: ranking only, centred head

With both fixes, part 1 started at popularity's level, as intended. Then nothing happened (`out/train_part1_ranking-only.log`):

| | MRR | nDCG@10 | Recall@10 | Median position | Cosine weight |
|---|---|---|---|---|---|
| step 0 | 0.0279 | 0.0282 | 5.6% | 302 | 2.0 |
| step 250 | 0.0288 | 0.0281 | 5.6% | 328 | 0.15 |
| step 500 | 0.0276 | 0.0264 | 5.4% | 328 | 0.08 |

The training loss stayed between 7.22 and 7.48 from step 50 to step 650, and the model kept turning the cosine weight down toward zero: it was switching off the "taste" part of its score and ranking by popularity alone. We stopped it after 43 minutes. (This attempt also ran into a memory problem, described in Notes.)

### Attempt 3: following the paper more closely

Comparing the lab with the paper at this point showed one real gap. The paper trains the ranking objective **together with a language-modelling objective** over the verbalized inputs and outputs, and the first two attempts had left that out. With so few examples per movie (about 5 each in 20,000), the language loss should matter: it trains *h* to predict the next title through the model's own word knowledge, which is shared between movies.

Adding it ran into memory first. Scoring all 152,000 vocabulary words at every position of 8 prompts of about 1,000 tokens peaked at 52 GB, more than the Mac has (`out/train_check300_all-positions.log`). So each step now scores 128 randomly chosen positions per prompt. On average that gives the same loss, and over many steps every part of the text is covered. The peak fell to 34.5 GB.

A 300-step test (`out/train_check300.log`):

| | MRR | Median position | Cosine weight | Ranking loss | Language loss |
|---|---|---|---|---|---|
| step 0 | 0.0253 | 316 | 2.0 | – | – |
| step 50 | – | – | – | 7.37 | 1.22 |
| step 100 | 0.0255 | 358 | 0.29 | 7.34 | 1.12 |
| step 200 | 0.0275 | 370 | 0.17 | 7.47 | 0.79 |

The language loss fell steadily; the model was learning to predict the history text, titles included. The ranking loss didn't move, and the cosine weight sank again.

### Why the two objectives didn't connect

[`lab/title_head.py`](lab/title_head.py) tests whether the language training changes *h* in a useful way at all. It scores movies with the LLM's own arithmetic for predicting words: the dot product of *h* with the word embeddings of each title. No ranking training is involved (200 validation people, `out/title_head.json`):

| Score | Original model, MRR | After 200 steps of attempt 3, MRR |
|---|---|---|
| popularity alone | 0.0336 | – |
| title's first word | 0.0009 | 0.0010 |
| first word + popularity | 0.0209 | 0.0184 |
| title, all words | 0.0037 | 0.0087 |
| all words + popularity | 0.0161 | 0.0301 |

The language training does move *h* toward the titles people liked next: the "all words" score more than doubles after only 200 steps. (The first word alone is useless, because the 3,706 titles start with only 996 different first tokens, such as "The".) But this is still below popularity.

The more telling result is what didn't happen in attempts 2 and 3. With 12 million trainable numbers and a few thousand examples, a model that is learning should at least start to **memorise** its training examples, so the training ranking loss should fall. It didn't move. That points to the training procedure rather than the amount of data, and it matches a known problem: fine-tuning a pretrained model while its new head is still untrained. The untrained head sends noisy gradients into the model, which distorts the features it should be building on (Kumar et al., 2022). Here, the model's way out was to switch the head off.

### Attempt 4: train the head first, then everything

The usual remedy is **LP-FT**: first train only the head on the frozen model's features (a *linear probe*), then fine-tune everything together starting from that head. The paper still applies, since everything is trained jointly in the end; LP-FT only makes sure the head works before it starts steering the LLM.

[`lab/probe.py`](lab/probe.py) does the first half. The untrained LLM reads 8,000 training prompts and the 500 validation prompts once, and their *h* vectors are saved. Training a head on saved vectors takes seconds instead of hours, so it tries 24 variants:

- **Features:** *h* centred, or centred and standardised (each dimension divided by its spread).
- **Head:** cosine with a learned weight, or a plain dot product.
- **Starting movie vectors:** the "fan of this movie" vectors from step 4, the title's word embeddings, or small random numbers.
- **Learning rate:** 1e-3 or 1e-2.

Each variant trains for 20 passes over the 8,000 examples and is scored on the 500 validation people after every pass; its best pass counts (`out/probe.json`):

| Head | MRR | nDCG@10 | Median position |
|---|---|---|---|
| popularity alone | 0.0323 | 0.0316 | 311 |
| **centred, cosine, title-word start, lr 1e-3** | **0.0346** | 0.0365 | 274 |
| centred, cosine, random start, lr 1e-3 | 0.0345 | 0.0369 | 268 |
| standardised, cosine, random start, lr 1e-3 | 0.0343 | 0.0369 | 273 |
| best dot-product head (standardised, "fan" start, lr 1e-3) | 0.0292 | 0.0281 | 428 |
| worst head (standardised, dot product, title-word start, lr 1e-2) | 0.0147 | 0.0148 | 1,174 |

Three things stand out:

- **On a frozen LLM, the head learns.** Its training loss falls steadily, where in the joint runs it never moved. That supports the diagnosis: the stalls came from training an untrained head and the LLM at the same time.
- **The frozen LLM's *h* holds little taste.** The best head beats popularity by only 7% in MRR. Almost all of the improvement has to come from fine-tuning.
- **Heads overfit quickly.** Every head peaks within a few passes and then memorises: at learning rate 1e-2 the training loss falls to about 0.02 while validation gets worse. Cosine heads beat dot-product heads clearly; the title-word and random starts end up the same, and the "fan" start is worst.

A second round (`python lab/probe.py --stage size`) tried smaller heads, with 64 or 128 numbers per movie behind a learned projection, and weight decay of 0, 0.01 or 0.1. None did better: 64 and 128 both reached an MRR of 0.0334, and weight decay changed nothing. That is expected for a cosine head, which only uses each vector's direction, so shrinking its length has no effect. Head size isn't the limit; the frozen *h* is.

**Fine-tuning from the probe's head.** Joint training (LoRA, the ranking loss and the language loss) then starts from the best probe head. The head learns slowly (1e-4, its cosine weight 1e-3) so it doesn't drift while LoRA reshapes *h*. Batches are built by token budget, at most 7,600 padded tokens each, which kept memory at a 25.4 GB peak. Part 1, 20,000 examples (`out/train_part1.json`):

| Step | Examples seen | MRR | nDCG@10 | Recall@10 | Median position | Cosine weight | Ranking loss | Language loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0346 | 0.0365 | 7.4% | 274 | 6.12 | – | – |
| 250 | 1,983 | 0.0369 | 0.0401 | 8.2% | 250 | 9.04 | 6.89 | 1.01 |
| 500 | 3,957 | 0.0464 | 0.0492 | 9.0% | 219 | 11.94 | 6.77 | 0.76 |
| 750 | 5,928 | 0.0454 | 0.0465 | 8.6% | 202 | 15.05 | 6.47 | 0.66 |
| 1,000 | 7,952 | 0.0555 | 0.0572 | 9.6% | 257 | 17.44 | 6.23 | 0.62 |
| 1,250 | 10,000 | 0.0518 | 0.0557 | 10.2% | 214 | 19.29 | 6.02 | 0.58 |
| 1,500 | 12,009 | 0.0600 | 0.0647 | 11.6% | 194 | 20.54 | 5.96 | 0.58 |
| 1,750 | 14,106 | 0.0558 | 0.0619 | 11.6% | 184 | 21.59 | 5.60 | 0.55 |
| 2,000 | 16,103 | 0.0588 | 0.0670 | 12.6% | 188 | 22.04 | 5.71 | 0.54 |
| 2,250 | 18,129 | 0.0603 | 0.0624 | 10.8% | 174 | 22.34 | 5.65 | 0.53 |
| 2,487 | 20,000 | **0.0621** | 0.0666 | 11.6% | 196 | 22.55 | 5.54 | 0.53 |

This time the model learned to rank:

- **MRR rose from 0.0346 to 0.0621,** 79% above where fine-tuning started and 92% above popularity. That closes 68% of the gap between popularity (0.0323) and the best classic method, item-kNN on the last 5 likes (0.0759). It also passes item-kNN on all likes (0.0411) and on the last 20 likes (0.0451) for these 500 people.
- **The cosine weight climbed steadily, from 6.1 to 22.6.** In every earlier attempt it sank toward zero. The model now trusts the taste part of its score more as training goes on, the opposite of switching it off.
- **Both losses fell together**: ranking from 6.89 to 5.54, language from 1.01 to 0.53. The two objectives now pull in the same direction.
- **Single checks are noisy.** With 500 people, MRR moves by a few thousandths from one check to the next (0.0555, then 0.0518, then 0.0600), so the trend across checks matters more than any one of them.

Part 1 took 2 hours 39 minutes.

### More data: a restart that went wrong, and one long run instead

The plan was a part 2: continue from part 1's final weights on the next 20,000 examples of the pool. After 250 steps its MRR had fallen to 0.0296, below popularity, and the cosine weight had dropped from 22.6 to 14.0 (`out/train_part2_rewarmed.json`). The language loss rose too, from 0.53 to 0.84, though the data was the same kind; on similar text it should have kept falling. Two things had reset at the restart:

- **The learning rate jumped back to its peak.** Part 1 ended at a tenth of its peak. Part 2 started a new schedule and warmed back up to the full peak within 100 steps.
- **The optimizer lost its memory.** Adam keeps running averages for every number it trains, which set how big each step is. They weren't saved, so part 2 started with empty ones, and Adam's first steps were full-size for every number, including all 3.3 million in the movie vectors.

Together they knocked the model out of what it had learned. `train.py` now saves the optimizer's state with a run's final weights and restores it with `--init`, so a later part keeps Adam's averages and continues at the learning rate where the earlier one ended.

For the data question, a gentle continuation at a low learning rate would mostly polish part 1, and any gain would be hard to attribute to the extra data. So instead, one run trains on all 40,000 examples under a single schedule, starting from the probe head like part 1. That gives two fully trained models to compare, 20,000 examples (part 1) against 40,000, like the paper's data-scaling comparison.

The 40,000-example run took 5 hours 22 minutes (`out/train_full40k.json`). Every other check:

| Step | Examples seen | MRR | nDCG@10 | Recall@10 | Median position | Cosine weight | Ranking loss | Language loss |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0346 | 0.0365 | 7.4% | 274 | 6.12 | – | – |
| 500 | 4,052 | 0.0409 | 0.0450 | 8.8% | 228 | 5.78 | 7.29 | 0.72 |
| 1,000 | 8,048 | 0.0476 | 0.0525 | 10.6% | 178 | 7.91 | 6.95 | 0.60 |
| 1,500 | 11,998 | 0.0613 | 0.0699 | 13.4% | 156 | 9.96 | 6.84 | 0.56 |
| 2,000 | 16,086 | 0.0744 | 0.0848 | 15.4% | 140 | 11.86 | 6.67 | 0.54 |
| 2,500 | 20,147 | 0.0731 | 0.0819 | 15.0% | 132 | 13.14 | 6.62 | 0.52 |
| 3,000 | 24,174 | 0.0836 | 0.0921 | 15.8% | 120 | 14.41 | 6.47 | 0.51 |
| 3,500 | 28,230 | 0.0897 | 0.1019 | 18.0% | 114 | 15.25 | 6.40 | 0.51 |
| 4,000 | 32,263 | 0.0846 | 0.0986 | 18.2% | 96 | 16.07 | 6.22 | 0.48 |
| 4,500 | 36,288 | **0.0952** | **0.1097** | 19.6% | 96 | 16.51 | 6.18 | 0.48 |
| 4,976 | 40,000 | 0.0938 | 0.1087 | 19.6% | 95 | 16.85 | 6.29 | 0.48 |

It started more slowly than part 1 (the stall at steps 250 to 500), then kept improving for the rest of the run. The best check, at step 4,500, is kept as `best`.

**20,000 against 40,000 examples**, each a full run from the same probe head, on the same 500 validation people:

| | MRR | nDCG@10 | Recall@10 | Median position |
|---|---|---|---|---|
| popularity | 0.0323 | 0.0316 | 5.8% | 311 |
| item-kNN, last 5 likes | 0.0759 | 0.0841 | 15.4% | 106 |
| GenRec, 20,000 examples (part 1, final) | 0.0621 | 0.0666 | 11.6% | 196 |
| GenRec, 40,000 examples (best check) | **0.0952** | **0.1097** | **19.6%** | **96** |
| GenRec, 40,000 examples (final) | 0.0938 | 0.1087 | 19.6% | 95 |

- **Doubling the training data raised MRR by about half,** from 0.0621 to 0.0938 at the end of each run. The paper reports the same direction: quality keeps rising with more Phase-2 data (paper, figure 4). We have two points, not a curve, so this says nothing yet about where it levels off.
- **With 40,000 examples, the LLM ranker beats item-kNN on these 500 people on every measure:** MRR 24% higher at the end of the run (0.0938 against 0.0759), nDCG@10 29% higher, and a better median position (95 against 106).
- **These 500 people are also how the best checkpoint was chosen,** so their scores flatter it a little. The real comparison is the test set in step 6, with an interval for how sure we can be.

## Step 6: the test set

Every choice so far, from head designs to the best checkpoint, was made on 500 validation people, which flatters the results a little. [`lab/evaluate.py`](lab/evaluate.py) scores the trained models on the 6,035 test people, each person's last 4–5★ rating, which nothing has touched. To tell a real gap from luck, it compares each model with item-kNN on the same people and resamples them 2,000 times (a paired bootstrap): if the middle 95% of the resampled differences is entirely above or below zero, the gap is real. Results are in `out/evaluate.json`.

| Method | MRR | nDCG@10 | Recall@10 | Median position | MRR vs item-kNN, 95% interval | nDCG@10 vs item-kNN, 95% interval |
|---|---|---|---|---|---|---|
| popularity | 0.0231 | 0.0222 | 4.6% | 340 | −0.0483 to −0.0394 | −0.0575 to −0.0468 |
| item-kNN, last 5 likes | 0.0671 | 0.0744 | 13.7% | 141 | – | – |
| GenRec, 20,000 examples | 0.0513 | 0.0566 | 10.7% | 221 | −0.0210 to −0.0105 | −0.0238 to −0.0115 |
| GenRec, 40,000 examples, best on validation | 0.0793 | 0.0901 | 16.6% | 118 | +0.0067 to +0.0176 | +0.0093 to +0.0218 |
| **GenRec, 40,000 examples, final** | **0.0820** | **0.0926** | **16.9%** | **115** | **+0.0094 to +0.0206** | **+0.0120 to +0.0246** |

- **Trained on 40,000 examples, GenRec beats item-kNN on every measure:** MRR 22% higher, nDCG@10 24% higher, the right movie in the top 10 for 16.9% of people instead of 13.7%, and a better median position (115 against 141). Both intervals lie entirely above zero, so this is not luck.
- **More training data decided it.** Trained on 20,000 examples, the same model loses to item-kNN, also clearly. Doubling the data raised MRR by 60%, from 0.0513 to 0.0820.
- **The final weights did slightly better than the checkpoint picked on validation** (0.0820 against 0.0793), a reminder that choosing on 500 people is partly choosing noise.
- Test scores are lower than validation scores for every method, item-kNN included (0.0671 on test, 0.0759 on the 500 validation people).

**By history length.** The prompt writes out a person's last 50 ratings and squeezes everything older into one summary line. MRR, split by how many ratings came before the test movie:

| | Up to 50 past ratings (1,865 people) | 51 to 150 (2,121) | Over 150 (2,049) |
|---|---|---|---|
| popularity | 0.0265 | 0.0218 | 0.0212 |
| item-kNN, last 5 likes | 0.0763 | 0.0674 | 0.0583 |
| GenRec, 20,000 examples | 0.0702 | 0.0480 | 0.0376 |
| **GenRec, 40,000 examples, final** | **0.1117** | **0.0768** | **0.0604** |

GenRec's lead is largest when the whole history fits in the prompt: 46% above item-kNN for people with up to 50 ratings, 14% for 51 to 150, and only 4% over 150. Every method finds long histories harder, but GenRec loses more of its lead. Either the one-line summary throws away information the model could use, or the last 50 ratings of a long history say less about the next movie. Step 7's first experiment, changing how many recent ratings the prompt writes out, tests that.

## Still to come

- **Step 7, the paper's experiments at small scale:** how much history to include (20, 50 or 100 recent ratings, like the paper's figure 5), reward weighting (5★ examples count double, a stand-in for the paper's reward models), a small Phase 1 (first teaching the LLM about the movies themselves), and possibly a larger model.

## Requirements

- A Mac with Apple Silicon. MLX does not run on Intel Macs, Linux or Windows.
- Python 3.10 or newer (the lab was run with 3.12).
- About 2 GB of disk for the model and data, plus about 50 MB per saved training run.
- 48 GB of memory for training: with the default batch of 8, memory peaks at about 35 GB. Ranking alone needs about 2 GB.

## Run it

From this folder:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r lab/requirements.txt
source .venv/bin/activate

python lab/data.py         # step 1: download MovieLens 1M (6 MB) and split it, a few seconds
python lab/baselines.py    # step 2: popularity and item-kNN, about 5 seconds
python lab/verbalize.py    # step 3: example prompts and token counts, about 40 seconds
python lab/model.py --limit 300   # step 4: zero-shot score, speed and memory, about 2 minutes
python lab/title_head.py --runs check300   # step 5: title-word head, needs a run's saved weights
python lab/probe.py                # step 5, LP-FT first half: 24 heads, about 15 minutes the first time
python lab/probe.py --stage size   # smaller heads and weight decay, under a minute
python lab/train.py --name part1 --head-from probe --head-lr 1e-4 --scale-lr 1e-3              # about 2.7 hours
python lab/train.py --name full40k --examples 40000 --head-from probe --head-lr 1e-4 --scale-lr 1e-3 --save-at 20000   # about 5.5 hours
python lab/evaluate.py part1:last full40k:best full40k:last   # step 6: test set, about 15 minutes per model
```

Training runs are best started detached (see Notes). Without uv: `python3 -m venv .venv && .venv/bin/pip install -r lab/requirements.txt`. The first run of `model.py` or `train.py` downloads Qwen2.5-0.5B (about 1 GB) and works out the starting movie vectors (cached in `cache/`).

## Layout

| Path | What it does |
|---|---|
| `lab/data.py` | Downloads MovieLens 1M and makes the train, validation and test splits |
| `lab/metrics.py` | MRR, nDCG@10 and Recall@10 for one right answer |
| `lab/baselines.py` | Random, popular and item-kNN recommenders |
| `lab/verbalize.py` | Writes a history as a chat, with the context-engineering settings |
| `lab/model.py` | The LLM, the pooled vector, the scoring head, and the starting movie vectors |
| `lab/train.py` | Phase-2 training with the ranking and language losses, optional reward weighting, two-part runs |
| `lab/centring.py` | Shows why the scoring head centres its vectors |
| `lab/title_head.py` | Scores movies with the LLM's own title-word embeddings, to test whether language training helps ranking |
| `lab/probe.py` | Trains scoring heads on the frozen LLM's saved vectors (the first half of LP-FT) |
| `lab/evaluate.py` | Scores trained models on the test set, with intervals for the difference from item-kNN |
| `out/` | Results as JSON, one file per script, plus training logs |
| `data/`, `cache/` | Downloaded data, movie vectors and trained weights (not in git) |

## Notes

- **MovieLens can't be redistributed.** Its licence forbids sharing the data, so `data/` is not in git and `data.py` downloads it from GroupLens.
- **Leave-one-out uses other people's later ratings.** Training and the item-kNN table include ratings other people made after a test person's last rating. This is the usual setup for MovieLens results, but a deployed system, like Netflix's, would only ever see the past.
- **Capping MLX's memory cache.** Each training batch is padded to its longest prompt, so batch shapes keep changing. MLX keeps freed GPU buffers for reuse, and with ever-changing sizes that cache only grows: an uncapped run reached 36 GB, pushed macOS into swap and slowed to about 7 seconds per step (`out/train_part1_uncapped.log`). `train.py` caps the cache at 2 GB and total use at 30 GB (`--memory-gb`).
- **Long runs should be detached.** A run started from a terminal or tool session stops when that session ends. `nohup python lab/train.py --name part1 > out/train_part1.log 2>&1 &` keeps it going.
- **Ties count half.** When several movies share the right movie's score, the right movie is placed halfway through the tie. Untrained methods hit ties often.
- **"Today" is the date of the answer's rating.** A real system knows the time of the request, so the prompt includes it. It doesn't reveal which movie is the answer.

## References

- Li, Sehgal, Rao, Houthooft, Hu, Zhu, Medapati, Li, Baltrunas, Huang, Rastogi, Aryafar. *GenRec: An LLM-Backed Recommendation Ranker at Netflix.* 2026. [arXiv 2608.10257](https://arxiv.org/abs/2608.10257)
- Harper and Konstan. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems, 2015. [doi:10.1145/2827872](https://doi.org/10.1145/2827872)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. [arXiv 2106.09685](https://arxiv.org/abs/2106.09685)
- Kumar, Raghunathan, Jones, Ma, Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022. [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)
- Qwen Team. *Qwen2.5 Technical Report.* 2024. [arXiv 2412.15115](https://arxiv.org/abs/2412.15115)
