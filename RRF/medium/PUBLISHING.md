# Publishing on Medium

## The quick way

Ask Claude to put the story on Medium. The `medium-post` skill (in `~/.claude/skills/medium-post`) converts
`rrf-medium.md`, uploads the images with captions and alt text, and fills a **draft** in your signed-in Chrome.
It never publishes; you review the draft and click Publish yourself.

## By hand

Medium's editor doesn't import Markdown files directly, and it has no tables, interactive elements or SVG charts.
This version is built around those limits:

- Every table and chart is a PNG in `images/`. The k slider became a still image.
- Only two heading levels are used (`##` = big heading, `###` = small heading), which is all Medium has.
- Everything else is plain paragraphs, bold, italic, links, lists, quotes and code blocks, which Medium supports.

## Steps

1. **Start a story** at medium.com → *Write*.
2. **Title and subtitle.** Type the `#` line as the title, then the `##` line under it as the subtitle.
   Medium styles them when you select the text and choose *Title* / *Subtitle* (or it detects them as the first two lines).
3. **Paste the text.** Open `rrf-medium.md` in a Markdown preview (VS Code: `Cmd+Shift+V`, or paste it into a GitHub Gist
   and view it), copy the rendered text section by section, and paste into Medium. Headings, bold, italic, links
   and lists survive a rich-text paste. Don't paste the raw Markdown source: Medium will show the `#` and `**` characters.
4. **Headings.** If a heading pastes as normal text, select it and click the big **T** (for `##`) or small **T** (for `###`).
5. **Images.** Wherever the Markdown has `![...](images/…png)`, delete the pasted placeholder and drag in that PNG.
   - Click the image → *Alt text*, and paste the text from inside the `![...]` brackets. It's written to be read aloud.
   - Type the italic line under each image into Medium's caption field (click just below the image).
6. **Quotes.** The `>` lines: select the text and click the quote button once (block quote) or twice (pull quote).
7. **Code blocks.** Put the cursor on an empty line, type three backticks ```` ``` ```` to start a code block, and paste
   the code. Do this for the Python example and the OpenSearch pipeline.
8. **Dividers.** The `---` lines: type `***` on an empty line, or use the *+* menu → separator.
9. **Tags.** Medium allows up to 5. Suggested: `Search`, `Information Retrieval`, `Machine Learning`, `RAG`, `OpenSearch`.
10. **Preview** before publishing, especially the images on a phone-width window.

## Before you publish

- If you push the `lab/` folder to GitHub, add a link to it in the "How we know these numbers are right" section.
  Readers trust numbers they can re-run.
- Medium shows images up to about 700 px wide. The PNGs are rendered at 2× (1400 px) so they stay sharp.

## Images

| File | Where it goes |
|---|---|
| `01-hero-merge.png` | Top of the story, under the subtitle |
| `02-reciprocals.png` | Step 1 |
| `03-formula.png` | Step 2 |
| `04-search-results.png` | Step 3a |
| `05-points-per-position.png` | Step 3b |
| `06-rrf-totals.png` | Step 3c |
| `07-k-points.png` | Step 4, replaces the interactive slider |
| `08-k-paper-chart.png` | "Where does 60 come from?" |
| `09-benchmark-table.png` | "Does RRF actually help?" |
| `10-head-to-head.png` | "Averages hide a mix" |
| `11-k-sweep.png` | "k = 60 isn't the best choice everywhere" |

To rebuild the images (for example after changing a number), run `node medium/build-images.mjs` from the project folder.
It needs Google Chrome and Node 22 or newer.
