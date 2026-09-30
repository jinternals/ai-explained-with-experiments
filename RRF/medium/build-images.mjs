// Renders the figures for the Medium post to PNG (2x) with headless Chrome.
// Every number is read from ../out/*.json, so the images always match the lab's latest run.
//
//   node medium/build-images.mjs
//
// Needs Google Chrome and Node 22+ (for the built-in WebSocket). Set CHROME to use another binary.

import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const WIDTH = 700; // CSS pixels, about Medium's body width
const SCALE = 2;   // device pixel ratio, so images are 1400 px wide

const path = (relative) => fileURLToPath(new URL(relative, import.meta.url));
const IMAGE_DIR = path("./images/");

const step3 = JSON.parse(await readFile(path("../out/step3_example.json"), "utf8"));
const bench = JSON.parse(await readFile(path("../out/benchmark.json"), "utf8"));

// ------------------------------------------------------------------ content helpers

const SHORT_NAMES = {
  "MED-5054": "Sweetener toxicity",
  "MED-4120": "Sucrose vs sweetener taste",
  "MED-1625": "Sugar substitutes",
  "MED-5053": "Sweeteners and tumours",
  "MED-3380": "Food dyes and ADHD",
  "MED-1624": "Aspartame and the brain",
  "MED-1623": "Aspartame and nerves",
  "MED-3521": "Cherry phenolics",
};

const DATASETS = [
  { id: "nfcorpus", name: "NFCorpus", about: "Nutrition and medical research" },
  { id: "scifact", name: "SciFact", about: "Scientific claims vs research abstracts" },
  { id: "fiqa", name: "FiQA", about: "Finance questions and forum answers" },
];

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const title = (doc) => doc.title.replace(/\.$/, "");
const shortName = (doc) => SHORT_NAMES[doc.id] ?? title(doc);
const ordinal = (n) => n + ({ 1: "st", 2: "nd", 3: "rd" }[n % 100 > 10 && n % 100 < 14 ? 0 : n % 10] ?? "th");
const three = (x) => x.toFixed(3);
const signed = (x, digits) => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(digits);

/** Rank with ties sharing a position: [0.5, 0.5, 0.2] -> [1, 1, 3]. */
const tiedRanks = (scores) => scores.map((s) => 1 + scores.filter((other) => other > s + 1e-12).length);

const relevantTag = (doc) => (doc.relevance > 0 ? ` <em class="rel">relevant</em>` : "");

// ------------------------------------------------------------------ figures

function heroFigure() {
  const chip = (rank, doc, score) =>
    `<div class="chip${doc.relevance > 0 ? " is-rel" : ""}"><span class="rk">${rank}</span><b>${esc(shortName(doc))}</b>${score ? `<i>${score}</i>` : ""}</div>`;
  const column = (heading, kind, chips) => `<div class="col ${kind}"><div class="col-head">${heading}</div>${chips.join("")}</div>`;

  const fused = step3.rrf.slice(0, step3.top_n);
  const ranks = tiedRanks(step3.rrf.map((row) => row.rrf));

  return `
    <div class="question"><span>A real question from a medical research test set</span>${esc(step3.query)}</div>
    <div class="merge">
      ${column("Keyword search", "kw", step3.keyword.map((doc, i) => chip(i + 1, doc)))}
      ${column("Meaning search", "mn", step3.meaning.map((doc, i) => chip(i + 1, doc)))}
      ${column("After RRF", "fused", fused.map((doc, i) => chip(ranks[i], doc, doc.rrf.toFixed(6))))}
    </div>
    <div class="key"><i></i>Marked relevant in the dataset's labels</div>`;
}

function reciprocalsFigure() {
  const rows = [1, 2, 4, 10].map((n) =>
    `<tr><td class="mono">${n}</td><td class="mono">1 ÷ ${n}</td><td class="num">${(1 / n).toFixed(2)}</td></tr>`);
  return `<table>
    <thead><tr><th>Number</th><th>Reciprocal</th><th class="num">As a decimal</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function formulaFigure() {
  const fraction = (kind, denominator) =>
    `<span class="frac ${kind}"><span>1</span><span>${denominator}</span></span>`;
  return `<div class="formula">
    <span>RRF score</span><span class="op">=</span>
    ${fraction("kw", "k + keyword rank")}<span class="op">+</span>${fraction("mn", "k + meaning rank")}
  </div>`;
}

function pointsFigure() {
  const k = step3.k;
  const rows = Array.from({ length: step3.top_n }, (_, i) => {
    const rank = i + 1;
    return `<tr><td>${ordinal(rank)}</td><td class="mono">${k} + ${rank} = ${k + rank}</td><td class="mono">1 ÷ ${k + rank}</td><td class="num">${(1 / (k + rank)).toFixed(6)}</td></tr>`;
  });
  return `<table>
    <thead><tr><th>Position</th><th>Sum</th><th>Fraction</th><th class="num">Points</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function searchResultsFigure() {
  const panel = (heading, kind, scoreLabel, hits, digits) => `
    <div class="panel ${kind}">
      <div class="panel-head"><span>${heading}</span><span>${scoreLabel}</span></div>
      ${hits.map((hit, i) => `
        <div class="hit"><span class="rk">${i + 1}</span><span>${esc(title(hit))}${relevantTag(hit)}</span><span class="sc">${hit.score.toFixed(digits)}</span></div>`).join("")}
    </div>`;
  return `<div class="panels">
    ${panel("Keyword search", "kw", "BM25 score", step3.keyword, 2)}
    ${panel("Meaning search", "mn", "Similarity", step3.meaning, 3)}
  </div>`;
}

function totalsFigure() {
  const ranks = tiedRanks(step3.rrf.map((row) => row.rrf));
  const cell = (rank, points) =>
    rank ? `<td class="num">${ordinal(rank)} → ${points.toFixed(6)}</td>` : `<td class="num muted">not in top ${step3.top_n} → 0</td>`;

  const rows = step3.rrf.map((row, i) => {
    const tied = ranks.filter((r) => r === ranks[i]).length > 1;
    return `<tr class="${i === 0 ? "win" : ""}">
      <td>${esc(title(row))}${relevantTag(row)}</td>
      ${cell(row.keyword_rank, row.keyword_points)}
      ${cell(row.meaning_rank, row.meaning_points)}
      <td class="num strong">${row.rrf.toFixed(6)}</td>
      <td class="num">${ordinal(ranks[i])}${tied ? " (tie)" : ""}</td>
    </tr>`;
  });

  return `<table>
    <thead><tr><th>Paper</th><th class="kw">Keyword</th><th class="mn">Meaning</th><th class="num">Total</th><th class="num">Final</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function kPointsFigure() {
  const block = (k) => {
    const points = [1, 2, 3, 4, 5].map((rank) => 1 / (k + rank));
    const star = 1 / (k + 1);
    const team = 2 / (k + 3);
    const winner = Math.abs(star - team) < 1e-12 ? "tie" : star > team ? "star" : "team";
    const bar = (value, rank) => `
      <div class="bar"><span>${ordinal(rank)}</span><div class="track"><div class="fill" style="width:${(value / points[0]) * 100}%"></div></div><span class="num">${value.toFixed(4)}</span></div>`;
    const duel = (who, how, score, isWinner) =>
      `<div class="duel${isWinner ? " won" : ""}"><b>${who}</b><span>${how}</span><span class="num">${score.toFixed(4)}</span></div>`;
    return `<div class="kblock">
      <div class="khead">k = ${k}</div>
      ${points.map((value, i) => bar(value, i + 1)).join("")}
      <div class="duels">
        ${duel("Star", "1st in one list only", star, winner === "star")}
        ${duel("Team player", "3rd in both lists", team, winner === "team")}
        <p class="verdict">${{ tie: "Exactly tied.", star: "The star wins.", team: "The team player wins." }[winner]}</p>
      </div>
    </div>`;
  };
  return `<div class="kgrid">${block(1)}${block(60)}</div>
    <p class="note">Bars are drawn relative to 1st place.</p>`;
}

function paperChartFigure() {
  // Cormack, Clarke & Büttcher (2009), Table 1: MAP for k = 0, 10, ..., 100 and 500.
  return `<svg viewBox="0 0 640 262" class="chart">
    <g class="grid"><line x1="60" y1="191" x2="610" y2="191"/><line x1="60" y1="153" x2="610" y2="153"/><line x1="60" y1="115" x2="610" y2="115"/><line x1="60" y1="77" x2="610" y2="77"/><line x1="60" y1="39" x2="610" y2="39"/></g>
    <g class="ty"><text x="52" y="195">0.207</text><text x="52" y="157">0.209</text><text x="52" y="119">0.211</text><text x="52" y="81">0.213</text><text x="52" y="43">0.215</text></g>
    <g class="tx"><text x="60" y="232">0</text><text x="152" y="232">20</text><text x="244" y="232">40</text><text x="336" y="232">60</text><text x="428" y="232">80</text><text x="520" y="232">100</text><text x="560" y="232">···</text><text x="600" y="232">500</text></g>
    <text x="335" y="256" class="axis">k</text>
    <text x="16" y="115" class="axis" transform="rotate(-90 16 115)">MAP</text>
    <polyline class="line" points="60,187.2 106,90.3 152,69.4 198,59.9 244,61.8 290,50.4 336,48.5 382,46.6 428,44.7 474,48.5 520,54.2"/>
    <line class="line dashed" x1="520" y1="54.2" x2="600" y2="137.8"/>
    <g class="dots"><circle cx="60" cy="187.2" r="3.5"/><circle cx="106" cy="90.3" r="3.5"/><circle cx="152" cy="69.4" r="3.5"/><circle cx="198" cy="59.9" r="3.5"/><circle cx="244" cy="61.8" r="3.5"/><circle cx="290" cy="50.4" r="3.5"/><circle cx="382" cy="46.6" r="3.5"/><circle cx="428" cy="44.7" r="3.5"/><circle cx="474" cy="48.5" r="3.5"/><circle cx="520" cy="54.2" r="3.5"/><circle cx="600" cy="137.8" r="3.5"/></g>
    <circle class="hi" cx="336" cy="48.5" r="6"/>
    <text x="336" y="26" class="label hi-label">k = 60 · 0.2145</text>
    <text x="72" y="200" class="label">k = 0 · 0.2072</text>
    <text x="610" y="160" class="label end">k = 500 · 0.2098</text>
  </svg>`;
}

function benchmarkTableFigure() {
  const rows = DATASETS.map(({ id, name, about }) => {
    const result = bench.datasets[id];
    const { keyword, meaning, rrf } = result["ndcg@10"];
    const bestSingle = Math.max(keyword, meaning);
    const change = rrf - bestSingle;
    const best = Math.max(keyword, meaning, rrf);
    const td = (value) => `<td class="num${value === best ? " strong" : ""}">${three(value)}</td>`;
    return `<tr>
      <td><b>${name}</b><small>${about}</small></td>
      <td class="num">${result.queries}</td>
      ${td(keyword)}${td(meaning)}${td(rrf)}
      <td class="num ${change >= 0 ? "up" : "down"}">${signed(change, 3)} (${signed((change / bestSingle) * 100, 0)}%)</td>
    </tr>`;
  });
  return `<table>
    <thead><tr><th>Test set</th><th class="num">Questions</th><th class="num kw">Keyword</th><th class="num mn">Meaning</th><th class="num">RRF</th><th class="num">RRF vs better searcher</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>
  <p class="note">nDCG@10: quality of the top 10 results, from 0 to 1. Higher is better.</p>`;
}

function headToHeadFigure() {
  const rows = DATASETS.map(({ id, name }) => {
    const result = bench.datasets[id];
    const vsMeaning = result["ndcg@10"].meaning >= result["ndcg@10"].keyword;
    const { better, same, worse } = vsMeaning ? result.rrf_vs_meaning : result.rrf_vs_keyword;
    return `<div class="h2h">
      <div class="h2h-label"><b>${name}</b><span>vs ${vsMeaning ? "meaning" : "keyword"} search</span></div>
      <div class="h2h-bar">
        <span class="better" style="flex:${better}">${better}</span><span class="same" style="flex:${same}">${same}</span><span class="worse" style="flex:${worse}">${worse}</span>
      </div>
    </div>`;
  });
  return `${rows.join("")}
    <div class="legend"><span><i class="better"></i>RRF better</span><span><i class="same"></i>Same score</span><span><i class="worse"></i>RRF worse</span></div>`;
}

function kSweepFigure() {
  const ks = ["1", "10", "60", "500"];
  const rows = DATASETS.map(({ id, name }) => {
    const sweep = bench.datasets[id].k_sweep;
    const best = Math.max(...ks.map((k) => sweep[k]));
    return `<tr><td><b>${name}</b></td>${ks.map((k) => `<td class="num${sweep[k] === best ? " strong best" : ""}">${three(sweep[k])}</td>`).join("")}</tr>`;
  });
  return `<table>
    <thead><tr><th>Test set</th>${ks.map((k) => `<th class="num">k = ${k}</th>`).join("")}</tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>
  <p class="note">nDCG@10 of RRF for different values of k. Best value per test set highlighted.</p>`;
}

// In story order.
const FIGURES = [
  ["01-hero-merge.png", heroFigure],
  ["02-reciprocals.png", reciprocalsFigure],
  ["03-formula.png", formulaFigure],
  ["04-search-results.png", searchResultsFigure],
  ["05-points-per-position.png", pointsFigure],
  ["06-rrf-totals.png", totalsFigure],
  ["07-k-points.png", kPointsFigure],
  ["08-k-paper-chart.png", paperChartFigure],
  ["09-benchmark-table.png", benchmarkTableFigure],
  ["10-head-to-head.png", headToHeadFigure],
  ["11-k-sweep.png", kSweepFigure],
];

// ------------------------------------------------------------------ styling

const FONTS = "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=JetBrains+Mono:wght@400;600&display=block";

const CSS = `
  :root { --ink:#161C2A; --muted:#586275; --rule:#D7DDE7; --ground:#F2F4F7; --kw:#2A55C6; --kw-soft:#E2E9FA; --mn:#A95A12; --mn-soft:#F7E8D8;
          --good:#2A7447; --good-soft:#DCEFE3; --bad:#AE3B2A; --bad-soft:#F6DFDA;
          --sans:"Bricolage Grotesque","Avenir Next","Segoe UI",sans-serif; --mono:"JetBrains Mono",Menlo,monospace; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #fff; color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.4; }
  .fig { padding: 24px; }
  .num, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

  .formula { display: flex; align-items: center; justify-content: center; gap: 14px; padding: 14px 0; font-family: var(--mono); font-size: 18px; }
  .frac { display: inline-flex; flex-direction: column; align-items: center; line-height: 1.35; }
  .frac > span:first-child { border-bottom: 2px solid currentColor; padding: 0 8px 3px; }
  .frac > span:last-child { padding: 3px 8px 0; }
  .frac.kw { color: var(--kw); } .frac.mn { color: var(--mn); }
  .op { color: var(--muted); }
  .muted { color: var(--muted); }
  .strong { font-weight: 700; }
  .note { margin: 14px 0 0; font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .rel { display:inline-block; font-style:normal; font-family:var(--mono); font-size:10px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
         color:var(--good); background:var(--good-soft); padding:1px 6px; border-radius:4px; margin-left:4px; vertical-align:1px; white-space:nowrap; }

  .question { font-size: 19px; font-weight: 700; padding-bottom: 12px; margin-bottom: 14px; border-bottom: 1px solid var(--rule); }
  .question span { display:block; font-family:var(--mono); font-size:10.5px; font-weight:400; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; }
  .merge { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .col { display: flex; flex-direction: column; gap: 6px; }
  .col-head { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; }
  .col.kw .col-head { color: var(--kw); } .col.mn .col-head { color: var(--mn); }
  .chip { display: flex; gap: 8px; align-items: baseline; padding: 7px 9px; border-radius: 6px; font-size: 13px; line-height: 1.3; border: 1px solid transparent; }
  .chip b { font-weight: 500; }
  .chip .rk { font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .chip i { font-style: normal; margin-left: auto; font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
  .kw .chip { background: var(--kw-soft); } .mn .chip { background: var(--mn-soft); } .fused .chip { background: var(--ground); border-color: var(--rule); }
  .chip.is-rel { border-color: var(--good); box-shadow: inset 3px 0 0 var(--good); }
  .key { margin-top: 14px; display: flex; gap: 8px; align-items: center; font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .key i { width: 13px; height: 13px; border-radius: 3px; border: 1px solid var(--good); box-shadow: inset 3px 0 0 var(--good); }

  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel { border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px 4px; }
  .panel-head { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); padding-bottom: 8px; }
  .panel.kw .panel-head span:first-child { color: var(--kw); } .panel.mn .panel-head span:first-child { color: var(--mn); }
  .hit { display: grid; grid-template-columns: 14px 1fr auto; gap: 8px; padding: 8px 0; border-top: 1px solid var(--rule); font-size: 13px; line-height: 1.35; }
  .hit .rk, .hit .sc { font-family: var(--mono); font-size: 11.5px; }
  .hit .rk { color: var(--muted); }
  .panel.kw .sc { color: var(--kw); } .panel.mn .sc { color: var(--mn); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-family: var(--mono); font-size: 10.5px; font-weight: 400; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); padding: 6px 8px; border-bottom: 1.5px solid var(--ink); }
  th.kw { color: var(--kw); } th.mn { color: var(--mn); }
  td { padding: 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  td.num, th.num { text-align: right; white-space: nowrap; font-size: 12px; }
  td small { display: block; color: var(--muted); font-size: 11.5px; }
  tr.win td { background: var(--good-soft); }
  td.up { color: var(--good); } td.down { color: var(--bad); }
  td.best { background: var(--good-soft); }

  .kgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
  .khead { font-size: 22px; font-weight: 700; margin-bottom: 10px; }
  .bar { display: grid; grid-template-columns: 30px 1fr 58px; gap: 8px; align-items: center; font-size: 12px; margin-bottom: 6px; }
  .bar span:first-child { font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .bar .num { text-align: right; font-size: 11.5px; }
  .track { height: 12px; background: var(--ground); border-radius: 3px; overflow: hidden; }
  .fill { height: 100%; background: var(--kw); border-radius: 3px; }
  .duels { margin-top: 14px; display: grid; gap: 6px; }
  .duel { display: grid; grid-template-columns: 1fr auto; padding: 8px 10px; border: 1px solid var(--rule); border-radius: 6px; font-size: 13px; }
  .duel span:not(.num) { grid-column: 1; color: var(--muted); font-size: 11.5px; }
  .duel .num { grid-row: 1 / span 2; grid-column: 2; align-self: center; font-size: 12.5px; }
  .duel.won { border-color: var(--ink); background: var(--ground); }
  .verdict { margin: 2px 0 0; font-size: 12.5px; font-weight: 500; }

  .chart { display: block; width: 100%; height: auto; }
  .chart text { font-family: var(--mono); font-size: 11px; fill: var(--muted); }
  .chart .ty text { text-anchor: end; } .chart .tx text, .chart .axis { text-anchor: middle; }
  .chart .axis { fill: var(--ink); font-size: 12px; }
  .chart .grid line { stroke: var(--rule); }
  .chart .line { fill: none; stroke: var(--kw); stroke-width: 2; }
  .chart .dashed { stroke-dasharray: 3 5; }
  .chart .dots circle { fill: var(--kw); }
  .chart .hi { fill: var(--mn); stroke: #fff; stroke-width: 2; }
  .chart .label { fill: var(--ink); font-size: 11.5px; }
  .chart .hi-label { fill: var(--mn); font-weight: 600; text-anchor: middle; }
  .chart .end { text-anchor: end; }

  .h2h { display: grid; grid-template-columns: 140px 1fr; gap: 14px; align-items: center; margin-bottom: 12px; }
  .h2h-label { display: flex; flex-direction: column; }
  .h2h-label b { font-weight: 500; font-size: 15px; }
  .h2h-label span { font-family: var(--mono); font-size: 10.5px; color: var(--muted); }
  .h2h-bar { display: flex; gap: 2px; height: 32px; border-radius: 6px; overflow: hidden; }
  .h2h-bar span { display: flex; align-items: center; justify-content: center; min-width: 34px; font-family: var(--mono); font-size: 12px; font-weight: 600; }
  .better { background: var(--good-soft); color: var(--good); } .same { background: var(--ground); color: var(--muted); } .worse { background: var(--bad-soft); color: var(--bad); }
  .legend { display: flex; gap: 18px; margin-top: 6px; font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
  .legend .better { border: 1px solid var(--good); } .legend .same { border: 1px solid var(--rule); } .legend .worse { border: 1px solid var(--bad); }
`;

const documentFor = (body) =>
  `<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="${FONTS}"><style>${CSS}</style></head><body><div class="fig">${body}</div></body></html>`;

// Waits for the web fonts, then reports the figure's height.
const MEASURE = `(async () => {
  const link = document.querySelector('link[rel="stylesheet"]');
  if (!link.sheet) await new Promise((done) => { link.onload = link.onerror = done; });
  await Promise.all(['500 15px "Bricolage Grotesque"', '700 15px "Bricolage Grotesque"', '400 12px "JetBrains Mono"', '600 12px "JetBrains Mono"']
    .map((font) => document.fonts.load(font)));
  await document.fonts.ready;
  return Math.ceil(document.querySelector(".fig").getBoundingClientRect().height);
})()`;

// ------------------------------------------------------------------ headless Chrome (DevTools protocol)

async function launchChrome() {
  const profile = await mkdtemp(join(tmpdir(), "rrf-chrome-"));
  const process = spawn(CHROME, [
    "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
    "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "about:blank",
  ], { stdio: ["ignore", "ignore", "pipe"] });

  const endpoint = await new Promise((resolve, reject) => {
    let log = "";
    process.stderr.on("data", (chunk) => {
      log += chunk;
      const match = log.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) resolve(new URL(match[1]));
    });
    process.on("exit", (code) => reject(new Error(`Chrome exited with code ${code}\n${log}`)));
  });

  return {
    port: endpoint.port,
    async close() {
      const exited = new Promise((resolve) => process.once("exit", resolve));
      process.kill();
      await exited; // Chrome writes to its profile until it has fully exited
      await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    },
  };
}

async function openTab(port) {
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });

  const pending = new Map();
  let lastId = 0;
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    message.error ? request.reject(new Error(message.error.message)) : request.resolve(message.result);
  };

  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++lastId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });

  return { send, close: () => socket.close() };
}

async function renderPng(tab, html, file) {
  await tab.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: 800, deviceScaleFactor: SCALE, mobile: false });
  const { frameTree } = await tab.send("Page.getFrameTree");
  await tab.send("Page.setDocumentContent", { frameId: frameTree.frame.id, html });

  const { result } = await tab.send("Runtime.evaluate", { expression: MEASURE, awaitPromise: true, returnByValue: true });
  const height = result.value;

  await tab.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height, deviceScaleFactor: SCALE, mobile: false });
  const { data } = await tab.send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: WIDTH, height, scale: 1 } });
  await writeFile(join(IMAGE_DIR, file), Buffer.from(data, "base64"));
  console.log(`${file}  ${WIDTH * SCALE}×${height * SCALE}`);
}

// ------------------------------------------------------------------ main

await mkdir(IMAGE_DIR, { recursive: true });
const chrome = await launchChrome();
try {
  const tab = await openTab(chrome.port);
  for (const [file, figure] of FIGURES) {
    await renderPng(tab, documentFor(figure()), file);
  }
  tab.close();
} finally {
  await chrome.close();
}
