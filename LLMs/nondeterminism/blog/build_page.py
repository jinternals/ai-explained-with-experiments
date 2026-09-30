"""Assemble blog/index.html from blog/page/ and the model run in out/.

    python blog/build_page.py [out/model-....json]

The batch-size explorer reads its data from the model results file, so after running
lab/model_experiment.py on another Mac this rebuilds the explorer with that Mac's answers.
The prose and tables quote the Apple M4 Pro run.
"""
import json, sys
from pathlib import Path

BLOG = Path(__file__).resolve().parent
OUT = BLOG.parent / "out"

src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(OUT.glob("model-*.json"))[0]
runs = json.loads(src.read_text())["A_plain"]
page = BLOG / "page"
html = ((page / "body.html").read_text()
        .replace("{{CSS}}", (page / "base.css").read_text())
        .replace("{{JS}}", (page / "runtime.js").read_text())
        .replace("{{DATA}}", json.dumps(runs, ensure_ascii=False)))
(BLOG / "index.html").write_text(html)
print(f"blog/index.html written ({len(html) // 1024} KB) using {src.name}")
