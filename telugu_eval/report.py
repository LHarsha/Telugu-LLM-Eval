"""Build results from saved outputs (offline) and export data for the site.

    python -m telugu_eval score         -> results/results.json, results/per_item.jsonl, results/RESULTS.md
    python -m telugu_eval judge         -> calls the judge model on failed short answers (API, cached)
    python -m telugu_eval build-site    -> score + site/data/*.json + README results block
"""

from __future__ import annotations

import html
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

from . import DATASET_VERSION, __version__, dataio
from .config import ModelConfig
from .judge import JUDGE_PROMPT, JUDGE_VERSION, cache_path, judge_key, judge_prompt, load_cache, run_judge
from .prompts import PROMPT_VERSION, build_prompt, template_hash
from .runner import RUNS_DIR, Selection, call_key, load_outputs, run_dir, select_items
from .scoring import GROUPS, aggregate, chance_accuracy, score_item

RESULTS_DIR = dataio.REPO_ROOT / "results"
SITE_DATA = dataio.REPO_ROOT / "site" / "data"
README = dataio.REPO_ROOT / "README.md"

LLM_DRAFT_AUTHOR = "llm-draft"

SPLIT_NOTES = {
    "reviewed": "Only items marked reviewed by the author are scored.",
    "all": "PRELIMINARY: includes draft items that have not been re-checked yet.",
    "examples": "FORMAT EXAMPLES ONLY: not benchmark data. For testing the pipeline.",
}


def latest_responses(cfg: ModelConfig, sel: Selection, runs_dir: Path = RUNS_DIR) -> dict[str, str]:
    """item_id -> response text of the latest successful call matching the current prompt+settings."""
    outs = load_outputs(run_dir(cfg, runs_dir) / "outputs.jsonl")
    by_key = {}
    for o in outs:  # later lines win
        if o.get("status") == "ok" and o.get("key"):
            by_key[o["key"]] = o.get("response_text", "")
    res = {}
    for it in sel.items:
        k = call_key(build_prompt(it, sel.passages.get(it.get("passage_id"))), cfg)
        if k in by_key:
            res[it["id"]] = by_key[k]
    return res


def score_model(cfg: ModelConfig, sel: Selection, runs_dir: Path = RUNS_DIR,
                judge_cfg: ModelConfig | None = None) -> tuple[dict, list]:
    responses = latest_responses(cfg, sel, runs_dir)
    scores = [score_item(it, responses.get(it["id"])) for it in sel.items]
    if judge_cfg is not None:
        cache = load_cache(cache_path(run_dir(cfg, runs_dir), judge_cfg.name))
        by_id = {it["id"]: it for it in sel.items}
        for s in scores:
            if s.status == "scored" and s.answer_type == "short" and not s.correct:
                k = judge_key(judge_prompt(by_id[s.item_id], s.pred), judge_cfg)
                if k in cache:
                    s.judge_correct = bool(cache[k]["verdict"])
    agg = aggregate(scores)
    agg.update({
        "name": cfg.name, "display_name": cfg.display_name, "kind": cfg.kind,
        "provider": cfg.provider, "model": cfg.model, "settings": cfg.public_dict(),
    })
    if judge_cfg is not None and "judge" in agg:
        shorts_failed = sum(1 for s in scores if s.status == "scored" and s.answer_type == "short" and not s.correct)
        agg["judge"]["coverage"] = f"{agg['judge']['judged_items']}/{shorts_failed} failed short answers judged"
        # A model judging its own answers may be biased; flag it so readers can discount it.
        agg["judge"]["self_judged"] = judge_cfg.name == cfg.name
    return agg, scores


def models_with_outputs(models: list[ModelConfig], runs_dir: Path = RUNS_DIR) -> list[ModelConfig]:
    return [m for m in models if (run_dir(m, runs_dir) / "outputs.jsonl").exists()]


def dataset_stats(sel: Selection) -> dict:
    d = {"n_items": len(sel.items), "chance_accuracy": chance_accuracy(sel.items)}
    for g in GROUPS:
        d[f"by_{g}"] = dict(sorted(Counter(it.get(g, "none") for it in sel.items).items()))
    d["n_passages"] = len({it.get("passage_id") for it in sel.items if it.get("passage_id")})
    # Provenance: items drafted with an LLM are counted so the site can disclose it,
    # whether or not they have since been reviewed.
    llm = [it for it in sel.items if is_llm_drafted(it)]
    d["n_llm_drafted"] = len(llm)
    d["n_llm_drafted_reviewed"] = sum(1 for it in llm if it.get("status") == "reviewed")
    d["n_unreviewed"] = sum(1 for it in sel.items if it.get("status") != "reviewed")
    return d


def is_llm_drafted(item: dict) -> bool:
    return item.get("origin") == LLM_DRAFT_AUTHOR or item.get("author") == LLM_DRAFT_AUTHOR


def provenance_note(ds: dict) -> str:
    n, n_rev, total = ds.get("n_llm_drafted", 0), ds.get("n_llm_drafted_reviewed", 0), ds["n_items"]
    if not n:
        return ""
    parts = []
    if n_rev:
        who = "All" if n_rev == total else f"{n_rev} of {total}"
        parts.append(f"{who} items were drafted with an LLM, then checked by a native Telugu speaker (single annotator).")
    if n - n_rev:
        parts.append(f"{n - n_rev} of {total} items were drafted with an LLM and are pending native-speaker verification.")
    return " ".join(parts)


def build_results(models: list[ModelConfig], split: str, *, data_dir: Path = dataio.DATA_DIR,
                  runs_dir: Path = RUNS_DIR, judge_cfg: ModelConfig | None = None) -> tuple[dict, list[dict]]:
    sel = select_items(split, data_dir)
    rows, per_item = [], []
    # Only enabled models appear on the leaderboard; disabled ones keep their raw runs in runs/.
    for m in models_with_outputs([m for m in models if m.enabled], runs_dir):
        agg, scores = score_model(m, sel, runs_dir, judge_cfg)
        if agg["n_scored"] == 0:
            continue
        rows.append(agg)
        for s in scores:
            per_item.append({"model": m.name, "item_id": s.item_id, "status": s.status, "pred": s.pred,
                             "method": s.method, "correct": s.correct, "judge_correct": s.judge_correct})

    def rank_key(r):
        return (not r["complete"], r["kind"] == "baseline", -(r["overall"]["accuracy"] or 0))
    rows.sort(key=rank_key)
    stats = dataset_stats(sel)
    # "all" with no drafts in it is the same item set as "reviewed": don't call it preliminary.
    split_note = SPLIT_NOTES["reviewed"] if split == "all" and not stats["n_unreviewed"] else SPLIT_NOTES[split]
    results = {
        "benchmark": "Telugu-LLM-Eval",
        "dataset_version": DATASET_VERSION,
        "tool_version": __version__,
        "generated": date.today().isoformat(),
        "split": split,
        "split_note": split_note,
        "prompt_version": PROMPT_VERSION,
        "template_hash": template_hash(),
        "dataset_fingerprint": sel.fingerprint(),
        "judge": ({"name": judge_cfg.name, "display_name": judge_cfg.display_name, "version": JUDGE_VERSION}
                  if judge_cfg else None),
        "dataset": stats,
        "models": rows,
    }
    results["notice"] = notice_text(results)
    return results, per_item


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def results_markdown(res: dict) -> str:
    cats = ["cultural", "code_mixed", "reading_comprehension"]
    head = "| Model | Type | Overall (95% CI) | Cultural | Code-mixed | Reading comp. | n |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in res["models"]:
        o = r["overall"]
        ci = f" ({pct(o['ci95'][0])}–{pct(o['ci95'][1])})" if o.get("ci95") else ""
        c = r["groups"]["category"]
        cells = [pct(c.get(k, {}).get("accuracy")) for k in cats]
        name = r["display_name"] + ("" if r["complete"] else " *(incomplete)*")
        lines.append(f"| {name} | {r['kind']} | {pct(o['accuracy'])}{ci} | " + " | ".join(cells) + f" | {r['n_scored']} |")
    ds = res["dataset"]
    note = [
        "",
        f"Accuracy in %, strict scoring (exact match after normalisation). Split: `{res['split']}`: {res['split_note']}",
        *([f"**Note:** {provenance_note(ds)}"] if provenance_note(ds) else []),
        f"{ds['n_items']} items; chance level {pct(ds['chance_accuracy'])}%. Prompt `{res['prompt_version']}`, "
        f"dataset {res['dataset_version']} (`{res['dataset_fingerprint']}`), generated {res['generated']}.",
    ]
    if not res["models"]:
        return "No results yet."
    return "\n".join(lines + note)


def write_results(res: dict, per_item: list[dict], out_dir: Path = RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    dataio.write_jsonl(out_dir / "per_item.jsonl", per_item)
    (out_dir / "RESULTS.md").write_text("# Results\n\n" + results_markdown(res) + "\n", encoding="utf-8", newline="\n")


def update_readme(res: dict, readme: Path = README) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        return False
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    new = before + start + "\n" + results_markdown(res) + "\n" + end + after
    if new != text:
        readme.write_text(new, encoding="utf-8", newline="\n")
    return True


# --- site export -------------------------------------------------------------

SAMPLES_PER_CATEGORY = 5


def sample_items(data_dir: Path = dataio.DATA_DIR) -> dict:
    """Up to 5 reviewed items per category for the public examples page.

    Falls back to the format examples while there are not enough reviewed
    items, and says so in the output.
    """
    real = select_items("reviewed", data_dir)
    ex = select_items("examples", data_dir)
    out = {"categories": {}}
    for cat in ("cultural", "code_mixed", "reading_comprehension"):
        picked = [it for it in real.items if it["category"] == cat][:SAMPLES_PER_CATEGORY]
        source, sel = "benchmark", real
        if not picked:
            picked = [it for it in ex.items if it["category"] == cat][:SAMPLES_PER_CATEGORY]
            source, sel = "format_examples", ex
        out["categories"][cat] = {
            "source": source,
            "items": [{k: it.get(k) for k in ("id", "subtopic", "script", "region", "difficulty", "question",
                                              "answer_type", "options", "gold_option", "gold_answers")}
                      | ({"passage": sel.passages.get(it["passage_id"])} if it.get("passage_id") else {})
                      for it in picked],
        }
    return out


def prompt_examples(data_dir: Path = dataio.DATA_DIR) -> dict:
    """The exact prompts sent to models, rendered from the format examples (kept in sync with the code)."""
    ex = select_items("examples", data_dir)
    out = {}
    for label, pred in (("mcq", lambda i: i["answer_type"] == "mcq" and i["category"] != "reading_comprehension"),
                        ("short", lambda i: i["answer_type"] == "short" and i["category"] != "reading_comprehension"),
                        ("reading_comprehension", lambda i: i["category"] == "reading_comprehension")):
        it = next((i for i in ex.items if pred(i)), None)
        if it:
            out[label] = build_prompt(it, ex.passages.get(it.get("passage_id")))
    return out


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


SITE_CATS = (("cultural", "Cultural"), ("code_mixed", "Code-mixed"), ("reading_comprehension", "Reading comp."))


def static_board(res: dict) -> tuple[str, str]:
    """Static HTML for the leaderboard facts strip and table (JS re-renders it with controls)."""
    ds = res["dataset"]
    n_models = sum(1 for m in res["models"] if m["kind"] != "baseline" and m["complete"])
    chance = f"{pct(ds['chance_accuracy'])}%" if ds.get("chance_accuracy") is not None else "–"
    facts = "".join(f"<div><b>{_esc(b)}</b><span>{_esc(s)}</span></div>" for b, s in (
        (ds["n_items"], "questions"), (3, "capabilities"),
        (n_models, "model evaluated" if n_models == 1 else "models evaluated"), (chance, "random-guess level")))
    head = ("<thead><tr><th>#</th><th>Model</th><th>Overall</th>"
            + "".join(f"<th>{n}</th>" for _, n in SITE_CATS) + "<th>Items</th></tr></thead>")
    if not res["models"]:
        return facts, head + f'<tbody><tr><td class="empty" colspan="{4 + len(SITE_CATS)}">No results yet.</td></tr></tbody>'
    rows, rank = [], 0
    for m in res["models"]:
        ranked = m["complete"] and m["kind"] != "baseline"
        rank += ranked
        o = m["overall"]
        ci = f'<span class="sub">{pct(o["ci95"][0])}–{pct(o["ci95"][1])}</span>' if o.get("ci95") else ""
        status = "" if m["complete"] else f'<span class="sub">incomplete: {m["n_scored"]}/{m["n_items"]} items answered so far</span>'
        cats = "".join(f'<td>{pct(m["groups"]["category"].get(k, {}).get("accuracy"))}</td>' for k, _ in SITE_CATS)
        cls = " ".join(c for c in ("baseline" if m["kind"] == "baseline" else "", "" if m["complete"] else "incomplete") if c)
        rows.append(f'<tr class="{cls}"><td class="rank">{rank if ranked else ""}</td>'
                    f'<td class="model"><b>{_esc(m["display_name"])}</b><span class="tag">{_esc(m["kind"])}</span>{status}</td>'
                    f'<td class="overall"><span class="num">{pct(o["accuracy"])}{ci}</span></td>{cats}<td>{m["n_scored"]}</td></tr>')
    return facts, head + "<tbody>" + "".join(rows) + "</tbody>"


def _replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        return text
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + content + end + after


def notice_text(res: dict) -> str:
    """Plain-text disclosure shown above the leaderboard (empty when nothing to disclose)."""
    parts = []
    ds = res["dataset"]
    if res["split"] != "reviewed" and (ds.get("n_unreviewed") or res["split"] == "examples"):
        parts.append(res["split_note"])
    if provenance_note(ds):
        parts.append(provenance_note(ds))
    return " ".join(parts)


def prerender_index(res: dict, index: Path = SITE_DATA.parent / "index.html") -> bool:
    if not index.exists():
        return False
    text = index.read_text(encoding="utf-8")
    facts, board = static_board(res)
    note = notice_text(res)
    notice = f'<p class="notice">{_esc(note)}</p>' if note else ""
    new = _replace_block(_replace_block(_replace_block(text, "facts", facts), "board", board), "notice", notice)
    if new != text:
        index.write_text(new, encoding="utf-8", newline="\n")
    return True


def export_site(res: dict, data_dir: Path = dataio.DATA_DIR, site_data: Path = SITE_DATA) -> None:
    site_data.mkdir(parents=True, exist_ok=True)
    (site_data / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    samples = sample_items(data_dir)
    samples["prompts"] = prompt_examples(data_dir)
    samples["judge_prompt_template"] = JUDGE_PROMPT
    (site_data / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    prerender_index(res, site_data.parent / "index.html")
    # The dataset itself is published with the site (open dataset).
    dl = site_data / "dataset"
    dl.mkdir(exist_ok=True)
    for f in (dataio.ITEMS_FILE, dataio.PASSAGES_FILE, "LICENSE"):
        src = data_dir / f
        if src.exists():
            shutil.copyfile(src, dl / f)


def collect_judge_work(cfg: ModelConfig, sel: Selection, runs_dir: Path = RUNS_DIR) -> list[tuple[dict, str | None]]:
    responses = latest_responses(cfg, sel, runs_dir)
    work = []
    for it in sel.items:
        if it["answer_type"] != "short" or it["id"] not in responses:
            continue
        s = score_item(it, responses[it["id"]])
        if not s.correct:
            work.append((it, s.pred))
    return work


def judge_models(models: list[ModelConfig], judge_cfg: ModelConfig, split: str, *, data_dir: Path = dataio.DATA_DIR,
                 runs_dir: Path = RUNS_DIR, log=print) -> int:
    sel = select_items(split, data_dir)
    status = 0
    for m in models_with_outputs(models, runs_dir):
        if m.kind == "baseline":
            continue
        work = collect_judge_work(m, sel, runs_dir)
        log(f"{m.name}: {len(work)} failed short answers to judge with {judge_cfg.name}")
        n, stop = run_judge(work, judge_cfg, cache_path(run_dir(m, runs_dir), judge_cfg.name), log=log)
        if stop:
            log(f"  judge stopped: {stop}. Re-run to resume.")
            status = 1
            break
    return status
