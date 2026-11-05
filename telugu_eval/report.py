"""Build results from saved outputs (offline) and export data for the site.

    python -m telugu_eval score         -> results/results.json, results/per_item.jsonl, results/RESULTS.md
    python -m telugu_eval judge         -> calls the judge model on failed short answers (API, cached)
    python -m telugu_eval build-site    -> score + site/data/*.json + README results block
"""

from __future__ import annotations

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
    return d


def build_results(models: list[ModelConfig], split: str, *, data_dir: Path = dataio.DATA_DIR,
                  runs_dir: Path = RUNS_DIR, judge_cfg: ModelConfig | None = None) -> tuple[dict, list[dict]]:
    sel = select_items(split, data_dir)
    rows, per_item = [], []
    for m in models_with_outputs(models, runs_dir):
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
    results = {
        "benchmark": "Telugu-LLM-Eval",
        "dataset_version": DATASET_VERSION,
        "tool_version": __version__,
        "generated": date.today().isoformat(),
        "split": split,
        "split_note": SPLIT_NOTES[split],
        "prompt_version": PROMPT_VERSION,
        "template_hash": template_hash(),
        "dataset_fingerprint": sel.fingerprint(),
        "judge": ({"name": judge_cfg.name, "display_name": judge_cfg.display_name, "version": JUDGE_VERSION}
                  if judge_cfg else None),
        "dataset": dataset_stats(sel),
        "models": rows,
    }
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


def export_site(res: dict, data_dir: Path = dataio.DATA_DIR, site_data: Path = SITE_DATA) -> None:
    site_data.mkdir(parents=True, exist_ok=True)
    (site_data / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    samples = sample_items(data_dir)
    samples["prompts"] = prompt_examples(data_dir)
    samples["judge_prompt_template"] = JUDGE_PROMPT
    (site_data / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
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
