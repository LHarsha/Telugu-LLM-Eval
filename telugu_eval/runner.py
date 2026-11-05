"""Run models over the dataset and save every raw output to disk.

Layout::

    runs/<model-name>/<prompt-version>/outputs.jsonl   append-only, one line per call
    runs/<model-name>/<prompt-version>/run.json        settings + history of invocations

Each output line carries a ``key`` = hash(prompt + model settings). A run skips
items that already have a successful output with the same key, so:

* an interrupted run (crash, free-tier daily limit) resumes where it stopped;
* growing the dataset from 30 to 150 items only calls the model on new items;
* editing an item, a passage, the prompt or the model settings re-runs exactly
  the affected items.

Old lines are never deleted; the scorer uses the latest successful line whose
key matches the current item.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import DATASET_VERSION, __version__, dataio
from .config import ModelConfig
from .prompts import PROMPT_VERSION, build_prompt, template_hash
from .providers import Provider, ProviderError, make_provider

RUNS_DIR = dataio.REPO_ROOT / "runs"
SPLITS = ("reviewed", "all", "examples")
MAX_CONSECUTIVE_ERRORS = 5


def _sha(obj) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Selection:
    split: str
    items: list[dict]
    passages: dict[str, str]          # passage_id -> text

    def fingerprint(self) -> str:
        used = sorted({it.get("passage_id") for it in self.items if it.get("passage_id")})
        return _sha({"items": self.items, "passages": {p: self.passages[p] for p in used}})[:16]


def select_items(split: str, data_dir: Path = dataio.DATA_DIR) -> Selection:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "examples":
        items = dataio.read_jsonl(data_dir / dataio.EXAMPLES_FILE).records
        passages = dataio.read_jsonl(data_dir / dataio.EXAMPLE_PASSAGES_FILE).records
    else:
        items = dataio.read_jsonl(data_dir / dataio.ITEMS_FILE).records
        passages = dataio.read_jsonl(data_dir / dataio.PASSAGES_FILE).records
        allowed = {"reviewed"} if split == "reviewed" else {"reviewed", "draft"}
        items = [it for it in items if it.get("status") in allowed]
    pmap = {p["passage_id"]: p["text"] for p in passages if "passage_id" in p and "text" in p}
    return Selection(split, sorted(items, key=lambda it: it["id"]), pmap)


def call_key(prompt: str, cfg: ModelConfig) -> str:
    return _sha({"prompt": prompt, "model": cfg.public_dict()})[:24]


def run_dir(cfg: ModelConfig, runs_dir: Path = RUNS_DIR) -> Path:
    return runs_dir / cfg.name / PROMPT_VERSION


def load_outputs(path: Path) -> list[dict]:
    return dataio.read_jsonl(path).records


def done_keys(outputs: list[dict]) -> set[str]:
    return {o["key"] for o in outputs if o.get("status") == "ok" and o.get("key")}


@dataclass
class RunSummary:
    model: str
    total: int
    skipped: int
    ok: int
    errors: int
    stopped_early: str | None
    out_dir: Path


def run_model(cfg: ModelConfig, selection: Selection, *, runs_dir: Path = RUNS_DIR,
              provider: Provider | None = None, limit: int | None = None,
              log=print) -> RunSummary:
    if provider is None and cfg.api_key_env and not cfg.api_key():
        msg = f"{cfg.name}: set {cfg.api_key_env} (environment or .env file) before running"
        log(msg)
        return RunSummary(cfg.name, len(selection.items), 0, 0, 0, msg, run_dir(cfg, runs_dir))
    out_dir = run_dir(cfg, runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "outputs.jsonl"
    already = done_keys(load_outputs(outputs_path))
    provider = provider or make_provider(cfg, log=log)

    todo = []
    for it in selection.items:
        prompt = build_prompt(it, selection.passages.get(it.get("passage_id")))
        key = call_key(prompt, cfg)
        if key not in already:
            todo.append((it, prompt, key))
    skipped = len(selection.items) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    started = _now()
    ok = errors = consecutive = 0
    stopped = None
    log(f"{cfg.name}: {len(selection.items)} items in split '{selection.split}', "
        f"{skipped} already done, {len(todo)} to run")
    with open(outputs_path, "a", encoding="utf-8", newline="\n") as fh:
        for n, (it, prompt, key) in enumerate(todo, 1):
            rec = {
                "item_id": it["id"], "key": key, "model": cfg.name, "split": selection.split,
                "prompt_version": PROMPT_VERSION, "template_hash": template_hash(),
                "timestamp": _now(), "prompt": prompt,
            }
            try:
                g = provider.generate(prompt)
                rec.update(status="ok", response_text=g.text, finish_reason=g.finish_reason,
                           usage=g.usage, latency_s=round(g.latency_s, 3), attempts=g.attempts,
                           request=g.request, raw=g.raw)
                ok += 1
                consecutive = 0
                log(f"  [{n}/{len(todo)}] {it['id']} ok ({g.latency_s:.1f}s) -> {g.text.strip()[:60]!r}")
            except ProviderError as e:
                if e.fatal:
                    # Not the item's fault (quota, key, model id): record nothing for
                    # the item, stop, and let the next run pick it up.
                    stopped = ("daily free-tier quota reached; re-run after the quota resets "
                               "(Gemini: midnight Pacific time)") if e.quota else f"fatal provider error: {e}"
                    log(f"  [{n}/{len(todo)}] {it['id']} STOP: {e}")
                    break
                rec.update(status="error", error=str(e), http_status=e.status)
                errors += 1
                consecutive += 1
                log(f"  [{n}/{len(todo)}] {it['id']} ERROR {e}")
            fh.write(dataio.dumps(rec) + "\n")
            fh.flush()
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                stopped = f"{consecutive} errors in a row"
                break

    _update_run_json(out_dir, cfg, selection, started, len(todo), ok, errors, stopped)
    if stopped:
        log(f"{cfg.name}: STOPPED: {stopped}. Re-run the same command to resume; finished items are kept.")
    log(f"{cfg.name}: {ok} ok, {errors} errors, {skipped} skipped -> {outputs_path}")
    return RunSummary(cfg.name, len(selection.items), skipped, ok, errors, stopped, out_dir)


def _update_run_json(out_dir: Path, cfg: ModelConfig, sel: Selection, started: str,
                     attempted: int, ok: int, errors: int, stopped: str | None) -> None:
    path = out_dir / "run.json"
    meta = {}
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta.update({
        "model": cfg.public_dict(),
        "prompt_version": PROMPT_VERSION,
        "template_hash": template_hash(),
        "dataset_version": DATASET_VERSION,
        "tool_version": __version__,
    })
    meta.setdefault("history", []).append({
        "started": started, "finished": _now(), "split": sel.split,
        "dataset_fingerprint": sel.fingerprint(), "n_items": len(sel.items),
        "attempted": attempted, "ok": ok, "errors": errors, "stopped_early": stopped,
    })
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def dry_run(cfg: ModelConfig, selection: Selection, runs_dir: Path = RUNS_DIR, log=print) -> None:
    already = done_keys(load_outputs(run_dir(cfg, runs_dir) / "outputs.jsonl"))
    todo = []
    for it in selection.items:
        p = build_prompt(it, selection.passages.get(it.get("passage_id")))
        if call_key(p, cfg) not in already:
            todo.append((it, p))
    temp = "n/a" if cfg.provider == "dummy" else (cfg.temperature if cfg.send_temperature else "not sent")
    log(f"{cfg.name}: would call {cfg.provider}/{cfg.model} on {len(todo)} of {len(selection.items)} items "
        f"(temperature: {temp})")
    if cfg.api_key_env and not cfg.api_key():
        log(f"  WARNING: {cfg.api_key_env} is not set")
    if cfg.rpm and todo:
        log(f"  rpm cap {cfg.rpm:g} -> at least {len(todo) / cfg.rpm:.1f} min")
    if todo:
        it, p = todo[0]
        log(f"--- first prompt ({it['id']}) ---\n{p}\n--- end ---")
