"""Optional LLM judge for short answers. Clearly labelled, never the headline.

Only short-answer outputs that FAIL the strict normalised match are sent to the
judge, with the question, every accepted gold answer and the model's extracted
answer. The judge answers CORRECT or INCORRECT. Verdicts are cached in
``runs/<model>/<prompt-version>/judge/<judge-name>.jsonl`` and keyed by a hash
of exactly what the judge saw, so scoring stays offline and reproducible.

Known weaknesses (documented in the README): the judge is itself an LLM with
imperfect Telugu, it can be lenient with near-misses, and if it belongs to the
same family as the judged model it may favour that model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import dataio
from .config import ModelConfig
from .providers import Provider, ProviderError, make_provider

JUDGE_VERSION = "j1"

JUDGE_PROMPT = """You are grading answers to a Telugu language test. Decide whether the student's answer means the same as one of the accepted answers.

Rules:
- Spelling variants, Telugu vs Romanized vs English forms of the same word, and extra words that do not change the meaning are CORRECT.
- A different item, a wrong number, a vaguer or broader answer, or an answer that adds a wrong claim is INCORRECT.
- If the student's answer is empty, it is INCORRECT.

Question:
{question}

Accepted answers:
{golds}

Student's answer:
{answer}

Reply with exactly one word: CORRECT or INCORRECT."""


def judge_prompt(item: dict, answer: str | None) -> str:
    golds = "\n".join(f"- {g}" for g in item["gold_answers"])
    return JUDGE_PROMPT.format(question=item["question"].strip(), golds=golds, answer=(answer or "").strip())


def judge_key(prompt: str, judge_cfg: ModelConfig) -> str:
    s = json.dumps({"v": JUDGE_VERSION, "prompt": prompt, "judge": judge_cfg.public_dict()}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def parse_verdict(text: str) -> bool | None:
    t = (text or "").strip().upper()
    words = re.findall(r"\b(INCORRECT|CORRECT)\b", t)
    if not words:
        return None
    return words[-1] == "CORRECT"


def cache_path(model_run_dir: Path, judge_name: str) -> Path:
    return model_run_dir / "judge" / f"{judge_name}.jsonl"


def load_cache(path: Path) -> dict[str, dict]:
    return {r["key"]: r for r in dataio.read_jsonl(path).records if r.get("verdict") is not None}


def run_judge(pending: list[tuple[dict, str | None]], judge_cfg: ModelConfig, cache_file: Path,
              provider: Provider | None = None, log=print) -> tuple[int, str | None]:
    """Judge (item, extracted_answer) pairs not already cached. Returns (n_new, stop_reason)."""
    cache = load_cache(cache_file)
    todo = []
    for item, ans in pending:
        p = judge_prompt(item, ans)
        k = judge_key(p, judge_cfg)
        if k not in cache:
            todo.append((item, ans, p, k))
    if not todo:
        return 0, None
    provider = provider or make_provider(judge_cfg, log=log)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(cache_file, "a", encoding="utf-8", newline="\n") as fh:
        for item, ans, p, k in todo:
            try:
                g = provider.generate(p)
            except ProviderError as e:
                if e.fatal:
                    return n, str(e)
                log(f"    judge error on {item['id']}: {e}")
                continue
            verdict = parse_verdict(g.text)
            fh.write(dataio.dumps({"key": k, "item_id": item["id"], "judge": judge_cfg.name,
                                   "judge_version": JUDGE_VERSION, "answer": ans, "prompt": p,
                                   "response_text": g.text, "verdict": verdict}) + "\n")
            fh.flush()
            n += 1
            log(f"    judged {item['id']}: {g.text.strip()[:20]!r} -> {verdict}")
    return n, None
