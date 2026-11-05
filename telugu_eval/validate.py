"""Dataset-level validation: everything in ``schema`` plus cross-record checks.

Cross-record checks:
* duplicate ids (items and passages)
* duplicate questions after normalisation (error) and near-duplicates (warning)
  - for reading comprehension a question is only a duplicate if it is on the
    same passage
* gold-letter balance for MCQ (a model biased to "B" should not score well)
* unused passages
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from . import dataio
from .normalize import normalize_for_match
from .schema import LETTERS, Report, check_item, check_passage

NEAR_DUP_THRESHOLD = 0.90
GOLD_BALANCE_MIN_ITEMS = 8
GOLD_BALANCE_MAX_SHARE = 0.40
TARGET_PER_CATEGORY = 50


@dataclass
class DatasetReport:
    report: Report = field(default_factory=Report)
    parse_errors: list = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    example_passages: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.parse_errors and not self.report.errors


def _dup_key(item: dict) -> str:
    q = normalize_for_match(item.get("question") or "")
    if item.get("category") == "reading_comprehension":
        return f"{item.get('passage_id')}::{q}"
    return q


def cross_checks(items: list[dict], passages: list[dict], rep: Report, *, check_balance: bool = True) -> None:
    # duplicate ids
    for kind, recs, key in (("item", items, "id"), ("passage", passages, "passage_id")):
        counts = Counter(r.get(key) for r in recs if r.get(key))
        for rid, c in counts.items():
            if c > 1:
                rep.error(str(rid), key, f"duplicate {kind} id ({c} records)")

    # duplicate / near-duplicate questions (ignore retired items)
    live = [it for it in items if it.get("status") != "retired"]
    seen: dict[str, str] = {}
    keyed = []
    for it in live:
        k = _dup_key(it)
        if not k.split("::")[-1]:
            continue
        if k in seen:
            rep.error(str(it.get("id")), "question", f"duplicate of {seen[k]} after normalisation")
        else:
            seen[k] = str(it.get("id"))
            keyed.append((str(it.get("id")), it.get("passage_id"), k.split("::")[-1]))
    # near duplicates: O(n^2) is fine at benchmark scale (hundreds of items)
    for i in range(len(keyed)):
        id_a, pa, qa = keyed[i]
        for j in range(i + 1, len(keyed)):
            id_b, pb, qb = keyed[j]
            if pa != pb:
                continue
            if abs(len(qa) - len(qb)) > 0.2 * max(len(qa), len(qb)):
                continue
            ratio = SequenceMatcher(None, qa, qb, autojunk=False).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                rep.warn(id_b, "question", f"near-duplicate of {id_a} (similarity {ratio:.2f})")

    # passage references usage
    used = {it.get("passage_id") for it in items}
    for p in passages:
        if p.get("passage_id") not in used:
            rep.warn(str(p.get("passage_id")), "passage_id", "passage has no questions yet")

    # gold letter balance
    if check_balance:
        mcq = [it for it in live if it.get("answer_type") == "mcq" and it.get("gold_option") in tuple(LETTERS)]
        if len(mcq) >= GOLD_BALANCE_MIN_ITEMS:
            c = Counter(it["gold_option"] for it in mcq)
            letter, n = c.most_common(1)[0]
            if n / len(mcq) > GOLD_BALANCE_MAX_SHARE:
                dist = ", ".join(f"{k}={c.get(k, 0)}" for k in "ABCD")
                rep.warn("dataset", "gold_option",
                         f"{letter} is gold in {n}/{len(mcq)} MCQ items ({dist}); spread correct answers across positions")


def validate_dataset(data_dir: Path = dataio.DATA_DIR) -> DatasetReport:
    out = DatasetReport()
    files = {
        "items": dataio.read_jsonl(data_dir / dataio.ITEMS_FILE),
        "passages": dataio.read_jsonl(data_dir / dataio.PASSAGES_FILE),
        "examples": dataio.read_jsonl(data_dir / dataio.EXAMPLES_FILE),
        "example_passages": dataio.read_jsonl(data_dir / dataio.EXAMPLE_PASSAGES_FILE),
    }
    for f in files.values():
        out.parse_errors.extend(f.errors)
    out.items = files["items"].records
    out.passages = files["passages"].records
    out.examples = files["examples"].records
    out.example_passages = files["example_passages"].records

    rep = out.report
    pids = {p.get("passage_id") for p in out.passages}
    ex_pids = {p.get("passage_id") for p in out.example_passages}

    for p in out.passages:
        rep.extend(check_passage(p))
    for p in out.example_passages:
        rep.extend(check_passage(p, example=True))
    for it in out.items:
        rep.extend(check_item(it, passage_ids=pids))
    for it in out.examples:
        rep.extend(check_item(it, example=True, passage_ids=ex_pids))

    cross_checks(out.items, out.passages, rep)
    cross_checks(out.examples, out.example_passages, rep, check_balance=False)

    # Examples must never leak into the scored set.
    real_q = {_dup_key(it) for it in out.items}
    for ex in out.examples:
        if _dup_key(ex) in real_q:
            rep.error(str(ex.get("id")), "question", "example item duplicated in items.jsonl (examples must stay out of scoring)")
    return out


def summary(out: DatasetReport) -> str:
    items = out.items
    lines = []
    by_status = Counter(i.get("status") for i in items)
    reviewed = [i for i in items if i.get("status") == "reviewed"]
    lines.append(f"items: {len(items)} total | reviewed {by_status.get('reviewed', 0)} | "
                 f"draft {by_status.get('draft', 0)} | retired {by_status.get('retired', 0)}")
    lines.append(f"passages: {len(out.passages)} | examples: {len(out.examples)} (never scored)")
    lines.append("")
    lines.append(f"{'category':24}{'reviewed':>9}{'all':>6}{'target':>8}")
    for cat in ("cultural", "code_mixed", "reading_comprehension"):
        n_rev = sum(1 for i in reviewed if i.get("category") == cat)
        n_all = sum(1 for i in items if i.get("category") == cat and i.get("status") != "retired")
        lines.append(f"{cat:24}{n_rev:>9}{n_all:>6}{TARGET_PER_CATEGORY:>8}")
    for fld in ("script", "difficulty", "answer_type", "region"):
        c = Counter(i.get(fld) for i in items if i.get("status") != "retired")
        if c:
            lines.append(f"{fld:12}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))))
    mcq = Counter(i.get("gold_option") for i in items if i.get("answer_type") == "mcq" and i.get("status") != "retired")
    if mcq:
        lines.append("gold letters: " + ", ".join(f"{k}={mcq.get(k, 0)}" for k in "ABCD"))
    return "\n".join(lines)


def run(data_dir: Path = dataio.DATA_DIR, *, strict: bool = False, quiet: bool = False) -> int:
    out = validate_dataset(data_dir)
    for e in out.parse_errors:
        print(e)
    issues = out.report.issues
    for i in sorted(issues, key=lambda x: (x.level != "error", x.record_id)):
        if quiet and i.level == "warning":
            continue
        print(i)
    if issues or out.parse_errors:
        print()
    print(summary(out))
    n_err = len(out.report.errors) + len(out.parse_errors)
    n_warn = len(out.report.warnings)
    print()
    print(f"{n_err} error(s), {n_warn} warning(s)")
    if n_err or (strict and n_warn):
        return 1
    return 0
