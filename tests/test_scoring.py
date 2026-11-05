"""Tests for answer extraction, matching, aggregation, judge plumbing and reports."""

import json

import pytest

from telugu_eval import CANARY, dataio
from telugu_eval.config import ModelConfig
from telugu_eval.judge import judge_key, judge_prompt, parse_verdict, run_judge
from telugu_eval.prompts import build_prompt
from telugu_eval.report import build_results, collect_judge_work, results_markdown, sample_items, update_readme
from telugu_eval.runner import call_key, run_model, select_items
from telugu_eval.scoring import (
    aggregate, bootstrap_ci, chance_accuracy, extract_mcq, extract_short, score_item, short_match,
)

OPTS = ["ఆంధ్రప్రదేశ్", "తెలంగాణ", "కర్ణాటక", "తమిళనాడు"]


# --- MCQ extraction ----------------------------------------------------------

@pytest.mark.parametrize("reply,expected", [
    ("Answer: B", "B"),
    ("answer: b", "B"),
    ("**Answer:** B", "B"),
    ("Answer: (B)", "B"),
    ("Answer: B. తెలంగాణ", "B"),
    ("Answer: B) తెలంగాణ", "B"),
    ("Final answer: C", "C"),
    ("Let me think.\nA seems wrong.\nAnswer: D", "D"),
    ("Answer: A\nWait, actually...\nAnswer: B", "B"),      # last answer line wins
    ("B", "B"),
    ("(b)", "B"),
    ("Answer: తెలంగాణ", "B"),                              # option text instead of letter
    ("తెలంగాణ", "B"),
    ("తెలంగాణ‌", "B"),                                # stray ZWNJ
])
def test_extract_mcq(reply, expected):
    ex = extract_mcq(reply, OPTS)
    assert ex.answer == expected, ex


@pytest.mark.parametrize("reply", [
    "", "I am not sure.", "Answer: E", "Answer: ", "A or B", "Answer: Andhra",
    "B is wrong and so is C, I would say A",   # no answer line, no bare letter
])
def test_extract_mcq_unparseable(reply):
    assert extract_mcq(reply, OPTS).answer is None


def test_extract_mcq_letter_followed_by_words():
    # The prompt asks for a letter, so a leading standalone letter is taken as the choice...
    assert extract_mcq("Answer: A state in south India", OPTS).answer == "A"
    # ...but a word that merely starts with a letter is not.
    assert extract_mcq("Answer: Andhra Pradesh", OPTS).answer is None


def test_extract_mcq_respects_option_count():
    assert extract_mcq("Answer: D", ["x", "y", "z"]).answer is None


# --- short answers -----------------------------------------------------------

def test_extract_short():
    assert extract_short("Answer: మామిడి").answer == "మామిడి"
    assert extract_short("**Answer:** మామిడి కాయ.").answer == "మామిడి కాయ."
    assert extract_short("thinking...\nAnswer: x\nAnswer: మామిడి").answer == "మామిడి"
    assert extract_short("మామిడి").method == "whole_reply"
    assert extract_short("line one\nline two").answer is None
    assert extract_short("Answer:").answer is None


@pytest.mark.parametrize("pred,golds,ok", [
    ("మామిడి", ["మామిడి", "mango"], True),
    ("మామిడి.", ["మామిడి"], True),
    ("  మా‍మిడి ", ["మామిడి"], True),           # ZWJ + spaces
    ("Mango", ["mango"], True),
    ("కై", ["కై"], True),                   # NFC
    ("౨", ["2"], True),                                # Telugu digit
    ("మామిడి కాయ", ["మామిడి"], False),                # strict: extra word fails (judge may accept)
    ("", ["మామిడి"], False),
    (None, ["మామిడి"], False),
])
def test_short_match(pred, golds, ok):
    assert short_match(pred, golds) is ok


def item(**kw):
    base = {"id": "cul-001", "category": "cultural", "script": "telugu", "difficulty": "easy", "region": "both",
            "answer_type": "mcq", "options": OPTS, "gold_option": "B", "question": "q"}
    base.update(kw)
    return base


def test_score_item():
    assert score_item(item(), "Answer: B").correct
    assert not score_item(item(), "Answer: C").correct
    s = score_item(item(), None)
    assert s.status == "missing" and not s.correct
    s = score_item(item(answer_type="short", gold_answers=["మామిడి"], options=None, gold_option=None), "Answer: మామిడి")
    assert s.correct and s.pred == "మామిడి"


# --- aggregation -------------------------------------------------------------

def test_bootstrap_ci_properties():
    lo, hi = bootstrap_ci([1] * 30 + [0] * 20)
    assert lo < 0.6 < hi and 0.4 < lo and hi < 0.8
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([1, 0] * 10) == bootstrap_ci([1, 0] * 10)  # seeded => reproducible


def test_aggregate_groups_missing_and_judge():
    scores = [
        score_item(item(id="cul-001"), "Answer: B"),
        score_item(item(id="cul-002", script="romanized"), "Answer: A"),
        score_item(item(id="cm-001", category="code_mixed", script="mixed", difficulty="hard"), "garbage"),
        score_item(item(id="cul-003"), None),
    ]
    short = score_item(item(id="rc-001", category="reading_comprehension", answer_type="short",
                            gold_answers=["రెండు"]), "Answer: రెండు ఎకరాలు")
    short.judge_correct = True
    scores.append(short)
    agg = aggregate(scores)
    assert agg["n_scored"] == 4 and agg["n_missing"] == 1 and not agg["complete"]
    assert agg["overall"]["accuracy"] == pytest.approx(1 / 4)
    assert agg["parse_failures"] == 1
    assert agg["groups"]["category"]["cultural"]["accuracy"] == pytest.approx(0.5)
    assert agg["groups"]["script"]["romanized"]["correct"] == 0
    assert "ci95" in agg["groups"]["category"]["cultural"]
    assert agg["judge"]["overall"]["accuracy"] == pytest.approx(2 / 4)   # judge reported separately
    assert agg["judge"]["judge_flipped_to_correct"] == 1
    assert agg["macro_category_accuracy"] == pytest.approx((0.5 + 0 + 0) / 3)


def test_chance_accuracy():
    items = [item(), item(options=["a", "b"]), item(answer_type="short", gold_answers=["x"])]
    assert chance_accuracy(items) == pytest.approx((0.25 + 0.5 + 0) / 3)
    assert chance_accuracy([]) is None


# --- judge -------------------------------------------------------------------

def test_parse_verdict():
    assert parse_verdict("CORRECT") is True
    assert parse_verdict("incorrect") is False
    assert parse_verdict("The answer is INCORRECT.") is False
    assert parse_verdict("Not correct... final: CORRECT") is True
    assert parse_verdict("maybe") is None


def test_judge_prompt_contains_all_golds_and_answer():
    it = item(answer_type="short", gold_answers=["మామిడి", "mango"], question="ఏ కాయ?")
    p = judge_prompt(it, "మామిడి కాయ")
    assert "- మామిడి\n- mango" in p and "మామిడి కాయ" in p and "ఏ కాయ?" in p


class FakeJudge:
    def __init__(self, reply="CORRECT"):
        self.reply, self.calls = reply, 0

    def generate(self, prompt):
        from telugu_eval.providers import Generation
        self.calls += 1
        return Generation(self.reply, "STOP", {}, {}, {}, 0.0, 1)


JUDGE = ModelConfig(name="j", provider="dummy", model="constant")


def test_run_judge_caches(tmp_path):
    it = item(answer_type="short", gold_answers=["మామిడి"])
    f = tmp_path / "judge" / "j.jsonl"
    fj = FakeJudge()
    assert run_judge([(it, "మామిడి కాయ")], JUDGE, f, provider=fj, log=lambda *a: None) == (1, None)
    assert run_judge([(it, "మామిడి కాయ")], JUDGE, f, provider=fj, log=lambda *a: None) == (0, None)
    assert fj.calls == 1
    rec = dataio.read_jsonl(f).records[0]
    assert rec["key"] == judge_key(judge_prompt(it, "మామిడి కాయ"), JUDGE) and rec["verdict"] is True


# --- end to end: run -> score -> report --------------------------------------

def make_dataset(tmp_path):
    items = [
        {"id": "cul-001", "category": "cultural", "subtopic": "f", "script": "telugu", "region": "both",
         "question": "ప్రశ్న ఒకటి", "answer_type": "mcq", "options": OPTS, "gold_option": "B",
         "difficulty": "easy", "author": "a", "status": "reviewed", "created": "2026-11-05", "canary": CANARY},
        {"id": "cm-001", "category": "code_mixed", "subtopic": "chat", "script": "romanized", "region": "none",
         "question": "Movie nachinda?", "answer_type": "short", "gold_answers": ["ledu", "లేదు"],
         "difficulty": "easy", "author": "a", "status": "reviewed", "created": "2026-11-05", "canary": CANARY},
        {"id": "rc-001", "category": "reading_comprehension", "subtopic": "s", "script": "telugu", "region": "none",
         "passage_id": "p-001", "question": "ఎన్ని ఎకరాలు?", "answer_type": "short", "gold_answers": ["రెండు"],
         "difficulty": "medium", "author": "a", "status": "draft", "created": "2026-11-05", "canary": CANARY},
    ]
    dataio.write_jsonl(tmp_path / dataio.ITEMS_FILE, items)
    dataio.write_jsonl(tmp_path / dataio.PASSAGES_FILE, [
        {"passage_id": "p-001", "title": "t", "script": "telugu", "text": "లక్ష్మికి రెండు ఎకరాల పొలం ఉంది.",
         "author": "a", "created": "2026-11-05", "canary": CANARY}])
    return items


class ScriptedModel:
    """Replies by item: maps a question substring to a reply."""

    def __init__(self, replies):
        self.replies = replies

    def generate(self, prompt):
        from telugu_eval.providers import Generation
        for k, v in self.replies.items():
            if k in prompt:
                return Generation(v, "STOP", {}, {}, {}, 0.0, 1)
        return Generation("no idea", "STOP", {}, {}, {}, 0.0, 1)


def test_end_to_end_score(tmp_path):
    make_dataset(tmp_path)
    runs = tmp_path / "runs"
    good = ModelConfig(name="good", provider="dummy", model="constant", display_name="Good")
    bad = ModelConfig(name="bad", provider="dummy", model="constant", display_name="Bad")
    sel = select_items("all", tmp_path)
    run_model(good, sel, runs_dir=runs, log=lambda *a: None,
              provider=ScriptedModel({"ప్రశ్న ఒకటి": "Answer: B", "Movie": "Answer: Ledu", "ఎకరాలు": "Answer: రెండు ఎకరాలు"}))
    run_model(bad, select_items("reviewed", tmp_path), runs_dir=runs, log=lambda *a: None,
              provider=ScriptedModel({"ప్రశ్న ఒకటి": "Answer: A"}))

    res, per_item = build_results([good, bad], "all", data_dir=tmp_path, runs_dir=runs)
    assert [r["name"] for r in res["models"]] == ["good", "bad"]         # complete before incomplete
    g, b = res["models"]
    assert g["complete"] and g["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert not b["complete"] and b["n_missing"] == 1
    assert res["split"] == "all" and "PRELIMINARY" in res["split_note"]
    assert res["dataset"]["n_items"] == 3 and res["dataset"]["by_category"]["cultural"] == 1
    assert len(per_item) == 6

    # reviewed split excludes the draft
    res_r, _ = build_results([good], "reviewed", data_dir=tmp_path, runs_dir=runs)
    assert res_r["models"][0]["n_items"] == 2 and res_r["models"][0]["overall"]["accuracy"] == 1.0

    # judge: the failed short answer is collected, judged once, and reported separately
    work = collect_judge_work(good, sel, runs)
    assert [(w[0]["id"], w[1]) for w in work] == [("rc-001", "రెండు ఎకరాలు")]
    from telugu_eval.judge import cache_path
    from telugu_eval.runner import run_dir
    run_judge(work, JUDGE, cache_path(run_dir(good, runs), JUDGE.name), provider=FakeJudge(), log=lambda *a: None)
    res_j, _ = build_results([good], "all", data_dir=tmp_path, runs_dir=runs, judge_cfg=JUDGE)
    m = res_j["models"][0]
    assert m["overall"]["accuracy"] == pytest.approx(2 / 3)               # strict unchanged
    assert m["judge"]["overall"]["accuracy"] == pytest.approx(1.0)         # judge-assisted separate
    assert res_j["judge"]["name"] == "j"

    # editing an item invalidates its old output (counts as missing, not silently reused)
    items = dataio.read_jsonl(tmp_path / dataio.ITEMS_FILE).records
    items[0]["question"] = "ప్రశ్న ఒకటి (మార్చబడింది)"
    dataio.write_jsonl(tmp_path / dataio.ITEMS_FILE, items)
    res2, _ = build_results([good], "all", data_dir=tmp_path, runs_dir=runs)
    assert res2["models"][0]["n_missing"] == 1

    md = results_markdown(res)
    assert "| Good | api |" in md and "(incomplete)" in md and "PRELIMINARY" in md


def test_call_key_matches_runner():
    it = item()
    cfg = ModelConfig(name="x", provider="dummy", model="constant")
    assert call_key(build_prompt(it), cfg) == call_key(build_prompt(dict(it)), cfg)


def test_readme_block_update(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro\n", encoding="utf-8")
    res = {"models": [], "dataset": {"n_items": 0, "chance_accuracy": None}, "split": "reviewed",
           "split_note": "", "prompt_version": "p1", "dataset_version": "v1", "dataset_fingerprint": "x", "generated": "d"}
    assert update_readme(res, readme)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "No results yet." in text and text.startswith("intro") and text.endswith("outro\n")


def test_sample_items_fall_back_to_examples(tmp_path):
    make_dataset(tmp_path)
    ex = [{"id": "ex-rc-1", "category": "reading_comprehension", "question": "q", "answer_type": "mcq",
           "options": ["a", "b"], "gold_option": "A", "passage_id": "ex-p-1", "status": "draft"}]
    dataio.write_jsonl(tmp_path / dataio.EXAMPLES_FILE, ex)
    dataio.write_jsonl(tmp_path / dataio.EXAMPLE_PASSAGES_FILE, [{"passage_id": "ex-p-1", "text": "పాఠం"}])
    s = sample_items(tmp_path)["categories"]
    assert s["cultural"]["source"] == "benchmark" and s["cultural"]["items"][0]["id"] == "cul-001"
    assert s["reading_comprehension"]["source"] == "format_examples"   # the rc item is only a draft
    assert s["reading_comprehension"]["items"][0]["passage"] == "పాఠం"
    json.dumps(s, ensure_ascii=False)
