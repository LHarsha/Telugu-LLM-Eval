"""Tests for prompts, model config, providers and the runner. Fully offline."""

import json

import pytest

from telugu_eval import CANARY, dataio
from telugu_eval.config import ConfigError, ModelConfig, load_models
from telugu_eval.prompts import build_prompt, template_hash
from telugu_eval.providers import (
    GeminiProvider, HTTPResult, OpenAICompatibleProvider, ProviderError, make_provider,
)
from telugu_eval.runner import load_outputs, run_model, select_items

MCQ = {"id": "cul-001", "category": "cultural", "answer_type": "mcq",
       "question": "బతుకమ్మ ఏ రాష్ట్ర పండుగ?", "options": ["ఆంధ్రప్రదేశ్", "తెలంగాణ", "కర్ణాటక", "తమిళనాడు"],
       "gold_option": "B"}
SHORT = {"id": "cul-002", "category": "cultural", "answer_type": "short", "question": "ఆవకాయను ఏ కాయతో చేస్తారు?",
         "gold_answers": ["మామిడి"]}
RC = {"id": "rc-001", "category": "reading_comprehension", "answer_type": "short", "passage_id": "p-001",
      "question": "రైతు పేరు ఏమిటి?", "gold_answers": ["లక్ష్మి"]}


# --- prompts ---------------------------------------------------------------

def test_mcq_prompt_has_lettered_options_and_no_gold():
    p = build_prompt(MCQ)
    assert "A. ఆంధ్రప్రదేశ్\nB. తెలంగాణ\nC. కర్ణాటక\nD. తమిళనాడు" in p
    assert "one of A, B, C or D" in p and "Answer: <letter>" in p
    assert "gold" not in p.lower()


def test_short_and_rc_prompts():
    assert "Answer: <your answer>" in build_prompt(SHORT)
    p = build_prompt(RC, "లక్ష్మి అనే రైతు ఉండేది.")
    assert "Passage:\nలక్ష్మి అనే రైతు ఉండేది." in p
    assert "లక్ష్మి" not in build_prompt(SHORT)
    with pytest.raises(ValueError):
        build_prompt(RC, None)


def test_three_option_mcq_letters():
    p = build_prompt({**MCQ, "options": ["x", "y", "z"]})
    assert "one of A, B or C" in p and "D." not in p


def test_template_hash_is_stable():
    assert template_hash() == template_hash() and len(template_hash()) == 16


# --- config ----------------------------------------------------------------

def write_toml(tmp_path, text):
    p = tmp_path / "models.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_repo_models_toml_loads():
    models = load_models()
    names = [m.name for m in models]
    assert "gemini-3.8-flash" in names and "random-baseline" in names
    # zero-cost rule: nothing enabled may need a paid key
    for m in models:
        if m.enabled:
            assert m.api_key_env in ("", "GEMINI_API_KEY"), m.name


def test_config_defaults_and_validation(tmp_path):
    p = write_toml(tmp_path, '[defaults]\nmax_output_tokens = 64\n[[model]]\nname="a"\nprovider="dummy"\nmodel="random"\n')
    (m,) = load_models(p)
    assert m.max_output_tokens == 64 and m.display_name == "a" and m.temperature == 0.0
    bad = {
        "unknown keys": '[[model]]\nname="a"\nprovider="dummy"\nmodel="x"\ntemprature=0\n',
        "provider must": '[[model]]\nname="a"\nprovider="openai"\nmodel="x"\n',
        "needs base_url": '[[model]]\nname="a"\nprovider="openai_compatible"\nmodel="x"\n',
        "duplicate": '[[model]]\nname="a"\nprovider="dummy"\nmodel="x"\n[[model]]\nname="a"\nprovider="dummy"\nmodel="x"\n',
        "name may only": '[[model]]\nname="a b"\nprovider="dummy"\nmodel="x"\n',
    }
    for msg, text in bad.items():
        with pytest.raises(ConfigError, match=msg):
            load_models(write_toml(tmp_path, text))


def test_public_dict_has_no_secret(monkeypatch):
    monkeypatch.setenv("SECRET_KEY_X", "sk-123")
    m = ModelConfig(name="a", provider="gemini", model="g", api_key_env="SECRET_KEY_X", send_temperature=False)
    d = m.public_dict()
    assert "sk-123" not in json.dumps(d) and d["temperature_sent"] is None


# --- providers -------------------------------------------------------------

class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload))
        r = self.responses.pop(0)
        return r if isinstance(r, HTTPResult) else HTTPResult(200, json.dumps(r))


def gem_cfg(**kw):
    return ModelConfig(name="g", provider="gemini", model="gemini-x", api_key_env="TEST_GEM_KEY", **kw)


def test_gemini_request_and_parse(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    resp = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "Answer: B"}]},
                            "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 9}}
    t = FakeTransport([resp])
    cfg = gem_cfg(send_temperature=False, extra={"generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}}})
    g = GeminiProvider(cfg, t, sleep=lambda s: None).generate("prompt")
    assert g.text == "Answer: B" and g.finish_reason == "STOP" and g.usage["totalTokenCount"] == 9
    url, headers, body = t.calls[0]
    assert url.endswith("/models/gemini-x:generateContent") and "key=" not in url
    assert headers == {"x-goog-api-key": "k"}
    assert "temperature" not in body["generationConfig"]
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert body["generationConfig"]["maxOutputTokens"] == 1024  # deep merge keeps defaults


def test_gemini_sends_temperature_zero_when_supported(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    t = FakeTransport([{"candidates": [{"content": {"parts": [{"text": "x"}]}}]}])
    GeminiProvider(gem_cfg(), t).generate("p")
    assert t.calls[0][2]["generationConfig"]["temperature"] == 0.0


def test_gemini_blocked_prompt(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    t = FakeTransport([{"promptFeedback": {"blockReason": "SAFETY"}}])
    g = GeminiProvider(gem_cfg(), t).generate("p")
    assert g.text == "" and g.finish_reason == "BLOCKED:SAFETY"


def test_missing_key_is_fatal(monkeypatch):
    monkeypatch.delenv("TEST_GEM_KEY", raising=False)
    with pytest.raises(ProviderError) as e:
        GeminiProvider(gem_cfg(), FakeTransport([])).generate("p")
    assert e.value.fatal


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    sleeps = []
    t = FakeTransport([
        HTTPResult(429, '{"error": {"details": [{"retryDelay": "7s"}]}}'),
        HTTPResult(503, "busy", {"Retry-After": "2"}),
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
    ])
    g = GeminiProvider(gem_cfg(), t, sleep=sleeps.append).generate("p")
    assert g.text == "ok" and g.attempts == 3 and sleeps == [8.0, 3.0]  # suggested wait + 1 s margin


def test_retries_exhausted_and_fatal_status(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    t = FakeTransport([HTTPResult(500, "x")] * 3)
    with pytest.raises(ProviderError) as e:
        GeminiProvider(gem_cfg(max_retries=2), t, sleep=lambda s: None).generate("p")
    assert e.value.status == 500 and not e.value.fatal and len(t.calls) == 3
    with pytest.raises(ProviderError) as e:
        GeminiProvider(gem_cfg(), FakeTransport([HTTPResult(402, "no credits")]), sleep=lambda s: None).generate("p")
    assert e.value.fatal


# Shape of the real Gemini free-tier daily-limit error (Nov 2026).
DAILY_429 = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
    "You exceeded your current quota. * Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-x, "
    "description: Generate content requests per day limit exceeded on Free Tier.\nPlease retry in 27m47.53s.",
    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1667s"}]}})


def test_daily_quota_stops_without_retrying(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    sleeps = []
    t = FakeTransport([HTTPResult(429, DAILY_429)])
    with pytest.raises(ProviderError) as e:
        GeminiProvider(gem_cfg(), t, sleep=sleeps.append).generate("p")
    assert e.value.quota and e.value.fatal and not sleeps and len(t.calls) == 1
    assert "limit: 20" in str(e.value) and "{" not in str(e.value)  # readable message, not raw JSON


def test_long_retry_delay_counts_as_quota(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    body = '{"error": {"message": "Resource exhausted. Please retry in 1h2m3s."}}'
    with pytest.raises(ProviderError) as e:
        GeminiProvider(gem_cfg(), FakeTransport([HTTPResult(429, body)]), sleep=lambda s: None).generate("p")
    assert e.value.quota


def test_retries_are_logged(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    logs = []
    t = FakeTransport([HTTPResult(503, "busy", {"Retry-After": "1"}), {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}])
    GeminiProvider(gem_cfg(), t, sleep=lambda s: None, log=logs.append).generate("p")
    assert logs and "retry 1/5" in logs[0]


def test_quota_stop_in_runner_keeps_progress(tmp_path):
    make_data(tmp_path)
    runs = tmp_path / "runs"

    class QuotaAfterOne(CountingProvider):
        def generate(self, prompt):
            if self.calls:
                self.calls.append(prompt)
                raise ProviderError("quota", status=429, quota=True)
            return super().generate(prompt)

    s = run_model(CFG, select_items("all", tmp_path), runs_dir=runs, provider=QuotaAfterOne(), log=lambda *a: None)
    assert s.ok == 1 and s.errors == 0 and "quota" in s.stopped_early
    outs = load_outputs(runs / "fake" / "p1" / "outputs.jsonl")
    assert [o["status"] for o in outs] == ["ok"]  # the quota stop is not recorded as an item error
    s2 = run_model(CFG, select_items("all", tmp_path), runs_dir=runs, provider=CountingProvider(), log=lambda *a: None)
    assert s2.skipped == 1 and s2.ok == 2


def test_rate_limit_spacing(monkeypatch):
    monkeypatch.setenv("TEST_GEM_KEY", "k")
    now = [0.0]
    sleeps = []
    ok = {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}
    prov = GeminiProvider(gem_cfg(rpm=6), FakeTransport([ok, ok]), sleep=sleeps.append, clock=lambda: now[0])
    prov.generate("a")
    now[0] = 4.0
    prov.generate("b")
    assert sleeps == [pytest.approx(6.0)]  # 10 s spacing, 4 s already elapsed


def test_openai_compatible(monkeypatch):
    monkeypatch.setenv("TEST_OAI_KEY", "sk")
    cfg = ModelConfig(name="o", provider="openai_compatible", model="m", base_url="https://x/v1/",
                      api_key_env="TEST_OAI_KEY")
    t = FakeTransport([{"choices": [{"message": {"content": "Answer: C"}, "finish_reason": "stop"}], "usage": {"total_tokens": 5}}])
    g = OpenAICompatibleProvider(cfg, t).generate("hi")
    url, headers, body = t.calls[0]
    assert url == "https://x/v1/chat/completions" and headers["Authorization"] == "Bearer sk"
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["temperature"] == 0.0
    assert g.text == "Answer: C"


def test_openai_compatible_without_key_for_local_server():
    cfg = ModelConfig(name="o", provider="openai_compatible", model="m", base_url="http://localhost:11434/v1")
    t = FakeTransport([{"choices": [{"message": {"content": "x"}}]}])
    OpenAICompatibleProvider(cfg, t).generate("hi")
    assert t.calls[0][1] == {}


def test_random_baseline_picks_valid_letters():
    prov = make_provider(ModelConfig(name="r", provider="dummy", model="random", extra={"seed": 1}))
    letters = {prov.generate(build_prompt(MCQ)).text for _ in range(40)}
    assert letters == {"Answer: A", "Answer: B", "Answer: C", "Answer: D"}
    assert prov.generate(build_prompt(SHORT)).text == "Answer:"


# --- runner ----------------------------------------------------------------

def make_data(tmp_path, statuses=("reviewed", "draft", "retired")):
    items = []
    for n, st in enumerate(statuses, 1):
        items.append({**MCQ, "id": f"cul-{n:03d}", "question": f"ప్రశ్న {n}", "status": st, "canary": CANARY})
    items.append({**RC, "status": "reviewed", "canary": CANARY})
    dataio.write_jsonl(tmp_path / dataio.ITEMS_FILE, items)
    dataio.write_jsonl(tmp_path / dataio.PASSAGES_FILE, [{"passage_id": "p-001", "text": "లక్ష్మి అనే రైతు."}])
    return items


class CountingProvider:
    def __init__(self, fail_ids=(), fatal=False):
        self.calls = []
        self.fail_ids = set(fail_ids)
        self.fatal = fatal

    def generate(self, prompt):
        from telugu_eval.providers import Generation
        self.calls.append(prompt)
        if any(f"ప్రశ్న {i}" in prompt for i in self.fail_ids):
            raise ProviderError("boom", status=500, fatal=self.fatal)
        return Generation("Answer: B", "stop", {"t": 1}, {"raw": True}, {"model": "x"}, 0.01, 1)


CFG = ModelConfig(name="fake", provider="dummy", model="constant")


def test_select_items_splits(tmp_path):
    make_data(tmp_path)
    assert [i["id"] for i in select_items("reviewed", tmp_path).items] == ["cul-001", "rc-001"]
    assert [i["id"] for i in select_items("all", tmp_path).items] == ["cul-001", "cul-002", "rc-001"]


def test_run_saves_raw_outputs_and_resumes(tmp_path):
    make_data(tmp_path)
    runs = tmp_path / "runs"
    sel = select_items("all", tmp_path)
    p = CountingProvider()
    s = run_model(CFG, sel, runs_dir=runs, provider=p, log=lambda *a: None)
    assert (s.ok, s.errors, s.skipped) == (3, 0, 0)
    outs = load_outputs(runs / "fake" / "p1" / "outputs.jsonl")
    assert {o["item_id"] for o in outs} == {"cul-001", "cul-002", "rc-001"}
    o = outs[0]
    assert o["status"] == "ok" and o["response_text"] == "Answer: B" and o["raw"] == {"raw": True}
    assert o["prompt"].startswith("You are taking a test") and o["key"]
    meta = json.loads((runs / "fake" / "p1" / "run.json").read_text(encoding="utf-8"))
    assert meta["prompt_version"] == "p1" and meta["history"][-1]["ok"] == 3

    # second run: nothing to do
    p2 = CountingProvider()
    s2 = run_model(CFG, sel, runs_dir=runs, provider=p2, log=lambda *a: None)
    assert s2.skipped == 3 and not p2.calls

    # editing one item re-runs only that item
    items = dataio.read_jsonl(tmp_path / dataio.ITEMS_FILE).records
    items[0]["question"] = "ప్రశ్న 1 మార్చబడింది"
    dataio.write_jsonl(tmp_path / dataio.ITEMS_FILE, items)
    p3 = CountingProvider()
    s3 = run_model(CFG, select_items("all", tmp_path), runs_dir=runs, provider=p3, log=lambda *a: None)
    assert s3.ok == 1 and s3.skipped == 2

    # changing model settings re-runs everything
    p4 = CountingProvider()
    cfg2 = ModelConfig(name="fake", provider="dummy", model="constant", max_output_tokens=7)
    assert run_model(cfg2, select_items("all", tmp_path), runs_dir=runs, provider=p4, log=lambda *a: None).ok == 3


def test_errors_are_recorded_and_retried_next_run(tmp_path):
    make_data(tmp_path)
    runs = tmp_path / "runs"
    sel = select_items("all", tmp_path)
    s = run_model(CFG, sel, runs_dir=runs, provider=CountingProvider(fail_ids=[2]), log=lambda *a: None)
    assert (s.ok, s.errors) == (2, 1)
    errs = [o for o in load_outputs(runs / "fake" / "p1" / "outputs.jsonl") if o["status"] == "error"]
    assert errs[0]["item_id"] == "cul-002" and errs[0]["http_status"] == 500
    p = CountingProvider()
    assert run_model(CFG, sel, runs_dir=runs, provider=p, log=lambda *a: None).ok == 1


def test_fatal_error_stops_run(tmp_path):
    make_data(tmp_path)
    p = CountingProvider(fail_ids=[1], fatal=True)
    s = run_model(CFG, select_items("all", tmp_path), runs_dir=tmp_path / "runs", provider=p, log=lambda *a: None)
    assert s.stopped_early and len(p.calls) == 1


def test_missing_key_stops_before_writing(tmp_path, monkeypatch):
    make_data(tmp_path)
    monkeypatch.delenv("TEST_GEM_KEY", raising=False)
    s = run_model(gem_cfg(), select_items("all", tmp_path), runs_dir=tmp_path / "runs", log=lambda *a: None)
    assert s.stopped_early and "TEST_GEM_KEY" in s.stopped_early
    assert not (tmp_path / "runs").exists()


def test_limit(tmp_path):
    make_data(tmp_path)
    s = run_model(CFG, select_items("all", tmp_path), runs_dir=tmp_path / "runs", provider=CountingProvider(),
                  limit=1, log=lambda *a: None)
    assert s.ok == 1
