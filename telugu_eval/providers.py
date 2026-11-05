"""One interface for every model: ``Provider.generate(prompt) -> Generation``.

Uses only urllib (no SDKs), so adding a provider never adds a dependency.
The HTTP call is injectable (``transport``) so tests run fully offline.
"""

from __future__ import annotations

import copy
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .config import ModelConfig

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# If the provider asks us to wait longer than this, it is a quota wall (e.g. a
# free-tier daily limit), not a hiccup: stop the run and resume later.
MAX_RETRY_WAIT_S = 300.0


@dataclass
class HTTPResult:
    status: int
    body: str
    headers: dict = field(default_factory=dict)


Transport = Callable[[str, dict, dict, float], HTTPResult]


def urllib_transport(url: str, headers: dict, payload: dict, timeout: float) -> HTTPResult:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HTTPResult(resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers))
    except urllib.error.HTTPError as e:
        return HTTPResult(e.code, e.read().decode("utf-8", "replace"), dict(e.headers or {}))


class ProviderError(RuntimeError):
    """A request failed permanently (after retries) or was refused."""

    def __init__(self, message: str, *, status: int | None = None, fatal: bool = False, body: str = "",
                 quota: bool = False):
        super().__init__(message)
        self.status = status
        self.fatal = fatal or quota  # True = stop the whole run (bad key, no credits, bad model id, quota)
        self.quota = quota           # True = daily/long quota reached; resume later
        self.body = body[:2000]


@dataclass
class Generation:
    text: str                       # the model's visible answer text
    finish_reason: str | None
    usage: dict
    raw: dict                       # full provider response (for transparency)
    request: dict                   # request body actually sent (no secrets)
    latency_s: float
    attempts: int


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Provider:
    def __init__(self, cfg: ModelConfig, transport: Transport | None = None, sleep=time.sleep, clock=time.monotonic,
                 log=None):
        self.cfg = cfg
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.clock = clock
        self.log = log or (lambda *a: None)
        self._last_call: float | None = None

    # subclasses implement these three
    def endpoint(self) -> str: ...
    def headers(self) -> dict: ...
    def body(self, prompt: str) -> dict: ...
    def parse(self, data: dict) -> tuple[str, str | None, dict]: ...

    def _throttle(self) -> None:
        if self.cfg.rpm and self._last_call is not None:
            gap = 60.0 / self.cfg.rpm - (self.clock() - self._last_call)
            if gap > 0:
                self.sleep(gap)
        self._last_call = self.clock()

    def generate(self, prompt: str) -> Generation:
        body = self.body(prompt)
        url, headers = self.endpoint(), self.headers()
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            t0 = time.monotonic()
            try:
                res = self.transport(url, headers, body, self.cfg.timeout_s)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                res = HTTPResult(0, f"network error: {e}")
            latency = time.monotonic() - t0
            if res.status == 200:
                try:
                    data = json.loads(res.body)
                except json.JSONDecodeError as e:
                    raise ProviderError(f"invalid JSON from provider: {e}", status=200, body=res.body) from e
                text, finish, usage = self.parse(data)
                return Generation(text, finish, usage, data, body, latency, attempt)
            retryable = res.status in RETRY_STATUS or res.status == 0
            msg = _error_message(res.body)
            if res.status == 429:
                wait = self._suggested_wait(res)
                if _is_daily_quota(res.body) or (wait is not None and wait > MAX_RETRY_WAIT_S):
                    raise ProviderError(f"free-tier quota reached for {self.cfg.model}: {msg}",
                                        status=429, quota=True, body=res.body)
            if not retryable or attempt > self.cfg.max_retries:
                fatal = res.status in (400, 401, 402, 403, 404)
                raise ProviderError(f"HTTP {res.status} after {attempt} attempt(s): {msg}",
                                    status=res.status, fatal=fatal, body=res.body)
            delay = self._backoff(attempt, res)
            self.log(f"    HTTP {res.status or 'network error'}; retry {attempt}/{self.cfg.max_retries} in {delay:.0f}s")
            self.sleep(delay)

    @staticmethod
    def _suggested_wait(res: HTTPResult) -> float | None:
        ra = {k.lower(): v for k, v in res.headers.items()}.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
        # Gemini: google.rpc.RetryInfo {"retryDelay": "17s"} in the error body
        m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', res.body)
        if m:
            return float(m.group(1))
        # ... or only in the message text: "Please retry in 27m47.5s."
        m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", res.body)
        if m:
            h, mi, s = m.groups()
            return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)
        return None

    @classmethod
    def _backoff(cls, attempt: int, res: HTTPResult) -> float:
        wait = cls._suggested_wait(res)
        if wait is not None:
            return min(wait + 1.0, MAX_RETRY_WAIT_S)
        return min(2.0 ** attempt + random.random(), 60.0)


def _error_message(body: str) -> str:
    """Short human-readable error from a JSON error body (falls back to raw text)."""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        msg = err.get("message") if isinstance(err, dict) else None
        if msg:
            return " ".join(str(msg).split())[:400]
    except (json.JSONDecodeError, AttributeError):
        pass
    return " ".join(body.split())[:300]


def _is_daily_quota(body: str) -> bool:
    b = body.lower()
    return "per day" in b or "perday" in b or "requests_per_day" in b


class GeminiProvider(Provider):
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def endpoint(self) -> str:
        return f"{self.BASE}/models/{self.cfg.model}:generateContent"

    def headers(self) -> dict:
        key = self.cfg.api_key()
        if not key:
            raise ProviderError(f"{self.cfg.name}: set {self.cfg.api_key_env} (env or .env)", fatal=True)
        return {"x-goog-api-key": key}  # header, never in the URL

    def body(self, prompt: str) -> dict:
        gen = {"maxOutputTokens": self.cfg.max_output_tokens}
        if self.cfg.send_temperature and self.cfg.temperature is not None:
            gen["temperature"] = self.cfg.temperature
        base = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
        return _deep_merge(base, self.cfg.extra)

    def parse(self, data: dict) -> tuple[str, str | None, dict]:
        usage = data.get("usageMetadata", {}) or {}
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "NO_CANDIDATES")
            return "", f"BLOCKED:{reason}", usage
        cand = cands[0]
        parts = (cand.get("content") or {}).get("parts") or []
        # skip thought-summary parts; keep only the visible answer
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return text, cand.get("finishReason"), usage


class OpenAICompatibleProvider(Provider):
    def endpoint(self) -> str:
        return self.cfg.base_url.rstrip("/") + "/chat/completions"

    def headers(self) -> dict:
        if not self.cfg.api_key_env:
            return {}
        key = self.cfg.api_key()
        if not key:
            raise ProviderError(f"{self.cfg.name}: set {self.cfg.api_key_env} (env or .env)", fatal=True)
        return {"Authorization": f"Bearer {key}"}

    def body(self, prompt: str) -> dict:
        b = {"model": self.cfg.model, "messages": [{"role": "user", "content": prompt}],
             "max_tokens": self.cfg.max_output_tokens}
        if self.cfg.send_temperature and self.cfg.temperature is not None:
            b["temperature"] = self.cfg.temperature
        return _deep_merge(b, self.cfg.extra)

    def parse(self, data: dict) -> tuple[str, str | None, dict]:
        choices = data.get("choices") or []
        if not choices:
            return "", "NO_CHOICES", data.get("usage") or {}
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content, choices[0].get("finish_reason"), data.get("usage") or {}


class DummyProvider(Provider):
    """Offline model. model="random": random MCQ letter; "constant": always extra['answer']."""

    def __init__(self, cfg: ModelConfig, *a, **kw):
        super().__init__(cfg, *a, **kw)
        self.rng = random.Random(cfg.extra.get("seed", 0))

    def endpoint(self) -> str:
        return "dummy://"

    def headers(self) -> dict:
        return {}

    def body(self, prompt: str) -> dict:
        return {"model": self.cfg.model, "prompt_chars": len(prompt)}

    def generate(self, prompt: str) -> Generation:
        if self.cfg.model == "constant":
            text = f"Answer: {self.cfg.extra.get('answer', 'A')}"
        elif self.cfg.model == "random":
            m = re.search(r"where <letter> is one of ([A-F, or]+)\.", prompt)
            letters = re.findall(r"[A-F]", m.group(1)) if m else []
            text = f"Answer: {self.rng.choice(letters)}" if letters else "Answer:"
        else:
            raise ProviderError(f"unknown dummy model {self.cfg.model!r}", fatal=True)
        return Generation(text, "stop", {}, {"text": text}, self.body(prompt), 0.0, 1)


def make_provider(cfg: ModelConfig, transport: Transport | None = None, **kw) -> Provider:
    """kw: sleep, clock, log."""
    cls = {"gemini": GeminiProvider, "openai_compatible": OpenAICompatibleProvider, "dummy": DummyProvider}[cfg.provider]
    return cls(cfg, transport, **kw)
