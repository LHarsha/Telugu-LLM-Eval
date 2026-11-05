"""Model registry: ``models.toml`` at the repo root.

Adding a model = adding one ``[[model]]`` table. Supported providers:

* ``gemini``             Google Gemini API (also serves open-weights Gemma models)
* ``openai_compatible``  any OpenAI-style /chat/completions endpoint: OpenRouter,
                         Hugging Face router, Groq, a local Ollama server, ...
* ``dummy``              offline fake model for tests and dry runs

API keys are read from environment variables (or a git-ignored ``.env``
file); they are never written to config, outputs or logs.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .dataio import REPO_ROOT

MODELS_FILE = REPO_ROOT / "models.toml"
PROVIDERS = ("gemini", "openai_compatible", "dummy")
KINDS = ("api", "open-weights", "baseline")


class ConfigError(ValueError):
    pass


@dataclass
class ModelConfig:
    name: str                       # stable id used in results and on the leaderboard
    provider: str
    model: str                      # provider's model string
    display_name: str = ""
    kind: str = "api"               # api | open-weights | baseline
    enabled: bool = True
    api_key_env: str = ""           # name of the env var holding the key ("" = no key)
    base_url: str = ""              # openai_compatible only
    temperature: float | None = 0.0
    send_temperature: bool = True   # False for models that deprecate/ignore it
    max_output_tokens: int = 1024
    rpm: float = 0.0                # client-side requests-per-minute cap (0 = none)
    timeout_s: float = 120.0
    max_retries: int = 5
    extra: dict = field(default_factory=dict)  # merged into the provider request body
    notes: str = ""

    def public_dict(self) -> dict:
        """Settings safe to store with results (no secrets)."""
        d = {k: getattr(self, k) for k in (
            "name", "provider", "model", "display_name", "kind", "base_url",
            "temperature", "send_temperature", "max_output_tokens", "extra", "notes")}
        d["temperature_sent"] = self.temperature if self.send_temperature else None
        return d

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_ALLOWED = set(ModelConfig.__dataclass_fields__)


def load_models(path: Path = MODELS_FILE) -> list[ModelConfig]:
    if not path.exists():
        raise ConfigError(f"{path} not found")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    defaults = data.get("defaults", {})
    out: list[ModelConfig] = []
    seen = set()
    for i, raw in enumerate(data.get("model", [])):
        merged = {**defaults, **raw}
        unknown = set(merged) - _ALLOWED
        if unknown:
            raise ConfigError(f"model #{i + 1}: unknown keys {sorted(unknown)}")
        for req in ("name", "provider", "model"):
            if not merged.get(req):
                raise ConfigError(f"model #{i + 1}: missing '{req}'")
        mc = ModelConfig(**merged)
        if mc.provider not in PROVIDERS:
            raise ConfigError(f"{mc.name}: provider must be one of {PROVIDERS}")
        if mc.kind not in KINDS:
            raise ConfigError(f"{mc.name}: kind must be one of {KINDS}")
        if mc.provider == "openai_compatible" and not mc.base_url:
            raise ConfigError(f"{mc.name}: openai_compatible needs base_url")
        if mc.name in seen:
            raise ConfigError(f"duplicate model name {mc.name!r}")
        if not all(c.isalnum() or c in "-_." for c in mc.name):
            raise ConfigError(f"{mc.name!r}: name may only use letters, digits, - _ .")
        seen.add(mc.name)
        mc.display_name = mc.display_name or mc.name
        out.append(mc)
    return out


def default_judge(path: Path = MODELS_FILE) -> str | None:
    """Name of the model used as LLM judge ([judge] model = "..."), if any."""
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return (tomllib.load(fh).get("judge") or {}).get("model")


def get_model(name: str, path: Path = MODELS_FILE) -> ModelConfig:
    for m in load_models(path):
        if m.name == name:
            return m
    raise ConfigError(f"no model named {name!r} in {path.name}")
