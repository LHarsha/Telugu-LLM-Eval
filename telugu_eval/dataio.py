"""Reading and writing the JSONL dataset files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ITEMS_FILE = "items.jsonl"
PASSAGES_FILE = "passages.jsonl"
EXAMPLES_FILE = "examples.jsonl"
EXAMPLE_PASSAGES_FILE = "examples_passages.jsonl"


@dataclass
class ParseError:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"[ERROR  ] {self.path}:{self.line}: {self.message}"


@dataclass
class Loaded:
    records: list[dict] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)  # source line of each record
    errors: list[ParseError] = field(default_factory=list)


def read_jsonl(path: Path) -> Loaded:
    out = Loaded()
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                out.errors.append(ParseError(path.name, n, f"invalid JSON: {e.msg} (col {e.colno})"))
                continue
            if not isinstance(rec, dict):
                out.errors.append(ParseError(path.name, n, "line is not a JSON object"))
                continue
            out.records.append(rec)
            out.lines.append(n)
    return out


def dumps(rec: dict) -> str:
    # ensure_ascii=False keeps Telugu readable in the file and in git diffs.
    return json.dumps(rec, ensure_ascii=False, separators=(", ", ": "))


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Atomically rewrite a JSONL file (temp file + rename), UTF-8, LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(dumps(r) + "\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
