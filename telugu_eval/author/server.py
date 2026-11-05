"""Local authoring server.

Runs on 127.0.0.1 only. Serves one HTML page and a tiny JSON API that reads
and writes ``data/items.jsonl`` / ``data/passages.jsonl``. Every save is
validated with the same checks as ``python -m telugu_eval validate``; an item
with errors is refused, warnings are shown but allowed.
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import CANARY, dataio
from ..normalize import clean_text, detect_script
from ..schema import (
    ALL_ITEM_FIELDS, ALL_PASSAGE_FIELDS, ANSWER_TYPES, CATEGORIES, CATEGORY_PREFIX,
    DIFFICULTIES, ITEM_ID_RE, PASSAGE_ID_RE, REGIONS, SCRIPTS, STATUSES,
    Report, check_item, check_passage,
)
from ..validate import cross_checks

STATIC = Path(__file__).resolve().parent / "static"
_lock = threading.Lock()


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.items_path = self.data_dir / dataio.ITEMS_FILE
        self.passages_path = self.data_dir / dataio.PASSAGES_FILE

    # -- reading -----------------------------------------------------------
    def items(self) -> list[dict]:
        return dataio.read_jsonl(self.items_path).records

    def passages(self) -> list[dict]:
        return dataio.read_jsonl(self.passages_path).records

    def next_item_id(self, category: str, items: list[dict]) -> str:
        prefix = CATEGORY_PREFIX[category]
        nums = [int(m.group(2)) for it in items
                if (m := ITEM_ID_RE.match(str(it.get("id", "")))) and m.group(1) == prefix]
        return f"{prefix}-{(max(nums) + 1) if nums else 1:03d}"

    def next_passage_id(self, passages: list[dict]) -> str:
        nums = [int(m.group(1)) for p in passages if (m := PASSAGE_ID_RE.match(str(p.get("passage_id", ""))))]
        return f"p-{(max(nums) + 1) if nums else 1:03d}"

    def state(self) -> dict:
        items, passages = self.items(), self.passages()
        subtopics: dict[str, list[str]] = {c: [] for c in CATEGORIES}
        for it in items:
            c, s = it.get("category"), it.get("subtopic")
            if c in subtopics and s and s not in subtopics[c]:
                subtopics[c].append(s)
        return {
            "items": items,
            "passages": passages,
            "subtopics": subtopics,
            "next_ids": {c: self.next_item_id(c, items) for c in CATEGORIES},
            "next_passage_id": self.next_passage_id(passages),
            "enums": {
                "category": CATEGORIES, "script": SCRIPTS, "region": REGIONS,
                "answer_type": ANSWER_TYPES, "difficulty": DIFFICULTIES, "status": STATUSES,
            },
            "today": date.today().isoformat(),
        }

    # -- preparing records -------------------------------------------------
    @staticmethod
    def prepare_item(raw: dict) -> dict:
        """Clean strings, drop fields that do not apply, fill defaults."""
        it = {k: raw.get(k) for k in ALL_ITEM_FIELDS if k in raw}
        for k in ("question", "subtopic", "source", "author_note", "author", "passage_id", "gold_option"):
            if isinstance(it.get(k), str):
                it[k] = clean_text(it[k])
        if isinstance(it.get("subtopic"), str):
            it["subtopic"] = it["subtopic"].lower()
        if it.get("answer_type") == "mcq":
            it["options"] = [clean_text(o) for o in (it.get("options") or [])]
            it.pop("gold_answers", None)
        elif it.get("answer_type") == "short":
            golds = [clean_text(g) for g in (it.get("gold_answers") or [])]
            it["gold_answers"] = [g for g in golds if g]
            it.pop("options", None)
            it.pop("gold_option", None)
        if it.get("category") != "reading_comprehension":
            it.pop("passage_id", None)
        for k in ("source", "author_note", "passage_id"):
            if it.get(k) in ("", None):
                it.pop(k, None)
        it.setdefault("status", "draft")
        it.setdefault("created", date.today().isoformat())
        it["canary"] = CANARY
        # stable key order = readable diffs
        return {k: it[k] for k in ALL_ITEM_FIELDS if k in it}

    @staticmethod
    def prepare_passage(raw: dict) -> dict:
        p = {k: raw.get(k) for k in ALL_PASSAGE_FIELDS if k in raw}
        for k in ("title", "text", "source", "author_note", "author"):
            if isinstance(p.get(k), str):
                p[k] = clean_text(p[k])
        if not p.get("script") and p.get("text"):
            p["script"] = detect_script(p["text"])
        for k in ("source", "author_note"):
            if p.get(k) in ("", None):
                p.pop(k, None)
        p.setdefault("created", date.today().isoformat())
        p["canary"] = CANARY
        return {k: p[k] for k in ALL_PASSAGE_FIELDS if k in p}

    # -- checking ----------------------------------------------------------
    def check_item_in_context(self, item: dict, original_id: str | None) -> Report:
        items = [it for it in self.items() if it.get("id") not in {original_id, item.get("id")}]
        passages = self.passages()
        rep = check_item(item, passage_ids={p.get("passage_id") for p in passages})
        ctx = Report()
        cross_checks(items + [item], [], ctx)
        rid = str(item.get("id"))
        for i in ctx.issues:
            if i.record_id == rid or i.record_id == "dataset":
                rep.issues.append(i)
        return rep

    # -- writing -----------------------------------------------------------
    def save_item(self, raw: dict, original_id: str | None) -> tuple[dict, Report]:
        with _lock:
            items = self.items()
            item = self.prepare_item(raw)
            if not item.get("id") and item.get("category") in CATEGORIES:
                item["id"] = self.next_item_id(item["category"], items)
                item = {k: item[k] for k in ALL_ITEM_FIELDS if k in item}  # id first
            rep = self.check_item_in_context(item, original_id)
            if rep.errors:
                return item, rep
            if original_id:
                if item.get("id") != original_id:
                    rep.error(original_id, "id", "ids cannot be changed when editing")
                    return item, rep
                idx = next((i for i, it in enumerate(items) if it.get("id") == original_id), None)
                if idx is None:
                    rep.error(original_id, "id", "item to update not found")
                    return item, rep
                items[idx] = item
            else:
                if any(it.get("id") == item["id"] for it in items):
                    rep.error(item["id"], "id", "id already exists")
                    return item, rep
                items.append(item)
            dataio.write_jsonl(self.items_path, items)
            return item, rep

    def save_passage(self, raw: dict, original_id: str | None) -> tuple[dict, Report]:
        with _lock:
            passages = self.passages()
            p = self.prepare_passage(raw)
            if not p.get("passage_id"):
                p["passage_id"] = self.next_passage_id(passages)
                p = {k: p[k] for k in ALL_PASSAGE_FIELDS if k in p}  # id first
            if original_id and p["passage_id"] != original_id:
                rep = Report()
                rep.error(original_id, "passage_id", "ids cannot be changed when editing")
                return p, rep
            rep = check_passage(p)
            others = [x for x in passages if x.get("passage_id") != original_id]
            if any(x.get("passage_id") == p["passage_id"] for x in others):
                rep.error(p["passage_id"], "passage_id", "id already exists")
            if rep.errors:
                return p, rep
            if original_id:
                idx = next((i for i, x in enumerate(passages) if x.get("passage_id") == original_id), None)
                if idx is None:
                    rep.error(original_id, "passage_id", "passage to update not found")
                    return p, rep
                passages[idx] = p
            else:
                passages.append(p)
            dataio.write_jsonl(self.passages_path, passages)
            return p, rep


def _issues(rep: Report) -> list[dict]:
    return [{"level": i.level, "id": i.record_id, "field": i.field, "message": i.message} for i in rep.issues]


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TeluguEvalAuthor/0.1"

        def log_message(self, fmt, *args):  # quieter console
            if not self.path.startswith("/api/state"):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def _same_origin(self) -> bool:
            # Block cross-site POSTs from other pages open in the browser.
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return re.fullmatch(r"http://(127\.0\.0\.1|localhost):\d+", origin) is not None

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, store.state())
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._same_origin():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            path = self.path.split("?", 1)[0]
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            original = body.get("original_id") or None
            if path == "/api/check":
                item = store.prepare_item(body.get("item") or {})
                if not item.get("id") and item.get("category") in CATEGORIES:
                    item["id"] = store.next_item_id(item["category"], store.items())
                rep = store.check_item_in_context(item, original)
                return self._json(200, {"issues": _issues(rep)})
            if path == "/api/items":
                item, rep = store.save_item(body.get("item") or {}, original)
                code = 400 if rep.errors else 200
                return self._json(code, {"saved": not rep.errors, "item": item, "issues": _issues(rep)})
            if path == "/api/passages":
                p, rep = store.save_passage(body.get("passage") or {}, original)
                code = 400 if rep.errors else 200
                return self._json(code, {"saved": not rep.errors, "passage": p, "issues": _issues(rep)})
            self._json(404, {"error": "unknown endpoint"})

    return Handler


def serve(data_dir: Path, *, port: int = 8765, open_browser: bool = True) -> None:
    store = Store(data_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store))
    url = f"http://127.0.0.1:{port}/"
    print(f"Authoring form running at {url}  (data: {store.data_dir})")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
