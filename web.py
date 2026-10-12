#!/usr/bin/env python3
"""Bare-bones web UI for the existing scheduling assistant.

Serves a static page and posts each chat turn to the same Orchestrator the
CLI uses. It does not decide who the patient is, which slot is real, or
whether a booking is allowed.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from orchestrator import Orchestrator
from tools import SchedulingAPIError, lookup_providers

ROOT = Path(__file__).resolve().parent / "static"
PORT = 8080

_bot = Orchestrator()
_lock = Lock()


def _when_parts(value):
    if not isinstance(value, str) or "T" not in value:
        return "", ""
    date, rest = value.split("T", 1)
    return date, rest[:5]


def _provider_name(provider_id):
    if not provider_id:
        return "Provider"
    try:
        payload = lookup_providers()
    except SchedulingAPIError:
        return "Provider"
    for provider in payload.get("providers") or []:
        if isinstance(provider, dict) and provider.get("providerId") == provider_id:
            return provider.get("name") or "Provider"
    return "Provider"


def _view(bot):
    """Describe the current turn for the page. Slot ids stay on the server."""
    confirmation = None
    slot = bot.selected_slot
    if isinstance(slot, dict) and slot in bot.slots:
        date, clock = _when_parts(slot.get("startTime"))
        confirmation = {
            "provider": _provider_name(slot.get("providerId")),
            "date": date or "Date from the scheduling system",
            "time": clock or "Time from the scheduling system",
            "location": slot.get("location") or "",
            "specialty": str(slot.get("specialty") or "").replace("_", " "),
        }

    choices = []
    if confirmation is None and bot.slots_shown:
        for number, item in enumerate(bot.slots, start=1):
            date, clock = _when_parts(item.get("startTime"))
            label = "{}. {} {}".format(number, date, clock).strip()
            choices.append({"label": label, "text": str(number)})
    return confirmation, choices


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            rel = "index.html"
        else:
            rel = path.lstrip("/")
        candidate = (ROOT / rel).resolve()
        if not str(candidate).startswith(str(ROOT.resolve())) or not candidate.is_file():
            self._json(404, {"error": "Not found."})
            return
        kind = "text/html; charset=utf-8"
        if candidate.suffix == ".css":
            kind = "text/css; charset=utf-8"
        elif candidate.suffix == ".js":
            kind = "text/javascript; charset=utf-8"
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/message":
            self._json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Send a text message."})
            return
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._json(400, {"error": "Type a message first."})
            return
        try:
            with _lock:
                reply = _bot.handle(text.strip())
                confirmation, choices = _view(_bot)
        except Exception:
            self._json(500, {"error": "The assistant hit a problem. Nothing was booked."})
            return
        self._json(200, {"reply": reply, "confirmation": confirmation, "choices": choices})

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("scheduling UI on http://localhost:{}".format(PORT))
    server.serve_forever()


if __name__ == "__main__":
    main()
