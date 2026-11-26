"""Public HTTP API for the agent: POST /ask {"question": "..."} -> {"answer", "tools", "grounded"}.

Follow-ups: {"question": "...", "history": [{"role": "user", ...}, {"role": "assistant", ...}, ...]}.

Run locally:   RALLY_DB=serve/vb_public.duckdb .venv/bin/uvicorn rallyagent.api:app --port 8080
Container:     see serve/Dockerfile

The browser sends only a question. The system prompt, tools, model (Claude Haiku 5.5) and limits are
fixed here, so the API key can't be used for anything but this agent. Guards, all set by env vars:

  MAX_QUESTION_CHARS  500    longer questions are rejected
  MAX_HISTORY         6      earlier messages a follow-up may send (user/assistant, alternating)
  MAX_HISTORY_CHARS   6000   total characters of that history
  RATE_PER_HOUR       10     questions per client IP per rolling hour
  DAILY_BUDGET_USD    1.00   the server stops answering once today's (UTC) spend reaches this
  MAX_CONCURRENT      4      agent runs at once; more wait, then get 503
  ASK_TIMEOUT_S       60     wall-clock limit per question
  ALLOWED_ORIGINS     https://m3tang.github.io   comma-separated, for CORS
  TURNSTILE_SECRET    (unset)  when set, requests must carry a valid Cloudflare Turnstile token
  TRUST_PROXY         1      take the client IP from X-Forwarded-For (Cloud Run sets it)
  PROXY_HOPS          1      how many X-Forwarded-For entries, from the right, proxies added

Counters live in memory, so run a single instance (Cloud Run --max-instances 1). A restart resets
them; the API key's own monthly spend limit in the Claude Console is the backstop. Each answered
question is logged as one JSON line (question, answer, cost, tools; the IP is hashed, not stored).
"""
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

import anthropic
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .agent import HAIKU, Agent, Config
from .pricing import cost_usd

log = logging.getLogger("rallyagent.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


MAX_QUESTION_CHARS = _env("MAX_QUESTION_CHARS", 500, int)
MAX_HISTORY = _env("MAX_HISTORY", 6, int)            # earlier messages a follow-up may carry
MAX_HISTORY_CHARS = _env("MAX_HISTORY_CHARS", 6000, int)
RATE_PER_HOUR = _env("RATE_PER_HOUR", 10, int)
DAILY_BUDGET_USD = _env("DAILY_BUDGET_USD", 1.0, float)
MAX_CONCURRENT = _env("MAX_CONCURRENT", 4, int)
ASK_TIMEOUT_S = _env("ASK_TIMEOUT_S", 60.0, float)
ALLOWED_ORIGINS = [o.strip() for o in _env("ALLOWED_ORIGINS", "https://m3tang.github.io").split(",") if o.strip()]
TURNSTILE_SECRET = _env("TURNSTILE_SECRET", "")
TRUST_PROXY = _env("TRUST_PROXY", "1") == "1"
PROXY_HOPS = _env("PROXY_HOPS", 1, int)

# The public agent: Haiku researches and writes, with tighter turn and token limits than the CLI.
PUBLIC = Config(HAIKU, None, max_turns=6, max_tokens=8000)
NOT_RELIABLE = ("Sorry, I couldn't answer that one reliably from the model's data. Try asking about match odds, "
                "a team's season, the NCAA tournament, or box-score stats from 2020 to 2025.")


class Limits:
    """Per-IP rolling-hour rate limit and a daily spend budget, in memory."""

    def __init__(self, per_hour, budget):
        self.per_hour, self.budget = per_hour, budget
        self.hits = defaultdict(deque)
        self.day, self.spent = None, 0.0

    def _today(self):
        d = dt.datetime.now(dt.timezone.utc).date()
        if d != self.day:
            self.day, self.spent = d, 0.0
        return d

    def allow(self, ip, now=None):
        now = time.time() if now is None else now
        q = self.hits[ip]
        while q and q[0] <= now - 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            return False, int(q[0] + 3600 - now) + 1
        q.append(now)
        return True, 0

    def budget_left(self):
        self._today()
        return round(max(self.budget - self.spent, 0.0), 4)

    def charge(self, usd):
        self._today()
        self.spent += usd


limits = Limits(RATE_PER_HOUR, DAILY_BUDGET_USD)
_slots = asyncio.Semaphore(MAX_CONCURRENT)
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = Agent(PUBLIC)
    return _agent


def client_ip(request: Request):
    """The visitor's IP. Proxies append to X-Forwarded-For, so earlier entries can be anything a client sent;
    only the last PROXY_HOPS entries were added by infrastructure. Cloud Run's front end adds one."""
    if TRUST_PROXY and (fwd := request.headers.get("x-forwarded-for")):
        hops = [h.strip() for h in fwd.split(",") if h.strip()]
        if hops:
            return hops[-min(PROXY_HOPS, len(hops))]
    return request.client.host if request.client else "unknown"


def _turnstile_ok(token, ip):
    if not TURNSTILE_SECRET:
        return True
    if not token:
        return False
    body = urllib.parse.urlencode({"secret": TURNSTILE_SECRET, "response": token, "remoteip": ip}).encode()
    try:
        with urllib.request.urlopen("https://challenges.cloudflare.com/turnstile/v0/siteverify", body, timeout=5) as r:
            return bool(json.load(r).get("success"))
    except OSError:
        return False


def parse_history(raw):
    """Earlier turns sent by the page, checked: a list of {"role", "content"} that alternates user/assistant,
    starts with user and ends with assistant (so the new question continues it), within the caps. The history
    is the client's word, so it is context only: the agent must fetch every number again for each answer."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_HISTORY:
        raise ValueError(f"history must be a list of at most {MAX_HISTORY} messages")
    out, total = [], 0
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError("each history message needs exactly role and content")
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want or not isinstance(m["content"], str) or not m["content"].strip():
            raise ValueError("history must alternate user and assistant messages, starting with user")
        total += len(m["content"])
        out.append({"role": want, "content": m["content"].strip()})
    if out and out[-1]["role"] != "assistant":
        raise ValueError("history must end with an assistant message")
    if total > MAX_HISTORY_CHARS:
        raise ValueError(f"history is too long; keep it under {MAX_HISTORY_CHARS} characters")
    return out


def _err(status, message, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status)


async def ask(request: Request):
    ip = client_ip(request)
    if int(request.headers.get("content-length") or 0) > 16384:
        return _err(413, "request too large")
    try:
        body = await request.json()
        question = str(body.get("question", "")).strip()
    except (ValueError, AttributeError):
        return _err(400, 'send JSON like {"question": "..."}')
    if not question:
        return _err(400, "question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        return _err(400, f"keep questions under {MAX_QUESTION_CHARS} characters")
    try:
        history = parse_history(body.get("history"))
    except ValueError as e:
        return _err(400, str(e))
    if not await asyncio.to_thread(_turnstile_ok, body.get("turnstile_token"), ip):
        return _err(403, "bot check failed; reload the page and try again")
    if limits.budget_left() <= 0:
        return _err(429, "the agent has hit its budget for today; try again tomorrow", retry="tomorrow")
    ok, wait = limits.allow(ip)
    if not ok:
        return _err(429, f"too many questions; try again in {wait // 60 + 1} minutes", retry_after_s=wait)

    t0 = time.monotonic()
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=10)
    except asyncio.TimeoutError:
        return _err(503, "busy; try again in a moment")
    try:
        r = await asyncio.wait_for(asyncio.to_thread(get_agent().ask, question, history), timeout=ASK_TIMEOUT_S)
    except asyncio.TimeoutError:
        # The worker thread may still finish and bill; its cost is not counted, which the Console limit covers.
        return _err(504, "that took too long; try a simpler question")
    except anthropic.RateLimitError:
        return _err(503, "busy; try again in a moment")
    except anthropic.APIError as e:
        log.warning(json.dumps({"event": "api_error", "error": f"{type(e).__name__}: {e}"[:300]}))
        return _err(502, "the model service had a problem; try again")
    finally:
        _slots.release()

    cost = cost_usd(r["usage_by_model"])
    limits.charge(cost)
    grounded = r["grounding"]["ok"] if "grounding" in r else False
    answer = r["answer"] if grounded and r["answer"] else NOT_RELIABLE
    tools = [{"name": c["name"], "args": c["args"]} for c in r["tool_calls"] if c["error"] is None]
    log.info(json.dumps({"event": "ask", "ip": hashlib.sha256(ip.encode()).hexdigest()[:12], "question": question,
                         "history_turns": len(history),
                         "answer": r["answer"], "shown": answer == r["answer"], "grounded": grounded,
                         "ungrounded": r.get("grounding", {}).get("ungrounded", []), "status": r["status"],
                         "tools": tools, "cost_usd": cost, "latency_s": round(time.monotonic() - t0, 2)}))
    return JSONResponse({"answer": answer, "grounded": grounded, "tools": tools})


async def health(request: Request):
    return JSONResponse({"ok": True, "budget_left_usd": limits.budget_left()})


app = Starlette(
    routes=[Route("/ask", ask, methods=["POST"]), Route("/health", health, methods=["GET"])],
    middleware=[Middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["POST", "GET"],
                           allow_headers=["content-type"], max_age=3600)],
)
