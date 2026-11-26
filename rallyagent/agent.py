"""The "Ask the model" agent.

    python -m rallyagent "Could Nebraska have beaten 2023 Texas?" [--config split|opus|haiku] [--verbose]

Flow for one question:
  1. research   a tool-use loop: the research model picks tools and arguments, tools run locally
                (cached, deterministic), results go back, until it stops calling tools
  2. write      (split config only) a stronger model writes the answer from the question, every tool
                result and the researcher's notes, with no tools
  3. ground     rallytools.grounding checks every number in the answer against the tool results; on a
                miss the writer gets one rewrite with the offending numbers named, and if numbers are still
                ungrounded the result is flagged (grounding.ok = False)

Configs (see CONFIGS): "split" uses Claude Haiku 5.5 to research and Claude Opus 5.5 to write; "opus" and
"haiku" use one model for both. Which is best is an eval question: python -m evals.run.

Requests to Claude Opus 5.5 opt into server-side refusal fallbacks (fallbacks="default"), so a safety
classifier false positive is retried on another model instead of failing the question.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import anthropic

from rallytools import grounding as G

from . import prompts as P
from .tools import TOOLS, result_text, run_tool

OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {OPUS}  # Claude Haiku 5.5 has no server-side fallback


@dataclass
class Config:
    research_model: str
    writer_model: str | None          # None: the research model writes the answer itself
    research_effort: str = "medium"
    writer_effort: str = "medium"
    max_turns: int = 8                # research model calls
    regenerate: bool = True           # one rewrite when the answer has ungrounded numbers
    max_tokens: int = 16000


CONFIGS = {
    "split": Config(HAIKU, OPUS),
    "opus": Config(OPUS, None),
    "haiku": Config(HAIKU, None),
}


class ServedModelMismatch(RuntimeError):
    """The API served a different model than requested, without a fallback explaining it."""


@dataclass
class Usage:
    by_model: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def add(self, resp, role, latency):
        u = resp.usage
        row = {"input_tokens": u.input_tokens or 0, "output_tokens": u.output_tokens or 0,
               "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
               "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        agg = self.by_model.setdefault(resp.model, dict.fromkeys(row, 0))
        for k, v in row.items():
            agg[k] += v
        self.calls.append({"role": role, "model": resp.model, "stop_reason": resp.stop_reason,
                           "latency_s": round(latency, 3), **row})

    def total(self):
        out = {}
        for agg in self.by_model.values():
            for k, v in agg.items():
                out[k] = out.get(k, 0) + v
        return out


KEYCHAIN_SERVICE = "vb-sim-agent-api-key"


def _client_kwargs():
    """Credentials for the client. The environment wins (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, as in
    CI); otherwise, on a Mac, the key stored in the login Keychain under KEYCHAIN_SERVICE:
        security add-generic-password -a "$USER" -s vb-sim-agent-api-key -w
    If neither is there, the SDK falls back to its own resolution (e.g. an `ant auth login` profile)."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    key = out.stdout.strip()
    return {"api_key": key} if out.returncode == 0 and key else {}


class Agent:
    def __init__(self, config="haiku", client=None):
        self.cfg = CONFIGS[config] if isinstance(config, str) else config
        self.client = client or anthropic.Anthropic(**_client_kwargs())

    # ---- one model call ----

    def _create(self, model, effort, usage, role, cache=True, **kw):
        # Cache the research loop (tools + system are shared by every question, and each turn extends the
        # last). The writer's prompt is unique per question, so caching it would only pay the write premium.
        params = dict(model=model, max_tokens=self.cfg.max_tokens, output_config={"effort": effort}, **kw)
        if cache:
            params["cache_control"] = {"type": "ephemeral"}
        if model in FALLBACK_MODELS:
            params.update(betas=[FALLBACK_BETA], fallbacks="default")
        t0 = time.monotonic()
        resp = self.client.beta.messages.create(**params)
        usage.add(resp, role, time.monotonic() - t0)
        fell_back = any(b.type == "fallback" for b in resp.content) or any(
            getattr(it, "type", "") == "fallback_message" for it in (getattr(resp.usage, "iterations", None) or []))
        if resp.model != model and not fell_back and not resp.model.startswith(model):
            raise ServedModelMismatch(f"requested {model}, served {resp.model}")
        return resp

    @staticmethod
    def _text(resp):
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()

    # ---- the flow ----

    def ask(self, question, history=None):
        """history: earlier turns [{"role": "user"|"assistant", "content": str}], oldest first, alternating and
        starting with "user". Context only: numbers in this answer must come from this answer's tool calls."""
        cfg, usage = self.cfg, Usage()
        history = list(history or [])
        calls, status = [], "ok"
        system = P.RESEARCH if cfg.writer_model else P.RESEARCH_SINGLE
        messages = [*({"role": m["role"], "content": m["content"]} for m in history), {"role": "user", "content": question}]
        resp = None
        for _ in range(cfg.max_turns):
            resp = self._create(cfg.research_model, cfg.research_effort, usage, "research",
                                system=system, tools=TOOLS, messages=messages)
            if resp.stop_reason == "refusal":
                return self._result("", calls, usage, "refusal", system)
            if resp.stop_reason == "max_tokens":
                status = "max_tokens"
            messages.append({"role": "assistant", "content": resp.content})  # thinking blocks kept as-is
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = []
            for b in uses:
                rec = run_tool(b.name, b.input)
                calls.append(rec)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": result_text(rec),
                                **({"is_error": True} if rec["error"] else {})})
            messages.append({"role": "user", "content": results})  # all results in one message
        else:
            status = "max_turns"
        notes = self._text(resp) if resp is not None else ""

        if cfg.writer_model:
            answer, wmsgs, stop = self._write(question, calls, notes, usage, history)
            if stop == "refusal":
                return self._result("", calls, usage, "refusal", P.WRITER)
        else:
            answer, wmsgs, stop = notes, None, resp.stop_reason

        g = G.check(answer, calls, question)
        regenerated = False
        if g["ungrounded"] and cfg.regenerate:
            regenerated = True
            answer = self._rewrite(question, answer, g["ungrounded"], calls, wmsgs, messages, system, usage)
            g = G.check(answer, calls, question)
        res = self._result(answer, calls, usage, status if stop != "max_tokens" else "max_tokens",
                           P.WRITER if cfg.writer_model else system)
        res["grounding"] = {"ok": not g["ungrounded"], "ungrounded": g["ungrounded"], "regenerated": regenerated}
        res["research_notes"] = notes if cfg.writer_model else None
        return res

    def _write(self, question, calls, notes, usage, history=()):
        ok = [{"tool": c["name"], "args": c["args"], "result": c["result"]} for c in calls if c["error"] is None]
        convo = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in history)
        content = ((f"Conversation so far (context only):\n{convo}\n" if convo else "") + f"Question: {question}\n\nTool results:\n"
                   + (json.dumps(ok, separators=(",", ":"), default=str) if ok else "(none)")
                   + f"\n\nResearcher's notes:\n{notes or '(none)'}")
        msgs = [{"role": "user", "content": content}]
        resp = self._create(self.cfg.writer_model, self.cfg.writer_effort, usage, "write", cache=False,
                            system=P.WRITER, messages=msgs)
        msgs.append({"role": "assistant", "content": resp.content})
        return self._text(resp), msgs, resp.stop_reason

    def _rewrite(self, question, answer, bad, calls, wmsgs, rmsgs, system, usage):
        fix = P.REGENERATE.format(bad=", ".join(bad), tools_hint="" if self.cfg.writer_model else
                                  ", calling a tool if you need a number you don't have")
        if self.cfg.writer_model:
            msgs = wmsgs + [{"role": "user", "content": fix}]
            resp = self._create(self.cfg.writer_model, self.cfg.writer_effort, usage, "rewrite", cache=False,
                                system=P.WRITER, messages=msgs)
            return self._text(resp) or answer
        # Single model: let it call tools again if it needs a number it doesn't have.
        msgs = rmsgs + [{"role": "user", "content": fix}]
        for _ in range(3):
            resp = self._create(self.cfg.research_model, self.cfg.research_effort, usage, "rewrite",
                                system=system, tools=TOOLS, messages=msgs)
            msgs.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                return self._text(resp) or answer
            results = []
            for b in uses:
                rec = run_tool(b.name, b.input)
                calls.append(rec)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": result_text(rec),
                                **({"is_error": True} if rec["error"] else {})})
            msgs.append({"role": "user", "content": results})
        return answer

    def _result(self, answer, calls, usage, status, system):
        models = list(usage.by_model)
        return {"answer": answer, "tool_calls": calls, "model": models[0] if len(models) == 1 else "+".join(models),
                "usage": usage.total(), "usage_by_model": usage.by_model, "model_calls": usage.calls,
                "stop_reason": {"ok": "end_turn"}.get(status, status), "status": status, "system": system}


# ---- entry points for the eval runner (python -m evals.run --answerer rallyagent.agent:answer_split) ----

_agents = {}


def _answer(config, case):
    if config not in _agents:
        _agents[config] = Agent(config)
    return _agents[config].ask(case["question"])


def answer_split(case):
    return _answer("split", case)


def answer_opus(case):
    return _answer("opus", case)


def answer_haiku(case):
    return _answer("haiku", case)


answer = answer_haiku  # default: cheapest, and as accurate on the evals so far


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    ap.add_argument("--config", default="haiku", choices=sorted(CONFIGS))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    r = Agent(a.config).ask(a.question)
    print(r["answer"] or f"(no answer: {r['status']})")
    if not r["grounding"]["ok"]:
        print(f"\n[flagged: numbers not from any tool: {', '.join(r['grounding']['ungrounded'])}]")
    if a.verbose:
        for c in r["tool_calls"]:
            print(f"\n> {c['name']}({json.dumps(c['args'])})" + (f"  ERROR {c['error']}" if c["error"] else ""))
        print(f"\nmodel calls: {json.dumps(r['model_calls'], indent=1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
