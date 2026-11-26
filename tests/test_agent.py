"""The agent's control flow against a scripted fake client (no API calls)."""
from types import SimpleNamespace as NS

import pytest

from rallyagent import agent as A
from rallyagent.tools import TOOLS


def text(t):
    return NS(type="text", text=t)


def tool_use(i, name, inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


def resp(model, content, stop):
    return NS(model=model, content=content, stop_reason=stop,
              usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                       cache_creation_input_tokens=0, iterations=None))


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.script.pop(0)


NEB_TEX = {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "season_b": 2023}


def test_split_flow_researches_then_writes():
    fake = FakeClient([
        resp(A.HAIKU, [tool_use(1, "win_probability", NEB_TEX)], "tool_use"),
        resp(A.HAIKU, [text("Nebraska 2025 favored; assumed 2025.")], "end_turn"),
        resp(A.OPUS, [text("Probably. The model makes 2025 Nebraska an 87% favorite over 2023 Texas.")], "end_turn"),
    ])
    r = A.Agent("split", client=fake).ask("Could Nebraska have beaten 2023 Texas?")
    assert r["answer"].startswith("Probably") and r["grounding"]["ok"] and not r["grounding"]["regenerated"]
    assert [c["name"] for c in r["tool_calls"]] == ["win_probability"]
    research, _, write = fake.requests
    assert research["model"] == A.HAIKU and research["tools"] == TOOLS and "fallbacks" not in research
    assert write["model"] == A.OPUS and "tools" not in write and write["fallbacks"] == "default" and "cache_control" not in write
    assert A.FALLBACK_BETA in write["betas"]
    assert "0.873" in write["messages"][0]["content"]          # the writer sees the tool result
    # Tool results go back as one user message, and the assistant turn is passed back unchanged.
    turn2 = fake.requests[1]["messages"]
    assert turn2[1]["role"] == "assistant" and turn2[2]["content"][0]["tool_use_id"] == "tu_1"
    assert set(r["usage_by_model"]) == {A.HAIKU, A.OPUS}


def test_ungrounded_answer_is_rewritten_then_flagged():
    fake = FakeClient([
        resp(A.HAIKU, [tool_use(1, "win_probability", NEB_TEX)], "tool_use"),
        resp(A.HAIKU, [text("done")], "end_turn"),
        resp(A.OPUS, [text("Nebraska is an 87% favorite and hit .412 in 2025.")], "end_turn"),
        resp(A.OPUS, [text("Nebraska is an 87% favorite.")], "end_turn"),          # rewrite fixes it
    ])
    r = A.Agent("split", client=fake).ask("Could Nebraska have beaten 2023 Texas?")
    assert r["grounding"] == {"ok": True, "ungrounded": [], "regenerated": True}
    assert ".412" in fake.requests[3]["messages"][-1]["content"]                     # told which number
    fake = FakeClient([
        resp(A.HAIKU, [tool_use(1, "win_probability", NEB_TEX)], "tool_use"),
        resp(A.HAIKU, [text("done")], "end_turn"),
        resp(A.OPUS, [text("Nebraska hit .412.")], "end_turn"),
        resp(A.OPUS, [text("Nebraska hit .412, really.")], "end_turn"),
    ])
    r = A.Agent("split", client=fake).ask("Could Nebraska have beaten 2023 Texas?")
    assert not r["grounding"]["ok"] and r["grounding"]["ungrounded"] == [".412"]


def test_single_model_answers_from_its_own_loop_and_tool_errors_reach_it():
    fake = FakeClient([
        resp(A.OPUS, [tool_use(1, "team_profile", {"team": "Nebrasak", "season": 2025})], "tool_use"),
        resp(A.OPUS, [tool_use(2, "team_profile", {"team": "Nebraska", "season": 2025})], "tool_use"),
        resp(A.OPUS, [text("Nebraska went 33-1 in 2025 and ranked No. 1.")], "end_turn"),
    ])
    r = A.Agent("opus", client=fake).ask("How good was Nebraska in 2025?")
    err = fake.requests[1]["messages"][2]["content"][0]
    assert err["is_error"] and "Did you mean: Nebraska" in err["content"]
    assert r["grounding"]["ok"] and len(fake.requests) == 3


def test_refusal_stop_is_reported_not_crashed():
    fake = FakeClient([resp(A.OPUS, [], "refusal")])
    r = A.Agent("opus", client=fake).ask("anything")
    assert r["status"] == "refusal" and r["answer"] == ""


def test_substituted_model_fails_loudly():
    fake = FakeClient([resp("claude-some-other-model", [text("hi")], "end_turn")])
    with pytest.raises(A.ServedModelMismatch):
        A.Agent("opus", client=fake).ask("anything")


def test_tool_schemas_are_strict_and_match_the_registry():
    from rallytools.registry import TOOLS as REG
    assert {t["name"] for t in TOOLS} == set(REG)
    for t in TOOLS:
        s = t["input_schema"]
        assert t["strict"] and s["additionalProperties"] is False and set(s["required"]) <= set(s["properties"])


def test_credentials_prefer_env_then_keychain(monkeypatch):
    import subprocess
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert A._client_kwargs() == {}                                   # SDK reads the env itself
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(A.sys, "platform", "darwin")
    found = subprocess.CompletedProcess([], 0, stdout="sk-test\n", stderr="")
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: found)
    assert A._client_kwargs() == {"api_key": "sk-test"}
    missing = subprocess.CompletedProcess([], 44, stdout="", stderr="not found")
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: missing)
    assert A._client_kwargs() == {}


def test_history_goes_before_the_question():
    fake = FakeClient([resp(A.HAIKU, [tool_use(1, "win_probability", NEB_TEX)], "tool_use"),
                       resp(A.HAIKU, [text("2025 Nebraska is an 87% favorite.")], "end_turn")])
    h = [{"role": "user", "content": "Could Nebraska have beaten 2023 Texas?"}, {"role": "assistant", "content": "Yes."}]
    r = A.Agent("haiku", client=fake).ask("And in 2023?", history=h)
    assert r["grounding"]["ok"]
    msgs = fake.requests[0]["messages"]
    assert [m["role"] for m in msgs[:3]] == ["user", "assistant", "user"] and msgs[2]["content"] == "Newest question: And in 2023?"
