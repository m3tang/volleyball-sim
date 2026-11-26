"""Question-log triage (evals/logs.py) and logged/live golden cases."""
import json

from evals import logs as L
from evals.grade import sufficient
from rallytools.registry import call


def _row(i, q, **kw):
    return {"id": str(i), "ts": f"2026-11-2{i % 9}T10:00:00Z", "question": q, "answer": kw.pop("answer", "Texas is 61%."),
            "shown": True, "grounded": True, "status": "ok", "tools": [{"name": "win_probability", "args": {}}],
            "cost_usd": 0.001, "latency_s": 3.0, "history_turns": 0, **kw}


def test_triage_sorts_worst_first(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DIR", tmp_path)
    monkeypatch.setattr(L, "ASKS", tmp_path / "asks.jsonl")
    monkeypatch.setattr(L, "INBOX", tmp_path / "inbox.md")
    rows = [
        _row(1, "Could Texas beat Stanford?", shown=False, ungrounded=["63%"]),
        _row(2, "How good was Nebrasak?", tool_errors=[{"name": "team_profile", "args": {}, "error": "no team named 'Nebrasak'"}]),
        _row(3, "Who's injured at Wisconsin?", answer="The data doesn't cover injuries.", tools=[]),
        _row(4, "And at home?", history_turns=2, history=[{"role": "user", "content": "Texas vs Stanford?"},
                                                         {"role": "assistant", "content": "Texas 55%."}]),
        _row(5, "What was Nebraska's record in 2025?"),          # already in logged.py: skipped
        _row(6, "Who has the best libero in the SEC?"),
        _row(7, "who has the best libero in the SEC"),           # duplicate after normalizing
    ]
    (tmp_path / "asks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    L.triage()
    md = (tmp_path / "inbox.md").read_text()
    sec = {s.split("\n")[0]: s for s in md.split("## ")[1:]}
    assert "Could Texas beat Stanford?" in sec["Withheld: failed the number check (1)"]
    assert "unchecked numbers: 63%" in md
    assert "Nebrasak" in sec["Tool errors (1)"]
    assert "Wisconsin" in sec["Declined: check the data really can't answer these (1)"]
    assert "> user: Texas vs Stanford?" in sec["Follow-ups (1)"]
    new = sec["New questions, not in the golden set (1)"]
    assert "libero" in new and "Nebraska's record" not in md


def test_live_tool_choice_accepts_either_team_order():
    spec = {"tool": "live_win_probability", "teams": [["Texas", 2025], ["Nebraska", 2025]], "home": None,
            "args": {"team_a": "Texas", "season_a": 2025, "team_b": "Nebraska", "season_b": 2025, "venue": "neutral",
                     "sets_a": 1, "sets_b": 2, "points_a": 20, "points_b": 23, "serving": "team_b"}}
    same = call("live_win_probability", {"team_a": "texas", "season_a": 2025, "team_b": "Nebraska", "sets_a": 1,
                                         "sets_b": 2, "points_a": 20, "points_b": 23, "serving": "team_b"})
    swapped = call("live_win_probability", {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "sets_a": 2,
                                            "sets_b": 1, "points_a": 23, "points_b": 20, "serving": "team_a"})
    wrong_server = call("live_win_probability", {"team_a": "Texas", "season_a": 2025, "team_b": "Nebraska", "sets_a": 1,
                                                 "sets_b": 2, "points_a": 20, "points_b": 23, "serving": "team_a"})
    wrong_score = call("live_win_probability", {"team_a": "Texas", "season_a": 2025, "team_b": "Nebraska", "sets_a": 1,
                                                "sets_b": 2, "points_a": 23, "points_b": 20, "serving": "team_b"})
    assert sufficient(spec, same) and sufficient(spec, swapped)
    assert not sufficient(spec, wrong_server) and not sufficient(spec, wrong_score)


def test_logged_cases_carry_history_and_read_answers_from_tools():
    cases = {c["id"]: c for c in map(json.loads, (L.HERE / "golden.jsonl").read_text().splitlines())}
    follow = next(c for c in cases.values() if c["category"] == "logged" and c.get("history"))
    assert [m["role"] for m in follow["history"]] == ["user", "assistant"]
    want = call("bracket_forecast", {"season": 2024})["result"]["actual_champion_pre_tournament_title_odds"]
    assert follow["expect"]["numbers"][0]["value"] == want
