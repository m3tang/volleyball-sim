"""The site's in-browser agent tools (web/template.html AGENT block) must return what rallytools does.

Runs tests/site_agent_eval.js in Node on a set of tool calls and answers, then compares with the Python
tools and grounding checker. Skips when node is not installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rallytools import core as C
from rallytools import grounding as G
from rallytools.registry import call

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

CALLS = [
    ("win_probability", {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "season_b": 2023}),
    ("win_probability", {"team_a": "USC", "season_a": 2025, "team_b": "Kansas State", "venue": "team_b_home"}),
    ("win_probability", {"team_a": "Pitt", "season_a": 2024, "team_b": "Penn State", "venue": "team_a_home"}),
    ("team_profile", {"team": "Penn State", "season": 2024}),
    ("team_profile", {"team": "Western Kentucky", "season": 2021}),
    ("season_rankings", {"season": 2022, "top": 5}),
    ("bracket_forecast", {"season": 2025, "top": 12}),
    ("advancement_odds", {"team": "Texas A&M", "season": 2025}),
    ("tournament_matches", {"season": 2024, "upsets_only": True}),
    ("tournament_matches", {"season": 2023, "team": "Texas"}),
    ("explain_upset", {"season": 2024, "team_a": "Missouri", "team_b": "SMU"}),
    ("explain_upset", {"season": 2025, "team_a": "Kentucky", "team_b": "Texas A&M"}),
]
ERRORS = [
    ("win_probability", {"team_a": "Nebrasak", "season_a": 2025, "team_b": "Texas"}, "Did you mean"),
    ("win_probability", {"team_a": "Nebraska", "season_a": 2019, "team_b": "Texas"}, "seasons available"),
    ("explain_upset", {"season": 2025, "team_a": "Nebraska", "team_b": "Stanford"}, "did not meet"),
    ("bracket_forecast", {"season": 2020}, "no bracket"),
]


def _close(a, b, path=""):
    """Equal, with floats allowed to differ in the last rounding place."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        assert abs(a - b) <= 1.01e-4, f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _run_js(jobs, tmp_path):
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps(jobs))
    out = subprocess.run(["node", str(ROOT / "tests" / "site_agent_eval.js"), str(f)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_js_tools_match_python(tmp_path):
    js = _run_js([{"name": n, "args": a} for n, a in CALLS], tmp_path)
    for (name, args), got in zip(CALLS, js):
        assert "error" not in got, f"{name}{args}: {got['error']}"
        py = call(name, args)
        assert py["error"] is None
        _close(got["result"], json.loads(json.dumps(py["result"])), name)


def test_js_tool_errors_are_messages(tmp_path):
    js = _run_js([{"name": n, "args": a} for n, a, _ in ERRORS], tmp_path)
    for (name, args, want), got in zip(ERRORS, js):
        assert "error" in got and not got["error"].startswith("CRASH") and want in got["error"], got


def test_js_grounding_matches_python(tmp_path):
    calls = [call("win_probability", {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "season_b": 2023}),
             call("advancement_odds", {"team": "Texas A&M", "season": 2025}),
             call("team_profile", {"team": "Tennessee", "season": 2025})]
    answers = [
        "Probably. 2025 Nebraska is an 87% favorite over 2023 Texas (Texas 12.7%, about 1 in 8). A 3-0 sweep is likeliest at 46.6%.",
        "Texas A&M had a 0.4% chance (about 1 in 250), roughly a 1-in-233 shot. It's best-of-5.",
        "Tennessee went 20-8 and ranked 12th. Made up: 71%, .412, 1,234, 33-2, 6th, under 1%.",
        "In the 2024 final (round 6), Penn State won in four sets.",
    ]
    jobs = [{"ground": {"answer": a, "calls": calls, "question": "Could Nebraska have beaten 2023 Texas?"}} for a in answers]
    js = _run_js(jobs, tmp_path)
    for a, got in zip(answers, js):
        py = G.check(a, calls, "Could Nebraska have beaten 2023 Texas?")
        assert got["ground"]["n"] == py["n"] and got["ground"]["ungrounded"] == py["ungrounded"], (a, got["ground"], py)
