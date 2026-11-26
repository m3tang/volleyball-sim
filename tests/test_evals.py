"""The grader and grounding checker on realistic prose, plus the harness self-test."""
import json

import pytest

from evals import run as R
from evals.answerers import spec_to_call
from evals.grade import grade
from rallytools import grounding as G
from rallytools.registry import call
from vbsim.data import DB

CASES = {c["id"]: c for c in R.load_cases()}


def answer(case_id, text, calls=None):
    """Grade `text` for a golden case, with the case's own tool call unless calls are given."""
    c = CASES[case_id]
    if calls is None:
        calls = [call(*spec_to_call(c["tools"][0], c))] if c["tools"] else []
    return grade(c, {"answer": text, "tool_calls": calls})["grade"]


# dream_matchup-001: "Could Nebraska have beaten 2023 Texas?" -> 2025 Nebraska 87.3% (or 2023 Nebraska 66%,
# either way the answer must say which Nebraska)
@pytest.mark.parametrize("text, correct", [
    ("Probably. The model makes 2025 Nebraska an 87% favorite over 2023 Texas.", 1),
    ("2025 Nebraska wins this one 87.3 percent of the time.", 1),
    ("2025 Nebraska would be a heavy favorite, around 0.87.", 1),
    ("2025 Nebraska wins about 88% of the time.", 0),     # wrong at the precision written
    ("2025 Nebraska is the favorite.", 0),               # no number
    ("Nebraska wins 87% of the time.", 0),               # doesn't say which Nebraska
    ("", 0),
])
def test_correct_at_stated_precision(text, correct):
    assert answer("dream_matchup-001", text)["correct"] == correct


def test_long_shot_formats():
    # bracket-015: Texas A&M 2025 title odds 0.43% (site shows "0.4% (about 1 in 250)")
    for text in ("The model gave Texas A&M a 0.4% chance (about 1 in 250).",
                 "Texas A&M had roughly a 1-in-233 shot before the tournament.",
                 "Just 0.43%, so about one in 233."):
        g = answer("bracket-015", text)
        assert g["correct"] == 1 and g["grounded"] == 1, text
    assert answer("bracket-015", "Texas A&M was at 0% before the tournament.")["correct"] == 0


def test_fan_names_count_as_facts():
    # bracket-011: model pick Pittsburgh 2024, 52%
    assert answer("bracket-011", "The model picked Pitt, at 52% to win the title.")["correct"] == 1
    # match_odds-002: favorite Southern California -> "USC" is fine
    g = answer("match_odds-002", "USC is the favorite at 60% even at Kansas State.")
    assert g["correct"] == 1


def test_hallucinated_numbers_are_flagged_but_tools_ground_the_rest():
    g = answer("dream_matchup-001", "2025 Nebraska is an 87% favorite. Nebraska also hit .412 and went 33-1.")
    assert g["correct"] == 1 and g["grounded"] == pytest.approx(1 / 3, abs=0.01)


def test_no_tools_means_ungrounded():
    g = answer("dream_matchup-001", "2025 Nebraska is an 87% favorite.", calls=[])
    assert g["correct"] == 1 and g["grounded"] == 0 and g["tool_choice"] == 0


def test_tool_choice_accepts_any_valid_route():
    c = CASES["dream_matchup-001"]  # 2025 Nebraska vs 2023 Texas, neutral
    swapped = call("win_probability", {"team_a": "texas", "season_a": 2023, "team_b": "Nebraska", "season_b": 2025})
    wrong_season = call("win_probability", {"team_a": "Nebraska", "season_a": 2024, "team_b": "Texas", "season_b": 2023})
    home = call("win_probability", {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "season_b": 2023,
                                    "venue": "team_a_home"})
    assert grade(c, {"answer": "", "tool_calls": [swapped]})["grade"]["tool_choice"] == 1
    assert grade(c, {"answer": "", "tool_calls": [wrong_season]})["grade"]["tool_choice"] == 0
    assert grade(c, {"answer": "", "tool_calls": [home]})["grade"]["tool_choice"] == 0
    # A record question answered via SQL instead of team_profile is fine when the case allows it.
    ace = CASES["team_profile-003"]
    assert any(t["tool"] == "query_stats" for t in ace["tools"])


def test_refusal_cases():
    g = answer("out_of_scope-001", "The data doesn't track injuries, so I can't say who was hurt.")
    assert g["correct"] == 1 and "tool_choice" not in g
    assert answer("out_of_scope-001", "Three starters missed time with ankle injuries.")["correct"] == 0
    # Declining an answerable question is wrong, and shows up in `refused`.
    g = answer("dream_matchup-001", "The data doesn't cover hypothetical matchups.")
    assert g["correct"] == 0 and g["refused"] == 1


def test_upset_reason_must_be_named():
    # upset-011: Ohio State over Minnesota 2022, top reason = scramble
    c = CASES["upset-011"]
    p = c["expect"]["numbers"][0]["value"]
    good = f"Ohio State was only a {100 * p:.0f}% underdog, and it won the scramble: long rallies after a dig went its way."
    bad = f"Ohio State was a {100 * p:.0f}% underdog and served tough."
    assert answer("upset-011", good)["correct"] == 1
    assert answer("upset-011", bad)["correct"] == 0


def test_grounding_exemptions():
    r = G.check("In the 2024 final (round 6, best-of-5) Penn State won in four sets.", [], "Who won the 2024 final?")
    assert r["n"] == 0


def test_harness_selftest():
    # Without the database (CI), the SQL cases are left out; everything else still runs for real.
    assert R.selftest(R.load_cases(need_db=None if DB.exists() else False))


def test_golden_answers_still_match_the_model():
    """A model or data rebuild that moves an answer must regenerate the golden set on purpose."""
    from evals import golden
    if not DB.exists():
        pytest.skip("golden generation needs data/vb.duckdb")
    new = golden.generate()
    old = [json.loads(l) for l in golden.OUT.read_text().splitlines()]
    assert new == old


def test_record_written_as_a_pair_counts():
    # team_profile-005: Tennessee 2025, 20-8, ranked 12th (found in the pilot: "20-8" was missed)
    assert answer("team_profile-005", "Tennessee went 20-8 in 2025 and the model ranked them 12th.")["correct"] == 1
    assert answer("team_profile-005", "Tennessee went 21-8 in 2025 and the model ranked them 12th.")["correct"] == 0


def test_a_caveat_is_not_a_refusal():
    g = answer("bracket-015", "The rally model can't say exactly why, but it gave Texas A&M a 0.4% chance "
                              "(about 1 in 250). The data doesn't cover injuries.")
    assert g["correct"] == 1 and g["refused"] == 0


def test_hyphenated_keywords_match():
    # upset-008: San Diego over Stanford 2022, top reason first-attack scoring (found in the upset comparison)
    c = CASES["upset-008"]
    p = c["expect"]["numbers"][0]["value"]
    assert answer("upset-008", f"San Diego had a {100 * p:.0f}% chance; the biggest factor was its first-attack scoring.")["correct"] == 1
