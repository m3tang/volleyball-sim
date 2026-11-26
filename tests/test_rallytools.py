import pytest

from rallytools import core as C
from vbsim.data import DB


def test_win_probability_matches_site():
    # Same numbers the site's match simulator shows (checked against the JS math in web/template.html).
    r = C.win_probability("Nebraska", 2025, "Texas A&M", 2025)
    assert r["p_team_a_wins"] == pytest.approx(0.8730, abs=1e-4)
    assert sum(r["final_score_probs_team_a_first"].values()) == pytest.approx(1, abs=1e-3)
    flipped = C.win_probability("Texas A&M", 2025, "Nebraska", 2025)
    assert flipped["p_team_a_wins"] == pytest.approx(1 - r["p_team_a_wins"], abs=1e-4)


def test_home_court_helps():
    neutral = C.win_probability("Kentucky", 2025, "Pittsburgh", 2025)["p_team_a_wins"]
    home = C.win_probability("Kentucky", 2025, "Pittsburgh", 2025, venue="team_a_home")["p_team_a_wins"]
    away = C.win_probability("Kentucky", 2025, "Pittsburgh", 2025, venue="team_b_home")["p_team_a_wins"]
    assert away < neutral < home


def test_team_names_resolve_loosely_and_fail_helpfully():
    assert C.resolve_team("penn state", 2024)["n"] == "Penn St."
    assert C.resolve_team("texas a&m", 2025)["n"] == "Texas A&M"
    with pytest.raises(C.ToolError, match="Did you mean"):
        C.resolve_team("Nebrasak", 2025)
    with pytest.raises(C.ToolError, match="seasons available"):
        C.win_probability("Nebraska", 2019, "Texas", 2025)


def test_bracket_odds_are_consistent():
    f = C.bracket_forecast(2025, top=64)
    assert sum(t["p_title"] for t in f["title_odds"]) == pytest.approx(1, abs=1e-3)
    a = C.advancement_odds("Texas A&M", 2025)
    p = [s["p"] for s in a["p_reach"]]
    assert p == sorted(p, reverse=True) and a["result"] == "won the title"
    assert a["p_reach"][-1]["p"] == f["actual_champion_pre_tournament_title_odds"]


def test_explain_upset_points():
    x = C.explain_upset(2025, "Kentucky", "Texas A&M")
    assert x["upset"] and x["winner"] == "Texas A&M"
    top = x["reasons"][0]
    assert top["points_for_winner"] == pytest.approx(
        (top["actual_rate"] - top["expected_rate"]) * top["chances"], abs=0.06)
    with pytest.raises(C.ToolError, match="did not meet"):
        C.explain_upset(2025, "Nebraska", "Stanford")


def test_odds_text_matches_site_rounding():
    assert C.odds_text(0.0043) == "0.4% (about 1 in 250)"
    assert C.odds_text(0.9959) == ">99%"
    assert C.odds_text(0.873) == "87%"


needs_db = pytest.mark.skipif(not DB.exists(), reason="data/vb.duckdb not present")


@needs_db
@pytest.mark.parametrize("sql, msg", [
    ("select * from events limit 1", "not allowed"),
    ("drop table matches", "only SELECT"),
    ("select 1; select 2", "exactly one"),
    ("select * from read_csv('/etc/passwd')", "disabled"),
    ("copy matches to 'x.csv'", "only SELECT"),
])
def test_sql_guardrails(sql, msg):
    from rallytools.sql import query_stats
    with pytest.raises(C.ToolError, match=msg):
        query_stats(sql)


@needs_db
def test_sql_row_limit_and_timeout():
    from rallytools.sql import query_stats
    r = query_stats("select match_id from matches", max_rows=5)
    assert r["row_count"] == 5 and r["truncated"]
    with pytest.raises(C.ToolError, match="longer than"):
        query_stats("select count(*) from rallies a, rallies b", timeout_s=1)


def test_upset_reasons_are_ranked_with_plain_labels():
    x = C.explain_upset(2024, "Missouri", "SMU")
    assert [r["rank"] for r in x["reasons"]] == list(range(1, len(x["reasons"]) + 1))
    assert x["biggest_factor"] == x["reasons"][0]
    assert x["biggest_factor"]["what"] == "Missouri's passing: how often Missouri got aced on SMU's serves"
