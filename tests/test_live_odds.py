"""Odds from a score in progress (vbsim.matchsim start=, rallytools live_win_probability)."""
import itertools

import numpy as np
import pytest

from rallytools import core as C
from vbsim.matchsim import set_win_prob

P_SRV, P_RCV = 0.58, 0.37


@pytest.mark.parametrize("t", [25, 15])
def test_set_from_any_score_satisfies_the_rally_recursion(t):
    """From every live score, P(win set) = p * P(after winning the rally) + (1 - p) * P(after losing it)."""
    def v(a, b, s):
        if C._set_over(a, b, t):
            return float(a > b)
        return float(set_win_prob(P_SRV, P_RCV, t, s, (a, b)))
    for a, b in itertools.product(range(t + 4), repeat=2):
        if C._set_over(a, b, t) or (max(a, b) > t and abs(a - b) > 1):
            continue
        for s in (True, False):
            p = P_SRV if s else P_RCV
            assert v(a, b, s) == pytest.approx(p * v(a + 1, b, True) + (1 - p) * v(a, b + 1, False), abs=1e-12)


def test_deuce_depends_only_on_lead_and_server():
    for s in (True, False):
        assert set_win_prob(P_SRV, P_RCV, 25, s, (24, 24)) == pytest.approx(set_win_prob(P_SRV, P_RCV, 25, s, (31, 31)))
        assert set_win_prob(P_SRV, P_RCV, 25, s, (25, 24)) == pytest.approx(set_win_prob(P_SRV, P_RCV, 25, s, (40, 39)))


def test_set_matches_simulation():
    rng = np.random.default_rng(0)
    wins, n = 0, 40000
    for _ in range(n):
        a, b, s = 20, 22, False
        while not C._set_over(a, b, 25):
            if rng.random() < (P_SRV if s else P_RCV):
                a, s = a + 1, True
            else:
                b, s = b + 1, False
        wins += a > b
    assert wins / n == pytest.approx(float(set_win_prob(P_SRV, P_RCV, 25, False, (20, 22))), abs=0.008)


def test_start_of_match_equals_pre_match_odds():
    pre = C.win_probability("Texas", 2025, "Nebraska", 2025, "team_a_home")["p_team_a_wins"]
    live = [C.live_win_probability("Texas", 2025, "Nebraska", 2025, "team_a_home", serving=s)["p_team_a_wins"]
            for s in ("team_a", "team_b")]
    assert sum(live) / 2 == pytest.approx(pre, abs=1e-4)


def test_winning_sets_and_rallies_moves_the_odds_the_right_way():
    f = lambda **k: C.live_win_probability("Texas", 2025, "Nebraska", 2025, **k)
    base = f()["p_team_a_wins"]
    assert f(sets_a=1)["p_team_a_wins"] > base > f(sets_b=1)["p_team_a_wins"]
    r = f(sets_a=1, sets_b=1, points_a=18, points_b=18)
    assert r["next_rally"]["if Texas wins it"] > r["p_team_a_wins"] > r["next_rally"]["if Nebraska wins it"]
    # Winning the 5th set at 14-13 ends the match.
    assert f(sets_a=2, sets_b=2, points_a=14, points_b=13)["next_rally"]["if Texas wins it"] == 1.0


@pytest.mark.parametrize("kw, msg", [({"sets_a": 3}, "match is over"), ({"points_a": 25, "points_b": 20}, "finished set"),
                                     ({"sets_a": 2, "sets_b": 2, "points_a": 15, "points_b": 13}, "sets go to 15"),
                                     ({"serving": "home"}, "serving")])
def test_bad_scores_are_tool_errors(kw, msg):
    with pytest.raises(C.ToolError, match=msg):
        C.live_win_probability("Texas", 2025, "Nebraska", 2025, **kw)
