import numpy as np
import pytest

from vbsim import fit as F
from vbsim.matchsim import match_win_prob, set_win_prob, simulate_set


def test_even_teams_are_fifty_fifty():
    p = match_win_prob(np.array([0.43]), np.array([0.57]))
    assert p[0] == pytest.approx(0.5, abs=1e-9)


def test_set_prob_matches_monte_carlo():
    rng = np.random.default_rng(0)
    p_srv, p_rcv = 0.46, 0.60  # A is the stronger team
    for target in (25, 15):
        exact = set_win_prob(np.array([p_srv]), np.array([p_rcv]), target, True)[0]
        n = 40000
        wins = sum(a > b for a, b in (simulate_set(p_srv, p_rcv, target, True, rng) for _ in range(n)))
        assert exact == pytest.approx(wins / n, abs=0.01)


def test_stronger_team_wins_more():
    rcv = np.linspace(0.50, 0.65, 8)
    p = match_win_prob(1 - 0.57 * np.ones(8), rcv)
    assert np.all(np.diff(p) > 0)


def test_bracket_advancement_matches_hand_calculation():
    from vbsim.bracket import Node, _fill_teams, advancement

    strength = {"A": 0.8, "B": 0.6, "C": 0.5, "D": 0.3}

    def win_prob(a_list, b_list, host):
        a = np.array([strength[t] for t in a_list])[:, None]
        b = np.array([strength[t] for t in b_list])[None, :]
        return a / (a + b)

    semi1 = Node(1, None, "A", "B", "A", None)
    semi2 = Node(1, None, "C", "D", "C", None)
    final = Node(2, None, semi1, semi2, "A", None)
    _fill_teams(final)
    adv = advancement(final, win_prob)
    p = lambda x, y: strength[x] / (strength[x] + strength[y])
    expected_a = p("A", "B") * (p("C", "D") * p("A", "C") + p("D", "C") * p("A", "D"))
    assert adv[id(final)]["A"] == pytest.approx(expected_a)
    assert sum(adv[id(final)].values()) == pytest.approx(1.0)


def test_fit_recovers_team_strengths():
    rng = np.random.default_rng(1)
    n_teams = 30
    true_off = rng.normal(0, 0.3, (n_teams, 1))
    true_def = rng.normal(0, 0.3, (n_teams, 1))
    t = rng.integers(0, n_teams, 3000)
    o = (t + rng.integers(1, n_teams, 3000)) % n_teams
    home = rng.choice([-1, 0, 1], 3000)
    eta = 0.28 + true_off[t, 0] - true_def[o, 0] + 0.1 * home
    n = np.full(3000, 80)
    k = rng.binomial(n, 1 / (1 + np.exp(-eta)))
    p = F.fit(np.stack([k, n - k], 1), t, o, home, n_teams, lam=1.0)
    assert np.corrcoef(p.off[:, 0], true_off[:, 0])[0, 1] > 0.9
    assert np.corrcoef(p.def_[:, 0], true_def[:, 0])[0, 1] > 0.9
    assert p.home[0] == pytest.approx(0.1, abs=0.03)


def test_form_pulls_odds_toward_even_and_keeps_symmetry():
    p_srv, p_rcv = np.array([0.45, 0.43]), np.array([0.62, 0.57])
    base = match_win_prob(p_srv, p_rcv)
    wide = match_win_prob(p_srv, p_rcv, form_sd=0.3)
    assert wide[0] < base[0] and wide[0] > 0.5
    assert wide[1] == pytest.approx(0.5, abs=1e-9)
