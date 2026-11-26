"""Predict the 2025 NCAA tournament from data available before it started, then score it.

  * Models are fitted on 2025 matches before the first round (prior: 2024 season).
  * Advancement odds are exact (every path through the bracket), not sampled.
  * Baseline: pick the team with the better regular-season win percentage.

Usage:  python -m analysis.bracket_2025
"""
import datetime as dt

import duckdb
import numpy as np

from vbsim import bracket as B
from vbsim import data as D
from vbsim import models as M
from vbsim.matchsim import match_win_prob

SEASON = 2025
TOURNEY = (dt.date(2025, 12, 4), dt.date(2025, 12, 21))
TABLES = ("sideout", "serve", "fb", "tr")
POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}  # standard bracket scoring, 1920 max
COLS = ["Rd 2", "Sweet 16", "Elite 8", "Final 4", "Final", "Champ"]


def fit_pretournament(d, idx, cls, season_year=SEASON, start=TOURNEY[0], half_life=None):
    """Fit on the season's matches before the tournament, with last season as the prior."""
    lam, rho, _ = M.TUNED[cls.name]
    season = {t: d[t]["match_id"] // 100000 for t in TABLES}
    prev = {t: season[t] == season_year - 1 for t in TABLES}
    has_prev = any(prev[t].any() for t in TABLES)
    prior = cls(len(idx), lam).fit(d, prev, idx).scaled(rho) if has_prev else None
    cutoff = np.datetime64(start)
    train = {t: (season[t] == season_year) & (d[t]["date"] < cutoff) for t in TABLES}
    return cls(len(idx), lam).fit(d, train, idx, prior=prior,
                                  weights=D.recency_weights(d, TABLES, start, half_life))


def make_win_prob(model, idx, form_sd):
    def win_prob(a_list, b_list, host):
        A, Bm = np.meshgrid(np.arange(len(a_list)), np.arange(len(b_list)), indexing="ij")
        a = np.array(a_list)[A.ravel()]
        b = np.array(b_list)[Bm.ravel()]
        h = np.where(a == host, 1.0, np.where(b == host, -1.0, 0.0))
        so_a = model.sideout(idx(a), idx(b), h)
        so_b = model.sideout(idx(b), idx(a), -h)
        return match_win_prob(1 - so_b, so_a, form_sd).reshape(A.shape)
    return win_prob


def record_before(con, cutoff, season=SEASON):
    rows = con.execute("""
      SELECT t, avg(won::INT), count(*) FROM (
        SELECT away_team AS t, winner = away_team AS won FROM matches WHERE season = ? AND date < ? AND winner IS NOT NULL
        UNION ALL
        SELECT home_team, winner = home_team FROM matches WHERE season = ? AND date < ? AND winner IS NOT NULL)
      GROUP BY t""", [season, cutoff, season, cutoff]).fetchall()
    return {t: (pct, n) for t, pct, n in rows}


def furthest(root):
    """Number of tournament matches each team won (6 = champion)."""
    wins = {t: 0 for t in root.teams}
    for n in B.walk(root):
        wins[n.winner] += 1
    return wins


def score_bracket(root, picks):
    return sum(POINTS[n.round] for n in B.walk(root) if picks[id(n)] == n.winner)


def baseline_picks(root, record):
    picks = {}

    def pick(node):
        sides = [pick(c) if isinstance(c, B.Node) else c for c in (node.left, node.right)]
        w = max(sides, key=lambda t: record.get(t, (0, 0))[0])
        picks[id(node)] = w
        return w

    pick(root)
    return picks


def game_metrics(root, win_prob):
    rows = []
    for n in B.walk(root):
        a = n.left.winner if isinstance(n.left, B.Node) else n.left
        b = n.right.winner if isinstance(n.right, B.Node) else n.right
        p_a = win_prob([a], [b], n.host)[0, 0]
        p_w = p_a if n.winner == a else 1 - p_a
        rows.append((n.round, a, b, n.winner, p_w))
    return rows


def main():
    d = D.load([SEASON - 1, SEASON])
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])
    root = B.rebuild(D.DB, SEASON, *TOURNEY)
    con = duckdb.connect(str(D.DB), read_only=True)
    record = record_before(con, TOURNEY[0])
    reached = furthest(root)

    results = {}
    for cls in (M.TouchModel, M.SideoutModel):
        model = fit_pretournament(d, idx, cls)
        wp = make_win_prob(model, idx, M.TUNED[cls.name][2])
        adv = B.advancement(root, wp)
        results[cls.name] = (model, wp, adv)

    model, wp, adv = results["touch"]
    odds = {t: [0.0] * 6 for t in root.teams}
    for n in B.walk(root):
        for t, p in adv[id(n)].items():
            odds[t][n.round - 1] = p
    total_title = sum(o[5] for o in odds.values())
    assert abs(total_title - 1) < 1e-9, total_title

    names = [f"lost {B.ROUND_NAMES[r]}" for r in range(1, 7)] + ["CHAMPION"]
    print("Pre-tournament odds, touch model (top 20 by title odds):")
    print(f"{'team':<22}{'record':>8}" + "".join(f"{c:>9}" for c in COLS) + "   actual")
    for t in sorted(odds, key=lambda t: -odds[t][5])[:20]:
        pct, n = record.get(t, (float('nan'), 0))
        print(f"{t:<22}{pct:>8.3f}" + "".join(f"{p:>9.1%}" for p in odds[t]) + f"   {names[reached[t]]}")

    print("\nGame-by-game (each actual matchup, at its actual site):")
    print(f"{'':<28}{'games':>6}{'log loss':>10}{'accuracy':>10}")
    print(f"{'coin flip':<28}{63:>6}{np.log(2):>10.4f}{'50.0%':>10}")
    for name, (_, wp_m, _) in results.items():
        rows = game_metrics(root, wp_m)
        pw = np.array([r[4] for r in rows])
        print(f"{name + ' model':<28}{len(pw):>6}{-np.log(pw).mean():>10.4f}{(pw > 0.5).mean():>10.1%}")
    rows = game_metrics(root, wp)
    base_correct = []
    for rnd, a, b, w, _ in rows:
        fav = a if record.get(a, (0, 0))[0] >= record.get(b, (0, 0))[0] else b
        base_correct.append(fav == w)
    print(f"{'better record':<28}{len(rows):>6}{'':>10}{np.mean(base_correct):>10.1%}")

    print("\nBy round (touch model): games, average P(actual winner), favorite won")
    for r in range(1, 7):
        sel = [x for x in rows if x[0] == r]
        pw = np.array([x[4] for x in sel])
        print(f"  {B.ROUND_NAMES[r]:<20}{len(sel):>3}{pw.mean():>8.1%}{(pw > 0.5).sum():>4}/{len(sel)}")

    print("\nBiggest upsets by the model's odds:")
    for rnd, a, b, w, p in sorted(rows, key=lambda x: x[4])[:6]:
        loser = b if w == a else a
        print(f"  {B.ROUND_NAMES[rnd]:<20} {w} over {loser}: model gave {w} {p:.0%}")

    picks = B.most_likely_bracket(root, adv)
    base = baseline_picks(root, record)
    print("\nBracket score (10/20/40/80/160/320 per round, max 1920):")
    print(f"  touch model bracket: {score_bracket(root, picks)}   champion pick: {picks[id(root)]}")
    print(f"  better-record bracket: {score_bracket(root, base)}   champion pick: {base[id(root)]}")

    # What the touch model says about the final.
    a, b = [c.winner for c in root.children()]
    ia, ib = idx([a]), idx([b])
    print(f"\nThe final, {a} vs {b} (neutral site), touch model step probabilities:")
    for who, (r_, s_) in ((a, (ia, ib)), (b, (ib, ia))):
        ex = model.explain(r_, s_, np.zeros(1))
        srv = ex["serve(ace, serr, inplay)"][0]
        fb = ex["first ball(win, lose, to_opp)"][0]
        tr = ex["transition(win, lose, to_opp)"][0]
        so = model.sideout(r_, s_, np.zeros(1))[0]
        print(f"  {who} receiving: opponent ace {srv[0]:.1%}, service error {srv[1]:.1%} | "
              f"first ball kill/win {fb[0]:.1%}, lost {fb[1]:.1%} | transition win {tr[0]:.1%} | sideout {so:.1%}")
    print(f"  {a} win probability: {wp([a], [b], None)[0, 0]:.1%}")


if __name__ == "__main__":
    main()
