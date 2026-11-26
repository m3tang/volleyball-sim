"""Does counting recent matches more improve predictions, especially in the tournament?

Each setting is a half-life in days: a match that many days before the prediction counts
half as much (None = every match in the season counts the same, the current model).

  Tuning: 2024 week-by-week predictions and the 2021-2024 NCAA tournaments.
  Test:   2025 week-by-week predictions and the 2025 tournament, run once.

Usage:  python -m analysis.recency
"""
import numpy as np

from analysis import bracket_2025 as BR
from analysis.evaluate_models import metrics, run_season
from vbsim import bracket as B
from vbsim import data as D
from vbsim import models as M

TABLES = ("sideout", "serve", "fb", "tr")
HALF_LIVES = (None, 180, 90, 60, 30, 15)


def tournament(d, idx, season, half_life):
    """Log loss over the tournament's actual matchups, plus the champion's pre-tournament title odds."""
    lam, rho, sd = M.TUNED["touch"]
    root = B.rebuild(D.DB, season)
    start = min(n.date for n in B.walk(root) if n.round == 1)
    model = BR.fit_pretournament(d, idx, M.TouchModel, season, start, half_life=half_life)
    wp = BR.make_win_prob(model, idx, sd)
    p_winner = np.array([g[4] for g in BR.game_metrics(root, wp)])
    title = B.advancement(root, wp)[id(root)][root.winner]
    return -np.log(p_winner).mean(), (p_winner > 0.5).mean(), title, root.winner


def main():
    d = D.load(list(range(2020, 2026)))
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])
    lam, rho, sd = M.TUNED["touch"]

    print("Tuning (2024 weekly; 2021-2024 tournaments, 252 matches)")
    print(f"{'half-life':>10}{'2024 weekly':>14}{'tourney log loss':>18}{'tourney picks':>15}   champion title odds 2021-24")
    tuned = {}
    for hl in HALF_LIVES:
        weekly = metrics(*(lambda r: (r["p_home"], r["home_won"]))(run_season(d, idx, M.TouchModel, 2024, lam, rho, sd, hl)))["log_loss"]
        res = [tournament(d, idx, s, hl) for s in (2021, 2022, 2023, 2024)]
        ll = np.mean([r[0] for r in res])
        acc = np.mean([r[1] for r in res])
        tuned[hl] = (weekly, ll)
        champs = ", ".join(f"{r[3]} {r[2]:.1%}" for r in res)
        print(f"{str(hl):>10}{weekly:>14.4f}{ll:>18.4f}{acc:>15.1%}   {champs}")

    best = min(tuned, key=lambda h: tuned[h][1] + tuned[h][0])
    print(f"\nChosen half-life: {best} (lowest combined weekly + tournament log loss on the tuning seasons)")

    print("\nTest: 2025, run once")
    for hl in (None, best):
        r = run_season(d, idx, M.TouchModel, 2025, lam, rho, sd, hl)
        m = metrics(r["p_home"], r["home_won"])
        t_ll, t_acc, title, champ = tournament(d, idx, 2025, hl)
        print(f"  half-life {str(hl):>5}: weekly log loss {m['log_loss']:.4f}, accuracy {m['accuracy']:.1%} | "
              f"tournament log loss {t_ll:.4f}, picks {t_acc:.1%}, {champ} title odds {title:.1%}")


if __name__ == "__main__":
    main()
