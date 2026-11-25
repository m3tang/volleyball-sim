"""Compare the sideout model and the touch model on match prediction.

Protocol (no peeking at the future):
  * Each season starts from last season's final team strengths, multiplied by rho
    (rho < 1 pulls teams back toward average to account for roster turnover).
  * Every 7 days the model is refitted on that season's matches played so far, then
    predicts the next 7 days' matches.
  * Settings (lam = shrinkage, rho = carryover) are tuned on 2024, then the chosen
    settings are run once on 2025 as the test.

Usage:  python analysis/evaluate_models.py
"""
import itertools
import time

import numpy as np

from vbsim import data as D
from vbsim import models as M
from vbsim.matchsim import match_win_prob

TABLES = ("sideout", "serve", "fb", "tr")
LAMS = (3, 10, 30, 100)
RHOS = (0.7, 0.85, 1.0)
WEEK = np.timedelta64(7, "D")


def season_of(match_ids):
    return np.asarray(match_ids) // 100000


def run_season(d, idx, cls, season, lam, rho, form_sd=0.0):
    """Weekly-refit predictions for one season. Returns per-match and per-rally results.

    Per match it keeps both teams' predicted sideout rates (so_home, so_away), so match
    odds can be recomputed for other form_sd values without refitting."""
    n = len(idx)
    in_season = {t: season_of(d[t]["match_id"]) == season for t in TABLES}
    in_prev = {t: season_of(d[t]["match_id"]) == season - 1 for t in TABLES}
    prior = cls(n, lam).fit(d, in_prev, idx).scaled(rho)

    mt = d["matches"]
    m_mask = season_of(mt["match_id"]) == season
    m_dates = mt["date"][m_mask]
    start, end = m_dates.min(), m_dates.max()

    out = {"p_home": [], "home_won": [], "date": [], "match_id": [], "so_home": [], "so_away": [],
           "row_match": [], "row_t": [], "row_n": [], "row_k": [], "row_p": [],
           "rally_ll": 0.0, "rally_n": 0}
    model, cutoff = prior, start
    while cutoff <= end:
        train = {t: in_season[t] & (d[t]["date"] < cutoff) for t in TABLES}
        if train["sideout"].any():
            model = cls(n, lam).fit(d, train, idx, prior=prior,
                                    warm=model if model is not prior else None)
        # Matches this week.
        wk = m_mask & (mt["date"] >= cutoff) & (mt["date"] < cutoff + WEEK)
        if wk.any():
            home, away = mt["home_team"][wk], mt["away_team"][wk]
            venue = np.ma.getdata(mt["venue_home"][wk])
            neutral = np.ma.getmaskarray(mt["venue_home"][wk])
            h = np.where(neutral, 0.0, np.where(venue == home, 1.0, -1.0))
            so_home = model.sideout(idx(home), idx(away), h)
            so_away = model.sideout(idx(away), idx(home), -h)
            out["p_home"].append(match_win_prob(1 - so_away, so_home, form_sd))
            out["home_won"].append(mt["winner"][wk] == home)
            out["date"].append(mt["date"][wk])
            out["match_id"].append(mt["match_id"][wk])
            out["so_home"].append(so_home)
            out["so_away"].append(so_away)
        # Rallies this week.
        s = d["sideout"]
        rw = in_season["sideout"] & (s["date"] >= cutoff) & (s["date"] < cutoff + WEEK)
        if rw.any():
            p = model.sideout(idx(s["t"][rw]), idx(s["o"][rw]), s["home"][rw])
            k, nn = s["k"][rw], s["n"][rw]
            out["rally_ll"] += float((k * np.log(p) + (nn - k) * np.log(1 - p)).sum())
            out["rally_n"] += int(nn.sum())
            for key, val in (("row_match", s["match_id"][rw]), ("row_t", s["t"][rw]),
                             ("row_n", nn), ("row_k", k), ("row_p", p)):
                out[key].append(val)
        cutoff = cutoff + WEEK
    for key in ("p_home", "home_won", "date", "match_id", "so_home", "so_away",
                "row_match", "row_t", "row_n", "row_k", "row_p"):
        out[key] = np.concatenate(out[key])
    out["week"] = ((out["date"] - start) // WEEK).astype(int)
    return out


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = y.astype(float)
    return {"matches": len(p),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "accuracy": float(((p > 0.5) == (y == 1)).mean())}


def summarize(name, r, mask=None):
    mask = np.ones(len(r["p_home"]), bool) if mask is None else mask
    m = metrics(r["p_home"][mask], r["home_won"][mask])
    return (f"{name:<34} {m['matches']:>7} {m['log_loss']:>9.4f} {m['brier']:>8.4f} {m['accuracy']:>8.1%}")


def calibration(r, bins=10):
    lines = ["  predicted     n   actual"]
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(r["p_home"], edges) - 1, 0, bins - 1)
    for i in range(bins):
        sel = b == i
        if sel.sum() >= 20:
            lines.append(f"  {r['p_home'][sel].mean():>8.1%} {sel.sum():>6} {r['home_won'][sel].mean():>8.1%}")
    return "\n".join(lines)


def main():
    t0 = time.time()
    d = D.load([2023, 2024, 2025])
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])
    print(f"loaded in {time.time() - t0:.0f}s; {len(idx)} teams")

    chosen = {}
    for cls in (M.SideoutModel, M.TouchModel):
        print(f"\nTuning {cls.name} model on 2024 (match log loss, lower is better):")
        best = None
        for lam, rho in itertools.product(LAMS, RHOS):
            r = run_season(d, idx, cls, 2024, lam, rho)
            m = metrics(r["p_home"], r["home_won"])
            rally = -r["rally_ll"] / r["rally_n"]
            print(f"  lam={lam:<5} rho={rho:<4} match={m['log_loss']:.4f}  rally={rally:.5f}  acc={m['accuracy']:.1%}")
            if best is None or m["log_loss"] < best[0]:
                best = (m["log_loss"], lam, rho)
        chosen[cls] = best[1:]
        print(f"  -> chosen lam={best[1]}, rho={best[2]}")

    print("\n=== Test: 2025 season, each model run once with its tuned settings ===")
    header = f"{'':<34} {'matches':>7} {'log loss':>9} {'brier':>8} {'accuracy':>8}"
    results = {}
    for cls, (lam, rho) in chosen.items():
        results[cls.name] = run_season(d, idx, cls, 2025, lam, rho)
    any_r = next(iter(results.values()))
    late = any_r["week"] >= 3
    december = (any_r["date"].astype("datetime64[M]").astype(int) % 12) == 11
    print(header)
    print(f"{'coin flip':<34} {len(any_r['p_home']):>7} {np.log(2):>9.4f} {0.25:>8.4f} {'50.0%':>8}")
    for name, r in results.items():
        print(summarize(f"{name}: all matches", r))
        print(summarize(f"{name}: week 4 onward", r, late))
        print(summarize(f"{name}: December (postseason)", r, december))
        print(f"{name}: rally log loss {-r['rally_ll'] / r['rally_n']:.5f} over {r['rally_n']:,} rallies")
    # Is the gap between the models bigger than noise? Paired bootstrap over test matches.
    def per_match_ll(r):
        p, y = np.clip(r["p_home"], 1e-6, 1 - 1e-6), r["home_won"].astype(float)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = per_match_ll(results["touch"]) - per_match_ll(results["sideout"])
    rng = np.random.default_rng(0)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nTouch minus sideout match log loss: {diff.mean():+.4f} "
          f"(95% interval {lo:+.4f} to {hi:+.4f}; negative favors the touch model)")

    for name, r in results.items():
        print(f"\nCalibration, {name} model, 2025 (home/first-listed team win probability):")
        print(calibration(r))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
