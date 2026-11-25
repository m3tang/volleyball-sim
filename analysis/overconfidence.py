"""Why are the match predictions overconfident, and how much should we widen them?

1. Momentum test. After the serving team wins k points in a row, is the receiving team less
   likely to win the next rally than it otherwise would be? Real sets are compared with
   simulated sets that have no momentum by construction (each rally independent). The
   simulations replay every real set with the same teams and first server, so streak
   selection and small-sample effects (the "hot hand" counting bias) hit both equally.
     null A: each team's actual sideout rate in that match (includes match-day form)
     null B: each team's predicted sideout rate from before the match (no form)
     null C: null B plus a random match-day form swing of the measured size (FORM_SD_CHECK)
2. Match-day form. How much does a team's sideout rate vary from match to match beyond what
   rally-to-rally chance explains? Estimated from the covariance of the two teams' residuals
   in the same match: when one team is "on", the other's sideout rate drops.
3. Fix. Add form_sd (see vbsim.matchsim.match_win_prob), tune it with lam and rho on 2024,
   test once on 2025.

Usage:  python -m analysis.overconfidence
"""
import itertools

import duckdb
import numpy as np

from analysis.evaluate_models import RHOS, metrics, run_season, calibration
from vbsim import data as D
from vbsim import models as M
from vbsim.matchsim import match_win_prob

TABLES = ("sideout", "serve", "fb", "tr")
FORM_SDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
LAMS = (3, 10, 30, 100)
RUN_BINS = (1, 2, 3, 4, 5, 6)  # 6 means 6+
FORM_SD_CHECK = 0.127  # from the residual covariance below


def predicted_rates(runs):
    """{(match_id, team): pre-match predicted sideout rate} from run_season outputs."""
    out = {}
    for r in runs:
        for m, t, p in zip(r["row_match"], r["row_t"], r["row_p"]):
            out[(int(m), t)] = float(p)
    return out


def load_sets(seasons):
    con = duckdb.connect(str(D.DB), read_only=True)
    rallies = con.execute(f"""
      WITH r AS (
        SELECT ra.match_id, ra.set_no, ra.rally_no, ra.serve_team, ra.receive_team, ra.sideout,
               ra.point_team, ra.usable
        FROM rallies ra JOIN matches m USING (match_id)
        WHERE m.season IN ({", ".join(map(str, seasons))}) AND m.clean AND m.n_point_summary_rows = 0),
      s AS (
        SELECT *, sum(chg) OVER (PARTITION BY match_id, set_no ORDER BY rally_no ROWS UNBOUNDED PRECEDING) AS streak
        FROM (SELECT *, (lag(point_team) OVER (PARTITION BY match_id, set_no ORDER BY rally_no)
                         IS DISTINCT FROM point_team)::INT AS chg FROM r)),
      t AS (SELECT *, row_number() OVER (PARTITION BY match_id, set_no, streak ORDER BY rally_no) AS pos FROM s)
      SELECT match_id, set_no, serve_team, receive_team, sideout,
             lag(pos) OVER w AS run_before, lag(point_team) OVER w AS prev_winner, usable
      FROM t WINDOW w AS (PARTITION BY match_id, set_no ORDER BY rally_no)
      ORDER BY match_id, set_no, rally_no""").fetchnumpy()
    sets = con.execute(f"""
      SELECT r.match_id, r.set_no, arg_min(r.serve_team, r.rally_no) AS first_server,
             any_value(m.away_team) AS away, any_value(m.home_team) AS home
      FROM rallies r JOIN matches m USING (match_id)
      WHERE m.season IN ({", ".join(map(str, seasons))}) AND m.clean AND m.n_point_summary_rows = 0
        AND r.serve_team IS NOT NULL
      GROUP BY r.match_id, r.set_no ORDER BY 1, 2""").fetchnumpy()
    in_match = con.execute(f"""
      SELECT r.match_id, r.receive_team, avg(r.sideout::INT) AS rate
      FROM rallies r JOIN matches m USING (match_id)
      WHERE m.season IN ({", ".join(map(str, seasons))}) AND m.clean AND m.n_point_summary_rows = 0 AND r.usable
      GROUP BY ALL""").fetchall()
    con.close()
    return rallies, sets, {(int(m), t): rate for m, t, rate in in_match}


def run_table(run_before, sideout):
    rb = np.minimum(np.asarray(run_before), RUN_BINS[-1])
    so = np.asarray(sideout, dtype=float)
    return {b: (int((rb == b).sum()), float(so[rb == b].mean())) for b in RUN_BINS}


def simulate_sets(set_no, p_recv_vs_first, p_first_vs_recv, rng):
    """Simulate many sets at once with independent rallies.

    p_recv_vs_first: P(the team receiving first serve wins a rally when the other team serves).
    p_first_vs_recv: P(the first server's team wins a rally when it is receiving).
    Returns (run_before, sideout) for every rally after the first in each set.
    """
    n = len(set_no)
    target = np.where(set_no == 5, 15, 25)
    a = np.zeros(n, int)            # first server's team
    b = np.zeros(n, int)
    a_serving = np.ones(n, bool)
    run = np.zeros(n, int)          # current server's consecutive points (0 before first point)
    active = np.ones(n, bool)
    out_run, out_so = [], []
    while active.any():
        idx = np.flatnonzero(active)
        p_sideout = np.where(a_serving[idx], p_recv_vs_first[idx], p_first_vs_recv[idx])
        so = rng.random(len(idx)) < p_sideout
        has_run = run[idx] > 0
        out_run.append(run[idx][has_run])
        out_so.append(so[has_run])
        # receiver wins -> serve changes, run restarts at 1; server wins -> run grows
        a_wins = np.where(a_serving[idx], ~so, so)
        a[idx] += a_wins
        b[idx] += ~a_wins
        new_serving = a_wins
        run[idx] = np.where(so, 1, run[idx] + 1)
        a_serving[idx] = new_serving
        done = ((a[idx] >= target[idx]) | (b[idx] >= target[idx])) & (np.abs(a[idx] - b[idx]) >= 2)
        active[idx[done]] = False
    return np.concatenate(out_run), np.concatenate(out_so)


def momentum_test(seasons, pred):
    rallies, sets, in_match = load_sets(seasons)
    ok = (rallies["usable"] & (rallies["prev_winner"] == rallies["serve_team"])
          & ~np.ma.getmaskarray(rallies["run_before"]))
    actual = run_table(np.ma.getdata(rallies["run_before"])[ok], rallies["sideout"][ok])

    m_ids = sets["match_id"].astype(int)
    first, away, home = sets["first_server"], sets["away"], sets["home"]
    other = np.where(first == away, home, away)

    def rates(lookup):
        r_vs_f = np.array([lookup.get((m, t), np.nan) for m, t in zip(m_ids, other)])
        f_vs_r = np.array([lookup.get((m, t), np.nan) for m, t in zip(m_ids, first)])
        keep = np.isfinite(r_vs_f) & np.isfinite(f_vs_r)
        return r_vs_f, f_vs_r, keep

    rng = np.random.default_rng(0)
    logit = lambda p: np.log(p) - np.log1p(-p)
    expit = lambda z: 1 / (1 + np.exp(-z))
    sims = {}
    for name, lookup, form in (("null A: in-match rates", in_match, 0.0),
                               ("null B: pre-match model", pred, 0.0),
                               ("null C: pre-match + form", pred, FORM_SD_CHECK)):
        r_vs_f, f_vs_r, keep = rates(lookup)
        rb, so = [], []
        for _ in range(3):
            r_p, f_p = r_vs_f[keep], f_vs_r[keep]
            if form > 0:
                # One form draw per match, shared by all its sets.
                u, inv = np.unique(m_ids[keep], return_inverse=True)
                dlt = rng.normal(0, form, len(u))[inv]
                r_p, f_p = expit(logit(r_p) + dlt), expit(logit(f_p) - dlt)
            x, y = simulate_sets(sets["set_no"][keep], r_p, f_p, rng)
            rb.append(x)
            so.append(y)
        sims[name] = run_table(np.concatenate(rb), np.concatenate(so))

    print("Receiving team's rally win rate after the server has won k points in a row:")
    print(f"  {'k':>3} {'rallies':>9} {'actual':>8}" + "".join(f"{n:>26}" for n in sims))
    for b in RUN_BINS:
        label = f"{b}+" if b == RUN_BINS[-1] else str(b)
        n, rate = actual[b]
        print(f"  {label:>3} {n:>9,} {rate:>8.1%}" + "".join(f"{s[b][1]:>26.1%}" for s in sims.values()))
    return actual, sims


def form_variance(runs):
    """Match-day form from the covariance of the two teams' residuals in each match."""
    rows = {}
    for r in runs:
        for m, t, n, k, p in zip(r["row_match"], r["row_t"], r["row_n"], r["row_k"], r["row_p"]):
            rows.setdefault(int(m), []).append((n, k, p))
    e_a, e_b, noise = [], [], []
    for vals in rows.values():
        if len(vals) != 2:
            continue
        (n1, k1, p1), (n2, k2, p2) = vals
        # (k - np) / (n p (1-p)) is roughly the logit-scale deviation from the prediction.
        e_a.append((k1 - n1 * p1) / (n1 * p1 * (1 - p1)))
        e_b.append((k2 - n2 * p2) / (n2 * p2 * (1 - p2)))
        noise += [1 / (n1 * p1 * (1 - p1)), 1 / (n2 * p2 * (1 - p2))]
    e_a, e_b = np.array(e_a), np.array(e_b)
    cov = np.cov(e_a, e_b)[0, 1]
    extra = np.var(np.concatenate([e_a, e_b])) - np.mean(noise)
    print(f"\nMatch-day form, {len(e_a):,} matches (logit scale):")
    print(f"  covariance of the two teams' residuals: {cov:+.4f}  -> form_sd ~ {np.sqrt(max(-cov, 0)):.3f}")
    print(f"  extra variance per team beyond rally chance: {extra:.4f} (sd {np.sqrt(max(extra, 0)):.3f})")


def tune_and_test(d, idx):
    chosen = {}
    for cls in (M.TouchModel, M.SideoutModel):
        best = None
        for lam, rho in itertools.product(LAMS, RHOS):
            r = run_season(d, idx, cls, 2024, lam, rho)
            for sd in FORM_SDS:
                p = match_win_prob(1 - r["so_away"], r["so_home"], sd)
                ll = metrics(p, r["home_won"])["log_loss"]
                if best is None or ll < best[0]:
                    best = (ll, lam, rho, sd)
        chosen[cls.name] = best[1:]
        print(f"\n{cls.name} model, tuned on 2024: lam={best[1]}, rho={best[2]}, form_sd={best[3]} "
              f"(2024 log loss {best[0]:.4f})")

    print("\n=== Test: 2025 ===")
    print(f"{'':<36}{'log loss':>9}{'brier':>8}{'accuracy':>9}")
    out = {}
    for cls in (M.TouchModel, M.SideoutModel):
        # "Before" = the settings chosen without form: touch (10, 0.85), sideout (30, 1.0).
        lam_old, rho_old = {"touch": (10, 0.85), "sideout": (30, 1.0)}[cls.name]
        old = run_season(d, idx, cls, 2025, lam_old, rho_old)
        lam, rho, sd = chosen[cls.name]
        new = run_season(d, idx, cls, 2025, lam, rho, form_sd=sd)
        for label, r in ((f"{cls.name}: before (no form)", old), (f"{cls.name}: with form_sd={sd}", new)):
            m = metrics(r["p_home"], r["home_won"])
            print(f"{label:<36}{m['log_loss']:>9.4f}{m['brier']:>8.4f}{m['accuracy']:>9.1%}")
        out[cls.name] = (old, new)

    old, new = out["touch"]
    ll = lambda r: -(r["home_won"] * np.log(r["p_home"]) + (1 - r["home_won"]) * np.log(1 - r["p_home"]))
    diff = ll(new) - ll(old)
    rng = np.random.default_rng(0)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nTouch model, with form minus without: {diff.mean():+.4f} log loss "
          f"(95% interval {lo:+.4f} to {hi:+.4f}; negative = better)")
    print("\nCalibration, touch model with form, 2025:")
    print(calibration(new))
    return chosen


def main():
    d = D.load([2023, 2024, 2025])
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])

    lam, rho = 10, 0.85  # touch settings without form, so the diagnostics see the old predictions
    runs = [run_season(d, idx, M.TouchModel, s, lam, rho) for s in (2024, 2025)]
    momentum_test([2024, 2025], predicted_rates(runs))
    form_variance(runs)
    chosen = tune_and_test(d, idx)
    print("\nSuggested TUNED:", {k: v for k, v in chosen.items()})


if __name__ == "__main__":
    main()
