"""XGBoost baseline vs the rally model, under the same rules.

Both models only use information from before each match:
  * Week by week: retrained every 7 days on matches already played (earlier seasons plus
    this season so far), tuned on 2024, tested once on 2025, on exactly the matches the
    rally model is tested on.
  * Tournaments: each NCAA tournament 2021-2025, predicted from data before the first round.

XGBoost sees one row per match: both teams' season-to-date stats as of that week (record,
strength of schedule, sideout rate, serving, first attack and scramble on both offense and
defense) plus last season's final numbers, the differences between the two teams, and
home court. The rally model sees every rally.

Usage:  python -m analysis.xgb_baseline
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from analysis import bracket_2025 as BR
from analysis.evaluate_models import metrics, run_season
from vbsim import bracket as B
from vbsim import data as D
from vbsim import models as M

TABLES = ("sideout", "serve", "fb", "tr")
SEASONS = list(range(2020, 2026))
WEEK = np.timedelta64(7, "D")


def frames(d):
    """Per-match, per-team count tables as DataFrames."""
    out = {t: pd.DataFrame({k: np.asarray(v) for k, v in d[t].items()}) for t in TABLES}
    m = d["matches"]
    out["matches"] = pd.DataFrame({
        "match_id": np.asarray(m["match_id"]), "season": np.asarray(m["season"]),
        "date": pd.to_datetime(np.asarray(m["date"])), "home": np.asarray(m["home_team"]),
        "away": np.asarray(m["away_team"]), "winner": np.asarray(m["winner"]),
        "neutral": np.ma.getmaskarray(m["venue_home"])})
    for t in TABLES:
        out[t]["season"] = out[t]["match_id"] // 100000
        out[t]["date"] = pd.to_datetime(out[t]["date"])
    return out


def team_stats(f, season, cutoff):
    """Season-to-date stats per team from matches strictly before `cutoff`."""
    def agg(table, team_col, cols):
        x = f[table]
        x = x[(x["season"] == season) & (x["date"] < cutoff)]
        return x.groupby(team_col)[list(cols)].sum()

    so = agg("sideout", "t", ["n", "k"])                          # team receiving
    so_def = agg("sideout", "o", ["n", "k"])                      # team serving
    sv = agg("serve", "t", ["ace", "serr", "inplay"])
    sv_def = agg("serve", "o", ["ace", "serr", "inplay"])
    fb = agg("fb", "t", ["win", "lose", "to_opp"])
    fb_def = agg("fb", "o", ["win", "lose", "to_opp"])
    tr = agg("tr", "t", ["win", "lose", "to_opp"])
    tr_def = agg("tr", "o", ["win", "lose", "to_opp"])

    s = pd.DataFrame(index=sorted(set(so.index) | set(so_def.index)))
    s["sideout"] = so["k"] / so["n"]
    s["serve_win"] = 1 - so_def["k"] / so_def["n"]
    tot = sv.sum(axis=1)
    s["ace"], s["serr"] = sv["ace"] / tot, sv["serr"] / tot
    tot = sv_def.sum(axis=1)
    s["ace_against"], s["serr_drawn"] = sv_def["ace"] / tot, sv_def["serr"] / tot
    for name, t in (("fb", fb), ("tr", tr)):
        tot = t.sum(axis=1)
        s[f"{name}_win"], s[f"{name}_lose"] = t["win"] / tot, t["lose"] / tot
    for name, t in (("fb_def", fb_def), ("tr_def", tr_def)):
        tot = t.sum(axis=1)
        s[f"{name}_win"], s[f"{name}_lose"] = t["win"] / tot, t["lose"] / tot

    m = f["matches"]
    m = m[(m["season"] == season) & (m["date"] < cutoff)]
    games = pd.concat([m[["home", "away", "winner"]].rename(columns={"home": "team", "away": "opp"}),
                       m[["away", "home", "winner"]].rename(columns={"away": "team", "home": "opp"})])
    games["won"] = (games["winner"] == games["team"]).astype(float)
    rec = games.groupby("team")["won"].agg(["mean", "count"]).rename(columns={"mean": "win_pct", "count": "played"})
    s = s.join(rec, how="outer")
    games["opp_pct"] = games["opp"].map(rec["win_pct"])
    s["sos"] = games.groupby("team")["opp_pct"].mean()
    return s


class Features:
    """Cached team stats by (season, cutoff), plus last season's final stats."""

    def __init__(self, f):
        self.f = f
        self.cache = {}
        self.final = {s: team_stats(f, s, pd.Timestamp("2100-01-01")) for s in SEASONS}

    def at(self, season, cutoff):
        key = (season, pd.Timestamp(cutoff).normalize())
        if key not in self.cache:
            cur = team_stats(self.f, season, key[1])
            prev = self.final.get(season - 1)
            if prev is not None:
                cur = cur.join(prev[["win_pct", "sideout", "serve_win", "fb_win", "tr_win"]].add_prefix("prev_"), how="outer")
            self.cache[key] = cur
        return self.cache[key]

    def rows(self, season, cutoff, a, b, home):
        """Feature matrix for matchups a vs b (home: 1 if a is at home, 0 neutral, -1 away)."""
        s = self.at(season, cutoff)
        A = s.reindex(a).reset_index(drop=True).add_prefix("a_")
        Bm = s.reindex(b).reset_index(drop=True).add_prefix("b_")
        diff = pd.DataFrame(A.values - Bm.values, columns=[c.replace("a_", "d_", 1) for c in A.columns])
        X = pd.concat([A, Bm, diff], axis=1)
        X["home"] = np.asarray(home, dtype=float)
        return X


def season_rows(feat, f, season):
    """All of a season's matches as training rows, each with features as of its own week."""
    m = f["matches"][f["matches"]["season"] == season].sort_values("date")
    start = m["date"].min()
    week = ((m["date"] - start) // pd.Timedelta(days=7)).astype(int)
    parts = []
    for k, g in m.groupby(week):
        cutoff = start + pd.Timedelta(days=7 * int(k))
        X = feat.rows(season, cutoff, g["home"].values, g["away"].values, np.where(g["neutral"], 0, 1))
        X["y"] = (g["winner"].values == g["home"].values).astype(int)
        X["match_id"] = g["match_id"].values
        X["week"] = int(k)
        parts.append(X)
    return pd.concat(parts, ignore_index=True)


def train(Xy, params):
    X = Xy.drop(columns=["y", "match_id", "week"])
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                              n_jobs=8, **params)
    model.fit(X, Xy["y"])
    return model


def walk_forward(feat, f, rows, season, params):
    """Weekly retraining through one season; returns P(home win) and outcomes per match."""
    past = pd.concat([rows[s] for s in SEASONS if 2021 <= s < season], ignore_index=True)
    cur = rows[season]
    preds = []
    for k in sorted(cur["week"].unique()):
        model = train(pd.concat([past, cur[cur["week"] < k]], ignore_index=True), params)
        test = cur[cur["week"] == k]
        p = model.predict_proba(test.drop(columns=["y", "match_id", "week"]))[:, 1]
        preds.append(pd.DataFrame({"match_id": test["match_id"].values, "p": p, "y": test["y"].values}))
    return pd.concat(preds, ignore_index=True)


def main():
    d = D.load(SEASONS)
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])
    f = frames(d)
    feat = Features(f)
    rows = {s: season_rows(feat, f, s) for s in SEASONS if s >= 2021}
    print("training rows per season:", {s: len(r) for s, r in rows.items()}, "| features:", rows[2025].shape[1] - 3)

    grid = [dict(max_depth=md, n_estimators=ne, learning_rate=lr, min_child_weight=mc, subsample=0.8, colsample_bytree=0.8)
            for md, ne, lr, mc in itertools.product((2, 3, 4), (200, 500), (0.03, 0.08), (5, 20))]
    print(f"\nTuning XGBoost on 2024 ({len(grid)} settings)...")
    best = None
    for prm in grid:
        r = walk_forward(feat, f, rows, 2024, prm)
        ll = metrics(r["p"].values, r["y"].values)["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, prm)
    print(f"  best 2024 log loss {best[0]:.4f} with {best[1]}")

    # --- 2025 week by week, same matches as the rally model
    lam, rho, sd = M.TUNED["touch"]
    rally = run_season(d, idx, M.TouchModel, 2025, lam, rho, sd)
    rally = pd.DataFrame({"match_id": rally["match_id"], "p_rally": rally["p_home"], "y": rally["home_won"].astype(int)})
    xg = walk_forward(feat, f, rows, 2025, best[1]).rename(columns={"p": "p_xgb"}).drop(columns="y")
    both = rally.merge(xg, on="match_id")
    print(f"\n=== 2025, week by week ({len(both)} matches) ===")
    print(f"{'':<14}{'log loss':>10}{'brier':>9}{'accuracy':>10}")
    for name, col in (("rally model", "p_rally"), ("XGBoost", "p_xgb")):
        m = metrics(both[col].values, both["y"].values)
        print(f"{name:<14}{m['log_loss']:>10.4f}{m['brier']:>9.4f}{m['accuracy']:>10.1%}")
    ll = lambda p, y: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    diff = ll(both["p_xgb"].values, both["y"].values) - ll(both["p_rally"].values, both["y"].values)
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    print(f"XGBoost minus rally log loss: {diff.mean():+.4f} (95% interval {np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}; negative favors XGBoost)")
    both["early"] = both["match_id"].isin(rows[2025][rows[2025]["week"] < 3]["match_id"])
    for label, mask in (("first 3 weeks", both["early"]), ("week 4 onward", ~both["early"])):
        a = metrics(both.loc[mask, "p_rally"].values, both.loc[mask, "y"].values)["log_loss"]
        b = metrics(both.loc[mask, "p_xgb"].values, both.loc[mask, "y"].values)["log_loss"]
        print(f"  {label:<14} rally {a:.4f} | XGBoost {b:.4f}")
    edges = np.linspace(0, 1, 6)
    print("calibration (predicted -> actual):")
    for name, col in (("rally", "p_rally"), ("XGBoost", "p_xgb")):
        b = np.clip(np.digitize(both[col], edges) - 1, 0, 4)
        print(f"  {name:<8}" + "  ".join(f"{both[col][b == i].mean():.0%}->{both['y'][b == i].mean():.0%}" for i in range(5) if (b == i).sum() > 30))

    # --- Tournaments 2021-2025, from data before the first round
    print("\n=== NCAA tournaments, predicted before the first round ===")
    print(f"{'season':<8}{'games':>6}{'rally ll':>10}{'XGB ll':>9}{'rally acc':>11}{'XGB acc':>9}")
    tot = {"r": [], "x": []}
    for s in range(2021, 2026):
        root = B.rebuild(D.DB, s)
        start_date = min(n.date for n in B.walk(root) if n.round == 1)
        start = pd.Timestamp(start_date)
        model = BR.fit_pretournament(d, idx, M.TouchModel, s, start_date)
        wp = BR.make_win_prob(model, idx, sd)
        past = pd.concat([rows[y] for y in SEASONS if 2021 <= y < s], ignore_index=True) if s > 2021 else rows[2021].iloc[:0]
        cur = rows[s]
        cur_dates = f["matches"].set_index("match_id").loc[cur["match_id"], "date"]
        xmodel = train(pd.concat([past, cur[(cur_dates < start).values]], ignore_index=True), best[1])
        games = []
        for n in B.walk(root):
            a = n.left.winner if isinstance(n.left, B.Node) else n.left
            b = n.right.winner if isinstance(n.right, B.Node) else n.right
            h = 1 if n.host == a else -1 if n.host == b else 0
            games.append((a, b, h, n.winner == a, wp([a], [b], n.host)[0, 0]))
        a, b, h, awon, pr = map(np.array, zip(*games))
        px = xmodel.predict_proba(feat.rows(s, start, a, b, h).drop(columns=[], errors="ignore"))[:, 1]
        y = awon.astype(int)
        mr, mx = metrics(pr, y), metrics(px, y)
        tot["r"].append(ll(pr, y)); tot["x"].append(ll(px, y))
        print(f"{s:<8}{len(y):>6}{mr['log_loss']:>10.4f}{mx['log_loss']:>9.4f}{mr['accuracy']:>11.1%}{mx['accuracy']:>9.1%}")
    r, x = np.concatenate(tot["r"]), np.concatenate(tot["x"])
    print(f"{'all':<8}{len(r):>6}{r.mean():>10.4f}{x.mean():>9.4f}")


if __name__ == "__main__":
    main()
