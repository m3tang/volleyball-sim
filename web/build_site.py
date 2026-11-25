"""Export everything the public page needs, then write site/index.html.

  web/template.html  the page (HTML, CSS, JS) with a __DATA__ placeholder
  web/data.json       what this script computes (also inlined into index.html)
  web/index.html      standalone page with data inline: double-click it or host it anywhere
  web/artifact.html   the same page for the Claude artifact (loads data.json beside it)

Usage:  python -m web.build_site               (refit and rebuild, about 2 minutes)
        python -m web.build_site --page-only   (template changes only, instant)
"""
import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np

from analysis import bracket_2025 as BR
from analysis.evaluate_models import metrics, run_season
from analysis.overconfidence import momentum_test, predicted_rates
from vbsim import bracket as B
from vbsim import data as D
from vbsim import models as M
from vbsim.matchsim import match_win_prob

HERE = Path(__file__).resolve().parent
SEASONS = list(range(2020, 2026))
TABLES = ("sideout", "serve", "fb", "tr")
MIN_MATCHES = 10
TOP_EARLIER = 64  # earlier seasons keep their top teams only (for dream matchups)


def r4(x):
    return [round(float(v), 4) for v in np.ravel(x)]


def r3(x):
    return [round(float(v), 3) for v in np.ravel(x)]


def raw_counts(d, season):
    """What each team actually did at each step in its matches, before any adjustment.
    Returns {team: {"rate": 12 raw rates in the same order as the model's numbers, "n": 6 counts}}."""
    from collections import defaultdict
    acc = defaultdict(lambda: np.zeros(18))
    # columns: 0-2 serving (ace, serr, total) | 3-5 receiving serve (opp ace, opp serr, total)
    #          6-8 first attack (win, lose, total) | 9-11 opponents' first attack | 12-14 scramble | 15-17 opp scramble
    for table, cols, own, opp in (("serve", ("ace", "serr", "inplay"), 0, 3), ("fb", ("win", "lose", "to_opp"), 6, 9),
                                  ("tr", ("win", "lose", "to_opp"), 12, 15)):
        t = d[table]
        mask = (t["match_id"] // 100000) == season
        a, b, c = (np.asarray(t[k])[mask] for k in cols)
        for team_col, base in (("t", own), ("o", opp)):
            for name, x, y, z in zip(np.asarray(t[team_col])[mask], a, b, c):
                v = acc[name]
                v[base] += x; v[base + 1] += y; v[base + 2] += x + y + z
    out = {}
    for name, v in acc.items():
        rates, ns = [], []
        for base in (0, 3, 6, 9, 12, 15):
            n = max(v[base + 2], 1)
            rates += [v[base] / n, v[base + 1] / n]
            ns.append(int(v[base + 2]))
        out[name] = {"rate": [round(float(r), 4) for r in rates], "n": ns}
    return out


def season_ratings(d, idx, con):
    """End-of-season touch-model strengths for every season, chained through the priors."""
    lam, rho, form_sd = M.TUNED["touch"]
    season_of = {t: d[t]["match_id"] // 100000 for t in TABLES}
    out, prev = {}, None
    for s in SEASONS:
        mask = {t: season_of[t] == s for t in TABLES}
        model = M.TouchModel(len(idx), lam).fit(d, mask, idx, prior=prev.scaled(rho) if prev else None)
        prev = model
        rec = dict((t, (w, l)) for t, w, l in con.execute("""
          SELECT t, count(*) FILTER (WHERE won), count(*) FILTER (WHERE NOT won) FROM (
            SELECT away_team t, winner = away_team won FROM matches WHERE season = ? AND winner IS NOT NULL
            UNION ALL SELECT home_team, winner = home_team FROM matches WHERE season = ? AND winner IS NOT NULL)
          GROUP BY t""", [s, s]).fetchall())
        names = sorted(t for t, (w, l) in rec.items() if w + l >= MIN_MATCHES and t in idx.ids)
        raw = raw_counts(d, s)
        ids = idx(names)
        avg = np.full(len(ids), len(idx))  # an index past every team = an average team
        # Rating: share of rallies won against a league-average team on a neutral floor
        # (half as receiver, half as server).
        padded = M.TouchModel(len(idx) + 1, lam)
        padded.parts = {k: type(p)(p.intercept, p.home, np.vstack([p.off, np.zeros((1, p.off.shape[1]))]),
                                   np.vstack([p.def_, np.zeros((1, p.def_.shape[1]))]))
                        for k, p in model.parts.items()}
        zero = np.zeros(len(ids))
        so_t = padded.sideout(ids, avg, zero)
        so_avg = padded.sideout(avg, ids, zero)
        rating = 0.5 * (so_t + (1 - so_avg))
        teams = []
        for name, i, rt in zip(names, ids, rating):
            p = model.parts
            teams.append({"n": name, "w": rec[name][0], "l": rec[name][1], "r": round(float(rt), 4),
                          "p": r3([p["serve"].off[i], p["serve"].def_[i], p["fb"].off[i], p["fb"].def_[i],
                                   p["tr"].off[i], p["tr"].def_[i]]),
                          "raw": raw.get(name)})
        teams.sort(key=lambda t: -t["r"])
        for rank, t in enumerate(teams, 1):
            t["rank"] = rank
        if s != SEASONS[-1]:
            teams = teams[:TOP_EARLIER]
        out[s] = {"teams": teams,
                  "b": {k: r4(p.intercept) for k, p in model.parts.items()},
                  "h": {k: r4(p.home) for k, p in model.parts.items()}}
        print(f"  {s}: {len(teams)} teams, top: {teams[0]['n']}")
    return out


def champions(con):
    rows = con.execute("""SELECT season, winner, CASE WHEN winner = away_team THEN home_team ELSE away_team END
      FROM matches WHERE (season, date) IN (SELECT (season, max(date)) FROM matches GROUP BY season)""").fetchall()
    return {s: {"champ": w, "runner_up": l} for s, w, l in rows}


# Order of the per-team numbers in a match breakdown, all from the team's side when it is
# receiving serve: sideout rate, aces against, opponent service errors, first-attack scoring,
# first-attack errors or blocked, scramble scoring (per possession after a dig).
BREAKDOWN = ("sideout", "aces_against", "serve_errors_for", "first_attack_score", "first_attack_lose", "scramble_score")


def match_counts(d, con, season):
    """Per-match counts for one season, keyed for quick lookup, plus block points."""
    counts = {}
    for table, cols in (("sideout", ("k", "n")), ("serve", ("ace", "serr", "inplay")),
                        ("fb", ("win", "lose", "to_opp")), ("tr", ("win", "lose", "to_opp"))):
        t = d[table]
        for i in np.flatnonzero((t["match_id"] // 100000) == season):
            counts[(table, int(t["match_id"][i]), t["t"][i])] = [float(t[c][i]) for c in cols]
    blocks = {(int(m), team): int(n) for m, team, n in con.execute("""
        SELECT match_id, point_team, count(*) FROM rallies
        WHERE season = ? AND usable AND end_event = 'Block' GROUP BY ALL""", [season]).fetchall()}
    matches = {(dt_, frozenset((a, h))): int(mid) for mid, dt_, a, h in con.execute(
        "SELECT match_id, date, away_team, home_team FROM matches WHERE season = ?", [season]).fetchall()}
    return counts, blocks, matches


def breakdown(model, idx, a, b, host, mid, counts, blocks):
    """Expected (pre-tournament model, this matchup and venue) vs actual, for both teams."""
    out = {"exp": {}, "act": {}, "n": {}, "blk": {}}
    for T, O in ((a, b), (b, a)):
        h = np.array([1.0 if host == T else -1.0 if host == O else 0.0])
        ti, oi = idx([T]), idx([O])
        srv = model.parts["serve"].probs(oi, ti, -h)[0]
        fb = model.parts["fb"].probs(ti, oi, h)[0]
        tr = model.parts["tr"].probs(ti, oi, h)[0]
        so = model.sideout(ti, oi, h)[0]
        keys = [("sideout", T), ("serve", O), ("fb", T), ("tr", T)]
        got = [counts.get((tbl, mid, team)) for tbl, team in keys]
        if any(g is None for g in got):
            return None
        (k, n), (ace, serr, inplay), (fw, fl, fo), (tw, tl, to) = got
        serves, fbn, trn = ace + serr + inplay, fw + fl + fo, tw + tl + to
        if min(n, serves, fbn) == 0:
            return None
        out["exp"][T] = r4([so, srv[0], srv[1], fb[0], fb[1], tr[0]])
        out["act"][T] = r4([k / n, ace / serves, serr / serves, fw / fbn, fl / fbn, tw / max(trn, 1)])
        out["n"][T] = [int(n), int(serves), int(serves), int(fbn), int(fbn), int(trn)]
        out["blk"][T] = blocks.get((mid, T), 0)
    return out


def bracket_export(d, idx, con, season):
    root = B.rebuild(D.DB, season)
    start = min(n.date for n in B.walk(root) if n.round == 1)
    record = BR.record_before(con, start, season)
    lam, rho, form_sd = M.TUNED["touch"]
    model = BR.fit_pretournament(d, idx, M.TouchModel, season, start)
    wp = BR.make_win_prob(model, idx, form_sd)
    adv = B.advancement(root, wp)
    picks = B.most_likely_bracket(root, adv)
    base = BR.baseline_picks(root, record)
    wins = BR.furthest(root)
    odds = {t: [0.0] * 6 for t in root.teams}
    for n in B.walk(root):
        for t, p in adv[id(n)].items():
            odds[t][n.round - 1] = p

    counts, blocks, mindex = match_counts(d, con, season)
    found = [0, 0]

    def ser(node):
        kids = [ser(c) if isinstance(c, B.Node) else c for c in (node.left, node.right)]
        a = node.left.winner if isinstance(node.left, B.Node) else node.left
        b = node.right.winner if isinstance(node.right, B.Node) else node.right
        mid = mindex.get((node.date, frozenset((a, b))))
        x = breakdown(model, idx, a, b, node.host, mid, counts, blocks) if mid else None
        found[0] += x is not None
        found[1] += 1
        return {"r": node.round, "host": node.host, "w": node.winner, "pick": picks[id(node)],
                "a": a, "b": b, "pa": round(float(wp([a], [b], node.host)[0, 0]), 4),
                "score": node.score, "x": x, "k": kids}

    games = BR.game_metrics(root, wp)
    base_right = 0
    for rnd, a, b, w, _ in games:
        fav = a if record.get(a, (0, 0))[0] >= record.get(b, (0, 0))[0] else b
        base_right += fav == w
    pw = np.array([g[4] for g in games])
    return {
        "tree": ser(root),
        "breakdowns": f"{found[0]}/{found[1]}",
        "teams": [{"n": t, "odds": r4(odds[t]), "wins": wins[t],
                   "rec": [round(record.get(t, (0, 0))[0], 3), record.get(t, (0, 0))[1]]}
                  for t in sorted(odds, key=lambda t: -odds[t][5])],
        "games": [{"r": g[0], "a": g[1], "b": g[2], "w": g[3], "p": round(float(g[4]), 4)} for g in games],
        "score": {"model": BR.score_bracket(root, picks), "record": BR.score_bracket(root, base),
                  "max": 1920, "model_champ": picks[id(root)], "record_champ": base[id(root)]},
        "acc": {"model": float((pw > 0.5).mean()), "record": base_right / len(games),
                "log_loss": float(-np.log(pw).mean()), "games": len(games)},
        "start": str(start),
        "champion": root.winner,
    }


def calibration_export(d, idx):
    lam, rho, form_sd = M.TUNED["touch"]
    r = run_season(d, idx, M.TouchModel, 2025, lam, rho, form_sd=form_sd)
    m = metrics(r["p_home"], r["home_won"])
    edges = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(r["p_home"], edges) - 1, 0, 9)
    bins = [{"pred": round(float(r["p_home"][b == i].mean()), 4), "actual": round(float(r["home_won"][b == i].mean()), 4),
             "n": int((b == i).sum())} for i in range(10) if (b == i).sum() >= 20]
    return {"bins": bins, "log_loss": m["log_loss"], "accuracy": m["accuracy"], "matches": m["matches"]}


def momentum_export(d, idx):
    runs = [run_season(d, idx, M.TouchModel, s, 10, 0.85) for s in (2024, 2025)]
    actual, sims = momentum_test([2024, 2025], predicted_rates(runs))
    keys = {"null B: pre-match model": "no_form", "null C: pre-match + form": "form", "null A: in-match rates": "in_match"}
    return {"k": list(actual), "n": [actual[k][0] for k in actual], "actual": [actual[k][1] for k in actual],
            **{keys[name]: [s[k][1] for k in actual] for name, s in sims.items()}}


def first_contact(con):
    rows = con.execute("""
      WITH r AS (
        SELECT ra.*, coalesce(a.n_attacks, 0) AS n_attacks
        FROM rallies ra JOIN matches m USING (match_id)
        LEFT JOIN (SELECT match_id, set_no, rally_no, count(*) AS n_attacks FROM events
                   WHERE event = 'Attack' GROUP BY ALL) a USING (match_id, set_no, rally_no)
        WHERE ra.usable AND m.clean AND m.n_point_summary_rows = 0)
      SELECT CASE end_event WHEN 'Ace' THEN 'ace' WHEN 'Service error' THEN 'serr'
               WHEN 'First ball kill' THEN 'fbk'
               ELSE CASE WHEN n_attacks <= 1 THEN 'first_attack_fails' ELSE 'long' END END AS k,
             count(*) AS n, avg(sideout::INT) AS so
      FROM r GROUP BY 1""").fetchall()
    total = sum(n for _, n, _ in rows)
    return {k: {"share": round(n / total, 4), "receiver_wins": round(so, 4)} for k, n, so in rows}


def elite_edge(d, idx):
    """Where do each season's top 10 teams gain their rally edge over an average team?
    Turn on one group of a team's strengths at a time (the rest at league average)."""
    lam, rho, _ = M.TUNED["touch"]
    season_of = {t: d[t]["match_id"] // 100000 for t in TABLES}
    shares, prev = [], None
    for s in SEASONS:
        mask = {t: season_of[t] == s for t in TABLES}
        model = M.TouchModel(len(idx), lam).fit(d, mask, idx, prior=prev.scaled(rho) if prev else None)
        prev = model
        n = len(idx)
        padded = M.TouchModel(n + 1, lam)
        pad = lambda a: np.vstack([a, np.zeros((1, a.shape[1]))])
        base = {k: (p.intercept, p.home, pad(p.off), pad(p.def_)) for k, p in model.parts.items()}

        def rally_share(team, keep):
            # keep: set of (part, "off"/"def") strengths left on for `team`; all else average.
            parts = {}
            for k, (b, h, off, de) in base.items():
                off2, de2 = np.zeros_like(off), np.zeros_like(de)
                if (k, "off") in keep: off2[team] = off[team]
                if (k, "def") in keep: de2[team] = de[team]
                parts[k] = type(model.parts[k])(b, h, off2, de2)
            padded.parts = parts
            t, a, z = np.array([team]), np.array([n]), np.zeros(1)
            return float(0.5 * (padded.sideout(t, a, z)[0] + 1 - padded.sideout(a, t, z)[0]))

        groups = [("serve", "off"), ("serve", "def"), ("fb", "off"), ("fb", "def"), ("tr", "off"), ("tr", "def")]
        everything = set(groups)
        ranks = []
        for name in set(d["sideout"]["t"][(d["sideout"]["match_id"] // 100000) == s]):
            i = idx.ids[name]
            ranks.append((rally_share(i, everything), i))
        for full, i in sorted(ranks, reverse=True)[:10]:
            gain = full - 0.5
            shares.append([(rally_share(i, {g}) - 0.5) / gain for g in groups])
    m = np.mean(shares, axis=0)
    labels = ["Aces and service errors", "Passing (avoiding aces)", "Scoring on the first attack",
              "Stopping the opponent's first attack", "Winning the scramble on offense", "Winning the scramble on defense"]
    return [{"label": l, "share": round(float(v / m.sum()), 4)} for l, v in zip(labels, m)]


def set_momentum(d, idx, con):
    """Do teams come back from 0-2, or hold a set-1 lead, as often as team strength predicts?"""
    from vbsim.matchsim import set_win_prob
    lam, rho, sd = M.TUNED["touch"]
    x, w = np.polynomial.hermite_e.hermegauss(21)
    w = w / w.sum()
    logit = lambda p: np.log(p) - np.log1p(-p)
    expit = lambda z: 1 / (1 + np.exp(-z))
    sets = {(m, n): side for m, n, side in con.execute("SELECT match_id, set_no, winner_side FROM sets").fetchall()}

    def win_from(h, a, set_no, s, f):
        if h == 3: return np.ones_like(s)
        if a == 3: return np.zeros_like(s)
        p = f if set_no == 5 else s
        return p * win_from(h + 1, a, set_no + 1, s, f) + (1 - p) * win_from(h, a + 1, set_no + 1, s, f)

    t = dict(down=0, back=0, down_exp=0.0, back_exp=0.0, n=0, hold=0, hold_exp=0.0)
    for season in (2024, 2025):
        r = run_season(d, idx, M.TouchModel, season, lam, rho)
        keep = np.array([(int(m), 1) in sets and (int(m), 2) in sets for m in r["match_id"]])
        mids, so_h, so_a, home_won = r["match_id"][keep], r["so_home"][keep], r["so_away"][keep], r["home_won"][keep]
        # All matches x all form nodes at once: arrays of shape (matches, nodes).
        a = expit(logit(so_h)[:, None] + sd * x[None, :])
        b = expit(logit(so_a)[:, None] - sd * x[None, :])
        flat = lambda f: f(1 - b.ravel(), a.ravel()).reshape(a.shape)
        S = flat(lambda ps, pr: 0.5 * (set_win_prob(ps, pr, 25, True) + set_win_prob(ps, pr, 25, False)))
        F = flat(lambda ps, pr: 0.5 * (set_win_prob(ps, pr, 15, True) + set_win_prob(ps, pr, 15, False)))
        t["down_exp"] += float((((1 - S) ** 2 + S ** 2) @ w).sum())
        t["back_exp"] += float((((1 - S) ** 2 * win_from(0, 2, 3, S, F) + S ** 2 * (1 - win_from(2, 0, 3, S, F))) @ w).sum())
        t["hold_exp"] += float(((S * win_from(1, 0, 2, S, F) + (1 - S) * (1 - win_from(0, 1, 2, S, F))) @ w).sum())
        s1 = np.array([sets[(int(m), 1)] for m in mids])
        s2 = np.array([sets[(int(m), 2)] for m in mids])
        hw = home_won.astype(bool)
        t["n"] += len(mids)
        t["hold"] += int(((s1 == "home") == hw).sum())
        down = s1 == s2
        t["down"] += int(down.sum())
        t["back"] += int(((s1 == "home") != hw)[down].sum())
    return {"down02": t["down"], "comeback": t["back"] / t["down"], "comeback_exp": t["back_exp"] / t["down_exp"],
            "matches": t["n"], "set1_holds": t["hold"] / t["n"], "set1_holds_exp": t["hold_exp"] / t["n"]}


def home_court(d, idx):
    """Two equal teams: how often does the home team win?"""
    from vbsim.matchsim import match_win_prob
    lam, rho, sd = M.TUNED["touch"]
    season_of = {t: d[t]["match_id"] // 100000 for t in TABLES}
    mask = {t: season_of[t] == SEASONS[-1] for t in TABLES}
    model = M.TouchModel(len(idx), lam).fit(d, mask, idx)
    model.parts = {k: type(p)(p.intercept, p.home, p.off * 0, p.def_ * 0) for k, p in model.parts.items()}
    z = np.zeros(1, dtype=int)
    so_home = model.sideout(z, z, np.ones(1))[0]
    so_away = model.sideout(z, z, -np.ones(1))[0]
    so_neutral = model.sideout(z, z, np.zeros(1))[0]
    return {"home_match": float(match_win_prob(np.array([1 - so_away]), np.array([so_home]), sd)[0]),
            "home_sideout": float(so_home), "away_sideout": float(so_away), "neutral_sideout": float(so_neutral)}


def findings(d, idx):
    con = duckdb.connect(str(D.DB), read_only=True)
    return {"elite": elite_edge(d, idx), "sets": set_momentum(d, idx, con), "home": home_court(d, idx)}


def sample_rally(con):
    """One real rally from the latest national final, as recorded, grouped into the model's steps.
    Picks the first rally with a serve, a first attack and at least two scramble possessions
    that ends in a kill."""
    mid, away, home, date = con.execute("""SELECT match_id, away_team, home_team, date FROM matches
        WHERE (season, date) = (SELECT (max(season), max(date)) FROM matches WHERE season = (SELECT max(season) FROM matches))""").fetchone()
    rallies = con.execute(f"""SELECT set_no, rally_no FROM rallies WHERE match_id = {mid} AND usable
        AND end_event = 'Kill' AND n_contacts BETWEEN 8 AND 12 ORDER BY set_no, rally_no""").fetchall()
    for set_no, rally_no in rallies:
        rows = con.execute(f"""SELECT actor_team, event, player, away_score_before, home_score_before, point_team
            FROM events WHERE match_id = {mid} AND set_no = {set_no} AND rally_no = {rally_no}
              AND row_type IN ('contact', 'outcome') ORDER BY rn""").fetchall()
        poss, out = 0, []
        for i, (team, event, player, a, h, pt) in enumerate(rows):
            if i == 0 or team != rows[i - 1][0]:
                poss += 1
            step = "The serve" if poss == 1 else "First attack" if poss == 2 else "Scramble"
            out.append({"poss": poss, "step": step, "team": team, "event": event, "player": player})
        if poss >= 4:
            return {"date": str(date), "away": away, "home": home, "set": set_no, "score": [rows[0][3], rows[0][4]],
                    "winner": rows[-1][5], "rows": out}
    return None


def totals(con):
    m, r = con.execute("""SELECT count(*), sum(n_usable_rallies) FROM matches""").fetchone()
    return {"matches": int(m), "rallies": int(r), "seasons": f"{SEASONS[0]}-{SEASONS[-1]}"}


DOC_START = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""


def render(payload=None):
    """Write two copies of the page.

    web/index.html     a complete standalone document with the data inline: open it by
                       double-clicking, or host it anywhere (GitHub Pages, any static host).
    web/artifact.html  the same page without document tags or inline data, for the Claude
                       artifact, which adds its own document wrapper and serves data.json beside it.
    """
    template = (HERE / "template.html").read_text()
    payload = payload or (HERE / "data.json").read_text()
    cut = template.index("</style>") + len("</style>")
    head, body = template[:cut], template[cut:]
    inline = '<script id="data" type="application/json">' + payload.replace("</", "<\\/") + "</script>\n"
    body = body.replace("<script>\n// MATH-START", inline + "<script>\n// MATH-START", 1)
    page = (DOC_START + head + "\n<style>[hidden] { display: none !important; } body { margin: 0; }</style>\n"
            + "</head>\n<body>" + body + "</body>\n</html>\n")
    (HERE / "index.html").write_text(page)
    (HERE / "artifact.html").write_text(template)
    return page


def main():
    import sys
    if "--page-only" in sys.argv:  # reuse web/data.json; only the template changed
        page = render((HERE / "data.json").read_text())
        print(f"wrote web/index.html ({len(page) / 1024:.0f} KB) from existing data.json")
        return
    con = duckdb.connect(str(D.DB), read_only=True)
    d = D.load(SEASONS)
    idx = D.TeamIndex()
    for t in TABLES:
        idx(d[t]["t"]), idx(d[t]["o"])
    x, w = np.polynomial.hermite_e.hermegauss(21)
    print("season ratings...")
    data = {
        "built": str(dt.date.today()),
        "totals": totals(con),
        "form_sd": M.TUNED["touch"][2],
        "gh": {"x": r4(x), "w": [float(v) for v in w / w.sum()]},
        "seasons": season_ratings(d, idx, con),
        "champions": champions(con),
        "first_contact": first_contact(con),
        "sample_rally": sample_rally(con),
    }
    print("brackets...")
    data["brackets"] = {}
    for s in SEASONS:
        try:
            data["brackets"][s] = bracket_export(d, idx, con, s)
        except ValueError as e:  # 2020 had a 48-team field with byes
            print(f"  {s}: skipped ({e})")
    data["bracket"] = data["brackets"][SEASONS[-1]]
    print("findings...")
    data["findings"] = findings(d, idx)
    print("calibration...")
    data["calibration"] = calibration_export(d, idx)
    print("momentum...")
    data["momentum"] = momentum_export(d, idx)

    payload = json.dumps(data, separators=(",", ":"))
    (HERE / "data.json").write_text(payload)
    page = render(payload)
    print(f"wrote web/data.json ({len(payload) / 1024:.0f} KB) and web/index.html ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
