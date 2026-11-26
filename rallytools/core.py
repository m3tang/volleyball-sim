"""Model tools, computed from the published parameters in web/data.json.

Everything here mirrors the math the site runs in the browser (web/template.html, MATH-START
to MATH-END), so a number from these tools is the number a reader sees on the site. Nothing
is refit: the functions are pure and deterministic, and data.json is loaded once.

Errors a caller can fix (unknown team, season out of range) raise ToolError with a message
written for the caller, including close matches for misspelled team names.
"""
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from vbsim.matchsim import set_win_prob

ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = ROOT / "web" / "data.json"

ROUND_NAMES = ["Round 1", "Round 2", "Regional semifinal", "Regional final", "National semifinal", "Final"]
# data.json bracket "odds"[i] = P(team wins its round i+1 match) = P(reaches the next stage).
REACH_NAMES = ["Round 2", "Regional semifinal (Sweet 16)", "Regional final (Elite 8)",
               "National semifinal (Final 4)", "Final", "Champion"]
VENUES = ("neutral", "team_a_home", "team_b_home")

# The 12 per-team numbers, in data.json order: (index, step, side, outcome class, name, description, good direction).
PARAMS = [
    (0, "serve", "off", 0, "serving_aces", "Ace rate on its serves", 1),
    (1, "serve", "off", 1, "serving_errors", "Service error rate", -1),
    (2, "serve", "def", 0, "passing_aces_allowed", "Opponents' ace rate on serves to this team", -1),
    (3, "serve", "def", 1, "passing_errors_drawn", "Opponents' service error rate", 1),
    (4, "fb", "off", 0, "first_attack_scoring", "Scores on its first attack after the pass", 1),
    (5, "fb", "off", 1, "first_attack_errors", "Loses the point on its first attack (error or blocked)", -1),
    (6, "fb", "def", 0, "first_attack_defense_kills_allowed", "Opponents score on their first attack", -1),
    (7, "fb", "def", 1, "first_attack_defense_errors_forced", "Opponents lose the point on their first attack", 1),
    (8, "tr", "off", 0, "scramble_scoring", "Scores on a possession after a dig", 1),
    (9, "tr", "off", 1, "scramble_errors", "Loses the point on a possession after a dig", -1),
    (10, "tr", "def", 0, "scramble_defense_kills_allowed", "Opponents score on a possession after a dig", -1),
    (11, "tr", "def", 1, "scramble_defense_errors_forced", "Opponents lose the point on a possession after a dig", 1),
]


class ToolError(ValueError):
    """A problem the caller can fix; the message says how."""


@lru_cache(maxsize=1)
def data():
    return json.loads(DATA_JSON.read_text())


def seasons():
    return sorted(int(s) for s in data()["seasons"])


def _r(x, n=4):
    return round(float(x), n)


def one_in(p):
    """"about 1 in N" with N rounded the way the site rounds it (oneIn in web/template.html)."""
    n = 1 / p
    nice = (round(n) if n < 10 else round(n / 5) * 5 if n < 100 else round(n / 50) * 50 if n < 1000
            else round(n / 500) * 500)
    return f"about 1 in {nice:,}"


def odds_text(p):
    """Odds as the site shows them: a percentage, plus "about 1 in N" for long shots (under 10%)."""
    if p >= 0.995 and p < 1:
        return ">99%"
    if 0 < p < 0.001:
        return f"<0.1% ({one_in(p)})"
    s = f"{100 * p:.1f}%" if p < 0.1 else f"{100 * p:.0f}%"
    if 0 < p < 0.1:
        s += f" ({one_in(p)})"
    return s


# ---------- name resolution ----------

# Names fans use that no spelling rule gets to. Keys and values are normalized (_norm) forms.
ALIASES = {"usc": "southern california", "miami": "miami (fl)", "pitt": "pittsburgh", "lmu": "lmu (ca)",
           "loyola marymount": "lmu (ca)", "saint mary's": "saint mary's (ca)", "st. mary's": "saint mary's (ca)",
           "northern iowa": "uni", "central florida": "ucf", "hawai'i": "hawaii", "fau": "florida atlantic",
           "texas a & m": "texas a&m", "tamu": "texas a&m", "uk": "kentucky", "nebby": "nebraska",
           "ole miss": "ole miss", "smu": "smu", "byu": "byu", "tcu": "tcu", "lsu": "lsu",
           "north carolina st.": "nc st.", "n.c. st.": "nc st.", "fiu": "fiu", "fgcu": "fgcu"}
# NCAA box-score abbreviations, expanded so "Western Kentucky" matches "Western Ky.".
ABBREV = {"fla.": "florida", "ky.": "kentucky", "mich.": "michigan", "ill.": "illinois", "colo.": "colorado",
          "ariz.": "arizona", "caro.": "carolina", "conn.": "connecticut", "tenn.": "tennessee", "ala.": "alabama",
          "ga.": "georgia", "la.": "louisiana", "mo.": "missouri", "ind.": "indiana", "wash.": "washington",
          "ark.": "arkansas", "miss.": "miss", "val.": "valley", "so.": "southern", "col.": "college", "u.": ""}


def _norm(name, alias=True):
    """Normalized team name; alias=False for free text (spelling rules only)."""
    x = name.lower().replace("&amp;", "&").replace("’", "'").strip()
    x = re.sub(r"\bstate\b", "st.", x)
    x = re.sub(r"\bst\b(?!\.)", "st.", x)
    for k, v in ABBREV.items():
        x = re.sub(r"(?<![\w.])" + re.escape(k) + r"(?=\s|$)", v, x)
    x = re.sub(r"\buniversity of\b|\bthe\b|\buniversity\b", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return ALIASES.get(x, x) if alias else x


def _season(season):
    try:
        s = int(season)
    except (TypeError, ValueError):
        raise ToolError(f"season must be a year like 2025, got {season!r}")
    if str(s) not in data()["seasons"]:
        raise ToolError(f"no data for season {s}; seasons available: {seasons()[0]}-{seasons()[-1]}")
    return s


def resolve_team(name, season):
    """Return the team record from data.json for a name as a person might type it."""
    s = _season(season)
    teams = data()["seasons"][str(s)]["teams"]
    by_norm = {_norm(t["n"]): t for t in teams}
    # Also match without a state tag in parentheses ("Saint Mary's" for "Saint Mary's (CA)") when unique.
    bare = {}
    for k, t in by_norm.items():
        bare.setdefault(re.sub(r"\s*\([a-z]{2}\)$", "", k), []).append(t)
    t = by_norm.get(_norm(name)) or next(iter(v[0] for k, v in bare.items() if k == _norm(name) and len(v) == 1), None)
    if t:
        return t
    close = difflib.get_close_matches(_norm(name), list(by_norm), n=3, cutoff=0.6)
    msg = f"no team named {name!r} in the {s} ratings."
    if close:
        msg += " Did you mean: " + ", ".join(by_norm[c]["n"] for c in close) + "?"
    if s != seasons()[-1]:
        msg += f" The published {s} ratings include only that season's top {len(teams)} teams."
    raise ToolError(msg)


# ---------- the touch model (mirrors MATH-START in web/template.html) ----------

def _step_probs(b, h, off, def_, home):
    e = np.exp(np.asarray(b) + np.asarray(off) - np.asarray(def_) + home * np.asarray(h))
    z = 1 + e.sum()
    return np.append(e / z, 1 / z)


def _ctx(sa, sb):
    A, B = data()["seasons"][str(sa)], data()["seasons"][str(sb)]
    mix = lambda x, y: [(u + v) / 2 for u, v in zip(x, y)]
    return {"b": {k: mix(A["b"][k], B["b"][k]) for k in ("serve", "fb", "tr")},
            "h": {k: mix(A["h"][k], B["h"][k]) for k in ("serve", "fb", "tr")}}


def _sideout(R, S, home_r, ctx):
    """P(receiver R wins the rally when S serves), with the step probabilities behind it."""
    P = lambda team, i: team["p"][i:i + 2]
    srv = _step_probs(ctx["b"]["serve"], ctx["h"]["serve"], P(S, 0), P(R, 2), -home_r)   # ace, serr, in play
    fb = _step_probs(ctx["b"]["fb"], ctx["h"]["fb"], P(R, 4), P(S, 6), home_r)           # win, lose, to opp
    trR = _step_probs(ctx["b"]["tr"], ctx["h"]["tr"], P(R, 8), P(S, 10), home_r)
    trS = _step_probs(ctx["b"]["tr"], ctx["h"]["tr"], P(S, 8), P(R, 10), -home_r)
    wR, tR, wS, tS = trR[0], trR[2], trS[0], trS[2]
    x = (wR + tR * (1 - wS - tS)) / (1 - tR * tS)
    y = wS + tS * (1 - x)
    return {"p": srv[1] + srv[2] * (fb[0] + fb[2] * (1 - y)), "srv": srv, "fb": fb, "tr": trR}


def _match_outcomes(so_a, so_b):
    """Set and final-score probabilities for A, averaged over match-day form like the site."""
    d = data()
    gh = [(x, w) for x, w in zip(d["gh"]["x"], d["gh"]["w"]) if w > 1e-9]
    x = np.array([g[0] for g in gh])
    w = np.array([g[1] for g in gh])
    logit = lambda p: np.log(p) - np.log1p(-p)
    expit = lambda z: 1 / (1 + np.exp(-z))
    sd = d["form_sd"]
    a, b = expit(logit(so_a) + sd * x), expit(logit(so_b) - sd * x)
    p_srv, p_rcv = 1 - b, a
    s = 0.5 * (set_win_prob(p_srv, p_rcv, 25, True) + set_win_prob(p_srv, p_rcv, 25, False))
    f = 0.5 * (set_win_prob(p_srv, p_rcv, 15, True) + set_win_prob(p_srv, p_rcv, 15, False))
    o = {"set": s, "3-0": s ** 3, "3-1": 3 * s ** 3 * (1 - s), "3-2": 6 * s * s * (1 - s) ** 2 * f,
         "2-3": 6 * s * s * (1 - s) ** 2 * (1 - f), "1-3": 3 * s * (1 - s) ** 3, "0-3": (1 - s) ** 3}
    out = {k: float((v * w).sum() / w.sum()) for k, v in o.items()}
    out["win"] = out["3-0"] + out["3-1"] + out["3-2"]
    return out


def _steps(r):
    return {"sideout": _r(r["p"]),
            "aced": _r(r["srv"][0]), "opponent_service_error": _r(r["srv"][1]),
            "first_attack_score": _r(r["fb"][0]), "first_attack_error_or_blocked": _r(r["fb"][1]),
            "scramble_score": _r(r["tr"][0])}


# ---------- tools ----------

def win_probability(team_a, season_a, team_b, season_b=None, venue="neutral"):
    """Exact match odds for team_a vs team_b, as on the site's match simulator."""
    if venue not in VENUES:
        raise ToolError(f"venue must be one of {', '.join(VENUES)}, got {venue!r}")
    sa = _season(season_a)
    sb = _season(season_b if season_b is not None else season_a)
    A, B = resolve_team(team_a, sa), resolve_team(team_b, sb)
    if A is B:
        raise ToolError("team_a and team_b are the same team and season")
    home = {"neutral": 0, "team_a_home": 1, "team_b_home": -1}[venue]
    ctx = _ctx(sa, sb)
    ra, rb = _sideout(A, B, home, ctx), _sideout(B, A, -home, ctx)
    o = _match_outcomes(ra["p"], rb["p"])
    fav = A["n"] if o["win"] >= 0.5 else B["n"]
    return {
        "team_a": {"name": A["n"], "season": sa}, "team_b": {"name": B["n"], "season": sb},
        "venue": venue, "cross_season": sa != sb,
        "p_team_a_wins": _r(o["win"]), "p_team_b_wins": _r(1 - o["win"]),
        "display": {A["n"]: odds_text(o["win"]), B["n"]: odds_text(1 - o["win"])},
        "favorite": fav,
        "p_team_a_wins_set": _r(o["set"]),
        "final_score_probs_team_a_first": {k: _r(o[k]) for k in ("3-0", "3-1", "3-2", "2-3", "1-3", "0-3")},
        "when_receiving_serve": {A["n"]: _steps(ra), B["n"]: _steps(rb)},
        "notes": ("Exact calculation over every rally and final score; averaged over match-day form."
                  + (" Cross-season matchups assume the overall level of play is the same in both seasons." if sa != sb else "")),
    }


def team_profile(team, season):
    """Rank, record, rating, and the 12 rally-step numbers for one team-season."""
    s = _season(season)
    t = resolve_team(team, s)
    S = data()["seasons"][str(s)]
    zero = [0.0, 0.0]
    steps = []
    for i, part, side, cls, name, desc, good in PARAMS:
        pair = t["p"][i - cls:i - cls + 2]
        adj = (_step_probs(S["b"][part], S["h"][part], pair, zero, 0) if side == "off"
               else _step_probs(S["b"][part], S["h"][part], zero, pair, 0))[cls]
        avg = _step_probs(S["b"][part], S["h"][part], zero, zero, 0)[cls]
        row = {"number": i + 1, "name": name, "measures": desc,
               "adjusted_rate": _r(adj), "league_average": _r(avg),
               "better_than_average": bool((adj - avg) * good > 0.0005), "model_value": t["p"][i]}
        if t.get("raw"):
            row["counted_rate"] = t["raw"]["rate"][i]
            row["counted_chances"] = t["raw"]["n"][i // 2]
        steps.append(row)
    return {"team": t["n"], "season": s, "rank": t["rank"], "teams_ranked": len(S["teams"]),
            "wins": t["w"], "losses": t["l"],
            "rating": t["r"],
            "rating_means": "share of rallies won against a league-average team on a neutral floor",
            "steps": steps,
            "notes": "counted = what happened in the team's own matches; adjusted = the model's estimate "
                     "against an average opponent. Ratings are end of season."}


def season_rankings(season, top=25, conference=None):
    """End-of-season model rankings. Earlier seasons publish only their top 64."""
    s = _season(season)
    top = max(1, min(int(top), 400))
    teams = data()["seasons"][str(s)]["teams"]
    if conference:
        raise ToolError("conference filtering is not in the published ratings; use query_stats on team_season")
    return {"season": s, "teams_published": len(teams),
            "rankings": [{"rank": t["rank"], "team": t["n"], "wins": t["w"], "losses": t["l"], "rating": t["r"]}
                         for t in teams[:top]]}


def _bracket(season):
    s = _season(season)
    b = data()["brackets"].get(str(s))
    if b is None:
        raise ToolError(f"no bracket for {s}. Brackets cover {', '.join(sorted(data()['brackets']))}; "
                        "2020 had a 48-team field with byes and is not modeled.")
    return s, b


def bracket_forecast(season, top=10):
    """The pre-tournament bracket forecast and how it scored."""
    s, b = _bracket(season)
    champ_odds = next(t for t in b["teams"] if t["n"] == b["champion"])["odds"][5]
    return {
        "season": s, "tournament_start": b["start"], "actual_champion": b["champion"],
        "actual_champion_pre_tournament_title_odds": champ_odds, "actual_champion_display": odds_text(champ_odds),
        "model_pick_champion": b["score"]["model_champ"],
        "title_odds": [{"team": t["n"], "p_title": t["odds"][5], "display": odds_text(t["odds"][5]),
                        "tournament_wins": t["wins"]} for t in b["teams"][:max(1, min(int(top), 64))]],
        "bracket_score": {"model": b["score"]["model"], "better_record_baseline": b["score"]["record"],
                          "max": b["score"]["max"], "baseline_champion_pick": b["score"]["record_champ"]},
        "matches_called": {"model_correct": round(b["acc"]["model"] * b["acc"]["games"]), "games": b["acc"]["games"],
                           "model_accuracy": _r(b["acc"]["model"]), "baseline_accuracy": _r(b["acc"]["record"]),
                           "model_log_loss": _r(b["acc"]["log_loss"])},
        "notes": "Odds are exact (every path through the bracket) from a model fit only on matches before round 1.",
    }


def advancement_odds(team, season):
    """One team's pre-tournament odds of reaching each stage, and how far it actually went."""
    s, b = _bracket(season)
    by = {_norm(t["n"]): t for t in b["teams"]}
    t = by.get(_norm(team))
    if t is None:
        close = difflib.get_close_matches(_norm(team), list(by), n=3, cutoff=0.6)
        raise ToolError(f"{team!r} was not in the {s} NCAA tournament field."
                        + (" Did you mean: " + ", ".join(by[c]["n"] for c in close) + "?" if close else ""))
    w = t["wins"]
    went = "won the title" if w == 6 else f"lost in the {ROUND_NAMES[w]}"
    return {"team": t["n"], "season": s,
            "p_reach": [{"stage": n, "p": p, "display": odds_text(p)} for n, p in zip(REACH_NAMES, t["odds"])],
            "tournament_wins": w, "result": went,
            "pre_tournament_win_pct": t["rec"][0], "pre_tournament_matches": t["rec"][1]}


def _nodes(tree):
    yield tree
    for k in tree["k"]:
        if isinstance(k, dict):
            yield from _nodes(k)


def _match_view(s, n):
    p_w = n["pa"] if n["w"] == n["a"] else 1 - n["pa"]
    loser = n["b"] if n["w"] == n["a"] else n["a"]
    return {"season": s, "round": ROUND_NAMES[n["r"] - 1], "winner": n["w"], "loser": loser,
            "score": n["score"], "host": n["host"], "model_pick": n["pick"],
            "p_winner_pre_match": _r(p_w), "display": odds_text(p_w), "upset": p_w < 0.5}


def tournament_matches(season, team=None, upsets_only=False):
    """NCAA tournament results with the model's pre-match odds."""
    s, b = _bracket(season)
    out = []
    for n in _nodes(b["tree"]):
        v = _match_view(s, n)
        if upsets_only and not v["upset"]:
            continue
        if team and _norm(team) not in (_norm(n["a"]), _norm(n["b"])):
            continue
        out.append(v)
    order = {r: i for i, r in enumerate(ROUND_NAMES)}
    out.sort(key=lambda v: (order[v["round"]], -v["p_winner_pre_match"]))
    if team and not out:
        raise ToolError(f"no {s} NCAA tournament match found for {team!r}")
    return {"season": s, "matches": out}


# Breakdown columns (web/build_site.py BREAKDOWN): sideout, aces against, opponent service errors,
# first-attack scoring, first-attack errors or blocked, scramble scoring. Each reason: (column, sign
# for the team itself, category). Mirrors reasons() in web/template.html.
REASONS = [(3, 1, "first_attack_scoring"), (4, -1, "first_attack_errors"), (5, 1, "scramble_scoring"),
           (1, -1, "aces_against"), (2, 1, "opponent_service_errors")]
# What each reason means, for the team it is about (T) and its opponent (O).
REASON_TEXT = {
    "first_attack_scoring": "{T} scoring on its first attack after the pass",
    "first_attack_errors": "{T}'s first-attack errors and blocked swings",
    "scramble_scoring": "{T} scoring on possessions after a dig (the scramble)",
    "aces_against": "{T}'s passing: how often {T} got aced on {O}'s serves",
    "opponent_service_errors": "{O}'s missed serves when serving to {T}",
}
BREAKDOWN_COLS = ["sideout", "aces_against", "opponent_service_errors", "first_attack_scoring",
                  "first_attack_errors_or_blocked", "scramble_scoring"]


def explain_upset(season, team_a, team_b):
    """Why a tournament match went the way it did: points swung by each part of the game,
    actual vs what the pre-tournament model expected for this matchup and venue."""
    s, b = _bracket(season)
    want = {_norm(team_a), _norm(team_b)}
    node = next((n for n in _nodes(b["tree"]) if {_norm(n["a"]), _norm(n["b"])} == want), None)
    if node is None:
        raise ToolError(f"{team_a} and {team_b} did not meet in the {s} NCAA tournament. "
                        "Breakdowns exist only for tournament matches.")
    view = _match_view(s, node)
    x = node["x"]
    if not x:
        return {**view, "breakdown_available": False,
                "notes": "No play-by-play was recorded for this match, so there is no breakdown."}
    W, L = view["winner"], view["loser"]
    reasons = []
    for T, sgn in ((W, 1), (L, -1)):
        for i, good, cat in REASONS:
            a, e, n = x["act"][T][i], x["exp"][T][i], x["n"][T][i]
            pts = good * (a - e) * n * sgn
            if pts >= 1:
                O = L if T == W else W
                reasons.append({"points_for_winner": round(pts, 1), "team": T, "category": cat,
                                "what": REASON_TEXT[cat].format(T=T, O=O),
                                "actual_rate": a, "expected_rate": e, "chances": n})
    reasons.sort(key=lambda r: -r["points_for_winner"])
    for k, r in enumerate(reasons, 1):
        r["rank"] = k
    table = {T: {c: {"expected": x["exp"][T][i], "actual": x["act"][T][i], "chances": x["n"][T][i]}
                 for i, c in enumerate(BREAKDOWN_COLS)} | {"block_points": x["blk"][T]} for T in (W, L)}
    return {**view, "breakdown_available": True,
            "biggest_factor": reasons[0] if reasons else None,
            "reasons": reasons, "by_team": table,
            "notes": "reasons are ranked by points swung, largest first: lead with rank 1 (biggest_factor) when "
                     "explaining the result. points_for_winner = extra points the winner gained from that part of "
                     "the game versus "
                     "the model's expectation, given how many chances each team had. Expected rates come from "
                     "the pre-tournament model for this matchup and venue."}


def model_facts():
    """Headline evaluation numbers and what the data does and does not cover."""
    d = data()
    out = {"data": d["totals"], "seasons_rated": seasons(), "brackets": sorted(int(s) for s in d["brackets"]),
           "calibration_2025": {"log_loss": _r(d["calibration"]["log_loss"]), "accuracy": _r(d["calibration"]["accuracy"]),
                                "matches": d["calibration"]["matches"]},
           "covers": "Division I women's volleyball play-by-play 2020-2025: rallies, touches, box scores, players.",
           "does_not_cover": ["injuries or lineups for a given match", "anything after the 2025 season",
                              "men's volleyball, other divisions, beach volleyball",
                              "betting lines", "full ratings for teams outside the top 64 before 2025"]}
    if "xgb" in d:
        out["xgboost_comparison"] = d["xgb"]
    return out
