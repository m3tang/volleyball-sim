"""Rebuild an NCAA tournament bracket from box scores and compute exact advancement odds.

The bracket is rebuilt from results: start at the final, then for each team step back to
its previous match in the tournament window, down to round 1. Other postseason events
held at the same time never connect to the final, so they drop out.

Venues: rounds 1-2 are played at "pods" of four teams, rounds 3-4 at regional sites,
the last two rounds at one neutral site. If any team played a home match at a site, that
team is the site's host and gets home court in every matchup played there, including
hypothetical ones.
"""
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Optional, Union

import duckdb
import numpy as np

ROUND_NAMES = {1: "Round 1", 2: "Round 2", 3: "Regional semifinal", 4: "Regional final",
               5: "National semifinal", 6: "Final"}


@dataclass
class Node:
    round: int
    date: object
    left: Union["Node", str]
    right: Union["Node", str]
    winner: str                       # actual winner
    venue_home: Optional[str]         # home team in the actual match (None = neutral)
    host: Optional[str] = None        # site host for any matchup at this node
    score: str = ""                   # actual set score, winner first ("3-1")
    teams: list = field(default_factory=list)

    def children(self):
        return [c for c in (self.left, self.right) if isinstance(c, Node)]


# The tournament label varies by season:
#   "2021 NCAA DI Volleyball Championship", "2023 Division I Women's Volleyball Championship",
#   "2024 NCAA Division I Women's Volleyball Championship", "NCAA Division I Volleyball Championship".
# It never matches the separate National Invitational (NIVC) event.
EVENT = r"(?:\d{4} )?(?:NCAA )?(?:DI|Division I) (?:Women's )?Volleyball Championship"
# Tournament opponents look like:
#   "#4 Coppin St. NCAA Division I Volleyball Championship"           -> this team at home
#   "@#1 Creighton NCAA Division I Volleyball Championship"           -> away at Creighton
#   "#2 UCLA @Lexington, KY (NCAA Division I Volleyball Championship)" -> neutral site
# The "#N" labels are not consistent per team (Texas A&M is #1 in some rows, #3 in others),
# so they are not treated as seeds.
_OPP = re.compile(r"^(?P<away>@)?(?:#\d+ )?(?P<name>.+?)(?: @(?P<city>[^(]+?))?\s*\(?" + EVENT + r"\)?$")


def _games(con, season, start, end):
    rows = con.execute(f"""
      SELECT strptime(left("Date", 10), '%m/%d/%Y')::DATE AS d, "Team", "Opponent", "Result"
      FROM team_match
      WHERE season_year = ? AND "Result" SIMILAR TO '[WL] [0-3]-[0-3]'
        AND regexp_matches("Opponent", ?)
        AND strptime(left("Date", 10), '%m/%d/%Y')::DATE BETWEEN ? AND ?
      ORDER BY d""", [season, EVENT, start, end]).fetchall()
    games = {}
    for d, team, opp_raw, result in rows:
        m = _OPP.match(opp_raw)
        if not m:
            raise ValueError(f"unrecognized tournament opponent: {opp_raw!r}")
        opp = m["name"].strip()
        key = (d, frozenset((team, opp)))
        if key in games:
            continue
        if m["away"]:
            venue_home = opp
        elif m["city"]:
            venue_home = None
        else:
            venue_home = team
        winner = team if result.startswith("W") else opp
        sets = result[2:]                           # this team's sets first, e.g. "1-3"
        score = sets if result.startswith("W") else "-".join(reversed(sets.split("-")))
        games[key] = {"date": d, "teams": (team, opp), "winner": winner, "venue_home": venue_home, "score": score}
    return list(games.values())


def rebuild(db_path, season, start=None, end=None):
    """Rebuild a 64-team bracket. start/end default to the whole calendar span of the season."""
    start = start or dt.date(season, 7, 1)
    end = end or dt.date(season + 1, 6, 30)
    con = duckdb.connect(str(db_path), read_only=True)
    games = _games(con, season, start, end)
    con.close()
    by_team = {}
    for g in games:
        for t in g["teams"]:
            by_team.setdefault(t, []).append(g)

    final = max(games, key=lambda g: g["date"])

    def build(g, rnd):
        kids = []
        for t in g["teams"]:
            if rnd == 1:
                kids.append(t)
                continue
            earlier = [p for p in by_team[t] if p["date"] < g["date"]]
            if not earlier:
                raise ValueError(f"{t}: no earlier tournament match before {g['date']} (round {rnd})")
            prev = max(earlier, key=lambda p: p["date"])
            if prev["winner"] != t:
                raise ValueError(f"{t} lost its previous match on {prev['date']} but played again")
            kids.append(build(prev, rnd - 1))
        return Node(rnd, g["date"], kids[0], kids[1], g["winner"], g["venue_home"], score=g["score"])

    root = build(final, 6)
    _fill_teams(root)
    _assign_hosts(root)
    n_games = sum(1 for _ in walk(root))
    if n_games != 63 or len(root.teams) != 64:
        raise ValueError(f"bracket incomplete: {n_games} games, {len(root.teams)} teams")
    return root


def walk(node):
    yield node
    for c in node.children():
        yield from walk(c)


def _fill_teams(node):
    for side in (node.left, node.right):
        if isinstance(side, Node):
            _fill_teams(side)
            node.teams += side.teams
        else:
            node.teams.append(side)


def _assign_hosts(root):
    # Sites: each round-2 node with its round-1 children; each round-4 node with its round-3
    # children; the final rounds are neutral.
    sites = [n for n in walk(root) if n.round in (2, 4)]
    for site in sites:
        nodes = [site] + site.children()
        hosts = {n.venue_home for n in nodes if n.venue_home is not None}
        host = hosts.pop() if len(hosts) == 1 else None
        for n in nodes:
            n.host = host


def advancement(root, win_prob):
    """Exact probability that each team wins each node.

    win_prob(a_list, b_list, host) -> matrix P(a[i] beats b[j]) at a site hosted by `host`.
    Returns {node id: {team: P(team wins this node)}}.
    """
    out = {}

    def solve(node):
        sides = []
        for side in (node.left, node.right):
            sides.append(solve(side) if isinstance(side, Node) else {side: 1.0})
        L, R = sides
        lt, rt = list(L), list(R)
        P = win_prob(lt, rt, node.host)                     # (|L|, |R|)
        pl = np.array([L[t] for t in lt])
        pr = np.array([R[t] for t in rt])
        if not np.isfinite(P).all():
            raise ValueError("win_prob returned non-finite values")
        # numpy 2.0 with Apple's Accelerate BLAS can raise false floating-point warnings in
        # matmul; inputs are checked above and outputs below.
        with np.errstate(all="ignore"):
            win_l = pl * (P @ pr)
            win_r = pr * ((1 - P).T @ pl)
        if not (np.isfinite(win_l).all() and np.isfinite(win_r).all()):
            raise ValueError("advancement produced non-finite values")
        res = dict(zip(lt, win_l)) | dict(zip(rt, win_r))
        out[id(node)] = res
        return res

    solve(root)
    return out


def most_likely_bracket(root, adv):
    """Pick the champion with the highest title odds, then fill each sub-bracket the same way,
    keeping picks consistent (a team picked to win a node also wins all nodes below it)."""
    picks = {}

    def pick(node, forced=None):
        probs = adv[id(node)]
        w = forced if forced is not None else max(probs, key=probs.get)
        picks[id(node)] = w
        for c in node.children():
            pick(c, w if w in c.teams else None)

    pick(root)
    return picks
