"""Generate the golden question set: evals/golden.jsonl (cases) and evals/golden.md (for review).

Usage:  python -m evals.golden            write both files
        python -m evals.golden --check    regenerate in memory and diff against golden.jsonl
                                          (catches a model or data change that moved an answer)

Every expected value is computed here, from rallytools.core (the same numbers the site shows)
or from a reference SQL query run directly against data/vb.duckdb. Nothing comes from a model.

A case:
  id, category (= tags[0]), tags, question
  expect.numbers  [{label, value, scale, alt?}]   all must appear in the answer (see evals/grade.py)
  expect.facts    [{label, any_of}]               one of the strings must appear (normalized)
  expect.refuse   true when the right answer is "the data doesn't cover that"
  tools           acceptable calls; any one is enough (see evals/grade.py tool_choice)
  ref             how the expected values were computed (tool call or SQL)
"""
import json
import random
import sys
from pathlib import Path

import duckdb

from rallytools import core as C
from vbsim.data import DB

HERE = Path(__file__).resolve().parent
OUT = HERE / "golden.jsonl"
MD = HERE / "golden.md"
SEED = 2025


def prob(label, value, alt=()):
    return {"label": label, "value": round(float(value), 4), "scale": "prob", **({"alt": [round(float(a), 4) for a in alt]} if alt else {})}


def count(label, value, alt=()):
    return {"label": label, "value": int(value), "scale": "count", **({"alt": [int(a) for a in alt]} if alt else {})}


def dec(label, value):
    return {"label": label, "value": round(float(value), 3), "scale": "decimal"}


def fact(label, *any_of):
    return {"label": label, "any_of": list(any_of)}


VENUE_PHRASE = {"neutral": [" on a neutral court", " at a neutral site", ""],
                "team_a_home": [" at {A}", " in {A}'s gym", " with {A} at home"],
                "team_b_home": [" at {B}", " in {B}'s gym", " with {A} on the road"]}
# How fans write names that the NCAA abbreviates.
FAN = {"Southern California": "USC", "Miami (FL)": "Miami", "LMU (CA)": "LMU", "UNI": "Northern Iowa",
       "Saint Mary's (CA)": "Saint Mary's", "Pittsburgh": "Pitt"}
FAN_ABBREV = {"Fla.": "Florida", "Ky.": "Kentucky", "Mich.": "Michigan", "Ill.": "Illinois", "Colo.": "Colorado",
              "Ariz.": "Arizona", "Caro.": "Carolina", "Tenn.": "Tennessee", "Ga.": "Georgia", "La.": "Louisiana"}


def fan(name):
    """A team name as a fan would type it ("Kansas State", "USC")."""
    if name in FAN:
        return FAN[name]
    for k, v in FAN_ABBREV.items():
        name = name.replace(k, v)
    return name.replace(" St.", " State") if name.endswith(" St.") else name


def team_fact(label, name):
    return fact(label, *dict.fromkeys([name, fan(name)]))
ROUND_PHRASE = {"Round 1": "first round", "Round 2": "second round", "Regional semifinal": "regional semifinal",
                "Regional final": "regional final", "National semifinal": "national semifinal", "Final": "national final"}
# Keywords that show an answer named a part of the game (for upset explanations).
REASON_WORDS = {
    "first_attack_scoring": ["first attack", "first-ball", "first ball", "off the pass"],
    "first_attack_errors": ["first attack", "first-ball", "first ball", "attack error", "hitting error", "blocked"],
    "scramble_scoring": ["scramble", "long rall", "after a dig", "transition"],
    "aces_against": ["ace", "passing", "pass"],
    "opponent_service_errors": ["service error", "serving error", "missed serve", "missed", "serve"],
}


class Gen:
    def __init__(self):
        self.rng = random.Random(SEED)
        self.cases = []
        self.con = duckdb.connect(str(DB), read_only=True)

    def add(self, category, question, numbers=(), facts=(), tools=(), ref="", tags=(), refuse=False, history=None):
        n = sum(c["category"] == category for c in self.cases) + 1
        self.cases.append({"id": f"{category}-{n:03d}", "category": category, "tags": [category, *tags],
                           "question": question,
                           "expect": {"numbers": list(numbers), "facts": list(facts), "refuse": refuse},
                           "tools": list(tools), "ref": ref, **({"history": history} if history else {})})

    def top(self, season, n):
        return [t["n"] for t in C.data()["seasons"][str(season)]["teams"][:n]]

    def sql(self, q, params=()):
        return self.con.execute(q, list(params)).fetchall()

    # ---- categories ----

    def match_odds(self, n=25):
        venues = ["neutral", "team_a_home", "team_b_home"]
        seasons = [2025] * 13 + [2024] * 4 + [2023] * 3 + [2022] * 3 + [2021] * 2
        for i, s in enumerate(seasons[:n]):
            a, b = self.rng.sample(self.top(s, 30), 2)
            v = venues[i % 3]
            r = C.win_probability(a, s, b, s, v)
            ph = self.rng.choice(VENUE_PHRASE[v]).format(A=fan(a), B=fan(b))
            A, B = fan(a), fan(b)
            tool = {"tool": "win_probability", "teams": [[a, s], [b, s]], "home": {"neutral": None, "team_a_home": a, "team_b_home": b}[v]}
            ref = f"win_probability({a!r}, {s}, {b!r}, {s}, {v!r})"
            yr = "" if s == 2025 else f" in {s}"
            kind = i % 4
            if kind == 0:
                self.add("match_odds", f"What are {A}'s chances of beating {B}{ph}{yr}?",
                         [prob(f"P({a} wins)", r["p_team_a_wins"])], tools=[tool], ref=ref, tags=[v, str(s)])
            elif kind == 1:
                fav = r["favorite"]
                p = r["p_team_a_wins"] if fav == a else r["p_team_b_wins"]
                self.add("match_odds", f"{A} vs {B}{ph}{yr}: who's favored, and by how much?",
                         [prob(f"P({fav} wins)", p)], [team_fact("favorite", fav)], [tool], ref, [v, str(s)])
            elif kind == 2:
                scores = r["final_score_probs_team_a_first"]
                best = max(scores, key=scores.get)
                rev = "-".join(reversed(best.split("-")))
                self.add("match_odds", f"If {A} played {B}{ph}{yr}, what's the most likely final score?",
                         [prob(f"P({best} for {a})", scores[best])], [fact("most likely score", best, rev)], [tool], ref, [v, str(s)])
            else:
                self.add("match_odds", f"How likely is {A} to sweep {B}{ph}{yr}?",
                         [prob(f"P({a} wins 3-0)", r["final_score_probs_team_a_first"]["3-0"])], tools=[tool], ref=ref, tags=[v, str(s)])

    def dream(self, n=15):
        # The seed question, as asked. "Nebraska" has no season: 2023 Nebraska (the team Texas beat in the 2023
        # final) and 2025 Nebraska (the latest) are both fair readings, so either passes if the answer says which.
        alts, tools = [], []
        for s in (2025, 2023):
            r = C.win_probability("Nebraska", s, "Texas", 2023)
            alts.append({"numbers": [prob(f"P(Nebraska {s} wins)", r["p_team_a_wins"])],
                         "facts": [fact(f"says it assumed {s} Nebraska", f"{s} Nebraska", f"Nebraska's {s}", f"Nebraska ({s})",
                                        f"{s} Huskers", f"{s} Cornhuskers")]})
            tools.append({"tool": "win_probability", "teams": [["Nebraska", s], ["Texas", 2023]], "home": None})
        self.add("dream_matchup", "Could Nebraska have beaten 2023 Texas?", tools=tools,
                 ref="win_probability('Nebraska', 2025 or 2023, 'Texas', 2023)", tags=["seed", "ambiguous season"])
        self.cases[-1]["expect"]["alternatives"] = alts
        champs = {int(s): v["champ"] for s, v in C.data()["champions"].items()}
        pairs = set()
        while len(pairs) < n - 1:
            sa, sb = self.rng.sample(range(2020, 2026), 2)
            a = self.rng.choice(self.top(sa, 8) + [champs[sa]] * 2)
            b = self.rng.choice(self.top(sb, 8) + [champs[sb]] * 2)
            if a == b or {(a, sa), (b, sb)} == {("Nebraska", 2025), ("Texas", 2023)}:
                continue
            if any({(a, sa), (b, sb)} == {(x, xs), (y, ys)} for x, xs, y, ys in pairs):
                continue
            pairs.add((a, sa, b, sb))
        for a, sa, b, sb in sorted(pairs):
            r = C.win_probability(a, sa, b, sb)
            q = self.rng.choice(["Who wins: {sa} {a} or {sb} {b}?", "Could {sa} {a} beat {sb} {b}?",
                                 "Dream matchup: {sa} {a} against {sb} {b}. What are the odds?"])
            fav, p = (a, r["p_team_a_wins"]) if r["p_team_a_wins"] >= 0.5 else (b, r["p_team_b_wins"])
            self.add("dream_matchup", q.format(a=fan(a), sa=sa, b=fan(b), sb=sb),
                     [prob(f"P({fav} {sa if fav == a else sb} wins)", p, alt=[1 - p])], [team_fact("favorite", fav)],
                     [{"tool": "win_probability", "teams": [[a, sa], [b, sb]], "home": None}],
                     f"win_probability({a!r}, {sa}, {b!r}, {sb})", ["cross-season"])

    def _record_agrees(self, team, season, w, l):
        bw, bl = self.sql("""SELECT count(*) FILTER (WHERE "Result" LIKE 'W%'), count(*) FILTER (WHERE "Result" LIKE 'L%')
                             FROM team_match WHERE season_year = ? AND "Team" = ?""", [season, team])[0]
        return (bw, bl) == (w, l)

    def team_profile(self, n=20):
        picks = []
        for s in (2025, 2025, 2025, 2025, 2025, 2025, 2024, 2024, 2024, 2023, 2023, 2022, 2022, 2021, 2021,
                  2025, 2024, 2023, 2022, 2025):
            picks.append((self.rng.choice(self.top(s, 40 if s == 2025 else 40)), s))
        for i, (t, s) in enumerate(picks[:n]):
            p = C.team_profile(t, s)
            tool = [{"tool": "team_profile", "team": t, "season": s}]
            ref = f"team_profile({t!r}, {s})"
            step = {x["name"]: x for x in p["steps"]}
            kind = i % 4
            if kind == 0 and self._record_agrees(t, s, p["wins"], p["losses"]):
                self.add("team_profile", f"What was {fan(t)}'s record in {s}, and where did the model rank them?",
                         [count("wins", p["wins"]), count("losses", p["losses"]), count("rank", p["rank"])],
                         tools=tool + [{"tool": "season_rankings", "season": s}], ref=ref, tags=[str(s)])
            elif kind in (0, 1):
                x = step["first_attack_scoring"]
                self.add("team_profile", f"Adjusted for opponents, how often did {fan(t)} score on its first attack in {s}?",
                         [prob("adjusted first-attack scoring", x["adjusted_rate"])], tools=tool, ref=ref, tags=[str(s)])
            elif kind == 2:
                # Three fair definitions: play-by-play serves (counted), opponent-adjusted, box-score aces/attempts.
                x = step["serving_aces"]
                box = self.sql('SELECT "Aces" / "SrvAtt" FROM team_season WHERE season_year = ? AND "Team" = ?', [s, t])[0][0]
                self.add("team_profile", f"What was {fan(t)}'s ace rate when serving in {s}?",
                         [prob("ace rate (counted, adjusted, or box score)", x["counted_rate"], alt=[x["adjusted_rate"]] + ([box] if box is not None else []))],
                         tools=tool + [{"tool": "query_stats"}], ref=ref, tags=[str(s)])
            else:
                x = step["scramble_scoring"]
                self.add("team_profile", f"How good was {fan(t)} in long rallies in {s}? What's its scramble scoring rate compared with the league?",
                         [prob("adjusted scramble scoring", x["adjusted_rate"], alt=[x.get("counted_rate", x["adjusted_rate"])]),
                          prob("league average", x["league_average"])], tools=tool, ref=ref, tags=[str(s)])

    def rankings(self, n=10):
        for s in (2020, 2021, 2022, 2023, 2024, 2025):
            r = C.season_rankings(s, 3)["rankings"]
            if s % 2:
                self.add("rankings", f"Who was the best team in {s} according to the rally model?",
                         [prob("rating", r[0]["rating"])], [team_fact("No. 1 team", r[0]["team"])],
                         [{"tool": "season_rankings", "season": s}], f"season_rankings({s})", [str(s)])
            else:
                self.add("rankings", f"Name the rally model's top three teams of {s}.", [],
                         [team_fact(f"No. {x['rank']}", x["team"]) for x in r],
                         [{"tool": "season_rankings", "season": s}], f"season_rankings({s})", [str(s)])
        for s in self.rng.sample(range(2021, 2026), 4)[: n - 6]:
            t = self.rng.choice(self.top(s, 25)[5:])
            p = C.team_profile(t, s)
            self.add("rankings", f"Where did {fan(t)} finish in the model's {s} rankings?", [count("rank", p["rank"])],
                     tools=[{"tool": "team_profile", "team": t, "season": s}, {"tool": "season_rankings", "season": s}],
                     ref=f"team_profile({t!r}, {s})", tags=[str(s)])

    def brackets(self, n=20):
        # Seed question, as asked.
        a = C.advancement_odds("Texas A&M", 2025)
        self.add("bracket", "Why did Texas A&M win 2025?",
                 [prob("pre-tournament title odds", a["p_reach"][-1]["p"])], [fact("champion", "Texas A&M")],
                 [{"tool": "advancement_odds", "team": "Texas A&M", "season": 2025},
                  {"tool": "bracket_forecast", "season": 2025},
                  {"tool": "explain_upset", "teams": ["Texas A&M", "Kentucky"], "season": 2025}],
                 "advancement_odds('Texas A&M', 2025)", ["seed", "2025"])
        for s in range(2021, 2026):
            f = C.bracket_forecast(s, 64)
            tb = [{"tool": "bracket_forecast", "season": s}]
            pick = f["model_pick_champion"]
            p_pick = next(t["p_title"] for t in f["title_odds"] if t["team"] == pick)
            self.add("bracket", f"Who did the model pick to win the {s} NCAA tournament, and with what odds?",
                     [prob(f"P({pick} title)", p_pick)], [team_fact("pick", pick)], tb, f"bracket_forecast({s})", [str(s)])
            champ = f["actual_champion"]
            self.add("bracket", f"Before the {s} tournament, what chance did the model give {fan(champ)} of winning it all?",
                     [prob(f"P({champ} title)", f["actual_champion_pre_tournament_title_odds"])],
                     tools=tb + [{"tool": "advancement_odds", "team": champ, "season": s}],
                     ref=f"bracket_forecast({s})", tags=[str(s)])
            bs = f["bracket_score"]
            self.add("bracket", f"How many bracket points did the model score in {s}, and how does that compare with picking the team with the better record?",
                     [count("model points", bs["model"]), count("baseline points", bs["better_record_baseline"])],
                     tools=tb, ref=f"bracket_forecast({s})", tags=[str(s)])
        field = [(t["n"], s) for s in range(2021, 2026) for t in C.data()["brackets"][str(s)]["teams"][:12]]
        for t, s in self.rng.sample(field, n - 16):
            a = C.advancement_odds(t, s)
            stage = self.rng.choice([2, 3, 4])
            name = {2: "the Elite Eight", 3: "the Final Four", 4: "the national final"}[stage]
            self.add("bracket", f"What were {fan(t)}'s odds of reaching {name} in {s}?",
                     [prob(f"P({t} reaches {name})", a["p_reach"][stage]["p"])],
                     tools=[{"tool": "advancement_odds", "team": t, "season": s}],
                     ref=f"advancement_odds({t!r}, {s})", tags=[str(s)])

    def upsets(self, n=15):
        rows = []
        for s in range(2021, 2026):
            for m in C.tournament_matches(s, upsets_only=True)["matches"]:
                x = C.explain_upset(s, m["winner"], m["loser"])
                if x["breakdown_available"] and x["reasons"]:
                    rows.append(x)
        for s in range(2021, 2026):  # the biggest upset of each tournament
            biggest = C.tournament_matches(s, upsets_only=True)["matches"]
            m = min(biggest, key=lambda v: v["p_winner_pre_match"])
            self.add("upset", f"What was the biggest upset of the {s} NCAA tournament by the model's pre-match odds?",
                     [prob(f"P({m['winner']} wins)", m["p_winner_pre_match"], alt=[1 - m["p_winner_pre_match"]])],
                     [team_fact("winner", m["winner"]), team_fact("loser", m["loser"])],
                     [{"tool": "tournament_matches", "season": s}], f"tournament_matches({s}, upsets_only=True)", [str(s)])
        for x in self.rng.sample(rows, n - 5):
            top = x["reasons"][0]
            self.add("upset", f"How did {fan(x['winner'])} upset {fan(x['loser'])} in the {x['season']} {ROUND_PHRASE[x['round']]}?",
                     [prob(f"P({x['winner']} wins)", x["p_winner_pre_match"], alt=[1 - x["p_winner_pre_match"]])],
                     [fact(f"top reason: {top['category']}", *REASON_WORDS[top["category"]])],
                     [{"tool": "explain_upset", "teams": [x["winner"], x["loser"]], "season": x["season"]}],
                     f"explain_upset({x['season']}, {x['winner']!r}, {x['loser']!r})", [str(x["season"]), x["round"]])

    def sql_stats(self, n=30):
        T = []
        for s in (2021, 2022, 2023, 2024, 2025):
            for stat, word in (("Kills", "kills"), ("Aces", "aces"), ("Digs", "digs")):
                T.append(("leader", s, stat, word))
        for s, team in ((2022, "Wisconsin"), (2024, "Penn St."), (2025, "Nebraska"), (2023, "Texas"), (2025, "Kentucky")):
            T.append(("hitpct", s, team, None))
            T.append(("team_aces", s, team, None))
        for s, stat, word in ((2022, "Aces", "aces"), (2024, "Digs", "digs"), (2025, "Kills", "kills")):
            T.append(("team_leader", s, stat, word))
        for s in (2023, 2025):
            T.append(("sideout", s, None, None))
        for kind, s, a, b in T[:n]:
            if kind == "leader":
                q = f'SELECT "Player", "Team", "{a}" FROM player_season WHERE season_year = {s} ORDER BY "{a}" DESC LIMIT 1'
                p, team, v = self.sql(q)[0]
                self.add("sql_stats", f"Who led Division I in {b} in {s}, and how many did she have?",
                         [count(b, v)], [fact("player", p)], [{"tool": "query_stats"}], q, [str(s), "leader"])
            elif kind == "hitpct":
                q = f"""SELECT "Hit Pct" FROM team_season WHERE season_year = {s} AND "Team" = '{a}'"""
                v = self.sql(q)[0][0]
                self.add("sql_stats", f"What was {fan(a)}'s hitting percentage in {s}?", [dec("hit pct", v)],
                         tools=[{"tool": "query_stats"}], ref=q, tags=[str(s), "team"])
            elif kind == "team_aces":
                q = f"""SELECT "Aces" FROM team_season WHERE season_year = {s} AND "Team" = '{a}'"""
                v = self.sql(q)[0][0]
                self.add("sql_stats", f"How many aces did {fan(a)} serve in {s}?", [count("aces", v)],
                         tools=[{"tool": "query_stats"}], ref=q, tags=[str(s), "team"])
            elif kind == "team_leader":
                q = f'SELECT "Team", "{a}" FROM team_season WHERE season_year = {s} ORDER BY "{a}" DESC LIMIT 2'
                (team, v), (_, v2) = self.sql(q)
                assert v != v2, f"tie for {a} in {s}"
                self.add("sql_stats", f"Which team had the most {b} in {s}, and how many?",
                         [count(b, v)], [team_fact("team", team)], [{"tool": "query_stats"}], q.replace("LIMIT 2", "LIMIT 1"),
                         [str(s), "team leader"])
            else:
                q = f"SELECT avg(sideout::INT) FROM rallies WHERE season = {s} AND usable"
                v = self.sql(q)[0][0]
                self.add("sql_stats", f"In {s}, how often did the receiving team win the rally?", [prob("sideout rate", v)],
                         tools=[{"tool": "query_stats"}], ref=q, tags=[str(s), "league"])

    def out_of_scope(self):
        qs = [
            ("Which Nebraska players were injured during the 2025 season?", "injuries"),
            ("Who will win the 2026 national championship?", "future"),
            ("What's the point spread for Texas vs Stanford?", "betting"),
            ("How did UCLA's men's volleyball team do in 2024?", "men"),
            ("Who won the 2025 Division II national title?", "other division"),
            ("What starting rotation did Pittsburgh use in the 2025 national semifinal?", "lineups"),
            ("How hard does Nebraska's best server hit her jump serve, in mph?", "ball tracking"),
            ("How many fans attended the 2025 national final?", "attendance"),
            ("Who coached Kentucky in 2023, and what's their career record?", "coaches"),
            ("How good was 2019 Stanford?", "before 2020"),
            ("Rank the best beach volleyball pairs of 2024.", "beach"),
            ("Will Kentucky's setter come back next season?", "future"),
            ("How would Nebraska do against Brazil's national team?", "outside NCAA"),
            ("What were the TV ratings for the 2024 final?", "ratings"),
            ("Should I bet on Texas A&M to repeat in 2026?", "betting"),
        ]
        for q, why in qs:
            self.add("out_of_scope", q, tools=[], ref="not in the data", tags=[why], refuse=True)

    def live_odds(self, n=6):
        """Odds from a score in progress. Questions say who serves, so the answer is determined."""
        rng = random.Random(SEED + 1)   # own stream, so earlier categories don't change
        ordinal = ["first", "second", "third", "fourth", "fifth"]
        states = [(1, 2, 20, 23, "b"), (2, 2, 10, 12, "a"), (2, 0, 15, 15, "a"), (0, 1, 24, 22, "a"),
                  (1, 1, 0, 0, "a"), (2, 1, 23, 24, "b")]
        for sa_, sb_, pa, pb, srv in states[:n]:
            a, b = rng.sample(self.top(2025, 30), 2)
            A, B = fan(a), fan(b)
            sets = f"leads {B} {sa_}-{sb_} in sets" if sa_ > sb_ else f"trails {B} {sa_}-{sb_} in sets" if sa_ < sb_ \
                else f"is tied with {B} {sa_}-{sb_} in sets"
            set_no = ordinal[sa_ + sb_]
            pts = f"is up {pa}-{pb}" if pa > pb else f"is down {pa}-{pb}" if pa < pb else f"is tied {pa}-{pb}"
            q = (f"{A} {sets} and {pts} in the {set_no} set, with {A if srv == 'a' else B} serving. "
                 f"What are {A}'s chances of winning the match from here?")
            if pa == pb == 0:
                q = (f"{A} {sets}, and {A if srv == 'a' else B} is about to serve to start the {set_no} set. "
                     f"What are {A}'s chances of winning the match from here?")
            args = {"team_a": a, "season_a": 2025, "team_b": b, "season_b": 2025, "venue": "neutral",
                    "sets_a": sa_, "sets_b": sb_, "points_a": pa, "points_b": pb, "serving": "team_" + srv}
            r = C.live_win_probability(**args)
            self.add("live_odds", q, [prob(f"P({a} wins from {sa_}-{sb_}, {pa}-{pb})", r["p_team_a_wins"])],
                     tools=[{"tool": "live_win_probability", "teams": [[a, 2025], [b, 2025]], "home": None, "args": args}],
                     ref=f"live_win_probability({args})", tags=["2025", f"set {sa_ + sb_ + 1}"])

    def logged(self):
        """Real questions from the public chat (evals/logged.py), with answers read from the tool result."""
        from evals.logged import LOGGED
        for e in LOGGED:
            name, args = e["call"]
            result = C_call(name, args)
            nums = []
            for scale, label, path in e.get("numbers", []):
                v = result
                for k in path.split("."):
                    v = v[int(k)] if isinstance(v, list) else v[k]
                nums.append({"prob": prob, "count": count, "decimal": dec}[scale](label, v))
            facts = [fact(label, *spellings) for label, spellings in e.get("facts", [])]
            history = [m for qa in e.get("history", []) for m in ({"role": "user", "content": qa[0]},
                                                                   {"role": "assistant", "content": qa[1]})]
            self.add("logged", e["question"], nums, facts, [] if e.get("refuse") else [spec_from_call(name, args)],
                     ref=f"{name}({args})", tags=["logged", e["asked"]] + (["follow-up"] if history else []),
                     refuse=e.get("refuse", False), history=history or None)


def C_call(name, args):
    from rallytools.registry import call
    r = call(name, args)
    if r["error"]:
        raise ValueError(f"logged case {name}({args}): {r['error']}")
    return r["result"]


def spec_from_call(name, args):
    """The grader's tool-choice spec for a call (see evals/grade.py sufficient)."""
    if name in ("win_probability", "live_win_probability"):
        sa = args["season_a"]
        sb = args.get("season_b") or sa
        v = args.get("venue", "neutral")
        home = None if v == "neutral" else args["team_a"] if v == "team_a_home" else args["team_b"]
        spec = {"tool": name, "teams": [[args["team_a"], sa], [args["team_b"], sb]], "home": home}
        return {**spec, "args": args} if name == "live_win_probability" else spec
    if name in ("team_profile", "advancement_odds"):
        return {"tool": name, "team": args["team"], "season": args["season"]}
    if name in ("season_rankings", "bracket_forecast", "tournament_matches"):
        return {"tool": name, "season": args["season"]}
    if name == "explain_upset":
        return {"tool": name, "season": args["season"], "teams": [args["team_a"], args["team_b"]]}
    return {"tool": name}


def generate():
    g = Gen()
    g.match_odds(); g.dream(); g.team_profile(); g.rankings(); g.brackets(); g.upsets(); g.sql_stats(); g.out_of_scope(); g.live_odds(); g.logged()
    ids = [c["id"] for c in g.cases]
    assert len(ids) == len(set(ids))
    qs = [c["question"].lower() for c in g.cases]
    assert len(qs) == len(set(qs)), "duplicate questions"
    return g.cases


def fmt_expect(c):
    e = c["expect"]
    if e["refuse"]:
        return "decline: not in the data"
    if e.get("alternatives"):
        return " OR ".join("(" + fmt_expect({"expect": {**alt, "refuse": False}}) + ")" for alt in e["alternatives"])
    parts = []
    for x in e["numbers"]:
        v = C.odds_text(x["value"]) if x["scale"] == "prob" else f"{x['value']:.3f}" if x["scale"] == "decimal" else f"{x['value']:,}"
        parts.append(f"{x['label']} = {v}" + (f" (or {', '.join(map(str, x['alt']))})" if x.get("alt") else ""))
    for f in e["facts"]:
        parts.append(f"{f['label']}: " + " / ".join(f["any_of"]))
    return "; ".join(parts)


def write_md(cases):
    from collections import Counter
    by = Counter(c["category"] for c in cases)
    lines = ["# Golden questions", "",
             f"{len(cases)} cases, generated by `python -m evals.golden` (seed {SEED}). Expected values come from "
             "`rallytools.core` (the numbers the site shows) or the reference SQL; none from a model.", "",
             "| category | cases |", "|---|---|"] + [f"| {k} | {v} |" for k, v in by.items()] + [""]
    for cat in by:
        lines += [f"## {cat}", "", "| id | question | expected | acceptable tools |", "|---|---|---|---|"]
        for c in (c for c in cases if c["category"] == cat):
            tools = ", ".join(sorted({t["tool"] for t in c["tools"]})) or "none needed"
            q = c["question"].replace("|", "\\|")
            lines.append(f"| {c['id']} | {q} | {fmt_expect(c).replace('|', '/')} | {tools} |")
        lines.append("")
    MD.write_text("\n".join(lines))


def main():
    cases = generate()
    if "--check" in sys.argv:
        old = [json.loads(l) for l in OUT.read_text().splitlines()]
        changed = [n["id"] for o, n in zip(old, cases) if o != n] + ([] if len(old) == len(cases) else ["(count changed)"])
        print("golden set up to date" if not changed else f"{len(changed)} cases changed: {', '.join(changed[:20])}")
        sys.exit(1 if changed else 0)
    OUT.write_text("".join(json.dumps(c) + "\n" for c in cases))
    write_md(cases)
    print(f"wrote {len(cases)} cases to {OUT.name} and {MD.name}")


if __name__ == "__main__":
    main()
