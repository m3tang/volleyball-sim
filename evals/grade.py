"""Score one answer against one golden case.

    grade(case, run) -> {"grade": {metric: number}, "detail": {...}}

run = {"answer": str, "tool_calls": [{"name", "args", "result", "error"}]}

Metrics (per case):
  correct      1 if every expected number and fact is in the answer (refusal cases: 1 if it declined)
  recall       share of expected numbers and facts present (refusal cases: same as correct)
  grounded     share of the answer's checked numbers that some tool returned (1.0 when it states none)
  tool_choice  1 if some call was sufficient (right tool, right teams/seasons/venue); absent for
               refusal cases, where no tool is required
  refused      1 if the answer declined: refusal wording and no checked numbers (reported for every case,
               so wrong refusals show up; an answer with a caveat is not a refusal)
Hallucination rate for a run = ungrounded numbers / checked numbers, pooled over all cases.
"""
import re

from rallytools import core as C
from rallytools import grounding as G

# Declining needs a statement about the data or tools, so "Texas doesn't have much of a chance" is not one.
_DATA = r"(?:data|dataset|database|model|tools?|numbers|records?|stats|information|play-by-play|box scores)"
REFUSAL = re.compile("|".join([
    _DATA + r"\s+(?:\w+\s+){0,2}?(?:does ?n[o']t|do ?n[o']t|can ?n?[o']t|cannot|won't|is ?n[o']t able to)\s+(?:\w+\s+){0,2}?"
            r"(?:cover|include|contain|have|track|record|go|extend|say|tell|answer|predict|know|reach|produce|give|"
            r"offer|publish|provide|generate|make)",
    r"(?:does ?n[o']t|do ?n[o']t|can ?n?[o']t|cannot|unable to)\s+(?:\w+\s+){0,3}?(?:have|see|find|access|answer|predict)\s+(?:\w+\s+){0,2}?" + _DATA,
    r"(?:not|is ?n[o']t|are ?n[o']t)\s+(?:in|part of|covered (?:by|in)|available in|included in|tracked in|recorded in)\s+(?:the|my|this|any)\s+(?:\w+\s+)?" + _DATA,
    r"(?:outside|beyond)\s+(?:of\s+)?(?:the|what|my)\s+(?:\w+\s+){0,3}?" + _DATA + r"|" + _DATA + r"\s+(?:only|just)\s+(?:covers|goes|has|includes)",
    r"no\s+(?:\w+\s+)?" + _DATA + r"\s+(?:on|about|for)",
    r"(?:can ?n?[o']t|cannot|won't)\s+(?:predict|forecast|tell you (?:who|what) will)",
]), re.I)


def refused(answer):
    return bool(REFUSAL.search(answer or ""))


def _norm_text(s):
    # Hyphens and spaces are interchangeable in prose ("first-attack scoring" names the first attack).
    return re.sub(r"(?<=[a-z])-(?=[a-z])", " ", C._norm(s or "", alias=False))


def _as_counts(m):
    """A record or score like "20-8" states two counts."""
    if m.kind == "pair":
        return [G.Mention(str(x), "num", float(x), 0, start=m.start) for x in m.pair]
    return [m]


def has_number(answer_mentions, spec):
    cands = [spec["value"], *spec.get("alt", [])]
    ms = [x for m in answer_mentions for x in (_as_counts(m) if spec["scale"] == "count" else [m])]
    return any(G.matches_value(m, v, spec["scale"]) for m in ms for v in cands)


def has_fact(answer, spec):
    a = _norm_text(answer)
    return any(_norm_text(x) in a for x in spec["any_of"])


def _resolve(team, season):
    try:
        return C.resolve_team(team, season)["n"]
    except C.ToolError:
        return C._norm(str(team))


def _ok(call):
    return call.get("error") is None and call.get("result") is not None


def sufficient(spec, call):
    """Does this call do what the spec needs?"""
    if call["name"] != spec["tool"] or not _ok(call):
        return False
    a = call.get("args") or {}
    t = spec["tool"]
    try:
        if t == "query_stats" or t == "model_facts":
            return True
        if t == "win_probability":
            sa = int(a["season_a"])
            sb = int(a.get("season_b") or sa)
            ta, tb = _resolve(a["team_a"], sa), _resolve(a["team_b"], sb)
            want = {(_resolve(n, s), s) for n, s in spec["teams"]}
            if {(ta, sa), (tb, sb)} != want:
                return False
            venue = a.get("venue", "neutral")
            home = None if venue == "neutral" else ta if venue == "team_a_home" else tb
            return home == (_resolve(spec["home"], spec["teams"][0][1]) if spec["home"] else None)
        if t in ("team_profile", "advancement_odds"):
            s = int(a["season"])
            return s == spec["season"] and _resolve(a["team"], s) == _resolve(spec["team"], s)
        if t in ("season_rankings", "bracket_forecast", "tournament_matches"):
            return int(a["season"]) == spec["season"]
        if t == "explain_upset":
            s = int(a["season"])
            got = {C._norm(a["team_a"]), C._norm(a["team_b"])}
            return s == spec["season"] and got == {C._norm(x) for x in spec["teams"]}
    except (KeyError, TypeError, ValueError):
        return False
    return False


def grade(case, run):
    answer = run.get("answer") or ""
    calls = run.get("tool_calls") or []
    e = case["expect"]
    g = G.check(answer, calls, case["question"])
    mentions = [G.Mention(**{k: m[k] for k in ("text", "kind", "value", "decimals", "bound", "pair", "start")})
                for m in g["mentions"]]
    # Declined = refusal wording and no checked numbers. "The model can't say exactly why... it gave them 0.4%"
    # is an answer with a caveat, not a refusal.
    phrase = refused(answer)
    did_refuse = phrase and g["n"] == 0
    if e["refuse"]:
        # Right: says the data doesn't cover it, and states nothing made up while doing so.
        correct = float(phrase and not g["ungrounded"])
        did_refuse = phrase
        recall = correct
        num_hits, fact_hits = [], []
    else:
        # A case may list alternatives (an ambiguous question with more than one fair reading); the best one counts.
        best = None
        for alt in e.get("alternatives") or [e]:
            nh = [has_number(mentions, x) for x in alt["numbers"]]
            fh = [has_fact(answer, x) for x in alt["facts"]]
            hits = nh + fh
            score = (float(bool(hits) and all(hits)), sum(hits) / len(hits) if hits else 0.0)
            if best is None or score > best[0]:
                best = (score, nh, fh, alt)
        (correct, recall), num_hits, fact_hits, e = best[0], best[1], best[2], {**e, **best[3]}
    out = {"correct": correct, "recall": round(recall, 4),
           "grounded": round(g["n_grounded"] / g["n"], 4) if g["n"] else 1.0,
           "refused": float(did_refuse)}
    if case["tools"]:
        out["tool_choice"] = float(any(sufficient(s, c) for s in case["tools"] for c in calls))
    detail = {"numbers": [{"label": x["label"], "hit": h} for x, h in zip(e["numbers"], num_hits)],
              "facts": [{"label": x["label"], "hit": h} for x, h in zip(e["facts"], fact_hits)],
              "checked_numbers": g["n"], "ungrounded": g["ungrounded"],
              "tools_called": [c["name"] for c in calls]}
    return {"grade": out, "detail": detail}
