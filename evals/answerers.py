"""Answerers: anything that takes a question and returns an answer plus the tool calls behind it.

    answer(case) -> {"answer": str, "tool_calls": [...], "model": str|None, "usage": dict|None,
                     "stop_reason": str|None, "system": str|None}

The real one is the agent (step 3), loaded by `--answerer module:function`. The ones here are
free and exist to test the harness and grader, per the eval checklist:
  oracle      calls the case's first acceptable tool for real and states the expected values;
              must score ~100% on everything (if not, the grader or a golden case is broken)
  no_tools    the oracle's text with no tool calls: correct, but every number is ungrounded
  perturbed   the oracle with each expected number nudged: wrong, and mostly ungrounded
  refuse_all  declines everything: right only on out-of-scope cases
  null        empty answer: right on nothing
"""
from rallytools import core as C
from rallytools.registry import call

REFUSAL_TEXT = ("The data doesn't cover that. It has Division I women's volleyball play-by-play and box "
                "scores from 2020 to 2025, and nothing about injuries, lineups, coaches, betting or later seasons.")


def spec_to_call(spec, case):
    t = spec["tool"]
    if t == "win_probability":
        (a, sa), (b, sb) = spec["teams"]
        venue = "neutral" if spec["home"] is None else "team_a_home" if spec["home"] == a else "team_b_home"
        return t, {"team_a": a, "season_a": sa, "team_b": b, "season_b": sb, "venue": venue}
    if t in ("team_profile", "advancement_odds"):
        return t, {"team": spec["team"], "season": spec["season"]}
    if t == "season_rankings":
        return t, {"season": spec["season"], "top": 25}
    if t == "bracket_forecast":
        return t, {"season": spec["season"], "top": 64}
    if t == "tournament_matches":
        return t, {"season": spec["season"], "upsets_only": True}
    if t == "explain_upset":
        return t, {"season": spec["season"], "team_a": spec["teams"][0], "team_b": spec["teams"][1]}
    if t == "query_stats":
        return t, {"sql": case["ref"]}
    raise ValueError(f"no oracle call for {t}")


def _fmt(x, shift=False):
    v = x["value"]
    if x["scale"] == "prob":
        if shift:
            v = v + 0.07 if v < 0.9 else v - 0.07
        return C.odds_text(v)
    if x["scale"] == "decimal":
        return f"{v + (0.021 if shift else 0):.3f}"
    return f"{v + (3 if shift else 0):,}"


def _compose(case, shift=False):
    e = case["expect"]
    if e["refuse"]:
        return REFUSAL_TEXT
    parts = [_fmt(x, shift) for x in e["numbers"]] + [f["any_of"][0] for f in e["facts"]]
    return "Per the rally model: " + "; ".join(parts) + "."


def _result(answer, calls):
    return {"answer": answer, "tool_calls": calls, "model": None, "usage": None, "stop_reason": "end_turn", "system": None}


def oracle(case):
    calls = []
    if case["tools"]:
        calls.append(call(*spec_to_call(case["tools"][0], case)))
    return _result(_compose(case), calls)


def no_tools(case):
    return _result(_compose(case), [])


def perturbed(case):
    r = oracle(case)
    r["answer"] = _compose(case, shift=True)
    return r


def refuse_all(case):
    return _result(REFUSAL_TEXT, [])


def null(case):
    return _result("", [])


BUILTIN = {"oracle": oracle, "no_tools": no_tools, "perturbed": perturbed, "refuse_all": refuse_all, "null": null}
