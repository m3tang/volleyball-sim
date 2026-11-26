"""Name -> function for every tool, with the same argument names as the MCP server.

The eval harness and the agent call tools through here, so they see exactly what an MCP client
sees. call() returns {"name", "args", "result", "error"}; caller-fixable problems come back in
"error" (the agent hands that text to the model), never as an exception.
"""
from . import core as C
from . import sql as Q

TOOLS = {
    "win_probability": lambda team_a, season_a, team_b, season_b=None, venue="neutral":
        C.win_probability(team_a, season_a, team_b, season_b, venue),
    "live_win_probability": lambda team_a, season_a, team_b, season_b=None, venue="neutral", sets_a=0, sets_b=0,
                                   points_a=0, points_b=0, serving="team_a":
        C.live_win_probability(team_a, season_a, team_b, season_b, venue, sets_a, sets_b, points_a, points_b, serving),
    "team_profile": lambda team, season: C.team_profile(team, season),
    "season_rankings": lambda season, top=25: C.season_rankings(season, top),
    "bracket_forecast": lambda season, top=10: C.bracket_forecast(season, top),
    "advancement_odds": lambda team, season: C.advancement_odds(team, season),
    "tournament_matches": lambda season, team=None, upsets_only=False: C.tournament_matches(season, team, upsets_only),
    "explain_upset": lambda season, team_a, team_b: C.explain_upset(season, team_a, team_b),
    "sql_schema": lambda: Q.schema(),
    "query_stats": lambda sql, max_rows=50: Q.query_stats(sql, max_rows),
    "model_facts": lambda: C.model_facts(),
}


def call(name, args=None):
    args = dict(args or {})
    rec = {"name": name, "args": args, "result": None, "error": None}
    fn = TOOLS.get(name)
    if fn is None:
        rec["error"] = f"unknown tool {name!r}; tools: {', '.join(TOOLS)}"
        return rec
    try:
        rec["result"] = fn(**args)
    except C.ToolError as e:
        rec["error"] = str(e)
    except TypeError as e:  # wrong or missing arguments
        rec["error"] = f"bad arguments for {name}: {e}"
    return rec
