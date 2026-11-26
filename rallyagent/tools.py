"""Tool definitions for the Messages API, matching the MCP server's tools (rallytools/server.py).

Schemas are strict (additionalProperties false), so tool inputs always validate. Descriptions say
when to call each tool, not just what it returns.
"""
import copy
import json
from functools import lru_cache

from rallytools.registry import call

SEASON = {"type": "integer", "description": "Fall season year, 2020-2025"}
TEAM = {"type": "string", "description": "Team name as commonly written, e.g. 'Texas A&M', 'Penn State', 'USC'"}
BRACKET_SEASON = {"type": "integer", "description": "NCAA tournament season, 2021-2025"}


def _tool(name, description, props, required):
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


TOOLS = [
    _tool("win_probability",
          "Exact odds that team_a beats team_b, with set and final-score probabilities and each team's rally-step "
          "rates. Call for any 'who wins', 'what are the chances', 'most likely score' or 'could X beat Y' question, "
          "including hypothetical matchups between teams from different seasons. Uses end-of-season ratings.",
          {"team_a": TEAM, "season_a": SEASON, "team_b": TEAM,
           "season_b": {"type": "integer", "description": "team_b's season if different from season_a"},
           "venue": {"type": "string", "enum": ["neutral", "team_a_home", "team_b_home"],
                     "description": "Where the match is played; neutral unless the question says whose gym"}},
          ["team_a", "season_a", "team_b"]),
    _tool("live_win_probability",
          "Odds that team_a wins the match from a score in progress: sets won, points in the current set and who "
          "serves next. Also returns the odds before the match and after either team wins the next rally. Call for "
          "'X leads 2-1', 'down 20-23 in the fourth', 'what are the odds from here' questions.",
          {"team_a": TEAM, "season_a": SEASON, "team_b": TEAM,
           "season_b": {"type": "integer", "description": "team_b's season if different from season_a"},
           "venue": {"type": "string", "enum": ["neutral", "team_a_home", "team_b_home"],
                     "description": "Where the match is played; neutral unless the question says whose gym"},
           "sets_a": {"type": "integer", "description": "Sets team_a has won, 0-2"},
           "sets_b": {"type": "integer", "description": "Sets team_b has won, 0-2"},
           "points_a": {"type": "integer", "description": "team_a's points in the set in progress"},
           "points_b": {"type": "integer", "description": "team_b's points in the set in progress"},
           "serving": {"type": "string", "enum": ["team_a", "team_b"],
                       "description": "Who serves next: the team that won the last rally; team_a if unknown"}},
          ["team_a", "season_a", "team_b", "sets_a", "sets_b", "points_a", "points_b"]),
    _tool("team_profile",
          "A team-season's rank, record, rating and its 12 rally-step numbers (serving, passing, first attack, "
          "defending the first attack, scramble offense and defense), counted and opponent-adjusted, next to league "
          "averages. Call for questions about how good a team was, its record, or a specific strength or weakness.",
          {"team": TEAM, "season": SEASON}, ["team", "season"]),
    _tool("season_rankings",
          "End-of-season model rankings by rating. Call for 'best team', 'top N', or 'where did X rank'. "
          "2020-2024 publish only the top 64 teams.",
          {"season": SEASON, "top": {"type": "integer", "description": "How many teams, default 25"}}, ["season"]),
    _tool("bracket_forecast",
          "The model's pre-tournament NCAA bracket forecast for a season: title odds, the model's champion pick, the "
          "actual champion and its pre-tournament odds, bracket score versus picking the better record, matches called.",
          {"season": BRACKET_SEASON, "top": {"type": "integer", "description": "How many title-odds rows, default 10"}},
          ["season"]),
    _tool("advancement_odds",
          "One tournament team's pre-tournament odds of reaching each round (Sweet 16, Elite 8, Final Four, final, "
          "title) and how far it actually went. Call for a specific team's tournament odds.",
          {"team": TEAM, "season": BRACKET_SEASON}, ["team", "season"]),
    _tool("tournament_matches",
          "NCAA tournament results for a season with the model's pre-match odds for each winner. Call to find "
          "upsets (upsets_only), a team's tournament path, or the biggest upset.",
          {"season": BRACKET_SEASON, "team": {"type": "string", "description": "Only this team's matches"},
           "upsets_only": {"type": "boolean", "description": "Only matches the winner had under 50% to win"}},
          ["season"]),
    _tool("explain_upset",
          "Why one NCAA tournament match went the way it did: points each part of the game swung toward the winner, "
          "actual versus what the pre-tournament model expected. Call for 'how did X beat Y' or 'why did X win' "
          "about a tournament match. Reasons come ranked by points swung; biggest_factor is rank 1. Lead your "
          "explanation with it, and keep the ranking when you mention others.",
          {"season": BRACKET_SEASON, "team_a": TEAM, "team_b": TEAM}, ["season", "team_a", "team_b"]),
    _tool("sql_schema", "Tables and columns query_stats can read. Call before writing SQL if unsure of a column.", {}, []),
    _tool("query_stats",
          "Run one read-only DuckDB SELECT over the box-score and play-by-play tables. Call for stats the model tools "
          "don't return: player and team box-score totals, leaders, head-to-head results, set scores, league-wide "
          "rally rates. On an error, read the message, fix the query and retry.",
          {"sql": {"type": "string", "description": "One SELECT statement"},
           "max_rows": {"type": "integer", "description": "Row cap, default 50, max 200"}}, ["sql"]),
    _tool("model_facts",
          "What the data covers and does not, and the model's tested accuracy. Call when unsure whether a question "
          "is answerable from this data.", {}, []),
]


@lru_cache(maxsize=4096)
def _cached(name, args_json):
    return call(name, json.loads(args_json))


def run_tool(name, args):
    """Run a tool, caching by (name, args): every tool is deterministic, so repeats are free."""
    rec = _cached(name, json.dumps(args or {}, sort_keys=True))
    return copy.deepcopy(rec)


def result_text(rec):
    """What the model sees: compact JSON, or the error message for the model to act on."""
    if rec["error"] is not None:
        return rec["error"]
    return json.dumps(rec["result"], separators=(",", ":"), default=str)
