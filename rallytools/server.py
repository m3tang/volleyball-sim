"""MCP server for the Rally Odds volleyball model.

Run:   .venv/bin/python -m rallytools.server            (stdio, for Claude Desktop / Claude Code)
Test:  .venv/bin/python -m rallytools.server --check    (calls every tool once and exits)

Every tool is read-only and deterministic. Model numbers come from web/data.json, the same
published parameters the site uses, so answers match what readers see on the site.
"""
import functools
import sys
from typing import Annotated, Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import core as C
from . import sql as Q

INSTRUCTIONS = """\
Tools for NCAA Division I women's volleyball, 2020-2025, from a rally-by-rally model (Rally Odds).
- Match odds between any two team-seasons (cross-season "dream matchups" too): win_probability.
- Odds from a score in progress (sets, points, who serves): live_win_probability.
- A team's strengths: team_profile. Rankings: season_rankings.
- NCAA tournaments 2021-2025: bracket_forecast, advancement_odds, tournament_matches, explain_upset.
- Anything else in the data (box scores, players, set scores, rallies): sql_schema, then query_stats.
- Coverage and headline accuracy: model_facts.
Every number you report should come from a tool result. The data has no injuries, lineups,
betting lines, or anything after the 2025 season; say so instead of guessing."""

mcp = MCPServer("rally-odds", title="Rally Odds volleyball model", instructions=INSTRUCTIONS, version="0.1.0")

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)

Season = Annotated[int, Field(ge=2020, le=2025, description="Fall season year, 2020-2025")]
Team = Annotated[str, Field(min_length=2, description="Team name as commonly written, e.g. 'Texas A&M', 'Penn State'")]
BracketSeason = Annotated[int, Field(ge=2021, le=2025, description="Tournament season, 2021-2025")]


def tool(fn):
    """Register fn as a read-only tool, turning caller-fixable errors into ToolError."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except C.ToolError as e:
            raise ToolError(str(e)) from e
    return mcp.tool(annotations=READ_ONLY)(wrapper)


@tool
def win_probability(
    team_a: Team, season_a: Season, team_b: Team,
    season_b: Annotated[Optional[int], Field(ge=2020, le=2025, description="team_b's season; defaults to season_a")] = None,
    venue: Literal["neutral", "team_a_home", "team_b_home"] = "neutral",
) -> dict:
    """Exact probability that team_a beats team_b in a best-of-five match, with set and final-score
    probabilities and each team's rally-step rates when receiving serve. Uses end-of-season ratings.
    Teams can be from different seasons (a hypothetical matchup)."""
    return C.win_probability(team_a, season_a, team_b, season_b, venue)


@tool
def live_win_probability(
    team_a: Team, season_a: Season, team_b: Team,
    season_b: Annotated[Optional[int], Field(ge=2020, le=2025, description="team_b's season; defaults to season_a")] = None,
    venue: Literal["neutral", "team_a_home", "team_b_home"] = "neutral",
    sets_a: Annotated[int, Field(ge=0, le=2, description="Sets team_a has won")] = 0,
    sets_b: Annotated[int, Field(ge=0, le=2, description="Sets team_b has won")] = 0,
    points_a: Annotated[int, Field(ge=0, le=60, description="team_a's points in the set in progress")] = 0,
    points_b: Annotated[int, Field(ge=0, le=60, description="team_b's points in the set in progress")] = 0,
    serving: Literal["team_a", "team_b"] = "team_a",
) -> dict:
    """Probability that team_a wins the match from a score in progress (sets won, points in the current set,
    who serves next), with the odds before the match and after each team wins the next rally."""
    return C.live_win_probability(team_a, season_a, team_b, season_b, venue, sets_a, sets_b, points_a, points_b, serving)


@tool
def team_profile(team: Team, season: Season) -> dict:
    """A team-season's rank, record, rating, and the model's 12 rally-step numbers (serving, passing,
    first attack, defending the first attack, scramble offense and defense), each as counted in its
    matches and adjusted for opponents, next to the league average."""
    return C.team_profile(team, season)


@tool
def season_rankings(season: Season, top: Annotated[int, Field(ge=1, le=400)] = 25) -> dict:
    """End-of-season model rankings by rating. 2020-2024 publish only the top 64 teams."""
    return C.season_rankings(season, top)


@tool
def bracket_forecast(season: BracketSeason, top: Annotated[int, Field(ge=1, le=64)] = 10) -> dict:
    """The model's pre-tournament NCAA bracket forecast: title odds, its champion pick, the actual
    champion and that team's pre-tournament odds, bracket score and matches called."""
    return C.bracket_forecast(season, top)


@tool
def advancement_odds(team: Team, season: BracketSeason) -> dict:
    """One tournament team's pre-tournament odds of reaching each round, and how far it went."""
    return C.advancement_odds(team, season)


@tool
def tournament_matches(
    season: BracketSeason,
    team: Annotated[Optional[str], Field(description="Only this team's matches")] = None,
    upsets_only: bool = False,
) -> dict:
    """NCAA tournament results for a season with the model's pre-match odds for the winner.
    An upset is a match the winner had under a 50% chance to win."""
    return C.tournament_matches(season, team, upsets_only)


@tool
def explain_upset(season: BracketSeason, team_a: Team, team_b: Team) -> dict:
    """Why an NCAA tournament match went the way it did: the points each part of the game swung
    toward the winner, comparing what happened with what the pre-tournament model expected for
    that matchup and venue. Works for any tournament match, upset or not. Reasons are ranked by points
    swung; biggest_factor is rank 1, so lead with it."""
    return C.explain_upset(season, team_a, team_b)


@tool
def sql_schema() -> dict:
    """Tables and columns query_stats can read, with notes on naming and quoting."""
    return Q.schema()


@tool
def query_stats(
    sql: Annotated[str, Field(description="One DuckDB SELECT statement over the tables in sql_schema")],
    max_rows: Annotated[int, Field(ge=1, le=Q.MAX_ROWS)] = 50,
) -> dict:
    """Run one read-only SQL SELECT over the play-by-play and box-score database. Only the tables in
    sql_schema are allowed; results are capped and long queries are stopped. On an error, read the
    message, fix the query and try again."""
    return Q.query_stats(sql, max_rows)


@tool
def model_facts() -> dict:
    """What the data covers and doesn't, and the model's tested accuracy (2025 held out, XGBoost baseline)."""
    return C.model_facts()


def _check():
    """Call each tool once through the MCP layer and print a one-line summary."""
    import anyio

    calls = [
        ("win_probability", {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas", "season_b": 2023}),
        ("live_win_probability", {"team_a": "Texas", "season_a": 2025, "team_b": "Nebraska", "sets_a": 1, "sets_b": 2, "points_a": 20, "points_b": 23, "serving": "team_b"}),
        ("team_profile", {"team": "Penn State", "season": 2024}),
        ("season_rankings", {"season": 2025, "top": 3}),
        ("bracket_forecast", {"season": 2025, "top": 3}),
        ("advancement_odds", {"team": "Texas A&M", "season": 2025}),
        ("tournament_matches", {"season": 2025, "upsets_only": True}),
        ("explain_upset", {"season": 2025, "team_a": "Texas A&M", "team_b": "Kentucky"}),
        ("sql_schema", {}),
        ("query_stats", {"sql": "select season, count(*) n from matches group by 1 order by 1"}),
        ("model_facts", {}),
        ("win_probability", {"team_a": "Nebrasak", "season_a": 2025, "team_b": "Texas"}),
        ("query_stats", {"sql": "delete from matches"}),
    ]

    async def run():
        tools = await mcp.list_tools()
        print(f"{len(tools)} tools: {', '.join(t.name for t in tools)}")
        for name, args in calls:
            try:
                res = await mcp.call_tool(name, args)
                print(f"  ok   {name}: {str(res)[:110]}")
            except ToolError as e:
                print(f"  err  {name}: {str(e)[:110]}")

    anyio.run(run)


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
    else:
        mcp.run()
