"""End-to-end: launch the MCP server as a subprocess over stdio, as an MCP client would."""
import json
import sys
from pathlib import Path

import anyio
import pytest

mcp = pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _run(fn):
    async def main():
        params = StdioServerParameters(command=sys.executable, args=["-m", "rallytools.server"], cwd=str(ROOT))
        async with Client(params) as client:
            return await fn(client)
    return anyio.run(main)


def _payload(res):
    return json.loads(res.content[0].text)


def test_stdio_lists_tools_and_answers():
    async def go(client):
        tools = {t.name: t for t in (await client.list_tools()).tools}
        res = await client.call_tool("win_probability",
                                     {"team_a": "Nebraska", "season_a": 2025, "team_b": "Texas A&M"})
        bad = await client.call_tool("query_stats", {"sql": "drop table matches"})
        return tools, res, bad

    tools, res, bad = _run(go)
    assert {"win_probability", "team_profile", "explain_upset", "query_stats", "bracket_forecast"} <= set(tools)
    assert all(t.annotations.read_only_hint for t in tools.values())
    venue = tools["win_probability"].input_schema["properties"]["venue"]
    assert set(venue["enum"]) == {"neutral", "team_a_home", "team_b_home"}
    assert not res.is_error and _payload(res)["p_team_a_wins"] == pytest.approx(0.873, abs=1e-4)
    assert bad.is_error and "only SELECT" in bad.content[0].text
