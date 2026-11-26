"""Typed, deterministic tools over the published Rally Odds model.

core.py   model tools (win probability, team profile, brackets, upset breakdowns), read from
          web/data.json so every number matches the public site exactly
sql.py    read-only DuckDB queries with a table allowlist, a row limit and a timeout
server.py MCP server exposing both (needs Python 3.10+ and the `mcp` package)
"""
