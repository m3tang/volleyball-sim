"""Read-only SQL over data/vb.duckdb, with guardrails for a model writing the queries.

Layers, outermost first:
  1. exactly one statement, and it must be a SELECT (WITH ... SELECT included)
  2. every table it reads must be on ALLOWED (DuckDB's own parser lists them)
  3. the connection is read-only with external access off, so file readers
     (read_csv, read_parquet, ATTACH, COPY) fail even if they slip past 1-2
  4. results are capped at max_rows, and a query running past timeout_s is interrupted
Errors come back as ToolError with a message the caller can act on (the bad column, the
allowed tables), so a model can fix its query and retry.
"""
import datetime as dt
import decimal
import threading
import duckdb

from vbsim.data import DB

from .core import ToolError

# events (30M touch rows) is left out on purpose: scans of it are slow, and every question
# the agent is meant to answer is covered by rallies and the box-score tables.
ALLOWED = {
    "matches": "One row per match. match_id, season, date, away_team, home_team, n_sets, away_sets, home_sets, "
               "winner, clean (play-by-play passed checks), n_usable_rallies.",
    "sets": "One row per set. match_id, set_no, away_final, home_final, winner_side ('away'/'home'), legal_final.",
    "rallies": "One row per rally (4.5M). match_id, season, set_no, rally_no, away_score_before, home_score_before, "
               "serve_team, receive_team, point_team, sideout (receiving team won), end_event ('Kill', 'Ace', "
               "'Service error', 'Block', 'Attack error', ...), n_contacts, server, usable.",
    "team_match": 'Box score per team per match. season_year, "Date" (text MM/DD/YYYY), "Team", "Conference", '
                  '"Opponent" (prefix "@" = away; " @City" = neutral), "Result" (e.g. "W 3-1"), "Kills", "Errors", '
                  '"Total Attacks", "Hit Pct", "Assists", "Aces", "SErr", "Digs", "RErr", "Block Solos", '
                  '"Block Assists", "BErr", "TB", "PTS". Quote mixed-case columns.',
    "team_season": 'Season box-score totals per team. season_year, "Team", "Conference", "S" (sets), "Kills", '
                   '"Errors", "Total Attacks", "Hit Pct", "Aces", "SErr", "Digs", "RErr", "TB", "PTS", "SrvAtt".',
    "player_match": 'Box score per player per match. season_year, team, "Date", "Player", "Opponent Team", '
                    '"Location", "Kills", "Errors", "TotalAttacks", "HitPct", "Assists", "Aces", "SErr", "Digs", '
                    '"BlockSolos", "BlockAssists", "PTS".',
    "player_season": 'Season totals per player. season_year, "Team", "Conference", "Player", "Yr", "Pos", "Ht", '
                     '"GP", "Kills", "Errors", "Total Attacks", "Hit Pct", "Assists", "Aces", "SErr", "Digs", '
                     '"Block Solos", "Block Assists", "TB", "PTS", "Hometown".',
}
NOTES = ("Team names match the site (e.g. 'Texas A&M', 'Penn St.'). matches.season and team_match.season_year are "
         "the fall season year. The NCAA tournament rows in team_match have the event name in \"Opponent\".")
MAX_ROWS = 200
TIMEOUT_S = 10.0


_lock = threading.Lock()
_conn = None


def _con():
    """One shared read-only connection (created once, thread-safely; each query uses its own cursor)."""
    global _conn
    with _lock:
        if _conn is None:
            con = duckdb.connect(str(DB), read_only=True)
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
            _conn = con
    return _conn


def schema():
    """Allowed tables with column notes, for writing queries."""
    return {"tables": ALLOWED, "notes": NOTES, "max_rows": MAX_ROWS, "timeout_seconds": TIMEOUT_S}


def _check(sql):
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ToolError("empty query")
    try:
        stmts = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        raise ToolError(f"SQL parse error: {e}")
    if len(stmts) != 1:
        raise ToolError("send exactly one statement")
    if stmts[0].type != duckdb.StatementType.SELECT:
        raise ToolError(f"only SELECT queries are allowed (got {stmts[0].type.name})")
    con = _con()
    try:
        tables = con.get_table_names(sql)
    except duckdb.Error as e:
        raise ToolError(f"SQL parse error: {e}")
    bad = sorted(t for t in tables if t.split(".")[-1].lower() not in ALLOWED)
    if bad:
        raise ToolError(f"table(s) not allowed: {', '.join(bad)}. Allowed: {', '.join(sorted(ALLOWED))}")
    return sql


def _jsonable(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, int) and abs(v) > 2 ** 53:
        return str(v)
    return v


def query_stats(sql, max_rows=MAX_ROWS, timeout_s=TIMEOUT_S):
    """Run one read-only SELECT. Returns columns, rows (capped) and whether rows were cut off."""
    sql = _check(sql)
    max_rows = max(1, min(int(max_rows), MAX_ROWS))
    cur = _con().cursor()
    timer = threading.Timer(timeout_s, cur.interrupt)
    timer.start()
    try:
        cur.execute(f"SELECT * FROM ({sql}) AS q LIMIT {max_rows + 1}")
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    except duckdb.InterruptException:
        raise ToolError(f"query ran longer than {timeout_s:.0f}s; add filters or aggregate")
    except duckdb.Error as e:
        raise ToolError(f"{type(e).__name__}: {e}")
    finally:
        timer.cancel()
        cur.close()
    truncated = len(rows) > max_rows
    rows = [[_jsonable(v) for v in r] for r in rows[:max_rows]]
    return {"columns": cols, "rows": rows, "row_count": len(rows), "truncated": truncated}
