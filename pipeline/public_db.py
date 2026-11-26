"""Build a small copy of data/vb.duckdb for the public agent: serve/vb_public.duckdb.

Keeps the box-score tables, matches and sets. Drops the 30M touch events and 4.5M rallies,
replacing the rallies with rally_summary: per season and rally-ending event, how many rallies and
how often the receiving team won them. That covers the league-wide rally questions the agent gets.

Usage:  python -m pipeline.public_db
"""
from pathlib import Path

import duckdb

from vbsim.data import DB

OUT = Path(__file__).resolve().parent.parent / "serve" / "vb_public.duckdb"
KEEP = ["matches", "sets", "team_match", "team_season", "player_match", "player_season"]


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.unlink(missing_ok=True)
    con = duckdb.connect(str(OUT))
    con.execute(f"ATTACH '{DB}' AS src (READ_ONLY)")
    for t in KEEP:
        con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}"')
    con.execute("""
      CREATE TABLE rally_summary AS
      SELECT season, coalesce(end_event, 'unknown') AS end_event, count(*) AS rallies,
             avg(sideout::INT) AS sideout_rate
      FROM src.rallies WHERE usable GROUP BY ALL
      UNION ALL
      SELECT season, 'all', count(*), avg(sideout::INT) FROM src.rallies WHERE usable GROUP BY season
      ORDER BY season, rallies DESC""")
    con.execute("DETACH src")
    con.execute("CHECKPOINT")
    con.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
