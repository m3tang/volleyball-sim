"""Load model inputs from data/vb.duckdb as per-match outcome counts.

Only clean matches with touch-level data are used (no old "Point X:" format), and only
rallies whose first possession is the serve by the serving team.

A possession is a run of consecutive touches by one team. Each non-serve possession by team T
ends one of three ways (from T's point of view):
  win     T wins the rally (kill, opponent error)
  lose    T loses the rally (attack error, blocked for a point, set or handling error)
  to_opp  the ball goes over and the opponent plays it
Block rows only appear on block points (one row per blocker), so a possession made only of
blocks ends the rally and counts as T's 'lose'. Block touches that stay in play are not
recorded: a covered block shows up as T attacking again within the same possession.
Kinds: 'fb' = the receiving team's first possession after the serve; 'tr' = any later one.
The serve possession ends as ace, service error, or in play.
"""
from pathlib import Path

import duckdb
import numpy as np

DB = Path(__file__).resolve().parent.parent / "data" / "vb.duckdb"


def _build_temp_tables(con, seasons):
    season_list = ", ".join(str(int(s)) for s in seasons)
    # Venue: neutral when the box-score opponent looks like "NJIT @New Rochelle, NY".
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE mm AS
    WITH tm AS (
      SELECT season_year AS season, "Date" AS date_raw, "Team" AS team,
             regexp_replace(regexp_replace("Opponent", '^@ ?', ''), ' @.*$', '') AS opp,
             regexp_matches("Opponent", '^.+ @') AS neutral
      FROM team_match)
    SELECT match_id, season, date, away_team, home_team, winner,
           CASE WHEN neutral THEN NULL ELSE home_team END AS venue_home
    FROM (
      SELECT m.match_id, m.season, m.date, m.away_team, m.home_team, m.winner,
             coalesce(bool_or(tm.neutral), false) AS neutral
      FROM matches m
      LEFT JOIN tm ON tm.season = m.season AND tm.date_raw = m.date_raw
        AND ((tm.team = m.home_team AND tm.opp = m.away_team) OR (tm.team = m.away_team AND tm.opp = m.home_team))
      WHERE m.season IN ({season_list}) AND m.clean AND m.n_point_summary_rows = 0
      GROUP BY m.match_id, m.season, m.date, m.away_team, m.home_team, m.winner)""")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE poss AS
    WITH ev AS (
      SELECT e.match_id, e.set_no, e.rally_no, e.rn, e.actor_team AS team, e.event,
             r.point_team, r.serve_team
      FROM events e
      JOIN rallies r USING (match_id, set_no, rally_no)
      JOIN mm USING (match_id)
      WHERE r.usable AND e.row_type IN ('contact', 'outcome') AND e.actor_team IS NOT NULL),
    seg AS (
      SELECT *, sum(chg) OVER (PARTITION BY match_id, set_no, rally_no ORDER BY rn
                               ROWS UNBOUNDED PRECEDING) AS poss_no
      FROM (SELECT *, (lag(team) OVER (PARTITION BY match_id, set_no, rally_no ORDER BY rn)
                       IS DISTINCT FROM team)::INT AS chg FROM ev))
    SELECT match_id, set_no, rally_no, poss_no, any_value(team) AS team,
           bool_and(event IN ('Block', 'Block error')) AS block_only,
           bool_or(event = 'Serve') AS has_serve,
           any_value(point_team) AS point_team, any_value(serve_team) AS serve_team
    FROM seg GROUP BY ALL""")

    # Rallies we can parse: the serve is the first possession, by the serving team, and only there.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE ok_rally AS
    SELECT match_id, set_no, rally_no FROM poss GROUP BY ALL
    HAVING bool_and(has_serve = (poss_no = 1))
       AND bool_or(poss_no = 1 AND team = serve_team)""")

    con.execute("""
    CREATE OR REPLACE TEMP TABLE poss_out AS
    WITH x AS (
      SELECT p.*,
             lead(team) OVER w AS n1_team, lead(block_only) OVER w AS n1_block, lead(team, 2) OVER w AS n2_team
      FROM poss p JOIN ok_rally USING (match_id, set_no, rally_no)
      WINDOW w AS (PARTITION BY match_id, set_no, rally_no ORDER BY poss_no)),
    y AS (
      SELECT *, lag(has_serve) OVER (PARTITION BY match_id, set_no, rally_no ORDER BY poss_no) AS prev_serve
      FROM x WHERE NOT block_only)
    SELECT match_id, team,
      CASE WHEN has_serve THEN 'serve' WHEN prev_serve THEN 'fb' ELSE 'tr' END AS kind,
      CASE WHEN n1_team IS NULL OR (n1_block AND n2_team IS NULL)
             THEN CASE WHEN point_team = team THEN 'win' ELSE 'lose' END
           ELSE 'to_opp' END AS outcome
    FROM y""")


def load(seasons, db=DB):
    """Return a dict of numpy-array tables for the given seasons."""
    con = duckdb.connect(str(db), read_only=True)
    _build_temp_tables(con, seasons)
    out = {}
    out["matches"] = con.execute("""
      SELECT match_id, season, date, away_team, home_team, winner, venue_home FROM mm ORDER BY date, match_id
    """).fetchnumpy()

    home_expr = """CASE WHEN mm.venue_home = {t} THEN 1 WHEN mm.venue_home IS NULL THEN 0 ELSE -1 END"""
    other = "CASE WHEN {t} = mm.away_team THEN mm.home_team ELSE mm.away_team END"

    # Sideout counts per match and receiving team (same parsed rallies as the touch model).
    out["sideout"] = con.execute(f"""
      SELECT r.match_id, mm.date, r.receive_team AS t, r.serve_team AS o,
             {home_expr.format(t='r.receive_team')} AS home,
             count(*) AS n, count(*) FILTER (WHERE r.sideout) AS k
      FROM rallies r JOIN ok_rally USING (match_id, set_no, rally_no) JOIN mm USING (match_id)
      GROUP BY ALL""").fetchnumpy()

    out["serve"] = con.execute(f"""
      SELECT p.match_id, mm.date, p.team AS t, {other.format(t='p.team')} AS o,
             {home_expr.format(t='p.team')} AS home,
             count(*) FILTER (WHERE outcome = 'win') AS ace,
             count(*) FILTER (WHERE outcome = 'lose') AS serr,
             count(*) FILTER (WHERE outcome NOT IN ('win', 'lose')) AS inplay
      FROM poss_out p JOIN mm USING (match_id) WHERE kind = 'serve'
      GROUP BY ALL""").fetchnumpy()

    for kind in ("fb", "tr"):
        out[kind] = con.execute(f"""
          SELECT p.match_id, mm.date, p.team AS t, {other.format(t='p.team')} AS o,
                 {home_expr.format(t='p.team')} AS home,
                 count(*) FILTER (WHERE outcome = 'win') AS win,
                 count(*) FILTER (WHERE outcome = 'lose') AS lose,
                 count(*) FILTER (WHERE outcome = 'to_opp') AS to_opp
          FROM poss_out p JOIN mm USING (match_id) WHERE kind = '{kind}'
          GROUP BY ALL""").fetchnumpy()
    con.close()
    return out


class TeamIndex:
    """Maps team names to integer ids across seasons."""

    def __init__(self):
        self.ids = {}

    def __call__(self, names):
        return np.array([self.ids.setdefault(n, len(self.ids)) for n in names], dtype=np.int64)

    def __len__(self):
        return len(self.ids)
