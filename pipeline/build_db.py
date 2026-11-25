"""Build data/vb.duckdb from the raw CSVs in data/raw/.

Tables produced:
  events       every play-by-play row, cleaned: match_id, rally_no, filled-in score, row_type, actor_team
  rallies      one row per rally: score before, serving team, point winner, ending event
  sets         one row per set: final score, winner, whether the score is a legal final
  matches      one row per match: teams, sets won, data-quality flags
  team_match, team_season, player_match, player_season   box-score tables as published

Facts about the source data that this script relies on (checked on 2024 D1):
  * `score` is "away-home" and is the score AFTER the event on that row. A rally's
    last row is the one where the score changes.
  * During media timeouts `score` holds the text "Media timeout" for the rally's
    middle rows; the rally's final row carries a real score, so we fill forward.
  * On error rows (Attack error, Service error, ...) `team` is the team that WON the
    point, not the team that made the error. Set/ball-handling errors are mixed. So
    actor_team comes from the player's team, learned from their unambiguous touches.
  * A few percent of matches use an older format with "Point X: ..." summary rows,
    substitutions and timeouts in the `event` column. Rallies are still found from
    score changes; those rows get a row_type other than 'contact'/'outcome'.

Usage:  python pipeline/build_db.py
"""
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "vb.duckdb"

CONTACTS = ("Serve", "Reception", "Set", "Attack", "Dig", "Block")
OUTCOMES = ("Kill", "First ball kill", "Ace", "Attack error", "Service error",
            "Set error", "Block error", "Ball handling error", "Dig error")
# Events whose `team` column reliably names the team that made the touch.
RELIABLE_TEAM = ("Serve", "Reception", "Set", "Attack", "Dig", "Kill", "First ball kill", "Ace")
# Errors where `team` is the opponent of the player who erred.
TEAM_IS_OPPONENT = ("Attack error", "Service error")


def sql_list(xs):
    return "(" + ", ".join("'" + x.replace("'", "''") + "'" for x in xs) + ")"


SEASONS = None  # set from --seasons; None means every file found


def files(level):
    paths = sorted(RAW.glob(f"wvb_{level}_div1_*.csv"))
    return [p for p in paths if SEASONS is None or season_of(p) in SEASONS]


def season_of(path):
    return int(path.stem.rsplit("_", 1)[-1])


def load_box_scores(con):
    for level, table in [("teammatch", "team_match"), ("teamseason", "team_season"),
                         ("playermatch", "player_match"), ("playerseason", "player_season")]:
        paths = files(level)
        if not paths:
            print(f"  (no {level} files; skipping {table})")
            continue
        union = " UNION ALL BY NAME ".join(
            f"SELECT {season_of(p)} AS season_year, * FROM read_csv('{p}', header = true, nullstr = 'NA')"
            for p in paths)
        con.execute(f"CREATE TABLE {table} AS {union}")


def load_pbp_file(con, path):
    """Load one season's play-by-play CSV into temp table raw_pbp, keeping file row order."""
    # Only some years have contestid (2025 does); use it as the match key when present.
    with open(path) as f:
        contest = "contestid" if "contestid" in f.readline() else "NULL::VARCHAR"
    # One thread so row_number() follows file order; rally order depends on it.
    con.execute("SET threads = 1")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE raw_pbp AS
        SELECT {season_of(path)} AS season, row_number() OVER () AS rn, {contest} AS contest_id,
               date, "set" AS set_raw, away_team, home_team, score, team, event, player, description
        FROM read_csv('{path}', all_varchar = true, header = true)""")
    con.execute("RESET threads")


def build_events(con):
    # Every match lies within one season, so process seasons one at a time, straight from
    # each CSV. Doing everything in one pass needed >5 GB of memory and spilled ~10 GB to disk.
    pbp = files("pbp")
    if not pbp:
        sys.exit(f"No play-by-play files in {RAW}. Run pipeline/download.py first.")
    for i, path in enumerate(pbp):
        t = time.time()
        load_pbp_file(con, path)
        build_events_season(con, season_of(path), first=(i == 0))
        print(f"  season {season_of(path)}: {time.time() - t:.0f}s", flush=True)
    con.execute("DROP TABLE raw_pbp")


def build_events_season(con, season, first):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE e0 AS
    SELECT season, rn, contest_id, date, away_team, home_team, TRY_CAST(set_raw AS INT) AS set_raw_no,
           score, team, event, player, description,
           CASE WHEN regexp_full_match(score, '\\d+-\\d+') THEN split_part(score, '-', 1)::INT END AS a_raw,
           CASE WHEN regexp_full_match(score, '\\d+-\\d+') THEN split_part(score, '-', 2)::INT END AS h_raw,
           coalesce(contest_id, date || '|' || away_team || '|' || home_team) AS mkey
    FROM raw_pbp""")

    # A new match starts when the match key changes, or the set number drops within
    # the same key (two matches between the same teams on the same day). The set-drop
    # check only looks at rows that have a set number.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE set_drops AS
    SELECT season, rn FROM (
      SELECT season, rn, mkey, set_raw_no, lag(set_raw_no) OVER w AS prev_set, lag(mkey) OVER w AS prev_key
      FROM e0 WHERE set_raw_no IS NOT NULL
      WINDOW w AS (ORDER BY rn))
    WHERE prev_key = mkey AND set_raw_no < prev_set""")
    # match_id = season * 100000 + match number within the season (unique across seasons).
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE e1_all AS
    SELECT *, {season} * 100000 + sum(new_match) OVER (ORDER BY rn ROWS UNBOUNDED PRECEDING) AS match_id
    FROM (
      SELECT e0.*,
        CASE WHEN lag(mkey) OVER (ORDER BY e0.rn) IS DISTINCT FROM mkey
               OR d.rn IS NOT NULL THEN 1 ELSE 0 END AS new_match
      FROM e0 LEFT JOIN set_drops d USING (season, rn))""")
    # Some files list a match twice (once from each team's page). With a contest ID we
    # can see that; keep the first copy.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE e1 AS
    SELECT * FROM e1_all
    WHERE contest_id IS NULL
       OR match_id IN (SELECT min(match_id) FROM e1_all WHERE contest_id IS NOT NULL
                       GROUP BY contest_id)""")
    dropped = con.execute("SELECT count(DISTINCT match_id) FROM e1_all").fetchone()[0] - \
        con.execute("SELECT count(DISTINCT match_id) FROM e1").fetchone()[0]
    if dropped:
        print(f"  season {season}: dropped {dropped} duplicate match copies")

    # Fill set number and score forward; compute score change on each row.
    con.execute("""
    CREATE OR REPLACE TEMP TABLE e2 AS
    SELECT *, a - prev_a AS da, h - prev_h AS dh
    FROM (
      SELECT *,
        coalesce(last_value(a_raw IGNORE NULLS) OVER ws, 0) AS a,
        coalesce(last_value(h_raw IGNORE NULLS) OVER ws, 0) AS h,
        coalesce(last_value(a_raw IGNORE NULLS) OVER wp, 0) AS prev_a,
        coalesce(last_value(h_raw IGNORE NULLS) OVER wp, 0) AS prev_h
      FROM (SELECT *, coalesce(last_value(set_raw_no IGNORE NULLS) OVER
                    (PARTITION BY match_id ORDER BY rn ROWS UNBOUNDED PRECEDING), 1) AS set_no
            FROM e1)
      WINDOW ws AS (PARTITION BY match_id, set_no ORDER BY rn ROWS UNBOUNDED PRECEDING),
             wp AS (PARTITION BY match_id, set_no ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))""")

    # Player -> team per match, from touches where `team` is reliable.
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE roster AS
    SELECT match_id, player, mode(team) AS player_team
    FROM e2 WHERE event IN {sql_list(RELIABLE_TEAM)} AND player IS NOT NULL
    GROUP BY ALL""")

    con.execute(("CREATE TABLE match_info AS " if first else "INSERT INTO match_info ") +
                "SELECT DISTINCT match_id, season, contest_id, date, away_team, home_team FROM e2")
    con.execute(("CREATE TABLE events AS " if first else "INSERT INTO events ") + f"""
    SELECT e.match_id, e.season, e.rn, e.set_no,
      coalesce(sum((e.da <> 0 OR e.dh <> 0)::INT) OVER
        (PARTITION BY e.match_id, e.set_no ORDER BY e.rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) + 1
        AS rally_no,
      e.prev_a AS away_score_before, e.prev_h AS home_score_before, e.a AS away_score, e.h AS home_score,
      CASE WHEN e.da = 1 AND e.dh = 0 THEN e.away_team
           WHEN e.dh = 1 AND e.da = 0 THEN e.home_team END AS point_team,
      (e.da <> 0 OR e.dh <> 0) AS score_changed,
      (e.da < 0 OR e.dh < 0 OR e.da + e.dh > 1) AS score_anomaly,
      e.score = 'Media timeout' AS in_media_timeout,
      CASE WHEN e.event IN {sql_list(CONTACTS)} THEN 'contact'
           WHEN e.event IN {sql_list(OUTCOMES)} THEN 'outcome'
           WHEN e.event IN ('Challenge request', 'Sanction') THEN 'admin'
           WHEN e.event ILIKE 'Timeout%' THEN 'timeout'
           WHEN e.event ILIKE '% subs:%' THEN 'substitution'
           WHEN e.event ILIKE '% starters:%' THEN 'lineup'
           WHEN e.event LIKE 'Point %' THEN 'point_summary'
           ELSE 'other' END AS row_type,
      CASE WHEN e.event IN {sql_list(CONTACTS + OUTCOMES)} THEN e.event END AS event,
      e.player,
      coalesce(r.player_team,
        CASE WHEN e.event IN {sql_list(RELIABLE_TEAM)} THEN e.team
             WHEN e.event IN {sql_list(TEAM_IS_OPPONENT)} THEN
               CASE WHEN e.team = e.away_team THEN e.home_team ELSE e.away_team END END) AS actor_team,
      e.team AS team_raw, e.event AS event_raw, e.description
    FROM e2 e LEFT JOIN roster r USING (match_id, player)""")


def build_rallies(con):
    con.execute("""
    CREATE TABLE rallies AS
    WITH r AS (
      SELECT match_id, set_no, rally_no,
        arg_min(away_score_before, rn) AS away_score_before,
        arg_min(home_score_before, rn) AS home_score_before,
        arg_min(actor_team, rn) FILTER (WHERE event = 'Serve') AS serve_team_pbp,
        arg_min(player, rn) FILTER (WHERE event = 'Serve') AS server,
        count(*) FILTER (WHERE event = 'Serve') AS n_serves,
        count(*) FILTER (WHERE row_type = 'contact') AS n_contacts,
        max(point_team) AS point_team,
        bool_or(score_changed) AS scored,
        bool_or(score_anomaly) AS score_anomaly,
        arg_max(event, rn) FILTER (WHERE row_type IN ('contact', 'outcome')) AS end_event,
        bool_or(row_type = 'point_summary') AS has_point_summary,
        min(rn) AS first_rn, max(rn) AS last_rn
      FROM events GROUP BY ALL)
    SELECT r.*, m.season, m.away_team, m.home_team,
      -- The winner of a rally serves the next one; use that when no Serve row exists.
      coalesce(serve_team_pbp, lag(point_team) OVER
        (PARTITION BY r.match_id, r.set_no ORDER BY rally_no)) AS serve_team,
      CASE WHEN coalesce(serve_team_pbp, lag(point_team) OVER
             (PARTITION BY r.match_id, r.set_no ORDER BY rally_no)) = m.away_team
           THEN m.home_team ELSE m.away_team END AS receive_team,
      (scored AND point_team IS NOT NULL AND NOT score_anomaly AND n_serves <= 1) AS usable
    FROM r JOIN match_info m USING (match_id)""")
    # Receive team is meaningless if serve team is unknown.
    con.execute("UPDATE rallies SET receive_team = NULL WHERE serve_team IS NULL")
    con.execute("""ALTER TABLE rallies ADD COLUMN sideout BOOLEAN;
                   UPDATE rallies SET sideout = point_team = receive_team WHERE usable""")


def build_sets_and_matches(con):
    con.execute("""
    CREATE TABLE sets AS
    SELECT match_id, set_no, a AS away_final, h AS home_final,
      CASE WHEN a > h THEN 'away' WHEN h > a THEN 'home' END AS winner_side,
      -- Legal final: winner reaches 25 (15 in set 5), wins by 2, and stopped as soon as it could.
      (greatest(a, h) >= CASE WHEN set_no = 5 THEN 15 ELSE 25 END
        AND abs(a - h) >= 2
        AND (greatest(a, h) = CASE WHEN set_no = 5 THEN 15 ELSE 25 END OR abs(a - h) = 2)) AS legal_final
    FROM (SELECT match_id, set_no, arg_max(away_score, rn) AS a, arg_max(home_score, rn) AS h
          FROM events GROUP BY ALL)""")

    con.execute("""
    CREATE TABLE matches AS
    SELECT m.match_id, m.season, m.contest_id, strptime(left(m.date, 10), '%m/%d/%Y')::DATE AS date,  -- "09/28/2024(2)" marks a 2nd match that day
      m.date AS date_raw, m.away_team, m.home_team,
      s.n_sets, s.away_sets, s.home_sets,
      CASE WHEN s.away_sets = 3 THEN m.away_team WHEN s.home_sets = 3 THEN m.home_team END AS winner,
      s.all_sets_legal,
      q2.n_rows, q3.n_rallies, q3.n_usable_rallies, q2.n_score_anomalies, q2.n_point_summary_rows,
      (s.all_sets_legal AND greatest(s.away_sets, s.home_sets) = 3 AND least(s.away_sets, s.home_sets) < 3
        AND q2.n_score_anomalies = 0) AS clean
    FROM match_info m
    JOIN (SELECT match_id, count(*) AS n_sets,
                 count(*) FILTER (WHERE winner_side = 'away') AS away_sets,
                 count(*) FILTER (WHERE winner_side = 'home') AS home_sets,
                 bool_and(legal_final) AS all_sets_legal
          FROM sets GROUP BY 1) s USING (match_id)
    JOIN (SELECT match_id, count(*) AS n_rows,
                 count(*) FILTER (WHERE row_type = 'point_summary') AS n_point_summary_rows,
                 count(*) FILTER (WHERE score_anomaly) AS n_score_anomalies
          FROM events GROUP BY 1) q2 USING (match_id)
    JOIN (SELECT match_id, count(*) AS n_rallies, count(*) FILTER (WHERE usable) AS n_usable_rallies
          FROM rallies GROUP BY 1) q3 USING (match_id)
    ORDER BY m.match_id""")


def main():
    global SEASONS
    import argparse
    ap = argparse.ArgumentParser(description="Build data/vb.duckdb from data/raw/*.csv")
    ap.add_argument("--seasons", type=int, nargs="+", help="only these seasons (for quick test builds)")
    ap.add_argument("--out", type=Path, default=DB, help=f"database path (default {DB})")
    args = ap.parse_args()
    SEASONS = set(args.seasons) if args.seasons else None

    t0 = time.time()
    tmp = args.out.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    for name, fn in [("box scores", load_box_scores), ("events", build_events),
                     ("rallies", build_rallies), ("sets and matches", build_sets_and_matches)]:
        t = time.time()
        print(f"{name}...", flush=True)
        fn(con)
        print(f"  {name}: {time.time() - t:.0f}s", flush=True)
    con.close()
    tmp.replace(args.out)  # only replace the old database once the build has succeeded
    print(f"Wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
