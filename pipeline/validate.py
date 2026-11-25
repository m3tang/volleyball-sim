"""Sanity checks on data/vb.duckdb. Prints a report and exits 1 if any check fails.

The key cross-check: match winners and set scores we rebuilt from play-by-play are
compared with the separately published team-match box scores (team_match.Result).

Usage:  python pipeline/validate.py
"""
import sys
from pathlib import Path

import duckdb

DB = Path(__file__).resolve().parent.parent / "data" / "vb.duckdb"

failures = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def main():
    con = duckdb.connect(str(DB), read_only=True)
    q = lambda s: con.execute(s).fetchall()

    print(con.sql("""
      SELECT season, count(*) AS matches, round(avg(clean::INT), 3) AS clean_share,
             sum(n_rallies) AS rallies, round(sum(n_usable_rallies) / sum(n_rallies), 3) AS usable_share,
             count(*) FILTER (WHERE n_point_summary_rows > 0) AS old_format
      FROM matches GROUP BY 1 ORDER BY 1"""))

    dup = q("""SELECT count(*) FROM (SELECT season, contest_id FROM matches WHERE contest_id IS NOT NULL
               GROUP BY ALL HAVING count(*) > 1)""")[0][0]
    check("no duplicate contest IDs", dup == 0, f"{dup} contest IDs appear more than once")
    # Same date and teams twice is legitimate: 2020 (COVID season) had same-day doubleheaders.
    dh = q("""SELECT count(*) FROM (SELECT season, date_raw, away_team, home_team FROM matches
              GROUP BY ALL HAVING count(*) > 1)""")[0][0]
    print(f"[INFO] same-day doubleheaders split by set number: {dh}")

    clean = q("SELECT avg(clean::INT) FROM matches")[0][0]
    check("clean matches >= 85%", clean >= 0.85, f"{clean:.1%} of matches are clean")

    usable = q("SELECT avg(usable::INT) FROM rallies")[0][0]
    check("usable rallies >= 98%", usable >= 0.98, f"{usable:.2%} of rallies are usable")

    serve_known = q("SELECT avg((serve_team IS NOT NULL)::INT) FROM rallies WHERE usable")[0][0]
    check("serving team known >= 99%", serve_known >= 0.99, f"{serve_known:.2%}")

    so = q("SELECT min(r), max(r) FROM (SELECT avg(sideout::INT) r FROM rallies WHERE usable GROUP BY season)")[0]
    check("sideout rate in 50-65% every season", 0.50 <= so[0] and so[1] <= 0.65,
          f"range {so[0]:.3f} to {so[1]:.3f}")

    # Cross-check against box scores. Opponent looks like "Iona", "@ Iona" (away game)
    # or "NJIT @New Rochelle, NY" (neutral site).
    res = q("""
      WITH tm AS (
        SELECT season_year AS season, "Date" AS date_raw, "Team" AS team,
               regexp_replace(regexp_replace("Opponent", '^@ ?', ''), ' @.*$', '') AS opp,
               "Result" AS result
        FROM team_match WHERE "Result" SIMILAR TO '[WL] [0-3]-[0-3]'),
      pm AS (
        SELECT m.*, CASE WHEN winner = away_team THEN away_sets ELSE home_sets END AS w_sets,
                    CASE WHEN winner = away_team THEN home_sets ELSE away_sets END AS l_sets
        FROM matches m WHERE clean)
      SELECT count(*) AS n,
             count(tm.result) AS matched,
             count(*) FILTER (WHERE tm.result = 'W ' || w_sets || '-' || l_sets) AS agree
      FROM pm LEFT JOIN tm
        ON tm.season = pm.season AND tm.date_raw = pm.date_raw AND tm.team = pm.winner
       AND tm.opp = CASE WHEN pm.winner = pm.away_team THEN pm.home_team ELSE pm.away_team END""")[0]
    n, matched, agree = res
    check("clean matches found in box scores >= 90%", matched / n >= 0.90, f"{matched}/{n} ({matched / n:.1%})")
    check("winner and set score agree with box score >= 99%", agree / max(matched, 1) >= 0.99,
          f"{agree}/{matched} ({agree / max(matched, 1):.2%})")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
