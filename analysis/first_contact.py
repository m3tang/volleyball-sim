"""First look: how often the receiving team wins the rally (sides out), split by what
happens after the serve and by how many attacks the rally lasts.

Usage:  python analysis/first_contact.py
"""
from pathlib import Path

import duckdb

DB = Path(__file__).resolve().parent.parent / "data" / "vb.duckdb"

con = duckdb.connect(str(DB), read_only=True)

# Only clean, touch-level matches (not the old "Point X:" format).
con.execute("""
CREATE TEMP TABLE r AS
SELECT ra.*, coalesce(a.n_attacks, 0) AS n_attacks
FROM rallies ra
JOIN matches m USING (match_id)
LEFT JOIN (SELECT match_id, set_no, rally_no, count(*) AS n_attacks FROM events
           WHERE event = 'Attack' GROUP BY ALL) a USING (match_id, set_no, rally_no)
WHERE ra.usable AND m.clean AND m.n_point_summary_rows = 0""")

print("What happens after the serve (2020-2025, Division 1 women):")
print(con.sql("""
SELECT CASE end_event
         WHEN 'Ace' THEN '1 ace'
         WHEN 'Service error' THEN '2 service error'
         WHEN 'First ball kill' THEN '3 first-ball kill by receiving team'
         ELSE CASE WHEN n_attacks <= 1 THEN '4 ends on first attack (error or block)'
                   ELSE '5 rally continues past first attack' END
       END AS outcome,
       count(*) AS rallies,
       round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_rallies,
       round(100 * avg(sideout::INT), 1) AS receiving_team_wins_pct
FROM r GROUP BY 1 ORDER BY 1"""))

print("Receiving team win % by rally length (number of attacks):")
print(con.sql("""
SELECT CASE WHEN n_attacks >= 6 THEN '6+' ELSE n_attacks::VARCHAR END AS attacks,
       count(*) AS rallies,
       round(100 * avg(sideout::INT), 1) AS receiving_team_wins_pct
FROM r WHERE end_event NOT IN ('Ace', 'Service error')
GROUP BY 1 ORDER BY min(n_attacks)"""))
