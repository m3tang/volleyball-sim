# volleyball-sim

A rally-level model of NCAA Division 1 women's volleyball, built to simulate matches
and the NCAA tournament bracket.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline/download.py      # ~4.5 GB of CSVs into data/raw/ (skips files you have)
.venv/bin/python pipeline/build_db.py      # builds data/vb.duckdb (~1.5 min, ~850 MB)
.venv/bin/python pipeline/validate.py      # sanity checks; exits 1 on failure
.venv/bin/python analysis/first_contact.py # first look at sideout rates
```

`build_db.py --seasons 2024 --out /tmp/test.duckdb` builds a single season quickly.

```sh
.venv/bin/python -m pytest -q tests          # match math and model fitting
.venv/bin/python -m analysis.evaluate_models # tune on 2024, test on 2025 (~1 min)
.venv/bin/python -m analysis.overconfidence  # momentum test, form estimate, form tuning (~3 min)
.venv/bin/python -m analysis.bracket_2025    # 2025 tournament odds and scoring
```

## Public page (`web/`)

Five views (home, brackets 2021-2025, matchup simulator, findings, how it works), switched by
`#bracket`, `#matchups`, `#findings` and `#model` links. The build writes:

- `web/index.html`: a complete standalone page with its data built in. Double-click it
  (or `open web/index.html`), or put it on any static host such as GitHub Pages.
- `web/artifact.html` + `web/data.json`: the same page for the Claude artifact preview.

```sh
.venv/bin/python -m web.build_site              # refit everything and rebuild (~2 min)
.venv/bin/python -m web.build_site --page-only  # after editing web/template.html (instant)
```

The page's JavaScript reimplements the touch model's rally chain and the exact set/match
math (between the `MATH-START` and `MATH-END` markers); it matches the Python model to within 4e-7.

## Models (`vbsim/`)

Both models estimate, for any two teams, how often the receiving team wins the rally.
`vbsim/matchsim.py` turns that into exact set and match win probabilities using the
scoring rules (rally scoring, winner serves, 25/15 win by 2, best of five).

- **Sideout model:** one logistic regression; each team has a receive strength and a serve strength.
- **Touch model:** the rally as a chain of possessions. The serve ends as an ace, a service
  error or in play; each possession after that ends as a win, a loss, or the ball going to
  the other team. First-ball possessions (right after the serve) and transition possessions
  get separate team strengths. The rally win probability is solved exactly from the chain.

Team strengths are opponent-adjusted, shrunk toward a prior (`lam`), and each season
starts from last season's strengths scaled by `rho`.

### Results (test season 2025, weekly refits, settings tuned on 2024)

| | log loss | Brier | accuracy |
|---|---|---|---|
| coin flip | 0.693 | 0.250 | 50.0% |
| sideout model | 0.472 | 0.155 | 76.6% |
| touch model | 0.472 | 0.155 | 77.3% |

The two models are statistically tied (touch minus sideout log loss −0.0005, 95% interval
−0.0037 to +0.0028). The touch model is the main engine because it predicts as well and
can explain *why* a team is favored.

### Overconfidence: match-day form, not momentum (`analysis/overconfidence.py`)

Without adjustment both models were overconfident at the extremes (teams given 3.9% won 7.2%).

- **Momentum:** after the server wins k points in a row, the receiving team's rally win rate
  falls (57.7% at k=1 to 52.3% at k≥6). Simulated sets with *no* momentum but a random
  match-day form swing reproduce this within 0.3 points for k=1-5, so the drop comes from
  good teams on good days producing runs, not from runs causing points. At k≥6 the real
  rate is 0.9 points lower than simulated, which is either a small late-run momentum
  effect or form that varies more than the simulation assumes.
- **Match-day form:** when one team beats its predicted sideout rate, its opponent falls short
  of theirs. The residual covariance puts the form swing at about 0.13 on the logit scale.
- **Fix:** `match_win_prob(..., form_sd)` averages match odds over a random form swing.
  Tuned on 2024 (`form_sd` = 0.1), on 2025 it lowers touch-model log loss from 0.4719 to
  0.4669 (95% interval for the change −0.0076 to −0.0022), and teams given 4.5% now win
  5.7% of the time. Tuned settings live in `vbsim.models.TUNED`.

## Tournament bracket (`vbsim/bracket.py`, `analysis/bracket_2025.py`)

The 2025 NCAA bracket (63 games) is rebuilt from box scores: tournament games are labeled
in the opponent field, and walking back from the final through each team's previous match
gives the bracket tree. Hosts come from the venue in the same field. The `#N` labels in that
field are not consistent per team, so they are not used as seeds.

Advancement odds are exact over every path through the bracket. 2025, using only data
from before the first round:

| | log loss (63 games) | games called right | bracket score (max 1920) |
|---|---|---|---|
| coin flip | 0.693 | 50% | |
| better regular-season record | | 68.3% | 800 |
| touch model (with match-day form) | 0.387 | 82.5% | 980 |

The model called 43 of 48 games in the first two rounds but only 3 of 7 from the regional
finals on. Champion Texas A&M had 0.4% title odds (0.2% before the form fix) and a 5%
chance against Nebraska in the regional final; Nebraska (undefeated) had 64.5%.

## Data

Pre-scraped CSVs from the [ncaavolleyballr](https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html)
project by Jeffrey R. Stevens (MIT license), originally from stats.ncaa.org. Division 1
women's, seasons 2020-2025. The 2020 season is the COVID season played into spring 2021.

### Tables in `data/vb.duckdb`

| table | one row per | notes |
|---|---|---|
| `events` | play-by-play row | filled-in score, `rally_no`, `row_type`, `actor_team` |
| `rallies` | rally | score before, `serve_team`, `point_team`, `sideout`, `usable` |
| `sets` | set | final score, `legal_final` |
| `matches` | match | teams, sets won, `clean` flag, data-quality counts |
| `match_info` | match | contest ID, raw date, teams |
| `team_match`, `team_season`, `player_match`, `player_season` | — | box scores as published |

`match_id` is `season * 100000 + n`. Use `rallies.usable` and `matches.clean` to filter.

### Things about the source data worth knowing

- `score` is **away-home**, and it is the score *after* that row's event. A rally ends on the
  row where the score changes.
- During media timeouts, `score` holds the text "Media timeout" for the middle rows of a rally.
- On error rows (`Attack error`, `Service error`) the `team` column is the team that **won the
  point**, not the team that erred. `Set error` and `Ball handling error` are mixed. The
  pipeline works out `actor_team` from each player's team in that match instead.
- Some matches (mostly 2020-2022) use an older format with "Point X: ..." summary rows,
  substitutions and timeouts in the `event` column. Their rallies are still usable; their
  touch sequences are less complete. See `matches.n_point_summary_rows`.
- The 2025 file includes `contestid` and lists about 1,700 matches twice; the pipeline keeps one copy.
- Dates like `09/28/2024(2)` mark a team's second match that day.
- About 8% of matches are not `clean` (a set stops early, or the score jumps), usually
  because the recording was cut short.

### Validation (current build)

- Winner and set score rebuilt from play-by-play agree with the published box score for
  **99.94%** of clean matches (23,892 of 23,907).
- 99.5% of 4.47 million rallies are usable; the serving team is known for 99.9%.
- The receiving team wins 56.5-57.1% of rallies in every season.
