# Rally Odds (volleyball-sim)

NCAA Division I women's volleyball forecasts built from rally-level play-by-play (2020–2025).
Public site: https://m3tang.github.io/volleyball-sim/ · Repo: github.com/m3tang/volleyball-sim

## Layout
- `pipeline/` download CSVs (`download.py`), build `data/vb.duckdb` per season (`build_db.py`), checks (`validate.py`)
- `vbsim/` models: `data.py` (load, TeamIndex), `fit.py` (`TUNED` settings), `models.py`, `matchsim.py` (exact set/match DP), `bracket.py` (rebuild + advancement odds)
- `analysis/` evaluations; results the site reads go in `analysis/results/`
- `web/build_site.py` → `web/data.json`, `web/index.html` (standalone, data inline), `web/artifact.html` (loads data.json)
- `web/template.html` holds all markup, CSS and JS. Edit this, never index.html/artifact.html.
- `rallytools/` model-as-tools: `core.py` (reads web/data.json, mirrors the site's JS math), `sql.py` (guarded read-only DuckDB), `server.py` (MCP server, mcp 2.x `MCPServer`; `--check` calls every tool)
- `tools/checks/` page checks (Node + puppeteer-core + jsdom; `npm install` there once)

## Commands
.venv is Python 3.12 (rebuilt from 3.9 for the mcp SDK). A full rebuild on 3.12 shifts some numbers in the 4th decimal (BLAS/optimizer noise); don't commit data.json for that alone.
```sh
.venv/bin/python -m pytest -q tests
.venv/bin/python -m web.build_site --page-only   # after template.html edits only (seconds)
.venv/bin/python -m web.build_site               # full refit + data.json (~1–2 min); needed when data/models change
.venv/bin/python -m rallytools.server --check      # MCP tools smoke test
.venv/bin/python -m analysis.xgb_baseline        # ~10 min, rewrites analysis/results/xgb_comparison.json
```
Pipe long build/analysis output through `tail`; it is verbose.

## Checking the page (run before committing site changes)
```sh
cd tools/checks
node views.js ../../web/index.html               # JS errors + every view renders
node pp.js <view> 390 /tmp/x.png 1               # phone width overflow check; views: home bracket matchups findings model
node pp.js <view> 1280 /tmp/x.png                # desktop
node el.js <view> "<css selector>" /tmp/x.png [width]   # screenshot one element (prefer over full pages)
node og.js / node readme-shots.js                # regenerate web/og.png / docs/images/*.jpg
```
Find code in template.html with `grep -n`, then read only that range; the file is large.

## Deploy
- Push to `main` → "Deploy site" workflow publishes `web/` to GitHub Pages. Watch with `gh run watch`.
- Claude artifact copy: publish `web/artifact.html` to https://claude.ai/artifact/W78Lz9nHJ9NWKQ89qphGah
  with `files={"data.json": "web/data.json"}`.
- Commit `web/index.html`, `web/artifact.html` and `web/data.json` along with template changes.
- No Claude/AI attribution lines in commits or PRs.

## Site copy rules
- Voice: a volleyball fan who is a data analyst. Plain, concise, short paragraphs.
- Never "we". Never call the model "it"; say "the model" or "the rally model".
- No clipped "x, y" fragment sentences or other AI-sounding phrasing.
- Titles state the takeaway ("The rally model beats XGBoost"), not the topic.
- Odds shown as % plus "1 in N" for long shots. Every chart gets an aria description.
- Don't mention old model versions.

## Model facts worth knowing
- Touch model chain: serve (ace / error / in play) → first-ball (win / lose / to opponent) → transition, solved in closed form; penalized multinomial logistic regression with team off/def effects.
- `TUNED = (lam=10, rho=1.0, form_sd=0.1)`: L2 prior, season carryover, match-day form (logit sd, Gauss–Hermite averaged).
- Evaluation is walk-forward: refit weekly, tuned on 2024, tested on 2025. Brackets use only pre-tournament matches.
- Tested and rejected: recency weighting (hurts tournaments). Tournament odds are calibrated; long-shot champions are underrated (hot streaks).
- XGBoost baseline (67 features) loses: 2025 log loss 0.497 vs 0.467; tournaments 0.460 vs 0.360.

## Data constraints
- Source: pre-scraped ncaavolleyballr CSVs (through 2025). stats.ncaa.org blocks automated access (Akamai 403); do not try to get around it.
- `pipeline/pull_season.R` is an unfinished attempt at 2026 data; left uncommitted on purpose.
- ncaa-api (henrygd) has point-level play-by-play only, not touch-level.
