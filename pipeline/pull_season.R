# Pull one season of NCAA Division I women's volleyball from stats.ncaa.org with the
# ncaavolleyballr package, in the same CSV format as the pre-scraped files in data/raw/.
#
# stats.ncaa.org blocks plain requests, so ncaavolleyballr drives a real Chrome browser
# (chromote). Google Chrome must be installed.
#
# Usage (from the repository root):
#   Rscript pipeline/pull_season.R 2026
#
# Writes:
#   data/raw/wvb_pbp_div1_<season>.csv        play-by-play, one row per touch
#   data/raw/wvb_teammatch_div1_<season>.csv  team box score, one row per team per match
#
# The run is resumable: every match is cached in data/raw/cache_<season>/ as it is fetched,
# so if it stops (sleep, network, rate limiting) just run the same command again.

args <- commandArgs(trailingOnly = TRUE)
season <- if (length(args) >= 1) as.integer(args[1]) else 2026L
delay <- if (length(args) >= 2) as.numeric(args[2]) else 2   # seconds between requests

for (pkg in c("ncaavolleyballr", "dplyr", "purrr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({
  library(ncaavolleyballr)
  library(dplyr)
  library(purrr)
})

# ncaavolleyballr 0.5.1 hard-codes 2025 as the latest season it accepts. Raise that limit
# for this run only so newer seasons can be requested.
if (ncaavolleyballr:::most_recent_season() < season) {
  utils::assignInNamespace("most_recent_season", function() season, ns = "ncaavolleyballr")
  # Its team-ID check also only knows the bundled (pre-2026) IDs, so skip that check too.
  utils::assignInNamespace("check_team_id", function(...) invisible(TRUE), ns = "ncaavolleyballr")
}

raw_dir <- file.path("data", "raw")
cache <- file.path(raw_dir, paste0("cache_", season))
dir.create(file.path(cache, "pbp"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(cache, "teammatch"), recursive = TRUE, showWarnings = FALSE)
say <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

# Team list for a season, fetched through the browser. (The package's get_teams() uses a
# plain request, which stats.ncaa.org blocks.)
teams_live <- function(season) {
  url <- paste0("https://stats.ncaa.org/team/inst_team_list?academic_year=", season + 1,
                "&conf_id=-1&division=1&sport_code=WVB")
  page <- ncaavolleyballr:::request_live_url(url)
  links <- rvest::html_elements(page, "table a")
  data.frame(team_name = rvest::html_text(links),
             team_url = rvest::html_attr(links, "href")) |>
    dplyr::mutate(team_id = stringr::str_extract(.data$team_url, "\\d+"), div = 1, yr = season) |>
    dplyr::filter(!is.na(.data$team_id)) |>
    dplyr::distinct(.data$team_id, .keep_all = TRUE)
}

# Cache helper: fetch once, then read from disk on every later run.
cached <- function(path, fetch) {
  if (file.exists(path)) return(readRDS(path))
  Sys.sleep(delay)
  value <- tryCatch(fetch(), error = function(e) {
    say("  failed:", conditionMessage(e))
    NULL
  })
  if (!is.null(value)) saveRDS(value, path)
  value
}

# 1. Division I teams for the season. Older package versions don't list newer seasons,
#    so fall back to fetching the team list from the NCAA site.
teams <- wvb_teams |> filter(yr == season, div == 1)
if (nrow(teams) == 0) {
  say("No", season, "teams bundled with this package version; fetching the team list.")
  teams <- cached(file.path(cache, "teams.rds"), function() teams_live(season))
}
say(nrow(teams), "Division I teams for", season)

# 2. Each team's schedule, and its per-match box scores.
say("Fetching schedules and team box scores...")
contests <- map2(teams$team_id, seq_len(nrow(teams)), function(id, i) {
  if (i %% 25 == 0) say("  team", i, "of", nrow(teams))
  cached(file.path(cache, "teammatch", paste0("contests_", id, ".rds")), function() find_team_contests(id))
}) |>
  list_rbind() |>
  filter(!is.na(contest)) |>
  distinct(contest, .keep_all = TRUE)
say(nrow(contests), "unique matches found so far this season")

teammatch <- map(teams$team_id, function(id) {
  cached(file.path(cache, "teammatch", paste0("stats_", id, ".rds")), function() team_match_stats(id, sport = "WVB"))
}) |> list_rbind()

# 3. Play-by-play for every match (the slow part: one page per match).
say("Fetching play-by-play...")
pbp <- map2(contests$contest, seq_len(nrow(contests)), function(cid, i) {
  if (i %% 100 == 0) say("  match", i, "of", nrow(contests))
  rows <- cached(file.path(cache, "pbp", paste0(cid, ".rds")), function() match_pbp(cid))
  if (is.null(rows) || nrow(rows) == 0) return(NULL)
  rows |> mutate(date = contests$date[i], contestid = as.character(cid))
}) |> list_rbind()

# 4. Write in the same column layout as the pre-scraped 2025 files.
cols <- c("date", "contestid", "set", "away_team", "home_team", "score", "team", "event", "player", "description")
pbp <- pbp |> select(any_of(cols))
missing <- setdiff(cols, names(pbp))
if (length(missing) > 0) stop("Play-by-play is missing columns: ", paste(missing, collapse = ", "))

pbp_file <- file.path(raw_dir, paste0("wvb_pbp_div1_", season, ".csv"))
tm_file <- file.path(raw_dir, paste0("wvb_teammatch_div1_", season, ".csv"))
write.csv(pbp, pbp_file, row.names = FALSE)
write.csv(teammatch, tm_file, row.names = FALSE)
say("Wrote", pbp_file, "(", nrow(pbp), "rows,", n_distinct(pbp$contestid), "matches )")
say("Wrote", tm_file, "(", nrow(teammatch), "rows )")
say("Done. Rebuild with: .venv/bin/python pipeline/build_db.py && .venv/bin/python -m web.build_site")
