# Plan

Next steps for Rally Odds, roughly in order of value for the effort. Updating the data for the 2026
season comes first once the NCAA replies; this list covers everything else.

## Done

- **Odds from any score** (was #4): the live scoreboard on the Match simulator and the
  `live_win_probability` tool for the chat.
- **Real questions as test cases** (was #7): `python -m evals.logs pull` and `triage`, with
  promoted cases in `evals/logged.py`. Repeat every few weeks.

## Model

1. **Fix the known weak spot: long-shot champions.** Each match gets an independent good or bad
   night, so a team that gets hot for a whole tournament is underrated. Three straight champions
   started below 2% (about 1 in 9,000 for all three). Try letting form carry over within a
   tournament, and backtest it on the 2021-25 brackets with the existing walk-forward setup.
   It either closes the gap or becomes an honest finding.
2. **Show the uncertainty.** A team fit on 30 matches and one fit on 12 get equally confident
   numbers. Show a range on match odds, or flag ratings that lean mostly on last season.
3. **Add a second baseline next to XGBoost.** Elo, or the ranking fans already know, gives a
   reference point most readers recognize.

## Site

4. **Shareable matchups.** Put teams, seasons and venue (and the live score) in the URL so a link
   reopens the same matchup.
5. **Bracket challenge.** Visitors fill in a bracket and see how it would have scored against the
   model's picks, using the bracket points the site already computes.

## Agent

6. **The AI judge.** Rate about 30 answers by hand, then calibrate a judge against those ratings.
   It catches what the number check can't, like an answer that puts the right number on the wrong
   team.
7. **Weekly evals.** Run a 20-question Haiku set on a schedule (about 2 cents a run) so a model
   update or prompt change can't break answers unnoticed.
8. **Keep the daily budget across restarts.** It lives in memory and resets each time the API
   starts after going idle. Store it (one Firestore record) before the site gets more traffic.

## Getting it seen

9. **Upset recaps (roadmap step 4).** Short AI-written paragraphs, generated at build time, on why
   each tournament upset happened.
10. **A launch write-up.** One post on the rally model, the XGBoost comparison, and the chat that
    checks every number. Share it with volleyball analytics people and on r/volleyball around the
    December tournament.

Top three for now: 1 (long-shot champions), 6 (the judge) and 4 (shareable matchups).
