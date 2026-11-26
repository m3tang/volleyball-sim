"""Real questions from the public chat, promoted to golden cases.

How a question gets here: `python -m evals.logs pull` copies the API's question log, `python -m evals.logs triage`
writes evals/logs/inbox.md (withheld answers, declines, tool errors and new kinds of question first), and you add
the ones worth keeping below. Then `python -m evals.golden` turns each into a case.

Each entry names the tool call that answers it. Expected numbers are read from that call's result at
generation time (a dotted path into the result, list positions as numbers), never typed in, so they stay
right if the model is refit. Facts are the words a right answer has to contain.

    question   as asked (fix typos only if the meaning is unchanged)
    asked      date it was logged
    history    earlier (question, answer) pairs, for follow-ups; answers as the chat showed them
    call       (tool, args) that answers it
    numbers    [(scale, label, path)]  scale: prob | count | decimal
    facts      [(label, [accepted spellings])]
    refuse     True when the right answer is to decline
"""

LOGGED = [
    dict(question="Who did the model pick to win the 2024 NCAA tournament?", asked="2026-11-26",
         call=("bracket_forecast", {"season": 2024}),
         numbers=[("prob", "P(Pittsburgh wins the 2024 title)", "title_odds.0.p_title")],
         facts=[("model's pick", ["Pittsburgh", "Pitt"])]),
    dict(question="And what were the actual champion's odds?", asked="2026-11-26",
         history=[("Who did the model pick to win the 2024 NCAA tournament?",
                   "The model picked Pittsburgh to win the 2024 NCAA tournament, with title odds of 52%. Nebraska was "
                   "next at 30%. Penn State won the title, which the model had at about 1 in 80 before the tournament.")],
         call=("bracket_forecast", {"season": 2024}),
         numbers=[("prob", "P(Penn State wins the 2024 title)", "actual_champion_pre_tournament_title_odds")],
         facts=[("champion", ["Penn State", "Penn St."])]),
    dict(question="What was Nebraska's record in 2025?", asked="2026-11-26",
         call=("team_profile", {"team": "Nebraska", "season": 2025}),
         numbers=[("count", "Nebraska wins", "wins"), ("count", "Nebraska losses", "losses")]),
    # Answered the earlier Texas A&M question again instead of this one (logged 2026-11-26 04:54).
    dict(question="Who was the best team of 2022?", asked="2026-11-26",
         history=[("Why did Texas A&M win the 2025 title?",
                   "Texas A&M won the 2025 title mostly by out-playing the model's expectations in the rally details that decide close sets, not by being the favorite. Before the tournament the rally model gave them about 0.4% (about 1 in 250) to win it all, and it picked Nebraska to win (64%).\n\nThe biggest edge came in the final against Kentucky, which A&M won 3-0. The model expected about 31% of A&M's scrambles after a dig to become points, but they scored on about 47% of them, which was the single largest swing in the match. Kentucky's scrambles fell well short of expectation too. A&M also got more free points from Kentucky's missed serves.\n\nIn the regional final against Nebraska (won 3-2 in Nebraska), A&M's first attacks after a good pass scored about 48% of the time against an expected 35%. That was the biggest factor in a match the model had at roughly 1 in 20 for A&M. A&M also reached the final four and the final as underdogs against Pittsburgh and Kentucky, both with the model's pick going the other way.")],
         call=("season_rankings", {"season": 2022, "top": 5}),
         numbers=[("count", "Texas 2022 wins", "rankings.0.wins"), ("count", "Texas 2022 losses", "rankings.0.losses")],
         facts=[("best team", ["Texas"])]),
]
