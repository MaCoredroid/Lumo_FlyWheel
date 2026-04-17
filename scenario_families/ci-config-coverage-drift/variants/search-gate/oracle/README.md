This directory contains the authored oracle for the richer `search-gate`
benchmark uplift.

Round 1 (`solution.patch`) repairs the visible CI drift by aligning
`pyproject.toml`, the GitHub workflow, and the workflow preview helper with the
renamed package `ci_app`. It also restores the preview helper's default search
suite selectors to stable boolean `pytest -k` expressions.

Round 2 (`solution_followup.patch`) fixes the latent hidden bug: punctuation-
heavy search suite labels must normalize into stable preview job ids and
selector tokens without leaking dots, brackets, or repeated punctuation.
