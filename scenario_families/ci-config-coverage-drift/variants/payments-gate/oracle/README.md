This directory contains the authored oracle for the richer `payments-gate`
benchmark uplift.

Round 1 (`solution.patch`) repairs the visible CI drift by aligning
`pyproject.toml`, the GitHub workflow, and the workflow preview helper with the
renamed package `ci_app`.

Round 2 (`solution_followup.patch`) fixes the latent hidden bug: separator-heavy
payment lane labels in the workflow preview helper must collapse repeated
punctuation into a stable single-hyphen dispatch slug.
