This directory contains the authored oracle for the richer `inventory-gate`
benchmark uplift.

Round 1 (`solution.patch`) repairs the visible CI drift by aligning
`pyproject.toml`, the GitHub workflow, and the workflow preview helper with the
renamed package `ci_app`. It also restores the preview helper's default
inventory artifact/report wiring to the local `make ci` contract.

Round 2 (`solution_followup.patch`) fixes the latent hidden bug: punctuation-
heavy inventory labels must normalize into stable preview job ids, artifact
names, and `reports/*.json` paths without repeated separators.
