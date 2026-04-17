## Payments Gate CI Contract

`make ci` is the repo-local contract used by the workflow and by local smoke
checks.

The workflow preview helper mirrors the package name wired through
`pyproject.toml` so release reviewers can spot drift before CI fails in GitHub.
Preview dispatch ids should stay stable when payment lane labels arrive with
extra punctuation or repeated separators.
