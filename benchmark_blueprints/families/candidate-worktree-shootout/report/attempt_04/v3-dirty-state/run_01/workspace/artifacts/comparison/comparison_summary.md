# Comparison Summary

Chosen strategy: Candidate B.

Candidate B is stronger because it puts separator-heavy normalization in
`service.compile_filters(...)`, which is the shared contract used by the
CLI, the scheduled importer, and the saved-view repair job. The final
workspace patch therefore fixes the visible CLI regression and the
direct service callers with one implementation path and one regression
test.

Rejected strategy: Candidate A.

Candidate A is weaker because it only patches `cli.py`. That makes the
current CLI tests pass, but the direct service probe still returns
unnormalized keys for separator-heavy labels, which matches the caller
matrix and the abandoned partial-work evidence in `repo_evidence/`.
