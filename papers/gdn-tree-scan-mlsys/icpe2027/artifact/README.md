# Artifact: Hosting Tree Speculation on a Recurrent-Hybrid Model (ICPE 2027 submission)

Anonymized for double-anonymous review. Contents:

- `paper/` — LaTeX source of the submission (ACM `acmart`, `sigconf`).
- `records/` — the raw records behind every table: recorded-request replays (`raw/replay/*/replay.jsonl`, server metrics, GPU timer sidecars), the captured-row sampler audit (`raw/replay/tree-pf0-*/pen_trace.jsonl`, `summaries/sampling_20261003/`), the full-model continuation run (`raw/q1v3v3/`, verdict and per-arm records; tensor objects omitted), the coding-agent study (`raw/workload/`: the corrected tree arm and its second attempt, the plain-decoding control and its second attempt, and the second attempts of native MTP-5 and SGLang EAGLE; per-task evaluator reports, agent traces, server metrics), the frozen continuation protocol sources (`code/q1v3v3/`), the serving-patch diff that corrects the tree sampler's penalty histories (`code/patcher-penfix.diff`), and the audit tooling (`code/pendiag/`, `code/penfix/`).
- `records/audit_penfix.py` — independent reduction that recomputes the replay pools, the history and Monte-Carlo verdicts with the top-k tie check, the continuation statistics, the coding-agent outcomes and two-attempt tallies, and the output-length comparability from `records/`; it verifies every file against `records/SYNC-MANIFEST.json` first. Run with Python 3.12: `python3 records/audit_penfix.py`.

Earlier records referenced by the audit (the native-arm replays, the first MTP-5 and SGLang agent attempts, and the pre-correction runs) are supplied in a companion archive on request through the submission system; the audit falls back to the values recorded in `records/AUDIT.json` when they are absent and says so.

Host names, user names, and the project and system names have been replaced; numeric results are unchanged.
