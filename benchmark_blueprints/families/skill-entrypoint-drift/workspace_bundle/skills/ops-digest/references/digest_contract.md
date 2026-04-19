# Digest Contract

Canonical command:

`python tools/digest_runner.py --input fixtures/incidents/sample_events.json --summary-length short`

Contract:
- Entrypoint: `tools/digest_runner.py`
- Required input flag: `--input`
- Summary flag: `--summary-length`
- Valid summary values: `short`, `long`
- Markdown heading order: `# Ops Digest`, `## Summary`, `## Events`

Reconciled drift:
- Removed the stale `scripts/build_digest.py` entrypoint from the skill workflow.
- Replaced the obsolete `--brief` flag with the live summary-length flag.
- Documented both repo-root and skill-relative command forms with paths that actually resolve.
