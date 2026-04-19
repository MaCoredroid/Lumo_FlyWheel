---
name: ops-digest
description: Build a weekly ops digest from the sample incident fixture.
---

# Ops Digest

Use `tools/digest_runner.py` as the canonical entrypoint. The legacy script path and boolean summary flag are stale.

From the repo root, run:

```sh
python tools/digest_runner.py --input fixtures/incidents/sample_events.json --summary-length short
```

Expected markdown shape:
- Title: `# Ops Digest`
- A `## Summary` section with bullet points
- A `## Events` section with one bullet per incident

Refactor note: the repo moved to `tools/digest_runner.py`, and the summary control is now `--summary-length {short,long}`.
