# Deviations from CARD.md, recorded as they were found

1. **Detector defects fixed before the kept run** — see the CARD.md addendum.
   The aborted launch's 4 canaries are in `runs/aborted_detector_bug_soak/` and are
   excluded from the result.

2. **P3 assistant turns are degraded (found mid-run, not fixed).**
   `soak.py` grows the P3 conversations by appending each round's assistant output.
   It reads that output from `runs/soak/full_P3_*.json`, but those files are only
   written when `fire(..., keep_full=True)`, and the P3 requests go through
   `run_concurrent()`, which does not set it. So the appended assistant turn falls back
   to `text_head * 8` (a ~1.3 kB repeat of the first 160 characters) instead of the real
   completion. Discovered while the soak was already running; a Python process cannot
   pick up an edit to its own loaded module, and a restart would have cost another
   ~6 minutes of startup plus all progress, so it was left alone.
   **Effect:** P3 still grows context toward `max_model_len` as intended — each round
   also appends a fresh ~3000-token filler document, which is what actually drives the
   context length — but the assistant turns in those conversations are repetitive text
   rather than the model's own prose. P0, P1, P2 and P4 are unaffected. The reported
   `max_context_tokens` is measured from the server's own `usage.prompt_tokens`, so it
   remains accurate.

3. **All traffic is "thinking" traffic.**
   With `--reasoning-parser qwen3` (kept for fidelity to the issue's launch command),
   the model's output lands in `message.reasoning`, and at the issue's `max_tokens: 100`
   the canary never closes its reasoning block — every canary ends
   `finish_reason: length` with `content: null`. Collapse detection is on the generated
   **token ids**, which cover reasoning and content alike, so detection is unaffected.
   But it does mean this soak exercises the reasoning path and rarely the content path.
   Recorded as a characterisation limit.
