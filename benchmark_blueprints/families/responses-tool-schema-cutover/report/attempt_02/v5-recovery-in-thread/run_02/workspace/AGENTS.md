# Responses Replay Repair

Variant: `v5-recovery-in-thread`

Incident context shows an earlier chronology-blind fix regressed replay snapshots after rollback.

Required solver behavior:
- Inspect at least one replay fixture before patching.
- Repair both `gateway/adapter.py` and `gateway/reducer.py` if needed.
- Keep repeated same-name tool calls distinct by `call_id` all the way through render output.
- Keep the public CLI summary format stable for successful runs.
- Update `codex/config.toml` and `docs/migrations/responses-cutover.md` to the Responses-native contract.

Anti-shortcuts:
- Do not dedupe repeated tool calls by tool name.
- Do not synthesize `call_id` from fixture position or visible ordinals.
- Do not delete fixtures or relax tests to get green.

