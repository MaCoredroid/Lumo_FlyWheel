# Reply 26 — campaign identity guard: the two dynamically loaded route dependencies added (narrow diff)

Written 2026-09-22T09:58:37Z (measured). No GPU launch; no new experiment. Verdict received: F1/F2/Q closed (`e1-campaign-final-redteam.md` a562a452f7a1cf7c…); this reply covers only the one required pre-data change.

## Diff (runner `e1/e1_run_cells.v2.sh` → `910e2dac38a86613`; previous `08e3aa21b08cdd5b`)
1. `campaign_identity.txt` now also freezes `scripts/fr13_device_multidraft_kernel.py` and `src/lumo_flywheel_serving/fr13_tree_conv_fused.py` (both mounted through /workspace and loaded dynamically at boot). The list is: patcher, `fr10_gdn_tree_kernel.py`, `fr10_decode_modes.py`, `fr13_device_multidraft_kernel.py`, `fr13_tree_conv_fused.py`, `gpu_oom_guard.sh`, plus the worktree HEAD; a dependency missing at freeze time refuses (exit 45); any change after the freeze refuses before every cell (exit 45).
2. One-line cosmetic fix in the same guard region: the vanished-process open error of the running-process check is silenced (it only produced a stray "/proc/<pid>/cmdline: No such file" line in the log).

Current hashes of the two added dependencies (as frozen by the next campaign): `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9` and `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e` — equal to the values in your message.

## CPU controls (no GPU)
- `e1/test_e1_campaign_stub.sh` `0957056f7061a6e9` (20/20): `identity_lists_dynamic_deps`; mutating EACH frozen dependency alone in the mirror (`fr13_device_multidraft_kernel.py`, `fr13_tree_conv_fused.py`, the patcher, `fr10_gdn_tree_kernel.py`, `fr10_decode_modes.py`) → runner exit 45 naming that file; restoring → the guard passes again (the run proceeds to the stub launcher, exit 10); snapshot tamper → 45; existing cell dir → 44; attempts/retry/not-qualified/exact env checks unchanged.
- `e1/test_e1_runner_terminal_gate.sh` `b3f3d0055980456a` (9/9) re-run with the mirror carrying both dependencies (F1 seal behaviour unchanged).
Outcomes: {"e1/tests_out_e1_campaign_stub.json": {"all_pass": true, "n": 20}, "e1/tests_out_e1_runner_terminal_gate.json": {"all_pass": true, "n": 9}}.

All other campaign sources are unchanged from reply-25 (driver `25aa6c2195957630`, summary `04a119321a2aa95a`, verify `e4c73148de56d75b`, preflight `6dd79084a8a7dcec`, aggregate `98beec9efac26f56`, manifest `c60a4a4c661e84a6`, freeze `11f5dd00b8d8b707`). Waiting for your verification of this narrow diff and the go; the launch command is in STATUS.


## Erratum (2026-09-22T10:01:19Z, clerical)
The header cited the earlier draft report hash (a562a452…); the settled-version review of record is `e1-campaign-final-redteam.md` sha f10d099c370a78844aaa69f546da1ff24788c5896ac92a2a2aa91aecc22ffb4e (f10d099c…). No content change.
