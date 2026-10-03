# Candidate target/MTP registry repair: independent bounded closure

Reviewed 2026-09-29 UTC. **The root repair is accepted at the source/CPU level. No concrete remaining blocker was found in this delta.** The prospective v3 propagation also closes the registry and malformed-counter findings in the prior bridge review. This is not a GPU result or launch approval; the parent retains final source/gate binding, readiness and execution ownership. No Docker, SSH, GPU, model, cache action, source edit, or gate edit was performed by this reviewer.

## Exact reviewed bytes

Paths are relative to `v2/experiments/review-response-20260927/tools/`.

| Artifact | SHA-256 |
|---|---|
| `q1_candidate_layer_registry_v1.py` | `3c6c4c06cffbc41a7ff7d6755b623cb01450f3fa66040b50e6c044ecdf269783` |
| `q1_candidate_hooks_v1_2.py` | `af4595e4b03d8f5c44a5ae0bca5125677e00b827666d8062fec7b43fdb2ac516` |
| `q1_patch_candidate_v1_2.py` | `e0418f8f544ae88cc1be445db98d20bcc97272fbe4363f945c22127e25771c8c` |
| `q1_candidate_job_v1_2.py` | `2560965556574b53cbcc6b914f00119723d3407af8763d77d34af0e362d731b7` |
| `run_q1_candidate_stage1_v2_7.sh` | `08190b5176bb1ba81a8b55c69ab54412f4ea5afc9bd2708a105de8f1af639591` |
| `q1_make_diag_launcher_v3_2_1.py` | `330e9134ef7c983dda9d654ceeee063fb1899b3a43b0220bfbb7a767e3d7796e` |
| `generated/fr14_leg3_launch_nomiddleware.q1diag.v3_2_1.sh` | `c5b2d634f7fe036e9d532f38a35f9e724c1d2f7085e5b6c32d74423f1bcff909` |
| Prospective propagated `q1_candidate_hooks_v3.py` | `72a399fcddb80255d7eff9b650cc3deb88a1a48d4ee1f814721a3b297fb5bdf2` |

## Root repair closure

The independent helper declares the precise Qwen target names: 48 `language_model.model.layers.*.linear_attn` layers and 16 target `.self_attn.attn` layers. It separately requires `mtp.layers.0.self_attn.attn`, refuses missing/extra target-like or MTP-like names and unbound layers, and returns target names only. Its name sets independently match both the production target-layer declarations and the archived accepted native root O0's actual 48/16 key sets.

All three root sites now use this helper: boot attestation, state snapshots, and hydration. Boot records the separate MTP name, but target backend checks, hydration destinations, and target-state snapshots receive only the 48/16 target lists. No tensor arithmetic, copy, layout, forced-token behavior, or numerical rule changes in the v1-to-v1.2 diff.

The full selection chain agrees: wrapper v2.7 hashes the helper and selects job v1.2, hooks v1.2, patcher v1.2, and diagnostic launcher v3.2.1. The job includes the mandatory helper source pin and current hook hash; the hook verifies that helper pin. The patcher changes only its hook import/marker/hash destination relative to v1. The generated diagnostic successor changes exactly five patcher filename occurrences from v1 to v1.2. Inverting those five substitutions restores the exact pinned v3.2 launcher bytes. The v2.6 unknown-ownership repair is retained in v2.7.

Independent CPU checks:

- A realistic 65-member registry selects target48/16 and excludes MTP. Missing MTP, missing target, extra MTP, unbound target, and incorrect target prefix all refuse.
- The actual root `_attest_boot` method, extracted through AST with metadata-only providers, accepts target48/16 plus separate MTP even when MTP has a different backend label; replacing a target backend with an incorrect label makes boot fatal. Thus MTP is separated without weakening target backend checks.
- Ran both actual CPU job builders on the same local archived native root record, real prefix fixture and generated-source manifest. Removing timestamp/canonical fields and the two intentionally changed source pins makes the outputs identical. Root-only, process A, R2, request parameters and all model/numerical settings remain unchanged.
- The actual new patcher applied in memory to both unchanged captured generated sources; both patched ASTs parse. Stripping inserted hook/import marker lines restores both originals byte-for-byte.
- Generator controls refuse changed pinned base, four/six occurrences (isolated with an in-memory fixture hash), and a conflicting existing output. No launcher was run by these checks.

## Prospective v3 closure

The same helper replaces all four v3 selection sites, including the new target-KV before hook, and its source pin is mandatory. R1 in `target-kv-bridge-v3-source-review.md` is closed for the v3 hash above.

Observed complete-event counters now require a nonnegative Python `int` before use. Fifteen independent exact-AST controls cover before/after counters: fractional values, numeric strings, booleans and negative values refuse; missing/stale/jumped after counts refuse; exact integer `before+1` reaches the O1 boundary. The before-counter's inherited absent→0 first-event convention remains explicitly bounded here: that zero is an initialization convention, not an assertion that a present counter was read. R2's coercion issue is closed. Root v1.2 remains the separately scoped instrumentation repair; this note does not silently add the prospective v3 witness to root jobs.

## Evidence and limits

Independent evidence beside this note: `candidate-target-registry-root-repair-cpu-audit.json` and `candidate-target-registry-root-repair-integration-cpu-audit.json` (the latter SHA `53e5ea5503bd18f9cef639d428af4d29893b205be8a5c567be16f8bc7147e431`). Prior failure notes remain historical. The checks are source, actual CPU job-builder, and AST/metadata controls, not new model experiments or tensor qualification. Target Bash syntax and full tensor fixture tests remain parent-owned; this reviewer did not claim to rerun them. Current NumPy absence in the local interpreter was already documented in the prior bridge review.

Final launch preparation must bind these exact successors. Old gates/jobs do not gain authority from this review, and the v3 witness still needs its own complete job/launcher integration before any use.
