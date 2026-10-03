# After-Z second-event binding: bounded source review

**Current disposition: F1 closed on repair2, SHA `a6850423e1318f88706f1b5a52fc343fa89ebc1d21fb2b5727c2dc457eda9b53`; bounded source/CPU PASS.** The initial and first-repair findings below remain preserved as review history. See the final closure section for tested scope.

2026-09-29. Initial source SHA-256 `8cfdda65b187e7884741ebe40897235adeffbcd52642845dfc3269c6d66a3c72`; supplied tests `bb9d6bb8b368fae523cbf0610633651a94ea8d6c1ed5cbb0d92263194d32d340`. **One concrete prior-event binding defect was reproduced.** The first repair closes constructor/callback cases but leaves a terminal recheck gap described below. No launch or numerical qualification is implied by this review.

## F1: prior first-event receipt is not tied throughout the second event

Initial source lines 25–29 call `HE.audit` on a prior receipt and the owner copied from that same receipt. This proves internal consistency, but does not bind its run/observation/request/process/runner/drafter to the current case or compare it with the preceding actual live census slot. Independent CPU controls passed the real `Binding`, `HB.Observer` tensor copy/gather audit, and `HE.audit`, then reached a successful `audit_seal` with each of: foreign prior observation, foreign prior run, a foreign prior request with internally matching request digests, and a different live preceding census. A prior receipt changed after construction also passed the next owner callback. These are synthetic ownership fixtures, not candidate model outputs. The unchanged valid second-event path passes as a control.

Minimum correction: bind the prior receipt to the current producer/case and exact preceding live census, retain its immutable snapshot, and check that snapshot throughout the new event, including the terminal seal. Keep the accepted first-event observer unchanged.

## First repair: partial closure, terminal gap remains

Repair SHA-256 `0b6cecd04afa48b95db68a60dd78fd2aff2753985d7aaedd5af836529329d3b8` adds current-context comparisons, last-census identity/content checks and a deep copy (27–36), and checks the retained receipt at owner callbacks (44). All five original bad constructor/callback cases now refuse, and the complete positive still passes.

However, `audit_seal` at 65–72 does not perform the retained-first-event check. Two independent follow-up controls complete the real hidden bridge, then either change `case.hidden_event['obs_id']` or replace the preceding live census slot with a foreign-request event before performing the second seal. Both still return success. This is the terminal boundary of F1, not a request for broader hardening. Reuse the same first-binding validation at owner boundaries and at `audit_seal`, with the appropriate pre-/post-seal census length, rather than maintaining divergent checks.

## Scope and retained evidence

Snapshots, a standalone reviewer reproducer, and raw CPU outcomes are preserved under paper-relative `p0/monitor/review-response-20260927/candidate-after-z-binding-independent/{initial,repair1}/`. The initial script uses the exact frozen binding and unchanged HB/HE helpers with real CPU tensors and minimal synthetic census records satisfying the existing HE schema. It does not import a vLLM engine or execute generated model code. Local command uses `/usr/bin/python3`, Torch 2.8.0, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1` and one Torch thread. The author's separate 12-control Python 3.12 log was read; its GPU/engine-free claims are not treated as independent runtime evidence.

The source preserves `nodes=[0]`, exact second TAW count, consecutive event index, monotonic forward step, same live pending/proposal identities, root-only acceptance, restored query identity, post-KV gather boundary, and a complete second census. No other critical source defect was found in the bounded review. Actual O2 input token verification, complete phase collector, source/caller installation, owner-level fail-stop/partial retention, and GPU admission remain explicitly outside this helper's current claim. No remote write, GPU/model execution, implementation edit, policy edit or gate action was taken.

## Repair2 final closure

Reviewed final source: `experiments/review-response-20260927/tools/q1_candidate_after_z_binding_v1.py`, SHA-256 `a6850423e1318f88706f1b5a52fc343fa89ebc1d21fb2b5727c2dc457eda9b53`. The new `_check_first(sealed)` (65–75) validates the unchanged first receipt, current case/producer identity, exact pre-/post-seal census length, actual preceding event with HE, and canonical equality to the retained first census. Both `owner` (44) and `audit_seal` (78) invoke it. This closes the terminal gap without changing HB/HE, numerical criteria, first-event live binding, production settings or model calls.

All **nine independent CPU controls pass** against a frozen copy: the complete valid second-event copy/forward/gather/selector/graph-owner/seal path succeeds and preserves the first receipt; the five original bad constructor/callback cases refuse; both residual terminal mutations refuse; and changing the current case observation immediately before seal also refuses. The positive records event index3, forward step11, complete-event count4, nodes `[0]`, and `qualification=False`. These are explicit synthetic census/tensor fixtures exercising real Binding/HB/HE logic, not actual model outputs or CUDA execution. No remote sync or remote execution was used for this closure, and no full unchanged suite rerun was required.

Frozen source, reviewer command and outcomes: `p0/monitor/review-response-20260927/candidate-after-z-binding-independent/repair2/`. `AUDIT.json` SHA-256 is `eee7ab25da55c8c34362318b910ed1fe4432ddc2bcadad4fb961aa2b942bffb7`. Run `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <repair2-directory>/review_controls.py`. The initial and repair1 snapshots remain intact. Five reused helper source hashes were verified unchanged from the initial reviewed snapshot; final canonical repair2 bytes were verified unchanged at seal time.

No residual blocker remains in the requested source delta. The integration and runtime boundaries listed above remain pending; this PASS is neither a qualification result nor a launch gate.
