# Candidate joint mapper: F1 repair closure

2026-09-29. **PASS for the bounded mapper source repair.** The selected-column/full-metadata blocker in `candidate-joint-mapper-source-review.md` is closed at the hashes below. This does not approve a connected candidate collector, GPU launch, result or full-Q1 qualification.

Reviewed `experiments/review-response-20260927/tools/q1_candidate_joint_common_o0_v1.py` SHA `6218966216cb22a2d5cd29b6d5ae5dc9597f8398a41fb6d12badd0c1466a8cb2`; tests SHA `90f546c865414513357003783d93f37d7d9cfbfc3b70d7a289b1cb8b418321c2`. Immutable repair copies and CPU results are in `p0/monitor/review-response-20260927/candidate-joint-mapper-independent/repair1/`; the initial source and invalid acceptances remain in its parent directory.

The changed mapper now requires actual prepared sequence extent `P+32`, the production aligned column `(P+31)//1024`, matching `runner.mamba_state_idx`, and SSI root agreement with the exact CPU/GPU/request allocation column. It validates the one physical spare page expanded across the 32-column active SSI window, complete pure-spec counts and the active speculative mask. The new prepared sequence, logical column and metadata counts join the before/after identity receipt. The original single union transaction, storage disjointness, source-byte authentication, bank views and mutation/failure behavior are unchanged.

Independent execution used `/usr/bin/python3`, Torch 2.8.0, CPU tensors only, `CUDA_VISIBLE_DEVICES=''`, OMP/MKL/torch threads set to one. The tests execute the snapshotted repair bytes rather than importing a moving mapper. All **11 supplied tests** passed. All **10 additional controls** passed:

- P=1020, sequence length 1052, running column 1: accepted.
- P=1020 with native-style column 0: refused.
- P=1020 with SSI pointing to another allocated page at column 0: refused.
- Stale P=65 native sequence length 66, index 999, nonzero prefill count, false active mask and width-one SSI: each refused.
- P=1020 with valid 32-row graph-padded SSI and inactive masks: accepted.
- P=1020 with a differing request-allocation page at the selected column: refused.

The P=1020 positives exercise the real live mapper and actual tensors, not a full import of a newly generated P=1020 source. The supplied positive exercises the complete existing native64→candidate256 union import at P=65, with exact MTP readback and untouched scratch checks. No new model/reference/candidate data was generated.

The column rule agrees with the pinned candidate's `gdn_attn.py:1066–1073,1153–1158`, `v1/worker/mamba_utils.py:575–592`, and immutable-image align helper `cpu-native-state-map-20260929T121600Z/utils.py:879–893`. The positive padded-row control respects `gdn_attn.py:1294–1304`; it does not invent an exactly-one-physical-row restriction.

No residual blocker was found in this repaired scope. The future hook must still bind the admitted joint source, retain natural/imported records separately, and poison the process after a post-latch failure. Graph selector observation, after-Z proposal/sealing, final caller/freeze and runtime admission remain the separate connection work identified in `candidate-joint-mtp-connection-source-review.md`.
