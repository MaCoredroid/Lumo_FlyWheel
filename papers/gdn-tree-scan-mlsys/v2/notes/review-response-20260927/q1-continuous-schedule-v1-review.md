# Continuous schedule v1: independent CPU/source review

**PASS for the authenticated fixed-input schedule and event-order helper. No concrete blocker found in that scope.** This is not the runtime hook adapter, numerical evidence, a process/repeat denominator or launch approval. No GPU, model, Docker, network, source implementation or gate was touched.

Reviewed source `tools/q1_continuous_schedule_v1.py` SHA **c7c1e53bc2cb2be78f9ef143f6eacd804c2c50765b23f9798fca142c1755afb1**. Fixed `fullmodel/fixtures/token-fixtures.sequences-v2.json` SHA **e9683d2f8097ecfe195331d9977f9e57e1ab8482f1d583e04c5268ad92777318**. These paths are relative to `experiments/review-response-20260927`.

The exact loader authenticates all input bytes before scheduling. It retains48 records/108 declared cycles and derives156 forward plans including48 final flushes. Those flushes close the existing final pending-token observations; they are not48 additional qualification cases. Calibration/evaluation retain24 records each. No global qualification denominator is invented.

Independent checks of every record confirm:

- Each interior root consumes the preceding z once at the same position. The actual forward owns the next cycle's commit while its root logits are assigned to the prior O2. Each cycle has exactly one O2 observer; initial hydration occurs once.
- `before_z_first` and `before_z_follow` belong to the active accepted path. Interior `after_z_first` is explicitly unclaimed. Only the final root-only flush supplies that last-cycle phase and stops after its genuine deferred seal. This matches the preceding design review without fabricating a separate interior root MTP pass.
- Next drafts are staged while the old publication owner remains active. `Cursor` advances the index only after the current seal; its next pre-forward admits the already-staged owner. Early promotion, omitted draft handoff, wrong commit and duplicate callbacks refuse.
- The output budget is exactly `len(flat_consumed_tokens)+1`, with EOS ignored. Every final z is consumed once; its emitted natural O2 bonus is not added to the target-consumption trace.
- Current geometry is represented separately: candidate256-token physical pages, native64-token physical blocks, logical64 export chunks, and1024-token SSI span. The SSI column `(start+31)//1024` matches the current candidate mapper's32-row prepared query (`q1_candidate_joint_common_o0_v1.py:25–29`). All18 retained boundary inputs cross256 and64; the old1024 input placement is preserved rather than described as an actual physical256 runtime observation. `runtime_geometry_verified` stays false.

**28 independent CPU controls passed.** These include the full48 connected cursor traces, population/geometry/phase/budget checks, ten malformed-record refusals, thirteen event-order/finalization refusals, and two exact-input-byte refusals. The terminal callback trace explicitly calls `drafts_staged(None)` then `sealed('terminal')`. Exact code, results and source copy are retained under `p0/monitor/review-response-20260927/continuous-schedule-independent/`. Source bytes matched again after the controls.

## Required integration boundaries, not new schedule blockers

1. Use the authenticated `load` result (or an equivalently hash-bound immutable job), not arbitrary caller dictionaries. `schedule(record)` is a coherence helper: it does not independently recheck every off-path token, topology ancestor or split field. A direct synthetic change to an inactive row can pass `schedule`; the same changed fixture is rejected by `load`'s exact SHA. Do not expose the direct helper as untrusted runtime admission.
2. The `Cursor` raises on an invalid callback but does not poison itself permanently. A future fail-stop hook must latch any such exception and reject later callbacks/forwards. A synthetic duplicate `sampled()` leaves the cursor at `accepted`; catching it and continuing would be caller misuse, not a valid repaired runtime sequence.
3. `drafts_staged('terminal')` after the last real cycle means prepare for the final root-only consumer, using the real proposal output as in current cycle0. `drafts_staged(None)` during the flush acknowledges its actual final proposer callback; it must not index or write another fixture. If production does not execute this callback, the sequence is incomplete—do not synthesize it to satisfy the cursor.
4. The helper does not observe actual state, owner generations, logits, MTP phases, graph epochs or publication bytes. The pending adapter must bind these checks at the real seams and preserve partial invalid evidence. Terminal plan fields do not repeat ordinary-forward page/SSI derived fields; derive those from the same final root position if the runtime adapter needs them, rather than assuming the ordinary item keys exist.

The prospective hookup still requires the versioned phase manifest, native checkpoint successor and independent raw audit described in `q1-continuous-joint-connection-design-review.md`. No unchanged importer, recurrence or publication primitive is reopened by this review.
