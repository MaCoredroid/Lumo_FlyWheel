# FR13 B=1 Speed + Lossless Pivot

Date: 2026-06-11 UTC

## Decision

Stop the current B=4 chase-down. B=4 serving is no longer the blocker; Step 3
showed a real quality deficit and a large per-forward speed deficit. The active
order is now:

1. Prove or fix B=1 strong lossless and speed, with accept/event as the
   required superset consequence check.
2. Only after B=1 is strong-lossless, near-native in forward cost, and
   measured at least native MTP-5 on accept/event, return to B=4 lossless and
   superset acceptance.

The speed target is effectively native parity. A tree/native forward ratio near
`1.0x` is the objective; a merely-sub-`1.1x` result is not enough unless the
acceptance math proves a real end-to-end win.

## Banked B=4 Stop Reason

Current Step 3 B=4 replay-on tree:

- Accept/event: tree `2.132045` vs native E5 K=5 `2.783088`, delta
  `-0.651043`.
- Replay-off is similar: tree `2.149832` vs native `2.783088`, delta
  `-0.633257`, so replay route itself is not the current quality root.
- Per-forward speed: tree replay-on `0.412014 s/fwd` vs native
  `0.263000 s/fwd`, ratio `1.566591x`.
- Replay-on did remove a real legacy cost: replay-on `0.412014 s/fwd` vs
  replay-off `0.480197 s/fwd`, about `14.2%` faster.
- The residual speed tax is structural in the current tree verifier shape:
  9-node tree verify rows versus native 5-token spine, plus any remaining
  GDN/tree scan, activation-ring, replay, TREE_ATTN, and committer overhead.

Conclusion: B=4 can run, but at current accept/event and forward ratio it
cannot beat native. Further B=4 debugging before B=1 speed+lossless risks
optimizing the wrong surface.

## Lossless Meaning

- Ground truth quality: pure target-model decode, no MTP.
- Deployed reference: native MTP-5, which verifies a single spine.
- Current object: a 9-node caterpillar tree verifier from the MTP head, with
  one branch off each spine position.
- Lossless claim: the tree verifier/committer preserves the underlying model's
  output distribution, not merely that it serves or accepts more tokens.
- Superset claim: because the tree contains the native MTP spine plus branches,
  a correct verifier should match native-spine quality and accept at least
  native MTP-5, with branches adding opportunity.
- B=1 consequence check: under the strong FR13 lossless preconditions, superset
  accept/event should follow. The B=1 tree must therefore measure at least the
  matched native MTP-5 accept/event before we return to B=4. The current clean
  B=1 caterpillar row (`2.1515` vs native `3.1613`) is a hard fail and means a
  lossless/superset precondition is still broken or unproven.

## Banked B=1 Facts To Reuse

- B=1 same-seed determinism at temp `0.6` was achieved after the per-request
  RNG, request-keyed buffers, and sequential remap fixes.
- The historical verify-path bar is the accepted FR13 gate:
  `within-E5-floor / argmax-lossless`, not literal `0.0` everywhere. For the
  live B=1 chasedown, that was operationalized as paired tree-vs-native event
  walks, first-fork classification, logit-margin checks, and served-stream
  loss classification.
- Old B=1 speed-tax rows are useful directionally but are not deployment
  verdicts: they used `BATCH_INVARIANT=1`, debug/logging regimes, and older
  replay state.
- The old B=1 lossless/acceptance localization named three seams:
  - S1: exact committer bonus-row bug on the `[0,2]` alternate leaf path.
  - S2: episodic verify-forward corruption on live multi-event rows.
  - S3: drafter spine tokens are not byte-identical to native; this dominates
    accept/event and is a superset blocker even if it is not by itself a
    p-lossless commit violation.
- Most recent B=1 chasedown state:
  - S1 was fixed in `4d45be27` and re-gated: `[0,2]` winners serve `st[2]`,
    `bonus_violations=0`, and true-spine `superset_violations=0`.
  - S2 branch-commit-class corruption was fixed by the committed-path conv
    window in `c0b53f5d`; the bound p0/gen-pos-16 corruption disappeared.
  - A residual spine-commit/gross-flip class remained on the legacy state path;
    the replay-route rebuild was the planned discriminator and is now the
    current route.
  - S3 remains an accept/superset blocker: drafter/verify shape makes the
    caterpillar spine less native-like than chain/native. It is not by itself a
    p-lossless violation, but it keeps accept/event below native and therefore
    can still kill end-to-end speed.
- Do not re-chase the fixed B=4 handoff crash, replay-route all-node HBM export,
  or old B=1 global-RNG/slot/racy-remap issues unless fresh evidence points
  back there.

## B=1 Lossless Bar To Use

Match the latest B=1 chasedown methodology, adjusted only for the current clean
speed regime. Source docs: `FR13_FLOOR_WORKFLOW_VERDICT.md`,
`FR13_ACCEPTANCE_LADDER_BIND.md`, `FR13_S1S2S3_DISCRIMINATE_BIND.md`,
`FR13_CONVFIX_AB_BIND.md`, and `FR13_REPLAY_GPU_GATES_BIND.md`.

- Use the pinned B=1 sequential battery from history:
  `output/fr13_acceptance_ladder/prompts_swe4.json`, `max_tokens=128`,
  greedy `temp=0.0/top_p=1.0/seed=1313`; include temp `0.6/top_p=0.95` after
  the greedy lossless path is clean.
- Pin BI state on both arms. For the speed regime this means BI off on both
  arms; if a diagnostic BI-on run is used, label it diagnostic and do not mix
  its lossless verdict with the speed verdict.
- Require same-seed repeat determinism before interpreting tree-vs-native
  differences.
- Re-run the S1/S2/S3 style reducer:
  - S1 bar: no wrong-row bonus reuse; `bonus_violations=0`; true-spine
    diagnostics, not leaf-order path-0 diagnostics.
  - S2 bar: no gross verify-forward corruption and no served-token fork outside
    the accepted native/cross-boot floor; any fork must be classified by margin
    and trigger context, not waved through.
  - S3 bar: report drafter/verify-shape acceptance deficit separately. It is
    not the p-lossless bar, but it is the superset/accept bar.
- Greedy pass condition: no remaining lossless-class served-stream fork under
  the S1/S2 classification. If tokens differ, classify them against the
  native/cross-boot floor before calling pass or fail.
- Temp-0.6 pass condition: output distribution within the historical E5/native
  self-noise floor, after the greedy S1/S2 path is clean.

## Next Gate

Run or bind a clean current B=1 gate with:

- Tree: `TREE_ATTN/tree_mtp`, 9-node caterpillar, `FR13_REPLAY_ROUTE=1`.
- Reference: native MTP-5, `FLASH_ATTN/naive_mtp`, `NUM_SPECULATIVE_TOKENS=5`.
- Pure target-model decode remains the conceptual ground truth, but the B=1
  live gate should stay native-MTP paired unless an existing no-MTP artifact is
  already available.
- `MAX_NUM_SEQS=1`.
- Deployment speed flags: `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`,
  `FR13_BI_TREE_ATTN=0`, diagnostics/capture envs unset.
- CUDA graph capture allowed and proven for both arms.
- Same prompts, seeds, temp/top-p, max-tokens, and prompt SHA across arms.
- Speed basis: vLLM `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`; also report
  warm decode TPS and accept/event.
- Lossless basis: the B=1 chasedown bar above; do not replace it with a new
  ad-hoc metric.
- Engagement asserts: tree drafts/event equals 9, native drafts/event equals 5.

Pass bar for moving past B=1:

- No known B=1 lossless-class violation remains under the historical S1/S2
  bar, with temp-0.6 distribution inside the historical floor once greedy is
  clean.
- Tree accept/event must be at least matched native MTP-5 on the B=1 workload
  as the measured consequence of the strong-lossless/superset proof; the
  intended result is tree > native, with branches adding opportunity rather
  than reducing acceptance.
- Tree/native per-forward ratio is near `1.0x`; do not accept a `1.1x` class
  result as good enough without an explicit end-to-end break-even calculation.
- Accept/event alone is not a lossless proof. Below-native accept/event is a
  theorem-check failure for the current implementation and blocks moving back
  to B=4.
