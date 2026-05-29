# Round F F_a Unique-Node GDN Cost Gate Close-Out

Date: 2026-05-29

## Decision

NO-SHIP.

Lossless cheap unique-node tree verification for Qwen3.6 Gated DeltaNet is not
plausibly achievable on this stack with the current explicit per-node
parent-state materialization design. The only known correct path for the GDN
rank-1 delta term requires materializing per-node recurrent state, which is the
bandwidth-bound work the spec is trying to avoid.

This stops at the spec section-11 gate: do not continue building an
expensive-but-correct verifier.

## Measurements

Real SWE-bench-Verified pinned subset:

- `output/round_f_unified/stage3_e3_eager_baseline_smoke.json`
- `output/round_f_unified/stage3_fa_unique_spine_mechanism_on_debug_smoke.json`
- `output/round_f_unified/stage3_fa_unique_spine_mechanism_on_layer0_debug_smoke.json`

E3 eager baseline:

- mean accepted/event: `2.000`
- mean event ms: `231.322`
- decode tps: `16.679`
- acc sha: `8c1dccbc8d0de1e2f0b2ae9a546029be6db5e188a048cc0805b41e60414aa85e`

F_a unique-node spine with explicit parent-state gather/scatter enabled:

- mean accepted/event: `0.364`
- mean event ms: `243.148`
- decode tps: `6.418`
- acc sha: `50c15859e412c34c0b558f693e075b1af8d476ec059e001635c59797c2fc11a8`
- `path_rows_zero_rate`: `1.0`
- `selected_eq_verified_rate`: `1.0`
- `verifier_path`: `LUMO_FA_UNIQUE_NODES/spine_expanded_parent_state_rows`
- GDN state rows: `[4, 1]`
- parent map: `[-2, -1, 0, 1]`

Layer-0 debug rerun with the same mechanism:

- mean accepted/event: `0.250`
- mean event ms: `263.936`
- decode tps: `5.734`

The measured physical invariants are good; the verifier state path is not.

## Localization

The FA-unique mechanism is not a draft-only stub:

- `fa_unique_gdn_layer` fired across all 48 GDN layers.
- `num_spec_decodes=4`, `num_spec_decode_tokens=4`.
- `spec_initial_state_indices_tensor=[17,18,19,20]`.
- `spec_state_indices_tensor=[18,19,20,21]`.
- `spec_query_start_loc=[0,1,2,3,4]`.

The live tensor layout is also not the primary fault:

- query shape: `[1,4,16,128]`
- key shape: `[1,4,16,128]`
- value shape: `[1,4,48,128]`
- `a/b` shape: `[4,48]`

The failure localizes to the parent-state recurrent path: the expanded
unique-node rows compute a different GDN state and reject almost everything on a
degenerate spine that should match E3.

## Cost Gate

A single GDN SSM state row is:

```text
48 heads * 128 value dim * 128 key dim * 2 bytes = 1,572,864 bytes ~= 1.50 MiB
```

The conv state row is:

```text
10,240 channels * 6 state len * 2 bytes = 122,880 bytes ~= 0.117 MiB
```

A correct explicit parent-state delta update has to read a parent state and
write a child state. Minimum state traffic is therefore about:

```text
(1.50 MiB + 0.117 MiB) * 2 ~= 3.24 MiB per node per GDN layer
```

For the Stage-3 spine with 3 selected nodes over 48 GDN layers:

```text
3 * 48 * 3.24 MiB ~= 467 MiB/event
```

If root row materialization is counted, this rises to about `622 MiB/event`.
This excludes q/k/v projection traffic, attention, logits, scheduling overhead,
and branch commit work.

The spec Stage-3 speed budget is:

```text
1.05 * E3_event_ms = 1.05 * 231.322 = 242.888 ms
```

The incorrect explicit path already measures `243.148 ms/event`, just over the
budget, and real correctness work would add, not remove, GDN delta work. The
decode-tps result is worse: `6.418 tps` versus E3 `16.679 tps`, and also worse
than the known path-row K=2 baseline from the operator brief, approximately
`12.7 tps`.

## Why STree Does Not Rescue This Stack

STree is the plausible cheap method for diagonal SSM tree verification:
tree-mask packing plus log-accumulated diagonal transitions avoids explicit
per-node state copies.

Qwen3.6 Gated DeltaNet is not diagonal-only. It is diagonal gating plus a
rank-1 delta update. The diagonal gate can use STree-style accumulation, but the
rank-1 delta term is order-dependent and has no cheap published tree kernel in
the STree implementation or literature checked by the monitor. Correctness for
that term requires explicit per-node parent state or a novel chunked tree-delta
kernel.

No specific plausibly-cheap correct mechanism was identified.

## Verdict

Do not continue implementing explicit per-node GDN tree verification for Round F
on this stack. It is the expensive path, and making it correct is not expected
to meet the speed objective.

Ship decision: NO-SHIP for F_a unique-node GDN tree verification.

Last achieved gate: measurement harness and invariants can exercise the real
FA-unique parent-state path with `path_rows==0`, but Stage-3 correctness and
speed both fail.
