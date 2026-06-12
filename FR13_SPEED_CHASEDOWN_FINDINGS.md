# FR13 speed chase-down findings

Date: 2026-06-12 UTC.

## Verdict

The current speed problem is not a thin miss and not a single already-fixed
diagnostic tax. Across the banked B=1 and B=4 artifacts, the remaining forward
tax follows the broader `tree_mtp` execution path: graph/row shape, tree-mode
scheduler/metadata, GDN/replay/state handling, and committer bookkeeping.

Several suspected dominant causes are now ruled out or bounded:

- Default diagnostic trace payload was removed; chain5 stayed around
  `0.303 s/fwd`.
- Branch width is not the primary B=1 speed tax; cat9 is only about `1.01x`
  slower than chain5.
- `TREE_ATTN` full-attention backend is not dominant; chain5 under
  `FLASH_ATTN` remains in the same slow class.
- Replay route itself is not dominant; replay-off is only about `2.7%` slower
  than replay-on in the clean B=1 attribution run.
- Tree GDN scan alone is not enough to explain the whole tax; no-tree-GDN
  fallback still pays most of the slowdown.
- Per-layer replay flag `.item()` checks are not the main wall-time bottleneck:
  reducing their DtoH sync class changed the profile but barely moved
  `s/fwd`, and the representative-layer variants failed safety.

The next valid speed work should preserve lossless semantics first, then target
the tree-mode graph/row/scheduler path or the GDN-state algorithm itself. Do
not continue representative-layer validation. If flag validation is optimized,
it must keep all-layer fresh/row checks and batch the DtoH read.

## Measurement rule

All speed rows below use the valid basis:

```text
request_decode_time_seconds_sum / spec_decode_num_drafts_total
```

Do not derive per-forward speed from TPS divided by accept/event. Warm decode
TPS and accept/event are useful context, but the speed denominator is the
speculative draft event count from `/metrics`.

## Current B=4 stop reason

Step 3 B=4 is frozen until B=1 clears speed and lossless/superset.

| arm | draft width | decode seconds | spec drafts | s/forward | accept/event | warm TPS |
|---|---:|---:|---:|---:|---:|---:|
| tree replay-on | 9 | 255.860496 | 621 | 0.412014 | 2.132045 | 7.566623 |
| native MTP-5 | 5 | 143.072108 | 544 | 0.263000 | 2.783088 | 14.314460 |
| tree replay-off | 9 | 285.237044 | 594 | 0.480197 | 2.149832 | 6.513880 |

Findings:

- Replay-on tree/native is `1.5666x` per forward.
- Replay-off/native is `1.8258x`.
- Replay-on is about `14.2%` faster than replay-off, so replay-on removed a
  real legacy cost.
- The remaining B=4 gap is structural enough that B=4 cannot beat native while
  it is both slower per forward and lower on accept/event.

Source: `FR13_STEP3_SPEED_FORENSICS.md`.

## Current B=1 gate

The clean B=1 cat9 gate fails both speed and superset acceptance.

| arm | draft width | decode seconds | spec drafts | s/forward | accept/event | warm TPS |
|---|---:|---:|---:|---:|---:|---:|
| cat9 tree | 9 | 50.765643 | 165 | 0.307671 | 2.151515 | 10.085561 |
| native MTP-5 | 5 | 27.051817 | 124 | 0.218160 | 3.161290 | 18.926640 |
| cat9 tree repeat | 9 | 50.754538 | 165 | 0.307603 | 2.151515 | 10.087768 |
| native repeat | 5 | 27.126764 | 124 | 0.218764 | 3.161290 | 18.874349 |

Findings:

- Cat9/native is `1.4103x` on the first window and `1.4061x` on repeat.
- Cat9 accept/event is `2.1515`, far below native `3.1613`.
- Same-seed repeat determinism passed for both arms.
- S1 internal bonus-row/superset diagnostics were clean.
- Served streams still fork from native on all four prompts, and this run did
  not capture the final-logit margin data needed for an S2 pass.

Source: `FR13_B1_CURRENT_GATE_BIND.md`.

## Chain-width discriminator

Chain5 isolates the speed tax from branch width.

| arm | draft width | s/forward | accept/event | warm TPS |
|---|---:|---:|---:|---:|
| native MTP-5 | 5 | 0.218160 | 3.161290 | 18.926640 |
| tree chain5 | 5 | 0.304121 | 3.256198 | 13.913580 |
| tree cat9 | 9 | 0.307671 | 2.151515 | 10.085561 |

Ratios:

- Chain5/native: `1.3940x`.
- Cat9/native: `1.4103x`.
- Cat9/chain5: `1.0117x`.

Findings:

- The large B=1 speed tax exists even at native-width chain5.
- Cat9 branch width adds little incremental per-forward cost over chain5 in
  the clean B=1 regime.
- Chain5 accept/event is above native in this run, but served streams still
  fork from native on all four prompts. This is not a lossless pass.

Source: `FR13_B1_CHAIN_SPEED_DISCRIMINATOR.md`.

## Backend ablation

`TREE_ATTN` is not the dominant speed tax.

| arm | backend/mode | s/forward | ratio vs native | accept/event |
|---|---|---:|---:|---:|
| native MTP-5 | `FLASH_ATTN/naive_mtp` | 0.218160 | 1.0000x | 3.161290 |
| chain5 tree | `TREE_ATTN/tree_mtp` | 0.304121 | 1.3940x | 3.256198 |
| chain5 tree | `FLASH_ATTN/tree_mtp` | 0.307007 | 1.4073x | 2.871212 |

Findings:

- Switching chain5 tree mode from `TREE_ATTN` to `FLASH_ATTN` does not move it
  toward native speed.
- The full-attention tree-bias backend is therefore not the dominant B=1 speed
  tax.

Source: `FR13_B1_BACKEND_ABLATION_BIND.md`.

## Replay and GDN attribution

The clean B=1 attribution run separates replay route and tree GDN from the
larger residual.

| arm | validity | s/forward | ratio vs native | accept/event |
|---|---|---:|---:|---:|
| native MTP-5 | valid reference | 0.217955 | 1.0000x | 2.850746 |
| chain5 replay-on | valid primary | 0.303595 | 1.3929x | 2.807407 |
| chain5 replay-off | diagnostic | 0.311789 | 1.4305x | 2.807407 |
| chain5 no-tree-GDN fallback | diagnostic | 0.293616 | 1.3471x | 2.946154 |

Findings:

- Replay-off is only `2.699%` slower than replay-on; replay is not the
  dominant cause of the residual slowdown.
- No-tree-GDN fallback still pays most of the tax, so the tree GDN kernel alone
  is not the whole explanation.
- The remaining surface is broader tree-mode row/graph/scheduler behavior
  shared by tree mode and the diagnostic fallback.

Source: `FR13_B1_SPEED_TAX_ATTRIBUTION_BIND.md`.

## Diagnostic trace removal

The launcher no longer enables high-volume diagnostics by default:

- `LUMO_MTP_DRAFT_TRACE_FILE`
- `LUMO_TREE_SAMPLER_DEBUG_LOG`
- `LUMO_TREE_PATH_LCP_LOG`

Clean no-log chain5 replay-on result:

| arm | decode seconds | spec drafts | s/forward | accept/event | warm TPS |
|---|---:|---:|---:|---:|---:|
| chain5 replay-on no-log | 39.737744 | 131 | 0.303342 | 2.931298 | 12.884476 |

Findings:

- Removing default diagnostics made clean speed verdicts trustworthy but did
  not uncover a near-native result.
- No-log Nsight reduced DtoH bytes/time versus the earlier log-heavy profile,
  but CUDA graph node events stayed at `43,270`.

Source: `FR13_B1_TRACELESS_SPEED_BIND.md`.

## Nsight profile evidence

The native-vs-chain5 profile shows the residual tree-mode surface, but it does
not yet give per-kernel attribution.

| signal | native MTP-5 | chain5 replay-on | chain/native |
|---|---:|---:|---:|
| s/forward | 0.217036 | 0.304396 | 1.4025x |
| CUDA graph node events | 14,368 | 43,270 | 3.0116x |
| graph node events per draft | 199.56 | 627.10 | 3.1425x |
| `cudaMemcpyAsync` calls | 7,865 | 18,512 | 2.3537x |
| GPU DtoH memcpy count | 326 | 7,752 | 23.7791x |

After no-log cleanup:

| signal | log-heavy chain5 | no-log chain5 | current / prior |
|---|---:|---:|---:|
| CUDA graph node events | 43,270 | 43,270 | 1.0000x |
| GPU DtoH memcpy count | 7,752 | 7,598 | 0.9801x |
| GPU DtoH memcpy bytes | 4,017,072 | 39,176 | 0.0098x |
| GPU DtoH memcpy time | 36.621952 ms | 7.293856 ms | 0.1992x |

Findings:

- The large DtoH byte surface in the first profile was mostly diagnostic trace
  payload.
- The graph-node and memcpy-call shape remained high after diagnostics were
  removed.
- Nsight kernel summary exports were empty, so there is no banked per-layer or
  per-kernel timing table yet.

Sources: `FR13_B1_PROFILE_BIND.md`, `FR13_B1_TRACELESS_SPEED_BIND.md`.

## Replay flag-sync chase

The latest flag-sync numbers look good only on repeatability, not on lossless
or wall-speed.

| variant | decode seconds | spec drafts | s/forward | accept/event | result |
|---|---:|---:|---:|---:|---|
| first-layer default | 43.308411 | 143 | 0.302856 | 2.566434 | safety failed |
| first-layer default repeat | 39.326723 | 130 | 0.302513 | 3.000000 | safety failed |
| strict all-layer repeat | 40.387729 | 133 | 0.303667 | 2.909774 | stable |
| last-layer default | 42.500922 | 141 | 0.301425 | 2.659574 | safety failed |
| last-layer default repeat | 42.474751 | 141 | 0.301239 | 2.659574 | default repeat pass |
| last-layer strict all-layer | 40.317079 | 133 | 0.303136 | 2.849624 | safety reference |

Last-layer safety:

- Default vs default repeat: `4/4` exact token sequences, pass.
- Default vs strict: `2/4` exact token sequences, fail.
- First mismatch: `prompt_id=1`, `sample_index=0`, `first_diff=25`,
  `default_token=471`, `strict_token=14`.

Profile signal from the first representative attempt:

| signal | prior no-log profile | representative flag-sync | current / prior |
|---|---:|---:|---:|
| CUDA graph node events | 43,270 | 43,270 | 1.0000x |
| GPU DtoH memcpy count | 7,598 | 814 | 0.1071x |
| GPU DtoH memcpy bytes | 39,176 | 12,040 | 0.3073x |
| GPU DtoH memcpy time | 7.293856 ms | 4.055552 ms | 0.5560x |
| `cudaStreamSynchronize` calls | 8,538 | 1,703 | 0.1995x |

Interpretation:

- The `.item()` checks were a real sync-class profiler smell.
- Removing most of them did not materially change wall speed: the run stayed
  around `0.301-0.303 s/fwd`.
- Representative-layer validation is unsafe. First-layer failed; last-layer
  recovered default repeatability but still differed from strict all-layer
  validation.
- The likely reason is semantic ordering/freshness: the old all-layer `.item()`
  checks acted as synchronization/freshness barriers across per-layer GDN
  replay staging. Checking one representative layer does not preserve the
  all-layer invariant.

Decision:

- Do not retry first-layer or last-layer representative checks.
- The only valid flag-sync optimization is all-layer batched validation:
  preserve every layer's fresh flag and staged-row check, but aggregate/read the
  small flag tensor matrix with one DtoH sync.
- Even if successful, this is expected to be a small wall-speed cleanup, not
  the main route to native parity.

Source: `FR13_B1_REPLAY_FLAG_SYNC_BIND.md`.

## Superset/lossless interaction

Speed and lossless remain coupled by the break-even math but must be proven
separately.

- Lossless target: preserve the target model output distribution.
- Deployed reference: native MTP-5 single spine.
- Current object: 9-node caterpillar tree from the MTP head, one branch off
  each spine position.
- Superset expectation: a correct tree contains the native MTP spine and should
  accept at least native MTP-5, with branches adding opportunity.

Current problem:

- Cat9 structurally includes the spine, but after partial accepts its future
  spine proposals stop matching chain/native on the same served prefix.
- The clean B=1 cat9 accept/event deficit, `2.1515` vs native `3.1613`, is
  therefore a theorem-precondition failure, not an accept-tuning target.
- Chain5 proves the tree backend can reach above native accept/event at
  native-width geometry, but chain5 still forks from native and is not a
  lossless pass.

Sources: `FR13_B1_SPEED_LOSSLESS_PIVOT.md`,
`FR13_B1_SUPERSET_PRECONDITION_BIND.md`,
`FR13_B1_CHAIN_SPEED_DISCRIMINATOR.md`.

## Current slow-point hypothesis

The strongest current hypothesis is:

```text
tree_mtp graph/row/scheduler path
  + GDN/replay/state handling
  + tree-mode metadata/committer bookkeeping
```

This is more precise than "HBM tax" and broader than "one kernel is slow".

What is not yet proven:

- Exact per-layer or per-kernel timing attribution.
- Whether the largest removable piece is graph-node/runtime overhead,
  GDN-state traffic, replay/activation ring staging, row materialization, or
  Python/scheduler metadata around `tree_mtp`.
- Whether a pure-spine fast path can safely reuse the native MTP-5 graph/row
  path while preserving the tree verifier contract.

## Next valid speed paths

1. All-layer batched flag validation.
   - Purpose: preserve strict all-layer semantics while reducing DtoH syncs.
   - Gate: default repeat exact match and default-vs-strict exact match.
   - Expected wall impact: probably small, because representative attempts did
     not move `s/fwd` materially.

2. Pure-spine fast path or native-path reuse for chain5.
   - Purpose: collapse the chain5 `tree_mtp` graph/row/scheduler overhead toward
     native MTP-5.
   - Gate: chain5 remains deterministic, does not introduce lossless-class
     forks beyond the accepted B=1 floor workflow, and stays at least native on
     accept/event.
   - Risk: chain5 is only a discriminator; cat9 still needs the branch/superset
     theorem to hold before returning to B=4.

3. GDN algorithmic state work.
   - Candidate: WY one-pass plus accept-only state commit, with persistent
     buffer preallocation and shared-prefix state reuse.
   - Rationale: this is the kernel-side hypothesis that can reduce binding HBM
     state traffic instead of just moving launches.
   - Gate: bit-exact/lossless validation first, then live speed.

4. Better profiler attribution.
   - Need a short profile that yields non-empty kernel/module timing, or an
     equivalent graph-node/NVTX table.
   - Required separation: `_tree_gdn_kernel`, `_tree_gdn_replay_kernel`,
     activation-ring copies, native GDN/update kernels, TREE_ATTN/FLASH_ATTN
     kernels, sampler/committer kernels, and graph runtime overhead.

5. Cat9 superset precondition fix.
   - Purpose: branches must add opportunity without degrading the native spine
     proposal sequence after partial accepts.
   - Gate: cat9 accept/event at least native MTP-5 on B=1, plus the historical
     S1/S2 lossless reducer.

## Do not re-chase

- B=4 draft-token handoff crash unless fresh evidence points there.
- Replay-off legacy all-node HBM export as if replay-on still paid it.
- Default diagnostic trace payload.
- `TREE_ATTN` as the dominant B=1 chain-width speed tax.
- Branch width as the main speed tax before fixing chain5.
- Representative-layer flag validation.
- TPS/accept hand-derived forward time.

## Pass bar before returning to B=4

B=1 must clear all of:

- Near-native per-forward speed, measured by `/metrics`.
- Strong lossless under the historical B=1 methodology, not just byte-repeat.
- Accept/event at least matched native MTP-5 as the superset consequence check.
- No new safety shortcut that differs from strict all-layer behavior.

Only after B=1 clears those gates should B=4 lossless/superset and deployment
speed be resumed.
