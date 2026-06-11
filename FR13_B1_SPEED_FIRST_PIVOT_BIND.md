# FR13 B=1 Speed-First Pivot Bind

Date: 2026-06-11 UTC

## Decision

Pause the live B=1 lossless state-parity discriminator and move the active
trail to B=1 speed-tax localization first.

Reason: the lossless/superset failure is now banked tightly enough to avoid
losing it, but even a perfectly lossless cat9 cannot beat native while the
chain-width tree path remains about `1.39x` to `1.41x` slower per forward than
native MTP-5.

## Banked Lossless/Superset State

`FR13_B1_SUPERSET_PRECONDITION_BIND.md` is the current lossless/superset
checkpoint:

- Structural spine inclusion is true for cat9.
- The first prompt's first event has byte-identical spine draft tokens versus
  chain5.
- After a partial cat9 accept, the next-event true-spine proposal no longer
  preserves the chain/native opportunity on the same served prefix.
- Therefore the below-native cat9 accept/event result is a strong-lossless
  theorem-precondition failure, not an accept/event tuning target.

The interrupted follow-up state-parity discriminator did not complete. Its
partial artifacts are ignored local state under
`output/fr13_b1_state_parity_bind/`; they are not a verdict.

## Banked Speed State

Current clean B=1 speed evidence:

- Native MTP-5: `0.218160 s/fwd`, accept/event `3.161290`.
- Tree chain5 / `TREE_ATTN`: `0.304121 s/fwd`, `1.394026x` native,
  accept/event `3.256198`.
- Tree cat9 / `TREE_ATTN`: `0.307671 s/fwd`, `1.410299x` native,
  accept/event `2.151515`.
- Tree chain5 / `FLASH_ATTN`: `0.307007 s/fwd`, `1.407257x` native.

The key speed result is that cat9 is only `1.011673x` slower than chain5, and
chain5 stays slow under `FLASH_ATTN`. That moves the speed front away from
branch-row width and away from the full-attention backend as the dominant tax.

## Active Speed Hypothesis

The B=1 chain-width tax is in the tree GDN/replay/state path. The next proof
must measure the residual rather than re-argue the old all-node HBM export:

- replay-on already removes `tree_state_all` all-node state export.
- remaining candidates are GDN tree scan work, replay activation-ring staging,
  committer-side `launch_tree_gdn_replay`, graph-size overhead, and any
  persistent tree-mode scheduler/metadata overhead present even at chain5.
- the basis for speed claims remains `/metrics`
  `request_decode_time_seconds_sum / spec_decode_num_drafts_total`; TPS divided
  by accept is not a per-forward measurement.

## Next Gate

Run a B=1 chain-width speed-tax discriminator:

- Reference: native MTP-5 / `FLASH_ATTN`, clean deployment flags.
- Primary tree arm: chain5 / `tree_mtp`, replay route ON, `FR10_ENABLE_TREE_GDN=1`.
- Diagnostics, clearly labeled:
  - replay route OFF versus ON, to bound replay staging/replay-kernel overhead
    versus legacy export/remap.
  - if needed, fallback/no-tree-GDN only as a diagnostic with
    `FR10_ALLOW_LINEAR_FALLBACK=1`; never bind it as a valid gate.
  - short `nsys` or `ncu` capture when practical, reporting top CUDA kernels
    by cumulative time and naming `_tree_gdn_kernel`,
    `_tree_gdn_replay_kernel`, memcpy/copy activity, native GDN kernels, and
    attention kernels when visible.

Pass condition for this phase is not end-to-end B=1 success. It is a measured
attribution that names the removable part of the `~1.39x` chain-width tax and
selects the next concrete fix.
