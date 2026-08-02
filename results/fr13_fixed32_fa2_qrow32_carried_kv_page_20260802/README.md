# Fixed32 B4 FA2 qrow32 carried K/V page address

Status: **host/source verified; GPU compile, SASS inspection, real-event byte
qualification, and timing pending**.

## Selection

After the initial K/V page-address fusion, each reverse-loop K advance still
resolved block `n - 1`, while the next unmasked iteration independently
resolved V at that same block. The qrow32 translation unit fixes noncausal,
nonlocal attention, so `n_masking_steps == 1`. Every unmasked-loop entry is
therefore dominated by the preceding guarded K advance for the same block.

## Change

The qrow32-only K-advance branch resolves the next relative page address once
and assigns both the K and V data pointers. The later V copy uses that carried
pointer. No K or V copy is moved; only the pointer assignment occurs earlier,
after the prior V async copy has completed.

For an `N`-block K sequence, active source resolver calls change from `2N - 1`
after the initial-only fusion to `N`: one initial partial-block resolution and
one full-block resolution for every remaining block. qrow16 and generic paged
fallbacks retain their original K and V resolver calls.

All carried addresses retain the exact canonical formula:

`physical_page * 1,048,576 + page_offset * 1,024 + column_offset`

## Preservation

- Launch geometry remains `6 x 4 x 4 = 96` CTAs per layer.
- K/V async-copy, fence, wait, GEMM, softmax, PV, and mask order is unchanged.
- The reverse K-block order and dynamic K length are unchanged.
- Initial final-tile clamping remains dynamic and shared by K/V.
- Page-table contents and physical-page selection remain dynamic.
- qrow16, stock/generic paged attention, and compile-time-dead append-KV paths
  are unchanged.

The resolver-call formulas above describe active specialized source. They are
not measured instruction counts or a performance claim. GPU compilation and
private SASS inspection must confirm realized codegen and resource usage before
the candidate is timing-eligible.

## Verification

- FA2 candidate, qrow32 gate, and static-gate host tests: `29 passed`.
- Python byte compilation: pass.
- Pinned-header transform: pass.
- Second-pass transform idempotence: pass.
- 262,144 modeled reverse-loop sequences covering 8,519,680 block states over
  64 threads, one through 64 K blocks, four nonidentity physical pages, and all
  final partial sizes 1 through 64: pass.
- Fail-closed rejection without fused static K/V addressing: pass.
- Source operation-order and fallback-retention checks: pass.
- `git diff --check`: pass.

No GPU command, synthetic performance probe, or real-task measurement was run.
Acceptance remains restricted to the standing real SWE-Verified task sets.

This directory contains aggregate source-level metadata only. It contains no
task text, prompts, responses, patches, model traffic, raw logs, environment or
process identities, binaries, credentials, or timing samples.
