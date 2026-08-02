# Fixed32 CFWD native full-value source candidate

Status: **default off; source/static verified only; not compiled, byte-qualified,
timed, or authorized for production**.

## Kernel

The candidate adds one private native vLLM `_C` CUDA operator. One CTA owns one
`(layer, request, value head)` across the full pinned `K=V=128` state matrix:

- 512 threads / 16 warps per CTA.
- One warp owns eight V rows.
- One lane owns four K columns for each of those rows.
- Each thread retains exactly 32 FP32 state elements before compiler effects.
- The first 128 threads load the BF16 K vector cooperatively into a 128-float
  shared buffer exactly once per root/accepted step.
- Four warp partials plus a warp-0 reduction normalize that one shared K vector.
- CTA barriers publish the node/scalars, norm, normalized K, and step completion.
- The 16 warps then update 16 disjoint eight-row V slices in the exact ordered
  root-plus-accepted recurrence and store only the final FP32 running bank row.

The static shared-memory source model is 548 bytes: 512 bytes for K, 16 bytes
for four norm partials, and 20 bytes for recurrence/control scalars. The pinned
committer state banks are FP32; no BF16 state-bank conversion was introduced.

## Static work model

The native candidate has one CTA per layer/request/value head: 2,304 CTAs/event
at B1 and 9,216 at B4. Relative to BV64, this removes the second CTA and its
duplicate 128-element K load, K-norm reduction, reciprocal square root, and
scalar/path metadata work per recurrence step. It does **not** remove initial or
final state-bank traffic, V loads, or state arithmetic.

| Occupancy | BV64 CTAs/event | Native CTAs/event | Native warps/event |
| --- | ---: | ---: | ---: |
| B1 | 4,608 | 2,304 | 36,864 |
| B4 | 18,432 | 9,216 | 147,456 |

The logical duplicate BF16 K traffic removed versus BV64 is 0.5625 MiB per
root-inclusive step at B1 and 2.25 MiB at B4. These are source-level counts,
not profiler DRAM measurements or speed claims. Compared with the unqualified
Triton BV128 source candidate, native CUDA keeps the same CTA count but uses 16
explicit warps and 32 state elements/thread instead of Triton's eight warps and
64 state elements/lane before compiler effects.

## Mandatory qualification

The source uses a different CUDA reduction tree and CUDA transcendental
codegen from the incumbent Triton implementation. Logical recurrence order is
preserved, but raw-byte equality is not inferred. The candidate is ineligible
until all of these pass in the pinned serving image:

1. `_C` compile for SM121 and resource inspection showing at most 64 registers
   per thread, two-CTA/SM eligibility, zero stack frame, zero local memory, and
   zero spill loads/stores.
2. Authenticated real SWE-Verified raw-byte comparison of all 48 FP32 running
   bank rows for every reachable accepted length 0 through 11. B1 uses a real
   task bracket; B4 uses the canonical exact4 campaign bracket.
3. Only after the byte credential exists, real SWE B1 and B4 full-step timing
   under the standing task-set rules.

The selector accepts only `diagnostic`, is default-off, fails closed on geometry,
dtype, guard, or operator drift, never emits a production credential, and is
not wired into the serving hot route by this source checkpoint.

## Verification

- New native source/contract/patch lifecycle suite: `34 passed`.
- New suite plus adjacent fixed32 committer/task-boundary suites: `89 passed`.
- Ruff, Python byte compilation, and `git diff --check`: pass.
- A broader sparse-worktree attempt reached `109 passed`; 10 harness cases
  could not run because the sparse checkout omits the canonical subset JSON
  files. Those failures did not exercise candidate code.
- No GPU command, CUDA compile, synthetic/probe performance run, real-task byte
  campaign, or timing measurement was executed.

This directory contains aggregate source metadata only. It contains no prompts,
responses, patches, traces, raw logs, process/container identities, credentials,
or timing samples.
