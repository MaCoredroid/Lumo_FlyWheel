# Fixed32 CFWD guarded full-value composition

Status: **default off; source/static verified; codegen, real byte gates, and
timing pending**.

This source-only checkpoint composes the guarded scan-bound specialization
with the previously reviewed full-value-tile change. It is bound to commit
`55d53ac5d4ef084e3a5bdd30771fa5a84bde73af` on branch
`agent/fixed32-cfwd-physical32-metadata-20260802`.

## Change

The prior `BV=64` geometry used two independent programs for the two halves of
each pinned 128-row value head. Those programs repeated accepted length, path
node, gate coefficient, bank offset, running-row metadata, K-vector loads, and
K normalization.

The composed candidate uses one parallel `BV=128` program per layer, request,
and value head. It adds no launch or persistent/transient buffer and creates no
serial recurrence dependency. The root-plus-accepted loop, active step count,
per-row FP32 state update, reduction axis, and final-store ownership are
unchanged. The ordered recurrence source hash remains
`d16ad65fe4affb85a85051bf8dc7530c17a34dd85826c05d6bd8adec67b1ce22`.

The composition retains the pre-replay device guard for accepted lengths
`[0,11]` and active physical nodes `[0,31]`, so the hot loop also retains zero
redundant bounds clamps.

## Static work model

Program count falls from 4,608 to 2,304 at B1 and from 18,432 to 9,216 at B4.
That removes one duplicate accepted-length, bank-offset, and running-row load
per eliminated program, plus two gate-coefficient loads. Per active recurrence
step it also removes one duplicate path-node load, one 128-element K-vector
load, one K-norm reduction, and one reciprocal square root per eliminated
program.

Total V/state element work remains parallel and unchanged. Pre-compiler FP32
state allocation rises from 32 to 64 elements per thread. Pinned-image codegen
must therefore establish registers, stack, local memory, and spill counts
before the candidate is eligible for a real byte gate or timing. No speedup is
claimed from source counts.

Wider grouping across value heads or layers was rejected because it would
introduce serial recurrence ownership or materially increase state lifetime.
The full value tile is the largest no-dependency reuse boundary in this Triton
layout.

## Evidence boundary

Host-only syntax and focused static suites passed. No Triton/CUDA codegen,
SASS/resource inspection, Docker/GPU execution, real SWE-Verified byte gate,
B1/B4 timing campaign, hardware-floor measurement, U95 acceptance test, or
synthetic probe was run for this bundle.

This bundle contains no prompts, responses, traces, raw logs, task IDs,
container identities, process identities, credentials, or secrets.
