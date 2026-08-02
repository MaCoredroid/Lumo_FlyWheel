# Fixed32 GDN ordered-root-loop SM121a codegen audit

Verdict: **CODEGEN_WIN_NOT_WIRED**.

This reduced offline artifact records a code-generation improvement for
`fixed32_gdn_single_launch_root_loop_v1`. The candidate replaces only the
five-way `tl.static_range(0, ROOT_STEPS)` expansion with the ordered
`tl.range(0, ROOT_STEPS)` loop. It remains one physical launch per layer with
the exact 32-node depth-first recurrence, 768 CTAs/request, and 8 warps/CTA.

The candidate is codegen-only. It is not referenced by the serving launcher
or runtime manifest, and the production kernel remains byte-for-byte unchanged
at SHA-256
`ca5ff6496c7cf3221996e6aa5971d36207e305e51f5c4a308f71d15165ab659a`.
No production, acceptance, or performance status is implied.

## Static codegen result

Two fresh-process, isolated-cache builds produced byte-identical reports and
cubins for the exact B1 and B4 live specializations.

| Metric | Static baseline | Ordered-root-loop | Delta |
| --- | ---: | ---: | ---: |
| Primary SASS instructions | 7,232 | 1,592 | -5,640 (-77.986726%) |
| Primary text bytes | 115,712 | 25,472 | -90,240 (-77.986726%) |
| Supplemental capmerc text bytes | 15,118 | 3,594 | -11,524 (-76.227014%) |
| Registers/thread | 97 | 112 | +15 (+15.463918%) |
| Registers/CTA | 24,832 | 28,672 | +3,840 |
| Stack / local bytes | 0 / 0 | 0 / 0 | unchanged |
| LDL / STL / calls | 0 / 0 / 0 | 0 / 0 / 0 | unchanged |

`cuobjdump` and `nvdisasm` independently count 1,592 primary instructions for
both batch specializations. The candidate also has zero indirect branches,
global scratch, and tensor memory in the retained resource summary.

The register increase is material and remains an unmeasured runtime risk. No
occupancy was queried or inferred.

## Work accounting

The launch geometry and structural recurrence work are unchanged:

- B1 grid: 48 x 16 x 1, or 768 CTAs in one launch;
- B4 grid: 48 x 16 x 4, or 3,072 CTAs in one launch;
- CTA-warps/request: 768 x 8 = 6,144;
- ordered recurrence warp-node steps/request: 32 x 6,144 = 196,608;
- ordered recurrence warp-node steps/B4 launch: 786,432.

For comparison only, multiplying static body instructions by CTA-warps gives
44,433,408 for the baseline and 9,781,248 for the candidate per request. This
is a static SASS footprint, not executed instruction work: the loop body runs
five times. It is not a latency, throughput, occupancy, or speed model.

## Source proof

The focused AST test normalizes the function name and docstring, converts
exactly one production outer root `static_range` call to `range`, and then
requires the complete production and candidate function ASTs to match. It also
pins the member loop as static, the branch path loop as ordered, the two node
helper call sites, absence of state export, and absence from the serving
launcher and runtime manifest.

The exact B1/B4 compile constants remain N=32, KH=16, VH=48, DK=128, DV=128,
BV=8, root steps 5, maximum branch length 7, maximum group members 3, groups
5, output scale `128**-0.5`, QK normalization on, raw gating on, h0 bank on,
counter/ring/flags export on, and scan alignment off. B1/B4 use flag rows 1/4.

## Rejected alternative

A more aggressive dynamic member loop reduced the body to 3,800 instructions
and 60,800 primary text bytes with 80 registers/thread, but emitted an 8-byte
stack frame plus one LDL and one STL. It was rejected after one build because
it violates the zero-spill constraint. That temporary source was restored and
is not retained.

## Boundary

The harness ran with `CUDA_VISIBLE_DEVICES` empty. Triton 3.6.0 and torch
2.10.0+cu130 compiled for `sm_121a` using the packaged CUDA 12.9
`ptxas-blackwell`; CUDA 13.0 `cuobjdump` and `nvdisasm` inspected the output.

No GPU kernel, Docker container, service, SWE task, synthetic probe, byte gate,
CUDA graph, timing, or acceptance campaign ran. Raw cubin, PTX, SASS, IR, and
compiler logs are not retained. A separate byte-parity and live qualification
would be required before considering any production wiring.
