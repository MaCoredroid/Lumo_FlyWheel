# Fixed32 CFWD gate/norm overlap source readiness

Status: **experimental, unselected, and source-structure verified; CUDA compile,
codegen, byte qualification, real-task timing, and performance evaluation are
pending**.

## Variants

`native/experimental/fr13_fixed32_cfwd_native_fullvalue_overlap.cu` separates
node publication from gate-scalar work. Threads `0..steps-1` publish each node
once, then one CTA barrier makes the table visible. Up to the first 36 threads
retain the canonical gate arithmetic while other warps can enter K
normalization and advance to its first barrier. The first normalization barrier
and final precompute barrier publish every gate scalar before recurrence reads
it.

This is a scheduling-overlap hypothesis, not a barrier-removal claim. The
canonical post-gate barrier is replaced by the earlier node-table barrier. Both
canonical and overlap-only sources contain five static `__syncthreads()` sites
and execute nine barrier arrivals per thread for a valid CTA: one metadata
barrier, one gate/node barrier, six barriers across three normalization waves,
and one final publication barrier.

`native/experimental/fr13_fixed32_cfwd_native_fullvalue_overlap_active_waves.cu`
adds a separable CTA-uniform guard. `active_waves = ceil(steps / 4)` is derived
from the shared, barrier-published step count and guards every per-wave shared
access and both barriers. It skips two runtime barriers for each inactive
trailing wave:

| Steps | Active waves | Norm barriers | Total valid-CTA barriers | Skipped vs overlap-only |
| ---: | ---: | ---: | ---: | ---: |
| 1-4 | 1 | 2 | 5 | 4 |
| 5-8 | 2 | 4 | 7 | 2 |
| 9-12 | 3 | 6 | 9 | 0 |

Both sources remain absent from the vLLM patcher, runtime selector, source
binding, and codegen checker. The canonical qualified source remains unchanged
at SHA256
`1c1a9813410dcf15bcbb4d23bec71ee16ddcd7e2dbe3b1a3698e58f71bd96985`.

## Validation

- Focused experimental and canonical source suite: `71 passed`.
- Ruff, Python byte compilation, canonical-source byte check, and whitespace
  checks: pass.
- No CUDA compile, codegen/resource/SASS inspection, Docker/container action,
  GPU query or launch, byte-correctness gate, real-task campaign, synthetic
  probe, timing measurement, hardware-floor result, or performance claim is
  represented here.

This bundle contains reduced source-readiness facts only. It excludes tasks,
model inputs or outputs, requests, responses, patches, environment dumps, raw
logs, process or container identities, binaries, objects, and timing samples.

From this directory, verify the manifest syntax and file checksums with:

```bash
python3 -m json.tool manifest.json >/dev/null
sha256sum --check SHA256SUMS
```
