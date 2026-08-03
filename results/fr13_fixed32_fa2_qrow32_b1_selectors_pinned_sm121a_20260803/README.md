# Fixed32 FA2 qrow32 B1 selectors, pinned SM121a binary

Status: **offline binary and ABI audit pass; real-task byte gates and timing
remain pending**.

This checkpoint productizes both fixed32 B1 query-row-32 kernels behind
independent, default-off selectors:

| arm | main grid/layer | main threads | context splits | combine |
| --- | ---: | ---: | ---: | --- |
| `no_split` | 24 CTAs | 64 | 1 | none |
| `split2` | 48 CTAs | 64 | 2 | stock FA2 split-K combine |

Both launchers revalidate the exact physical32 B1 geometry in C++. The
`split2` arm obtains `oaccum` and LSE scratch through FA2's stock
`set_params_splitkv` path. Production requires an arm-bound real-task PASS,
K64/root1, the canonical SWE-Verified exact4 identity, all 16 tree-attention
layers, and a pinned candidate-library digest. It has no steady-state device
synchronization and no silent fallback.

The pinned library is `5eec90f317cf6126cd57ab7f77b392ae6a1430d28210dcb31756abe788ef3467`
(300,140,712 bytes). It is intentionally not stored in Git; the manifest
records its complete identity and reproducible toolchain. Compared with the
qualified qrow16 reference library, its defined dynamic symbols, undefined
dynamic symbols, `DT_NEEDED` entries, and `RUNPATH` are byte-for-byte list
identical. The two new host launchers are local symbols and the two added
device images target `sm_121a`.

Pinned compiler resources are spill-free:

| kernel | registers | stack | local | static shared | SASS instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
| qrow32 B1 no-split | 252 | 0 | 0 | 1,024 B | 3,992 |
| qrow32 B1 split2 main | 254 | 0 | 0 | 1,024 B | 3,656 |
| stock FA2 combine specialization | 44 | 0 | 0 | 1,072 B | 1,080 |

No GPU, synthetic probe, task payload, prompt, response, credential, raw
SASS, object, or shared library is included here. Admission still requires
each arm to pass the raw-byte gate on the canonical real B1 task, followed by
full-step wall/TPS and phase timing on the standing four-task SWE-Verified
set. This checkpoint makes no speedup or hardware-floor claim.
