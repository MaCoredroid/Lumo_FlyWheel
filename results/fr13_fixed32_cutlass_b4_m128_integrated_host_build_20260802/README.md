# Fixed32 B4 M128 integrated host build

Status: pinned host compile/link and static binary audit pass with a recorded
`RUNPATH` drift. The candidate remains default off and is not
acceptance-valid.

## Result

The integrated `persistent_b4_m128_static` source compiled for `sm_121a` and
linked into the complete `_C_stable_libtorch` extension. The generated
dispatch is pinned by SHA256
`319bda31b05222e17eedefa65dbe328e87a9afd335301a75cf4bbb150911aedc`.
The exact linked output is installed read-only at:

`/home/mark/fr13_m128_static_build/bin/_C_stable_libtorch.persistent_b4_m128_integrated_direct_linear_static_coord_scalar_8eba8a4756234f70.abi3.so`

Its SHA256 is
`8eba8a4756234f706b5f2fc7d90ca47aa3e81905162d4395bda8e2a707752bce`,
its size is 113,253,424 bytes, and its mode is `0555`.

The build and audit were isolated from the live campaign. They did not launch
a GPU kernel, import the extension, use Docker, or run a synthetic/probe or
real-task workload. The linked binary, raw logs, raw symbol lists, cubins, and
raw SASS are not published; paths and cryptographic identities are reduced
here.

## Kernel audit

Both integrated output-dtype candidates retain the prior static resource
tuple: 168 registers, zero stack, zero local memory, 1,024 bytes static shared
memory, 384 threads per CTA, 1,792 parameter bytes, and 2,688 bytes in
`CONSTANT[0]`. Their resource dump is byte-identical to the prior static
binary's complete 309-record dump. The earlier static audit established that
those 309 records comprise 307 unchanged stock records plus the two static
M128 candidates.

For both BF16 and FP16, integration reduces the candidate body from 1,440 to
1,160 SASS instructions and from 23,040 to 18,560 text bytes. That is 280
instructions and 4,480 bytes fewer, or 19.444%. Relative to the incumbent
dynamic-persistent M128 body, the integrated body is 536 instructions and
8,576 bytes smaller, or 31.604%.

Both integrated bodies retain 128 `QMMA.16832.F32.E4M3.E4M3`, 128 `FFMA`, 72
`FMUL`, 48 `LDSM.16.M88.4`, 16 `STSM.16.M88.2`, and 32 dtype-specific output
packs. Neither contains `LDL`, `STL`, `LD.LOCAL`, `ST.LOCAL`, or `CALL`. Exact
instruction order and register operands are not claimed unchanged; the raw
body hashes and opcode-count deltas are published instead.

## Preservation audit

The integrated and prior-static extensions each contain 17 embedded cubins:
16 `sm_121a` and one `sm_89`. Sixteen cubins are byte-identical. The sole
changed cubin has the same 16-function device symbol inventory, and 14 of its
16 function bodies are byte-identical. Only the BF16 and FP16 static-M128
candidates changed. In particular, both incumbent dynamic-persistent M128
bodies remain byte-identical.

The host dynamic-symbol inventory remains 1,297 defined and 182 undefined
symbols. The undefined set and all nine `DT_NEEDED` entries are exact. The
defined-set delta is limited to replacing the two old candidate
`FusionCallbacks<LinearCombination>` weak `get_grid_shape` definitions with
the two integrated candidate `Sm90TreeVisitor<Sm90Compute<multiplies>>`
definitions. No stock definition changed.

One packaging difference is intentionally not hidden: the prior binary's
`RUNPATH` was
`/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64`,
whereas this build records
`/home/mark/fr13_streamk_build/venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64:`.
The exact linked binary was preserved rather than rewritten with `patchelf`.
Consequently this is not an exact host-ABI equivalence claim, and runtime load
compatibility remains untested.

## Scope boundary

Static-code reduction does not establish latency, throughput, occupancy, raw
output equivalence, or hardware-floor progress. The next valid gate is the
authenticated real SWE-Verified exact4 B4 raw-byte comparison under the pinned
K64/root1 physical root-plus-31 configuration. Full-step B4 timing is allowed
only after both Tail23 and Hydra27 byte gates pass.
