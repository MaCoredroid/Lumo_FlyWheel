# Fixed32 GDN ordered-root-loop live qualification readiness

Status: **READY_NOT_EXECUTED**.

This reduced artifact binds the default-off
`fixed32_gdn_single_launch_root_loop_v1` candidate to source commit
`08dc6187a0ccf70b13f8c988f77b02d126ffdc79` and to its exact B1/B4
`sm_121a` code generation. It is not a byte-parity PASS, timing result,
acceptance result, performance claim, or production authorization.

## Serving boundary

The existing production node helper and static-root kernel bodies are unchanged:

- `_tree_gdn_fixed32_single_launch_node` source SHA-256:
  `39734a9dcfaf14c45de0fd35d8e0b12f6c099a8c3850fdc4c2b472dd3229ca6c`
- `_tree_gdn_kernel_fixed32_single_launch` source SHA-256:
  `870fe9943a8e33b7dff6457b4e8524ef85173cbbea4ddd432a3effd9991cd03d`

The production selector remains default-off. During either live gate, the
candidate runs only as a shadow comparison and the captured reference graph is
always served. Candidate output and state can become production-eligible only
after an exact, source-bound PASS for that batch profile.

Each authenticated task comparison covers all 48 layers and exact raw bytes for
`out`, `ring_k`, `ring_v`, `ring_a`, `ring_b`, `flags`, and `counter`. Candidate
state and every authoritative surface are restored to the reference snapshot.
The B1 runner accepts exactly the first canonical task as a diagnostic; the B4
runner requires the canonical exact4 task set. Capture-only, speed-probe,
probe-only, synthetic, and unauthenticated routes are rejected.

## Offline build

Two fresh processes with isolated Triton caches compiled the exact B1 and B4
specializations for `sm_121a`. Both passes were deterministic for every audited
variant. The ordered-root-loop candidate uses one launch per layer, 768 CTAs per
request, 8 warps per CTA, 112 registers per thread, 1,592 primary SASS
instructions, and 25,472 primary text bytes. Stack, local memory, LDL, STL,
calls, indirect branches, global scratch, and tensor memory are all zero.

The register count is an unmeasured runtime risk. Static code size does not
establish latency, throughput, occupancy, or acceptance behavior.

## Boundary

`CUDA_VISIBLE_DEVICES` was empty. No GPU kernel, Docker container, service,
SWE-Verified task, synthetic probe, byte gate, CUDA graph, timing arm, or
acceptance campaign was launched. Raw cubin, PTX, SASS, IR, compiler logs, task
data, prompts, and responses are not retained. A real B1 diagnostic followed by
the exact4 B4 gate is still required.
