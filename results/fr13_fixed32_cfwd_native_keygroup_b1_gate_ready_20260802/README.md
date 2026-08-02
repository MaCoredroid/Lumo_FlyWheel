# Fixed32 native key-group CFWD B1 gate readiness

Status: **the exact K64/root1 B1 real-task byte-gate path is integrated and
source-bound; the full vLLM rebuild and live gate are pending**.

## Candidate

The candidate keeps one CTA per `(layer, request, key_head)` and processes the
three value heads sequentially. The exact full translation unit previously
passed its SM121a resource contract with 64 registers per thread, 6,568 source
static shared bytes, 7,592 reported shared bytes, and zero stack, spills, local
memory, `LDL`, `STL`, or `CALL`. The frozen object from that compile has SHA256
`8a20199ee6ad357f6188aed5551fa2697c0185ca86edf63f4fa2d8b8f29649b0`.

The candidate remains default-off, diagnostic-only, timing-ineligible, and
unauthorized for production.

## Binary provenance

Binding issuance now requires all of the following before a live run:

1. Exact pinned vLLM commit and exact patched source bytes.
2. The canonical full `vllm/_C.abi3.so` build target, not the stable extension
   or an arbitrary shared object.
3. Exactly one candidate CUDA object reachable from that full target.
4. A forced rebuild after making the pinned candidate source newer than the
   existing object and extension.
5. Candidate object and full-extension timestamps in source, object, link
   order, plus bound object and extension SHA256 identities.
6. An exact mode-0400, single-link private binding checked again at install and
   at postvalidation.

This closes the stale-object and wrong-extension gap in the earlier gate
integration. No live binary or binding is represented by this reduced artifact.

## Real B1 byte gate

The launcher is pinned to the existing K64 block map, root reduction enabled,
batch size one, concurrency one, and one resolved SWE-Verified task. It disables
all competing kernel candidates and autocommit. The postvalidator requires:

- observed accepted lengths `0..11`;
- candidate and reference captures for every accepted length;
- byte equality for every FP32 bank row across all 48 layers;
- reference output restored and served after each newly qualified depth;
- exact candidate source, binary, install, selector, and runtime-manifest
  identities; and
- a chat-traffic audit reconstructed from the raw task evidence rather than
  trusting self-reported audit flags.

This run is a correctness diagnostic only. It is not a timing run and cannot
make a hardware-floor or production-acceptance claim.

## Verification and remaining work

- Focused B1 boundary, runtime-arm, committer, selector, binary, and gate suite:
  `169 passed`.
- Ruff, Python byte compilation, runtime-manifest self-test, and
  `git diff --check`: pass.
- No Docker/container action, GPU query, CUDA launch, SWE task, synthetic/probe
  timing, or performance measurement was run for this integration.
- Remaining live step: build the pinned full `_C.abi3.so`, issue its private
  binding, and run the new real K64/root1 B1 all-depth gate.
- B1 timing and canonical B4 exact4 byte-gate/timing remain pending and must use
  the standing real SWE-Verified task-set rules.

This directory contains reduced readiness facts only. It excludes prompts,
responses, patches, raw task payloads, raw logs, environment dumps, process and
container identities, credentials, binaries, objects, and timing samples.
