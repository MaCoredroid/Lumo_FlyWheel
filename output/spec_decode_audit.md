# Speculative Decode Audit

Generated: 2026-05-06T01:35:00Z

## Running Runtime

- container: `lumo-vllm-l0c-fp8-cutlass-run30`
- served model: `qwen3.5-27b`
- vLLM version: `0.19.0`
- API port: `9950`

## Result

- `vllm.spec_decode` import: failed (`ModuleNotFoundError`)
- `vllm serve --help` grep for `speculative|ngram|prompt-lookup|draft|eagle|mtp`: no matching launch flags
- current Round 0 measured warm output throughput: `6.955 tok/s`
- Track B target: `15.000 tok/s`

## Decision

Round 1 speculative decoding cannot be launched on the currently running vLLM container. Prefix caching is active and useful for prefill, but the decode-side 2x target requires either:

1. A vLLM build with speculative decoding / ngram prompt lookup / MTP launch support for this Qwen hybrid-attention model.
2. A repo-owned PLD proposer/verifier integration added outside the current container's exposed CLI surface.

Until one of those is present, the completion audit must remain incomplete.
