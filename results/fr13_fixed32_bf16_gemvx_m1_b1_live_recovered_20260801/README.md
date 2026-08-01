# Fixed32 full-vocabulary M1 real-B1 recovered qualification

Status: `PASS_RECOVERED_AFTER_TEARDOWN_ONLY_FAILURE`

This package records a real SWE-Verified B1 shadow qualification for the
full-vocabulary BF16 M1 draft-head kernel. The task
`astropy__astropy-12907` resolved, the terminal fixed32 census closed at 960
events, and all 1,191,936,000 compared BF16 elements matched the stock result.
Stock logits were returned throughout the run.

The original outer launcher exited with `serve_rc=16` after the task, final
flush, boundary snapshot, and kernel result were complete. The failure was in
traffic-audit teardown:

```text
TypeError: build_fixed32_chat_traffic_audit() got an unexpected keyword argument 'concurrency'
```

The recovery did not rerun or modify the measured kernel workload. It rebuilt
the chat-traffic audit from the immutable task boundary, authenticated ingress
ledgers, work census, trace, and evaluation artifacts. The validator was made
strictly compatible with both the legacy and current audit schemas. It then
validated the live result and issued and verified the production sidecar.

## Bound identities

- Measured source commit: `cabab34ca6286395bda87210ec1e55875f1ef02b`
- Recovery implementation commit: `ddefac9c07b0f38a87f0ac383821bbcd0479b8f7`
- Candidate CUDA source SHA-256:
  `26ea8aad9f891b5e758a39464209d6f82008a10fac8da4c02ee052e839218a54`
- Candidate SO SHA-256:
  `7d6c549e741d8fbbc54732ba5873a8c01f7f089f15a8589ef51eb49a45f5e6d5`
- Live result SHA-256:
  `e193dabd1a07c6866ecb6f483562973c9f3dee22a043f1662f2fd51c5c01626e`
- Rebuilt traffic audit SHA-256:
  `5a745f57c19b7d2d649258c4b7827f54781dbc7d88ecc042695efb7a3617d2b5`
- Final flush SHA-256:
  `ec75d5cb553a2623009a78602cde0c4627db1a9302224f758ffba8ee529670d0`
- Final boundary SHA-256:
  `8e596cd5f925fc5ba38db039d3ff9652fa320fe6f20c6cec38feaf4172c403d0`
- Production sidecar SHA-256:
  `fc8b186bde7f9f15acee484b188fb9b99c120d3ee47cff4af4d79e4edf364850`

## Scope

This is a one-task kernel byte qualification and a production credential. It
is not a performance measurement, floor-acceptance result, B4 result, or clean
outer-wrapper run. Full-wall timing remains subject to the standing exact4 and
exact16 real SWE-Verified rules.
