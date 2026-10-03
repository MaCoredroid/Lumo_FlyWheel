# Root diagnostic request RNG repair — independent source review

Date: 2026-09-29. Scope: the failed root-only R2 diagnostic and a prospective request-parameter repair. No GPU, model, container, cache, remote action, implementation edit, or gate change was performed. This is not workload, timing, sampling-distribution, or full qualification evidence.

**Disposition: the proposed request `seed: 0` → `seed: null` change is scientifically appropriate for this bounded forced-continuation diagnostic.** Retain engine `--seed 0`, temperature 1, top_p 1, the production fixed32 sampler, all forced root/draft/acceptance products, and the raw O2 exact-greedy criterion. Record this as an explicit prospective diagnostic-request amendment, not as an unchanged request. There is no source-level reason to change the production RNG guard, supply synthetic uniforms, pin uniforms to 0.5, or reset its generator between cases.

## Source-grounded mechanism

`scripts/fr13_device_multidraft_kernel.py:4483–4531` rejects every nonempty per-request generator map before drawing. With no map and no supplied uniforms, line 4527 performs one `target.uniform_(generator=_fr13_bulk_gen(target.device))` call. Lines 8054–8066 construct that cached process-local generator once, using `torch.initial_seed() & 0x7FFFFFFF`, then reuse it. Engine seed zero remains zero after this mask. The optional uniform-pin path at 4528 and 8046 must remain disabled.

The actual native runner export, `identity/native_source/vllm__v1__worker__gpu_model_runner.py:1138–1145`, creates a request generator only for `SamplingType.RANDOM_SEED`. The generated candidate runner has the same branch at 1198–1205. The exported `gpu_input_batch.py:390–393` adds only non-None generators to the sampling map. `gpu_worker.py:274–275,690–692` initializes/restores the model seed. The fixed32 logit-direct route still requires sampled rather than all-greedy metadata (`fr13_device_multidraft_kernel.py:7429`), so temperature zero is not the minimal repair.

Four independent stdlib controls extracted the actual AST functions/branches, without importing Torch: nonempty-map rejection precedes any draw; None/empty maps each make one draw and reuse a single seed-zero generator; the native runner allocates only for RANDOM_SEED; the input-batch branch omits generator=None. These are source controls, not execution of CUDA sampling. Source hashes and results are in `q1-root-request-rng-repair-independent-audit.json` (SHA-256 `ee66b221a62a7ca26d1536eb5151b2ed862a8b442dcbef318372b96a502e1cdf`).

## Why the narrow diagnostic remains meaningful

Root hooks `q1_candidate_hooks_v1_2.py:358–399` preserve natural sampled products as diagnostics, then impose the prescribed root, all draft rows, and accepted products. Its O2 hook at 430–449 captures raw logits before sampling and compares smallest-ID greedy argmax with the pinned native observation. The native reference request remains greedy (`q1_reference_driver_v2.py:104`); it does not require identical stochastic trajectories. Changing the candidate's request-generator policy therefore need not change the prescribed token/state comparison or the O2 decision criterion.

It does change natural random draws and their stream offsets. Warmup and prior requests can advance the cached bulk generator, so engine seed zero alone does not prove per-request random repeatability or agreement with native sampling. Keep those claims excluded. The kernel comment's assertion of distribution equality is not independently established here. This repair does not fix or qualify the separately observed native-64/candidate-1024 KV hydration mismatch.

## Minimal integration obligations

1. Use a versioned job/request policy with `seed: null`, update the wrapper's seed-zero request assertion and hash bindings, and preserve the failed seed-zero artifacts. The current driver already serializes `rq["seed"]` directly (`q1_candidate_driver_v1_1.py:78,86`), so JSON null is the smallest payload change; omitting the field requires corresponding driver handling. Preserve all engine, token, path, temperature, top_p, capacity and numerical settings.
2. Confirm the actual installed OpenAI/SamplingParams path resolves null to sampled RANDOM with `seed is None`, then generator=None and an empty B1 generator map. Those parser sources were not in the inspected local exports, so the AST controls do not establish that mapping. A connected CPU parser check or the existing diagnostic's metadata attestation can establish it. Retain effective engine seed zero and the bulk-generator route; do not bypass the guard or rewrite the production sampler.
3. Keep forced-token/publication and raw O2 checks mandatory. The scope remains the same root-only R2 diagnostic, with no workload extension or new sampling-preservation assertion.

## Preserved failure and exact source identities

The local failed run `runs/q1-candidate-stage1/q1-candidate-stage1-20260929T080536Z` records request seed zero, HTTP 500, and the exact fixed32 generator-map exception (`engine.log:275–279`). Its case seal is invalid with no O2 and also records the earlier KV planner mismatch; it supplies no passing numerical observation. Engine log SHA-256: `da2cfc272e5a9066ea9926083c7c0afa97c5e180d531aef188373d21cae16cd4`. Case JSON file SHA-256: `cc28d95e65c01dfa7dd81dca6fc5eafd0edda941be472eda41c509120ce33ac2` (internal record seal differs by design).

Key SHA-256 pins:

- Production multidraft kernel: `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9`.
- Native runner export: `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0`.
- Generated candidate runner: `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79`.
- Input batch export: `2d6d6d4293ce70b1bc7a7fe47217727148959706c8ad6245e4ad060cd6a9a2bf`.
- GPU worker export: `3b2fc4e3a2610c67e7c7d2017d04b4f50a8560aec8b85a486a5451f9db97744a`.
- Root hooks v1.2: `af4595e4b03d8f5c44a5ae0bca5125677e00b827666d8062fec7b43fdb2ac516`.

All reviewed job/driver/wrapper/native-driver and failure-file hashes are retained in the companion audit. This recommendation authorizes no launch and does not review the forthcoming implementation successor.
