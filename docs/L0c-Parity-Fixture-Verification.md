# L0c parity-fixture verification — operator runbook

How to convince yourself a (re-)built parity fixture is sound *before* you spend agent budget on an L0c round against it. Companion to `HLD-Serving-Backend-AutoResearch-v0_2-L0KernelPlan.md` §2.2 / §6 — the HLD specifies the contract; this doc is the runbook.

## What the fixture is for

The fixture is the **correctness substrate** for L0c kernel mutation. It pins:

- `probes_input.jsonl` — the deterministic probe prompts (16 for siblings, 64 for the heavy family per HLD §2.4)
- `<kernel>_reference_logits.npz` — first-token logits from running those probes against the unmutated kernel
- `<kernel>_reference_state.npz` — recurrent-state snapshots at token 1 and token 1024 (DeltaNet only)
- `<kernel>_v1.yaml` — schema + `tolerances` + `generated_against.reference_baseline` (the runtime config that produced the reference logits) + `weight_version_id`

The L0c parity probe re-runs the same probes after each mutation and compares element-wise against the .npz references. Any element outside `rtol/atol` → mutation rejected as `parity_logit_diverged` / `parity_state_diverged`.

The fixture is **not** about latency. Performance comes from `RealMeasurementHarness.measure()` during the per-iteration n=2 paired step.

## When you'd run a verification

After:

- Running `scripts/regenerate_deltanet_parity_fixture.py` (rebuilt against a new L0b winner)
- Rotating model weights (`weight_version_id` changes)
- Switching kernel containers / docker images
- Bumping vLLM, Triton, FA, or FlashInfer versions
- Editing the bind-mounted kernel source (the file vLLM `import`s)

If any of these change between fixture build and L0c round, the L0c parity gate becomes meaningless — every mutation will fail or pass for reasons unrelated to the patch. The verifier catches this in 30 minutes instead of letting a 12-hour agent round chase ghosts.

## Three verifications, in increasing cost

### 1. Schema + content-hash check (~2s, no GPU)

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_parity_fixture.py tests/test_build_parity_fixture.py -q
```

Confirms:

- Fixture YAML parses + has all required fields per HLD §6.6.6
- `weight_version_id` matches `DEFAULT_WEIGHT_VERSION_ID`
- `reference_baseline` matches `parity_fixture.REFERENCE_BASELINE` (the global default, **or** the override threaded through `expected_reference_baseline=` if the fixture was built against a non-default L0b winner)
- All referenced blobs exist (`probes_input.jsonl`, `*_reference_logits.npz`, `*_reference_state.npz`)
- `fixture_content_hash` (yaml + every referenced blob, sorted-key delimiter-framed per §6.6.6) recomputes byte-equal

Failures here are configuration mistakes — wrong probe count, missing npz, mismatched weight version. Fix the fixture YAML or rerun the builder; do not proceed.

### 2. Pre-flight check against the L0c base bundle (~5s, no GPU)

The L0c runner does this automatically (commit `51afda7`, `_assert_fixture_matches_base`): it reads `fixture.generated_against.reference_baseline` and demands every key in it equals the corresponding key in `base.kernel_selection`. The round refuses with a precise diff if any field differs (e.g. `attention_backend: fixture=flash-attn-4 base=vllm-default`).

You can dry-run this without spawning vLLM by importing the runner and calling `_assert_fixture_matches_base(fixture_yaml, base_bundle, fixture_path)`. Or simply rely on the L0c round to refuse fast if there's a mismatch.

### 3. No-mutation parity probe (~30 min, full GPU)

The decisive test. Bring up vLLM with the bind-mounted kernel **unmodified** + the L0b-winner bundle activated, then run the parity probe against the fixture. Expect `pass: true` with `tolerance_overshoot ≈ 0`.

```bash
PYTHONPATH=src python3 scripts/verify_deltanet_parity_fixture.py \
  --base-bundle output/tuned_configs/responses-sdk-adapter-cutover-heavy/<sha>/<bundle>.yaml \
  --kernel-target deltanet
```

The script (a) bind-mounts `output/auto_research/l0c_kernel_workdir/chunk_delta_h.py` over the container's pristine kernel, (b) sets `LUMO_P2B_*` env vars and bind-mounts the debug-export staging dir, (c) `server.load_tuned_config(base_bundle)` then `server.start()`, (d) calls `run_parity_probe()` directly, (e) tears down.

Interpreting the output:

| `pass` | `reason` | What it means |
|---|---|---|
| `true` | `ran_passed` | Fixture aligned with this runtime. L0c against this fixture is meaningful. |
| `false` | `parity_logit_diverged` | Live logits don't match `<kernel>_reference_logits.npz` within `rtol_logit/atol_logit`. Either the runtime drifted from fixture-capture time (kernel bytes, weight version, attention backend, prefix caching) or the debug-export path is non-deterministic. Look at `tolerance_overshoot` — small (<2x) likely run-to-run noise; large (>10x) signals a structural mismatch like the flash-attn-4 vs vllm-default issue from canary v2. |
| `false` | `parity_state_diverged` | Same as logits but for DeltaNet recurrent state at tok 1 or tok 1024. State diverges before logits when a kernel-internal computation drifts but the output projection masks it. |
| `false` | `capture_failed` | vLLM didn't emit the expected `.pt` exports. Check that `LUMO_P2B_VLLM_DEBUG_EXPORT=1`, `LUMO_P2B_DEBUG_PROBE_REQUEST_IDS="*"`, `LUMO_P2B_DEBUG_EXPORT_DIR=<path>` are set and the staging dir is bind-mounted host↔container at the same path. |
| `false` | `endpoint_unreachable` | The probe's `/v1/completions` POSTs got a non-2xx. The lumo inference proxy only whitelists `/v1/responses` and `/v1/chat/completions` — make sure you're hitting the engine port directly (`runtime.port`, not `proxy_port`). |
| `false` | `comparison_failed` | The .npz reference itself failed to load — likely fixture corruption. Re-run `pytest tests/test_parity_fixture.py`. |

### 4. P5b fp8_e5m2 KV purity attestation (~10 min, full GPU)

Required only when the base stack's `kv_cache_dtype` is `fp8_e5m2` (i.e., the L0b empirical winner). HLD v0.3.3 §7.X. Skipped automatically by the script when the base bundle is already bf16/fp16.

```bash
PYTHONPATH=src python3 scripts/p5b_fp8_kv_purity_attestation.py \
  --base-bundle output/tuned_configs/.../<bundle>_4866bc3f.yaml
```

The script (a) brings up vLLM with the fp8_e5m2 base bundle and captures 16 short-prompt-short-output probes, (b) synthesizes a sibling bundle with `kv_cache_dtype: bf16` (everything else identical) and captures the same probes, (c) compares element-wise logit divergence against the fixture tolerances `rtol_logit=1e-3, atol_logit=1e-3`. Output: `output/p5b_fp8_kv_purity_<timestamp>.json`.

| Outcome | Meaning |
|---|---|
| `status: PASS` | fp8 KV introduces no divergence beyond the parity gate's noise floor; L0c rounds bootstrapped against this base will be measuring kernel mutations, not quantization noise. |
| `status: FAIL`, `halt_code: fp8_kv_purity_violation` | fp8 KV introduces divergence beyond `rtol/atol` on >0 of 16 probes. Fix paths: re-capture parity fixture against bf16 KV, OR loosen tolerance with explicit justification, OR change base stack's `kv_cache_dtype`. |
| `status: skipped`, `reason: base_kv_already_safe` | Base bundle is bf16/fp16 — no FP8 noise risk to validate, no work to do. |

The 16 probes are deterministic (same row generator as the parity fixture), so two consecutive runs against the same base bundle byte-equal each other modulo accumulation noise within `rtol/atol`.

## Common ways a fixture goes stale

1. **L0b converges to a new kernel_selection.** Caught by the pre-flight check (verification 2). Fix: regenerate fixture against the new winner via `scripts/regenerate_deltanet_parity_fixture.py --reference-baseline-bundle <new bundle>`.
2. **Model weight rotation.** `weight_version_id` in fixture metadata stops matching `model_registry.yaml`. Caught by verification 1. Fix: regenerate.
3. **Kernel source edit.** Someone touched `output/auto_research/l0c_kernel_workdir/chunk_delta_h.py` between fixture build and L0c round; not caught by 1 or 2 (the bytes aren't bound to the fixture). This can make every mutation fail with the same first-probe logit overshoot, including a no-mutation/base smoke. Fix: restore the kernel bytes used at fixture capture or regenerate the fixture against the new base kernel. Snapshotting kernel sha into fixture YAML remains an open follow-up.
4. **vLLM image rebuild.** New container brings new pristine kernel bytes; if the bind-mount points at a now-unrelated host file, vLLM imports the wrong kernel. Caught by verification 3 (large tolerance_overshoot). Fix: re-snapshot the bind-mount source from the new image and rebuild fixture.
5. **Triton autotune cache drift.** The DeltaNet fixture was captured with `/tmp/lumo-fixture-rebuild-triton`; L0c must use that same cache root unless a no-mutation verifier has passed with another root. A different cache can produce first-token logit drift before any mutation is applied.

## What "pass" *cannot* tell you

- **Not** that all mutations will be evaluated correctly. A mutation could change a code path the 16/64 probes don't exercise. Probe coverage is finite by design — HLD §6.4 acknowledges this.
- **Not** that the runtime is fast. Parity is a correctness predicate, not a performance one.
- **Not** that the mutation is semantically equivalent to the original. Two kernels can produce the same logits on these probe inputs and disagree on others. The fixture is a sample-based correctness gate, not a proof.

## What "fail" *probably* tells you

- The most common cause of fail is **config drift between fixture-capture and round-run** — same code path the canary v2 hit (flash-attn-4 fixture × vllm-default base bundle gave overshoot 28.6 on every mutation, including no-op ones). Always check verification 2 first.
- The next most common cause is **non-determinism in the debug-export path** — prefix caching not flushed between probe captures, fp8 GEMM accumulation order varying, etc. The fixture builder runs `--reproducibility-runs 3` and asserts pairwise tolerance to surface this; if reproducibility passes at fixture build but verification 3 fails, the divergence is probably between *fixture-capture-side* and *L0c-probe-side* code paths (different request shapes, different KV-cache state, etc.) — investigate `parity_probe.py` vs `build_parity_fixture.py` request construction.
