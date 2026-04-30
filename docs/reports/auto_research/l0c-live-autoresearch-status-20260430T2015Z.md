# L0c Live Auto-Research Status - 2026-04-30

## Scope

This report records what the current serving auto-research loop is building and what has been observed from the live L0c DeltaNet round. It is based on on-disk round artifacts only; the running tmux/Codex/vLLM processes were not interrupted.

## What We Are Building

The active work is the L0c kernel-mutation arm of serving auto-research for `qwen3.5-27b` on the `responses-sdk-adapter-cutover-heavy` workload.

The system is meant to run a Karpathy-style propose/test loop for serving kernels:

- A Codex agent proposes one source patch per attempt against the selected kernel source.
- The controller applies the patch to an isolated kernel workdir.
- A real vLLM harness restarts the serving stack, runs the DeltaNet parity fixture, and only measures candidates that pass parity.
- Passing candidates are measured against the same workload shape as the paired baseline.
- The round continues until an accepted-candidate cap, total-attempt cap, timeout, or blocker condition is reached.

For the current round, the mutable target is:

- Kernel target: `deltanet`
- Kernel source: `output/auto_research/l0c_kernel_workdir/chunk_delta_h.py`
- In-container kernel path: `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py`
- Parity fixture: `benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/deltanet_v1.yaml`
- Fixture id: `responses-sdk-adapter-cutover-deltanet-v1`
- Base bundle: `output/tuned_configs/responses-sdk-adapter-cutover-heavy/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/20260426T2339070000_4866bc3f.yaml`

The round is intentionally correctness-first. Logit/state parity is a hard gate; speed is irrelevant for any patch that fails parity.

## Live Round

Active tmux session:

```text
l0c_deltanet_long_20260430T183323Z
```

Active round directory:

```text
output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-deltanet-20260430T183324Z
```

Configured caps:

- `accepted_iteration_cap: 24`
- `total_attempt_cap: 72`
- `round_timeout_hours: 48.0`
- `base_measurements: 5`
- `agent_runtime: codex`
- `per_iteration_codex_wall_clock_s: 7200`

## Observations So Far

Baseline measurement completed all five baseline measurement files:

- `baselines/measurement_01.json`
- `baselines/measurement_02.json`
- `baselines/measurement_03.json`
- `baselines/measurement_04.json`
- `baselines/measurement_05.json`

Candidate `001`:

- Wrote `mutation.patch`.
- Passed parity at checkpoints `[1, 1024]`.
- `parity_check.json` has `pass: true`, `reason: ran_passed`.
- Canonical measurement trace exists.
- Objective mean in the latest artifact read: `0.0449845`.
- This is below the recent paired-baseline level seen in the prior round (`0.056204`), so it does not currently look like a speed winner.

Candidate `002`:

- Wrote `mutation.patch`.
- Failed parity.
- Rejection row was written to `mutations_rejected.tsv`.
- Rejection reason: `parity_state_diverged`.
- First diverging probe: `0`.
- The parity error was state-digest mismatch, not logit overshoot.

Candidate `003`:

- Wrote `mutation.patch`.
- Latest artifact read shows `parity_check.json` with `pass: true`, `reason: ran_passed`.
- No measurement trace was present in the artifact read used for this report, so it should be treated as in-progress at report time.

## Behavioral Notes

The repeated `lumo-vllm-l0c-live` messages and temporary `127.0.0.1:8100` connection failures are expected during the current harness design. The controller restarts vLLM between activation/parity/measurement phases to isolate kernel activation, Triton/cache state, debug export state, prefix-cache state, and measurement metrics.

This restart model is expensive but currently defensible for correctness. A future warm-server optimization should only be used after we can prove in-process reset semantics for:

- kernel bundle activation,
- Triton/autotune state,
- prefix cache,
- debug export directories,
- metrics windows,
- and request/proxy state.

## Current Assessment

The auto-research loop is functioning mechanically:

- It bootstrapped a high-cap real L0c round.
- It measured the baseline.
- It spawned Codex-backed candidate attempts.
- It accepted parity-clean candidates into measurement.
- It rejected parity-breaking candidates with structured reasons.
- It continued after rejection without manual intervention.

The search quality is still the open problem. The observed safe candidates continue to be small DeltaNet metadata/cache/load-shape mutations, and the measured objective so far has not exceeded the paired baseline. That suggests the controller and parity gate are doing their job, while the proposer needs stronger guidance toward changes with real memory-traffic or launch/path impact rather than changes that are merely parity-safe.

## Manual Monitoring Commands

Attach to the live session:

```bash
tmux attach -t l0c_deltanet_long_20260430T183323Z
```

Inspect the latest artifacts without attaching:

```bash
ROUND=output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-deltanet-20260430T183324Z
find "$ROUND" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -200
```

Check ledgers:

```bash
ROUND=output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-deltanet-20260430T183324Z
for f in "$ROUND"/measurements.tsv "$ROUND"/results.tsv "$ROUND"/mutations_rejected.tsv "$ROUND"/run_log.json; do
  [ -f "$f" ] && printf '\n## %s\n' "$f" && cat "$f"
done
```

