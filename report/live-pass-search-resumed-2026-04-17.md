# Live Pass Search Report - 2026-04-17

## Scope

Continued the live pass search on `main` using the required real-task path:

```bash
cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family <family> --variant <variant> --json
```

Goal was to find the first genuine `pass=true` live run with clean LLD-04 telemetry (`anomalies=[]`, `telemetry_summary.n_tasks=1`).

## Live Stack Bring-Up

The repo-local serving path required host-memory recovery before launch. Without it, `torch.cuda.mem_get_info()` reported about `10.6 / 117.5 GiB` free and `ModelServer.start()` stayed in `_wait_vram_free()`.

Bring-up sequence that worked:

```bash
cd /home/mark/shared/lumoFlyWheel
set -a; source ./.lumo.local.env; set +a
printf '%s\n' "$LUMO_SUDO_PASSWORD" | sudo -S bash -lc 'sync; echo 3 > /proc/sys/vm/drop_caches; swapoff -a || true; swapon -a || true'
PYTHONPATH=src ./.venv/bin/python -m lumo_flywheel_serving.cli serve qwen3.5-27b
```

Observed healthy live stack after warm-up:

- upstream vLLM health on `http://127.0.0.1:8000/health`
- inference proxy on `http://127.0.0.1:8001/v1`
- direct upstream `POST /v1/responses` and proxy `POST /v1/responses` both succeeded
- proxy intentionally rejects non-inference paths such as `GET /v1/models` with `403 Blocked by codex-bench-proxy: inference paths only`

## Attempts

1. `normalizer-api-migration/alert-routing`
   - Result: countable failure
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family normalizer-api-migration --variant alert-routing --json
     ```
   - Telemetry: clean (`anomalies=[]`, `telemetry_summary.n_tasks=1`)
   - One-task elapsed: `400.0410509277135s`
   - Grading failure: `alert-routing follow-up lifecycle normalization slice did not pass`
   - Result artifact: `output/live_codex_long_task/normalizer-api-migration/alert-routing/20260417T232546Z/result.json`

2. `ci-config-coverage-drift/inventory-gate`
   - Result: countable failure
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family ci-config-coverage-drift --variant inventory-gate --json
     ```
   - Telemetry: clean (`anomalies=[]`, `telemetry_summary.n_tasks=1`)
   - One-task elapsed: `207.02092340867966s`
   - Grading failures:
     - `inventory-gate hidden workflow-preview artifact slice did not pass`
     - `inventory-gate punctuation-heavy preview artifact slice did not pass`
   - Result artifact: `output/live_codex_long_task/ci-config-coverage-drift/inventory-gate/20260417T233249Z/result.json`

3. `alert-dedupe-investigation/payments-oncall`
   - Result: infra failure, not countable
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family alert-dedupe-investigation --variant payments-oncall --json
     ```
   - Telemetry capture still wrote a clean single-task row before failure (`anomalies=[]`)
   - Failure: `Codex transport to localvllm failed before completion`
   - Codex stderr evidence:
     - `failed to parse function arguments: trailing characters at line 1 column 32`
   - Result artifact: `output/live_codex_long_task/alert-dedupe-investigation/payments-oncall/20260417T233654Z/result.json`

4. `report-cli-markdown-evolution/release-readiness`
   - Result: countable failure
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant release-readiness --json
     ```
   - Telemetry: clean (`anomalies=[]`, `telemetry_summary.n_tasks=1`)
   - One-task elapsed: `162.61323301773518s`
   - Grading failures:
     - `release-readiness hidden renderer slice did not pass`
     - `release-readiness follow-up/docs slice did not pass`
   - Result artifact: `output/live_codex_long_task/report-cli-markdown-evolution/release-readiness/20260417T233726Z/result.json`

5. `owner-field-cross-layer/release-gate`
   - Result: countable failure
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family owner-field-cross-layer --variant release-gate --json
     ```
   - Telemetry: clean (`anomalies=[]`, `telemetry_summary.n_tasks=1`)
   - One-task elapsed: `211.63371683750302s`
   - Grading failures:
     - `release-gate hidden owner persistence slice did not pass`
     - `release-gate separator-heavy release routing follow-up slice did not pass`
   - Result artifact: `output/live_codex_long_task/owner-field-cross-layer/release-gate/20260417T234036Z/result.json`

6. `alert-dedupe-investigation/search-oncall`
   - Result: infra failure, not countable
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family alert-dedupe-investigation --variant search-oncall --json
     ```
   - Telemetry capture still wrote a clean single-task row before failure (`anomalies=[]`)
   - Failure: `Codex transport to localvllm failed before completion`
   - Codex stderr evidence:
     - `failed to parse function arguments: trailing characters at line 1 column 32`
   - Result artifact: `output/live_codex_long_task/alert-dedupe-investigation/search-oncall/20260417T234424Z/result.json`

7. `normalizer-api-migration/catalog-sync`
   - Result: countable failure
   - Live command:
     ```bash
     cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --family normalizer-api-migration --variant catalog-sync --json
     ```
   - Telemetry: clean (`anomalies=[]`, `telemetry_summary.n_tasks=1`)
   - One-task elapsed: `366.1707670474425s`
   - Grading failures:
     - `catalog-sync hidden migration slice did not pass`
     - `catalog-sync follow-up source-label normalization slice did not pass`
   - Result artifact: `output/live_codex_long_task/normalizer-api-migration/catalog-sync/20260417T234452Z/result.json`

## Outcome

No genuine passing family/variant was found in this resumed live search.

- First genuine pass: none
- Exact successful live command: none
- First successful one-task elapsed: not applicable
- First successful clean LLD-04 telemetry capture: not applicable

## Remaining Gaps

- The serving path is workable only after the host-memory recovery step; without it, the initial minimum-VRAM gate blocks launch on this host.
- Five countable live runs all produced clean single-task telemetry and all failed on richer hidden follow-up slices rather than visible-test milestones.
- The `alert-dedupe-investigation` family hit the same Codex/local-vLLM transport/parser failure twice:
  - `failed to parse function arguments: trailing characters at line 1 column 32`
- No repo bug specific to `lumoFlyWheel` was identified during these attempts, so no code change was made.
