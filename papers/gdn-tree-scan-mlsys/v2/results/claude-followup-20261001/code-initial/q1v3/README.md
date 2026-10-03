# q1v3 — LumoTree full-model multi-cycle correctness check

Decision rules: `PROTOCOL.md` (frozen). This file: how to run it.

## Files

| file | role |
|---|---|
| `PROTOCOL.md` | claim, arms, cases, surfaces, frozen tolerances and verdict rules |
| `cases.py`, `cases.v1.json`, `fixtures/requests/*.json` | predeclared case set (33 multi-cycle cases, 147 cycles, 10 negative controls) and the frozen chat bodies |
| `q1v3_hooks.py`, `q1v3_common.py` | in-engine hooks (forcing, O0 capture/import, state + logits capture, negative-control mutations); paged-state helpers |
| `patch_runner.py` | inserts the hook calls (native: 4 anchors in the stock runner; candidate: 6 + 2 anchors in the production-patched runner / rejection sampler) |
| `serve_native_q1v3.sh` | native spec-off engine (mirrors `serve_native.sh`) + KV pin + hooks |
| `make_cand_launch.py` | generates the q1v3 copy of the deployed serve-only chain (5 exact-once edits; sources sha-pinned) |
| `client.py` | per-arm observation driver (control file, chat request, wait for seal) |
| `run_all.sh` | **single end-to-end driver**: A → CAND → B → V → P → `reduce.py` |
| `reduce.py` | authenticates records, computes distances, writes `VERDICT.json` |
| `freeze.py`, `FREEZE.json` | sha256 freeze of everything above; `run_all.sh` refuses on any change |
| `tests/` | CPU tests incl. an engine stand-in that runs hooks + reducer end-to-end |

## Before the run (CPU only)

```bash
cd /home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/q1v3
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --basetemp=$PWD/.pytest-tmp tests; rm -rf .pytest-tmp
python3 freeze.py --write        # only after review; commit FREEZE.json together with the sources
Q1V3_RUNS=$PWD/.dryrun bash run_all.sh --dry-run && rm -rf .dryrun   # freeze + cases + launch-chain generation + plans, no GPU
```

## Run (after the SWE study has finished)

```bash
cd /home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp/q1v3
nohup setsid bash run_all.sh --wait-idle > /home/mark/shared/lumotree-v2exp-runs/q1v3-launch.out 2>&1 &
```

`--wait-idle` polls until no container and no v2exp queue / pipeline (including `v2exp_pipeline_after.sh`) / run_*arm script is running, then starts; it re-checks before every stage.
Without it the driver refuses when the GPU is busy. Output: `/home/mark/shared/lumotree-v2exp-runs/q1v3/<run-id>/`
(`driver.log`, `o0/`, `A|CAND|B|V|P/` each with `cases/`, `objects/`, `responses/`, `boot.*.json`,
`patch_receipt.json`, `engine.log`; `STAGE-*.json`; `VERDICT.json`). Files written by the containers are owned by root.

Resume / partial: `bash run_all.sh --run-id <run-id> --stages CAND,B,V,P` (completed stages are skipped; move a
failed stage's directory aside first). Re-reduce only:
`CUDA_VISIBLE_DEVICES= python3 reduce.py --run <RUN> --out <RUN>/VERDICT.json`. Skip the optional arm: `--stages A,CAND,B,V`.

## Expected cost

| item | estimate |
|---|---|
| boots | 4 native × ~5.5 min + candidate vehicle ~6 min (READY in 330–345 s in recent v2exp runs) |
| native arm | 33 observations: 7 cold prefills (~1 350 tok/s, ~2.7 min total) + ~10–15 s per case (O0 import, 25–75 decode steps, 4–7 state captures) ≈ 10–12 min; A adds ~2 min of O0 capture |
| candidate arm | 44 observations (33 + 1 repeat + 10 controls) ≈ 12–15 min |
| total GPU wall | ≈ 1.6 h with P (≈ 1.3 h without); budget ceiling 3 h |
| GPU/host memory | native: weights ≈ 25 GB + pinned 40 GiB KV (V: ≈ 52.7 GiB) + transients < 1 GB; candidate: deployed GPU_UTIL 0.70 (52.66 GiB KV in Codex's stage-1 boot). Each stage requires MemFree ≥ 85 GiB after `recover_host_memory` (natives) or the vehicle's own floor (candidate). Hooks hold one prefix's O0 on the host (≤ 4.1 GB for the 60k prefix). |
| disk | ≈ 0.7 GB per observation (FP32 SSM snapshots dominate) → ≈ 130 GB per run + ≈ 15 GB O0 store; the driver requires ≥ 300 GiB free |
| reducer (CPU) | ≈ 15–30 min (re-hashes and compares every object) |

## Reused vs new

Reused (ideas and verified anchors, re-implemented without the freeze/seal bureaucracy): Codex's hook anchor
strings (stock runner A1/A2/A3/A5; generated runner S0/S1/S2/S4/S5; rejection-sampler S3, all re-verified here
against the captured sources), the TAW 5-tuple forcing (`q1_forced_acceptance.py`), the native GDN running-row
derivation (`non_spec_state_indices_tensor`), the candidate running row (`spec_state_indices_tensor[:,0]`), the
O0-import idea (common O0 for every arm), Codex's six prefixes and cycle-0 tree assignments, the native serving
flags (`serve_native.sh`) and the deployed vehicle + signed-request path (`run_tree_arm.sh`, `fixed32_auth.py`).
New: multi-cycle forcing with no re-hydration, per-cycle state/KV/logit capture, KV pin + capacity-matched native
arm, packed-decode variant arm, five negative controls, single driver, calibrated reducer, CPU engine stand-in.

## Known limitations / what was not verified without a GPU

See PROTOCOL.md §8. Not verifiable on CPU: that the anchors bind identically in the *live* container (checked at
boot; the patcher refuses otherwise), the in-engine tensor attribute names on real metadata objects, the chat
rendering parity across arms (checked at run time, R2), timings and memory, and that the negative-control wrappers
do not trip a production device assertion (they run last and are designed to stay inside the device contracts).
