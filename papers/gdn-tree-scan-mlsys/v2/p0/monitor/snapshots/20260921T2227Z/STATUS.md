# v2 bounded numerical experiments — execution status

Worktree: `/home/mark/lumo-paper-v2-20260921`, branch `codex/paper-v2-pilot-20260921`, base `984f613d7885d9bf7898866827552b68a3199d0c`.
Session: Claude Fable 5.1 interactive, tmux `lumo-v2-20260921` (created 2026-09-21T21:45:29Z per `p0/handoff-status.json`), handoff `p0/HANDOFF-CLAUDE.md` (sha256 `a06f3a35…1fa7b2`).
Host clock is UTC (`date +%Z` = UTC). Times below are MEASURED (file mtimes, `date -u`, telemetry stamps) or marked *observed-by*.

## Authorization scope (from handoff)

Bounded scope only: E7a -> E2/E7b -> E1, E3 conditional. No external messages, uploads, publication, push, new quantization campaign, or broad task-quality campaign. Do not touch `/home/mark/shared/lumoFlyWheel` checkout state, its captures, tmux sessions, unrelated processes, or shared model weights. Historical captures are read-only inputs and remain historical. Synthetic inputs are labeled synthetic. The old pre-fix WY state-write bug is NOT the principal baseline (the vendored WY kernel is the post-fix form at commit `8a975837`).

## Stage log (measured chronology)

| UTC (measured) | Stage | State | Evidence |
| --- | --- | --- | --- |
| 21:45:29Z | Handoff session created | — | `p0/handoff-status.json` |
| observed-by 21:47:36Z | Handoff acknowledged; STATUS.md first written; resource check | done | `uptime` output stamped 21:47:36 in session transcript |
| 21:54:47Z | E7a step 1 inventory conclusion written | done | `STATUS.md` mtime (previous version; the "~22:20" stamp it carried was a GUESS and is retracted) |
| 21:57:41Z | Corrected WY kernel vendored from `8a975837` | done | `e7a/legacy_wy_8a975837.py` mtime; sha256 `e7c2a2a8…58c1` (with provenance header; raw git object `7716f0d5…7cda`) |
| 21:58:03Z | Historical payload manifest written | done | `e7a/historical_payload_manifest.json` `generated_utc` |
| observed-by 22:03:45Z | Core algebra self-test on layer-62 payload (CPU fp64/fp32) | PASS (identities) | transcript; rerun reproducible via `e7a/e7a_selftest_core.py` |
| 22:03:15Z | New Triton kernels (B-fs, C-nm, compact commit) written | compiled-import OK in image | `e7a/e7a_kernels.py` mtime; container import test |
| 22:06:00Z | Fresh GPU contention telemetry | CONTENDED | `e7a/telemetry/gpu_contention_20260921T220600Z.txt` |
| 22:06:32Z | ULP metric restricted to significant elements | done | `e7a/e7a_core.py` mtime |
| 22:08:53Z | Device harness written; Codex review #1 findings addressed | done | `e7a/e7a_device.py`; `p0/monitor/2026-09-21-claude-reply-01.md` |
| 22:13:03Z–22:15:29Z | Smoke runs 1–2 (layout slip, manifest crash) | superseded | `out-20260921T221303Z-e7a-smoke/`, `out-20260921T221529Z-e7a-smoke/` (no result JSON / crashed manifest; kept as record) |
| 22:16:54Z | Codex review #2 (six harness defects) | received | `p0/monitor/2026-09-21-review-02.md` |
| 22:20:45Z | Review-02 fixes + CPU regression tests | ALL PASS (22 checks) | `e7a/tests_out/review02_regression_20260921T222045Z.json` |
| 22:20:51Z–22:22:49Z | Smoke run 3 with fixed harness (layer-62, iters=50) | completed; **metrics NOT promoted** | `out-20260921T222051Z-e7a-smoke/` (result JSON, completed manifest, telemetry) |
| 22:2xZ | Review-02 reply written | done | `p0/monitor/2026-09-21-claude-reply-02.md` |

## Current state

**Stage:** E7a step 3 — extended ladder with the fixed harness (historical layers 62/1/12/0 + synthetic chain depth 1/5/11, binary depth 3, tiny-gate regime, B4 arm), timing at 200 iters under the shared-device label.
**Model inference performed so far in this worktree:** none (kernel-level work only).
**Raw artifact directories:** `papers/gdn-tree-scan-mlsys/v2/experiments/out-<utc>-e7a-*/` (per run; `telemetry/` has 1 Hz utilization + 1 Hz compute-process inventory for the whole window).

## Codex review #2 (22:16:54Z) — resolutions

All six findings fixed; reply with evidence: `p0/monitor/2026-09-21-claude-reply-02.md`. Summary: node-major factor slicing + `compare()` shape assertion; manifest sources/flags split with start+final+crash manifests; exponent-input masking (core + both Triton kernels) removes off-path inf·0 NaNs, non-finite entries counted/excluded; monotone signed ULP ordinal with ±0 policy; immutable replay h0 source row (column 15) with repeat-bitwise validation (True on all 10 prefixes); pipefail-safe telemetry plus 1 Hz compute-process inventory for attribution. Smoke-3 observations are listed in the reply as unpromoted.

Resolved harness item from smoke-3: the vendored June-8 (`8a975837`) **sequential** scan kernel's per-node state export disagrees with the oracle (max_abs 0.163) while its outputs match. Cause (source-verified in `e7a/legacy_wy_8a975837.py` lines ~342-346): for every non-root node the export track restarts from `b_h0` and loops `tl.range(1, i + 1)`, skipping the root node's rank-1 update; the output path (`state_i`) does not skip it. This is a defect of that intermediate historical revision (the June-10 serving kernel's exported `serving_state` matches the oracle at 4.6e-7, so it was fixed between June 8 and June 10). `A_legacy_state*` rows are excluded from all tables; `A_legacy_out*` rows stay (bit-identical to the production scan). The WY kernel in the same file is unaffected (state at fp32 floor).

## Codex review #1 (22:01:38Z) — resolutions

Reply with evidence: `p0/monitor/2026-09-21-claude-reply-01.md` (review file preserved unmodified).

1. **Timing contention.** Refreshed telemetry at 22:06:00Z: unrelated `VLLM::EngineCore` is now PID 2736995 (34,135 MiB; the server was restarted twice since 21:47), GPU util samples over 5 s = 0/72/0/0/56 % → the device IS intermittently contended by the other server's work. Rule adopted: every E7a timing table is labeled **"shared-device diagnostic"**; `run_in_image.sh` records `telemetry/gpu_util_1hz.csv` + compute-app snapshots before/after each run, and the harness brackets each timing block with a fixed matmul probe. No comparative timing claim will be made without an uncontended window (contention record showing ~0 % baseline util for the whole window). The unrelated server is never stopped.
2. **Chronology.** Guessed "~22:00/~22:20" stamps retracted; table above uses measured mtimes/telemetry stamps or *observed-by*.
3. **Independent-prefix accounting.** Tensor-level check of the 9 unique historical payloads (see reply file): all 9 differ jointly in (q,k,v,a,b,h0); the six layer-0 files share identical layer parameters (A_log, dt_bias) as expected; two fr10-era layer-0 files share identical q/k/v but different a/b/h0 (lineage murky, debugging-era captures); l1/l12 are close (max|Δq| 0.87) and may be the same forward pass. Token ids / positions are NOT stored, so **the number of independent (prompt, position) prefixes is unrecoverable** → historical payloads are counted as **≤ 9 dependent historical operand sets from ≤ 3 known prompt sources, ONE topology (10-node caterpillar, B1)**, usable for algebra/diagnostics only. They contribute **0** to the 8-prefix pilot and **0** to the 32-prefix confirmation; those require fresh captures under the pinned configuration. Reducers key results by `provenance.sha256` and carry `synthetic`/`historical` labels; layer count is never reported as prefix count.

## Blockers / resource notes

- Unrelated host-level vLLM server (`Qwen/Qwen3.8-27B-FP8`, `--gpu-memory-utilization 0.50`, from `/home/mark/shared/exp54928`) resident and ACTIVE on the GB10. Not touched. Full-model E2/E7b/E1 cannot use the old 0.6-utilization config while it is resident; E7a kernel work fits in the remaining device memory (40.5 GiB free seen from the image at 21:5x).
- Historical GDN payloads cover ONE topology; depth 1/11, chain-vs-branch, B4 arms use SYNTHETIC operands (labeled) for the algebra pilot; fresh captures are required for confirmation.

## Results so far (algebra, CPU, layer-62 historical payload; NOT a GPU result)

- fp64: A(scan-checkpoint) == A(per-node replay) exactly; B and C factors vs oracle u ≤ 6.7e-16; all-node states ≤ 1.3e-15; every accepted-prefix commit (10 paths) ≤ 1.3e-15 for A-replay / B-compact / C-compact. The three mechanisms are the same algebra.
- fp32 torch mirrors vs fp64 oracle: out max_abs 1.3–2.3e-8 (rel 2–4e-7), state max_abs 2.6–4.0e-7 (rel 0.7–1.1e-7), factors rel ~2e-7 — all at the fp32 floor; B and C fp32 differ from each other only in last bits.
- Historical served kernel bytes (June 2026 scan): out (bf16) within 1 bf16 ULP of oracle; state (fp32) max_abs 4.6e-7 (rel 1.3e-7) — consistent with the archived "fp32 floor" claims for the sequential scan on this payload.
- Gate ranges on this payload: g ∈ [−3.32, −4.7e-7], cum_g min −16.4 → P_min 7.6e-8, min visible decay ratio 5.3e-7; no fp32 underflow at depth 5.

## Next step

Run the E7a ladder with the fixed harness (background, ~30 min), then write the reducer (stage-isolation table, decay-range table, memory table, shared-device timing table with attribution status) and update the paper only with evidence-backed rows. Fresh-capture prefixes (8 pilot / 32 confirmation) remain blocked on model-runtime qualification + memory headroom (unrelated server comes and goes; see telemetry).

## Evidence limits carried forward (do not drop)

- Historical loaded FA2/JIT hashes and July model-file identity are unknown; present hashes do not recover them.
- H3 42.74/39.95/32.85 tok/s = pure-decode event accounting, not full-campaign throughput; H3 and H4 not additive.
- July H3 patchers reconstructed -> two target-logit constraint passes; not relabelable as uniform temp-0.36.
- P0 memory: ~52 GiB available, swap occupied; recheck before any full-model work; do not blindly reuse 0.6 gpu-mem-util.
- FA2 load check was CPU-only library load, not GPU dispatch/numerical qualification.
- E7a mechanisms B and C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems.
