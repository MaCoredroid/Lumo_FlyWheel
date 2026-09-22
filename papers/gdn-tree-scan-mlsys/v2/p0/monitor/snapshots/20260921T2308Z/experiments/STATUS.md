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
| 22:25Z | Review-02 reply written | done | `p0/monitor/2026-09-21-claude-reply-02.md` (mtime 22:25) |
| 22:26:16Z–22:36:06Z | **E7a ladder** (4 historical + 5 synthetic operand sets + B4 arm; iters=200) | completed, status=completed, start/end hashes match, 122 cubins, 0 invalid cells | `out-20260921T222616Z-e7a-ladder/` (+ `source_snapshot/` of the loaded modules, `telemetry/container_host_pid.txt`) |
| 22:31:54Z | Codex review #3 (non-finite statistics; PID-ownership attribution; B4/fix-date/matched-work/snapshot boundaries) | received | `p0/monitor/2026-09-21-review-03.md` |
| 22:33:37Z | Ladder container host PID recorded while running (2748559 = only nvidia-smi compute PID) | done | `out-…-ladder/telemetry/container_host_pid.txt` |
| 22:36:2xZ | Review-03 fixes applied AFTER the ladder exited (compare finite pairs; runner PID record; matched-output timing; reducer ownership attribution) + CPU tests ALL PASS (25) | done | `e7a/tests_out/review03_regression_*.json`; post-run source hashes: core `9588afe4…`, device `e99cbf4b…`, runner `2ef28156…`, reducer `21e859e5…` |
| 22:36:58Z–22:42:33Z | Tiny-gate synthetic pass (chain 11/5, caterpillar; regime tiny-gates) on the fixed sources | completed; status=completed; hashes match; 0 invalid cells; no foreign PID | `out-20260921T223658Z-e7a-tinygates/` |
| 22:37Z | Review-03 reply written | done | `p0/monitor/2026-09-21-claude-reply-03.md` (mtime 22:37) |
| 22:47:46Z | Codex review #4 (paper wording: bf16 agreement range, numeric vs bitwise, B4 control scope, causal wording, chronology) | received | `p0/monitor/2026-09-21-review-04.md` |
| 22:55:43Z | USER-authorized host memory cleanup by Codex (drop_caches, guarded swapoff/swapon) → MemAvailable 102.6 GiB, swap 0 | done (not by this session) | `p0/monitor/2026-09-21-memory-cleanup.json` |
| 22:59:27Z | Review-04 wording corrections applied (main.tex/abstract/STATUS); harness v3: integer-view bitwise metrics, seam-ablation oracles, B4 controls for all 4 requests | done; CPU tests ALL PASS | `e7a/tests_out/v3_regression_*.json` |
| 23:00:26Z | v3 reruns launched (ladder-v3 → tinygates-v3, serial) | running | `out-20260921T230026Z-e7a-ladder-v3/` |

## Current state

**Stage:** E7a step 1 (algebra/mechanism pilot) COMPLETE at kernel level: ladder + tiny-gate runs reduced and promoted with labels; paper updated (PDF stale). **E7a step 2 (fresh 8-prefix pilot / 32-prefix confirmation) BLOCKED** by the launcher's own gates (see Blockers). E2/E7b and E1 NOT started (they require the fresh-capture boot).
**Model inference performed so far in this worktree:** none (kernel-level work only).
**Raw artifact directories:** `papers/gdn-tree-scan-mlsys/v2/experiments/out-<utc>-e7a-*/` (929 MB total, ~930 MB of which are per-run Triton caches whose cubins are hashed in the manifests); nothing committed or pushed (handoff: no push; worktree left with untracked `papers/gdn-tree-scan-mlsys/v2/` as delivered plus the new experiment files).

## Codex review #3 (22:31:54Z) — resolutions

Reply with evidence: `p0/monitor/2026-09-21-claude-reply-03.md`. (1) `compare()` computes finite-pair-only statistics with `n_finite`, NaN when no valid pair; reducer renders **INVALID** for any non-finite candidate OR reference and never summarizes such rows (ladder: 585 comparison dicts, 0 non-finite → metrics unchanged). (2) attribution keys on the recorded container host PID (`docker inspect .State.Pid`), foreign PIDs ⇒ shared, missing record ⇒ unverified, coverage gaps and a ≤5 % probe-drift criterion enforced; fixture with a foreign python3 PID now labels SHARED-DEVICE. Corrections: the smoke did NOT exercise B4 (first B4 = ladder); the legacy export "fix date" is withdrawn (which binary changed the path, and when, is not established); timing rows carry WORK labels (B/C verify rows use fp32 output stores, native helpers allocate) and are not speedups; matched bf16-output B/C timing + preallocated native closures added for later runs.

## E7a pilot results (kernel-level; ladder `out-20260921T222616Z-e7a-ladder`; 4 HISTORICAL dependent operand sets [layers 62/1/12/0, caterpillar depth 5, B1] + SYNTHETIC chain depth 1/5/11, binary depth 3, caterpillar; NOT fresh prefixes; NOT the authors' systems)

- **Algebra:** fp64 identities of A/B/C ≈ 1e-15 (CPU self-test + unit tests on chain-11/binary-3/caterpillar).
- **Verifier (stage 1), ieee fp32 dots:** all realizations at the fp32 floor vs the fp64 oracle on all 9 sets (out max-rel 0.9e-7–5.2e-7; factors U max-rel 1.7e-7–3.4e-7). The **production scan's fp32-store output is bit-identical to the pinned native spec-update kernel on 9/9 sets** (bf16-store outputs agree in 99.998–100 % of elements; those are separate compilations); B_fs and C_nm bf16 outputs equal native's bf16 outputs in 99.97–99.99 % of elements.
- **Commit (stages 2/3):** compact reconstruction from IDENTICAL oracle factors ≈ fp32 floor (max-rel 1.0e-7–2.1e-7) ⇒ commit arithmetic itself is at the floor; own-factor commits (B_fs, C_nm) and A's replay all at max-rel 1.0e-7–3.7e-7 vs oracle; A replay vs native_sg state max-rel ≤1.3e-7 with 98.3–99.3 % bit-exact elements.
- **Precision is decisive:** tf32 tensor-core dots (a plausible fragment-MMA realization; the Bole paper does not specify precision) degrade out to max-rel 6.5e-4–1.4e-3, factors to 4.0e-4–7.6e-4, commits to 3.0e-4–8.1e-4 (2.6e-4–8.9e-4 from oracle factors), and bf16-output agreement with native to 74–79 %.
- **The two pinned native paths disagree by design:** one-token decode (`fused_recurrent…packed_decode`) vs spec-update (`fused_sigmoid_gating…`) differ by state max-abs 5.7e-4–3.5e-2 (max-rel 2.2e-4–1.7e-3) and only 75–94 % bf16-output equality; the decode kernel matches its OWN exact oracle (bf16-rounded beta, x/sqrt norm) at the floor (max-rel 1.2e-7–4e-7). Any tree kernel can agree with only one of them at the floor; the handoff's "do not merge references" rule is measured, not assumed.
- **Depth (synthetic chain 11):** commit error grows mildly with accepted depth (A replay 5.3e-7 → 1.35e-6 abs; B/C 5.3e-7 → 1.0e-6), still ≈1e-7 relative.
- **Controls:** sibling-reorder relabeling bitwise-invariant for A on 9/9 and for B/C on 8/9 (binary-3: last-bit differences from changed dot accumulation order, 92–96 % exact); N_PAD 16→32 not bitwise (≤1.2e-7 abs); Neumann with d−1 terms fails (U err up to 0.29) — the nilpotency term is load-bearing; replay repeat launches bitwise-stable on all paths.
- **B4 (synthetic, batched B/C kernels):** per-request at floor; batched result bitwise identical to the B1 result for the same request (batch invariance) for both B_fs and C_nm; the production plain-route scan has no batched kernel outside fixed32 mode (4 launches).
- **Transient memory (per request, per layer):** compact factors U+cum_g 396 KB; replay writes (depth+1)×3.1 MB committed rows; full per-node state export (legacy kernels) 50 MB; production scan exports nothing (register checkpoint 128 KB/program).
- **Timing (label: NO FOREIGN COMPUTE PROCESS SAMPLED in this window; probe drift ≤4.2 % on all blocks (criterion ≤5 %); rows do different work; one layer, B1, µs sync-median/pipelined):** A scan 89/65 (bf16 out); B_fs[ieee] verify 110/97 and C_nm[ieee] verify 94–99/81–91 (fp32 out, NOT matched); tf32 verify 31–42; commit: replay depth-5 104/90 (writes 6 rows) vs compact 18–19/9.5–10 (writes 1 row); zero-accept replay 54/36; chain-11: scan 133, replay 207; binary-3 (15 nodes): scan 166; native spec-update chain-5 158 (wrapper allocs); legacy WY with full export 760–1650. These are diagnostic kernel timings, not serving throughput.
- **Tiny-gate regime (SYNTHETIC; `out-20260921T223658Z-e7a-tinygates`, review-03-fixed sources, no foreign compute PID, probe drift ≤2.7 %):** chain-11/chain-5/caterpillar with cum_g down to −912 (P_min = 0 in fp32, min visible decay ratio 0 / 3e-195 / 2e-192): **0 non-finite comparisons**; all fp32 realizations at the floor (out rel 0.9–5.8e-7; commits rel 1.2–1.9e-7; A scan bit-identical to native 3/3; B/C bf16 agreement 99.997–100 %); tf32 commits rel 1.1–1.4e-3. Decay ratios are evaluated as exp(cum_i − cum_j) with masked exponent inputs, never as literal quotients of underflowed P — that is why nothing goes non-finite. With vanishing gates the ancestor coupling vanishes, so the Neumann d−1 control passes there (1.8–2.0e-7); it fails only when coupling exists (ladder: up to 0.29). Matched bf16-output timing: B_fs 107–109 µs, C_nm 95–99 µs (≈constant in node count) vs sequential scan 60 µs (6 nodes) → 90 (10) → 133 (12); compact commit 18.8 µs vs replay 104 (depth 5) / 208 (depth 11).
- **Historical-provenance note:** payload `serving_out` bytes equal the production scan's bf16 output for the layer-62 and layer-12 captures and differ for the two captures taken while the June WY kernel was serving (wy_l0, wy_l1) — a provenance check of which kernel served, not a correctness metric.
- **Excluded:** `A_legacy_state*` (June-8 vendored sequential export skips the root update; source-inspected; fix history unknown).

## Codex review #2 (22:16:54Z) — resolutions

All six findings fixed; reply with evidence: `p0/monitor/2026-09-21-claude-reply-02.md`. Summary: node-major factor slicing + `compare()` shape assertion; manifest sources/flags split with start+final+crash manifests; exponent-input masking (core + both Triton kernels) removes off-path inf·0 NaNs, non-finite entries counted/excluded; monotone signed ULP ordinal with ±0 policy; immutable replay h0 source row (column 15) with repeat-bitwise validation (True on all 10 prefixes); pipefail-safe telemetry plus 1 Hz compute-process inventory for attribution. Smoke-3 observations are listed in the reply as unpromoted.

Resolved harness item from smoke-3: the vendored June-8 (`8a975837`) **sequential** scan kernel's per-node state export disagrees with the oracle (max_abs 0.163) while its outputs match. Cause (source-verified in `e7a/legacy_wy_8a975837.py` lines ~342-346): for every non-root node the export track restarts from `b_h0` and loops `tl.range(1, i + 1)`, skipping the root node's rank-1 update; the output path (`state_i`) does not skip it. This is a defect of that intermediate historical revision; the June-10 payload's exported `serving_state` matches the oracle at 4.6e-7, but which binary changed the export path, and when, is NOT established (P0: historical identities unknown). `A_legacy_state*` rows are excluded from all tables; `A_legacy_out*` rows stay (bit-identical to the production scan). The WY kernel in the same file is unaffected (state at fp32 floor).

## Codex review #1 (22:01:38Z) — resolutions

Reply with evidence: `p0/monitor/2026-09-21-claude-reply-01.md` (review file preserved unmodified).

1. **Timing contention.** Refreshed telemetry at 22:06:00Z: unrelated `VLLM::EngineCore` is now PID 2736995 (34,135 MiB; the server was restarted twice since 21:47), GPU util samples over 5 s = 0/72/0/0/56 % → the device IS intermittently contended by the other server's work. Rule adopted: every E7a timing table is labeled **"shared-device diagnostic"**; `run_in_image.sh` records `telemetry/gpu_util_1hz.csv` + compute-app snapshots before/after each run, and the harness brackets each timing block with a fixed matmul probe. No comparative timing claim will be made without an uncontended window (contention record showing ~0 % baseline util for the whole window). The unrelated server is never stopped.
2. **Chronology.** Guessed "~22:00/~22:20" stamps retracted; table above uses measured mtimes/telemetry stamps or *observed-by*.
3. **Independent-prefix accounting.** Tensor-level check of the 9 unique historical payloads (see reply file): all 9 differ jointly in (q,k,v,a,b,h0); the six layer-0 files share identical layer parameters (A_log, dt_bias) as expected; two fr10-era layer-0 files share identical q/k/v but different a/b/h0 (lineage murky, debugging-era captures); l1/l12 are close (max|Δq| 0.87) and may be the same forward pass. Token ids / positions are NOT stored, so **the number of independent (prompt, position) prefixes is unrecoverable** → historical payloads are counted as **≤ 9 dependent historical operand sets from ≤ 3 known prompt sources, ONE topology (10-node caterpillar, B1)**, usable for algebra/diagnostics only. They contribute **0** to the 8-prefix pilot and **0** to the 32-prefix confirmation; those require fresh captures under the pinned configuration. Reducers key results by `provenance.sha256` and carry `synthetic`/`historical` labels; layer count is never reported as prefix count.

## Blockers / resource notes (updated 23:0xZ after the user-authorized memory cleanup)

**Memory gate: RESOLVED for now.** After Codex's user-authorized cleanup (22:55:43Z) MemAvailable = 102.6 GiB, MemFree ≈ 102 GiB, swap used = 0. The speed launcher's gate (`MemFree >= 100 GiB and swap_used == 0`) and the native launcher's gate (`MemAvailable >= 80 GiB and swap_used == 0`) both pass at this instant; recheck immediately before every boot. Between-run cache/swap recovery is authorized only after confirming no unrelated compute workload and enough RAM to swap in; it is recorded outside timed regions and swap is always restored.

**Supported configuration for the audited FP8 checkpoint (selected):** `scripts/fr10_launch_speed_server.sh` at HEAD — `SERVED_MODEL_PATH` defaults to `/models/qwen3.6-27b-fp8` (read-only mount `/models`), applies `fr10_phase4_patch_vllm_tree_gdn.py`, serves `tree_mtp` with the caterpillar `TREE` default, forwards `FR10_TREE_GDN_CAPTURE_PAYLOAD` (one-shot GDN operand capture), records `served_model.json` (checkpoint identity) and `repo_identity.json`, and uses the STOCK image attention stack (no forked FA2 mount; `ATTENTION_BACKEND` must be set to `TREE_ATTN` for tree correctness — its unset default is `FLASH_ATTN`). Owned container name (`CONTAINER=e7a-capture-<utc>`), own `LOG_DIR` under this worktree, `GPU_UTIL` and `MAX_MODEL_LEN` chosen for a capture boot (not the old 0.6 blindly; recorded). Native reference server for prefix selection / E1 native arm: `scripts/fr13_launch_native_mtp_server.sh` (stock vLLM MTP-5, FP8 default, `FLASH_ATTN`, `GPU_UTIL` default 0.6).

**The HEAD tree launcher remains unusable for FP8**: `SERVED_MODEL_PATH` is `readonly` = `/models/qwen3.8-27b-nvfp4-radixark` (not caller-overridable by design); no override is attempted.

**FA2 binary qualification (source-level finding):** the available forked binary (`_vllm_fa2_C.abi3.so`, 301,219,928 B, sha256 `97fa2519…c0ae1`) is exactly the binary recorded by all four June capture directories (`fr13_forked_fa2.sha256` at the in-container path `/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`) → it is the June tree-bias fork used for the historical operand captures (lineage established). It is NOT the August promoted split-K candidate (`28570f83…`), and the only HEAD gate (`fr13_fa2_qrow32_gqa_pair_gate.py::validate_candidate`) is hard-wired to that candidate's size/sha/source closure and raises "GQA-pair candidate SO identity drifted" for any other file → no supported HEAD gate can certify the `97fa` fork, and the b34 live gate additionally requires a pushed upstream commit and an empty Docker host. Its numerical qualification therefore remains the archived June-era A/B (2-ULP floor), not a current-revision pass. The selected capture configuration does not depend on it (stock image FA2 `d5c5c5de…`, stock Triton `TREE_ATTN`); this is recorded, not hidden.

Other notes:
- Unrelated host-level vLLM server (`Qwen/Qwen3.8-27B-FP8`, from `/home/mark/shared/exp54928`) was resident and active 21:47–~22:12Z, absent since; never touched. Every run's compute-process inventory is recorded.
- Historical GDN payloads: ONE topology, dependent operand sets; zero fresh prefixes.
- Tree launcher pre-gate side effects noted by review 04 (pkill of `sample_dcgm_during_task.py`, global cache drop/swapoff) — the speed launcher only does `docker rm -f` of ITS OWN container name plus `recover_host_memory()`; no `sample_dcgm_during_task.py` process exists on the host (checked 22:5xZ).

## Results so far (algebra, CPU, layer-62 historical payload; NOT a GPU result)

- fp64: A(scan-checkpoint) == A(per-node replay) exactly; B and C factors vs oracle u ≤ 6.7e-16; all-node states ≤ 1.3e-15; every accepted-prefix commit (10 paths) ≤ 1.3e-15 for A-replay / B-compact / C-compact. The three mechanisms are the same algebra.
- fp32 torch mirrors vs fp64 oracle: out max_abs 1.3–2.3e-8 (rel 2–4e-7), state max_abs 2.6–4.0e-7 (rel 0.7–1.1e-7), factors rel ~2e-7 — all at the fp32 floor; B and C fp32 differ from each other only in last bits.
- Historical served kernel bytes (June 2026 scan): out (bf16) within 1 bf16 ULP of oracle; state (fp32) max_abs 4.6e-7 (rel 1.3e-7) — consistent with the archived "fp32 floor" claims for the sequential scan on this payload.
- Gate ranges on this payload: g ∈ [−3.32, −4.7e-7], cum_g min −16.4 → P_min 7.6e-8, min visible decay ratio 5.3e-7; no fp32 underflow at depth 5.

## Paper

`main.tex`/`abstract.tex` updated (main.tex mtime 22:59:27Z) with the E7a pilot (new Section `sec:e7a-pilot` + Table `tab:e7a`; evidence-table and planned-table captions/rows; reconstruction-numerics sentence; abstract sentence). **`main.pdf` is STALE**: no LaTeX toolchain on this host; rebuild with the Mac workflow (`latexmk -pdf main.tex`). `notes/design/experiment-matrix.csv` E7a row marked executed-pilot.

## Next step (in progress)

1. v3 reruns → reduce → reply-04 with bitwise evidence → restore precise "bitwise" wording in the paper only where the integer-view metric supports it.
2. Fresh-prefix pilot (E7a step 2): build a candidate prefix pool from local text, boot the NATIVE MTP-5 reference server (stock), score top-1/top-2 margins and context strata at prefix ends, freeze 8 pilot + 32 confirmation prefixes with seeds/IDs BEFORE any tree output is viewed; then one capture boot per pilot prefix with the speed launcher (`TREE_ATTN`, capture hook), operands → the E7a harness (fresh rows, labeled fresh).
3. E2/E7b continuation harness per `experiments/e2/COVERAGE_MAP.md`; E1 after qualification.

## Evidence limits carried forward (do not drop)

- Historical loaded FA2/JIT hashes and July model-file identity are unknown; present hashes do not recover them.
- H3 42.74/39.95/32.85 tok/s = pure-decode event accounting, not full-campaign throughput; H3 and H4 not additive.
- July H3 patchers reconstructed -> two target-logit constraint passes; not relabelable as uniform temp-0.36.
- P0 memory: ~52 GiB available, swap occupied; recheck before any full-model work; do not blindly reuse 0.6 gpu-mem-util.
- FA2 load check was CPU-only library load, not GPU dispatch/numerical qualification.
- E7a mechanisms B and C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems.
