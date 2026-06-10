# FR13 Overhead Decomposition Plan — static counts, labeled ranges, profile protocol (2026-06-10)

**Status**: PLAN + static inventory. CPU-only synthesis from existing code/artifacts — **NO GPU was used for this doc.**
**Binding context**: `FR13_SPEED_TAX_BASELINE.md` (the table + validity scope + deployment-regime spec) and
`FR13_BUG_CLASS_PLAYBOOK.md` class 12 (**no number leaves this flow unlabeled** — every cell below carries a label).
**Source pins**: legacy = `main @ 9001ef43`; replay = `origin/fr13-replay-route @ 33e9a8d0` (branch is actively
pushed by another workflow — **re-grep the replay counts before binding them into a run**); patcher =
`scripts/fr10_phase4_patch_vllm_tree_gdn.py`; kernel = `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`.

**Labels used**: `[EXACT-CODE]` counted from source, exact. `[MEASURED]` from an existing artifact (validity caveats carried).
`[DERIVED]` arithmetic on exact/measured inputs. `[EST]` reasoned range, NOT a measurement. `[UNMEASURED]` no number exists anywhere.

---

## 0. The gap being decomposed [MEASURED, B=1/BI=1/instrumented debug regime — direction-only per the validity scope]

From the FR13-TAX table (sum-basis s/fwd vs the w78aq6xum bandwidth-floor prediction `[P]`):

| arm | s/fwd [M] | pred floor [P] | **own-floor overhead [DERIVED]** | vs native measured [DERIVED] |
|---|---|---|---|---|
| native_mtp5_bi1 | 0.2127 | 102.8 ms | **+109.9 ms (2.07x own floor)** | baseline |
| legacy_cat9 | 0.3936 | 116.6 ms | **+277.0 ms** | +180.9 ms |
| replay_cat9_b2 | 0.3270 | 100.9 ms | **+226.1 ms** | +114.3 ms |
| legacy − replay |  |  |  | **66.6 ms** (the replay saving) |

The regime is **OVERHEAD-dominated**: even native runs at 2.07x its own bandwidth floor, and the tree arms carry
2.0–2.5x native's overhead. The bandwidth model is silent about all of it. This doc inventories the unmeasured
suspects (playbook class 12), bounds each with a labeled range, and folds the discriminating measurements into the
deployment-regime campaign already specified in `FR13_SPEED_TAX_BASELINE.md`.

---

## 1. Static counts [EXACT-CODE unless labeled otherwise]

### 1.1 Capture-mode structure (suspect a)

- Compilation config on every non-eager boot (tree AND native): `mode=VLLM_COMPILE`, `cudagraph_mode=FULL_AND_PIECEWISE`,
  capture sizes `[1..80]`; `splitting_ops` includes `vllm::gdn_attention_core`, `vllm::unified_attention_with_output`,
  `vllm::unified_kv_cache_update`. Cites: `output/fr13_method_a_bi_campaign/T1/cuda_graph_proof.txt` (engine init line),
  `output/fr13_accept_only_20260610T002243Z/tree/docker_full.log:26,141,150-151`. [MEASURED boot-log]
- Tree boots DO capture: `PIECEWISE=8 (largest=80)` + `FULL=4 (largest=40)`; FULL sizes = uniform-decode
  q_len=10 × num_reqs 1..4 → {10,20,30,40} [DERIVED from largest=40 + max_num_seqs=4; the size list is not printed].
- TREE_ATTN is graph-capturable **only because our patch** declares
  `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` (patcher :7297-7298); stock TREE_ATTN inherits NEVER →
  dispatcher downgrades FULL→PIECEWISE (`FR13_TREE_ATTN_CUDAGRAPH.md`). [EXACT-CODE]
- Model = **48 GDN + 16 full-attn layers** (`/models/qwen3.6-27b-fp8/config.json`, full_attention_interval=4). [EXACT-CODE]
- PIECEWISE forward structure: 64–80 split points → **65–81 graph segments + 64–80 eager attention/GDN op regions per
  forward** [DERIVED from splitting_ops × layer counts; the 64-vs-80 range is whether `unified_kv_cache_update` splits
  separately on the 16 full-attn layers in this build — unverified].
- FULL graphs ENGAGE live on tree arms: the gate-4 root cause was committer pending-dict aliasing **across per-batch-size
  FULL graphs** (pins 48 × 201.3 MB ≈ 9.4 GiB scratch; `FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md:43`), and boot1 crashed
  DURING capture because `.cpu()` diagnostics ran inside the captured forward (`FR13_REPLAY_GPU_GATES_BIND.md:19`). [MEASURED]
- **LEGACY statically cannot FULL-capture the tree forward**: `tree_state_all = torch.empty` fp32, 201.3 MB **per GDN
  layer per step, allocated inside the forward**; the replay branch comments call it "the capture-blocking allocation"
  and remove it under `FR13_REPLAY_ROUTE=1` (`store_node_states=False`). [EXACT-CODE]
- **Capture mode of the speed-tax-table boots is UNRECORDED**: `boot*_launch.log` files in `output/fr13_replay_gpu_gates/`
  are 4-line wrappers (free -h + container hash); no `cudagraph_mode`/`Capturing` lines exist in fr13_replay_gpu_gates,
  fr13_convfix_ab, or fr13_s1s2s3_discriminate. Only inference: boot1 = `failed_capture_crash`, boot1a/boot3 eager,
  b1/b2 "captured, FINAL_LOGIT only" per the bind doc. The native 0.2127 arm has NO saved cuda_graph_proof either
  (mode inferred from launcher default ENFORCE_EAGER=0, `scripts/fr10_launch_speed_server.sh:80,222`). [MEASURED-ABSENCE]
- Runtime FULL-vs-PIECEWISE **dispatch mix per step is UNMEASURED everywhere**: every saved init line shows
  `cudagraph_metrics=False`, `enable_logging_iteration_details=False`. [MEASURED-ABSENCE]

### 1.2 Host-sync inventory per step (suspect b) [EXACT-CODE, gate-walked: every grep hit mapped to its governing flag]

Serving path = diagnostics OFF (all FR10_METRICS / FR12_* / FR13_*LOG / capture flags at defaults). Greedy decode.

| site | LEGACY (main@9001ef43) | REPLAY (fr13-replay-route@33e9a8d0) |
|---|---|---|
| per-layer patched GDN forward (×48) | **0 syncs** (all 159 grep hits flag-gated OFF) | **0 syncs** (165 hits, +6 are FR13_REPLAY_BOUNDARY taps, default OFF) |
| committer replay-launch glue (×48 layers/commit) | n/a (replay-only) | **2 unconditional `.item()`/layer = 96/step** (greedy: replay-patcher L4339/L4345; sampled twin L4823/L4829) |
| greedy committer entry (×1/step) | **6 small D2H** (`.cpu().tolist()` parents/drafts/parent_targets/self_targets/bonus/counts, main L3472-3476,L3482) | same 6 |
| sampled committer (×1/step, sampled arms only) | +2 LARGE D2H (rows×vocab fp32 softmax ≈5.5 MB each, L3890-3891) + 1–2 `.item()`/request | same |
| committer publish glue (×1/step) | ~3 pageable H2D `.copy_(torch.tensor(...))` + R×(len+1) scalar device writes (L3684-3691,L3730-3744) | same |
| REQKEY rewrite / tree-metadata / depth-positions / mamba copy-meta (×1/step) | **0 GPU syncs** (`.tolist()` hits are numpy/CPU); ~5 pageable H2D + python loops | same |
| tree_attn override | **net −4 syncs**: deletes 4 native TreeAttention `.item()` calls (build-time ints; patcher comment :6373-6376 — done for capture) | same |
| `torch.cuda.synchronize` | **ZERO anywhere on main** | only in FR13_REPLAY_BOUNDARY_LOG eager diagnostics (default OFF, fail-loud on capture) |

The 96/step replay count **re-derives the remediation's "~96/step" exactly** (2 × 48 GDN layers). The sync→launch→sync
pattern also serializes the 48 replay Triton launches (defeats launch-ahead). `output_scale` is a python float fixed at
metadata-builder init — no sync there.

### 1.3 Python-in-forward inventory (suspect c) [EXACT-CODE counts; launch counts DERIVED ±30%]

Per GDN layer per step, serving path, both routes (conv carrier identical; scan publish differs):
- **per-layer per-step tree re-parse**: `json.loads(SPEC_CONFIG)` + `ast.literal_eval` + sort + parent/index/path0/branch
  list rebuild (main-patcher L866-905) — pure python, EVERY layer EVERY step;
- ~10–15 `os.environ.get` lookups + module getattr in try/except;
- conv section per request row: ~6 window ops, bias clone/expand, python loop over conv width (4) × 3–5 tap casts/muls,
  `triton_ex2_silu_bf16` launch, python loop over tree_n nodes (10 at cat9) × ~6 ops, `torch.stack`, `conv_state.index_copy_`;
- scan section per row: 5–7 `.contiguous()` slices, `launch_tree_gdn_prepared` (~30 python validations + 1 Triton
  dispatch); LEGACY adds the 201.3 MB fp32 `tree_state_all` alloc + per-row ssm `index_copy_` publish; REPLAY replaces
  with 4 ring `.copy_` + flag `fill_` device writes (capture-safe by design).
- **Derived eager launch count**: ~90–110 launches per (layer,row) → **~4.5–5.5k launches/step at B=1 cat9** across 48 GDN
  layers, vs ~2–3 fused launches/layer native [DERIVED by op-listing, ±30%]. PIECEWISE regime: ~180–430 launches +
  65–81 segment replays [DERIVED].
- Drafter: patched eagle.py MTP rollout (~5 sequential mini-forwards + caterpillar assembly in python); **no drafter
  capture phase appears in any boot log** — drafter graph status UNVERIFIED. [MEASURED-ABSENCE]

### 1.4 Pad rule (suspect d) [EXACT-CODE]

- `n = N_draft + 1` (root included); `n_pad = 1 << (n - 1).bit_length()`, cap 16 (`NotImplementedError` above), at
  patcher :226-231 (live init) and `padded_nodes()` kernel :74-78 (offline only). **NO pad-forcing env exists** (grep
  verified). Single-source: per-step launch reads `tree_n_pad = visible_mask.size(0)` (:2252) with separate
  `n_actual = tree_n` (:2251, :2912-2913); replay rings are init-sized to n_pad (kernel :758).
- Boundaries: pad8 covers N=4..7; pad16 covers N=8..15; N=16 → n=17 → pad32 REJECTED (catalog `node16_REJECTED`).
- Pad is a **discrete kernel-cost regime**: `N_PAD` is `tl.constexpr`; `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K), fp32)`
  (kernel :458) + nested `tl.static_range(0, N_PAD)` ancestry loops guarded `j < N_ACTUAL` (:459-462). pad8→pad16 =
  ~4x static-unrolled ops + 2x cache footprint (the known N_PAD=16 spill regime; num_warps=8 interim, kernel :843);
  48 launches/forward at B=1 (one per GDN layer per request).
- `n_actual < n_pad` is ALREADY the deployed configuration (cat9: n_actual=10, n_pad=16; mask rows ≥ n are all-zero,
  init loops :234-240) — forcing pad16 at n_actual=6 exercises the identical mechanism. [EXACT-CODE]
- **The sweep catalog CANNOT isolate pad as-is**: chain5 is the only pad8 point and confounds N (5 vs 9+), topology class
  (`fr10_tree_has_sibling` False vs True, patcher :298), and pad simultaneously. The 0.0108 s/fwd-per-node OLS slope was
  fit ACROSS the pad boundary. [EXACT-CODE + catalog read]

### 1.5 BI=1 (suspect e) [MEASURED-elsewhere]

`reference_fr10_speed_measurement_pitfalls`: BATCH_INVARIANT=1 = known slow-GEMM regime, half of the FR10 "8 TPS" double
artifact; OFF for speed. The deployment spec already mandates **BI=0 both arms**. BI=1 on TREE_ATTN additionally requires
the FR13_BI_TREE_ATTN allowlist patch (stock vLLM refuses; batch_invariant gate log line 63). No paired BI=0/BI=1 number
exists for THIS model on GB10 [UNMEASURED].

---

## 2. Suspect ledger — expected-magnitude RANGES (every value labeled)

Decomposition frame: `overhead(arm) = SHARED (native-class, present on all arms) + TREE-EXCESS (tree arms only)`.

### SHARED class (explains native's +109.9 ms; also present on tree arms)

| id | suspect | range (ms/fwd) | label + basis |
|---|---|---|---|
| S1 | BI=1 slow-GEMM penalty | **30–80** | [EST] direction proven (pitfalls ref), magnitude unmeasured on this model; the single largest native suspect |
| S2 | floor-model optimism (27 GB ÷ 273 GB/s assumes peak streaming; real B=1 fp8 effective BW + attention + non-overlap) | **10–40** | [EST] model-error, not arm work; unvalidated — no BW profile exists |
| S3 | window instrumentation (FR10_METRICS=1, jsonl tracers, FINAL_LOGIT fp32 DtoH+torch.save: ~6 MB/fwd native, ~9.9 MB/fwd tree) | **5–25 native / 10–40 tree** | [EST] sizes EXACT-CODE, cost estimated; inside the 0.213/0.327/0.394 table numbers per the validity scope |
| S4 | vLLM per-step host work outside graphs (scheduler, sampler glue, detokenize, API) at B=1 | **5–15** | [EST] generic vLLM B=1 step overhead class |

SHARED sum: **50–160 ms** vs native measured **+109.9 ms** → native is plausibly FULLY accounted at mid-range (~105).

### TREE-EXCESS class (on top of SHARED; explains legacy +180.9 / replay +114.3 ms over native)

| id | suspect | route | range (ms/fwd) | label + basis |
|---|---|---|---|---|
| T1 | graph breaks + python-in-forward + eager/piecewise dispatch (suspects a+c, inseparable until profiled) | both | **10–101** | [EST bracketed by MEASURED]: eager-vs-captured deltas = replay +56 ms (b2 vs b3, LOWER bound — captured side carried the logit hook) and legacy +101 ms (b1 vs b1a, UPPER-confounded by b1a instrumentation), `output/fr13_replay_gpu_gates/*_probe.json`. Static side: eager ≈4.5–5.5k launches/step → 25–80 ms dispatch [EST]; PIECEWISE ≈ 10–40 ms [EST]; FULL ≈ ~0 (python baked out). Actual value depends on the UNRECORDED capture mode of each table boot |
| T2 | legacy scratch traffic: 48 × 201.3 MB fp32 `tree_state_all` alloc+write+publish-read + per-row ssm `index_copy_` | legacy only | **35–75** | [EST] 9.66 GB/fwd [EXACT-CODE size] at 130–270 GB/s effective; anchored by the MEASURED legacy−replay delta 66.6 ms (replay removes exactly this) |
| T3 | replay committer syncs + serialized launches: 96 `.item()`/step + 48 de-overlapped Triton replay launches | replay only | **1–6** | [EST] 96 [EXACT-CODE] × 5–30 µs idle-stream + de-overlap; = 0.4–2.6% of the 226 ms overhead. The remediation's "<1% of 99 ms" sits at the LOW end and is NOT an upper bound (busy-stream syncs are unbounded above) |
| T4 | committer entry D2H + publish H2D (greedy) | both | **0.05–0.4** | [EST] 6 small D2H [EXACT-CODE] × 5–30 µs + ~10–20 pageable H2D × 2–10 µs; sampled arms add 0.2–1 ms [EST] (2 × 5.5 MB D2H + fp32 vocab softmax) |
| T5 | pad8→pad16 kernel regime step (cat9 runs pad16) | both | **2–40** | [EST] 48 launches × +50..+750 µs/launch; upper end only in the h_cache spill regime; sits inside the 36–40 ms/fwd N5→N9 residual (measured step +42–46 ms vs +6.3 ms row-traffic prediction) — currently CONFOUNDED, P1/P2 arms isolate |
| T6 | TREE_ATTN vs FLASH_ATTN per-kernel cost on the 16 full-attn layers + 10-vs-6 q-rows beyond row-traffic floor | both | **0–30** | [UNMEASURED → EST placeholder] no number exists anywhere; stage-1 top-kernels table pins it |
| T7 | drafter rollout delta (patched eagle.py python caterpillar assembly vs native MTP-5 rollout; capture status unknown) | both | **0–20** | [UNMEASURED → EST placeholder] only the patch DELTA counts (native also rolls 5 MTP steps) |
| T8 | REQKEY rewrite + tree-metadata + depth-positions python per step | both | **0.5–5** | [EST] pure python loops + json/ast parse, counts EXACT-CODE |

### 2.1 Sanity reconciliation — do the ranges account for ~110 / ~277 / ~226 ms?

- **Native +109.9**: S1+S2+S3+S4 = 50–160 [EST]. **Accounted at mid-range.** Dominated by two unmeasured shares (S1 BI, S2 floor fidelity).
- **Legacy +277.0**: SHARED(tree) 55–175 + T1 10–101 + T2 35–75 + T4 ~0.1–0.4 + T5 2–40 + T6 0–30 + T7 0–20 + T8 0.5–5
  = **103–446 [EST]**, midpoint ≈ 275. **Plausibly accounted**, but the range is wide because T1's value depends on the
  unrecorded capture mode and T2/T5/T6 are unprofiled.
- **Replay +226.1**: legacy ledger − T2 + T3 → predicted replay−legacy delta = **−29 to −74 ms** vs MEASURED −66.6 ms.
  **Consistent**, and the measured delta anchors T2 toward its upper half (~60–70 ms) if T1 is route-equal.
- **Cross-check that fails informatively**: if the captured tree boots actually replayed FULL graphs per step (T1 → ~0),
  the explained tree-excess shrinks to ~38–170 ms and the ledger's LOW end falls short of the measured excesses by
  **up to ~80–120 ms** → an unexplained mass that ONLY the stage-1/stage-2 measurements below can assign.

### 2.2 What remains UNEXPLAINED (the honest list)

1. **T1's actual value per table boot** — capture mode unrecorded; the 56–101 ms eager brackets are
   instrumentation-asymmetric in opposite directions (captured arms carried the logit hook; b1a carried extra captures).
2. **S1 (BI) share of native's 110 ms** — no paired BI=0/BI=1 number on this model; could be 30 ms or 80 ms.
3. **S2 floor fidelity** — the 102.8/116.6/100.9 ms floors are peak-BW predictions; nobody has measured achieved BW.
4. **T6 (TREE_ATTN kernel delta)** — zero evidence either way.
5. **T5-vs-per-node-python split** inside the +42–46 ms N5→N9 step — structurally confounded in the current catalog.
6. **Drafter (T7) capture status and cost** — absence of capture log lines is weak evidence.

**CORRECTION bound by this doc**: `FR13_WHY_SLOWER_VERDICT.md:23` ("Eager-launch tax: ≤2.7% of the floor … CUDA-graph
capture removes it") counted GPU **hardware** launch overhead only. The MEASURED eager-vs-captured deltas are
56–101 ms/fwd (17–26%) — the missing mass is CPU-side python dispatch + per-layer patched glue. That line is
RELABELED: "GPU HW launch overhead only; total eager tax measured 56–101 ms/fwd, see FR13_OVERHEAD_DECOMP_PLAN.md".

---

## 3. PROFILE PROTOCOL — folded into the deployment-regime campaign (`FR13_SPEED_TAX_BASELINE.md` spec)

The deployment campaign (B=4, BI=0, FR10_METRICS=0 + all LUMO logging unset, FULL capture proven, SWE-Verified
workload, /metrics-delta basis) remains the speed-of-record. The stages below ADD evidence arms around it; every
diagnostic window is LABELED and never substitutes for a speed-of-record number.

### Stage 0 — capture-proof mandate (every boot, zero cost)
- Save `docker logs $CONTAINER > $RUN_DIR/docker_full.log` per arm, ALWAYS (the speed-tax table lost this; the
  4-line `*_launch.log` wrappers are insufficient). Format reference:
  `output/fr13_b4_eager_bisect_20260609T203718Z/{tree,native}/docker_full.log` + `eager_proof.txt`.
- Emit `cuda_graph_proof` per arm = {engine-init `cudagraph_mode` value, `splitting_ops` list, count of
  `Capturing CUDA graph` lines, `Profiling CUDA graph memory: PIECEWISE=n FULL=n` line, assert NO eager/PIECEWISE-fallback
  warning}. Expect FULL largest=40 (tree, q10) vs largest=24 (native MTP-5, q6) — record the q-len asymmetry.
  This satisfies the deployment spec's existing "cuda_graph_proof per arm" requirement by DEFINING its evidence format.

### Stage 1 — ONE eager torch.profiler diagnostic window per arm class [LABEL: diagnostic-only; counts/shares bind, absolute ms do NOT]
- Mechanism: launcher already sets `VLLM_SERVER_DEV_MODE=1` (exposes `/start_profile` + `/stop_profile`); add a
  flag-gated `-e VLLM_TORCH_PROFILER_DIR=/logs/torch_profile` passthrough (default unset = zero serving impact).
  `ENFORCE_EAGER=1` already supported (launcher :215). Boot with **BI=0, FR10_METRICS=0, ALL diag/capture envs OFF**
  so the trace sees the serving path. Eager by policy: capture instrumentation crashes graphs (boot1 crash precedent).
- Window: warm 2 requests; `/start_profile`; ONE pinned prompt (`prompts_swe4.json`), max_tokens=64; `/stop_profile`.
- Arms: native-eager | legacy-cat9-eager | replay-cat9-eager | legacy-chain5-eager | (optional) chain5+FORCE_PAD16-eager.
- Extraction per decode step (CPU post-processing, chrome trace, with_stack):
  (i) `cudaLaunchKernel` count → **suspect a** (launch-count delta tree-vs-native; pins the 4.5–5.5k estimate);
  (ii) count + summed time of `cudaStreamSynchronize` / `cudaMemcpy*DtoH` / `aten::item`/`aten::_local_scalar_dense`,
  attributed via stacks → **suspect b** (must reproduce 96/step replay + 6/step committer entry; converts counts→time
  and bounds per-sync cost as a measured range);
  (iii) CPU-op time share between kernels → **suspect c** (python-in-forward, legacy vs replay vs native);
  (iv) top-30 kernels by cuda_time incl. tree-GDN per-launch duration at pad8 vs pad16 → **suspects d + T6**
  (per-launch kernel duration is meaningful even eager; TREE_ATTN-vs-FLASH per-kernel cost falls out of the same table).
- The native-eager vs native-captured pair additionally splits S4 (generic vLLM step cost) from patch-added python.

### Stage 2 — captured-mode dispatch evidence (inside the deployment campaign)
- Speed-of-record arms (B=4, BI=0, METRICS=0): cuda_graph_proof per Stage 0. If a tree arm cannot reach FULL,
  **that finding IS the suspect-(a) verdict** for that route.
- ONE SEPARATE labeled diagnostic boot per tree arm with vLLM's own observability flags (confirmed present-but-off in
  v0.19.2rc1.dev134+gfe9c3d6c5): `--observability-config '{"cudagraph_metrics": true,
  "enable_logging_iteration_details": true}'` → **per-step FULL/PIECEWISE/eager dispatch-mode counts over the pinned
  battery — the one number no existing log contains.** Launcher gains a flag-gated `FR13_VLLM_EXTRA_ARGS` hook
  (default empty) to append the flag. NEVER the speed-of-record window (logging overhead).
- nsys: availability in `vllm/vllm-openai:cu130-nightly` UNVERIFIED — check `docker exec <c> which nsys` at next boot.
  If present: one ~30 s steady-state window = the only direct launch count under graph replay (`cudaGraphLaunch` vs
  residual `cudaLaunchKernel` per step). OPTIONAL — Stage 1 counts + Stage 2 proofs bound the same question.

### Stage 3 — pad-isolation arms (suspect d) added to the deployment sweep
- **P2 (definitive, same-N)**: chain5 natural (n=6, pad8) vs chain5 + `FR13_FORCE_N_PAD=16`. Requires the minimal
  diagnostic env: at patcher :227, after deriving n_pad, read `FR13_FORCE_N_PAD`; if set/nonzero validate
  (power of two, ≥ derived, ≤ 16, else raise), override, and LOG the served n_pad at boot (assert evidence); launcher
  adds `-e FR13_FORCE_N_PAD="${FR13_FORCE_N_PAD:-0}"`. Default 0 = byte-identical current behavior. Semantically safe:
  n_actual < n_pad is the deployed cat9 configuration already; mask rows ≥ n are all-zero; kernel guards `j < N_ACTUAL`.
  Pre-flight: one-line audit that the FA2/TREE_ATTN bias-mask sizing has no independent pad assumption.
- **P1 (no-code-change cross-check)**: caterpillar7 (N=7, n=8, pad8) vs caterpillar8 (N=8, n=9, pad16) — dN=1, both
  depth-5 (no drafter-engagement caveat), both has_sibling=True. Pad step ≈ (cat8 − cat7) − within-pad16 per-node slope
  (slope from N ∈ {9,12,13,15}). `--expected-draft-count 7/8` engagement asserts.
- REJECTED: "chain5 + 6th dummy node" — one real node only reaches n=7 (still pad8); crossing to pad16 needs +3 nodes
  (= P1 anyway) and any real node changes drafter/verifier/sampler work — NOT a pad isolation.
- Bind only if P1 and P2 estimates AGREE.

### Stage 4 — BI (suspect e): no deliverable-regime measurement needed
Deployment spec already mandates BI=0 both arms. If a number for S1 is wanted (to close the native-110 ledger):
ONE paired native boot BI=0 vs BI=1 on the same pinned battery [LABEL: diagnostic].

### Suspect → measurement map

| suspect | discriminating measurement |
|---|---|
| (a) graph breaks / launches | Stage 0 + Stage 2 proofs & dispatch-mode counts; Stage 1 launch counts; optional nsys |
| (b) host syncs | Stage 1 sync counts/times vs the §1.2 static inventory (96/step, 6/step) |
| (c) python-in-forward | Stage 1 CPU-share, legacy vs replay vs native-eager |
| (d) pad8→pad16 | Stage 3 P2 (definitive) + P1 (cross-check) + Stage 1 per-launch kernel durations |
| (e) BI=1 | known direction (pitfalls ref); deployment = BI=0; optional Stage 4 paired native boot |
| T2 scratch traffic | already anchored by the measured legacy−replay 66.6 ms; Stage 1 trace attributes the alloc+copy time directly |
| T6 TREE_ATTN delta | Stage 1 top-kernels table (same trace, free) |

### Invalidation rules (class 9/12 carried)
- Any boundary-tap / METRICS / capture flag ON invalidates a speed number (taps add 2 full-device synchronizes per
  instrumented layer per commit).
- Stage-1 absolute milliseconds never bind (eager + profiler overhead + B=1); only counts, shares, and per-launch
  kernel durations do.
- Replay arms inherit the live accept-bug confound until the gate-4 fix lands; per-forward ratios remain informative
  at matched shape (per the baseline doc), accept-dependent columns do not.

---

## 4. Source pins / reproduce

- Readers (CPU, read-only): boot-log capture inventory (`output/fr13_method_a_bi_campaign/T1/cuda_graph_proof.txt`,
  `output/fr13_accept_only_20260610T002243Z/tree/docker_full.log`, `output/fr13_replay_gpu_gates/*`), static sync/python
  gate-walk of patcher main@9001ef43 vs replay@33e9a8d0, pad/topology audit of patcher + kernel + sweep catalog.
- Measured per-forward inputs: `output/fr13_replay_gpu_gates/{b1,b1a,b2,b3}*_probe.json`,
  `output/fr13_speed_tax_gate/backfill_reduce.json` (table basis).
- This doc is the static-counts + ranges + protocol companion to `FR13_SPEED_TAX_BASELINE.md`; the deployment campaign
  executes Stages 0–3 in one GPU window class (post replay-accept-fix, per the baseline doc's scheduling note).
