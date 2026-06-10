# FR-13 Bug-Class Playbook — feed this to EVERY workflow agent

**Purpose (user directive 2026-06-10):** every bug we chased down is a *category* that recurs. Workflow agents must know these signatures BEFORE debugging, and must reuse the banked experiments instead of re-running them. **Workflow authors: include this file (or its relevant rows) in every debugging/build prompt.**

## Part 1 — The bestiary (signature → discriminator → fix pattern)

| # | class | signature | discriminator that cracked it | fix pattern | where it recurs |
|---|---|---|---|---|---|
| 1 | **Slot/batch-position-keyed state** (`cc008587` REQKEY) | first event of a request corrupted; "failing rows MOVE between runs"; **fixed-point determinism** (run2≡run3, run4≡run5 — deterministic given predecessor, NOT random) | B=1 sequential same-seed repeats ×N + the fixed-point test | key by request-id; explicit first-event zeroing; pre-forward rewrite of slot buffers | any persistent buffer indexed by batch row |
| 2 | **Global RNG in per-request paths** (`cc008587` PER_REQ_GEN) | same-seed non-repro at sampled tokens (lcp=1 at token 1); greedy unaffected | greedy-vs-sampled split (greedy clean ⇒ RNG) | seed from `sampling_metadata.generators[req]` | any committer/sampler drawing without `generator=` |
| 3 | **In-place overlapping parallel writes (kernel race)** (`cc008587` REMAP_SEQ) | corruption on accepted_len≥2; torch-det-warn SILENT (Triton invisible); src[1..L] overlaps dst[0..L-1] | detector silence + fixed-point ⇒ not an op; index-math inspection | gather-then-scatter / double-buffer; never in-place permute | any kernel writing rows it may also read |
| 4 | **Spine-only-valid column arithmetic on branch winners** (`c0b53f5d` conv fix) | EPISODIC whole-forward corruption (10-25× baseline) AFTER branch commits; transient, recovers | bind the trigger context (prev accepted_len + winner path) + forced-spine A/B + fix-ON/OFF | derive from the committed PATH's tokens; snapshot BEFORE in-place mutations | every consumer of accepted_len-based column math (h0, conv, KV slots) |
| 5 | **Wrong-row special case / index-order assumption** (`4d45be27` S1) | wrong token served on one winner class; diagnostics silently checking the wrong path | re-derive committer math from logs (163/163 re-derivation exposed the 14) | compute the true index (walk first-children); DELETE special cases; fix the diagnostics too | leaf-order vs BFS vs spine indexing anywhere |
| 6 | **Lazy alloc / per-step objects under CUDA capture** (gate-4 #2; replay-build V2) | works EAGER, breaks CAPTURED; "offline bit-identical, live broken"; per-batch-size graphs alias different buffers | eager-vs-captured bisect | init-time persistent preallocation (the `:184-201` pattern); captured device `fill_` handshakes; NO per-step dicts/allocs in flagged paths | every new buffer in a captured region |
| 7 | **Zero-accept / edge-row staleness** (gate-4 #1) | next-event h0 clamps to col 0 which path-only updates never refresh | the "next event's h0 correctness" check (NOT just "rejected rows untouched") | explicit zero-accept publish path; root→col 0 unconditionally | every publish/commit path's empty-accept case |
| 8 | **Offline single-forward ≠ live multi-step** (gate-4 lesson; replay live bug 2026-06-10 — now PROVEN TWICE) | all CPU/offline proofs pass; live serving corrupts. **The measured-boundary formulation (replay GPU gates):** everything up to the producer's bank bytes can be byte-PROVEN (A/B 126/126) and the route still fails live ⇒ **the defect lives in the INTERVAL between producer-write and consumer-read**: ordering, intervening writers (e.g. native copy paths like `get_temporal_copy_spec` moving/cloning state rows on persistent-batch churn), index/keying drift at request churn, stale non-consumed columns that legacy would have filled but the new route leaves | **boundary instrument**: capture the consumer's input AS-READ (h0 at event N+1) and byte-diff vs the producer's output AS-WRITTEN (event N) — the diff names the interval actor; plus live B=1 same-seed repeat FIRST, and a per-event wiring trace (slot, snapshot values, columns read/written, native copy ops in between) | snapshot at producer time; ordering asserts; audit EVERY native machinery that touches the same rows between steps; account for layout differences the new route introduces (stale columns) | every cross-step contract — and ANY route that changes WHAT is persisted while native machinery still assumes the old layout |
| 9 | **Silent fallback / vacuous instrument** (FR10_REQUIRE_TREE; diag[12]; launcher silent-OFF `FR13_FA2_PREFILL_NATIVE`) | a run "passes" while measuring nothing | engagement asserts (sentinel in logs, backend line, flag in container env) BEFORE trusting any number | fail-loud on disengagement; boot needles; record flag-state headers in every artifact | every flag-gated feature + every launcher |
| 10 | **Shared-source ≠ shared-SASS (codegen identity)** (matrix R4) | two kernels inline the same body but compile differently (constexpr/pressure) | byte A/B on captured payloads, int-view equality (NEVER atol), SASS hash pin | one shared body + identical constexprs/num_warps + the A/B gate re-armed per toolchain | any "bit-exact by re-execution" claim |
| 11 | **Batch-composition / BI-flag sensitivity** | native itself only 0.714 draft-identical across BI flag; near-ties flip on sub-ULP shifts | native-vs-native control arms; pin BI on BOTH arms | per-shape comparators; same-flag-state pairing; floors measured not assumed | every cross-arm comparison |
| 12 | **Measurement traps** (multiple retractions) | TPS÷accept hand-rolls (retracted 2×); prompt-pairing mismatch (lcp=0 artifact, burned 3-4 boots); per-pos counters indexing accepted-path-length ("branches added 0" artifact); single-draw floors (0.0593 vs measured 0.113); non-like-for-like trajectories after fixes | — | raw counters only; capture-once pinned prompts; source-index traces; multi-sample p95 floors; label every estimate | every reduce phase |

## Part 2 — Banked experiments (REUSE, do not re-run / re-derive)
All raw workflow results: `research/fr13_workflows/INDEX.md` (each row = synthesis + adversarial verify, auditable). Binds: `FR13_*_BIND.md` + `FR13_LADDER_LOG.md`. Key reusable instruments:
- **Same-seed native floor protocol** (`FR13_NUM_SPLITS_NATIVE_FLOOR_BIND.md`): measured floor bag-TV **0.113**; seed-pair ≈0.11; the 0.0593 single-draw is superseded.
- **3-arm corruption gate** `scripts/fr13_corruption_gate.py` (tree/native/noise, self-noise-corrected).
- **Fixed-point determinism test** + B=1 sequential battery (cracked class 1).
- **torch-det-warn detector** (`FR13_TORCH_DET_WARN`, inert default) — op-level nondeterminism stacks (silence = classes 1/3/4, not ops).
- **Forced-spine A/B** (`FR13_FORCE_SPINE_COMMIT`, DIAGNOSTIC-ONLY never bound) — splits commit-dependent vs forward-shape effects.
- **Eager-vs-captured bisect** — splits class 6 from logic bugs.
- **Lockstep drafter comparison** (identical committed prefixes) + per-depth accept tables (tree vs native measured on the same prompts).
- **Branch-token oracle** `scripts/fr13_branch_token_oracle.py` (native-on-path; prefill-shaped — has near-tie noise, live-vs-live is binding).
- **Capture-payload harness** (`FR10_TREE_GDN_CAPTURE_PAYLOAD` :2683-2739) — saves h0+k/v/raw_a/raw_b = the byte-A/B vehicle.
- **Pinned prompts** `output/fr13_acceptance_ladder/prompts_swe4.json` (capture-once rule) + reference arms in `output/fr13_s1s2s3_discriminate/` + `output/fr13_convfix_ab/`.
- **Gate-transfer matrix** `FR13_REPLAY_GATE_TRANSFER_MATRIX.md` — which evidence transfers vs re-runs; drift risks R1-R10.
- **Kernel lineage** `FR13_GDN_KERNEL_LINEAGE.md` (stop+report on factual changes).

## Part 3 — Standing rules for every workflow prompt
1. Quote the relevant bestiary rows in the agent prompt (at minimum: classes 6/7/8/9 for anything live; 10 for any kernel; 12 for any reduce).
2. First gate of any live campaign = B=1 same-seed bit-identical repeat (class 8).
3. Engagement asserts before any number (class 9); flag-state+seeds header in every artifact.
4. Adversarial verify phase always; holds=false ⇒ do not act on the synthesis.
5. Reuse Part-2 instruments by path; extending them is fine, re-implementing inline is not (`FR13_MEASURE_HARNESS.md`).
