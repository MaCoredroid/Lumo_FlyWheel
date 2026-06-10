# FR13 cost-gate research — RAW workflow results (2026-06-09)

Raw, unedited outputs of the four parallel research workflows launched to price the FR13 B=4
cost-gate (lossless feasibility × speed × drift-tracking × prior-art). The distilled,
human-readable verdicts are in the top-level `FR13_*_VERDICT.md` / `FR13_DRIFT_TRACKER_DESIGN.md`;
these are the **source** results (synthesis + adversarial-verify, verbatim) so claims are auditable.

| file | workflow | verify | verdict |
|---|---|---|---|
| `why_slower_wacoxe6i2.raw.json` | wacoxe6i2 — why slower (MEASURED) | holds=True | 2.336× = 1.432× fwd × 1.632× per-fwd; WY one-pass is the per-fwd lever; native 76% saturated + 0 branch accepts → conditional/capped → `FR13_WHY_SLOWER_VERDICT.md` |
| `nondet_source_ws5783inp.raw.json` | ws5783inp — B=4 non-det source | holds=True | diffuse (state/bank-row feedback loop, not the det-by-construction kernels); 53% floor is a diff-seed artifact; 2 cheap residual probes before STOP → `FR13_NONDET_SOURCE_VERDICT.md` |
| `drift_tracking_wet1mdi93.salvaged.json` | wet1mdi93 — robust drift tracking | flow failed (1 agent socket error); **3 readers salvaged from transcripts** | multi-sample p95 floor + forced-decode logit-KL/TV + K-seed tracker → `FR13_DRIFT_TRACKER_DESIGN.md` |
| `git_archaeology_w6m7imv2j.raw.json` | w6m7imv2j — prior-fix archaeology | holds=True | GDN scan EXONERATED (beec984a/e4a6a2f2/45519178 batch-invariant by construction); real B=4 carrier is OUTSIDE the scan = TREE_ATTN reduction-order + fp8 GEMMs, BOTH addressable via the **forked-FA2 -inf-bias path (fe21cb73) under VLLM_BATCH_INVARIANT+FLASH_ATTN** (untested lever, reuses prior art). LOST seam `_fr13_tree_mamba_initial_seed_tokens` (prose-only). **Launcher trap:** `FR13_FA2_PREFILL_NATIVE` silently OFF on `fr10_launch_speed_server.sh` → any B=4 number from it is invalid; only `fr13_launch_forked_fa2_tree_server.sh` sets it ON. |

Note: `output/` is gitignored, so these were copied here (tracked) to survive context resets.

## Kernel-speed flows (the HBM-tax half)
| file | workflow | verify | verdict |
|---|---|---|---|
| `replay_no_hbm_tax_wznd54pmo.raw.json` | wznd54pmo — remove HBM tax from battle-tested replay kernel | holds=True | **ACCEPT-ONLY COMMIT = the win**: bit-IDENTICAL (rejected nodes' states are published then provably dead; only accepted path's final state survives the remap), ~2-4 days plumbing, erases +35.8% durable tax → native 1.0x HBM, keeps the proven batch-invariant kernel. Recompute-from-spine (free under 99ms floor, ~<1%) = secondary for the transient spill. DOMINATES WY on GB10. Per-node S = 3.0 MiB (not 151 MB — that's the per-seq all-layers aggregate). |
| `wy_breakthrough_wox8pnjx8.raw.json` | wox8pnjx8 — WY at the within-argmax B=4 bar | holds=True | WY failed the WRONG bar (byte-exact); within-floor residual ~1 bf16 ULP (4.19e-9 vs non-MTP = 10000× closer than shipping MTP-5). PARKED-NOT-DEAD but **dominated**: within-floor not byte-exact, 6/6 spine argmax was "single-event coincidence/Gate A never passed" (contested), FLOP advantage worthless bandwidth-bound. Pursue only if state tax empirically dominates B=4 TPS. |

**Speed verdict:** accept-only commit (low-risk, bit-identical, keeps proven kernel) is the clear HBM-tax removal — WY not needed. **SEPARATE from the B=4 lossy carrier** (both flows confirm the carrier — conv/h0 bank-row + fp8-GEMM — is OUTSIDE the GDN scan and persists under accept-only/recompute/WY alike).

## Consolidated cost-gate read (all 4 flows)
- **AGREE:** GDN scan is exonerated; B=4 carrier is outside it; tree is 2.336× slower and conditional/capped even if lossless (native 76% saturated, 0 branch accepts = drafter bug).
- **CONTESTED (key uncertainty):** ws5783inp says carrier = *diffuse* state/bank-row wiring → stop; w6m7imv2j says carrier = TREE_ATTN/fp8-GEMM *batch-variance* → fixable via the FLASH-compatible FA2 path under `VLLM_BATCH_INVARIANT`. **Untested decisive lever distinguishes them.**
- **Cheap decisive battery (~3-5 boots, reuses prior art):** (1) confirm B=4 harness launcher = forked-fa2 (prefill-native ON), else re-run; (2) forked-FA2 -inf-bias under `VLLM_BATCH_INVARIANT=1`+FLASH-compat → if B=4 non-det drops, carrier is fixable; (3) `num_splits=1` one-liner; (4) same-seed (1313v1313) native floor probe.
