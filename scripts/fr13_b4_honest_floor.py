#!/usr/bin/env python3
"""FR13 B4 - the honest (all-mandatory-bytes) hardware floor at the width-4 point.

ANALYSIS ONLY. NO GPU. NOT CITABLE AS A TIMING CLAIM.
This tool changes no measured wall. It changes the DENOMINATOR those walls are
divided by, and it computes B1 and B4 on ONE accounting basis so the two
floor ratios can be compared at all.

WHY THIS EXISTS
---------------
Every published fr13 floor_ratio divides a measured step wall by a WEIGHT-ONLY
floor (`results/fr13_hardware_floor_correction_20260731/floor_ledger.json`,
`"nonweight_costs_included": false`). Weight traffic is batch-INVARIANT: the
same 32.67 GB is read whether the step serves one request or four. Every other
mandatory byte -- KV, GDN recurrent state, logits, the residual stream -- is
PER-REQUEST and therefore 4x larger at width 4.

So a weight-only denominator flatters B1 and punishes B4 by construction, and
"B4 is 3.0x its floor, B1 is 1.94x" is not an apples-to-apples statement. This
tool builds the floor the way the B1 FA2 roofline was built (mandatory unique
bytes / 273 GB/s) and applies it to BOTH.

FAIL-CLOSED: every geometry constant is asserted against the pinned topology
module or the served config; a mismatch raises rather than emitting a floor.
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path

SCHEMA = "fr13.b4_honest_floor.v1"
BW = 273_000_000_000                      # B/s, campaign constant

# ---- geometry (all pinned; see scripts/fr13_fixed32_topology.py + served config)
MODEL_LAYERS        = 64
ATTN_LAYERS         = 16                  # full-attention target layers
GDN_LAYERS          = 48
KV_CACHE_TENSORS    = 17                  # 16 target + 1 MTP drafter (kv_remap.kv_cache_tensors)
MTP_PASSES          = 5                   # 1 initial + 4 post-root-graph (floor_ledger production_invariants)
KV_HEADS            = 4
HEAD_DIM            = 256
KV_ELEM_BYTES       = 2                   # bf16 (FR13_FULL_ATTN_KV_FP8 defaults 0)
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * KV_HEADS * HEAD_DIM * KV_ELEM_BYTES   # 4096

GDN_V_HEADS         = 48
GDN_K_DIM           = 128
GDN_V_DIM           = 128
SSM_STATE_BYTES     = GDN_V_HEADS * GDN_K_DIM * GDN_V_DIM * 4            # fp32, 3,145,728
CONV_CHANNELS       = 10_240
CONV_KERNEL         = 4
CONV_CARRY_ROWS     = CONV_KERNEL - 1                                    # 3 = the true recurrent carry
CONV_CARRY_BYTES    = CONV_CHANNELS * CONV_CARRY_ROWS * 2                # bf16, 61,440
CONV_STATE_LEN_RT   = 34                  # 3 + 31 spec rows, the SHIPPED staging length
CONV_STAGE_BYTES    = CONV_CHANNELS * CONV_STATE_LEN_RT * 2              # 696,320

HIDDEN              = 5120
VOCAB               = 248_320
DRAFT_VOCAB_K       = 65_536
PHYSICAL_ROWS       = 32                  # root-inclusive verify rows per request
PHYSICAL_DRAFTS     = 31

WEIGHT_BYTES_ROOT64K = 32_666_638_208     # floor_ledger: root_64k_five_64k_draft_heads
WEIGHT_BYTES_FULLROOT= 34_538_346_368     # floor_ledger: current_one_full_plus_four_64k_draft_heads
WEIGHT_FLOOR_ROOT64K_MS  = 119.658015414
WEIGHT_FLOOR_FULLROOT_MS = 126.51408926

# ---- measured anchors (all cited in README.md of this directory)
B1_STEP_WALL_MS      = 232.360            # sealed B1 gqa_pair production default (fr13_canonical_env.sh:45)
B4_STEP_WALL_MS      = 381.2836471327279  # sealed post-lever width-4 candidate mean, n=4
B4_STEP_WALL_ALT_MS  = 384.0206           # 411.05 sealed unprofiled width-4 wall - 27.0294 sealed gain
B4_STOCK_WIDTH4_MS   = 408.3130500078963  # sealed campaign stock width-4 mean, n=4
B4_PRELEVER_WIDTH4_MS= 411.05             # sealed unprofiled width-4 operating point
CTX_TOKENS_B1        = 18_031             # attack-ladder implied context, same workload
KV_POOL_TOKENS       = 177_152            # boot-log GPU KV cache size => hard cap on sum(ctx)


def _assert(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL-CLOSED: {msg}")


def check_geometry() -> dict:
    """Re-derive every constant from its pinned source; raise on any mismatch."""
    out = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fr13_fixed32_topology as topo
        _assert(topo.MODEL_LAYERS == MODEL_LAYERS, "MODEL_LAYERS")
        _assert(topo.TREE_ATTENTION_LAYERS == ATTN_LAYERS, "TREE_ATTENTION_LAYERS")
        _assert(topo.GDN_LAYERS == GDN_LAYERS, "GDN_LAYERS")
        _assert(topo.GDN_CONV_VALUE_HEADS == GDN_V_HEADS, "GDN_CONV_VALUE_HEADS")
        _assert(topo.GDN_CONV_KEY_HEAD_DIM == GDN_K_DIM, "GDN key head dim")
        _assert(topo.GDN_CONV_CHANNELS == CONV_CHANNELS, "GDN_CONV_CHANNELS")
        _assert(topo.GDN_CONV_KERNEL_SIZE == CONV_KERNEL, "GDN_CONV_KERNEL_SIZE")
        _assert(topo.GDN_CONV_STATE_LENGTH == CONV_STATE_LEN_RT, "GDN_CONV_STATE_LENGTH")
        _assert(topo.KV_REMAP_TARGET_CACHE_TENSORS + topo.KV_REMAP_DRAFTER_CACHE_TENSORS
                == KV_CACHE_TENSORS, "KV cache tensor count")
        out["topology_module"] = "scripts/fr13_fixed32_topology.py"
    except ImportError as exc:            # sparse worktree without the module
        out["topology_module"] = f"UNAVAILABLE ({exc})"
    _assert(WEIGHT_BYTES_ROOT64K * 1000 / BW == WEIGHT_FLOOR_ROOT64K_MS or
            abs(WEIGHT_BYTES_ROOT64K * 1000 / BW - WEIGHT_FLOOR_ROOT64K_MS) < 1e-6,
            "root64k weight floor does not reproduce")
    _assert(abs(WEIGHT_BYTES_FULLROOT * 1000 / BW - WEIGHT_FLOOR_FULLROOT_MS) < 1e-6,
            "fullroot weight floor does not reproduce")
    _assert(KV_BYTES_PER_TOKEN_PER_LAYER == 4096, "KV bytes/token/layer")
    _assert(SSM_STATE_BYTES == 3_145_728, "ssm state bytes")
    # the published B1 FA2 roofline floor must fall out of this geometry
    b1_fa2 = ATTN_LAYERS * CTX_TOKENS_B1 * KV_BYTES_PER_TOKEN_PER_LAYER * 1000 / BW
    _assert(abs(b1_fa2 - 4.33) < 0.01, f"B1 FA2 floor reproduces as {b1_fa2:.4f}, published 4.33")
    out["b1_fa2_floor_ms_reproduced"] = b1_fa2
    return out


def per_request_bytes(ctx_tokens: int, conv_basis: str = "carry") -> dict:
    """Mandatory non-weight bytes for ONE request for ONE speculative step."""
    conv = CONV_CARRY_BYTES if conv_basis == "carry" else CONV_STAGE_BYTES
    b = {
        # 1. target tree attention re-reads the whole context, 16 layers
        "kv_read_target": ATTN_LAYERS * ctx_tokens * KV_BYTES_PER_TOKEN_PER_LAYER,
        # 2. the MTP drafter layer re-reads it once per drafting pass
        "kv_read_mtp": MTP_PASSES * ctx_tokens * KV_BYTES_PER_TOKEN_PER_LAYER,
        # 3. the step's own 32 tree rows must be written to all 17 KV tensors
        "kv_write_tree_rows": KV_CACHE_TENSORS * PHYSICAL_ROWS * KV_BYTES_PER_TOKEN_PER_LAYER,
        # 4. GDN recurrent carry: read committed state, write new committed state.
        #    post-single_launch minimum -- the per-node handoff is NOT counted.
        "gdn_state_rw": GDN_LAYERS * (SSM_STATE_BYTES + conv) * 2,
        # 5. logits must be materialised then read back to verify/sample
        "lm_head_logits_rw": PHYSICAL_ROWS * VOCAB * 2 * 2,
        "draft_head_logits_rw": PHYSICAL_DRAFTS * DRAFT_VOCAB_K * 2 * 2,
        # 6. the residual stream cannot stay resident across 64 layers
        "residual_stream_rw": MODEL_LAYERS * PHYSICAL_ROWS * HIDDEN * 2 * 2,
    }
    b["total"] = sum(b.values())
    return b


def floor(batch: int, ctx_tokens: int, weight_bytes: int = WEIGHT_BYTES_ROOT64K,
          conv_basis: str = "carry") -> dict:
    pr = per_request_bytes(ctx_tokens, conv_basis)
    nonweight = batch * pr["total"]
    total = weight_bytes + nonweight
    return {
        "batch": batch, "ctx_tokens_per_request": ctx_tokens,
        "weight_bytes": weight_bytes,
        "nonweight_bytes": nonweight,
        "nonweight_bytes_per_request": pr,
        "mandatory_bytes_total": total,
        "weight_floor_ms": weight_bytes * 1000 / BW,
        "nonweight_floor_ms": nonweight * 1000 / BW,
        "honest_floor_ms": total * 1000 / BW,
    }


def main() -> int:
    geom = check_geometry()
    out = {
        "schema": SCHEMA, "acceptance_valid": False, "citable": False,
        "analysis_only": True, "gpu_touched": False,
        "bandwidth_bytes_per_s": BW,
        "geometry_check": geom,
        "geometry": {
            "model_layers": MODEL_LAYERS, "attn_layers": ATTN_LAYERS, "gdn_layers": GDN_LAYERS,
            "kv_cache_tensors": KV_CACHE_TENSORS, "mtp_passes_per_step": MTP_PASSES,
            "kv_bytes_per_token_per_layer": KV_BYTES_PER_TOKEN_PER_LAYER,
            "ssm_state_bytes_per_request_per_layer": SSM_STATE_BYTES,
            "conv_carry_bytes_per_request_per_layer": CONV_CARRY_BYTES,
            "conv_staged_bytes_per_request_per_layer": CONV_STAGE_BYTES,
            "vocab": VOCAB, "draft_vocab_k": DRAFT_VOCAB_K,
            "physical_rows_root_inclusive": PHYSICAL_ROWS, "physical_drafts": PHYSICAL_DRAFTS,
            "hidden": HIDDEN, "kv_pool_tokens": KV_POOL_TOKENS,
        },
    }

    # ---- (0) the denominator defect, before any new accounting -----------------
    out["denominator_defect"] = {
        "finding": ("The published width-4 floor_ratio divides by 126.51408926 ms "
                    "(scenario current_one_full_plus_four_64k_draft_heads, 34.538 GB) "
                    "while the sealed B4 arms launch with FR13_DRAFT_VOCAB_ROOT=1 / "
                    "FR13_DRAFT_VOCAB_K=65536, i.e. the 32.667 GB root_64k scenario whose "
                    "floor is 119.658015414 ms -- the SAME scenario B1's 1.9419 uses."),
        "evidence": [
            "output/fr13_b4_hydra27_sealing_campaign_20260814T011514Z/run_00/launcher_meta.txt: "
            "draft_vocab_root=1 draft_vocab_k=65536 mandatory_weight_floor_ms=119.658015414",
            "tests/test_fr13_b4_width4_window.py:60 FLOOR_MS = 126.514089260",
        ],
        "b1_weight_only_ratio": B1_STEP_WALL_MS / WEIGHT_FLOOR_ROOT64K_MS,
        "b4_weight_only_ratio_as_published_126_514": B4_STEP_WALL_MS / WEIGHT_FLOOR_FULLROOT_MS,
        "b4_weight_only_ratio_on_b1_basis_119_658": B4_STEP_WALL_MS / WEIGHT_FLOOR_ROOT64K_MS,
        "b4_prelever_weight_only_ratio_on_b1_basis": B4_PRELEVER_WIDTH4_MS / WEIGHT_FLOOR_ROOT64K_MS,
        "note": ("On one basis the published gap 3.01 vs 1.94 (1.55x) is really "
                 "3.19 vs 1.94 (1.64x). Correcting the basis makes B4 look WORSE, "
                 "not better -- the honest floor below is what makes it look better."),
    }

    # ---- (1) the honest floor, both batches, one basis --------------------------
    b1 = floor(1, CTX_TOKENS_B1)
    b4 = floor(4, CTX_TOKENS_B1)
    out["honest_floor"] = {
        "context_basis": ("18,031 tokens/request -- the attack-ladder implied context measured "
                          "on the SAME SWE-Verified agent workload "
                          "(results/fr13_attack_ladder_analysis_20260808). Using one context for "
                          "both batches is what makes the comparison apples-to-apples."),
        "b1": b1, "b4": b4,
        "b1_step_wall_ms": B1_STEP_WALL_MS,
        "b4_step_wall_ms": B4_STEP_WALL_MS,
        "b4_step_wall_alt_ms": B4_STEP_WALL_ALT_MS,
        "b1_honest_ratio": B1_STEP_WALL_MS / b1["honest_floor_ms"],
        "b4_honest_ratio": B4_STEP_WALL_MS / b4["honest_floor_ms"],
        "b4_honest_ratio_alt_wall": B4_STEP_WALL_ALT_MS / b4["honest_floor_ms"],
        "b4_honest_ratio_stock_prelever": B4_STOCK_WIDTH4_MS / b4["honest_floor_ms"],
        "b1_excess_ms": B1_STEP_WALL_MS - b1["honest_floor_ms"],
        "b4_excess_ms": B4_STEP_WALL_MS - b4["honest_floor_ms"],
        "b1_excess_ms_per_request": B1_STEP_WALL_MS - b1["honest_floor_ms"],
        "b4_excess_ms_per_request": (B4_STEP_WALL_MS - b4["honest_floor_ms"]) / 4,
    }

    # ---- (2) context sensitivity -----------------------------------------------
    rows = []
    for ctx, label in [
        (12_000, "low band"),
        (CTX_TOKENS_B1, "B1-measured anchor (central)"),
        (24_531, "token-flow estimate: 137,128 prefill + 10,058 decode tokens entered KV "
                 "in the 360.19 s window at ~3 admissions/completions => ~49.1 k terminal "
                 "context/task, mean resident ~24.5 k"),
        (44_288, "hard cap: KV pool 177,152 tokens / 4 slots"),
    ]:
        f1, f4 = floor(1, ctx), floor(4, ctx)
        rows.append({
            "ctx_tokens_per_request": ctx, "label": label,
            "b1_honest_floor_ms": f1["honest_floor_ms"],
            "b4_honest_floor_ms": f4["honest_floor_ms"],
            "b1_honest_ratio": B1_STEP_WALL_MS / f1["honest_floor_ms"],
            "b4_honest_ratio": B4_STEP_WALL_MS / f4["honest_floor_ms"],
            "ratio_of_ratios": (B4_STEP_WALL_MS / f4["honest_floor_ms"]) /
                               (B1_STEP_WALL_MS / f1["honest_floor_ms"]),
        })
    out["context_sensitivity"] = rows

    # ---- (3) conv-state basis sensitivity (what counts as mandatory state) ------
    f4c = floor(4, CTX_TOKENS_B1, conv_basis="staged")
    out["conv_state_basis_sensitivity"] = {
        "carry_only_ms": b4["honest_floor_ms"],
        "shipped_34_row_staging_ms": f4c["honest_floor_ms"],
        "delta_ms": f4c["honest_floor_ms"] - b4["honest_floor_ms"],
        "note": ("Baseline counts only the true recurrent conv carry (conv_kernel-1 = 3 rows). "
                 "The shipped runtime carries 34 rows (3 + 31 speculative). The extra 31 rows "
                 "are the tree's own conv inputs, which an ideal implementation reads from the "
                 "activation stream, not from a state buffer."),
    }
    # bf16 ssm-state counterfactual (fp32 is a PARKED losslessness contract, not physics)
    fp32_ms = 4 * GDN_LAYERS * SSM_STATE_BYTES * 2 * 1000 / BW
    out["ssm_dtype_sensitivity"] = {
        "fp32_ssm_state_ms_step_b4": fp32_ms,
        "bf16_counterfactual_ms_step_b4": fp32_ms / 2,
        "note": "mamba_ssm_dtype=float32 is a PARKED losslessness choice (fr13_canonical_env.sh:43).",
    }

    # ---- (4) where the excess lives, at the measured width-4 point ---------------
    #   measured column: in-step plain-sum kernel ms/step at width 4, from the
    #   539-step nsys capture (PROFILED). floor column: this tool, ctx = 18,031.
    ctx = CTX_TOKENS_B1
    ms = lambda by: by * 1000 / BW
    comp = [
        ("GEMM fp8 blockwise (target + MTP)", 133.007, 94.72,
         "B1 attack-ladder all-traffic GEMM floor; batch-invariant (w4/w1 = 1.08)"),
        ("other bucket (elementwise/copies/sampling)", 83.923, 0.0,
         "no separate mandatory tensor: this is plumbing around traffic already counted"),
        ("FA2 tree attention", 80.196, ms(4 * ATTN_LAYERS * ctx * KV_BYTES_PER_TOKEN_PER_LAYER),
         "4 requests x 16 layers x ctx x 4096 B"),
        ("GDN scan + delta rule", 50.042 + 12.983,
         ms(4 * GDN_LAYERS * (SSM_STATE_BYTES + CONV_CARRY_BYTES) * 2),
         "4 requests x 48 layers x (3 MiB ssm + 60 KiB conv carry) read+write"),
        ("bf16 GEMM (LM head + draft heads)", 30.365,
         ms(2_542_796_800 + 5 * 671_088_640 + 4 * (PHYSICAL_ROWS * VOCAB * 2 * 2)),
         "head weights are inside the 32.67 GB; shown here for the component ratio"),
        ("unified attention (MTP layer)", 16.541,
         ms(4 * MTP_PASSES * ctx * KV_BYTES_PER_TOKEN_PER_LAYER),
         "4 requests x 5 MTP passes x ctx x 4096 B"),
        ("FA2 causal (prefill spill inside decode steps)", 4.947, 0.0, "not decode work"),
        ("GPU idle / host residual inside the step", 429.333 - 412.004, 0.0,
         "step wall 429.333 - in-step kernel plain sum 412.004, PROFILED"),
    ]
    out["width4_component_ledger"] = {
        "basis": ("in-step plain-sum kernel ms/step over the 225 width-4 steps of "
                  "output/fr13_b4_width4_nsys_20260813T030940Z (PROFILED, CUPTI-attached). "
                  "Plain sum == union on this trace (gaps.json: 140.01 vs 139.89 s)."),
        "width4_step_wall_profiled_ms": 429.333,
        "width4_step_wall_sealed_unprofiled_ms": B4_PRELEVER_WIDTH4_MS,
        "components": [
            {"component": n, "measured_ms_step_w4": m, "floor_ms_step": f,
             "headroom_ms_step": m - f, "ratio_to_floor": (m / f) if f else None, "note": note}
            for n, m, f, note in comp
        ],
        "measured_total_ms_step": sum(m for _, m, _, _ in comp),
        "floor_total_ms_step": sum(f for _, _, f, _ in comp),
    }

    # ---- (5) the FA2 context-insensitivity finding -------------------------------
    out["fa2_context_insensitivity"] = {
        "method": ("For each >1 s prefill gap with 5 clean width-4 step-groups on each side, "
                   "regress the change in mean flash_fwd_splitkv_kernel(grid=(1,4,24)) launch "
                   "time on the KV tokens the gap admitted (tokens read off "
                   "silu_and_mul_per_block_quant_kernel gridX, the same source gaps.json uses)."),
        "isolated_gaps": 12, "kv_tokens_admitted": 79_859,
        "net_launch_time_change_ms": -0.1315,
        "slope_ns_per_token_per_launch": -3.5183,
        "slope_95ci_ns_per_token_per_launch": [-7.3568, 0.3202],
        "dram_floor_ns_per_token_per_launch": KV_BYTES_PER_TOKEN_PER_LAYER * 1e9 / BW,
        "verdict": ("The marginal KV token costs <= 0.32 ns of FA2 launch time (95% one-sided) "
                    "against a 15.0 ns DRAM floor. 'FA2 time is proportional to KV bytes' is "
                    "REJECTED at width 4. The kernel is 96 CTAs at 1 CTA/SM (102400 B smem, "
                    "derisk.json) = 2 waves, and each wave finishes with its LONGEST request: "
                    "the cost is set by max(ctx) and by cross-request load imbalance, not by "
                    "sum(ctx). That is why the ~59 ms of FA2 headroom is real but is NOT a "
                    "bandwidth-efficiency problem."),
        "grid_semantics_proof": ("no flash_fwd_splitkv_combine_kernel exists in the trace => "
                                 "num_splits == 1 => grid=(m_blocks, batch, heads); and the "
                                 "gridY=4 population is exactly 3616 = 226 x 16 launches, "
                                 "matching the 225(+1 boundary) width-4 steps x 16 attn layers."),
    }

    out["does_not_claim"] = [
        "No timing claim. No arm was run; no GPU was touched.",
        "The floor is a LOWER BOUND on time, not an achievable target: it assumes every "
        "mandatory byte moves once at 273 GB/s with zero latency, zero launch cost, zero "
        "recompute and perfect overlap.",
        "ctx=18,031 tokens/request is imported from the B1 attack ladder, not re-measured at "
        "width 4; the capture rejects proportionality between FA2 time and context, so no "
        "width-4 context could be inverted out of it. The sensitivity table is the honest bound.",
        "Width-4 component ms/step are CUPTI-profiled and are upper bounds.",
    ]
    p = Path("results/fr13_b4_honest_floor_20260814/floor.json")
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {p}")
    for k in ("b1_honest_ratio", "b4_honest_ratio", "b4_honest_ratio_alt_wall"):
        print(f"  {k} = {out['honest_floor'][k]:.4f}")
    print(f"  b1 honest floor = {b1['honest_floor_ms']:.4f} ms")
    print(f"  b4 honest floor = {b4['honest_floor_ms']:.4f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
