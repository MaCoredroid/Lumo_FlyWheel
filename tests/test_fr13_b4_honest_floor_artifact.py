"""Pin the FR13 B4 honest-floor artifact and the arithmetic behind it.

The artifact is ANALYSIS ONLY. What is pinned here is (a) that it never claims
otherwise, (b) that the floor arithmetic reproduces from the ledger geometry
rather than from a copied literal, and (c) the two load-bearing findings, so a
later edit cannot quietly soften them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARTIFACT = Path("results/fr13_b4_honest_floor_20260814")
BW = 273_000_000_000

# floor_ledger.json scenarios -- the whole point of the artifact is that these
# two are not interchangeable.
WEIGHT_BYTES_ROOT64K = 32_666_638_208
WEIGHT_FLOOR_ROOT64K_MS = 119.658015414
WEIGHT_FLOOR_FULLROOT_MS = 126.51408926

B1_STEP_WALL_MS = 232.360  # fr13_canonical_env.sh:45, sealed gqa_pair default


def _load(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text())


def test_sha256sums_cover_every_artifact_and_verify() -> None:
    lines = (ARTIFACT / "SHA256SUMS").read_text().strip().splitlines()
    recorded = {ln.split()[1].lstrip("*"): ln.split()[0] for ln in lines}
    present = {p.name for p in ARTIFACT.iterdir() if p.name != "SHA256SUMS"}
    assert present == set(recorded), (present, set(recorded))
    for name, digest in recorded.items():
        actual = hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest()
        assert actual == digest, name


def test_artifact_never_claims_acceptance() -> None:
    for name in ("floor.json", "other_bucket.json"):
        doc = _load(name)
        assert doc["acceptance_valid"] is False, name
        assert doc["citable"] is False, name
        assert doc["does_not_claim"], name
    assert _load("floor.json")["gpu_touched"] is False


def test_weight_floor_scenarios_reproduce_from_bytes() -> None:
    assert abs(WEIGHT_BYTES_ROOT64K * 1000 / BW - WEIGHT_FLOOR_ROOT64K_MS) < 1e-6


def test_denominator_defect_is_stated_and_numerically_right() -> None:
    d = _load("floor.json")["denominator_defect"]
    assert abs(d["b1_weight_only_ratio"] - B1_STEP_WALL_MS / WEIGHT_FLOOR_ROOT64K_MS) < 1e-9
    # the published B4 ratio uses the 126.514 scenario; the arms launch the
    # 119.658 one, so the correct weight-only ratio is strictly LARGER.
    assert (d["b4_weight_only_ratio_on_b1_basis_119_658"]
            > d["b4_weight_only_ratio_as_published_126_514"])
    ratio = (d["b4_weight_only_ratio_on_b1_basis_119_658"]
             / d["b4_weight_only_ratio_as_published_126_514"])
    assert abs(ratio - WEIGHT_FLOOR_FULLROOT_MS / WEIGHT_FLOOR_ROOT64K_MS) < 1e-9
    assert "draft_vocab_root=1" in " ".join(d["evidence"])


def test_honest_floor_reproduces_the_published_b1_fa2_roofline() -> None:
    doc = _load("floor.json")
    # 4.33 ms is published in results/fr13_attack_ladder_analysis_20260808
    assert abs(doc["geometry_check"]["b1_fa2_floor_ms_reproduced"] - 4.33) < 0.01
    g = doc["geometry"]
    assert g["kv_bytes_per_token_per_layer"] == 4096
    assert g["ssm_state_bytes_per_request_per_layer"] == 3_145_728
    assert g["attn_layers"] == 16 and g["gdn_layers"] == 48
    assert g["kv_cache_tensors"] == 17


def test_honest_floor_is_internally_consistent() -> None:
    h = _load("floor.json")["honest_floor"]
    for key in ("b1", "b4"):
        f = h[key]
        assert f["mandatory_bytes_total"] == f["weight_bytes"] + f["nonweight_bytes"]
        assert abs(f["honest_floor_ms"] - f["mandatory_bytes_total"] * 1000 / BW) < 1e-9
    # width 4 pays exactly 4x the per-request non-weight traffic
    assert h["b4"]["nonweight_bytes"] == 4 * h["b1"]["nonweight_bytes"]
    # the weight term is batch-invariant
    assert h["b4"]["weight_bytes"] == h["b1"]["weight_bytes"] == WEIGHT_BYTES_ROOT64K
    assert abs(h["b1_honest_ratio"] - B1_STEP_WALL_MS / h["b1"]["honest_floor_ms"]) < 1e-9


def test_the_answer_to_the_question_survives_every_context_choice() -> None:
    """The honest gap must be smaller than the published one, at every C."""
    doc = _load("floor.json")
    published_gap = (doc["denominator_defect"]["b4_weight_only_ratio_as_published_126_514"]
                     / doc["denominator_defect"]["b1_weight_only_ratio"])
    rows = doc["context_sensitivity"]
    assert len(rows) >= 4
    for r in rows:
        assert r["ratio_of_ratios"] < published_gap, r["ctx_tokens_per_request"]
    # and it must shrink monotonically as context grows
    ctxs = [r["ctx_tokens_per_request"] for r in rows]
    gaps = [r["ratio_of_ratios"] for r in rows]
    assert ctxs == sorted(ctxs)
    assert gaps == sorted(gaps, reverse=True)


def test_fa2_context_insensitivity_finding_is_pinned() -> None:
    f = _load("floor.json")["fa2_context_insensitivity"]
    lo, hi = f["slope_95ci_ns_per_token_per_launch"]
    assert lo < 0 < hi, "the CI must straddle zero for the rejection to hold"
    # the whole finding: the upper CI bound is far below the DRAM floor
    assert hi < f["dram_floor_ns_per_token_per_launch"] / 10
    assert abs(f["dram_floor_ns_per_token_per_launch"] - 4096 * 1e9 / BW) < 1e-6
    assert f["isolated_gaps"] >= 10 and f["kv_tokens_admitted"] > 50_000


def test_other_bucket_reconciles_against_the_parent_reducer() -> None:
    doc = _load("other_bucket.json")
    o = doc["other_bucket"]
    # gaps.json publishes 26.05 s of out-of-step 'other' over the window
    assert o["out_of_step_reconciles_to_gaps_json"] is True
    assert abs(o["out_of_step_s"] - 26.048) < 0.05
    # the capture's own width histogram
    assert doc["width_histogram"] == {"1": 4, "2": 74, "3": 237, "4": 225}
    # the bucket is the second largest item at width 4, after the GEMM only
    fam = {k: v["w4"] for k, v in doc["family_ms_step_by_width"].items() if k != "other"}
    ranked = sorted(fam.items(), key=lambda kv: -kv[1])
    assert ranked[0][0] == "GEMM fp8 blockwise"
    assert ranked[1][0] == "FA2 tree"
    assert o["ms_step_w4_profiled"] > ranked[1][1], "other must outrank FA2 at width 4"
    assert o["ms_step_w4_profiled"] < ranked[0][1], "other must not outrank the GEMM"


def test_other_bucket_reducible_estimate_carries_error_bars() -> None:
    o = _load("other_bucket.json")["other_bucket"]
    lo, hi = o["reducible_ms_step_w4"]
    assert 0 < lo < hi < o["ms_step_w4_profiled"]
    cb = o["cupti_bound"]
    assert (cb["ms_step_w4_lower_bound_all_cupti_charged_here"]
            < cb["central_pro_rata"]
            <= cb["ms_step_w4_upper_bound_no_cupti_charged_here"])
    for cls in o["classes"].values():
        band = cls["reducible_band"]
        assert 0.0 <= band[0] <= band[1] <= 1.0
        assert cls["note"]


def test_sampler_math_is_recorded_as_width_invariant() -> None:
    """9.2 ms/step of sampler work on padded max-width buffers is the finding."""
    kernels = {k["short_name"]: k for k in _load("other_bucket.json")["kernels"]}
    for name in ("tensor_kernel_scan_innermost_dim", "cunn_SoftMaxForward"):
        k = kernels[name]
        assert k["class"] == "sampling_and_verification", name
        assert k["w4_over_w2"] is not None and k["w4_over_w2"] < 1.10, (name, k["w4_over_w2"])
    total = sum(kernels[n]["ms_step_w4"] for n in
                ("tensor_kernel_scan_innermost_dim", "cunn_SoftMaxForward"))
    assert total > 8.0
