from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.fr13_hardware_floor_ledger import (
    INITIAL_MTP_FORWARD_PASSES,
    MODEL_ROOT,
    MTP_FORWARD_BYTES_PER_PASS,
    MTP_FORWARD_PASSES,
    POST_ROOT_GRAPH_MTP_FORWARD_PASSES,
    build_ledger,
    verify_pinned_constants,
)
from scripts.fr13_fixed32_topology import (
    MTP_FORWARD_CALLS as CENSUS_POST_ROOT_MTP_FORWARD_CALLS,
)


REPO = Path(__file__).resolve().parents[1]
CURATED = (
    REPO
    / "results"
    / "fr13_fixed32_b1_nsys_20260731T013952Z_curated"
)
CORRECTION = REPO / "results" / "fr13_hardware_floor_correction_20260731"
FR14 = REPO / "results" / "fr14_nvfp4_port_20260816"
HISTORICAL_LEDGER_SHA256 = (
    "3507bc71b5b7cf72d090de00d45b03a736493121be0a1680e4db19519714c8b4"
)


def test_historical_four_pass_floor_artifact_remains_immutable() -> None:
    historical_bytes = (CURATED / "floor_ledger.json").read_bytes()
    published = json.loads(historical_bytes)
    summary = json.loads(
        (CURATED / "diagnostic_summary.json").read_text(encoding="ascii")
    )

    assert sha256(historical_bytes).hexdigest() == HISTORICAL_LEDGER_SHA256
    assert published["schema"] == "fr13.speculative_step_weight_ledger.v1"
    assert published["components"]["mtp_forward"]["passes"] == 4
    scenarios = published["scenarios"]
    floor = summary["floor_accounting"]

    legacy = scenarios["legacy_target_plus_verifier_head"]
    assert legacy["mandatory_weight_bytes"] == floor[
        "legacy_exact_tensor_ledger"
    ]["bytes"]
    assert legacy["mandatory_weight_floor_ms"] == floor[
        "legacy_exact_tensor_ledger"
    ]["floor_ms"]

    current = scenarios["current_one_full_plus_four_64k_draft_heads"]
    assert current["drafter_head_bytes"] == floor["current_algorithm"][
        "drafter_head_bytes"
    ]
    assert current["mandatory_weight_bytes"] == floor["current_algorithm"][
        "mandatory_weight_bytes_per_step"
    ]
    assert current["mandatory_weight_floor_ms"] == floor[
        "current_algorithm"
    ]["mandatory_weight_floor_ms"]

    root_64k = scenarios["root_64k_five_64k_draft_heads"]
    assert root_64k["mandatory_weight_bytes"] == floor[
        "root_64k_projection"
    ]["mandatory_weight_bytes_per_step"]
    assert root_64k["mandatory_weight_floor_ms"] == floor[
        "root_64k_projection"
    ]["mandatory_weight_floor_ms"]


def test_weight_ledger_counts_initial_and_post_root_mtp_forwards() -> None:
    generated = build_ledger()
    invariants = generated["production_invariants"]
    mtp = generated["components"]["mtp_forward"]

    assert INITIAL_MTP_FORWARD_PASSES == 1
    assert POST_ROOT_GRAPH_MTP_FORWARD_PASSES == 4
    assert (
        POST_ROOT_GRAPH_MTP_FORWARD_PASSES
        == CENSUS_POST_ROOT_MTP_FORWARD_CALLS
    )
    assert MTP_FORWARD_PASSES == 5
    assert invariants == {
        "initial_mtp_forward_passes_per_event": 1,
        "post_root_graph_mtp_forward_passes_per_event": 4,
        "total_mtp_forward_passes_per_event": 5,
        "drafter_head_passes_per_event": 5,
    }
    assert mtp["initial_passes"] == 1
    assert mtp["post_root_graph_passes"] == 4
    assert mtp["passes"] == 5
    assert mtp["bytes"] == 5 * MTP_FORWARD_BYTES_PER_PASS


def test_historical_v2_correction_artifact_remains_immutable() -> None:
    """The v2 (fp8, Qwen3.6) correction artifact is now history too.

    FR14 re-derived the ledger against the NVFP4 checkpoint, so build_ledger()
    no longer reproduces this artifact -- it reproduces the FR14 one (see
    test_fr14_weight_ledger_matches_published_artifact). What survives is the
    artifact's own internal consistency and its self-declared numbers, so a
    later edit still cannot quietly rewrite the fp8-era record.
    """
    published_bytes = (CORRECTION / "floor_ledger.json").read_bytes()
    published = json.loads(published_bytes)
    manifest = json.loads(
        (CORRECTION / "correction_manifest.json").read_text(
            encoding="ascii"
        )
    )

    assert published["schema"] == "fr13.speculative_step_weight_ledger.v2"
    assert sha256(published_bytes).hexdigest() == manifest["artifacts"][
        "corrected_floor_ledger_sha256"
    ]
    assert manifest["historical_artifact"]["sha256"] == (
        HISTORICAL_LEDGER_SHA256
    )
    # The fp8-era MTP forward pass: 477_199_744 B. FR14's BF16 MTP shard is
    # 849_398_784 B/pass, which is exactly why the floor did not halve.
    assert manifest["correction"]["added_mandatory_bytes_per_event"] == (
        477_199_744
    )
    assert published["components"]["mtp_forward"]["bytes_per_pass"] == (
        477_199_744
    )
    assert published["production_invariants"] == {
        "initial_mtp_forward_passes_per_event": INITIAL_MTP_FORWARD_PASSES,
        "post_root_graph_mtp_forward_passes_per_event": (
            POST_ROOT_GRAPH_MTP_FORWARD_PASSES
        ),
        "total_mtp_forward_passes_per_event": MTP_FORWARD_PASSES,
        "drafter_head_passes_per_event": MTP_FORWARD_PASSES,
    }

    root_64k = published["scenarios"]["root_64k_five_64k_draft_heads"]
    assert root_64k["mandatory_weight_bytes"] == 32_666_638_208
    assert root_64k["mandatory_weight_floor_ms"] == 119.658015414
    assert manifest["corrected_root64_acceptance"]["floor_ms"] == (
        root_64k["mandatory_weight_floor_ms"]
    )
    assert manifest["corrected_root64_acceptance"]["one_sided_u95_cap_ms"] == (
        137.606717726
    )

    nsys = manifest["nsys_confirmation"]
    assert nsys["draft_head_kernel_instances"] == (
        nsys["dfwd_range_instances"] * MTP_FORWARD_PASSES
    )
    assert nsys["mtp_gemm_instances"] == (
        nsys["dfwd_range_instances"]
        * MTP_FORWARD_PASSES
        * nsys["mtp_gemms_per_forward"]
    )


def test_fr14_weight_ledger_matches_published_artifact() -> None:
    """The LIVE ledger is the FR14 ARM-B one and it is bound to its artifact.

    Arm A's ledger (floor_ledger.json, conservative unsloth bytes) stays banked
    in the same directory as the other half of the bytes ablation; the live
    binding moved to floor_ledger_radixark.json when the constant train
    re-pointed the stack, exactly as the FR13->FR14 train moved it off
    results/fr13_hardware_floor_correction_20260731/floor_ledger.json.
    """
    generated = build_ledger()
    published = json.loads((FR14 / "floor_ledger_radixark.json").read_bytes())

    assert published == generated
    assert generated["schema"] == "fr13.speculative_step_weight_ledger.v3"
    assert generated["model_root"] == "/models/qwen3.8-27b-nvfp4-radixark"
    assert MTP_FORWARD_BYTES_PER_PASS == 849_398_784

    root_64k = generated["scenarios"]["root_64k_five_64k_draft_heads"]
    assert root_64k["mandatory_weight_bytes"] == 25_210_209_416
    assert root_64k["mandatory_weight_floor_ms"] == 92.345089436
    # Still not 0.5x of FR13 even with the aggressive recipe: 0.772x. The BF16
    # MTP head grew and PHASE 1 keeps the five K64 draft-head reads in BF16.
    # Pin the ratio so a future "NVFP4 halves it" edit cannot land quietly.
    assert 0.77 < root_64k["mandatory_weight_floor_ms"] / 119.658015414 < 0.78

    # THE ABLATION ITSELF: arm A is banked beside us and must remain the
    # conservative-bytes record, so the delta this campaign measures cannot be
    # rewritten by editing one file.
    arm_a = json.loads((FR14 / "floor_ledger.json").read_bytes())
    arm_a_root = arm_a["scenarios"]["root_64k_five_64k_draft_heads"]
    assert arm_a["model_root"] == "/models/qwen3.8-27b-nvfp4"
    assert arm_a_root["mandatory_weight_bytes"] == 27_977_022_848
    assert arm_a_root["mandatory_weight_floor_ms"] == 102.479937172
    assert (
        arm_a_root["mandatory_weight_bytes"]
        - root_64k["mandatory_weight_bytes"]
    ) == 2_766_813_432

    # Nearly the whole delta is the HEAD, not the backbone. Guard the claim.
    head = generated["components"]["nvfp4_head"]
    assert head["bytes"] == 715_161_608
    assert head["bf16_head_reference_bytes"] == 248_320 * 5_120 * 2
    assert head["bf16_head_reference_bytes"] - head["bytes"] == 1_827_635_192


def test_fr14_phase2_projection_is_labelled_not_pinned() -> None:
    """The FP4 draft-head read is a projection; nothing may quote it as a floor."""
    generated = build_ledger()
    phase2 = generated["projected_scenarios"]["root_64k_five_nvfp4_draft_heads"]

    assert phase2["mandatory_weight_bytes"] == 22_798_484_616
    assert phase2["mandatory_weight_floor_ms"] == 83.510932659
    assert phase2["draft_64k_nvfp4_head_bytes"] == 65_536 * 2_560 + 65_536 * 320
    assert "PROJECTION, NOT PINNED" in phase2["status"]
    # It must NOT be reachable as a live scenario.
    assert "root_64k_five_nvfp4_draft_heads" not in generated["scenarios"]


def test_fr14_ledger_reproduces_by_summing_the_checkpoint() -> None:
    """The pinned bytes must be re-derivable from the served files.

    Skipped when the checkpoint is not mounted (CPU-only CI); when it IS
    present this is the measured-never-derived gate: every byte term is a sum
    of real safetensors tensor spans, not a scaled FR13 constant.
    """
    if not MODEL_ROOT.is_dir():
        pytest.skip(f"served checkpoint not present: {MODEL_ROOT}")
    report = verify_pinned_constants(MODEL_ROOT)
    assert report["problems"] == []
    assert report["pass"] is True
    assert report["target_model_bytes"] == 16_892_610_688
    assert report["mtp_forward_bytes_per_pass"] == 849_398_784
    # The verifier head is the 4-tensor NVFP4 set as shipped -- no lm_head
    # surgery in arm B. If this ever equals the BF16 product the head silently
    # got dequantised and 6.7 ms of the floor is fiction.
    assert report["lm_head_bytes_on_disk"] == 715_161_608
    assert report["lm_head_tensor_count"] == 4
    assert report["lm_head_bytes_on_disk"] != 248_320 * 5_120 * 2

    # The drafter is found BY NAME across the three shards: RadixArk puts the
    # 15 mtp.* tensors inside model-00003-of-00003.safetensors, so a
    # filename-keyed ledger would have missed them entirely.
    assert report["mtp_tensor_count"] == 15
    assert report["shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    # Whole-ledger cross-check against RadixArk's own qualification.json
    # output_indexed_payload_bytes: catches a truncated shard that happens to
    # leave one bucket intact.
    assert report["checkpoint_total_tensor_bytes"] == 21_921_428_072
    assert report["checkpoint_total_tensor_count"] == 2_194


def test_curated_attribution_binds_source_and_historical_lifecycle() -> None:
    summary = json.loads(
        (CURATED / "diagnostic_summary.json").read_text(encoding="ascii")
    )
    attribution_bytes = (CURATED / "nsys_attribution.json").read_bytes()
    attribution = json.loads(attribution_bytes)
    provenance = summary["provenance"]

    assert provenance["measured_git_sha"] == (
        "1a7a765447c8ce6068e0dd5d3a344d58ace85f2b"
    )
    assert provenance["attribution_json_sha256"] == sha256(
        attribution_bytes
    ).hexdigest()
    assert provenance["runtime_manifest_sha256"] == attribution[
        "provenance"
    ]["runtime_manifest"]["sha256"]
    assert provenance["exact4_subset_sha256"] == attribution["provenance"][
        "exact4_subset"
    ]["sha256"]
    assert (
        summary["publication"][
            "capture_exercised_terminal_ledger_snapshot_fix"
        ]
        is False
    )
    assert summary["publication"]["acceptance_valid"] is False
