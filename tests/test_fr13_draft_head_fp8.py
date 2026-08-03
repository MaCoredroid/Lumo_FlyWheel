from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
MEASURE = REPO / "scripts" / "fr13_measure.py"
TIMING_RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_fp8_timing.sh"
TIMING_REDUCER = REPO / "scripts" / "fr13_draft_head_fp8_timing.py"
RUNTIME_MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_FP8" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head FP8 replacement snippet not found")


def test_candidate_is_default_off_strict_and_orthogonal_to_cutlass_wave() -> None:
    snippet = _eagle_snippet()
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert '"FR13_DRAFT_HEAD_FP8", "0"' in snippet
    assert '_fr13_dh_fp8_raw not in ("0", "1")' in snippet
    assert "_fr13_dh_fp8," in snippet
    assert "not _fr13_is_fixed32" in snippet
    assert "not _fr13_dvk_root" in snippet
    assert "not _fr13_single_logits" in snippet
    assert "_fr13_dvk_configured != 65536" in snippet
    assert "FR13_DRAFT_HEAD_FP8=${FR13_DRAFT_HEAD_FP8:-0}" in launcher
    assert 'case "$FR13_DRAFT_HEAD_FP8" in' in launcher
    assert "FR13_FIXED32_CUTLASS_WAVE" not in snippet[
        snippet.index("_fr13_dh_fp8_raw") : snippet.index(
            "def _fr13_dvk_prepare"
        )
    ]


def test_weight_quantization_is_once_before_root_and_bound_to_sm121() -> None:
    snippet = _eagle_snippet()
    prepare = snippet[
        snippet.index("def _fr13_dvk_prepare") : snippet.index(
            "def _fr13_dh_pad_logits"
        )
    ]

    assert 'not getattr(self, "_fr13_dh_fp8_ready", False)' in prepare
    assert "tuple(torch.cuda.get_device_capability()) != (12, 1)" in prepare
    assert "CUTLASS_BLOCK_FP8_SUPPORTED" in prepare
    assert "per_block_cast_to_fp8" in prepare
    assert "block_size=[128, 128]" in prepare
    assert "use_ue8m0=False" in prepare
    assert "tuple(_fr13_dh_fp8_qw.shape) != (65536, 5120)" in prepare
    assert "tuple(_fr13_dh_fp8_qw.stride()) != (5120, 1)" in prepare
    assert "tuple(_fr13_dh_fp8_ws.shape) != (512, 40)" in prepare
    assert "tuple(_fr13_dh_fp8_ws.stride()) != (40, 1)" in prepare
    assert snippet.index("_fr13_dh_fp8_quant_weight(") < snippet.index(
        "_fr10_logits, _fr10_root_map = _fr13_dvk_logits("
    )


def test_fp8_output_directly_serves_all_draft_logits_without_bf16_shadow() -> None:
    snippet = _eagle_snippet()
    helper = snippet[
        snippet.index("def _fr13_dh_fp8_logits") : snippet.index(
            "def _fr13_dvk_logits"
        )
    ]

    assert "per_token_group_quant_fp8" in helper
    assert "column_major_scales=True" in helper
    assert "use_ue8m0=False" in helper
    assert "cutlass_scaled_mm" in helper
    assert "_fr13_dh_fp8_qw," in helper
    assert "_fr13_dh_fp8_ws," in helper
    assert "[128, 128]" in helper
    assert "torch.bfloat16" in helper
    assert "return _fr13_dh_fp8_out" in helper
    assert "quant_method.apply" not in helper
    assert "compute_logits" not in helper
    assert "proposal_logits=fp8_output_direct" in helper
    assert "bf16_shadow_calls=0" in helper
    dispatch = snippet[
        snippet.index("def _fr13_dvk_logits") : snippet.index(
            "def _fr13_dvk_real_ids"
        )
    ]
    assert "if _fr13_dh_fp8_on:" in dispatch
    assert "_logits = _fr13_dh_fp8_logits(_h)" in dispatch
    assert "BF16 fallback is forbidden" in dispatch


def test_root_plus_four_capture_engagement_has_no_synchronize() -> None:
    snippet = _eagle_snippet()
    engagement = snippet[
        snippet.index("def _fr13_dh_fp8_note_selection") : snippet.index(
            "def _fr13_dh_fp8_logits"
        )
    ]

    assert "_fr13_dh_fp8_selected_root_calls > 1" in engagement
    assert "_fr13_dh_fp8_selected_capture_calls > 4" in engagement
    assert "self._fr13_dh_fp8_selected_root_calls != 1" in engagement
    assert "self._fr13_dh_fp8_selected_capture_calls != 4" in engagement
    assert '"selected_root_calls": 1' in engagement
    assert '"captured_loop_calls": 4' in engagement
    assert '"fallback_calls": 0' in engagement
    assert '"steady_state_synchronizations": 0' in engagement
    assert "torch.cuda.synchronize" not in engagement
    assert snippet.count("_fr13_dh_fp8_note_replay(") == 3


def test_exact_fp8_traffic_floor_and_cap_math() -> None:
    vocab = 65_536
    hidden = 5_120
    calls = 5
    scale_rows = vocab // 128
    scale_cols = hidden // 128

    bf16_head_bytes = calls * vocab * hidden * 2
    fp8_head_bytes = calls * vocab * hidden
    fp32_scale_bytes = calls * scale_rows * scale_cols * 4
    candidate_head_bytes = fp8_head_bytes + fp32_scale_bytes
    candidate_full_step_bytes = 32_666_638_208 - bf16_head_bytes + candidate_head_bytes
    floor_ms = candidate_full_step_bytes / 273_000_000
    cap_ms = floor_ms * 1.15

    assert bf16_head_bytes == 3_355_443_200
    assert fp8_head_bytes == 1_677_721_600
    assert fp32_scale_bytes == 409_600
    assert candidate_head_bytes == 1_678_131_200
    assert candidate_full_step_bytes == 30_989_326_208
    assert abs(floor_ms - 113.514015414) < 1e-9
    assert abs(cap_ms - 130.541117726) < 1e-9
    snippet = _eagle_snippet()
    for value in (
        "1678131200",
        "30989326208",
        "113.514015414",
        "130.541117726",
    ):
        assert value in snippet


def test_floor_sequence_and_launcher_bind_fp8_candidate_ledger() -> None:
    sequence = SEQUENCE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_DRAFT_HEAD_FP8=${FR13_DRAFT_HEAD_FP8:-0}" in sequence
    assert "FR13_MANDATORY_WEIGHT_BYTES=30989326208" in sequence
    assert "FR13_WEIGHT_FLOOR_MS=113.514015414" in sequence
    assert "_fixed32_expected_mandatory_weight_bytes=30989326208" in launcher
    assert "_fixed32_expected_weight_floor_ms=113.514015414" in launcher
    assert '-e FR13_DRAFT_HEAD_FP8="$FR13_DRAFT_HEAD_FP8" \\' in launcher
    assert "FR13_DRAFT_HEAD_FP8_SOURCE_SHA256" in launcher
    assert "FR13_DRAFT_HEAD_FP8_SOURCE_COMMIT" in launcher

    command = f"""
set -euo pipefail
run_variant() {{ :; }}
export BSIZE=1 CONC=1 TAG=fp8_floor_test
export FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_HEAD_FP8=1
source {SEQUENCE}
printf '%s %s' "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.stdout == "30989326208 113.514015414"


def test_deploy_speed_reducer_binds_fp8_to_exact_k64_root_ledger() -> None:
    measure = MEASURE.read_text(encoding="utf-8")

    assert '(65_536, 1, 1): (' in measure
    assert "30_989_326_208" in measure
    assert "113.514015414" in measure
    assert 'os.environ.get("FR13_DRAFT_HEAD_FP8", "0")' in measure
    assert '_draft_head_fp8_raw not in {"0", "1"}' in measure
    assert '"draft_head_fp8": bool(_draft_vocab_config[2])' in measure
    assert "(0, 0, 1):" not in measure
    assert "(65_536, 0, 1):" not in measure


def test_drafter_replacement_snippet_compiles() -> None:
    compile(
        "class _C:\n    def propose(self):\n" + _eagle_snippet(),
        "<fr13_draft_head_fp8_snippet>",
        "exec",
    )


def test_real_b1_gate_validates_direct_engagement_and_acceptance() -> None:
    from scripts import fr13_draft_head_fp8_gate as gate

    source_sha = "a" * 64
    source_commit = "b" * 40
    engagement = {
        "schema": gate.SCHEMA,
        "status": "ENGAGED",
        "source_commit": source_commit,
        "candidate_source_sha256": source_sha,
        "served_batch_size": 1,
        "geometry": gate.GEOMETRY,
        "candidate": gate.CANDIDATE,
        "traffic": gate.TRAFFIC,
        "selected_root_calls": 1,
        "captured_loop_calls": 4,
        "fallback_calls": 0,
        "drafter_graph_id": 7,
        "drafter_graph_signature": gate.GRAPH_SIGNATURE,
        "observed_measured_replays_at_least": 1,
        "capture_origin": "unmeasured",
        "execution_basis": "cudagraph_replay",
        "forward_step_index": 3,
        "runtime_mode": "FULL",
        "steady_state_synchronizations": 0,
    }
    gate._validate_engagement(
        engagement, source_sha=source_sha, source_commit=source_commit
    )
    broken = json.loads(json.dumps(engagement))
    broken["candidate"]["bf16_shadow_calls"] = 1
    try:
        gate._validate_engagement(
            broken, source_sha=source_sha, source_commit=source_commit
        )
    except ValueError as error:
        assert "engagement contract drifted" in str(error)
    else:
        raise AssertionError("BF16 shadow engagement did not fail closed")

    events = 10.0
    accepted = 37.0
    acceptance = {
        "schema": "fr13.measure.deploy_speed.v1",
        "kind": "speed",
        "instrument": "OFF",
        "regime": "deployment",
        "batch_size": 1,
        "n_tasks": 1,
        "task_instance_ids": [gate.INSTANCE],
        "engagement": {
            "tok_per_draft": 31.0,
            "expected_tok_per_draft": 31.0,
            "engaged": True,
        },
        "per_task": [{"instance_id": gate.INSTANCE}],
        "mandatory_weight_bytes": 30_989_326_208,
        "weight_floor_ms": 113.514015414,
        "accept_per_event": accepted / events,
        "committed_per_event": accepted / events + 1.0,
        "raw_counter_delta_aggregate": {
            "vllm:spec_decode_num_drafts_total": events,
            "vllm:spec_decode_num_draft_tokens_total": events * 31.0,
            "vllm:spec_decode_num_accepted_tokens_total": accepted,
        },
    }
    summary = gate._validate_acceptance(acceptance)
    assert summary == {
        "events": 10,
        "accepted_drafts": 37,
        "accepted_drafts_per_event": 3.7,
        "committed_tokens_per_event": 4.7,
    }


def test_real_b1_runner_uses_canonical_task_and_only_gates_not_tunes() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert "FR13_GATE_DRAFT_HEAD_FP8=${FR13_GATE_DRAFT_HEAD_FP8:-0}" in runner
    assert "FR13_GATE_DRAFT_HEAD_FP8 must be the only enabled kernel candidate" in runner
    assert "FR13_B1_WORKLOAD_PROFILE" in runner
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in runner
    assert "astropy__astropy-12907" in runner
    assert "FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON=/logs/fr13_draft_head_fp8.engagement.json" in runner
    assert "scripts/fr13_measure.py deploy-speed" in runner
    assert "--expected-tok-per-draft 31" in runner
    assert "scripts/fr13_draft_head_fp8_gate.py" in runner
    assert "draft_head_fp8_real_b1_gate.json" in runner


def test_gate_validator_can_rebuild_a_promotion_credential() -> None:
    validator = (
        REPO / "scripts" / "fr13_draft_head_fp8_gate.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--gate-result", type=Path)' in validator
    assert 'parser.add_argument("--expected-gate-sha256")' in validator
    assert "gate_result != result" in validator
    assert "does not match rebuilt raw evidence" in validator
    assert "fr13.fixed32.draft_head_fp8_promotion_credential.v1" in validator
    assert '"performance_tuning_eligible": True' in validator
    assert '"formal_floor_acceptance_eligible": False' in validator


def test_timing_runner_is_real_exact4_and_uses_distinct_arm_floors() -> None:
    runner = TIMING_RUNNER.read_text(encoding="utf-8")
    reducer = TIMING_REDUCER.read_text(encoding="utf-8")
    manifest = RUNTIME_MANIFEST.read_text(encoding="utf-8")

    assert "config/fr13_fixed32/subset_b4_four.json" in runner
    assert "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5" in runner
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in runner
    assert 'run_arm "$STOCK_ARM" 0' in runner
    assert 'run_arm "$CANDIDATE_ARM" 1' in runner
    assert "scripts/fr13_bigdenom_swe_serve_variant.sh" in runner
    assert "scripts/fr13_measure.py deploy-speed" in runner
    assert "--gate-result \"$GATE_RESULT_JSON\"" in runner
    assert "--expected-gate-sha256 \"$GATE_RESULT_SHA256\"" in runner
    assert "30989326208" in runner
    assert "113.514015414" in runner
    assert "32666638208" in runner
    assert "119.658015414" in runner
    assert "synthetic" not in runner.lower()
    assert "probe-only" not in runner.lower()

    assert '"gpu_components_ms_per_step"' in reducer
    assert '"measured_tps_fullstep_wall"' in reducer
    assert '"accept_per_event"' in reducer
    assert '"wall_residual_ms"' in reducer
    assert "T_CRITICAL_ONE_SIDED_95_DF3" in reducer
    assert '"formal_floor_acceptance_eligible": False' in reducer
    assert "Formal Tail23/Hydra27 acceptance remains separate" in reducer

    for path in (
        "scripts/fr13_draft_head_fp8_gate.py",
        "scripts/fr13_draft_head_fp8_timing.py",
        "scripts/fr13_run_b1_draft_head_fp8_timing.sh",
    ):
        assert f'"{path}"' in manifest
