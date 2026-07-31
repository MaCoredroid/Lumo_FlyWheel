from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = ROOT / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
VALIDATOR = ROOT / "scripts" / "fr13_dfwd_unified_bm8_gate.py"
MATRIX = (
    ROOT
    / "results"
    / "fr13_fixed32_dfwd_unified_bm8_source_20260731"
    / "kernel_matrix.json"
)


def _load_patcher(monkeypatch):
    monkeypatch.setenv("FR13_FIXED32_MODE", "tail6_fixed32")
    spec = importlib.util.spec_from_file_location("fr13_bm8_patcher", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validator():
    spec = importlib.util.spec_from_file_location("fr13_bm8_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_unified_source() -> str:
    return '''import torch

logger = object()
is_batch_invariant = False
float8_info = torch.finfo(current_platform.fp8_dtype())


def kernel_2d():
    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        P = tl.exp(S - m_j[:, None])
        alpha = tl.exp(M - m_j)


def kernel_3d():
    # iterate through tiles (now limited to the sliding window range)
    for j in range(
        max(segm_idx * tiles_per_segment, tile_start),
        min((segm_idx + 1) * tiles_per_segment, tile_end),
    ):
        seq_offset = j * TILE_SIZE + offs_t


def reduce_segments():
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    segm_output *= tl.exp(segm_max - overall_max)[:, None]


def unified_attention(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap, q_descale,
    k_descale, v_descale, seq_threshold_3D=None,
    num_par_softmax_segments=None, softmax_segm_output=None,
    softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
    output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None,
    use_alibi_sqrt=False,
):
    use_qq_bias = qq_bias is not None
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    if use_qq_bias:
        if True:
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    launch(
        scale=softmax_scale,
    )
'''


def test_bm8_patcher_emits_compilable_default_stock_live_gate(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_patcher(monkeypatch)
    target = tmp_path / "triton_unified_attention.py"
    target.write_text(_minimal_unified_source(), encoding="utf-8")
    module.TRITON_UNIFIED_ATTN_PATH = target

    assert module._patch_triton_unified_attention_fr13() is True
    emitted = target.read_text(encoding="utf-8")
    compile(emitted, str(target), "exec")

    assert "FR13_DFWD_UNIFIED_BM8_LIVE_GATE" in emitted
    assert 'os.environ.get("FR13_DFWD_UNIFIED_BM8_INTERNAL") == "1"' in emitted
    assert "BLOCK_M = 8" in emitted
    assert "_FR13_DFWD_UNIFIED_BM8_DISPATCHES += 1" in emitted
    assert "refused false stock-vs-stock dispatch" in emitted
    assert "if _fr13_dfwd_bm8" in emitted
    assert "else q.shape[0] // BLOCK_Q + num_seqs" in emitted
    assert "query_snapshot.copy_(q)" in emitted
    assert "seq_lens_snapshot.copy_(seqused_k)" in emitted
    assert emitted.index("torch.cuda.is_current_stream_capturing()") < emitted.index(
        "FR13 DFWD unified BM8 internal selector leaked"
    )
    assert '"served_return": "stock captured drafter graph unchanged"' in emitted
    assert '"performance_measurement": False' in emitted
    assert "_fr13_dfwd_unified_bm8_real_task_marker" in emitted
    assert "_fr13_dfwd_unified_bm8_candidate_identity" in emitted
    assert "_stat.S_IMODE(metadata.st_mode) != 0o444" in emitted
    assert '"candidate_identity": candidate_identity' in emitted
    assert '"task_marker": task_marker' in emitted


def test_bm8_keeps_every_valid_b1_gqa_lane_mapping() -> None:
    def valid_lanes(block_m: int):
        return [
            (offset // 6, offset % 6)
            for offset in range(block_m)
            if offset // 6 < 1
        ]

    expected = [(0, head) for head in range(6)]
    assert valid_lanes(16) == expected
    assert valid_lanes(8) == expected
    assert 16 - len(expected) == 10
    assert 8 - len(expected) == 2


def test_drafter_replay_hook_and_launcher_are_fail_closed() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert patcher.count("_fr13_dfwd_unified_bm8_live_replay(") == 3
    assert patcher.count("stock captured drafter graph unchanged") == 2
    assert (
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB="
        "${FR13_DFWD_UNIFIED_BM8_LIVE_AB:-0}"
    ) in launcher
    assert "FR13 DFWD unified BM8 internal selector is launcher-private" in launcher
    assert (
        "requires the pinned real SWE B1 diagnostic and exact source identity"
        in launcher
    )
    assert (
        '-e FR13_DFWD_UNIFIED_BM8_LIVE_AB='
        '"$FR13_DFWD_UNIFIED_BM8_LIVE_AB" \\'
    ) in launcher
    assert "DFWD unified BM8 replay hook missing" in launcher
    assert "FR13_DFWD_UNIFIED_BM8_REAL_EVENT_PATH" in launcher
    assert "FR13_DFWD_UNIFIED_BM8_IDENTITY_JSON" in launcher
    assert "FR13_DFWD_UNIFIED_BM8_SOURCE_COMMIT" in launcher
    assert "temporary.chmod(0o444)" in launcher
    assert "fixed32-bm8-real-event-arm" in (
        ROOT / "scripts/run_swe_bench_q36_a.py"
    ).read_text(encoding="utf-8")
    assert 'FR13_GATE_BM8=${FR13_GATE_BM8:-0}' in runner
    assert "FR13_GATE_BM8 must be the only enabled kernel candidate" in runner
    assert "fr13_dfwd_unified_bm8_gate.py verify" in runner
    assert "FR13_DFWD_UNIFIED_BM8_PRODUCTION" not in launcher


def _identity(source_commit: str) -> dict:
    return {
        "schema": "fr13.fixed32.dfwd_unified_bm8.identity.v1",
        "source_commit": source_commit,
        "production_enabled": False,
        "candidate": {
            "kernel": "kernel_unified_attention_2d",
            "stock_block_m": 16,
            "stock_block_q": 2,
            "candidate_block_m": 8,
            "candidate_block_q": 1,
            "required_calls": 4,
        },
        "files": {
            label: {"path": path, "sha256": str(index + 1) * 64}
            for index, (label, path) in enumerate(
                {
                    "patcher": (
                        "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py"
                    ),
                    "unified_attention": (
                        "/usr/local/lib/python3.12/dist-packages/vllm/v1/"
                        "attention/ops/triton_unified_attention.py"
                    ),
                    "eagle_replay_hook": (
                        "/usr/local/lib/python3.12/dist-packages/vllm/v1/"
                        "spec_decode/eagle.py"
                    ),
                }.items()
            )
        },
    }


def _live(identity: dict, instance_id: str) -> dict:
    calls = [
        {
            "call_index": index,
            "seq_len": 128 + index,
            "bytes": 12288,
            "raw_byte_mismatches": 0,
            "stock_sha256": str(index + 4) * 64,
            "candidate_sha256": str(index + 4) * 64,
        }
        for index in range(4)
    ]
    return {
        "schema": "fr13.fixed32.dfwd_unified_bm8_live_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": instance_id,
        "concurrency": 1,
        "batch_size": 1,
        "task_marker": f"swe_verified:{instance_id}",
        "candidate_identity": identity,
        "calls": calls,
        "geometry": {
            "query_shape": [1, 24, 256],
            "kv_heads": 4,
            "stock_block_m": 16,
            "stock_block_q": 2,
            "candidate_block_m": 8,
            "candidate_block_q": 1,
            "valid_query_heads_per_kv": 6,
        },
        "candidate_dispatch": "launcher-private BM8 exact B1 selector",
        "candidate_dispatches": 4,
        "served_return": "stock captured drafter graph unchanged",
        "performance_measurement": False,
    }


def test_live_validator_requires_task_and_exact_candidate_identity(
    tmp_path: Path,
) -> None:
    module = _load_validator()
    source_commit = "a" * 40
    instance_id = "astropy__astropy-12907"
    identity = _identity(source_commit)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity) + "\n", encoding="ascii")
    identity_path.chmod(0o444)
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(_live(identity, instance_id)) + "\n", encoding="ascii")

    result = module.verify(
        live_result=live_path,
        identity_path=identity_path,
        expected_source_commit=source_commit,
        expected_instance_id=instance_id,
    )

    assert result["status"] == "PASS"
    assert result["calls"] == 4
    assert result["raw_byte_mismatches"] == 0
    assert result["performance_measurement"] is False
    assert result["production_enabled"] is False

    identity_path.chmod(0o644)
    with pytest.raises(module.GateError, match="mode is not 0444"):
        module.verify(
            live_result=live_path,
            identity_path=identity_path,
            expected_source_commit=source_commit,
            expected_instance_id=instance_id,
        )
    identity_path.chmod(0o444)

    tampered = _live(identity, instance_id)
    tampered["task_marker"] = "swe_verified:django__django-10000"
    live_path.write_text(json.dumps(tampered) + "\n", encoding="ascii")
    with pytest.raises(module.GateError, match="provenance or dispatch"):
        module.verify(
            live_result=live_path,
            identity_path=identity_path,
            expected_source_commit=source_commit,
            expected_instance_id=instance_id,
        )


def test_bm8_real_task_arm_uses_distinct_canonical_marker(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_swe_bench_q36_a as orchestrator

    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    logs.chmod(0o700)
    path = (logs / "fr13_dfwd_unified_bm8.real_event.arm").resolve()
    arm = orchestrator._Fixed32Bm8RealTaskArm(
        path=path,
        instance_id="astropy__astropy-12907",
    )

    arm.start()
    assert path.read_text(encoding="ascii") == (
        "swe_verified:astropy__astropy-12907\n"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert arm.as_dict()["schema"] == "fr13-fixed32-bm8-real-task-arm-v1"
    assert arm.artifact_name == "fixed32_bm8_real_task_arm.json"
    arm.finish()
    assert not path.exists()
    assert arm.rotated_path.is_file()


def test_source_artifact_has_no_gpu_or_speed_claim() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert matrix["status"] == "SOURCE_IMPLEMENTED_GPU_GATE_NOT_RUN"
    assert matrix["default_behavior_changed"] is False
    assert matrix["gpu_used"] is False
    assert matrix["synthetic_or_probe_timing_used"] is False
    assert matrix["live_gate"]["run_completed"] is False
    assert matrix["live_gate"]["served_return"] == (
        "stock captured drafter graph unchanged"
    )
    assert matrix["candidate"]["invalid_row_reduction_fraction"] == 0.8
    assert matrix["floor_accounting"]["optimistic_candidate_group_ceiling_ms"] == (
        6.967564
    )
