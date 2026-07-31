from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
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

    assert patcher.count("_fr13_dfwd_unified_bm8_live_replay(") == 3
    assert patcher.count("stock captured drafter graph unchanged") == 2
    assert (
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB="
        "${FR13_DFWD_UNIFIED_BM8_LIVE_AB:-0}"
    ) in launcher
    assert "FR13 DFWD unified BM8 internal selector is launcher-private" in launcher
    assert "requires fixed32 B1 and an instance id" in launcher
    assert (
        '-e FR13_DFWD_UNIFIED_BM8_LIVE_AB='
        '"$FR13_DFWD_UNIFIED_BM8_LIVE_AB" \\'
    ) in launcher
    assert "DFWD unified BM8 replay hook missing" in launcher


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
