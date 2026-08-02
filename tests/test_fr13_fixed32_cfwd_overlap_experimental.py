from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SOURCE = ROOT / "native/fr13_fixed32_cfwd_native_fullvalue.cu"
EXPERIMENTAL_SOURCE = (
    ROOT / "native/experimental/fr13_fixed32_cfwd_native_fullvalue_overlap.cu"
)
ACTIVE_WAVES_SOURCE = ROOT / (
    "native/experimental/"
    "fr13_fixed32_cfwd_native_fullvalue_overlap_active_waves.cu"
)
CONTROL_SOURCE = (
    ROOT / "src/lumo_flywheel_serving/fr13_cfwd_native_fullvalue_cuda.py"
)
PATCHER_SOURCE = ROOT / "scripts/fr13_patch_vllm_cfwd_native_fullvalue_cuda.py"
CANONICAL_SHA256 = (
    "1c1a9813410dcf15bcbb4d23bec71ee16ddcd7e2dbe3b1a3698e58f71bd96985"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_experiment_is_unselected_and_canonical_source_is_frozen() -> None:
    canonical = CANONICAL_SOURCE.read_bytes()
    assert hashlib.sha256(canonical).hexdigest() == CANONICAL_SHA256

    runtime_sources = _read(CONTROL_SOURCE) + _read(PATCHER_SOURCE)
    assert EXPERIMENTAL_SOURCE.name not in runtime_sources
    assert ACTIVE_WAVES_SOURCE.name not in runtime_sources


def test_overlap_source_publishes_each_node_once_before_gate_work() -> None:
    source = _read(EXPERIMENTAL_SOURCE)
    node_publish = source[
        source.index("  const int steps = shared_steps;") :
        source.index("  const int gate_task_count = steps * kHeadGroup;")
    ]
    assert "if (thread_id < steps)" in node_publish
    assert "const int step = thread_id;" in node_publish
    assert "if (step > 0)" in node_publish
    assert "step - 1" in node_publish
    assert node_publish.count("shared_nodes[step] = node;") == 1
    assert node_publish.count("__syncthreads();") == 1
    assert node_publish.index("shared_nodes[step] = node;") < node_publish.index(
        "__syncthreads();"
    )


def test_gate_workers_overlap_first_norm_wave_without_post_gate_barrier() -> None:
    source = _read(EXPERIMENTAL_SOURCE)
    gate_work = source[
        source.index("  const int gate_task_count = steps * kHeadGroup;") :
        source.index("  const int step_slot = warp / kNormPartialWarps;")
    ]
    assert "if (thread_id < gate_task_count)" in gate_work
    assert "const int step = thread_id / kHeadGroup;" in gate_work
    assert "const int local_value_head = thread_id % kHeadGroup;" in gate_work
    assert "const int node = shared_nodes[step];" in gate_work
    assert "accepted_paths[" not in gate_work
    assert "shared_nodes[step] =" not in gate_work
    assert "__syncthreads();" not in gate_work

    norm_work = source[
        source.index("  const int step_slot = warp / kNormPartialWarps;") :
        source.index("  // Process one value head at a time")
    ]
    assert norm_work.count("__syncthreads();") == 3
    assert "Publish every normalized K row and gate scalar" in norm_work


def test_overlap_preserves_incumbent_gate_norm_and_recurrence_arithmetic() -> None:
    canonical = _read(CANONICAL_SOURCE)
    experimental = _read(EXPERIMENTAL_SOURCE)

    gate_math_start = "    const int value_head = key_head * kHeadGroup"
    gate_math_end = "        sigmoid(load_bf16(b_rings + ab_offset));\n"
    canonical_gate = canonical[
        canonical.index(gate_math_start) :
        canonical.index(gate_math_end) + len(gate_math_end)
    ]
    experimental_gate = experimental[
        experimental.index(gate_math_start) :
        experimental.index(gate_math_end) + len(gate_math_end)
    ]
    assert experimental_gate == canonical_gate

    norm_start = "  const int step_slot = warp / kNormPartialWarps;"
    canonical_norm_end = "  // Publish every immutable normalized K row"
    experimental_norm_end = "  // Publish every normalized K row and gate scalar"
    assert experimental[
        experimental.index(norm_start) : experimental.index(experimental_norm_end)
    ] == canonical[canonical.index(norm_start) : canonical.index(canonical_norm_end)]

    recurrence_start = "  // Process one value head at a time"
    assert experimental[experimental.index(recurrence_start) :] == canonical[
        canonical.index(recurrence_start) :
    ]


def test_overlap_source_keeps_same_static_barrier_count() -> None:
    canonical = _read(CANONICAL_SOURCE)
    experimental = _read(EXPERIMENTAL_SOURCE)
    assert canonical.count("__syncthreads();") == 5
    assert experimental.count("__syncthreads();") == 5


def test_active_wave_variant_uses_only_a_cta_uniform_trailing_wave_guard() -> None:
    overlap = _read(EXPERIMENTAL_SOURCE)
    active_waves = _read(ACTIVE_WAVES_SOURCE)
    loop_marker = "#pragma unroll\n  for (int wave = 0;"
    guard = """  const int active_precompute_waves =
      (steps + kStepsPerWave - 1) / kStepsPerWave;
"""
    guarded_loop = """#pragma unroll
  for (int wave = 0; wave < kPrecomputeWaves; ++wave) {
    // steps is CTA-uniform, so every thread skips the same trailing barriers.
    if (wave >= active_precompute_waves) {
      continue;
    }
"""
    expected = overlap.replace(loop_marker, guard + loop_marker, 1).replace(
        """#pragma unroll
  for (int wave = 0; wave < kPrecomputeWaves; ++wave) {
""",
        guarded_loop,
        1,
    )
    assert active_waves == expected
    assert active_waves.count("const int steps = shared_steps;") == 1
    assert "\n  steps =" not in active_waves
    setup = active_waves[
        active_waves.index("if (thread_id == 0)") :
        active_waves.index("const int steps = shared_steps;")
    ]
    assert setup.index("shared_steps =") < setup.index("__syncthreads();")
    guard_definition = active_waves[
        active_waves.index("const int active_precompute_waves =") :
        active_waves.index("#pragma unroll", active_waves.index("const int active"))
    ]
    assert all(
        divergent_name not in guard_definition
        for divergent_name in ("thread_id", "warp", "lane", "blockIdx")
    )
    guarded_wave = active_waves[
        active_waves.index("if (wave >= active_precompute_waves)") :
        active_waves.index("// Publish every normalized K row")
    ]
    for per_wave_operation in (
        "const int step = wave * kStepsPerWave + step_slot;",
        "normalized_ks[step][key_index] = key_value;",
        "norm_partials[step_slot][norm_warp] = warp_partial;",
        "inverse_norms[step_slot] =",
        "normalized_ks[step][key_index] = __fmul_rn(",
    ):
        assert guarded_wave.index("continue;") < guarded_wave.index(
            per_wave_operation
        )
    assert guarded_wave.index("continue;") < guarded_wave.index(
        "__syncthreads();"
    )
    assert guarded_wave.count("__syncthreads();") == 2
    assert active_waves.count("__syncthreads();") == 5


def test_active_wave_count_covers_exactly_the_live_steps() -> None:
    expected = {
        **dict.fromkeys(range(1, 5), 1),
        **dict.fromkeys(range(5, 9), 2),
        **dict.fromkeys(range(9, 13), 3),
    }
    assert {steps: (steps + 3) // 4 for steps in range(1, 13)} == expected
