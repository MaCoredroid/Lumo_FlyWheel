from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumo_flywheel_serving import fr13_gdn_warpgroup_cuda as candidate


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "native/fr13_fixed32_gdn_warpgroup.cu"
PATCHER_SOURCE = ROOT / "scripts/fr13_patch_vllm_gdn_warpgroup_cuda.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("warpgroup_patcher", PATCHER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection_kwargs() -> dict[str, object]:
    return {
        "mode": "hydra27_fixed32",
        "batch_size": 4,
        "n_actual": 32,
        "n_pad": 32,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "dim_k": 128,
        "dim_v": 128,
        "block_v": 8,
        "use_qk_l2norm": True,
        "scan_align": True,
        "ring_export": True,
        "flags_export": True,
        "h0_use_accepted_column": False,
        "op_available": True,
    }


def test_static_schedule_is_exact_single_writer_fixed32() -> None:
    contract = candidate.validate_static_schedule()
    assert contract == {
        "root_path": (0, 1, 4, 9, 14),
        "root_depth": 5,
        "group_parents": (14, 0, 1, 4, 9),
        "group_sizes": (3, 2, 2, 2, 2),
        "active_member_warps": 11,
        "max_branch_depth": 7,
        "logical_depth": 12,
        "physical_recurrence_depth": 12,
        "writer_nodes": 32,
    }
    writers = candidate.ROOT_PATH + tuple(
        node
        for group in candidate.GROUP_PATHS
        for path in group
        for node in path
    )
    assert sorted(writers) == list(range(32))
    assert len(writers) == len(set(writers))


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_resource_contract_is_one_launch_and_legal_shape(batch: int) -> None:
    resource = candidate.resource_contract(batch)
    assert resource["threads_per_cta"] == 640
    assert resource["warps_per_cta"] == 20
    assert resource["active_member_warps"] == 11
    assert resource["inactive_member_warps"] == 9
    assert resource["static_shared_bytes"] == 20_480
    assert resource["launches_per_layer"] == 1
    assert resource["ctas_per_layer"] == batch * 768
    assert resource["physical_recurrence_depth"] == 12
    assert resource["hbm_parent_state_exports"] == 0
    assert resource["hbm_parent_state_reads"] == 0
    assert resource["compile_gate"]["registers_per_thread_at_most"] == 102
    assert resource["compile_gate"]["local_bytes"] == 0


def test_selector_is_default_off() -> None:
    assert candidate.resolve_candidate(
        **_selection_kwargs(), environ={}
    ) is None
    assert candidate.resolve_candidate(
        **_selection_kwargs(), environ={candidate.SELECTOR_ENV: "0"}
    ) is None


def test_selector_is_exact_and_never_authorizes_production() -> None:
    selection = candidate.resolve_candidate(
        **_selection_kwargs(),
        environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
    )
    assert selection is not None
    assert selection["candidate"] == candidate.CANDIDATE
    assert selection["batch_size"] == 4
    assert selection["production_authorized"] is False
    assert selection["fallback_on_error"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", None),
        ("batch_size", 8),
        ("n_actual", 31),
        ("n_pad", 64),
        ("num_key_heads", 8),
        ("num_value_heads", 32),
        ("dim_k", 64),
        ("dim_v", 64),
        ("block_v", 16),
        ("use_qk_l2norm", False),
        ("scan_align", False),
        ("ring_export", False),
        ("flags_export", False),
        ("h0_use_accepted_column", True),
        ("op_available", False),
    ],
)
def test_armed_selector_fails_closed_on_contract_drift(
    field: str, value: object
) -> None:
    kwargs = _selection_kwargs()
    kwargs[field] = value
    with pytest.raises(RuntimeError):
        candidate.resolve_candidate(
            **kwargs,
            environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
        )


def test_unknown_selector_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="must be unset"):
        candidate.resolve_candidate(
            **_selection_kwargs(),
            environ={candidate.SELECTOR_ENV: "production"},
        )


def test_incumbent_byte_gate_is_explicit_and_reference_serving() -> None:
    plan = candidate.incumbent_byte_gate_plan()
    assert plan["reference_route"] == "force_incumbent_fixed32_path_bv8"
    assert plan["surfaces"] == (
        "output",
        "ring_k",
        "ring_v",
        "ring_a",
        "ring_b",
        "flags",
        "invocation_counter",
    )
    assert plan["comparison"] == "raw_bytes"
    assert plan["restore_before_candidate"] is True
    assert plan["reference_always_served_during_qualification"] is True
    assert plan["production_credential_emitted"] is False
    assert plan["timing_eligible"] is False


def test_operator_probe_and_launch_are_source_bound() -> None:
    calls: list[tuple[object, ...]] = []

    def op(*args: object) -> None:
        calls.append(args)

    torch_module = SimpleNamespace(
        ops=SimpleNamespace(
            _C=SimpleNamespace(fr13_fixed32_gdn_warpgroup=op)
        )
    )
    assert candidate.operator_available(torch_module)
    selection = candidate.resolve_candidate(
        **_selection_kwargs(),
        environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
    )
    assert selection is not None
    tensors = {name: object() for name in (
        "out", "q", "k", "v", "raw_a", "raw_b", "a_log", "dt_bias",
        "h0", "h0_indices", "ring_k", "ring_v", "ring_a", "ring_b",
        "flags", "invocation_counter",
    )}
    candidate.launch_candidate(
        torch_module=torch_module,
        selection=selection,
        **tensors,
        h0_index_row=0,
        h0_index_batch_stride=13,
        h0_bank_stride=48 * 128 * 128,
        output_scale=128**-0.5,
        count_invocation=True,
    )
    assert len(calls) == 1
    assert calls[0][16] == 4
    assert calls[0][-5:] == (True, True, True, True, True)


def test_native_source_encodes_block_cooperation_and_unique_writers() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;" in source
    assert "static_assert(kThreadsPerBlock == 640);" in source
    assert "static_assert(kParentSharedBytes == 20480);" in source
    assert "__launch_bounds__(kThreadsPerBlock, 1)" in source
    assert "__shared__ float parent_states[kGroups * kStateElements];" in source
    assert source.count("__syncthreads();") == 2
    assert "const int group = warp / kWarpsPerGroup;" in source
    assert "const int member = warp % kWarpsPerGroup;" in source
    assert "value_tile == 0 && value_head % kHeadGroup == 0" in source
    assert "ring_export && value_tile == 0 && lane == 0" in source
    assert "state_export" not in source
    assert "properties->major == 12 && properties->minor == 1" in source
    assert "properties->maxThreadsPerBlock >= kThreadsPerBlock" in source
    assert "properties->sharedMemPerBlock >= kParentSharedBytes" in source


def test_patcher_adds_one_source_to_pinned_extension_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _load_patcher()
    fixture = {
        Path("CMakeLists.txt"): patcher.CMAKE_ANCHOR,
        Path("csrc/ops.h"): patcher.OPS_ANCHOR,
        Path("csrc/torch_bindings.cpp"): patcher.BINDINGS_ANCHOR,
    }
    for relative, text in fixture.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        patcher,
        "PINNED_SHA256",
        {
            relative: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for relative, text in fixture.items()
        },
    )
    assert patcher.patch_source_root(tmp_path, CUDA_SOURCE) is True
    assert patcher.patch_source_root(tmp_path, CUDA_SOURCE) is False
    cmake = (tmp_path / "CMakeLists.txt").read_text(encoding="utf-8")
    ops = (tmp_path / "csrc/ops.h").read_text(encoding="utf-8")
    bindings = (tmp_path / "csrc/torch_bindings.cpp").read_text(
        encoding="utf-8"
    )
    assert cmake.count("csrc/fr13_fixed32_gdn_warpgroup.cu") == 1
    assert "void fr13_fixed32_gdn_warpgroup(" in ops
    assert bindings.count('ops.impl("fr13_fixed32_gdn_warpgroup"') == 1
    assert (
        tmp_path / "csrc/fr13_fixed32_gdn_warpgroup.cu"
    ).read_bytes() == CUDA_SOURCE.read_bytes()


def test_patcher_rejects_partial_install(tmp_path: Path) -> None:
    patcher = _load_patcher()
    for relative in patcher.PINNED_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("anchor\n", encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text(
        f"# {patcher.MARKER}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="partial"):
        patcher.patch_source_root(tmp_path, CUDA_SOURCE)
