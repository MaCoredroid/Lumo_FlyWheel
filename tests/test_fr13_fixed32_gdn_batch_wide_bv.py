from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    triton_stub.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel  # noqa: E402


@pytest.mark.parametrize("candidate_bv", (16, 32, 64, 128))
def test_combined_bv_resolver_accepts_only_explicit_wide_candidates(
    candidate_bv: int,
) -> None:
    assert (
        kernel._fr13_resolve_fixed32_batch_gdn_bv(
            "tail6_fixed32",
            env_name="FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            sidecars=(),
            environ={"FR13_FIXED32_BATCH_GDN_BV_CANDIDATE": str(candidate_bv)},
            geom_override={"BV": 8},
        )
        == candidate_bv
    )


@pytest.mark.parametrize("candidate", ("", "8", "15", "24", "256", "x"))
def test_combined_bv_resolver_rejects_invalid_candidates(candidate: str) -> None:
    environ = {"FR13_FIXED32_BATCH_GDN_BV_CANDIDATE": candidate} if candidate else {}
    if not candidate:
        assert (
            kernel._fr13_resolve_fixed32_batch_gdn_bv(
                "tail6_fixed32",
                env_name="FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
                sidecars=(),
                environ=environ,
                geom_override={"BV": 8},
            )
            is None
        )
        return
    with pytest.raises(RuntimeError, match="16, 32, 64, or 128"):
        kernel._fr13_resolve_fixed32_batch_gdn_bv(
            "tail6_fixed32",
            env_name="FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            sidecars=(),
            environ=environ,
            geom_override={"BV": 8},
        )


def test_combined_bv_resolver_rejects_mode_geometry_and_source_drift(
    tmp_path: Path,
) -> None:
    name = "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE"
    with pytest.raises(RuntimeError, match="exact fixed32 mode"):
        kernel._fr13_resolve_fixed32_batch_gdn_bv(
            None,
            env_name=name,
            sidecars=(),
            environ={name: "64"},
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="pinned exactly to BV=8"):
        kernel._fr13_resolve_fixed32_batch_gdn_bv(
            "tail6_fixed32",
            env_name=name,
            sidecars=(),
            environ={name: "64"},
            geom_override={"BV": 64},
        )
    sidecar = tmp_path / "candidate.flag"
    sidecar.write_text("32\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="agreeing sources"):
        kernel._fr13_resolve_fixed32_batch_gdn_bv(
            "tail6_fixed32",
            env_name=name,
            sidecars=(str(sidecar),),
            environ={name: "64"},
            geom_override={"BV": 8},
        )


@pytest.mark.parametrize("batch", (2, 3, 4))
@pytest.mark.parametrize("block_v", (16, 32, 64, 128))
def test_two_launch_contract_is_batch_invariant(batch: int, block_v: int) -> None:
    contract = kernel.fixed32_batch_gdn_launch_contract(
        batch,
        n_actual=32,
        n_pad=32,
        block_v=block_v,
        dim_v=128,
    )
    assert contract["physical_launches_per_layer"] == 2
    assert contract["level_grid_z"] == (batch, 11 * batch)
    assert contract["path_programs"] == 12 * batch
    assert contract["physical_rows_per_request"] == 32


@pytest.mark.parametrize("batch", (0, 1, 5))
def test_two_launch_contract_rejects_invalid_batch(batch: int) -> None:
    with pytest.raises(ValueError, match=r"batch_size must be in \[2, 4\]"):
        kernel.fixed32_batch_gdn_launch_contract(
            batch,
            n_actual=32,
            n_pad=32,
            block_v=64,
            dim_v=128,
        )


@pytest.mark.parametrize("n_actual,n_pad", ((31, 32), (32, 31), (33, 32)))
def test_two_launch_contract_rejects_nonfixed_physical_rows(
    n_actual: int, n_pad: int
) -> None:
    with pytest.raises(ValueError, match="exact 32-row physical tree"):
        kernel.fixed32_batch_gdn_launch_contract(
            4,
            n_actual=n_actual,
            n_pad=n_pad,
            block_v=64,
            dim_v=128,
        )


@pytest.mark.parametrize("block_v", (0, 3, 24, 256))
def test_two_launch_contract_rejects_invalid_block_v(block_v: int) -> None:
    with pytest.raises(ValueError, match="BLOCK_V"):
        kernel.fixed32_batch_gdn_launch_contract(
            4,
            n_actual=32,
            n_pad=32,
            block_v=block_v,
            dim_v=128,
        )


def test_combined_selector_is_default_off_and_b1_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_byte_ab_control", lambda: (False, None)
    )
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_production_control", lambda: None
    )
    assert kernel.fixed32_batch_gdn_selector(1) is None
    assert kernel.fixed32_batch_gdn_selector(4) is None


@pytest.mark.parametrize("batch", (2, 3, 4))
def test_combined_candidate_requires_diagnostic_and_selects_all_b2_b4(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_production_control", lambda: None
    )
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_byte_ab_control", lambda: (False, None)
    )
    with pytest.raises(RuntimeError, match="requires the batched GDN diagnostic"):
        kernel.fixed32_batch_gdn_selector(batch)
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_byte_ab_control", lambda: (True, None)
    )
    assert kernel.fixed32_batch_gdn_selector(batch) == "diagnostic"
    assert kernel.fixed32_batch_gdn_selector(1) is None


def test_wide_live_pass_is_source_bound_and_production_validates_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_path = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(pass_path))
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION", "1")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", 64)
    source_sha = "a" * 64
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_source_sha256", lambda: source_sha
    )

    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:astropy__astropy-12907",
        batch=3,
        layer_keys=set(range(48)),
        reference_bv=8,
        candidate_bv=64,
    )
    assert not pass_path.exists()

    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:astropy__astropy-12907",
        batch=4,
        layer_keys=set(range(48)),
        reference_bv=8,
        candidate_bv=64,
    )
    payload = json.loads(pass_path.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13.fixed32.batch_gdn.live_pass.v2"
    assert payload["source_sha256"] == source_sha
    assert payload["physical_rows_per_request"] == 32
    assert payload["reference_physical_launches_per_layer"] == 8
    assert payload["candidate_physical_launches_per_layer"] == 2
    assert payload["compared_byte_surfaces"] == list(
        kernel._FR13_FIXED32_BATCH_GDN_BV_BYTE_SURFACES
    )
    assert kernel._fr13_fixed32_batch_gdn_production_control() == payload

    payload["source_sha256"] = "b" * 64
    pass_path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(RuntimeError, match="PASS record is invalid"):
        kernel._fr13_fixed32_batch_gdn_production_control()


def test_launcher_wires_fail_closed_combined_bv_sidecars() -> None:
    launcher = (ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    assert "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE" in launcher
    assert "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION" in launcher
    assert "fr13_fixed32_batch_gdn_bv_candidate.flag" in launcher
    assert "fr13_fixed32_batch_gdn_bv_production.flag" in launcher
    assert "B1 path-BV and B2-B4 batched wide-BV selectors" in launcher
    assert (
        "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE requires exactly one "
        "eager or graph byte diagnostic" in launcher
    )
    assert "byte diagnostic requires MAX_NUM_SEQS=4" in launcher
    assert "_fixed32_expected_metrics=1" in launcher
    assert "_fixed32_expected_eager=1" in launcher
    assert "requires MAX_NUM_SEQS=2, 3, or 4" in launcher


def test_graph_byte_diagnostic_launcher_contract_is_fail_closed() -> None:
    launcher = (
        ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")

    assert "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB-0" in launcher
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB must be exactly 0 or 1"
        in launcher
    )
    assert (
        "fixed32 eager and graph batched GDN diagnostics are mutually exclusive"
        in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB requires ENFORCE_EAGER=0"
        in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB requires FR10_METRICS=1"
        in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB requires FR13_RING_EXPORT=1"
        in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB requires FR13_FLAGS_INKERNEL=1"
        in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB requires MAX_NUM_SEQS=4"
        in launcher
    )
    assert "is incompatible with the B1 diagnostic" in launcher
    assert "fr13_fixed32_batch_gdn_graph_byte_ab.enabled" in launcher
    graph_branch_start = launcher.index(
        'if [[ "$_fr13_batch_gdn_graph_byte_ab" == "1" ]]'
    )
    graph_branch_end = launcher.index(
        "if (( _fr13_batch_gdn_diagnostic_count == 1 ))",
        graph_branch_start,
    )
    graph_branch = launcher[graph_branch_start:graph_branch_end]
    assert "fr13_fixed32_batch_gdn_graph_byte_ab.enabled" in graph_branch
    assert "fr13_fixed32_batch_gdn_byte_ab.enabled" not in graph_branch
    assert (
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH=/logs/"
        "fr13_fixed32_batch_gdn_byte_ab.real_event.arm" in launcher
    )
    serve = (
        ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")
    assert (
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=1 requires a fixed32 arm"
        in serve
    )
    assert (
        "fixed32 graph byte diagnostic requires MAX_NUM_SEQS_OVR=4 and "
        "SWE_CONCURRENCY=4" in serve
    )


def test_eager_gate_refreshes_a_stable_nonzero_ssi_registration() -> None:
    patcher = (
        ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text(encoding="utf-8")

    assert (
        "self.fr13_fixed32_spec_state_indices_tensor = ("
        in patcher
    )
    assert (
        "int(self.spec_state_indices_tensor.shape[0]) < ("
        in patcher
    )
    assert '"                spec_state_indices=(\\n"' in patcher
    assert (
        '"                    self.fr13_fixed32_spec_state_indices_tensor\\n"'
        in patcher
    )
    assert "max_batch_size=_fr13_fixed32_ssi_capacity" in patcher
    assert "and not self.use_full_cuda_graph" in patcher
    assert (
        "].copy_(spec_state_indices_tensor, non_blocking=True)"
        in patcher
    )
    assert "].fill_(PAD_SLOT_ID)" in patcher
    assert (
        "max_batch_size=min(4, int(self.decode_cudagraph_max_bs))"
        not in patcher
    )


def test_exact4_b4_live_gate_runner_is_non_timing_and_fail_closed() -> None:
    runner = (
        ROOT / "scripts" / "fr13_run_b4_gdn_wide_live_gate.sh"
    ).read_text(encoding="utf-8")
    assert "config/fr13_fixed32/subset_b4_four.json" in runner
    assert (
        "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
        in runner
    )
    assert "export BSIZE=4" in runner
    assert "export CONC=4" in runner
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in runner
    assert (
        "FR10_METRICS=1 ENFORCE_EAGER=0 "
        "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in runner
    )
    assert "FR13_FIXED32_BATCH_GDN_BYTE_AB=0" in runner
    assert "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=1" in runner
    assert "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=" in runner
    assert "FR13_FIXED32_BATCH_GDN_PRODUCTION=0" in runner
    assert "FR13_DFWD_UNIFIED_BM8_LIVE_AB=0" in runner
    assert "FR13_FIXED32_B1_DIAGNOSTIC=0" in runner
    assert (
        "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 "
        "FR13_CFWD_GPU_TIMER=1"
        in runner
    )
    assert "runner_sha256=" in runner
    assert "B4 diagnostic runner changed during execution" in runner
    assert 'record.get("batch") == 4' in runner
    assert 'record.get("carrier_nonzero") is not True' in runner
    assert 'record.get("status") == "mismatch_reference_served"' in runner
    assert "fr13.fixed32.batch_gdn.graph_live_pass.v1" in runner
    assert '"gate_mode": "post_replay_shadow"' in runner
    assert '"capture_records": 48' in runner
    assert '"real_task_authenticated": True' in runner
    assert 'record.get("graph_id") != graph_id' in runner
    assert 'record.get("graph_signature") != graph_signature' in runner
    assert 'record.get("graph_baseline_byte_equal") is not True' in runner
    assert 'record.get("graph_comparisons")' in runner
    assert '"graph_baseline_out"' in runner
    assert '"floor_acceptance_eligible": False' in runner
    assert '"production_default_enabled": False' in runner
    assert "exact4_b4_graph_byte_diagnostic" in runner
