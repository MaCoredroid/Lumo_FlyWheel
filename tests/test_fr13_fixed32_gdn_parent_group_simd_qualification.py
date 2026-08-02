from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


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


def _campaign_identity(name: str) -> dict[str, object]:
    contract = kernel._FR13_FIXED32_GDN_PARENT_GROUP_SIMD_CAMPAIGNS[name]
    markers = tuple(
        f"swe_verified:{task_id}" for task_id in contract["task_ids"]
    )
    marker_sha256 = hashlib.sha256(
        ("\n".join(markers) + "\n").encode("ascii")
    ).hexdigest()
    reduced = {
        "campaign": name,
        "subset_sha256": contract["subset_sha256"],
        "task_count": len(markers),
        "task_markers_sha256": marker_sha256,
    }
    reduced["campaign_identity_sha256"] = hashlib.sha256(
        json.dumps(
            reduced, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    return reduced


def _pass_payload(
    *, campaign: str, gate: str, identity: dict[str, str]
) -> dict[str, object]:
    campaign_identity = _campaign_identity(campaign)
    return {
        "schema": "fr13.fixed32.gdn_parent_group_simd.b4_live_pass.v1",
        "status": "pass",
        "candidate": "fixed32_gdn_parent_group_simd_v2",
        "kernel": "tree_gdn_parent_group_simd_width4_v2",
        "reference_kernel": "per_request_tree_gdn_path_bv8",
        "mode": "hydra27_fixed32",
        **campaign_identity,
        "gate": gate,
        "batch": 4,
        "physical_rows_per_request": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "simd_width": 4,
        "groups": 5,
        "group_max_path_lengths": [7, 7, 1, 1, 1],
        "grid_z_per_request": [1, 5],
        "event_grid_z": [4, 20],
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "logical_critical_path": 12,
        "physical_critical_path": 12,
        "parent_loads": 5,
        "kernel_source_sha256": identity["kernel_source_sha256"],
        "parent_contract_sha256": identity["parent_contract_sha256"],
        "writer_sha256": identity["writer_sha256"],
        "compared_byte_surfaces": list(
            kernel._FR13_FIXED32_GDN_PARENT_GROUP_SIMD_B4_SURFACES
        ),
        "layer_count": 48,
        "layer_keys": [f"0x{index:x}" for index in range(1, 49)],
        "graph_id": 71 if gate == "graph" else None,
        "graph_signature": "a" * 64 if gate == "graph" else None,
        "capture_records": 48 if gate == "graph" else None,
        "raw_byte_equal": True,
        "state_restored": True,
        "reference_served": True,
        "real_task_authenticated": True,
        "campaign_authenticated": True,
        "production_default_enabled": False,
    }


@pytest.mark.parametrize("campaign", ("exact4", "exact16"))
def test_campaign_marker_requires_complete_canonical_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: str
) -> None:
    contract = kernel._FR13_FIXED32_GDN_PARENT_GROUP_SIMD_CAMPAIGNS[campaign]
    marker = tmp_path / "campaign.arm"
    marker.write_text(
        "\n".join(
            f"swe_verified:{task_id}" for task_id in contract["task_ids"]
        )
        + "\n",
        encoding="ascii",
    )
    marker.chmod(0o444)
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_REAL_EVENT_PATH", str(marker)
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_CAMPAIGN_CACHE", None
    )

    observed = kernel._fr13_fixed32_gdn_parent_group_simd_campaign()
    assert observed == _campaign_identity(campaign)
    assert not any("astropy__" in str(value) for value in observed.values())

    marker.chmod(0o644)
    marker.write_text(marker.read_text(encoding="ascii").splitlines()[0] + "\n")
    marker.chmod(0o444)
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_CAMPAIGN_CACHE", None
    )
    with pytest.raises(RuntimeError, match="not the canonical exact4 or exact16"):
        kernel._fr13_fixed32_gdn_parent_group_simd_campaign()


def test_production_requires_four_distinct_source_bound_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {
        "candidate": "fixed32_gdn_parent_group_simd_v2",
        "kernel": "tree_gdn_parent_group_simd_width4_v2",
        "kernel_source_sha256": "b" * 64,
        "parent_contract_sha256": "c" * 64,
        "writer_sha256": "d" * 64,
    }
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PARENT_GROUP", True)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_identity",
        lambda: identity,
    )
    monkeypatch.setenv("FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION", "1")
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION_CREDENTIAL",
        None,
    )
    paths = {}
    for campaign in ("exact4", "exact16"):
        for gate in ("eager", "graph"):
            path = tmp_path / f"{campaign}.{gate}.json"
            path.write_text(
                json.dumps(
                    _pass_payload(
                        campaign=campaign,
                        gate=gate,
                        identity=identity,
                    ),
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
            path.chmod(0o444)
            env_name = (
                "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_"
                f"{campaign.upper()}_{gate.upper()}_PASS_PATH"
            )
            monkeypatch.setenv(env_name, str(path))
            paths[(campaign, gate)] = path

    credential = (
        kernel._fr13_fixed32_gdn_parent_group_simd_production_control()
    )
    assert credential["candidate"] == "fixed32_gdn_parent_group_simd_v2"
    assert len(credential["pass_sha256"]) == 4
    assert len(credential["credential_sha256"]) == 64

    paths[("exact4", "eager")].chmod(0o644)
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION_CREDENTIAL",
        None,
    )
    with pytest.raises(RuntimeError, match="immutable single-link exact4/eager"):
        kernel._fr13_fixed32_gdn_parent_group_simd_production_control()
    paths[("exact4", "eager")].chmod(0o444)

    bad = json.loads(paths[("exact16", "graph")].read_text(encoding="ascii"))
    bad["candidate"] = "fixed32_batch_gdn_bv8_v1"
    paths[("exact16", "graph")].chmod(0o644)
    paths[("exact16", "graph")].write_text(
        json.dumps(bad) + "\n", encoding="ascii"
    )
    paths[("exact16", "graph")].chmod(0o444)
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION_CREDENTIAL",
        None,
    )
    with pytest.raises(RuntimeError, match="exact16/graph live PASS is invalid"):
        kernel._fr13_fixed32_gdn_parent_group_simd_production_control()


def test_eager_mismatch_permanently_blocks_later_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {
        "candidate": "fixed32_gdn_parent_group_simd_v2",
        "kernel": "tree_gdn_parent_group_simd_width4_v2",
        "kernel_source_sha256": "b" * 64,
        "parent_contract_sha256": "c" * 64,
        "writer_sha256": "d" * 64,
    }
    state = {
        "campaign_identity_sha256": _campaign_identity("exact4")[
            "campaign_identity_sha256"
        ],
        "passed": {(4, index) for index in range(1, 49)},
        "attempts": {},
        "waiting_announced": set(),
        "failed": True,
    }
    path = tmp_path / "must-not-exist.pass.json"
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_EAGER_STATE", state
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_pass_path",
        lambda _campaign, _gate: path,
    )

    with pytest.raises(RuntimeError, match="permanently blocked by a mismatch"):
        kernel._fr13_fixed32_gdn_parent_group_simd_live_pass_emit(
            campaign_identity=_campaign_identity("exact4"),
            gate="eager",
            layer_keys=set(range(1, 49)),
        )
    assert not path.exists()

    launcher = Path(kernel.__file__).read_text(encoding="utf-8")
    assert "eager gate previously mismatched" in launcher
    assert 'grouped_eager_enabled and not bool(gate_state["failed"])' in launcher


@pytest.mark.parametrize("selector", ("eager", "graph", "production"))
def test_b4_selector_detection_does_not_authorize_production(
    monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    enabled = {"eager": False, "graph": False, "production": False}
    enabled[selector] = True

    def selector_enabled(*, env_name: str, default_path: str) -> bool:
        del default_path
        return enabled[
            "eager" if env_name.endswith("B4_EAGER") else "graph"
        ]

    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_enabled",
        selector_enabled,
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_production_armed",
        lambda: enabled["production"],
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_production_control",
        lambda: pytest.fail("B1 selector detection authorized production"),
    )

    assert kernel._fr13_fixed32_gdn_parent_group_simd_b4_selector_armed()


@pytest.mark.parametrize(
    ("selector", "staging_rows"),
    (
        ("graph", 1),
        ("production", 1),
        ("eager", 2),
        ("graph", 2),
        ("eager", 3),
        ("graph", 3),
    ),
)
def test_b4_selector_executes_per_request_preseed_on_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    staging_rows: int,
) -> None:
    for name in ("B4_EAGER", "B4_GRAPH"):
        monkeypatch.setenv(
            f"FR13_FIXED32_GDN_PARENT_GROUP_SIMD_{name}",
            "1" if selector == name.removeprefix("B4_").lower() else "0",
        )
        monkeypatch.setenv(
            f"FR13_FIXED32_GDN_PARENT_GROUP_SIMD_{name}_ENABLED_PATH",
            str(tmp_path / f"{name}.disabled"),
        )
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION",
        "1" if selector == "production" else "0",
    )
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION_ARM_PATH",
        str(tmp_path / "production.disabled"),
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PARENT_GROUP", True)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_ROUTE_REQUESTED", True)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS", None
    )
    monkeypatch.setattr(
        kernel, "_read_tree_gdn_geom_override", lambda: {"BV": 8}
    )
    for control in (
        "scan_align_on",
        "npad_invariant_on",
        "parent_gather_on",
        "hc_internal_on",
    ):
        monkeypatch.setattr(kernel, control, lambda: False)
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_production_control",
        lambda: pytest.fail("B1 preseed authorized grouped production"),
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_byte_ab_control",
        lambda: pytest.fail("B1 preseed entered the grouped source gate"),
    )

    launches = []

    class FakePathKernel:
        def __getitem__(self, grid):
            def launch(*_args, **_kwargs):
                launches.append(grid)

            return launch

    class ForbiddenGroupedKernel:
        def __getitem__(self, _grid):
            pytest.fail("B1 preseed launched the grouped SIMD kernel")

    monkeypatch.setattr(kernel, "_tree_gdn_path_kernel", FakePathKernel())
    monkeypatch.setattr(
        kernel,
        "_tree_gdn_path_kernel_fixed32_parent_group",
        ForbiddenGroupedKernel(),
    )

    n_actual = n_pad = 32
    num_heads = 1
    dim = 8
    descriptor = torch.zeros((1,), dtype=torch.int32)
    state = {
        "schedule": "fixed32",
        "route_armed": True,
        "selfcheck_armed": False,
        "fixed32_contract": {},
        "fixed32_parent_group": {},
        "levels": (
            (descriptor, descriptor, 5, 1, descriptor),
            (descriptor, descriptor, 7, 11, descriptor),
        ),
        "export": torch.zeros((32, num_heads, dim, dim)),
        "emask": torch.zeros((32, 32), dtype=torch.int32),
        "engaged_announced": True,
    }
    cache_key = kernel._subtree_cache_key(
        n_actual, num_heads, dim, dim, torch.device("cpu")
    )
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_CACHE", {cache_key: state})

    q = torch.zeros((n_actual, num_heads, dim))
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    gate = torch.zeros((n_actual, num_heads))
    h0 = torch.zeros((num_heads, dim, dim))
    strict = torch.zeros((n_pad, n_pad), dtype=torch.int32)
    flags = torch.zeros((staging_rows,), dtype=torch.int32)
    counter = torch.zeros((), dtype=torch.int32)

    kernel.launch_tree_gdn_prepared(
        q,
        k,
        v,
        gate,
        gate,
        h0,
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=strict,
        out=torch.zeros_like(q),
        raw_a=gate,
        raw_b=gate,
        A_log=torch.zeros((num_heads,)),
        dt_bias=torch.zeros((num_heads,)),
        invocation_counter=counter,
        ring_k=torch.zeros_like(q),
        ring_v=torch.zeros_like(v),
        ring_a=torch.zeros_like(gate),
        ring_b=torch.zeros_like(gate),
        staging_flags=flags,
        staging_rows=staging_rows,
    )

    assert len(launches) == 2
    assert state["last_physical_execution"]["route"] == "fixed32_path"
    assert state["last_physical_execution"]["batch_size"] == staging_rows
    assert state["last_physical_execution"]["batched"] is False


def _graph_record(layer_key: int) -> dict[str, object]:
    surfaces = kernel._FR13_FIXED32_GDN_PARENT_GROUP_SIMD_B4_SURFACES
    state = {
        "out": torch.full((2,), 7, dtype=torch.uint8),
        "export": torch.arange(32, dtype=torch.uint8),
        "ring_k": torch.full((2,), 11, dtype=torch.uint8),
        "ring_v": torch.full((2,), 12, dtype=torch.uint8),
        "ring_a": torch.full((2,), 13, dtype=torch.uint8),
        "ring_b": torch.full((2,), 14, dtype=torch.uint8),
        "flags": torch.tensor([1, 1, 1, 1], dtype=torch.int32),
        "invocation_counter": torch.tensor(9, dtype=torch.int32),
    }
    compact = torch.arange(20, dtype=torch.uint8)

    def snapshot():
        return {name: state[name].clone() for name in surfaces}

    def restore(value):
        for name in surfaces:
            state[name].copy_(value[name])

    def run_reference():
        state["invocation_counter"].fill_(13)
        return {
            "block_v": 8,
            "physical_launches": 8,
            "kernel_structure": "per_request_tree_gdn_path_bv8",
            "compact_export": compact.clone(),
        }

    def run_candidate(block_v: int):
        state["invocation_counter"].fill_(13)
        state["export"][:20].copy_(compact)
        return {
            "block_v": block_v,
            "physical_launches": 2,
            "kernel_structure": "tree_gdn_parent_group_simd_width4_v2",
            "compact_export": compact.clone(),
        }

    return {
        "layer_key": layer_key,
        "snapshot": snapshot,
        "restore": restore,
        "run_reference": run_reference,
        "run_candidate": run_candidate,
        "carrier_nonzero": lambda: True,
        "byte_equal": torch.equal,
        "surface_names": surfaces,
    }


def test_graph_shadow_waits_for_campaign_then_passes_without_task_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = "e" * 64
    state = {
        "status": "armed",
        "candidate": "fixed32_gdn_parent_group_simd_v2",
        "kernel": "tree_gdn_parent_group_simd_width4_v2",
        "campaign": None,
        "campaign_identity_sha256": None,
        "graph_id": None,
        "graph_signature": None,
        "batch_size": None,
        "records": 0,
    }
    captures = {
        91: {
            "batch_size": 4,
            "graph_signature": signature,
            "records": tuple(_graph_record(index) for index in range(1, 49)),
            "layer_keys": frozenset(range(1, 49)),
        }
    }
    campaign = _campaign_identity("exact4")
    campaign_reads = iter((None, campaign))
    emitted = []
    passed = []
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_graph_control",
        lambda: True,
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_campaign",
        lambda: next(campaign_reads),
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_STATE", state
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_CAPTURES", captures
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_identity",
        lambda: {
            "candidate": "fixed32_gdn_parent_group_simd_v2",
            "kernel": "tree_gdn_parent_group_simd_width4_v2",
            "kernel_source_sha256": "1" * 64,
            "parent_contract_sha256": "2" * 64,
            "writer_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_emit",
        lambda payload: emitted.append(payload),
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_live_pass_emit",
        lambda **payload: passed.append(payload),
    )

    waiting = kernel.fixed32_gdn_parent_group_simd_graph_gate_on_replay(
        91, signature, 4, 48
    )
    assert waiting["status"] == "waiting_for_authenticated_campaign"
    observed = kernel.fixed32_gdn_parent_group_simd_graph_gate_on_replay(
        91, signature, 4, 48
    )
    assert observed["status"] == "passed"
    assert observed["campaign"] == "exact4"
    assert len(emitted) == 48
    assert len(passed) == 1
    assert all("swe_verified:" not in json.dumps(record) for record in emitted)


@pytest.mark.parametrize("terminal_status", ("failed", "passed"))
def test_graph_capture_cannot_reset_terminal_qualification(
    monkeypatch: pytest.MonkeyPatch, terminal_status: str
) -> None:
    state = {
        "status": terminal_status,
        "candidate": "fixed32_gdn_parent_group_simd_v2",
        "kernel": "tree_gdn_parent_group_simd_width4_v2",
        "campaign": "exact4" if terminal_status == "passed" else None,
        "campaign_identity_sha256": (
            _campaign_identity("exact4")["campaign_identity_sha256"]
            if terminal_status == "passed"
            else None
        ),
        "graph_id": 91 if terminal_status == "passed" else None,
        "graph_signature": "e" * 64 if terminal_status == "passed" else None,
        "batch_size": 4 if terminal_status == "passed" else None,
        "records": 48 if terminal_status == "passed" else 0,
    }
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_gdn_parent_group_simd_graph_control",
        lambda: True,
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PARENT_GROUP", True)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_STATE", state
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_CONTEXT", None
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_CAPTURES", {}
    )

    with pytest.raises(RuntimeError, match=f"terminal.*{terminal_status}"):
        kernel.fixed32_gdn_parent_group_simd_graph_capture_begin(92, 4)
    assert kernel._FR13_FIXED32_GDN_PARENT_GROUP_SIMD_GRAPH_CONTEXT is None
    assert state["status"] == terminal_status


def test_patcher_uses_distinct_grouped_graph_hooks_and_census_identity() -> None:
    source = (
        ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
    ).read_text(encoding="utf-8")
    for name in (
        "fixed32_gdn_parent_group_simd_graph_capture_begin",
        "fixed32_gdn_parent_group_simd_graph_capture_end",
        "fixed32_gdn_parent_group_simd_graph_gate_on_replay",
    ):
        assert name in source
    assert '"physical_candidate": event["gdn_physical_candidate"]' in source
    assert '"credential_sha256": event["gdn_credential_sha256"]' in source
    assert 'character not in "0123456789abcdef"' in source
    assert 'b1_source_candidate = batch == 1' in source
    assert 'not physical_batched' in source
    assert "fixed32_batch_gdn_graph_live_gate_on_replay" in source


def test_production_launcher_installs_immutable_passes_without_symlinks(
    tmp_path: Path,
) -> None:
    launcher = (
        ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    anchor = launcher.index('  python3 - "$LOG_DIR" \\\n')
    start = launcher.index("<<'PY'\n", anchor) + len("<<'PY'\n")
    end = launcher.index("\nPY\n", start)
    installer = launcher[start:end]
    source_dir = tmp_path / "source"
    destination = tmp_path / "destination"
    source_dir.mkdir()
    destination.mkdir()
    sources = []
    for index in range(4):
        path = source_dir / f"pass-{index}.json"
        path.write_text("{}\n", encoding="ascii")
        path.chmod(0o444)
        sources.append(path)

    completed = subprocess.run(
        [sys.executable, "-", str(destination), *(str(path) for path in sources)],
        input=installer,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    installed = sorted(destination.glob("*.pass.json"))
    assert len(installed) == 4
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        and path.stat().st_nlink == 1
        and path.read_bytes() == b"{}\n"
        for path in installed
    )

    linked = source_dir / "linked.json"
    linked.symlink_to(sources[0])
    rejected_destination = tmp_path / "rejected"
    rejected_destination.mkdir()
    rejected = subprocess.run(
        [
            sys.executable,
            "-",
            str(rejected_destination),
            str(linked),
            *(str(path) for path in sources[1:]),
        ],
        input=installer,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert not tuple(rejected_destination.iterdir())


def test_real_campaign_launcher_wires_only_distinct_grouped_credentials() -> None:
    launcher = (
        ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    serve = (
        ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT / "scripts" / "fr13_run_b4_gdn_parent_group_simd_live_gate.sh"
    ).read_text(encoding="utf-8")

    for name in (
        "fr13_fixed32_gdn_parent_group.arm",
        "fr13_fixed32_gdn_parent_group_simd_b4_eager.enabled",
        "fr13_fixed32_gdn_parent_group_simd_b4_graph.enabled",
        "fr13_fixed32_gdn_parent_group_simd_b4.real_event.arm",
        "fr13_fixed32_gdn_parent_group_simd.production.arm",
    ):
        assert name in launcher
    for campaign in ("EXACT4", "EXACT16"):
        for gate in ("EAGER", "GRAPH"):
            pass_name = (
                "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_"
                f"{campaign}_{gate}_PASS_PATH"
            )
            assert pass_name in launcher
    assert "zip(sources, names, strict=True)" in launcher
    assert 'getattr(os, "O_NOFOLLOW", 0)' in launcher
    assert "stat.S_IMODE(before.st_mode) != 0o444" in launcher
    assert "identity(os.fstat(source_fd)) != identity(before)" in launcher
    assert "os.replace(temporary, destination)" in launcher
    assert "grouped SIMD must be the only fixed32 GDN/kernel candidate" in launcher
    assert "grouped SIMD qualification/production requires exact B4 concurrency" in serve
    assert "CAMPAIGN=${CAMPAIGN:-exact4}" in runner
    assert "GATE=${GATE:-graph}" in runner
    assert 'ENFORCE_EAGER="$GATE_ENFORCE_EAGER"' in runner
    assert "FR13_FIXED32_GDN_PARENT_GROUP_SIMD_PRODUCTION=0" in runner
    assert "LUMO_SWE_AUTOCOMMIT=0" in runner
    assert 'FR13_DRAFT_VOCAB_K=65536' in runner
    assert 'FR13_DRAFT_VOCAB_ROOT=0' in runner
