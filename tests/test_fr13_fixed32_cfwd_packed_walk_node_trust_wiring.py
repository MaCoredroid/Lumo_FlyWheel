from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "fr13_device_multidraft_cfwd_packed_v3.py"
OVERLAY = (
    ROOT / "scripts" / "fr13_cfwd_packed_walk_node_trust_runtime_overlay.py"
)
CANDIDATE = ROOT / "scripts" / "fr13_cfwd_packed_walk_node_trust_kernel.py"
RUNNER = (
    ROOT / "scripts" / "fr13_run_b1_cfwd_packed_walk_node_trust_live_gate.sh"
)
BASE_RUNNER = ROOT / "scripts" / "fr13_run_b1_cfwd_logit_direct_live_gate.sh"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = ROOT / "scripts" / "fr13_runtime_manifest.py"
SELECTOR = "FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB"
PRODUCTION_SELECTOR = "FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION"


def _load(path: Path, name: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _topology() -> SimpleNamespace:
    return SimpleNamespace(
        PHYSICAL_DRAFTS=31,
        PHYSICAL_ROWS=32,
        WALK_CAP=12,
        OUTPUT_PUBLISH_CAPACITY=32,
        ACCEPTED_PATH_CAPACITY=16,
    )


def _armed_wrapper(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv(PRODUCTION_SELECTOR, "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "0")
    wrapper = _load(WRAPPER, name)
    return wrapper, wrapper._base, wrapper._node_trust_overlay


def test_wrapper_keeps_packed_v3_walk_unmodified_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SELECTOR, raising=False)
    monkeypatch.delenv(PRODUCTION_SELECTOR, raising=False)
    wrapper = _load(WRAPPER, "fr13_node_trust_default_off_test")
    base = wrapper._base
    assert not getattr(
        base, "_FR13_CFWD_PACKED_WALK_NODE_TRUST_INSTALLED", False
    )
    assert base._fr13_cfwd_logit_direct_walk_cuda.__module__ == base.__name__
    assert "_fr13_fixed32_taw_packed_physical_slot_commit_kernel" in (
        inspect.getsource(base._fr13_cfwd_logit_direct_walk_cuda)
    )


def test_armed_selector_accepts_only_hydra27_physical32_b1_b4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base, overlay = _armed_wrapper(
        monkeypatch, "fr13_node_trust_exact_selector_test"
    )
    assert base._FR13_CFWD_PACKED_WALK_NODE_TRUST_INSTALLED is True
    for batch in (1, 4):
        assert overlay._select(
            _topology(), {"mode": "hydra27_fixed32", "batch_size": batch}
        ) == "node_trust"
    for entry in (
        {"mode": "tail6_fixed32", "batch_size": 1},
        {"mode": "hydra27_fixed32", "batch_size": 2},
    ):
        with pytest.raises(RuntimeError, match="exact Hydra27 physical32 B1/B4"):
            overlay._select(_topology(), entry)
    drifted = _topology()
    drifted.PHYSICAL_ROWS = 31
    with pytest.raises(RuntimeError, match="exact Hydra27 physical32 B1/B4"):
        overlay._select(
            drifted, {"mode": "hydra27_fixed32", "batch_size": 1}
        )


def test_armed_selector_requires_existing_diagnostic_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, overlay = _armed_wrapper(
        monkeypatch, "fr13_node_trust_shadow_selector_test"
    )
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")
    with pytest.raises(RuntimeError, match="requires CFWD diagnostic mode"):
        overlay._select(
            _topology(), {"mode": "hydra27_fixed32", "batch_size": 1}
        )


def test_production_selector_requires_and_reuses_base_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SELECTOR, "0")
    monkeypatch.setenv(PRODUCTION_SELECTOR, "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "1")
    wrapper = _load(WRAPPER, "fr13_node_trust_production_selector_test")
    base = wrapper._base
    overlay = wrapper._node_trust_overlay
    monkeypatch.setattr(
        base,
        "_fr13_cfwd_logit_direct_selector",
        lambda **_kwargs: "production",
    )
    assert overlay._select(
        _topology(), {"mode": "hydra27_fixed32", "batch_size": 1}
    ) == "node_trust"
    contract = base._fr13_cfwd_packed_walk_node_trust_runtime_contract()
    assert contract["selector"] == "production"
    assert contract["reference_always_served"] is False


def test_walk_routes_candidate_outputs_into_existing_comparator_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base, overlay = _armed_wrapper(
        monkeypatch, "fr13_node_trust_walk_route_test"
    )
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        overlay._NODE_TRUST_MODULE, "launch_packed_walk_node_trust", launch
    )
    outputs = [object() for _ in range(5)]
    entry = {
        "mode": "hydra27_fixed32",
        "batch_size": 4,
        "output_tokens": outputs[0],
        "output_lens": outputs[1],
        "accepted_path_rows": outputs[2],
        "accepted_lens": outputs[3],
        "last_row": outputs[4],
    }
    decisions = (object(), object())
    bonus = object()
    observed = base._fr13_cfwd_logit_direct_walk_cuda(
        _topology(), entry, bonus, decisions
    )
    assert observed == tuple(outputs)
    assert len(calls) == 1
    assert calls[0]["self_token"] is decisions[0]
    assert calls[0]["event"] is decisions[1]
    assert calls[0]["bonus_token"] is bonus
    assert calls[0]["producer_contract"] == overlay._producer_contract(
        "hydra27_fixed32"
    )
    commit_source = inspect.getsource(base.fr13_fixed32_cfwd_logit_direct_commit)
    assert commit_source.index("_fr13_cfwd_logit_direct_walk_cuda") < (
        commit_source.index("_fr13_cfwd_logit_direct_compare")
    )


def test_selector_and_source_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _load(OVERLAY, "fr13_node_trust_fail_closed_test")
    monkeypatch.setenv(PRODUCTION_SELECTOR, "0")
    monkeypatch.setenv(SELECTOR, "yes")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "0")
    with pytest.raises(RuntimeError, match="requires CFWD diagnostic mode"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv(PRODUCTION_SELECTOR, "1")
    with pytest.raises(RuntimeError, match="diagnostic and production are exclusive"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(SELECTOR, "0")
    monkeypatch.setenv(PRODUCTION_SELECTOR, "1")
    with pytest.raises(RuntimeError, match="requires CFWD production mode"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(PRODUCTION_SELECTOR, "yes")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(PRODUCTION_SELECTOR, "0")
    monkeypatch.setenv(SELECTOR, "0")
    assert overlay.install(SimpleNamespace())["installed"] is False
    wrapper_source = WRAPPER.read_text(encoding="ascii")
    overlay_sha = hashlib.sha256(OVERLAY.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert overlay_sha in wrapper_source
    assert candidate_sha == overlay.NODE_TRUST_SOURCE_SHA256
    assert candidate_sha in OVERLAY.read_text(encoding="ascii")


def test_runtime_contract_is_default_off_and_reuses_shadow_comparator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base, _ = _armed_wrapper(
        monkeypatch, "fr13_node_trust_runtime_contract_test"
    )
    contract = base._fr13_cfwd_packed_walk_node_trust_runtime_contract()
    assert contract["candidate_default_off"] is True
    assert contract["selector"] == "diagnostic"
    assert contract["reference_always_served"] is True
    assert contract["shadow_comparator"] == "_fr13_cfwd_logit_direct_compare"
    assert contract["mode"] == "hydra27_fixed32"
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows"] == 32
    assert contract["walk_levels"] == 12
    assert contract["installed"] is True


def test_runner_is_source_bound_real_swe_k64_b1_shadow() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    source = RUNNER.read_text(encoding="ascii")
    base = BASE_RUNNER.read_text(encoding="ascii")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert f"{SELECTOR}=1" in source
    assert "fr13_run_b1_cfwd_logit_direct_live_gate.sh" in source
    assert hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() in source
    assert hashlib.sha256(OVERLAY.read_bytes()).hexdigest() in source
    assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest() in source
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in base
    assert "FR13_DRAFT_VOCAB_K=65536 FR13_DRAFT_VOCAB_ROOT=1" in base
    assert "real_swe_verified_one_task" in base
    assert "PROBE_ONLY" not in source and "CAPTURE_ONLY" not in source
    assert f'-e {SELECTOR}="${SELECTOR}"' in launcher
    assert f'{PRODUCTION_SELECTOR}=${{{PRODUCTION_SELECTOR}:-0}}' in launcher
    assert f'-e {PRODUCTION_SELECTOR}="${PRODUCTION_SELECTOR}"' in launcher
    assert "node trust production requires the source-bound" in launcher
    assert "Hydra27 physical32 K64/root1 B1 shadow gate" in launcher
    for path in (CANDIDATE, OVERLAY, RUNNER):
        assert str(path.relative_to(ROOT)) in manifest
