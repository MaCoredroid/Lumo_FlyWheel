from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "fr13_device_multidraft_cfwd_packed_v3.py"
OVERLAY = (
    ROOT / "scripts" / "fr13_cfwd_packed_walk_active_depth_runtime_overlay.py"
)
CANDIDATE = ROOT / "scripts" / "fr13_cfwd_packed_walk_active_depth_kernel.py"
RUNNER = (
    ROOT / "scripts" / "fr13_run_b1_cfwd_packed_walk_active_depth_live_gate.sh"
)
BASE_RUNNER = ROOT / "scripts" / "fr13_run_b1_cfwd_logit_direct_live_gate.sh"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = ROOT / "scripts" / "fr13_runtime_manifest.py"
SELECTOR = "FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_BYTE_AB"


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
    monkeypatch.setenv("FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB", "0")
    monkeypatch.setenv("FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION", "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "0")
    wrapper = _load(WRAPPER, name)
    return wrapper, wrapper._base, wrapper._active_depth_overlay


def test_wrapper_keeps_packed_v3_walk_unmodified_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SELECTOR, raising=False)
    monkeypatch.delenv(
        "FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB", raising=False
    )
    monkeypatch.delenv(
        "FR13_CFWD_PACKED_WALK_NODE_TRUST_PRODUCTION", raising=False
    )
    wrapper = _load(WRAPPER, "fr13_active_depth_default_off_test")
    base = wrapper._base
    assert not getattr(
        base, "_FR13_CFWD_PACKED_WALK_ACTIVE_DEPTH_INSTALLED", False
    )
    assert base._fr13_cfwd_logit_direct_walk_cuda.__module__ == base.__name__


def test_armed_selector_accepts_only_hydra27_physical32_b1_b4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, overlay = _armed_wrapper(
        monkeypatch, "fr13_active_depth_exact_selector_test"
    )
    for batch in (1, 4):
        assert overlay._select(
            _topology(), {"mode": "hydra27_fixed32", "batch_size": batch}
        ) == "active_depth"
    with pytest.raises(RuntimeError, match="exact Hydra27 physical32 B1/B4"):
        overlay._select(
            _topology(), {"mode": "tail6_fixed32", "batch_size": 1}
        )
    drifted = _topology()
    drifted.WALK_CAP = 11
    with pytest.raises(RuntimeError, match="exact Hydra27 physical32 B1/B4"):
        overlay._select(
            drifted, {"mode": "hydra27_fixed32", "batch_size": 1}
        )


def test_walk_routes_candidate_into_existing_comparator_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base, overlay = _armed_wrapper(
        monkeypatch, "fr13_active_depth_walk_route_test"
    )
    calls = []
    monkeypatch.setattr(
        overlay._ACTIVE_DEPTH_MODULE,
        "launch_active_depth_packed_walk",
        lambda **kwargs: calls.append(kwargs),
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
    assert base._fr13_cfwd_logit_direct_walk_cuda(
        _topology(), entry, bonus, decisions
    ) == tuple(outputs)
    assert len(calls) == 1
    assert calls[0]["self_token"] is decisions[0]
    assert calls[0]["event"] is decisions[1]
    assert calls[0]["bonus_token"] is bonus
    assert calls[0]["base_contract"] == overlay._active_depth_base_contract(
        "hydra27_fixed32"
    )


def test_selector_and_source_bindings_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _load(OVERLAY, "fr13_active_depth_fail_closed_test")
    monkeypatch.setenv(SELECTOR, "yes")
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(SELECTOR, "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "0")
    with pytest.raises(RuntimeError, match="requires CFWD diagnostic mode"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    monkeypatch.setenv("FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB", "1")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        overlay.install(SimpleNamespace())
    monkeypatch.setenv(SELECTOR, "0")
    assert overlay.install(SimpleNamespace())["installed"] is False
    wrapper_source = WRAPPER.read_text(encoding="ascii")
    overlay_sha = hashlib.sha256(OVERLAY.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
    assert overlay_sha in wrapper_source
    assert candidate_sha == overlay.ACTIVE_DEPTH_SOURCE_SHA256
    assert candidate_sha in OVERLAY.read_text(encoding="ascii")


def test_runtime_contract_is_default_off_diagnostic_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base, _ = _armed_wrapper(
        monkeypatch, "fr13_active_depth_runtime_contract_test"
    )
    contract = base._fr13_cfwd_packed_walk_active_depth_runtime_contract()
    assert contract["candidate_default_off"] is True
    assert contract["selector"] == "diagnostic"
    assert contract["reference_always_served"] is True
    assert contract["shadow_comparator"] == "_fr13_cfwd_logit_direct_compare"
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows"] == 32
    assert contract["walk_levels"] == 12
    assert contract["walk_termination"] == "first_reject_or_leaf_or_fixed_cap"


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
    assert "Hydra27 physical32 K64/root1 B1 shadow gate" in launcher
    for path in (CANDIDATE, OVERLAY, RUNNER):
        assert str(path.relative_to(ROOT)) in manifest
