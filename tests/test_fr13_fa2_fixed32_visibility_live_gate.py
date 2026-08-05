from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
B1_RUNNER = REPO / "scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh"
B4_RUNNER = REPO / "scripts/fr13_run_b4_fa2_qrow32_live_gate.sh"


def _module(relative: str, name: str):
    path = REPO / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_visibility_arms_are_explicit_and_incumbent_defaults_are_unchanged() -> None:
    patcher = PATCHER.read_text(encoding="ascii")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    b1_runner = B1_RUNNER.read_text(encoding="ascii")
    b4_runner = B4_RUNNER.read_text(encoding="ascii")

    assert '"visibility": {' in patcher
    assert "c5ab32a6ae4e615f1e77a4997db5429152053c549e761fb11d90b33bb3959a79" in patcher
    assert "805635d6881dbf73287d66c10541880b7cf93bcb6bf7b04e50efd3d32728b0aa" in patcher
    assert 'candidate_arm == "visibility" and fixed32_mode != "hydra27_fixed32"' in patcher
    assert '""|qrow32|gqa_pair|visibility)' in launcher
    assert '""|nosplit|split2|visibility)' in launcher
    assert '""|nosplit) ;;' in launcher
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty or nosplit" in launcher
    assert 'LIVE_AB_ARM=${FR13_QROW32_LIVE_AB_ARM:-qrow32}' in b4_runner
    assert 'LIVE_ARM=${FR13_QROW32_B1_LIVE_ARM:-split2}' in b1_runner
    assert 'FR13_FA2_QROW32_B1_PRODUCTION_ARM= \\' in b1_runner


def test_runtime_contract_selects_only_the_explicit_visibility_binary() -> None:
    contract = _module("scripts/fr13_fixed32_contract.py", "visibility_contract")
    common = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
    }
    b1_env = {
        **common,
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "visibility",
        "FR13_FA2_QROW32_B1_SO_SHA256": contract.QROW32_B1_VISIBILITY_FA2_SHA256,
    }
    assert contract._expected_runtime_fa2_identity(b1_env) == (
        contract.QROW32_B1_VISIBILITY_FA2_SIZE,
        contract.QROW32_B1_VISIBILITY_FA2_SHA256,
    )

    b4_env = {
        **common,
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "",
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "1",
        "FR13_FA2_QROW32_LIVE_PAGED_AB_ARM": "visibility",
        "FR13_FA2_QROW32_SO_SHA256": contract.QROW32_B4_VISIBILITY_FA2_SHA256,
    }
    assert contract._expected_runtime_fa2_identity(b4_env) == (
        contract.QROW32_B4_VISIBILITY_FA2_SIZE,
        contract.QROW32_B4_VISIBILITY_FA2_SHA256,
    )
    b4_env["FR13_FA2_QROW32_SO_SHA256"] = contract.QROW32_B4_GQA_PAIR_FA2_SHA256
    with pytest.raises(contract.ContractError, match="not the pinned candidate"):
        contract._expected_runtime_fa2_identity(b4_env)


def test_b1_visibility_contract_does_not_change_split2_source() -> None:
    sidecar = _module(
        "scripts/fr13_qrow32_b1_pass_sidecar.py", "visibility_b1_sidecar"
    )
    visibility = sidecar._candidate_contract("visibility")
    split2 = sidecar._candidate_contract("split2")
    assert visibility == {
        "sha256": sidecar.VISIBILITY_CANDIDATE_SHA256,
        "size": sidecar.VISIBILITY_CANDIDATE_SIZE,
        "source_files": sidecar.VISIBILITY_SOURCE_FILES,
        "source_closure_sha256": sidecar.VISIBILITY_SOURCE_CLOSURE_SHA256,
    }
    assert split2["sha256"] == sidecar.CANDIDATE_SHA256
    assert split2["size"] == 300_154_616
    assert split2["source_files"][
        "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu"
    ] == "223542ecf9bcc8837022aaceeca7468e4a8c866b528c4327c68f924dc4ab344d"


def test_b4_candidate_validator_closes_binary_and_regenerated_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _module(
        "scripts/fr13_fa2_fixed32_visibility_gate.py", "visibility_b4_gate"
    )
    candidate_raw = b"visibility candidate"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_raw)
    monkeypatch.setattr(gate, "CANDIDATE_SIZE", len(candidate_raw))
    monkeypatch.setattr(gate, "CANDIDATE_SHA256", _sha(candidate_raw))

    source = tmp_path / "fa2"
    contents = {
        "csrc/flash_attn/flash_api.cpp": b"api\n",
        "csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu": b"kernel\n",
    }
    for relative, raw in contents.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    files = {relative: _sha(raw) for relative, raw in contents.items()}
    status = tuple(f" M {relative}" for relative in contents)
    head = "1" * 40
    closure = {"fa2_head": head, "files": files}
    monkeypatch.setattr(gate, "SOURCE_FILES", files)
    monkeypatch.setattr(gate, "SOURCE_STATUS", status)
    monkeypatch.setattr(gate, "FA2_HEAD", head)
    monkeypatch.setattr(
        gate,
        "SOURCE_CLOSURE_SHA256",
        hashlib.sha256(gate._canonical_bytes(closure)).hexdigest(),
    )

    def fake_git(repo: Path, *args: str) -> str:
        assert repo == source
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "\n".join(status)
        raise AssertionError(args)

    monkeypatch.setattr(gate, "_git", fake_git)
    assert gate.validate_b4(candidate, source)["candidate_arm"] == "visibility"
    (source / next(iter(contents))).write_bytes(b"drift")
    with pytest.raises(gate.GateError, match="source hash drifted"):
        gate.validate_b4(candidate, source)
