from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
SIDECAR_SCRIPT = REPO / "scripts" / "fr13_qrow16_pass_sidecar.py"
PATCHER = REPO / "scripts" / "fr13_patch_fa2_tree_bias.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SFWD_GATE = REPO / "scripts" / "fr13_sfwd_conv_postprep_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("qrow_sidecar", SIDECAR_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sfwd_gate_module():
    spec = importlib.util.spec_from_file_location("sfwd_gate", SFWD_GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _patcher_literal(name: str) -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing patcher literal: {name}")


def _live(candidate_sha256: str) -> dict[str, object]:
    output_sha = _sha(b"output")
    lse_sha = _sha(b"lse")
    return {
        "schema": "fr13.fixed32.fa2_qrow16_live_paged_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": "astropy__astropy-12907",
        "concurrency": 1,
        "physical_rows": 32,
        "candidate_so_sha256": candidate_sha256,
        "graph_id": 123,
        "runtime_mode": "FULL",
        "layer_name": "language_model.model.layers.3.self_attn",
        "operands": {
            "query_shape": [32, 24, 256],
            "key_cache_shape": [8, 1024, 4, 256],
            "value_cache_shape": [8, 1024, 4, 256],
            "block_table_shape": [1, 8],
            "query_start_loc": [0, 32],
            "seq_lens": [4096],
            "max_seqlen_k": 4096,
            "tree_bias_shape": [1, 32, 32],
        },
        "output": {
            "dtype": "torch.bfloat16",
            "bytes": 393216,
            "raw_byte_mismatches": 0,
            "stock_sha256": output_sha,
            "candidate_sha256": output_sha,
        },
        "lse": {
            "dtype": "torch.float32",
            "bytes": 3072,
            "raw_byte_mismatches": 0,
            "stock_sha256": lse_sha,
            "candidate_sha256": lse_sha,
        },
        "candidate_dispatch": "qrow16 internal exact-geometry require",
        "served_return": "stock captured graph output unchanged",
        "performance_measurement": False,
    }


def test_issue_and_verify_bind_live_pass_to_exact_candidate(tmp_path: Path) -> None:
    module = _module()
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"exact qrow candidate")
    candidate_sha = module.sha256_file(candidate)
    live = tmp_path / "live.json"
    live.write_text(json.dumps(_live(candidate_sha), sort_keys=True) + "\n")
    live_sha = module.sha256_file(live)
    sidecar = tmp_path / "production-pass.json"

    issued = module.issue_sidecar(
        live_result=live,
        expected_live_sha256=live_sha,
        candidate_so=candidate,
        expected_candidate_sha256=candidate_sha,
        out=sidecar,
    )

    assert issued["status"] == "PASS"
    assert issued["candidate_so_sha256"] == candidate_sha
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    verified = module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=module.sha256_file(sidecar),
        candidate_so=candidate,
        expected_candidate_sha256=candidate_sha,
    )
    assert verified == issued
    with pytest.raises(ValueError, match="pass sidecar raw SHA-256 mismatch"):
        module.verify_sidecar(
            sidecar_path=sidecar,
            expected_sidecar_sha256="0" * 64,
            candidate_so=candidate,
            expected_candidate_sha256=candidate_sha,
        )
    candidate.write_bytes(b"different build")
    with pytest.raises(ValueError, match="attested candidate SO SHA-256 mismatch"):
        module.verify_sidecar(
            sidecar_path=sidecar,
            expected_sidecar_sha256=module.sha256_file(sidecar),
            candidate_so=candidate,
            expected_candidate_sha256=candidate_sha,
        )


def test_sidecar_rejects_mismatch_and_wrong_build(tmp_path: Path) -> None:
    module = _module()
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"candidate")
    candidate_sha = module.sha256_file(candidate)
    payload = _live(candidate_sha)
    payload["output"]["raw_byte_mismatches"] = 1  # type: ignore[index]
    live = tmp_path / "live.json"
    live.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="output comparison drifted"):
        module.issue_sidecar(
            live_result=live,
            expected_live_sha256=module.sha256_file(live),
            candidate_so=candidate,
            expected_candidate_sha256=candidate_sha,
            out=tmp_path / "pass.json",
        )
    with pytest.raises(ValueError, match="candidate SO SHA-256 mismatch"):
        module.issue_sidecar(
            live_result=live,
            expected_live_sha256=module.sha256_file(live),
            candidate_so=candidate,
            expected_candidate_sha256="0" * 64,
            out=tmp_path / "wrong.json",
        )


def test_production_selector_is_default_off_attested_and_fail_closed() -> None:
    patcher = PATCHER.read_text()
    launcher = LAUNCHER.read_text()

    assert 'os.environ.get("FR13_FA2_QROW16_PRODUCTION", "0") != "1"' in patcher
    assert "FR13_FA2_QROW16_INTERNAL_PRODUCTION_ATTESTED" in patcher
    assert "FR13 qrow16 production geometry drifted" in patcher
    assert "FR13 qrow16 production did not capture all target tree layers" in patcher
    assert "len(layers) != 16" in patcher
    assert "FR13_FA2_QROW16_PRODUCTION_CAPTURE_END" in patcher
    assert "_FR13_FA2_QROW16_BATCH_STRIDE_SENTINEL = 1179791667" in patcher
    assert "torch.as_strided(" in patcher
    assert "FR13_FA2_QROW16_INTERNAL_DISPATCH" not in patcher
    assert "_fr13_qrow16_production_bias" in patcher
    assert "_fr13_fa2_qrow16_production_end" in patcher
    assert '"--fixed32-query-tile16-production"' in patcher

    assert "FR13_FA2_QROW16_PRODUCTION=${FR13_FA2_QROW16_PRODUCTION:-0}" in launcher
    assert "qrow16 live A/B and production are mutually exclusive" in launcher
    assert "fr13_qrow16_pass_sidecar.py issue" in launcher
    assert "fr13_qrow16_pass_sidecar.py verify" in launcher
    assert "FR13_FA2_QROW16_INTERNAL_PRODUCTION_ATTESTED=1" in launcher
    assert "--fixed32-query-tile16-production" in launcher


def test_qrow16_eager_selector_admits_only_one_authenticated_sfwd_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = {"os": os, "torch": torch}
    exec(
        _patcher_literal("FIXED32_QUERY_TILE16_PRODUCTION_HELPERS"),
        namespace,
    )
    begin = namespace["_fr13_fa2_qrow16_production_begin"]
    exact = {
        "layer": type("Layer", (), {"layer_name": "layer.0"})(),
        "query": torch.empty((32, 24, 256), dtype=torch.bfloat16),
        "key_cache": torch.empty((1, 1024, 4, 256), dtype=torch.bfloat16),
        "value_cache": torch.empty((1, 1024, 4, 256), dtype=torch.bfloat16),
        "cu_seqlens_q": torch.tensor((0, 32), dtype=torch.int32),
        "max_seqlen_q": 32,
        "seqused_k": torch.tensor((32,), dtype=torch.int32),
        "max_seqlen_k": 32,
        "causal": False,
        "window_size": (-1, -1),
        "block_table": torch.zeros((1, 1), dtype=torch.int32),
        "num_splits": 0,
        "tree_bias": torch.zeros((32, 32), dtype=torch.float32),
    }
    for name, value in {
        "FR13_FA2_QROW16_PRODUCTION": "1",
        "FR13_FA2_QROW16_INTERNAL_PRODUCTION_ATTESTED": "1",
        "FR13_FA2_QROW16_PRODUCTION_PASS_SIDECAR_SHA256": "1" * 64,
        "FR13_FA2_QROW16_SO_SHA256": "2" * 64,
        "ENFORCE_EAGER": "1",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB": "1",
    }.items():
        monkeypatch.setenv(name, value)

    candidate = begin(**exact)
    assert candidate is not None
    assert tuple(candidate.shape) == (1, 32, 32)

    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "1")
    with pytest.raises(RuntimeError, match="eager SFWD routes overlap"):
        begin(**exact)

    monkeypatch.setenv("FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0")
    monkeypatch.setenv("FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB", "0")
    assert begin(**exact) is None


def test_sfwd_gate_binds_qrow16_to_conv_postprep_eager_engagement(
    tmp_path: Path,
) -> None:
    module = _sfwd_gate_module()
    sidecar = {
        "schema": "fr13.fixed32.fa2_qrow16_production_pass.v1",
        "status": "PASS",
        "candidate_so_sha256": module.QROW16_SHA256,
        "live_result_sha256": module.QROW16_PASS_SHA256,
    }
    sidecar_path = tmp_path / "fr13_fa2_qrow16_production_pass.json"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n", encoding="ascii"
    )
    capture = {
        "schema": "fr13.fixed32.fa2_qrow16_eager_production_engagement.v1",
        "status": "ENGAGED",
        "runtime_mode": "EAGER",
        "batch_size": 1,
        "layers": [f"layer.{index}" for index in range(16)],
        "layer_count": 16,
        "candidate_so_sha256": module.QROW16_SHA256,
        "pass_sidecar_sha256": hashlib.sha256(
            sidecar_path.read_bytes()
        ).hexdigest(),
        "dispatch": "qrow16 exact geometry; no fallback",
        "sfwd_state_fusion_production": False,
        "sfwd_conv_postprep_byte_ab": True,
    }
    capture_path = tmp_path / "fr13_fa2_qrow16_production_capture.json"
    capture_path.write_text(
        json.dumps(capture, sort_keys=True) + "\n", encoding="ascii"
    )

    sidecar_raw, capture_raw = module._validate_qrow_evidence(tmp_path)
    assert sidecar_raw == sidecar_path.read_bytes()
    assert capture_raw == capture_path.read_bytes()

    capture["sfwd_conv_postprep_byte_ab"] = False
    capture_path.write_text(
        json.dumps(capture, sort_keys=True) + "\n", encoding="ascii"
    )
    with pytest.raises(module.GateError, match="production evidence drifted"):
        module._validate_qrow_evidence(tmp_path)
