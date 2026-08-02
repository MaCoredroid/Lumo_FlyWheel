from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SIDECAR_SCRIPT = REPO / "scripts" / "fr13_qrow16_pass_sidecar.py"
PATCHER = REPO / "scripts" / "fr13_patch_fa2_tree_bias.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _module():
    spec = importlib.util.spec_from_file_location("qrow_sidecar", SIDECAR_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
