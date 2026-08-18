from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_fa2_qrow32_gate.py")
    spec = importlib.util.spec_from_file_location("fr13_fa2_qrow32_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="ascii")
    return path


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module()
    fa2 = tmp_path / "fa2"
    source = fa2 / "csrc/flash_attn/src"
    contents = {
        "flash_fwd_launch_template.h": b"launch\n",
        "flash_fwd_kernel.h": b"// FR13_FA2_TREE_BIAS_TILE_EARLYOUT\n",
        "flash_fwd_split_hdim256_bf16_sm80.cu": b"stock\n",
    }
    for name, data in contents.items():
        _write(source / name, data)
    monkeypatch.setattr(
        module,
        "EXACT_SAFE_SOURCE_SHA256",
        {name: _sha(data) for name, data in contents.items()},
    )
    _write(
        source / "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
        module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT,
    )
    _write(
        fa2 / "csrc/flash_attn/flash_api.cpp",
        module.FIXED32_QUERY_TILE32_API_DECLARATION
        + module.FIXED32_QUERY_TILE32_API_GATE,
    )
    _write(
        fa2 / "CMakeLists.txt",
        'file(GLOB FA2_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")\n',
    )
    stock_object = _write(tmp_path / "stock.o", b"stock object")
    qrow_object = _write(
        tmp_path / "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu.o",
        b"qrow object",
    )
    monkeypatch.setattr(module, "EXACT_SAFE_STOCK_OBJECT_SHA256", _sha(b"stock object"))
    final = _write(tmp_path / "final.txt", "ninja: no work to do.\n")
    return module, fa2, stock_object, qrow_object, final


def test_build_gate_accepts_fresh_configure_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, fa2, stock_object, qrow_object, final = _build_fixture(
        tmp_path, monkeypatch
    )
    manifest = _write(
        tmp_path / "build.ninja",
        "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu\n",
    )
    initial = _write(
        tmp_path / "initial.txt",
        "[1/2] Building CUDA object flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu.o\n"
        "[2/2] Linking CXX shared library _vllm_fa2_C.abi3.so\n",
    )
    args = argparse.Namespace(
        fa2_src=fa2,
        stock_object=stock_object,
        qrow_object=qrow_object,
        build_manifest=manifest,
        explicit_compile_log=None,
        initial_dry_run=initial,
        explicit_link_log=None,
        final_dry_run=final,
        output=None,
    )
    result = module.verify_build(args)
    assert result["status"] == "PASS"
    assert result["object_discovery_route"] == "fresh_configure_discovered_object"
    assert result["required_patch_flags"] == [
        "--tree-bias-tile-earlyout",
        "--fixed32-query-tile32",
    ]


def test_build_gate_accepts_explicit_object_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, fa2, stock_object, qrow_object, final = _build_fixture(
        tmp_path, monkeypatch
    )
    compile_log = _write(
        tmp_path / "compile.log",
        "nvcc -c csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu "
        f"-o {qrow_object.name}\n",
    )
    link_log = _write(
        tmp_path / "link.log",
        f"c++ -shared api.o {qrow_object.name} -o _vllm_fa2_C.abi3.so\n",
    )
    args = argparse.Namespace(
        fa2_src=fa2,
        stock_object=stock_object,
        qrow_object=qrow_object,
        build_manifest=None,
        explicit_compile_log=compile_log,
        initial_dry_run=None,
        explicit_link_log=link_log,
        final_dry_run=final,
        output=None,
    )
    result = module.verify_build(args)
    assert result["object_discovery_route"] == "explicit_object_compile_and_append"


def _live_result(module, candidate_sha256: str, source_commit: str):
    layer_records = []
    for layer_name in module.TARGET_LAYERS:
        layer_records.append(
            {
                "layer_name": layer_name,
                "output": {"raw_byte_mismatches": 0},
                "lse": {"raw_byte_mismatches": 0},
                "slots": [
                    {
                        "slot": slot,
                        "output": {"raw_byte_mismatches": 0},
                        "lse": {"raw_byte_mismatches": 0},
                    }
                    for slot in range(4)
                ],
            }
        )
    return {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "task_ids": list(module.TASK_IDS),
        "subset_sha256": module.EXACT4_SUBSET_SHA256,
        "concurrency": 4,
        "batch_size": 4,
        "physical_rows_per_slot": 32,
        "total_query_rows": 128,
        "fixed32_mode": "hydra27_fixed32",
        "candidate_so_sha256": candidate_sha256,
        "source_commit": source_commit,
        "runtime_mode": "FULL",
        "layer_count": 16,
        "target_layers": list(module.TARGET_LAYERS),
        "stock_calls": 16,
        "candidate_calls": 16,
        "operands": {
            "query_shape": [128, 24, 256],
            "query_start_loc": [0, 32, 64, 96, 128],
            "slot_coverage": [0, 1, 2, 3],
            "key_cache_tail_shape": [1024, 4, 256],
            "seq_lens": [1000, 2000, 3000, 4000],
            "suffix_start_mod64": [8, 48, 24, 0],
        },
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "layers": layer_records,
        "served_return": "stock captured graph output unchanged",
        "fallback_allowed": False,
        "performance_measurement": False,
    }


def test_live_verifier_requires_all_layers_slots_and_real_exact4(tmp_path: Path) -> None:
    module = _module()
    candidate = _write(tmp_path / "candidate.so", b"candidate")
    candidate_sha256 = _sha(b"candidate")
    source_commit = "a" * 40
    result_path = _write(
        tmp_path / "live.json",
        json.dumps(_live_result(module, candidate_sha256, source_commit)),
    )
    campaign_arm = _write(
        tmp_path / "campaign-arm.json",
        json.dumps(
            {
                "schema": "fr13-fixed32-taw-campaign-arm-v1",
                "state": "ended",
                "run_classification": "b4_taw_diagnostic",
                "batch_size": 4,
                "concurrency": 4,
                "task_count": 4,
                "instances_total": 4,
                "started_at": "2026-08-10T00:00:00Z",
                "ended_at": "2026-08-10T01:00:00Z",
                "verdict_counts": {"resolved": 3, "unresolved": 1},
                "subset_sha256": module.EXACT4_SUBSET_SHA256,
                "task_ids": list(module.TASK_IDS),
                "marker": (
                    "swe_verified:campaign4_" + module.EXACT4_SUBSET_SHA256
                ),
            }
        ),
    )
    campaign = _write(
        tmp_path / "campaign.json",
        json.dumps(
            {
                "schema": "fr13-fixed32-qwen-campaign-provenance-v1",
                "metric_scope": "concurrent_campaign_union",
                "concurrency": 4,
                "task_ids": list(module.TASK_IDS),
            }
        ),
    )
    args = argparse.Namespace(
        result=result_path,
        campaign_arm=campaign_arm,
        campaign_provenance=campaign,
        candidate_so=candidate,
        fixed32_mode="hydra27_fixed32",
        source_commit=source_commit,
    )
    verification = module.verify_live(args)
    assert verification["status"] == "PASS"
    assert verification["layer_count"] == 16
    assert verification["slot_coverage"] == [0, 1, 2, 3]

    tampered = _live_result(module, candidate_sha256, source_commit)
    tampered["batch_size"] = 4.0
    result_path.write_text(json.dumps(tampered), encoding="ascii")
    with pytest.raises(module.GateError, match="batch_size drifted"):
        module.verify_live(args)

    tampered = _live_result(module, candidate_sha256, source_commit)
    tampered["output_raw_byte_mismatches"] = False
    result_path.write_text(json.dumps(tampered), encoding="ascii")
    with pytest.raises(module.GateError, match="output_raw_byte_mismatches drifted"):
        module.verify_live(args)

    tampered = _live_result(module, candidate_sha256, source_commit)
    tampered["layers"][7]["slots"][2]["lse"]["raw_byte_mismatches"] = 1
    result_path.write_text(json.dumps(tampered), encoding="ascii")
    with pytest.raises(module.GateError, match="per-slot byte mismatch"):
        module.verify_live(args)

    result_path.write_text(
        json.dumps(_live_result(module, candidate_sha256, source_commit)),
        encoding="ascii",
    )
    arm_payload = json.loads(campaign_arm.read_text(encoding="ascii"))
    arm_payload["instances_total"] = 4.0
    campaign_arm.write_text(json.dumps(arm_payload), encoding="ascii")
    with pytest.raises(
        module.GateError, match="completed canonical exact4 campaign"
    ):
        module.verify_live(args)

    arm_payload["instances_total"] = 4
    arm_payload["verdict_counts"] = {"resolved": 2, "unresolved": 1}
    campaign_arm.write_text(json.dumps(arm_payload), encoding="ascii")
    with pytest.raises(
        module.GateError, match="completed canonical exact4 campaign"
    ):
        module.verify_live(args)
