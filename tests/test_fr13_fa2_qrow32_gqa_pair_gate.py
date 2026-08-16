from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_fa2_qrow32_gqa_pair_gate.py")
    spec = importlib.util.spec_from_file_location("fr13_gqa_pair_gate_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="ascii")
    return path


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_fixture(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str]:
    candidate_bytes = b"pinned gqa-pair candidate"
    candidate = _write(tmp_path / "candidate.so", candidate_bytes)
    monkeypatch.setattr(module, "CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module, "CANDIDATE_SHA256", _sha(candidate_bytes))

    fa2 = tmp_path / "fa2"
    pair_name = (
        "csrc/flash_attn/src/"
        "flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"
    )
    contents = {
        "csrc/flash_attn/flash_api.cpp": (
            module.FIXED32_QUERY_GQA_PAIR32_API_DECLARATION
            + "\n"
            + module.FIXED32_QUERY_GQA_PAIR32_API_GATE
        ).encode("ascii"),
        "csrc/flash_attn/flash_api_torch_lib.cpp": b"torch registration\n",
        "csrc/flash_attn/src/flash.h": b"api declarations\n",
        pair_name: module.FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT.encode("ascii"),
        "csrc/flash_attn/src/flash_fwd_kernel.h": b"kernel specialization\n",
        "csrc/flash_attn/src/utils.h": b"static head traits\n",
    }
    for relative, raw in contents.items():
        _write(fa2 / relative, raw)
    source_hashes = {name: _sha(raw) for name, raw in contents.items()}
    head = "1" * 40
    status = tuple(f" M {name}" for name in contents)
    closure = {"fa2_head": head, "files": source_hashes}
    monkeypatch.setattr(module, "SOURCE_FILES", source_hashes)
    monkeypatch.setattr(module, "SOURCE_STATUS", status)
    monkeypatch.setattr(module, "FA2_HEAD", head)
    monkeypatch.setattr(
        module,
        "SOURCE_CLOSURE_SHA256",
        module._sha256_bytes(module._canonical_bytes(closure)),
    )

    def fake_git(repo: Path, *args: str) -> str:
        assert repo == fa2
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "\n".join(status)
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git", fake_git)
    return candidate, fa2, head


def _summary(dtype: str, shape: list[int], byte_count: int) -> dict[str, object]:
    digest = _sha(f"{dtype}:{shape}".encode("ascii"))
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": byte_count,
        "raw_byte_mismatches": 0,
        "stock_sha256": digest,
        "candidate_sha256": digest,
    }


def _live_result(module, mode: str, source_commit: str) -> dict[str, object]:
    layers = []
    for layer_name in module.qrow32_gate.TARGET_LAYERS:
        layers.append(
            {
                "layer_name": layer_name,
                "output": _summary(
                    "torch.bfloat16", [128, 24, 256], 128 * 24 * 256 * 2
                ),
                "lse": _summary("torch.float32", [24, 128], 24 * 128 * 4),
                "slots": [
                    {
                        "slot": slot,
                        "query_rows": [slot * 32, (slot + 1) * 32],
                        "output": _summary(
                            "torch.bfloat16",
                            [32, 24, 256],
                            32 * 24 * 256 * 2,
                        ),
                        "lse": _summary(
                            "torch.float32", [24, 32], 24 * 32 * 4
                        ),
                    }
                    for slot in range(4)
                ],
            }
        )
    return {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "task_ids": list(module.qrow32_gate.TASK_IDS),
        "subset_sha256": module.qrow32_gate.EXACT4_SUBSET_SHA256,
        "concurrency": 4,
        "batch_size": 4,
        "physical_rows_per_slot": 32,
        "total_query_rows": 128,
        "fixed32_mode": mode,
        "candidate_arm": "gqa_pair",
        "selector_sentinel": module.FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL,
        "candidate_so_sha256": module.CANDIDATE_SHA256,
        "candidate_so_size": module.CANDIDATE_SIZE,
        "fa2_head": module.FA2_HEAD,
        "fa2_source_closure_sha256": module.SOURCE_CLOSURE_SHA256,
        "source_commit": source_commit,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "runtime_mode": "FULL",
        "layer_count": 16,
        "target_layers": list(module.qrow32_gate.TARGET_LAYERS),
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
        "layers": layers,
        "incumbent_dispatch": "stock FA2 exact geometry; no fallback",
        "candidate_dispatch": "qrow32 GQA-pair exact geometry; no fallback",
        "served_return": "stock captured graph output unchanged",
        "fallback_allowed": False,
        "performance_measurement": False,
    }


def _arm_args(
    module,
    tmp_path: Path,
    candidate: Path,
    fa2: Path,
    source_commit: str,
    mode: str,
) -> argparse.Namespace:
    result = _write(
        tmp_path / f"{mode}.live.json",
        json.dumps(_live_result(module, mode, source_commit)),
    )
    # The campaign summary the runner actually writes. The TAW campaign-arm
    # artifact this used to model is only emitted under
    # FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1, which this gate must not set.
    # Resolution counts are deliberately mixed: a raw-byte gate must pass
    # regardless of how many instances resolved.
    campaign_arm = _write(
        tmp_path / f"{mode}.arm.json",
        json.dumps(
            {
                "model_name_or_path": "qwen3.8-27b-nvfp4-radixark::qwen-code-0.19.4::q38-a",
                "started_at": "2026-08-10T05:42:06Z",
                "ended_at": "2026-08-10T06:33:40Z",
                "instances_total": 4,
                "verdict_counts": {"resolved": 1, "failed": 3},
                "resolved_rate": 0.25,
            }
        ),
    )
    campaign = _write(
        tmp_path / f"{mode}.provenance.json",
        json.dumps(
            {
                "schema": "fr13-fixed32-qwen-campaign-provenance-v1",
                "metric_scope": "concurrent_campaign_union",
                "concurrency": 4,
                "task_ids": list(module.qrow32_gate.TASK_IDS),
            }
        ),
    )
    return argparse.Namespace(
        result=result,
        campaign_arm=campaign_arm,
        campaign_provenance=campaign,
        candidate_so=candidate,
        fa2_source=fa2,
        fixed32_mode=mode,
        source_commit=source_commit,
    )


def test_candidate_identity_is_binary_and_regenerated_source_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate, fa2, _ = _identity_fixture(module, tmp_path, monkeypatch)
    result = module.validate_candidate(candidate, fa2)
    assert result["candidate_arm"] == "gqa_pair"
    assert result["selector_sentinel"] == 0x20014

    (fa2 / next(iter(module.SOURCE_FILES))).write_bytes(b"drift")
    with pytest.raises(module.GateError, match="source hash drifted"):
        module.validate_candidate(candidate, fa2)


@pytest.mark.parametrize("mode", ["tail6_fixed32", "hydra27_fixed32"])
def test_arm_gate_requires_all_layer_bf16_fp32_raw_byte_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    module = _module()
    candidate, fa2, source_commit = _identity_fixture(module, tmp_path, monkeypatch)
    args = _arm_args(module, tmp_path, candidate, fa2, source_commit, mode)
    verified = module.verify_arm(args)
    assert verified["status"] == "PASS"
    assert verified["logical_topology"] == (
        "Tail23" if mode == "tail6_fixed32" else "Hydra27"
    )
    assert verified["layer_count"] == 16
    assert verified["slot_coverage"] == [0, 1, 2, 3]

    payload = json.loads(args.result.read_text(encoding="ascii"))
    payload["layers"][5]["slots"][2]["lse"]["dtype"] = "torch.bfloat16"
    args.result.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(module.GateError, match="raw-byte contract drifted"):
        module.verify_arm(args)


def test_arm_gate_rejects_wrong_gqa_selector_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate, fa2, source_commit = _identity_fixture(module, tmp_path, monkeypatch)
    args = _arm_args(
        module, tmp_path, candidate, fa2, source_commit, "hydra27_fixed32"
    )
    payload = json.loads(args.result.read_text(encoding="ascii"))
    payload["selector_sentinel"] = 0x20013
    args.result.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(module.GateError, match="selector_sentinel drifted"):
        module.verify_arm(args)

    payload = _live_result(module, "hydra27_fixed32", source_commit)
    payload["layers"][0]["output"]["candidate_sha256"] = "0" * 64
    args.result.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(module.GateError, match="digest identity drifted"):
        module.verify_arm(args)


def test_dual_gate_hash_binds_tail23_and_hydra27(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate, fa2, source_commit = _identity_fixture(module, tmp_path, monkeypatch)
    tail_args = _arm_args(
        module, tmp_path, candidate, fa2, source_commit, "tail6_fixed32"
    )
    hydra_args = _arm_args(
        module, tmp_path, candidate, fa2, source_commit, "hydra27_fixed32"
    )
    tail = _write(
        tmp_path / "tail.verification.json",
        json.dumps(module.verify_arm(tail_args)),
    )
    hydra = _write(
        tmp_path / "hydra.verification.json",
        json.dumps(module.verify_arm(hydra_args)),
    )
    args = argparse.Namespace(
        tail_verification=tail,
        hydra_verification=hydra,
        candidate_so=candidate,
        fa2_source=fa2,
        source_commit=source_commit,
    )
    verified = module.verify_dual(args)
    assert verified["qualified_topologies"] == ["Tail23", "Hydra27"]
    assert verified["timing_eligible"] is False
    assert verified["production_eligible"] is False

    payload = json.loads(hydra.read_text(encoding="ascii"))
    original = dict(payload)
    payload.pop("live_result_sha256")
    hydra.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(module.GateError, match="Hydra27 GQA-pair arm"):
        module.verify_dual(args)

    payload = original
    payload["logical_topology"] = "Tail23"
    hydra.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(module.GateError, match="Hydra27 GQA-pair arm"):
        module.verify_dual(args)


def _declared(runner: str, key: str) -> str:
    """Read a key=value the runner bakes into its launcher_meta.txt banner."""
    matches = re.findall(rf"\\n{re.escape(key)}=([^\\%]*?)\\n", runner)
    assert matches, f"{key} is not declared in the runner banner"
    assert len(set(matches)) == 1, f"{key} is declared inconsistently: {matches}"
    return matches[0]


def test_dual_runner_is_default_off_real_exact4_and_non_timing() -> None:
    runner = Path(
        "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_live_gate.sh"
    ).read_text(encoding="ascii")
    assert "FR13_RUN_B4_QROW32_GQA_PAIR_LIVE_GATE:-0" in runner
    assert "run_arm tail6_fixed32 tail23" in runner
    assert "run_arm hydra27_fixed32 hydra27" in runner
    assert "FR13_QROW32_LIVE_AB_ARM=gqa_pair" in runner
    assert "task_count_per_topology=4" in runner
    assert "draft_vocab_k=65536" in runner
    assert "verify-dual" in runner

    # production_enabled is not a constant of the GQA-pair arm -- it is a
    # property of the RUN MODE. The byte gate always returns the stock captured
    # output, so it must declare 0 and must declare that it served the
    # reference; the timing pair actually serves the candidate on one arm, so
    # it must declare the opposite. Asserting the byte gate's 0 in isolation
    # would keep passing if the two ever silently swapped meaning, so the
    # invariant under test is the equivalence, checked in both directions.
    assert _declared(runner, "timing_eligible") == "0"
    assert _declared(runner, "production_enabled") == "0"
    assert _declared(runner, "reference_always_served") == "1"

    timing = Path(
        "scripts/fr13_run_b4_fa2_qrow32_gqa_pair_timing.sh"
    ).read_text(encoding="ascii")
    assert _declared(timing, "timing_eligible") == "1"
    assert "reference_always_served" not in timing
    assert "FR13_FA2_QROW32_B4_PRODUCTION_ARM=\"$production_arm\"" in timing

    patcher = Path("scripts/fr13_patch_fa2_tree_bias.py").read_text(
        encoding="ascii"
    )
    helper = patcher[patcher.index("FIXED32_QUERY_TILE32_LIVE_AB_HELPERS") :]
    assert '"gqa_pair": {' in helper
    assert '"sentinel": 131092' in helper
    assert 'FR13_FA2_QROW32_LIVE_PAGED_AB_ARM' in helper
    assert 'int(num_splits) == int(candidate_contract["num_splits"])' in helper
    assert "FR13 qrow32 binary/source provenance drifted" in helper
