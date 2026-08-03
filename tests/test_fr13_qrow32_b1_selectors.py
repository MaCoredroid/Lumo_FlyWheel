from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
TIMING_RUNNER = REPO / "scripts/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison(seed: str, dtype: str, shape: list[int]) -> dict[str, object]:
    digest = hashlib.sha256(seed.encode("ascii")).hexdigest()
    return {
        "dtype": dtype,
        "shape": shape,
        "bytes": 4096,
        "raw_byte_mismatches": 0,
        "stock_sha256": digest,
        "candidate_sha256": digest,
    }


def _live_payload(candidate_sha256: str, arm: str) -> dict[str, object]:
    config = {
        "no_split": (1179791668, 0, "not applicable"),
        "split2": (
            1179791669,
            2,
            "stock FA2 set_params_splitkv via num_splits=2",
        ),
    }[arm]
    layers = []
    for index in range(3, 64, 4):
        name = f"language_model.model.layers.{index}.self_attn.attn"
        layers.append(
            {
                "layer_name": name,
                "output": _comparison(name + "-o", "torch.bfloat16", [32, 24, 256]),
                "lse": _comparison(name + "-l", "torch.float32", [1, 24, 32]),
            }
        )
    return {
        "schema": "fr13.fixed32.fa2_qrow32_b1_live_paged_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": "astropy__astropy-12907",
        "concurrency": 1,
        "batch_size": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "runtime_mode": "FULL",
        "candidate_so_sha256": candidate_sha256,
        "arm": arm,
        "selector_sentinel": config[0],
        "candidate_num_splits": config[1],
        "split_scratch_allocation": config[2],
        "layer_count": 16,
        "layers": layers,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "served_return": "stock captured graph output unchanged",
        "performance_measurement": False,
    }


def test_selectors_are_independent_default_off_and_split2_uses_stock_api() -> None:
    text = PATCHER.read_text()
    helpers = text.split("FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS", 1)[1]
    production = helpers.split("def _fr13_fa2_qrow32_b1_production_begin", 1)[1]

    assert '"no_split": {"sentinel": 1179791668, "num_splits": 0}' in helpers
    assert '"split2": {"sentinel": 1179791669, "num_splits": 2}' in helpers
    assert 'os.environ.get(env_name, "")' in helpers
    assert '"--fixed32-query-tile32-b1-live-ab"' in text
    assert '"--fixed32-query-tile32-b1-production"' in text
    assert 'num_splits=(\n                                _fr13_qrow32_b1_selection["num_splits"]' in text
    assert "stock FA2 set_params_splitkv via num_splits=2" in helpers
    assert "torch.cuda.synchronize()" not in production
    assert '"candidate_served": True, "fallback_allowed": False' in helpers
    assert "FR13 qrow32 B1 production silently fell back" in helpers


def test_launcher_requires_real_k64_gate_and_arm_bound_exact4() -> None:
    text = LAUNCHER.read_text()

    assert "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty, no_split, or split2" in text
    assert "FR13 qrow32 B1 live gate requires the canonical K64/root1 real task" in text
    assert "fr13_qrow32_b1_pass_sidecar.py issue" in text
    assert "fr13_qrow32_b1_pass_sidecar.py verify" in text
    assert 'if [[ -n "\\${FR13_FA2_QROW32_B1_PRODUCTION_ARM}" ]]; then' in text
    assert "FR13_FA2_QROW32_B1_INTERNAL_ATTESTED=1" in text
    assert "--fixed32-query-tile32-b1-live-ab" in text
    assert "--fixed32-query-tile32-b1-production" in text
    assert "astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398" in text
    assert "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5" in text


def test_timing_runner_requires_arm_bound_real_exact4_engagement() -> None:
    text = TIMING_RUNNER.read_text()

    assert ': "${QROW32_B1_ARM:?set QROW32_B1_ARM to no_split or split2}"' in text
    assert "fr13_qrow32_b1_pass_sidecar.py verify" in text
    assert 'qrow.get("candidate_served") is not True' in text
    assert 'qrow.get("fallback_allowed") is not False' in text
    assert 'qrow.get("arm") != candidate_arm' in text
    assert 'qrow.get("num_splits") != (0 if candidate_arm == "no_split" else 2)' in text
    assert '"qrow32_b1_arm": candidate_arm' in text
    assert '"qrow32_b1_num_splits": 0 if candidate_arm == "no_split" else 2' in text


@pytest.mark.parametrize("arm", ["no_split", "split2"])
def test_sidecar_is_arm_bound(tmp_path: Path, arm: str) -> None:
    module = _module(SIDECAR, f"qrow32_sidecar_{arm}")
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes((arm + "-candidate").encode("ascii"))
    candidate_sha256 = module.sha256_file(candidate)
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(
            _live_payload(candidate_sha256, arm),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    live_sha256 = module.sha256_file(live)
    sidecar = tmp_path / "pass.json"

    issued = module.issue_sidecar(
        live_result=live,
        expected_live_sha256=live_sha256,
        candidate_so=candidate,
        expected_candidate_sha256=candidate_sha256,
        arm=arm,
        out=sidecar,
    )
    sidecar_sha256 = module.sha256_file(sidecar)
    verified = module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=sidecar_sha256,
        candidate_so=candidate,
        expected_candidate_sha256=candidate_sha256,
        arm=arm,
    )
    assert issued["arm"] == arm
    assert verified["arm"] == arm

    wrong_arm = "split2" if arm == "no_split" else "no_split"
    with pytest.raises(ValueError, match="contract drifted"):
        module.verify_sidecar(
            sidecar_path=sidecar,
            expected_sidecar_sha256=sidecar_sha256,
            candidate_so=candidate,
            expected_candidate_sha256=candidate_sha256,
            arm=wrong_arm,
        )
