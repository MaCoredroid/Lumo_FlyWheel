from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
LIVE_GATE = REPO / "scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh"
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
        "reference_sha256": digest,
        "candidate_sha256": digest,
    }


def _live_payload(
    module, candidate_sha256: str, source_commit: str, patch_sha256: str
) -> dict[str, object]:
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
        "schema": module.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": module.EXACT4_TASK_IDS[0],
        "concurrency": 1,
        "batch_size": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "runtime_mode": "FULL",
        "candidate_so_sha256": candidate_sha256,
        "candidate_so_size": module.CANDIDATE_SIZE,
        "arm": "split2",
        "selector_sentinel": 1179791669,
        "candidate_num_splits": 2,
        "split_scratch_allocation": "stock FA2 set_params_splitkv via num_splits=2",
        "reference_selector_sentinel": 1179791667,
        "reference_dispatch": "qrow16 incumbent exact geometry; no fallback",
        "candidate_dispatch": "qrow32 B1 split2 exact geometry; no fallback",
        "fa2_head": module.FA2_HEAD,
        "fa2_source_closure_sha256": module.SOURCE_CLOSURE_SHA256,
        "source_commit": source_commit,
        "patch_source_sha256": patch_sha256,
        "layer_count": 16,
        "layers": layers,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "served_return": "qrow16 captured graph output unchanged",
        "performance_measurement": False,
    }


def test_selectors_are_split2_only_default_off_and_qrow16_served() -> None:
    text = PATCHER.read_text()
    helpers = text.split("FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS", 1)[1]
    production = helpers.split("def _fr13_fa2_qrow32_b1_production_begin", 1)[1]

    assert '"split2": {"sentinel": 1179791669, "num_splits": 2}' in helpers
    assert '"no_split": {"sentinel": 1179791668' not in helpers
    assert 'os.environ.get(env_name, "")' in helpers
    assert 'tree_bias = _fr13_fa2_qrow32_b1_live_register(' in text
    assert "_FR13_FA2_QROW32_B1_QROW16_REFERENCE_SENTINEL = 1179791667" in helpers
    assert '"reference_sha256"' in helpers
    assert '"served_return": "qrow16 captured graph output unchanged"' in helpers
    assert "torch.cuda.synchronize()" not in production
    assert '"candidate_served": True, "fallback_allowed": False' in helpers
    assert "FR13 qrow32 B1 production silently fell back" in helpers


def test_launcher_requires_exact_binary_source_graph_and_real_gate() -> None:
    text = LAUNCHER.read_text()

    assert "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty or split2" in text
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty or split2" in text
    assert "FR13 qrow32 B1 live gate requires the canonical K64/root1 real task" in text
    assert '"${FR13_FIXED32_MODE:-}" == "hydra27_fixed32"' in text
    assert '"${ENFORCE_EAGER:-0}" == "0"' in text
    assert '"${CUDAGRAPH_MODE:-}" == "FULL_AND_PIECEWISE"' in text
    assert "5eec90f317cf6126cd57ab7f77b392ae6a1430d28210dcb31756abe788ef3467" in text
    assert "c10888e721335ff99f93dabdfea7d8a524fbd7e21e8aee3f425f50af06bf5d84" in text
    assert "--patch-source scripts/fr13_patch_fa2_tree_bias.py" in text
    assert "--patch-source /workspace/scripts/fr13_patch_fa2_tree_bias.py" in text
    assert "FR13_FA2_QROW32_B1_INTERNAL_ATTESTED=1" in text
    assert "astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398" in text


def test_live_gate_is_authenticated_one_task_non_timing_qrow16_served() -> None:
    text = LIVE_GATE.read_text()

    assert "FR13_RUN_QROW32_SPLIT2_LIVE_GATE" in text
    assert "subset_b1_diagnostic_one.json" in text
    assert "fixed32_chat_traffic_audit.json" in text
    assert "all(value is True for value in checks.values())" not in text
    assert "any(value is not True for value in checks.values())" in text
    assert '"served_return": "qrow16 captured graph output unchanged"' in text
    assert "FR13_SFWD_GPU_TIMER=0" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "fr13_qrow32_b1_pass_sidecar.py validate-source" in text
    assert 'PYTHONPATH="$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"' in text


def test_live_gate_inline_contract_import_resolves_from_repo_root() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "scripts")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts import fr13_fixed32_contract, fr13_qrow32_b1_pass_sidecar",
        ],
        cwd=REPO,
        env=environment,
        check=True,
    )


def test_timing_runner_is_pass_gated_exact4_graph_only() -> None:
    text = TIMING_RUNNER.read_text()

    assert "FR13_RUN_QROW32_SPLIT2_TIMING" in text
    assert ': "${QROW32_B1_PASS:?set QROW32_B1_PASS' in text
    assert "fr13_qrow32_b1_pass_sidecar.py validate-source" in text
    assert "fr13_qrow32_b1_pass_sidecar.py verify" in text
    assert "FR13_FA2_QROW32_B1_PRODUCTION_ARM=split2" in text
    assert "ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "fr13_qrow32_split2_timing.py" in text
    assert "exact16_rule=only_after_exact4_u95_clears_cap" in text
    assert "subset_b4_four.json" in text
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0" in text


def test_sidecar_is_binary_source_and_split2_bound(tmp_path: Path) -> None:
    module = _module(SIDECAR, "qrow32_sidecar")
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"split2-candidate")
    module.CANDIDATE_SIZE = candidate.stat().st_size
    module.CANDIDATE_SHA256 = module.sha256_file(candidate)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    patch_sha256 = module.sha256_file(PATCHER)
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(
            _live_payload(module, module.CANDIDATE_SHA256, source_commit, patch_sha256),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    sidecar = tmp_path / "pass.json"
    issued = module.issue_sidecar(
        live_result=live,
        expected_live_sha256=module.sha256_file(live),
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm="split2",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
        out=sidecar,
    )
    verified = module.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=module.sha256_file(sidecar),
        candidate_so=candidate,
        expected_candidate_sha256=module.CANDIDATE_SHA256,
        arm="split2",
        patch_source=PATCHER,
        expected_source_commit=source_commit,
    )
    assert issued["arm"] == "split2"
    assert verified["source_commit"] == source_commit
    assert verified["patch_source_sha256"] == patch_sha256

    with pytest.raises(ValueError, match="must be split2"):
        module.validate_live_result(
            _live_payload(
                module, module.CANDIDATE_SHA256, source_commit, patch_sha256
            ),
            candidate_sha256=module.CANDIDATE_SHA256,
            arm="no_split",
        )
