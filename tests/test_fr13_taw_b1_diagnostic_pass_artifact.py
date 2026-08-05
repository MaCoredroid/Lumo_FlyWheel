from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_taw_b1_diagnostic_pass_20260731T162536Z"
)
SOURCE = ROOT / "scripts" / "fr13_device_multidraft_kernel.py"


def _load_taw_module():
    spec = importlib.util.spec_from_file_location(
        "fr13_taw_b1_diagnostic_pass_artifact",
        SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_at(commit: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "show",
            f"{commit}:scripts/fr13_device_multidraft_kernel.py",
        ],
        cwd=ROOT,
        text=True,
    )


def _candidate_math_projection(source: str, proof: dict) -> str:
    tree = ast.parse(source)
    definitions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constant_names = tuple(
        proof["candidate_math_projection"]["constant_ast_sha256"]
    )
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in constant_names:
                values[target.id] = node.value
    function_names = tuple(
        proof["candidate_math_projection"]["function_ast_sha256"]
    )
    payload = {
        "functions": {
            name: ast.dump(
                definitions[name],
                annotate_fields=True,
                include_attributes=False,
            )
            for name in function_names
        },
        "constants": {
            name: ast.dump(
                values[name],
                annotate_fields=True,
                include_attributes=False,
            )
            for name in constant_names
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_diagnostic_pass_cannot_arm_current_production() -> None:
    taw = _load_taw_module()
    payload = json.loads(
        (ARTIFACT / "diagnostic_pass.json").read_text(encoding="ascii")
    )
    requirement = json.loads(
        (ARTIFACT / "current_production_requirement.json").read_text(
            encoding="ascii"
        )
    )

    assert payload["status"] == "pass"
    assert payload["schema"] == (
        "fr13.fixed32.taw_native_precompute.diagnostic_pass.v1"
    )
    assert payload["production_eligible"] is False
    assert payload["source_contract_sha256"] == (
        "fe73ad35a916e41532575e29a5f9f6442d1081d0d1c0d0fc18210fdc8f0f56f8"
    )
    assert requirement["required_payload"]["source_contract_sha256"] == (
        taw._FR13_FIXED32_TAW_SOURCE_SHA256
    )
    assert payload["source_contract_sha256"] != (
        taw._FR13_FIXED32_TAW_SOURCE_SHA256
    )
    with pytest.raises(RuntimeError, match="different candidate/source"):
        taw._fr13_fixed32_taw_native_production_pass(
            path=str(ARTIFACT / "diagnostic_pass.json"),
            expected_mode="hydra27_fixed32",
            expected_batch=1,
        )


def test_old_candidate_math_projection_is_not_current_physical_slot_source() -> None:
    proof = json.loads(
        (ARTIFACT / "source_equivalence.json").read_text(encoding="ascii")
    )
    run_source = _source_at(proof["run_source"]["commit"])
    current_source = SOURCE.read_text(encoding="utf-8")

    run_projection = _candidate_math_projection(run_source, proof)
    current_projection = _candidate_math_projection(current_source, proof)
    assert run_projection == proof["candidate_math_projection"]["run_sha256"]
    assert current_projection == (
        proof["candidate_math_projection"]["current_sha256"]
    )
    assert run_projection != current_projection
    assert proof["candidate_math_projection"]["equal"] is False
    assert proof["conclusion"]["candidate_math_equivalent"] is False
    assert (
        proof["conclusion"][
            "diagnostic_evidence_informative_for_current_candidate"
        ]
        is False
    )
    assert proof["conclusion"]["full_source_contract_equivalent"] is False
    assert proof["conclusion"]["production_eligible"] is False


def test_run_evidence_is_b1_only_and_zero_mismatch() -> None:
    evidence = json.loads(
        (ARTIFACT / "run_evidence.json").read_text(encoding="ascii")
    )
    markers = evidence["marker_counts"]
    census = evidence["work_census"]

    assert evidence["status"] == "DIAGNOSTIC_BYTE_EQUIVALENCE_PASS"
    assert evidence["production_eligible"] is False
    assert evidence["mode"] == "hydra27_fixed32"
    assert evidence["batch_size"] == 1
    assert evidence["task"]["verdict"] == "resolved"
    assert markers["periodic_taw_pass"] == 5
    assert markers["periodic_root_checks"] == [128, 256, 384, 512, 640]
    assert markers["taw_mismatch"] == 0
    assert markers["current_schema_live_pass"] == 0
    assert census["event_rows"] == 762
    assert census["diagnostic_route_rows"] == 762
    assert census["run_source_contract_rows"] == 762
