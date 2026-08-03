from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
CANDIDATE = ROOT / "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py"
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
MANIFEST = ROOT / "scripts/fr13_runtime_manifest.py"
HOST_CREDENTIAL = (
    ROOT / "scripts/fr13_gdn_gqa_group3_production_credential.py"
)


def _kernel_function_source(name: str) -> str:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _resolver_namespace() -> dict[str, object]:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_SIDECARS",
        "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH_SIDECARS",
        "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_PASS",
    }
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            == "_fr13_resolve_fixed32_gdn_gqa_group3_production"
        )
    ]
    namespace: dict[str, object] = {
        "__file__": os.fspath(KERNEL),
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "Path": Path,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            KERNEL,
            "exec",
        ),
        namespace,
    )
    namespace["_fr13_fixed32_gdn_gqa_group3_source_sha256"] = (
        lambda: _combined_source_sha256()
    )
    return namespace


def _combined_source_sha256() -> str:
    payload = (
        b"fr10_gdn_tree_kernel.py\0"
        + KERNEL.read_bytes()
        + b"\0fr13_gdn_gqa_group3.py\0"
        + CANDIDATE.read_bytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _host_credential_module():
    spec = importlib.util.spec_from_file_location(
        "fr13_gdn_gqa_group3_production_credential_test",
        HOST_CREDENTIAL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _credential(mode: str = "hydra27_fixed32", batch: int = 1) -> dict[str, object]:
    is_hydra = mode == "hydra27_fixed32"
    return {
        "schema": "fr13.fixed32.gdn_single_launch.real_task_credential.v3",
        "status": "PASS",
        "credential_scope": f"{'hydra27' if is_hydra else 'tail23'}:b{batch}",
        "run_classification": (
            "one_real_swe_verified_b1_graph_byte_diagnostic"
            if batch == 1
            else "real_swe_verified_exact4_b4_graph_byte_diagnostic"
        ),
        "runtime_profile": "fixed32",
        "runtime_sequence": "scripts/fr13_fixed32_floor_timers_seq.sh",
        "candidate": "fixed32_gdn_single_launch_gqa_group3_v1",
        "reference": "fixed32_gdn_two_launch_reference_v1",
        "mode": mode,
        "logical_topology": "Hydra27" if is_hydra else "Tail23",
        "logical_drafts": 27 if is_hydra else 23,
        "valid_mask": 0x7ABDFFFF if is_hydra else 0x7A9CE7FF,
        "batch_size": batch,
        "expected_batch": batch,
        "concurrency": batch,
        "task_ids": (
            ["astropy__astropy-12907"]
            if batch == 1
            else [
                "astropy__astropy-12907",
                "astropy__astropy-13033",
                "astropy__astropy-13236",
                "astropy__astropy-13398",
            ]
        ),
        "physical_rows": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": (
            "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
        ),
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "raw_byte_equal": True,
        "reference_served": True,
        "state_restored": True,
        "task_completion_verified": True,
        "finalized_ingress_verified": True,
        "runtime_launch_end_stable": True,
        "external_launch_end_stable": True,
        "graph_work_census_verified": True,
        "batch_specific_pass_verified": True,
        "production_enabled": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "floor_acceptance_eligible": False,
        "arm_git_head_source_commit_match": True,
        "source_commit_git_show_verified": True,
        "source_closure_sha256": "1" * 64,
        "kernel_source_sha256": hashlib.sha256(KERNEL.read_bytes()).hexdigest(),
        "gqa_group3_source_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "candidate_source_sha256": _combined_source_sha256(),
        "source_commit": "a" * 40,
        "runtime_manifest_source_commit": "a" * 40,
        "runtime_manifest_canonical_sha256": "2" * 64,
        "runtime_manifest_sha256": "3" * 64,
        "entrypoint_sha256": "4" * 64,
        "common_runner_sha256": "5" * 64,
        "reducer_sha256": "6" * 64,
    }


def test_host_credential_is_exact_head_candidate_profile_mode_batch_bound(
    tmp_path: Path,
) -> None:
    module = _host_credential_module()
    path = tmp_path / "credential.json"
    valid = _credential()
    path.write_text(json.dumps(valid, sort_keys=True) + "\n", encoding="ascii")
    result = module.validate_credential(
        path,
        source_commit="a" * 40,
        profile="fixed32",
        mode="hydra27_fixed32",
        batch=1,
    )
    assert result["source_commit"] == "a" * 40

    mutations = {
        "source_commit": "b" * 40,
        "runtime_manifest_source_commit": "b" * 40,
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "runtime_profile": "other",
        "mode": "tail6_fixed32",
        "batch_size": 4,
    }
    for key, value in mutations.items():
        stale = dict(valid)
        stale[key] = value
        path.write_text(
            json.dumps(stale, sort_keys=True) + "\n", encoding="ascii"
        )
        with pytest.raises(module.CredentialError, match=key):
            module.validate_credential(
                path,
                source_commit="a" * 40,
                profile="fixed32",
                mode="hydra27_fixed32",
                batch=1,
            )


@pytest.mark.parametrize(
    ("mode", "batch"),
    (
        ("hydra27_fixed32", 1),
        ("tail6_fixed32", 4),
        ("hydra27_fixed32", 4),
    ),
)
def test_host_credential_accepts_only_supported_b1_b4_profiles(
    tmp_path: Path, mode: str, batch: int
) -> None:
    module = _host_credential_module()
    path = tmp_path / "credential.json"
    path.write_text(
        json.dumps(_credential(mode, batch), sort_keys=True) + "\n",
        encoding="ascii",
    )
    result = module.validate_credential(
        path,
        source_commit="a" * 40,
        profile="fixed32",
        mode=mode,
        batch=batch,
    )
    assert (result["mode"], result["batch_size"]) == (mode, batch)


def test_host_credential_rejects_unsupported_profile_pair_and_duplicate_json(
    tmp_path: Path,
) -> None:
    module = _host_credential_module()
    path = tmp_path / "credential.json"
    path.write_text(
        json.dumps(_credential(), sort_keys=True) + "\n", encoding="ascii"
    )
    with pytest.raises(module.CredentialError, match="runtime profile"):
        module.validate_credential(
            path,
            source_commit="a" * 40,
            profile="other",
            mode="hydra27_fixed32",
            batch=1,
        )
    with pytest.raises(module.CredentialError, match="mode/batch"):
        module.validate_credential(
            path,
            source_commit="a" * 40,
            profile="fixed32",
            mode="tail6_fixed32",
            batch=1,
        )
    path.write_text('{"schema":"x","schema":"y"}\n', encoding="ascii")
    with pytest.raises(module.CredentialError, match="duplicate JSON key"):
        module.validate_credential(
            path,
            source_commit="a" * 40,
            profile="fixed32",
            mode="hydra27_fixed32",
            batch=1,
        )


def test_production_resolver_is_default_off_and_exact_source_batch_bound(
    tmp_path: Path,
) -> None:
    namespace = _resolver_namespace()
    resolve = namespace[
        "_fr13_resolve_fixed32_gdn_gqa_group3_production"
    ]
    common = {
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
    }
    assert resolve(
        "hydra27_fixed32",
        environ=common,
        arm_sidecars=(),
        batch_sidecars=(),
        geom_override={"BV": 8},
    ) is None
    assert resolve(
        "hydra27_fixed32",
        environ={
            **common,
            "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "0",
        },
        arm_sidecars=(),
        batch_sidecars=(),
        geom_override={"BV": 8},
    ) is None

    credential_path = tmp_path / "credential.json"
    credential_path.write_text(
        json.dumps(_credential(), sort_keys=True) + "\n", encoding="ascii"
    )
    armed = {
        **common,
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "1",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH": "1",
    }
    result = resolve(
        "hydra27_fixed32",
        environ=armed,
        arm_sidecars=(),
        batch_sidecars=(),
        geom_override={"BV": 8},
        pass_path=os.fspath(credential_path),
    )
    assert result["candidate"] == "fixed32_gdn_single_launch_gqa_group3_v1"

    stale = _credential()
    stale["candidate_source_sha256"] = "0" * 64
    credential_path.write_text(
        json.dumps(stale, sort_keys=True) + "\n", encoding="ascii"
    )
    with pytest.raises(RuntimeError, match="invalid, stale"):
        resolve(
            "hydra27_fixed32",
            environ=armed,
            arm_sidecars=(),
            batch_sidecars=(),
            geom_override={"BV": 8},
            pass_path=os.fspath(credential_path),
        )


def test_production_dispatch_replaces_full_graph_launch_only_for_credential_batch() -> None:
    b1 = _kernel_function_source("launch_tree_gdn_prepared")
    b4 = _kernel_function_source("launch_tree_gdn_prepared_fixed32_batch")
    selector = _kernel_function_source("fixed32_batch_gdn_selector")
    preseed = _kernel_function_source("subtree_preseed")
    candidate_launch = CANDIDATE.read_text(encoding="utf-8")
    candidate_tree = ast.parse(candidate_launch)
    launch_node = next(
        node
        for node in candidate_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_fixed32_gdn_gqa_group3_source_candidate"
    )
    launch_source = ast.get_source_segment(candidate_launch, launch_node)
    assert launch_source is not None

    assert "_fr13_fixed32_gdn_gqa_group3_production_for_batch(1)" in b1
    assert "_FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH(" in b1
    assert 'return (\n            "gqa_group3"' in selector
    assert "_fr13_fixed32_gdn_gqa_group3_production_for_batch(4)" in selector
    assert 'selector == "gqa_group3"' in b4
    assert 'elif selector == "gqa_group3":\n        _launch_batched' in b4
    assert "maxnreg=(128 if _gate_export else None)" in b1
    assert "maxnreg=(128 if gate_export else None)" in b4
    assert "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None" in preseed
    assert "torch.empty" not in launch_source
    assert "torch.zeros" not in launch_source
    assert ".item()" not in candidate_launch
    assert "cuda.synchronize" not in candidate_launch


def test_b4_selector_uses_only_the_exact_production_credential_batch() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "_fr13_fixed32_gdn_gqa_group3_production_for_batch",
        "fixed32_batch_gdn_selector",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, object] = {
        "os": os,
        "_FR13_FIXED32_MAX_BATCH": 4,
        "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE": None,
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH": False,
        "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": {"batch_size": 4},
        "_fr13_fixed32_batch_gdn_byte_ab_control": lambda: (False, None),
        "_fr13_fixed32_batch_gdn_graph_byte_ab_control": lambda: False,
        "_fr13_fixed32_batch_gdn_production_control": lambda: None,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=definitions, type_ignores=[])
            ),
            KERNEL,
            "exec",
        ),
        namespace,
    )
    select = namespace["fixed32_batch_gdn_selector"]
    assert select(4) == "gqa_group3"
    assert select(2) is None
    assert select(3) is None

    namespace["_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION"] = {"batch_size": 1}
    assert select(4) is None


def test_patcher_launcher_and_manifest_bind_production_credential() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH" in patcher
    assert "GDN GQA-group3 production requires exact credentialed" in patcher
    assert "B1-or-B4 K64/root1 physical32 FULL-graph contract" in patcher
    assert "production did not replace the captured" in patcher
    assert '"incumbent launch: " + repr(executed_gdn)' in patcher
    assert "fr13_fixed32_gdn_gqa_group3.production_credential.json" in launcher
    assert "fr13_fixed32_gdn_gqa_group3.production_batch.flag" in launcher
    assert "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_PASS_PATH=" in launcher
    assert "FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON" in launcher
    assert "git rev-parse --verify 'HEAD^{commit}'" in launcher
    assert launcher.count("fr13_gdn_gqa_group3_production_credential.py") == 2
    assert launcher.count('--source-commit "$_fr13_gdn_gqa_group3_source_commit"') == 2
    assert "scripts/fr13_gdn_gqa_group3_production_credential.py" in manifest
    assert "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py" in manifest
