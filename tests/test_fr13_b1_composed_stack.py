from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _load_timing_module():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_b1_composed_stack_timing.py"
    spec = importlib.util.spec_from_file_location("fr13_b1_composed_timing_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gate_module():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_b1_composed_stack_gate.py"
    spec = importlib.util.spec_from_file_location("fr13_b1_composed_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sfwd_gate_module():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_sfwd_conv_postprep_gate.py"
    spec = importlib.util.spec_from_file_location("fr13_sfwd_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, raw: bytes) -> tuple[Path, str]:
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_composed_gate_helper_runs_directly() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/fr13_b1_composed_stack_gate.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validate-graph-credentials" in completed.stdout
    assert "validate-eager-credentials" in completed.stdout


def test_gate_wrappers_bind_independent_same_boot_credentials() -> None:
    graph = _text("fr13_run_gdn_single_launch_live_gate.sh")
    helper = _text("fr13_b1_composed_stack_gate.py")
    eager = _text("fr13_run_b1_target_sfwd_conv_postprep_live_gate.sh")
    assert "FR13_FA2_QROW32_B1_LIVE_AB_ARM=\"$QROW32_LIVE_ARM\"" in graph
    assert "QROW32_LIVE_ARM=$QROW32_GATE_ARM" in graph
    assert '--qrow-arm "$QROW32_LIVE_ARM"' in graph
    assert "FR13_FIXED32_GDN_PATH_BV_CANDIDATE=\"$FR13_GDN_GATE_CANDIDATE\"" in graph
    assert "FR13_DFWD_K64_TOP3=\"$COMBINED_GRAPH_GATE\"" in graph
    assert '--qrow-output "$ARMDIR/$QROW32_VERIFICATION_FILE"' in graph
    assert 'QROW32_VERIFICATION_FILE="qrow32_${QROW32_LIVE_ARM}_live_verification.json"' in graph
    assert "--dfwd-output \"$ARMDIR/dfwd_k64_top3_credential.json\"" in graph
    assert 'qrow.get("live_result_sha256") != _sha256(qrow_live_raw)' in helper
    assert "Gate-A shared evidence" in helper
    assert 'export RUNROOT="$RUNROOT_REL"' in eager
    assert "--target-live-pass" in eager
    assert "sfwd_conv_postprep_k64_root_b1_gate.json" in eager


def test_composed_gate_reads_attempt_parity_from_v3_ingress(tmp_path: Path) -> None:
    module = _load_gate_module()
    arm = tmp_path / "arm"
    arm.mkdir()
    health = {
        "swe_orchestrator_rc": 0,
        "tasks": [
            {
                "instance_id": module.TASK_ID,
                "codex_timed_out": False,
                "verdict": "resolved",
            }
        ],
    }
    traffic = {
        "schema": "fr13-fixed32-chat-task-provenance-audit-v3",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "mode": "hydra27_fixed32",
        "subset": {
            "sha256": module.SUBSET_SHA256,
            "task_ids": [module.TASK_ID],
            "task_count": 1,
        },
        "checks": {"all_canonical_tasks_validated": True},
        "ingress": {"exact_proxy_engine_attempt_parity": True},
        "tasks": {module.TASK_ID: {}},
    }
    (arm / "health.json").write_text(json.dumps(health), encoding="ascii")
    traffic_path = arm / "fixed32_chat_traffic_audit.json"
    traffic_path.write_text(json.dumps(traffic), encoding="ascii")

    module._validate_health_and_traffic(arm)

    traffic["ingress"]["exact_proxy_engine_attempt_parity"] = False
    traffic_path.write_text(json.dumps(traffic), encoding="ascii")
    with pytest.raises(module.GateError, match="traffic audit drifted"):
        module._validate_health_and_traffic(arm)


def test_runner_wires_optional_cfwd_smoke_and_exact4_evidence() -> None:
    runner = _text("fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh")
    wrapper = _text("fr13_run_b1_composed_stack_timing.sh")
    smoke_wrapper = _text("fr13_run_b1_composed_cfwd_production_smoke.sh")
    cfwd_wrapper = _text("fr13_run_b1_composed_cfwd_stack_timing.sh")
    manifest = _text("fr13_runtime_manifest.py")
    required = (
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1",
        "FR13_DFWD_K64_TOP3=1",
        'FR13_FIXED32_CUTLASS_WAVE="$TARGET_SELECTOR"',
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1",
        'FR13_DFWD_GPU_TIMER="$COMPOSED_STACK"',
        'FR13_CFWD_GPU_TIMER="$COMPOSED_STACK"',
        "validate-graph-credentials",
        "validate-eager-credentials",
        "fr13_b1_composed_stack_timing.py",
        "TARGET_SFWD_COMBINED_SUMMARY_SHA256",
        "production engaged layer=",
        "issue-production-smoke",
        "validate-production-smoke",
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=1",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1",
        "--cfwd-engagement",
        "--cfwd-smoke-credential",
    )
    for value in required:
        assert value in runner
    assert "FR13_B1_COMPOSED_STACK_TIMING=1" in wrapper
    assert "FR13_B1_COMPOSED_CFWD_SMOKE=1" in smoke_wrapper
    assert "FR13_B1_COMPOSED_CFWD_PRODUCTION=1" in smoke_wrapper
    assert "FR13_RUN_B1_U8_CFWD_SFWD_TIMING=1" in cfwd_wrapper
    assert "fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh" in cfwd_wrapper
    assert "FR13_RUN_QROW32_NOSPLIT_TIMING" not in cfwd_wrapper
    for source in (
        "scripts/fr13_run_b1_k64_qrow32_b1_sfwd_stack_timing.sh",
        "scripts/fr13_run_b1_composed_stack_timing.sh",
        "scripts/fr13_run_b1_composed_cfwd_production_smoke.sh",
        "scripts/fr13_run_b1_composed_cfwd_stack_timing.sh",
        "scripts/fr13_run_b1_u8_cfwd_sfwd_stack_timing.sh",
        "scripts/fr13_b1_composed_stack_timing.py",
        "scripts/fr13_cfwd_logit_direct_gate.py",
        "scripts/fr13_qrow32_b1_pass_sidecar.py",
        "scripts/fr13_qrow32_split2_timing.py",
    ):
        assert source in manifest


def test_sfwd_admits_taw_only_for_exact_composed_cfwd_tuple() -> None:
    launcher = _text("fr13_launch_forked_fa2_tree_server.sh")
    required = (
        '_fr13_sfwd_cfwd_composed=0',
        '"$_fr13_sfwd_qrow16_production" == "0"',
        '"$_fr13_sfwd_qrow32_production" == "1"',
        '"${FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION:-0}" == "1"',
        '"${FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH:-}" == "1"',
        '"${FR13_DFWD_K64_TOP3:-0}" == "1"',
        '"${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "identity_wide256_fullgrid_b1"',
        '"${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}" == "0"',
        '"${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION:-0}" == "1"',
        '"${FR13_CFWD_LOGIT_DIRECT_BYTE_AB:-0}" == "0"',
        '"${FR13_CFWD_LOGIT_DIRECT_PRODUCTION:-0}" == "1"',
        '&& "$_fr13_sfwd_cfwd_composed" != "1"',
    )
    for value in required:
        assert value in launcher


def test_six_way_environment_rejects_every_single_selector_near_miss() -> None:
    module = _load_gate_module()
    exact = list(module.SIX_WAY_ENV)
    module.validate_six_way_environment(("\n".join(exact) + "\n").encode("ascii"))
    for index, line in enumerate(exact):
        name, _separator, _value = line.partition("=")
        near_miss = list(exact)
        near_miss[index] = f"{name}=near-miss"
        with pytest.raises(module.GateError, match="environment drifted"):
            module.validate_six_way_environment(
                ("\n".join(near_miss) + "\n").encode("ascii")
            )


def test_cfwd_engagement_requires_candidate_served_return() -> None:
    module = _load_gate_module()
    source_commit = "1" * 40
    credential_sha = "2" * 64
    payload = {
        "schema": module.CFWD_ENGAGEMENT_SCHEMA,
        "status": "engaged",
        "candidate": module.cfwd_gate.CANDIDATE,
        "mode": "hydra27_fixed32",
        "batch_size": 1,
        "source_commit": source_commit,
        "candidate_source_sha256": module.cfwd_gate.CANDIDATE_SOURCE_SHA256,
        "integration_source_schema": module.cfwd_gate.INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": module.cfwd_gate.INTEGRATION_SOURCE_SHA256,
        "production_pass_sha256": credential_sha,
        "served_return": module.CFWD_SERVED_RETURN,
        "producer_pid": 123,
    }
    module._validate_cfwd_engagement(
        payload,
        source_commit=source_commit,
        credential_sha256=credential_sha,
    )
    payload["served_return"] = "reference products"
    with pytest.raises(module.GateError, match="served-return"):
        module._validate_cfwd_engagement(
            payload,
            source_commit=source_commit,
            credential_sha256=credential_sha,
        )


def test_production_smoke_credential_binds_all_component_hashes() -> None:
    module = _load_gate_module()
    source_commit = "1" * 40
    component_hashes = {
        key: hashlib.sha256(key.encode("ascii")).hexdigest()
        for key, _path_name, _sha_name in module.COMPONENT_ARGUMENTS
    }
    payload = {
        "schema": module.PRODUCTION_SMOKE_SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "mode": "hydra27_fixed32",
        "batch_size": 1,
        "task_id": module.TASK_ID,
        "task_count": 1,
        "subset_sha256": module.SUBSET_SHA256,
        "runtime_mode": "FULL",
        "production_paths": {
            "qrow32_nosplit": True,
            "gdn_gqa_group3": True,
            "dfwd_k64_top3": True,
            "target_gemm_selector": module.TARGET_SELECTOR,
            "sfwd_conv_postprep": True,
            "taw_native_precompute": True,
            "cfwd_logit_direct": True,
        },
        "component_credential_sha256s": component_hashes,
        "runtime_evidence_sha256s": {"health": "2" * 64},
        "cfwd_served_return": module.CFWD_SERVED_RETURN,
        "performance_measurement": False,
        "timing_eligible": False,
        "exact4_eligible": True,
    }
    module._validate_production_smoke_payload(
        payload,
        source_commit=source_commit,
        component_hashes=component_hashes,
    )
    mutated = dict(component_hashes, cfwd_credential="3" * 64)
    with pytest.raises(module.GateError, match="credential drifted"):
        module._validate_production_smoke_payload(
            payload,
            source_commit=source_commit,
            component_hashes=mutated,
        )


def test_gate_b_summary_binds_target_sfwd_manifest_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_gate_module()
    monkeypatch.setattr(module, "_validate_source_commit", lambda *_: None)
    source_commit = "1" * 40
    target, target_sha = _write(tmp_path / "target", b"target\n")
    sfwd, sfwd_sha = _write(tmp_path / "sfwd", b"sfwd\n")
    manifest, manifest_sha = _write(tmp_path / "manifest", b"manifest\n")
    sha_fields = {
        key: hashlib.sha256(key.encode()).hexdigest()
        for key in (
            "records_sha256",
            "runtime_manifest_sha256",
            "external_manifest_sha256",
            "host_readiness_sha256",
            "diagnostic_sha256",
            "task_bracket_sha256",
            "terminal_sha256",
            "traffic_sha256",
            "container_env_sha256",
            "docker_log_sha256",
            "qrow16_sidecar_sha256",
            "qrow16_capture_sha256",
        )
    }
    summary_payload = {
        "schema": module.SFWD_COMBINED_SCHEMA,
        "status": "pass",
        "candidate": module.SFWD_CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": manifest_sha,
        "task_id": module.TASK_ID,
        "task_count": 1,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "qrow16_production": True,
        "layer_count": 48,
        "reference_returned": True,
        "candidate_returned": False,
        "decision_exact": True,
        "no_fallback": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "live_pass_sha256": sfwd_sha,
        "combined_target_selector": "identity_wide256_fullgrid_b1",
        "combined_target_live_pass_sha256": target_sha,
        **sha_fields,
    }
    summary, summary_sha = _write(
        tmp_path / "summary",
        (json.dumps(summary_payload, sort_keys=True) + "\n").encode(),
    )
    args = SimpleNamespace(
        repo=ROOT,
        source_commit=source_commit,
        combined_summary=summary,
        combined_summary_sha256=summary_sha,
        target_live=target,
        target_live_sha256=target_sha,
        sfwd_pass=sfwd,
        sfwd_pass_sha256=sfwd_sha,
        source_manifest=manifest,
        source_manifest_sha256=manifest_sha,
    )
    module.validate_eager_credentials(args)
    summary_payload["combined_target_live_pass_sha256"] = "0" * 64
    summary.write_text(json.dumps(summary_payload, sort_keys=True) + "\n")
    args.combined_summary_sha256 = hashlib.sha256(summary.read_bytes()).hexdigest()
    with pytest.raises(module.GateError, match="combined_target_live_pass_sha256"):
        module.validate_eager_credentials(args)


def test_combined_sfwd_gate_validates_target_pass_against_repo_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_sfwd_gate_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    arm = tmp_path / "arm"
    logs = arm / "logs"
    logs.mkdir(parents=True)
    bracket_path = (
        arm
        / "swe_out"
        / "only"
        / "per_task"
        / module.TASK_ID
        / "fixed32_task_boundary.json"
    )
    bracket_path.parent.mkdir(parents=True)
    bracket_path.write_text("{}\n", encoding="ascii")
    installed = logs / "fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
    installed.write_text("manifest\n", encoding="ascii")
    installed.chmod(0o400)
    marker = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
    marker.write_text(module.TASK_MARKER + "\n", encoding="ascii")
    marker.chmod(0o444)

    source_commit = "1" * 40
    launch_raw = b"manifest\n"
    source_sha = hashlib.sha256(launch_raw).hexdigest()
    files = {
        module.MODULE_SOURCE: {"sha256": "2" * 64},
        module.KERNEL_SOURCE: {"sha256": "3" * 64},
    }
    launch = {"source_commit": source_commit}
    readiness = {
        "schema": module.READINESS_SCHEMA,
        "status": "ready_for_one_real_swe_verified_hydra27_b1_byte_gate",
        "candidate": module.CANDIDATE,
        "source_commit": source_commit,
        "upstream_commit": source_commit,
        "source_manifest_sha256": source_sha,
        "source_file_count": len(module.SOURCE_FILES),
        "candidate_source_sha256": files[module.MODULE_SOURCE]["sha256"],
        "candidate_kernel_source_sha256": files[module.KERNEL_SOURCE]["sha256"],
        "task_id": module.TASK_ID,
        "task_subset_sha256": module.SUBSET_SHA256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": module.BLOCKS_SHA256,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "qrow16_production": True,
        "qrow16_fa2_sha256": module.QROW16_SHA256,
        "qrow16_live_pass_sha256": module.QROW16_PASS_SHA256,
        "compared_byte_surfaces": list(module.BYTE_SURFACES),
        "required_layer_count": module.LAYERS,
        "reference_always_served": True,
        "candidate_returned": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "gpu_or_docker_used": False,
        "launched": False,
    }
    record_pairs = {(str(layer), f"prefix-{layer}") for layer in range(48)}
    record_template = {
        "schema": "fr13.fixed32.sfwd_conv_postprep.byte_ab.v1",
        "status": "pass",
        "candidate": module.CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": source_sha,
        "fixed32_mode": "hydra27_fixed32",
        "task_marker": module.TASK_MARKER,
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "compared_byte_surfaces": list(module.BYTE_SURFACES),
        "mismatches": 0,
        "differing_bytes": 0,
        "zero_diff": True,
        "real_task_authenticated": True,
        "reference_always_served": True,
        "candidate_returned": False,
        "reference_decision": "serve_incumbent",
        "candidate_decision": "shadow_only",
        "decision_exact": True,
        "qrow16_production": True,
        "qrow16_fa2_sha256": module.QROW16_SHA256,
        "qrow16_live_pass_sha256": module.QROW16_PASS_SHA256,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "comparisons": [
            {"name": name, "byte_equal": True, "differing_bytes": 0}
            for name in module.BYTE_SURFACES
        ],
    }
    records = []
    for layer, prefix in sorted(record_pairs):
        record = dict(record_template, layer_key=layer, layer_prefix_sha256=prefix)
        records.append(json.dumps(record, sort_keys=True))
    records_raw = ("\n".join(records) + "\n").encode("ascii")
    target_path = tmp_path / "target-pass.json"
    target_raw = b"target-pass\n"
    target_path.write_bytes(target_raw)
    target_so = tmp_path / "target.so"
    target_so.write_bytes(b"so\n")

    json_by_name = {
        "launch.json": (launch, launch_raw),
        "end.json": (launch, launch_raw),
        "sfwd_conv_postprep_host_readiness.json": (readiness, b"readiness\n"),
        "fr13_fixed32_sfwd_conv_postprep.live_pass.json": ({}, b"pass\n"),
        "fixed32_b1_diagnostic.json": (
            {"task_ids": [module.TASK_ID], "floor_acceptance_eligible": False},
            b"diagnostic\n",
        ),
        "fixed32_final_flush_skipped.json": (
            {
                "schema": "fr13-fixed32-eager-kernel-terminal-v1",
                "run_classification": "eager_kernel_byte_diagnostic",
                "acceptance_valid": False,
                "flush_protocol_used": False,
            },
            b"terminal\n",
        ),
        "fixed32_chat_traffic_audit_skipped.json": (
            {
                "run_classification": "eager_kernel_byte_diagnostic",
                "acceptance_valid": False,
            },
            b"traffic\n",
        ),
        "fixed32_task_boundary.json": (
            {
                "schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
                "instance_id": module.TASK_ID,
                "acceptance_valid": False,
            },
            b"bracket\n",
        ),
    }
    monkeypatch.setattr(module, "_load_json", lambda path: json_by_name[path.name])
    monkeypatch.setattr(module, "_validate_source_manifest", lambda *_args, **_kwargs: files)
    monkeypatch.setattr(module, "_validate_live_pass", lambda *_args, **_kwargs: record_pairs)
    monkeypatch.setattr(module, "_validate_qrow_evidence", lambda *_args: (b"sidecar\n", b"capture\n"))

    expected_env = [
        "FR13_FIXED32_CUTLASS_WAVE=identity_wide256_fullgrid_b1_byte_ab",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1",
        f"FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256={source_sha}",
        f"FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT={source_commit}",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0",
        "FR13_FA2_QROW16_PRODUCTION=1",
        f"FR13_FA2_QROW16_SO_SHA256={module.QROW16_SHA256}",
        f"FR13_FA2_QROW16_LIVE_PASS_SHA256={module.QROW16_PASS_SHA256}",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "ENFORCE_EAGER=1",
    ]
    regular_by_name = {
        "runtime_manifest.at_launch.json": b"runtime\n",
        "runtime_manifest.at_end.json": b"runtime\n",
        "external_manifest.at_launch.json": b"external\n",
        "external_manifest.at_end.json": b"external\n",
        installed.name: launch_raw,
        marker.name: (module.TASK_MARKER + "\n").encode("ascii"),
        "fr13_fixed32_sfwd_conv_postprep.byte_ab.jsonl": records_raw,
        "container_env.txt": ("\n".join(expected_env) + "\n").encode("ascii"),
        "docker_after_tasks.log": (
            b"[FR13_DRAFT_VOCAB] shim built K=65536 x\n"
            b"[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 x\n"
        ),
        target_path.name: target_raw,
    }
    monkeypatch.setattr(module, "_regular", lambda path, **_kwargs: regular_by_name[path.name])

    validated: dict[str, object] = {}

    class QualificationError(RuntimeError):
        pass

    def validate_live_result(path, digest, candidate_so, patch_source, **kwargs):
        validated.update(
            path=path,
            digest=digest,
            candidate_so=candidate_so,
            patch_source=patch_source,
            kwargs=kwargs,
        )

    fake_cutlass = types.SimpleNamespace(
        QualificationError=QualificationError,
        validate_live_result=validate_live_result,
    )
    monkeypatch.setitem(sys.modules, "fr13_cutlass_streamk_pass", fake_cutlass)
    output = tmp_path / "combined.json"
    module.validate_gate(
        SimpleNamespace(
            repo=repo,
            arm_dir=arm,
            source_commit=source_commit,
            task_id=module.TASK_ID,
            manifest_launch=tmp_path / "launch.json",
            manifest_end=tmp_path / "end.json",
            target_live_pass=target_path,
            target_candidate_so=target_so,
            output=output,
        )
    )
    assert validated["patch_source"] == repo / "scripts/fr13_patch_cutlass_fixed32_wave.py"
    assert json.loads(output.read_text(encoding="ascii"))[
        "combined_target_live_pass_sha256"
    ] == hashlib.sha256(target_raw).hexdigest()


def test_composed_reducer_emits_phase_tps_u95_and_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_timing_module()
    base = {
        "schema": "fr13.fixed32.fa2_qrow32_nosplit.exact4_timing.v1",
        "run_classification": "real_swe_verified_exact4_qrow32_nosplit",
        "descriptive_equal_task_one_sided_u95": {
            "mean_ms": 130.0,
            "u95_ms": 135.0,
            "descriptive_screen_pass": True,
        },
        "exact16_eligible": True,
    }
    monkeypatch.setattr(module.qrow_timing, "reduce_timing", lambda **_: base)
    measure, _ = _write(
        tmp_path / "measure.json",
        (
            json.dumps(
                {
                    "step_wall_ms": 130.0,
                    "measured_tps_fullstep_wall": 40.0,
                    "s_per_fwd_gpu": 0.08,
                    "drafter_gpu_ms_per_step": 20.0,
                    "committer_gpu_ms_per_step": 10.0,
                    "overhead_other_ms_per_event": 20.0,
                    "accept_per_event": 4.0,
                    "committed_per_event": 5.0,
                    "floor_ratio": 130.0 / 119.658015414,
                    "derived_tps_fullstep_gpu": 45.0,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    required_env = (
        "FR13_FIXED32_MODE=hydra27_fixed32",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "MAX_NUM_SEQS=1",
        "SWE_CONCURRENCY=1",
        "ENFORCE_EAGER=0",
        "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
        "FR10_METRICS=1",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=1",
        "FR13_DFWD_K64_TOP3=1",
        "FR13_FIXED32_CUTLASS_WAVE=identity_wide256_fullgrid_b1",
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=1",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1",
        "FR13_CONV_WB_BATCHED=1",
        "FR13_SFWD_GPU_TIMER=1",
        "FR13_DFWD_GPU_TIMER=1",
        "FR13_CFWD_GPU_TIMER=1",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0",
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1",
        "FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0",
        "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=1",
    )
    container, _ = _write(
        tmp_path / "container_env.txt", ("\n".join(required_env) + "\n").encode()
    )
    docker_lines = list(module.DFWD_MARKERS) + [
        f"{module.SFWD_MARKER}layers.{index} B=1 rows=32"
        for index in range(48)
    ]
    docker, _ = _write(
        tmp_path / "docker.log", ("\n".join(docker_lines) + "\n").encode()
    )
    target_binary, _ = _write(
        tmp_path / "target_binary.json",
        (
            json.dumps(
                {
                    "schema": "fr13.fixed32.cutlass_streamk_binary.v2",
                    "selector": module.TARGET_SELECTOR,
                    "production_enabled": True,
                    "qualification_profile": "k64_root",
                    "source": {"sha256": module.TARGET_SHA256},
                    "destination": {"sha256": module.TARGET_SHA256},
                    "installed_mode": "0555",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    named: dict[str, tuple[Path, str]] = {}
    for name in (
        "gqa",
        "target_sidecar",
        "sfwd_pass",
        "sfwd_manifest",
        "qrow_credential",
        "dfwd_credential",
        "eager_summary",
    ):
        named[name] = _write(tmp_path / name, f"{name}\n".encode())
    gqa_arm, _ = _write(tmp_path / "gqa_arm", b"1\n")
    gqa_batch, _ = _write(tmp_path / "gqa_batch", b"1\n")
    taw, taw_sha = _write(tmp_path / "taw", b"taw\n")
    cfwd, cfwd_sha = _write(tmp_path / "cfwd.json", b"{}\n")
    cfwd_engagement, _ = _write(tmp_path / "cfwd-engagement.json", b"{}\n")
    component_hashes = {
        "qrow32_composed": named["qrow_credential"][1],
        "gqa3": named["gqa"][1],
        "dfwd_top3": named["dfwd_credential"][1],
        "sfwd_pass": named["sfwd_pass"][1],
        "sfwd_manifest": named["sfwd_manifest"][1],
        "target_sfwd_summary": named["eager_summary"][1],
        "taw_production": taw_sha,
        "cfwd_credential": cfwd_sha,
    }
    smoke, smoke_sha = _write(
        tmp_path / "smoke.json",
        (
            json.dumps(
                {"component_credential_sha256s": component_hashes}, sort_keys=True
            )
            + "\n"
        ).encode("ascii"),
    )
    monkeypatch.setattr(
        module.composed_gate, "_validate_production_smoke_payload", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        module.composed_gate.cfwd_gate,
        "_validate_credential",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module.composed_gate, "_validate_cfwd_engagement", lambda *_args, **_kwargs: None
    )
    args = SimpleNamespace(
        cfwd_production=True,
        subset=tmp_path / "unused-subset",
        measure=measure,
        baseline=tmp_path / "unused-baseline",
        engagement=tmp_path / "unused-engagement",
        health=tmp_path / "unused-health",
        traffic_audit=tmp_path / "unused-traffic",
        source_commit="1" * 40,
        patch_source_sha256="2" * 64,
        pass_sha256="3" * 64,
        pass_sidecar_sha256="4" * 64,
        runner_sha256="5" * 64,
        block_map_sha256="6" * 64,
        floor_ms=119.658015414,
        cap_ms=137.6067177261,
        arm="composed",
        container_env=container,
        docker_log=docker,
        gqa3_production_credential=named["gqa"][0],
        gqa3_production_arm=gqa_arm,
        gqa3_production_batch=gqa_batch,
        gqa3_pass_sha256=named["gqa"][1],
        target_production_sidecar=named["target_sidecar"][0],
        target_production_sidecar_sha256=named["target_sidecar"][1],
        target_binary_record=target_binary,
        sfwd_production_pass=named["sfwd_pass"][0],
        sfwd_pass_sha256=named["sfwd_pass"][1],
        sfwd_production_manifest=named["sfwd_manifest"][0],
        sfwd_manifest_sha256=named["sfwd_manifest"][1],
        qrow_composed_credential=named["qrow_credential"][0],
        qrow_composed_credential_sha256=named["qrow_credential"][1],
        dfwd_credential=named["dfwd_credential"][0],
        dfwd_credential_sha256=named["dfwd_credential"][1],
        target_sfwd_combined_summary=named["eager_summary"][0],
        target_sfwd_combined_summary_sha256=named["eager_summary"][1],
        taw_production_pass=taw,
        taw_production_pass_sha256=taw_sha,
        cfwd_production_pass=cfwd,
        cfwd_production_pass_sha256=cfwd_sha,
        cfwd_engagement=cfwd_engagement,
        cfwd_smoke_credential=smoke,
        cfwd_smoke_credential_sha256=smoke_sha,
    )
    result = module.reduce_composed(args)
    assert result["schema"] == module.SCHEMA
    assert result["run_classification"] == (
        "real_swe_verified_exact4_b1_composed_cfwd_kernel_stack"
    )
    assert result["phase_breakdown_ms_per_event"] == {
        "sfwd_verify_gpu": 80.0,
        "dfwd_drafter_gpu": 20.0,
        "cfwd_committer_gpu": 10.0,
        "host_and_unattributed": 20.0,
        "wall_full_step": 130.0,
    }
    assert result["full_step_tps"] == {"wall": 40.0, "gpu_components": 45.0}
    assert result["acceptance"]["descriptive_screen_pass"] is True
    assert result["production_evidence"]["sfwd_engaged_layer_count"] == 48
    assert result["composed_stack"]["cfwd_logit_direct"] is True
    assert result["production_evidence"]["cfwd_served_return"] == (
        "logit-direct candidate products"
    )


def test_forked_launcher_forwards_patch_contract_into_container() -> None:
    launcher = (ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    assert '-e CUDAGRAPH_MODE="$CUDAGRAPH_MODE"' in launcher
    assert '-e MAX_NUM_SEQS="$MAX_NUM_SEQS"' in launcher
    assert '-e SWE_CONCURRENCY="${SWE_CONCURRENCY:-}"' in launcher
