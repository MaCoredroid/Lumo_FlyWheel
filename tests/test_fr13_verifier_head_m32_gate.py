from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GENERIC_RUNNER = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_verifier_head_m32_live_gate.sh"
BIGDENOM = REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
VALIDATOR = REPO / "scripts" / "fr13_verifier_head_m32_gate.py"
SWE_RUNNER = REPO / "scripts" / "run_swe_bench_q36_a.py"


def _module():
    spec = importlib.util.spec_from_file_location("fr13_verifier_gate", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patcher_module():
    spec = importlib.util.spec_from_file_location("fr13_verifier_patcher", PATCHER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_block() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "verifier_head_block"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("verifier-head runtime block not found")


def _runtime_function(block: str):
    source = (
        "def compute_logits(self, hidden_states, logits):\n"
        + textwrap.indent(textwrap.dedent(block), "    ")
        + "    return logits\n"
    )
    namespace = {"os": os}
    exec(compile(source, "<verifier-head-shadow>", "exec"), namespace)
    return namespace["compute_logits"]


def _live() -> dict[str, object]:
    module = _module()
    return {
        "schema": module.LIVE_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "instance_id": module.EXPECTED_INSTANCE,
        "source_commit": "b" * 40,
        "patch_source_sha256": "c" * 64,
        "candidate_so_sha256": module.EXPECTED_SO_SHA256,
        "kernel_source_sha256": module.EXPECTED_KERNEL_SHA256,
        "task_marker": f"swe_verified:{module.EXPECTED_INSTANCE}",
        "topology": module.EXPECTED_TOPOLOGY,
        "geometry": module.EXPECTED_GEOMETRY,
        "comparison_calls": 1,
        "compared_elements": module.EXPECTED_ELEMENTS,
        "compared_bytes": module.EXPECTED_BYTES,
        "raw_bf16_mismatches": 0,
        "reference_preservation_mismatches": 0,
        "reference_sha256": "d" * 64,
        "candidate_sha256": "d" * 64,
        "reference_always_served": True,
        "candidate_returned": False,
        "served_return": "incumbent BF16 logits object unchanged",
        "performance_measurement": False,
        "timing_eligible": False,
    }


def test_runtime_shadow_is_exact_pinned_and_incumbent_served() -> None:
    source = PATCHER.read_text(encoding="utf-8")
    assert "# FR13_VERIFIER_HEAD_M32_SHADOW" in source
    assert "and tuple(hidden_states.shape) == (32, 5120)" in source
    assert "and tuple(logits.shape) == (32, 248320)" in source
    assert "FR13 verifier-head real-task marker drifted" in source
    assert "FR13 verifier-head shadow requires eager Hydra27" in source
    assert "torch.ops.fr13_verifier_head.bf16_m32_out(" in source
    assert '"compared_elements": 7946240' in source
    assert '"reference_always_served": True' in source
    assert '"candidate_returned": False' in source
    assert "_fr13_vh_reference_words = memoryview(" in source
    assert "_fr13_vh_temporary.chmod(0o644)" in source
    assert "return logits" in source
    assert "return _fr13_vh_candidate" not in source
    runtime = _runtime_block()
    assert "spawned EngineCore receives a curated environment" in runtime
    assert 'os.environ.get("FR13_VERIFIER_HEAD' not in runtime
    assert 'os.environ.get("FR13_FIXED32' not in runtime


def test_runtime_waits_through_prefill_until_exact_m32() -> None:
    source = PATCHER.read_text(encoding="utf-8")
    exact = source.index("_fr13_vh_exact_m32 = (")
    attempted = source.index("self._fr13_vh_shadow_attempted = True", exact)
    guarded = source.index(
        "if _fr13_vh_marker_path.exists() and _fr13_vh_exact_m32:", exact
    )
    assert exact < guarded < attempted


def test_patch_time_bake_is_default_off_and_rejects_disabled_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "FR13_VERIFIER_HEAD_M32_SHADOW",
        "FR13_VERIFIER_HEAD_M32_SO",
        "FR13_VERIFIER_HEAD_M32_SO_SHA256",
        "FR13_VERIFIER_HEAD_M32_KERNEL_SOURCE",
        "FR13_VERIFIER_HEAD_M32_KERNEL_SOURCE_SHA256",
        "FR13_VERIFIER_HEAD_M32_PATCH_SOURCE_SHA256",
        "FR13_VERIFIER_HEAD_M32_SOURCE_COMMIT",
        "FR13_VERIFIER_HEAD_M32_INSTANCE_ID",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    patcher = _patcher_module()
    config = patcher._fr13_verifier_head_m32_patch_config()
    assert config["enabled"] is False

    rendered = patcher._fr13_verifier_head_m32_render_block(
        _runtime_block(), config, "a" * 64
    )
    assert "no executable hook emitted" in rendered
    compute_logits = _runtime_function(rendered)
    assert "getattr" not in compute_logits.__code__.co_names
    incumbent = object()
    assert compute_logits(SimpleNamespace(), object(), incumbent) is incumbent

    monkeypatch.setenv("FR13_VERIFIER_HEAD_M32_SO_SHA256", "a" * 64)
    with pytest.raises(RuntimeError, match="disabled with live credentials"):
        patcher._fr13_verifier_head_m32_patch_config()


def test_launcher_is_default_off_and_mounts_only_exact_candidate() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "FR13_VERIFIER_HEAD_M32_SHADOW=${FR13_VERIFIER_HEAD_M32_SHADOW:-0}" in source
    assert "disabled verifier-head shadow forbids candidate credentials" in source
    assert "5b5e8c3051f29bc4f65ef93c96ed22ef38ef07a1754e9c36a167e5158f71f4b7" in source
    assert "7cbc9f5157d8e93ee35930b028d97d0c3b1a26a9d79aa87ec6061928f8161768" in source
    assert (
        "verifier-head shadow requires eager Hydra27 physical32 K64/root1 real-B1"
        in source
    )
    assert '== "/logs/fr13_verifier_head_m32.live.json"' in source
    assert '== "/logs/fr13_fixed32_cutlass_streamk.real_event.arm"' in source
    assert (
        '-v "$FR13_VERIFIER_HEAD_M32_SO:/tmp/fr13_verifier_head_m32.abi3.so:ro"'
        in source
    )
    assert (
        '-e FR13_VERIFIER_HEAD_M32_SO="$_FR13_VERIFIER_HEAD_M32_CONTAINER_SO"' in source
    )


def test_real_runner_binds_k64_root1_physical32_without_timing() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "One real SWE-Verified K64/root1 B1 raw-BF16" in source
    assert "export FR13_B1_WORKLOAD_PROFILE=k64_root" in source
    assert "export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907" in source
    assert "export FR13_GATE_VERIFIER_HEAD_M32=1" in source
    assert "export FR13_GATE_GDN_BV=0" in source
    assert "export ENFORCE_EAGER=1" in source
    assert "exact4" not in source.lower()
    assert "timing" not in source.lower()


def test_generic_runner_requires_clean_terminal_evidence() -> None:
    source = GENERIC_RUNNER.read_text(encoding="utf-8")
    assert (
        "FR13_GATE_VERIFIER_HEAD_M32 must be the only enabled kernel candidate"
        in source
    )
    assert (
        "FR13_VERIFIER_HEAD_M32_REAL_EVENT_PATH=/logs/fr13_fixed32_cutlass_streamk.real_event.arm"
        in source
    )
    assert "fixed32_final_flush.json" in source
    assert "fixed32_chat_traffic_audit.json" in source
    assert "scripts/fr13_verifier_head_m32_gate.py" in source
    assert "timing_eligible=0" in source
    assert "reference_always_served=1" in source


def test_shared_runner_keeps_verifier_credentials_empty_when_disabled() -> None:
    source = GENERIC_RUNNER.read_text(encoding="utf-8")
    assert "FR13_GATE_VERIFIER_HEAD_M32_KERNEL_SHA256=\n" in source
    assert "VERIFIER_HEAD_M32_PATCH_SHA256=\n" in source
    assert "VERIFIER_HEAD_M32_SOURCE_COMMIT=\n" in source
    assert "VERIFIER_HEAD_M32_INSTANCE_ID=\n" in source
    assert 'if [[ "$FR13_GATE_VERIFIER_HEAD_M32" == "1" ]]; then' in source
    assert (
        'FR13_VERIFIER_HEAD_M32_PATCH_SOURCE_SHA256="$VERIFIER_HEAD_M32_PATCH_SHA256"'
        in source
    )
    assert (
        'FR13_VERIFIER_HEAD_M32_SOURCE_COMMIT="$VERIFIER_HEAD_M32_SOURCE_COMMIT"'
        in source
    )
    assert (
        'FR13_VERIFIER_HEAD_M32_INSTANCE_ID="$VERIFIER_HEAD_M32_INSTANCE_ID"' in source
    )


def test_bigdenom_reuses_authenticated_single_task_arm() -> None:
    source = BIGDENOM.read_text(encoding="utf-8")
    assert '${FR13_VERIFIER_HEAD_M32_SHADOW:-0}" == "1"' in source
    assert (
        '--fixed32-cutlass-real-event-arm "$FIXED32_CUTLASS_REAL_EVENT_ARM_PATH"'
        in source
    )


def test_swe_runner_admits_shadow_on_stock_wave_only() -> None:
    source = SWE_RUNNER.read_text(encoding="utf-8")
    assert '"FR13_VERIFIER_HEAD_M32_SHADOW"' in source
    assert 'fixed32_verifier_head_shadow = verifier_head_shadow_text == "1"' in source
    assert 'if cutlass_wave != "stock":' in source
    assert "or fixed32_verifier_head_shadow" in source
    assert "if fixed32_cutlass_diagnostic or fixed32_verifier_head_shadow:" in source
    assert "verifier-head M32 shadow must be the only kernel diagnostic" in source


def test_live_validator_accepts_only_exact_zero_diff_record() -> None:
    module = _module()
    payload = _live()
    assert (
        module.validate_live_result(
            payload,
            expected_source_commit="b" * 40,
            expected_patch_sha256="c" * 64,
        )
        == payload
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("raw_bf16_mismatches", 1),
        ("reference_preservation_mismatches", 1),
        ("comparison_calls", 2),
        ("compared_bytes", 1),
        ("candidate_returned", True),
        ("timing_eligible", True),
        ("status", "FAIL"),
    ],
)
def test_live_validator_rejects_false_pass(key: str, value: object) -> None:
    module = _module()
    payload = _live()
    payload[key] = value
    with pytest.raises(ValueError, match="live PASS contract drifted"):
        module.validate_live_result(
            payload,
            expected_source_commit="b" * 40,
            expected_patch_sha256="c" * 64,
        )


def test_gate_output_is_private_and_explicitly_unqualified(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "gate.json"
    payload = {
        "schema": module.GATE_SCHEMA,
        "status": "PASS",
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
    }
    module._write(path, payload)
    assert json.loads(path.read_text(encoding="ascii")) == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_runtime_manifest_closes_over_gate_sources() -> None:
    source = MANIFEST.read_text(encoding="utf-8")
    for relative in (
        "scripts/fr13_verifier_head_m32_gate.py",
        "scripts/fr13_run_b1_verifier_head_m32_live_gate.sh",
        "csrc/fr13_bf16_verifier_head_m32_sm121a.cu",
    ):
        assert f'"{relative}"' in source
