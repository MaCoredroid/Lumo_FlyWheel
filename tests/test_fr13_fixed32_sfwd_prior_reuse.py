from __future__ import annotations

import ast
import hashlib
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "lumo_flywheel_serving" / "fr13_sfwd_prior_reuse.py"
KERNEL_MODULE_PATH = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_prior_reuse_descriptorless.py"
)
OLD_KERNEL_PATH = ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GENERIC_RUNNER_PATH = ROOT / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
RUNNER_PATH = ROOT / "scripts" / "fr13_run_b1_sfwd_prior_reuse_gate.sh"
GATE_PATH = ROOT / "scripts" / "fr13_sfwd_prior_reuse_gate.py"
PRODUCTION_PATH = (
    ROOT / "src" / "lumo_flywheel_serving" / "fr13_sfwd_state_fusion_production.py"
)

sys.path.insert(0, str(ROOT / "src"))
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import fr13_sfwd_prior_reuse as candidate  # noqa: E402


def _function_source(name: str, *, path: Path = MODULE_PATH) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _direct_call_keywords(name: str, *, path: Path) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trees = [tree]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and f"{name}(" in node.value
        ):
            wrapped = "def _fragment():\n" + textwrap.indent(
                textwrap.dedent(node.value), "    "
            )
            trees.append(ast.parse(wrapped))
    return [
        {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        for candidate_tree in trees
        for node in ast.walk(candidate_tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        )
    ]


def _source_manifest(tmp_path: Path, commit: str) -> tuple[Path, str]:
    source_sha = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    kernel_source_sha = hashlib.sha256(KERNEL_MODULE_PATH.read_bytes()).hexdigest()
    payload = {
        "schema": candidate.SOURCE_MANIFEST_SCHEMA,
        "candidate": candidate.CANDIDATE,
        "source_commit": commit,
        "files": {
            candidate.SOURCE_RELATIVE_PATH: {
                "bytes": MODULE_PATH.stat().st_size,
                "sha256": source_sha,
            },
            candidate.KERNEL_SOURCE_RELATIVE_PATH: {
                "bytes": KERNEL_MODULE_PATH.stat().st_size,
                "sha256": kernel_source_sha,
            },
        },
    }
    path = tmp_path / "source_manifest.json"
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    path.write_text(raw, encoding="ascii")
    return path, hashlib.sha256(raw.encode("ascii")).hexdigest()


def test_contract_closes_row32_c64_for_b1_b4() -> None:
    for batch in (1, 2, 3, 4):
        contract = candidate.fixed32_sfwd_prior_reuse_contract(
            batch, tree_rows=32, conv_width=4, conv_state_len=34
        )
        assert contract["candidate"] == candidate.CANDIDATE
        assert contract["physical_rows_per_request"] == 32
        assert contract["conv_rows_per_program"] == 32
        assert contract["conv_row_groups_per_request"] == 1
        assert contract["conv_block_c"] == 64
        assert contract["conv_num_warps"] == 16
        assert contract["topology_host_validation"] == "exact_parent_each_launch"
        assert contract["source_descriptor_device_validation"] is False
        assert contract["source_descriptor_launcher_argument"] is False
    for geometry in ((0, 32, 4, 34), (1, 31, 4, 34), (1, 32, 3, 34)):
        with pytest.raises(ValueError):
            candidate.fixed32_sfwd_prior_reuse_contract(
                geometry[0],
                tree_rows=geometry[1],
                conv_width=geometry[2],
                conv_state_len=geometry[3],
            )


def test_gate_is_default_off_and_uses_authenticated_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enabled = tmp_path / "enabled"
    event = tmp_path / "event"
    monkeypatch.setattr(candidate, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    assert candidate.fixed32_sfwd_prior_reuse_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (False, None)
    enabled.write_text("1\n", encoding="ascii")
    assert candidate.fixed32_sfwd_prior_reuse_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (True, None)
    event.write_text("swe_verified:astropy__astropy-12907\n", encoding="ascii")
    assert candidate.fixed32_sfwd_prior_reuse_gate_control(
        environ={}, enabled_path=str(enabled), event_path=str(event)
    ) == (True, "swe_verified:astropy__astropy-12907")


def test_byte_gate_binds_manifest_and_compares_both_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "1" * 40
    manifest, manifest_sha = _source_manifest(tmp_path, commit)
    records = tmp_path / "records.jsonl"
    live_pass = tmp_path / "live_pass.json"
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH", str(manifest)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256", manifest_sha
    )
    monkeypatch.setenv("FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT", commit)
    monkeypatch.setenv("FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB_PATH", str(records))
    monkeypatch.setenv("FR13_FIXED32_SFWD_PRIOR_REUSE_PASS_PATH", str(live_pass))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    candidate._STATE.update(
        task_marker=None,
        batch=None,
        passed={},
        attempts={},
        source_binding=None,
    )
    reference_out = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    reference_stage = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    record = candidate.fixed32_sfwd_prior_reuse_byte_gate(
        task_marker="swe_verified:astropy__astropy-12907",
        layer_prefix="model.layers.0.mixer",
        layer_key=17,
        batch_size=1,
        reference_out=reference_out,
        candidate_out=reference_out.clone(),
        reference_source_stage=reference_stage,
        candidate_source_stage=reference_stage.clone(),
    )
    assert record["zero_diff"] is True
    assert record["source_manifest_sha256"] == manifest_sha
    assert record["candidate_kernel_source_sha256"] == hashlib.sha256(
        KERNEL_MODULE_PATH.read_bytes()
    ).hexdigest()
    assert [item["name"] for item in record["comparisons"]] == [
        "conv_out",
        "commit_source_stage",
    ]
    assert record["reference_always_served"] is True
    assert record["production_eligible"] is False
    live_pass.write_text("stale\n", encoding="ascii")
    mismatch_stage = reference_stage.clone()
    mismatch_stage[0, 0] = 5.0
    mismatch = candidate.fixed32_sfwd_prior_reuse_byte_gate(
        task_marker="swe_verified:astropy__astropy-12907",
        layer_prefix="model.layers.0.mixer",
        layer_key=17,
        batch_size=1,
        reference_out=reference_out,
        candidate_out=reference_out.clone(),
        reference_source_stage=reference_stage,
        candidate_source_stage=mismatch_stage,
    )
    assert mismatch["zero_diff"] is False
    assert not live_pass.exists()


def test_pass_requires_k64_root1_and_carries_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv(
        "FR13_DRAFT_VOCAB_BLOCKS",
        str(ROOT / "scripts" / "fr13_dvk_subset_blocks.json"),
    )
    monkeypatch.setenv("FR13_FIXED32_SFWD_PRIOR_REUSE_PASS_PATH", str(output))
    binding = {
        "source_commit": "2" * 40,
        "source_manifest_sha256": "3" * 64,
        "candidate_source_sha256": hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
        "candidate_kernel_source_sha256": hashlib.sha256(
            KERNEL_MODULE_PATH.read_bytes()
        ).hexdigest(),
    }
    candidate._pass_emit(
        task_marker="swe_verified:astropy__astropy-12907",
        batch=1,
        layers={
            index: hashlib.sha256(str(index).encode()).hexdigest()
            for index in range(48)
        },
        source_binding=binding,
    )
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["candidate"] == candidate.CANDIDATE
    assert payload["source_commit"] == binding["source_commit"]
    assert payload["source_manifest_sha256"] == binding["source_manifest_sha256"]
    assert payload["compared_byte_surfaces"] == [
        "conv_out",
        "commit_source_stage",
    ]
    with pytest.raises(RuntimeError, match="B1-only"):
        candidate._pass_emit(
            task_marker="swe_verified:astropy__astropy-12907",
            batch=2,
            layers={
                index: hashlib.sha256(str(index).encode()).hexdigest()
                for index in range(48)
            },
            source_binding=binding,
        )


def test_launcher_uses_packed_xgather_kernel_and_exact_layout() -> None:
    kernel = _function_source(
        "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel",
        path=KERNEL_MODULE_PATH,
    )
    launcher = _function_source("launch_fixed32_sfwd_prior_reuse")
    assert kernel.index("prior_0 = tl.load(") < kernel.index(
        "for tap in tl.static_range(0, WIDTH - 1):"
    )
    assert "tl.where(source_row == 1, prior_1, prior_2)" in kernel
    assert "current_x * current_weight" in kernel
    assert "source_stage + stage_offset + 2 * C + offs_c" in kernel
    assert "source_flat" not in kernel
    assert "x_stride_row" not in kernel
    assert "weight_stride_c" not in kernel
    assert "weight_stride_w" not in kernel
    assert "x_batch = x + pid_b * N * X_STRIDE_ROW" in kernel
    assert "weight_channels = conv_weights + offs_c * WIDTH" in kernel
    assert kernel.count("tl.load(x_batch") == 1
    assert "tl.gather(current_x, x_index, axis=0)" in kernel
    assert "grid = (batch, triton.cdiv(channels, BLOCK_C))" in launcher
    assert "fixed32_specialized_layout_contract(" in launcher
    assert "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel[grid](" in launcher
    assert "X_STRIDE_ROW=X_ROW_STRIDE" in launcher
    assert "ROWS_PER_PROGRAM=ROWS_PER_PROGRAM" in launcher
    assert "BLOCK_C=BLOCK_C" in launcher
    assert "num_warps=NUM_WARPS" in launcher
    assert "source_flat" not in launcher
    assert ".cpu(" not in launcher
    assert ".item(" not in launcher
    assert ".tolist(" not in launcher


def test_launcher_requires_exact_host_parent_vector() -> None:
    parent = list(candidate.FIXED32_PARENT)
    assert candidate._validate_fixed32_tree_parent(parent) == candidate.FIXED32_PARENT
    parent[-1] = 0
    with pytest.raises(RuntimeError, match="host parent vector drifted"):
        candidate._validate_fixed32_tree_parent(parent)
    with pytest.raises(RuntimeError, match="host parent vector drifted"):
        candidate._validate_fixed32_tree_parent(parent[:-1])
    for malformed in (torch.tensor(candidate.FIXED32_PARENT), [False], None):
        with pytest.raises(ValueError, match="host int list/tuple"):
            candidate._validate_fixed32_tree_parent(malformed)


def test_wiring_is_exclusive_reference_served_and_preserves_old_pass() -> None:
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    generic = GENERIC_RUNNER_PATH.read_text(encoding="utf-8")
    gate = GATE_PATH.read_text(encoding="utf-8")
    production = PRODUCTION_PATH.read_text(encoding="utf-8")
    assert "_FR13_FIXED32_SFWD_PRIOR_REUSE_IMPORT" in patcher
    assert "launch_fixed32_sfwd_prior_reuse" in patcher
    assert "fixed32_sfwd_prior_reuse_byte_gate" in patcher
    assert "mixed_qkv_spec = _fr10_tree_conv_out" in patcher
    assert '_fr13_sfwd_candidate_kind == "prior_reuse"' in patcher
    assert "SFWD prior-reuse must be the only SFWD candidate" in launcher
    assert "fr13_fixed32_sfwd_prior_reuse_byte_ab.enabled" in launcher
    assert "FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256" in launcher
    assert (
        '_fr13_sfwd_state_fusion_timing="${FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB:-0}"'
        in launcher
    )
    assert "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB:-0" in generic
    assert "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0" in runner
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0" in runner
    assert "unset FR13_NEEDS_ALLOW FR10_ALLOW_LINEAR_FALLBACK" in runner
    assert "source_manifest.at_launch.json" in runner
    assert "source_manifest.at_end.json" in runner
    assert "source_descriptor_in_kernel=false" in runner
    assert "current_x_global_loads_per_element=1" in runner
    assert "conv_num_warps=16" in runner
    assert "topology_host_validation=exact_parent_each_launch" in runner
    assert "source_descriptor_device_validation=false" in runner
    assert "source_descriptor_launcher_argument=false" in runner
    assert "tree_parent=_fr10_parent" in patcher
    assert "x_stride=16384,1" in runner
    assert "reference_gdn_source_bound" in gate
    assert "fr13_sfwd_prior_reuse_descriptorless.py" in gate
    assert "candidate_kernel_source_sha256" in gate
    assert f'CANDIDATE = "{candidate.CANDIDATE}"' in gate
    assert "tuple(mixed_qkv_spec.shape)" in patcher
    assert "_fr13_sfwd_candidate_out = torch.empty(" in patcher
    assert candidate.CANDIDATE not in production
    assert 'CANDIDATE = "fixed32_sfwd_state_fusion_v1"' in production
    assert hashlib.sha256(OLD_KERNEL_PATH.read_bytes()).hexdigest() == (
        "6c0f0ad607f15ea2727c2a9b244b1fe1c5ddb88268d70264c08f10470a5d2098"
    )


def test_both_prior_reuse_calls_match_descriptorless_launcher_signature() -> None:
    launcher_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    launcher = next(
        node
        for node in launcher_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_fixed32_sfwd_prior_reuse"
    )
    required = {argument.arg for argument in launcher.args.kwonlyargs}
    calls = _direct_call_keywords(
        "launch_fixed32_sfwd_prior_reuse", path=PATCHER_PATH
    )

    assert len(calls) == 2
    assert all(keywords == required for keywords in calls)
    assert all("tree_parent" in keywords for keywords in calls)
    assert all("source_flat" not in keywords for keywords in calls)
