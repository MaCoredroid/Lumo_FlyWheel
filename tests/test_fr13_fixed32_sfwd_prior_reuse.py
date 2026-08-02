from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
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

GATE_SPEC = importlib.util.spec_from_file_location("fr13_sfwd_prior_reuse_gate", GATE_PATH)
assert GATE_SPEC is not None and GATE_SPEC.loader is not None
gate_module = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(gate_module)


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


def test_contract_closes_adaptive_row32_geometry_for_b1_b4() -> None:
    for batch in (1, 2, 3, 4):
        contract = candidate.fixed32_sfwd_prior_reuse_contract(
            batch, tree_rows=32, conv_width=4, conv_state_len=34
        )
        assert contract["candidate"] == candidate.CANDIDATE
        assert contract["physical_rows_per_request"] == 32
        assert contract["conv_rows_per_program"] == 32
        assert contract["conv_row_groups_per_request"] == 1
        expected_geometry = (128, 2) if batch == 1 else (256, 4)
        assert contract["conv_block_c"] == expected_geometry[0]
        assert contract["conv_num_warps"] == expected_geometry[1]
        assert contract["conv_peak_live_x"] == 5
        assert contract["conv_live_x_sum"] == 116
        assert contract["conv_activation_window"] == 2
        assert contract["conv_peak_live_acc"] == 2
        assert contract["conv_peak_live_x_with_deferred_stage"] == 5
        assert contract["conv_live_x_sum_with_deferred_stage"] == 125
        assert contract["x_global_loads_per_channel"] == 32
        assert contract["x_reload_count"] == 0
        assert tuple(contract["conv_node_order"]) == (
            27, 25, 23, 18, 13, 8, 3, 0, 2, 7, 12, 17, 22, 1, 5, 6,
            4, 10, 11, 9, 15, 16, 14, 20, 21, 19, 24, 26, 28, 29, 30, 31,
        )
        assert contract["topology_host_validation"] == "exact_parent_each_launch"
        assert contract["source_descriptor_device_validation"] is False
        assert contract["source_descriptor_launcher_argument"] is False
        assert contract["candidate"] == (
            "fixed32_sfwd_channel_serial_r32_b1c128w2_bxc256w4_"
            "u32x2_frontier5_loadonce_act2_v4"
        )
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


def test_launcher_uses_channel_serial_kernel_and_exact_layout() -> None:
    kernel = _function_source(
        "_fr13_fixed32_sfwd_channel_serial_kernel",
        path=KERNEL_MODULE_PATH,
    )
    launcher = _function_source("launch_fixed32_sfwd_prior_reuse")
    assert kernel.index("prior_0 = tl.load(") < kernel.index("x_0 = tl.load(")
    assert "tap_0 = (" not in kernel
    assert "tap_1 = (" not in kernel
    assert "tap_2 = (" not in kernel
    assert "for node in tl.static_range(0, N):" not in kernel
    assert "product_3 = (x_31 * weight_3)" in kernel
    assert "tl.gather" not in kernel
    assert "stage_batch + 2 * C + offs_c" in kernel
    assert "source_flat" not in kernel
    assert "x_stride_row" not in kernel
    assert "weight_stride_c" not in kernel
    assert "weight_stride_w" not in kernel
    assert "CONV_STRIDE_ROW" in kernel
    assert "bank_row * CONV_STRIDE_ROW + offs_c" in kernel
    assert "prior_base + C" in kernel
    assert "prior_base + 2 * C" in kernel
    assert "ssi_stride_b" not in kernel
    assert "ssi_stride_s" not in kernel
    assert "spec_state_indices + pid_b * N" in kernel
    assert "x_batch = x + pid_b * N * X_STRIDE_ROW" in kernel
    assert "weight_channels = conv_weights + offs_c * WIDTH" in kernel
    assert "weight_pair_01" in kernel
    assert "weight_pair_23" in kernel
    assert "tl.pointer_type(tl.uint64)" not in kernel
    assert kernel.count("tl.load(x_batch") == 32
    assert kernel.count("tl.store(out_batch +") == 32
    assert kernel.count("tl.store(stage_batch + ((WIDTH - 1) +") == 32
    for row in range(32):
        assert kernel.count(
            f"x_{row} = tl.load(x_batch + {row} * X_STRIDE_ROW + offs_c)"
        ) == 1
    assert "Exact load-once order keeps the peak 5 frontier" in kernel
    assert "two independent activation chains" in kernel
    assert 'block_c = int(contract["conv_block_c"])' in launcher
    assert 'num_warps = int(contract["conv_num_warps"])' in launcher
    assert "grid = (batch, triton.cdiv(channels, block_c))" in launcher
    assert "fixed32_specialized_layout_contract(" in launcher
    assert "int(conv_state.stride(1)) != 1" in launcher
    assert "int(conv_state.stride(2)) != channels" in launcher
    assert "int(conv_state.stride(0)) < channels * state_len" in launcher
    assert "not spec_state_indices.is_contiguous()" in launcher
    assert "int(spec_state_indices.shape[1]) != rows" in launcher
    assert "int(conv_weights.data_ptr()) % 4 != 0" in launcher
    assert "x_out_source_storage_alias" in launcher
    assert "tensor.untyped_storage().data_ptr()" in launcher
    assert "_fr13_fixed32_sfwd_channel_serial_kernel[grid](" in launcher
    assert "X_STRIDE_ROW=X_ROW_STRIDE" in launcher
    assert "ROWS_PER_PROGRAM=ROWS_PER_PROGRAM" not in launcher
    assert "BLOCK_C=block_c" in launcher
    assert "num_warps=num_warps" in launcher
    assert "source_flat" not in launcher
    assert "CONV_STRIDE_ROW=int(conv_state.stride(0))" in launcher
    assert "int(spec_state_indices.stride(" not in launcher
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
    assert "host-readiness" in runner
    assert "sfwd_prior_reuse_host_readiness.json" in runner
    assert "source_descriptor_in_kernel=false" in runner
    assert "x_global_loads_per_channel=32" in runner
    assert "x_reload_count=0" in runner
    assert "conv_peak_live_x=5" in runner
    assert "conv_live_x_sum=116" in runner
    assert "conv_activation_window=2" in runner
    assert "conv_peak_live_acc=2" in runner
    assert "conv_peak_live_x_with_deferred_stage=5" in runner
    assert "conv_live_x_sum_with_deferred_stage=125" in runner
    assert "conv_node_order=27,25,23,18,13,8,3,0" in runner
    assert "conv_block_c=128" in runner
    assert "conv_num_warps=2" in runner
    assert "topology_host_validation=exact_parent_each_launch" in runner
    assert "source_descriptor_device_validation=false" in runner
    assert "source_descriptor_launcher_argument=false" in runner
    assert "tree_parent=_fr10_parent" in patcher
    assert "x_stride=16384,1" in runner
    assert "conv_state_layout=bank,channel,state" in runner
    assert "conv_state_stride=2097152,1,10240" in runner
    assert "spec_state_indices_width=32" in runner
    assert "reference_gdn_source_bound" in gate
    assert "fr13_sfwd_prior_reuse_descriptorless.py" in gate
    assert "candidate_kernel_source_sha256" in gate
    assert "HOST_READINESS_SCHEMA" in gate
    assert "docker/chat_templates/qwen3-openai-codex.jinja" in gate
    assert candidate.CANDIDATE in gate
    assert "tuple(mixed_qkv_spec.shape)" in patcher
    assert "_fr13_sfwd_candidate_out = torch.empty(" in patcher
    assert candidate.CANDIDATE not in production
    assert 'CANDIDATE = "fixed32_sfwd_state_fusion_v1"' in production
    assert hashlib.sha256(OLD_KERNEL_PATH.read_bytes()).hexdigest() == (
        "f70b73659fa18645a58b16d63b523cd4e22d1f5422bc05cc2a03dd8be2b551f4"
    )


def test_host_readiness_is_pushed_host_only_and_binds_strict_b1_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    python_path = repo / ".venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    python_path.chmod(0o755)
    fa2_relative = "output/stock-fa2.so"
    fa2 = repo / fa2_relative
    fa2.parent.mkdir(parents=True)
    fa2.write_bytes(b"stock-fa2")

    source_relative = "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
    kernel_relative = (
        "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
    )
    subset_relative = "config/subset.json"
    blocks_relative = "scripts/blocks.json"
    template_relative = "docker/chat-template.jinja"
    source_files = (
        subset_relative,
        template_relative,
        source_relative,
        kernel_relative,
        blocks_relative,
    )
    subset_sha = "a" * 64
    blocks_sha = "b" * 64
    template_sha = "c" * 64
    commit = "1" * 40
    files = {
        relative: {"bytes": 1, "sha256": "d" * 64}
        for relative in source_files
    }
    files[subset_relative]["sha256"] = subset_sha
    files[blocks_relative]["sha256"] = blocks_sha
    files[template_relative]["sha256"] = template_sha
    manifest_payload = {
        "schema": gate_module.SOURCE_SCHEMA,
        "candidate": gate_module.CANDIDATE,
        "source_commit": commit,
        "reference_gdn_source_bound": True,
        "files": files,
    }
    manifest = tmp_path / "source-manifest.json"
    manifest_raw = (
        json.dumps(manifest_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    manifest.write_bytes(manifest_raw)

    monkeypatch.setattr(gate_module, "SOURCE_FILES", source_files)
    monkeypatch.setattr(gate_module, "SUBSET_RELATIVE", subset_relative)
    monkeypatch.setattr(gate_module, "SUBSET_SHA256", subset_sha)
    monkeypatch.setattr(gate_module, "DRAFT_VOCAB_BLOCKS_RELATIVE", blocks_relative)
    monkeypatch.setattr(gate_module, "DRAFT_VOCAB_BLOCKS_SHA256", blocks_sha)
    monkeypatch.setattr(gate_module, "CHAT_TEMPLATE_RELATIVE", template_relative)
    monkeypatch.setattr(gate_module, "CHAT_TEMPLATE_SHA256", template_sha)
    monkeypatch.setattr(gate_module, "FA2_REPO_RELATIVE", fa2_relative)
    monkeypatch.setattr(gate_module, "FA2_SIZE", len(b"stock-fa2"))
    monkeypatch.setattr(
        gate_module, "FA2_SHA256", hashlib.sha256(b"stock-fa2").hexdigest()
    )

    pushed = True

    def fake_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
            return ""
        if arguments == ("symbolic-ref", "--short", "HEAD"):
            return "agent/test"
        if arguments == ("rev-parse", "@{upstream}"):
            return commit if pushed else "2" * 40
        raise AssertionError(arguments)

    def fake_manifest(args) -> None:
        Path(args.output).write_bytes(manifest_raw)

    monkeypatch.setattr(gate_module, "_git_stdout", fake_git)
    monkeypatch.setattr(gate_module, "write_source_manifest", fake_manifest)
    output = tmp_path / "readiness.json"
    args = types.SimpleNamespace(
        repo=str(repo),
        source_commit=commit,
        source_manifest=str(manifest),
        fa2_so=str(fa2),
        output=str(output),
    )
    gate_module.write_host_readiness(args)
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["source_commit"] == payload["upstream_commit"] == commit
    assert payload["source_binding"]["file_count"] == len(source_files)
    assert payload["byte_gate"]["compared_surfaces"] == [
        "conv_out",
        "commit_source_stage",
    ]
    assert payload["byte_gate"]["required_layer_count"] == 48
    assert payload["byte_gate"]["reference_always_served"] is True
    assert payload["launch_policy"] == {
        "default_off": True,
        "host_only_preflight": True,
        "gpu_or_docker_used": False,
        "launched": False,
        "runtime_correctness_qualified": False,
    }

    pushed = False
    with pytest.raises(gate_module.GateError, match="not pushed"):
        gate_module.write_host_readiness(args)
