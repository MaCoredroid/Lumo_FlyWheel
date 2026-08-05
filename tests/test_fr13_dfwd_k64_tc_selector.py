from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "fr13_dfwd_k64_tc_selector.py"
PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNTIME_MANIFEST = ROOT / "scripts" / "fr13_runtime_manifest.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("fr13_k64_tc_selector", SELECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(commit: str, batch: int = 1) -> dict[str, str]:
    return {
        "FR13_DRAFT_HEAD_K64_TC": "1",
        "FR13_DRAFT_HEAD_K64_TC_SOURCE_COMMIT": commit,
        "FR13_FIXED32_MODE": "hydra27_fixed32",
        "MAX_NUM_SEQS": str(batch),
        "SWE_CONCURRENCY": str(batch),
        "ENFORCE_EAGER": "0",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_BLOCKS": (
            "/workspace/scripts/fr13_dvk_subset_blocks.json"
        ),
        "FR13_DRAFTER_SINGLE_LOGITS": "1",
        "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
        "FR13_FIXED32_ACTIVE_NODES": "27",
        "NUM_SPECULATIVE_TOKENS": "31",
    }


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "new"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and "FR13_DRAFT_HEAD_K64_TC" in node.value.value
        ):
            return node.value.value
    raise AssertionError("K64 Tensor Core Eagle replacement not found")


@pytest.mark.parametrize(
    ("batch", "operation"),
    [
        (1, "gemm_m1_tc16x256x64_s2_out"),
        (4, "gemm_m4_tc16x256x64_s2_out"),
    ],
)
def test_validator_accepts_only_exact_b1_b4(batch: int, operation: str) -> None:
    selector = _load_selector()
    commit = "a" * 40
    result = selector.validate_environment(_env(commit, batch), ROOT, commit)
    assert result["status"] == "PASS"
    assert result["batch_size"] == batch
    assert result["operation"] == operation
    assert result["physical_drafts"] == 31
    assert result["active_nodes"] == 27
    assert result["proposal_only"] is True
    assert result["target_authority_changed"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("FR13_FIXED32_MODE", "tail6_fixed32"),
        ("MAX_NUM_SEQS", "2"),
        ("SWE_CONCURRENCY", "4"),
        ("ENFORCE_EAGER", "1"),
        ("CUDAGRAPH_MODE", "PIECEWISE"),
        ("FR13_DRAFT_VOCAB_ROOT", "0"),
        ("FR13_DRAFT_VOCAB_K", "32768"),
        ("FR13_DRAFTER_SINGLE_LOGITS", "0"),
        ("FR13_FIXED32_PHYSICAL_DRAFTS", "27"),
        ("FR13_FIXED32_ACTIVE_NODES", "31"),
        ("NUM_SPECULATIVE_TOKENS", "27"),
        ("FR13_DRAFT_HEAD_B14_WARP4_PAIR8", "1"),
        ("FR13_DRAFT_HEAD_PAD_ROWS", "32"),
        ("FR13_DRAFT_HEAD_M32_PRODUCTION", "1"),
        ("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "1"),
        ("FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION", "1"),
        ("FR13_DRAFT_HEAD_FP8", "1"),
        ("FR13_DFWD_K64_TOP3", "1"),
    ],
)
def test_validator_fails_closed_on_geometry_or_competing_mode(
    key: str, value: str
) -> None:
    selector = _load_selector()
    commit = "b" * 40
    env = _env(commit)
    env[key] = value
    with pytest.raises(selector.SelectorError):
        selector.validate_environment(env, ROOT, commit)


def test_validator_rejects_source_commit_drift() -> None:
    selector = _load_selector()
    with pytest.raises(selector.SelectorError, match="source commit"):
        selector.validate_environment(_env("a" * 40), ROOT, "b" * 40)


@pytest.mark.parametrize("rel_name", ["SOURCE_REL", "SO_REL", "BUILD_REL", "MANIFEST_REL"])
def test_validator_rejects_authenticated_input_tamper(
    tmp_path: Path, rel_name: str
) -> None:
    selector = _load_selector()
    for name in ("SOURCE_REL", "SO_REL", "BUILD_REL", "MANIFEST_REL"):
        rel = getattr(selector, name)
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    tampered = tmp_path / getattr(selector, rel_name)
    tampered.chmod(0o644)
    tampered.write_bytes(tampered.read_bytes() + b"x")
    commit = "c" * 40
    with pytest.raises(selector.SelectorError, match="SHA-256 drifted"):
        selector.validate_environment(_env(commit), tmp_path, commit)


def test_runtime_dispatch_is_default_off_direct_and_fail_closed() -> None:
    snippet = _eagle_snippet()
    wrapped = "def _patched_propose_draft_token_ids():\n" + textwrap.indent(
        textwrap.dedent(snippet), "    "
    )
    compile(wrapped, "<fr13_k64_tc_eagle>", "exec")
    setup = snippet[
        snippet.index("_fr13_dh_tc_raw") : snippet.index("def _fr13_dvk_prepare")
    ]
    helper = snippet[
        snippet.index("def _fr13_dh_pair8_logits") : snippet.index(
            "def _fr13_dvk_logits"
        )
    ]
    dispatch = snippet[
        snippet.index("def _fr13_dvk_logits") : snippet.index(
            "def _fr13_dvk_real_ids"
        )
    ]
    assert '"FR13_DRAFT_HEAD_K64_TC", "0"' in setup
    assert '_FR13_FIXED32_MODE != "hydra27_fixed32"' in setup
    assert "int(batch_size) not in (1, 4)" in setup
    assert "!= (3, 3, 3, 3, 3)" in setup
    assert "fr13_bf16_gemm_k64_tc16x256x64_s2.cu" in snippet
    assert "fr13_bf16_k64_tc_head" in snippet
    assert "gemm_m1_tc16x256x64_s2_out" in snippet
    assert "gemm_m4_tc16x256x64_s2_out" in snippet
    assert "self._fr13_dh_pair8_op(" in helper
    assert '_fr13_dh_pair8_runtime_marker = (' in helper
    assert '"FR13_DRAFT_HEAD_K64_TC"' in helper
    assert '"FR13_DRAFT_HEAD_B14_WARP4_PAIR8"' in helper
    assert 'f"[{_fr13_dh_pair8_runtime_marker}] engaged "' in helper
    assert 'f"[{_fr13_dh_pair8_marker}] engaged "' not in helper
    assert "quant_method.apply" not in helper
    assert "compute_logits" not in helper
    assert "torch.cuda.synchronize" not in helper
    assert "target_authority_unchanged=1" in helper
    assert "if _fr13_dh_tc_on:" in dispatch
    assert "incumbent fallback is forbidden" in dispatch
    assert "rejection" not in helper.lower()
    assert "accept" not in helper.lower()


def test_capture_attestation_requires_root_plus_exact_four_loop_calls() -> None:
    snippet = _eagle_snippet()
    assert "self._fr13_dh_tc_selected_capture_calls != 4" in snippet
    assert "self._fr13_dh_tc_selected_root_calls < 1" in snippet
    assert "self._fr13_dh_tc_fallback_calls != 0" in snippet
    assert "[FR13_DRAFT_HEAD_K64_TC] graph" in snippet


def test_launcher_and_runtime_manifest_close_real_route() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    manifest = RUNTIME_MANIFEST.read_text(encoding="utf-8")
    assert "FR13_DRAFT_HEAD_K64_TC=${FR13_DRAFT_HEAD_K64_TC:-0}" in launcher
    assert 'case "$FR13_DRAFT_HEAD_K64_TC" in' in launcher
    assert "scripts/fr13_dfwd_k64_tc_selector.py" in launcher
    assert '-e FR13_DRAFT_HEAD_K64_TC="$FR13_DRAFT_HEAD_K64_TC"' in launcher
    assert "FR13_FIXED32_PHYSICAL_DRAFTS=" in launcher
    assert "FR13_FIXED32_ACTIVE_NODES=" in launcher
    assert "scripts/fr13_dfwd_k64_tc_selector.py" in manifest
    assert "csrc/fr13_bf16_gemm_k64_tc16x256x64_s2.cu" in manifest
    assert "fr13_bf16_k64_tc16x256x64_s2.abi3.so" in manifest


def test_python_and_shell_sources_parse() -> None:
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(SELECTOR), str(PATCHER)],
        check=True,
    )
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
