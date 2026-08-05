from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts/fr13_dfwd_k64_fp8_selector.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("fr13_k64_fp8_selector", SELECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(commit: str, batch: int = 1) -> dict[str, str]:
    return {
        "FR13_DRAFT_HEAD_FP8": "1",
        "FR13_DRAFT_HEAD_FP8_STATIC_IO": "1",
        "FR13_DRAFT_HEAD_FP8_ARM": f"hydra27_k64_fp8_b{batch}",
        "FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON": (
            "/logs/fr13_draft_head_fp8.engagement.json"
        ),
        "FR13_DRAFT_HEAD_FP8_SOURCE_COMMIT": commit,
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
        "FR13_MANDATORY_WEIGHT_BYTES": "30989326208",
        "FR13_WEIGHT_FLOOR_MS": "113.514015414",
    }


@pytest.mark.parametrize("batch", [1, 4])
def test_selector_accepts_exact_b1_b4(batch: int) -> None:
    selector = _load_selector()
    commit = "a" * 40
    result = selector.validate_environment(_env(commit, batch), ROOT, commit)
    assert result["status"] == "PASS"
    assert result["batch_size"] == batch
    assert result["operation"] == "vllm_cutlass_block_fp8_scaled_mm_static_io"
    assert result["five_call_mandatory_bytes"] == 1_678_131_200
    assert result["full_step_mandatory_bytes"] == 30_989_326_208
    assert result["proposal_only"] is True
    assert result["target_authority_changed"] is False
    assert result["static_smoke_is_performance_evidence"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("FR13_DRAFT_HEAD_FP8_STATIC_IO", "0"),
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
        ("FR13_DRAFT_HEAD_K64_TC", "1"),
        ("FR13_DRAFT_HEAD_B14_WARP4_PAIR8", "1"),
        ("FR13_DRAFT_HEAD_PAD_ROWS", "32"),
        ("FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION", "1"),
        ("FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION", "1"),
        ("FR13_DFWD_K64_TOP3", "1"),
    ],
)
def test_selector_fails_closed_on_geometry_or_competing_mode(
    key: str, value: str
) -> None:
    selector = _load_selector()
    commit = "b" * 40
    env = _env(commit)
    env[key] = value
    with pytest.raises(selector.SelectorError):
        selector.validate_environment(env, ROOT, commit)


def test_selector_rejects_source_commit_and_arm_drift() -> None:
    selector = _load_selector()
    with pytest.raises(selector.SelectorError, match="source commit"):
        selector.validate_environment(_env("a" * 40), ROOT, "b" * 40)
    env = _env("a" * 40)
    env["FR13_DRAFT_HEAD_FP8_ARM"] = "unsafe arm"
    with pytest.raises(selector.SelectorError, match="canonical"):
        selector.validate_environment(env, ROOT, "a" * 40)


@pytest.mark.parametrize(
    "rel_name", ["SOURCE_REL", "SMOKE_SOURCE_REL", "SMOKE_RESULT_REL"]
)
def test_selector_rejects_authenticated_input_tamper(
    tmp_path: Path, rel_name: str
) -> None:
    selector = _load_selector()
    for name in ("SOURCE_REL", "SMOKE_SOURCE_REL", "SMOKE_RESULT_REL"):
        rel = getattr(selector, name)
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    tampered = tmp_path / getattr(selector, rel_name)
    tampered.write_bytes(tampered.read_bytes() + b"x")
    commit = "c" * 40
    with pytest.raises(selector.SelectorError, match="SHA-256 drifted"):
        selector.validate_environment(_env(commit), tmp_path, commit)
