from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle_r64_u8.py"
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu"


def load_builder():
    spec = importlib.util.spec_from_file_location("fr13_dfwd_u8_builder", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_is_pinned_to_the_reviewed_source_and_toolchain() -> None:
    module = load_builder()
    assert module.SOURCE == SOURCE
    assert module.SOURCE_SHA256 == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert module.EXPECTED_TORCH == "2.11.0+cu130"
    assert module.EXPECTED_CUDA == "13.0"
    assert module.EXPECTED_ARCH == "12.1a"


def test_builder_keeps_candidate_separate_and_default_off() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    assert "fr13_bf16_k64_m1_r64_u8_sm121a" in source
    assert "gemvx_m1_shuffle_r64_u8_out" in source
    assert '"production_default_enabled": False' in source
    assert '"runtime_wired": False' in source
    assert '"gpu_runtime_used": False' in source
    assert '"--frandom-seed=fr13_bf16_k64_m1_r64_u8"' in source
    assert "FR13_DRAFT_HEAD" not in source
