from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ARTIFACT = Path(
    "results/fr13_fixed32_cfwd_packed_walk_active_depth_sm121a_codegen_20260805"
)
SOURCE = Path("scripts/fr13_cfwd_packed_walk_active_depth_kernel.py")


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_artifact_binds_exact_candidate_source() -> None:
    summary = _summary()
    assert summary["candidate_revision"] == "cd1398aee"
    assert summary["source_sha256"]["candidate"] == hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    assert summary["claim_scope"] == (
        "static_sm121a_codegen_no_gpu_runtime_claim"
    )


def test_artifact_static_delta_is_clean_for_b1_b4() -> None:
    summary = _summary()
    for batch in ("b1", "b4"):
        base = summary["builds"]["base"][batch]
        candidate = summary["builds"]["candidate"][batch]
        assert candidate["registers"] == 31 < base["registers"] == 44
        assert candidate["ldg"] == 2 < base["ldg"] == 24
        assert candidate["stg"] == 8 < base["stg"] == 41
        assert candidate["static_noncontrol_sass_instructions"] == 81
        assert base["static_noncontrol_sass_instructions"] == 496
        assert candidate["encoded_sass_instructions"] == 96
        assert base["encoded_sass_instructions"] == 512
        assert candidate["bra"] == 2
        assert all(
            candidate[name] == 0
            for name in ("stack_bytes", "local_bytes", "ldl", "stl", "calls")
        )


def test_checked_in_summary_passes_unchanged_verifier() -> None:
    path = ARTIFACT / "verify_codegen_outputs.py"
    spec = importlib.util.spec_from_file_location("active_depth_verify", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(_summary())


def test_readme_scopes_runtime_claims() -> None:
    readme = (ARTIFACT / "README.md").read_text(encoding="ascii")
    normalized = " ".join(readme.split())
    assert "static codegen evidence only" in normalized
    assert "does not establish GPU byte equality or runtime speed" in normalized
    assert "real SWE-Verified B1/B4 byte gate" in normalized
