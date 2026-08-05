from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m4_shuffle_r64_u8.py"
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"


def _assignment(source: str, name: str) -> object:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name}")


def test_builder_pins_source_and_deployed_toolchain() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    assert _assignment(source, "SOURCE_SHA256") == (
        "a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e"
    )
    assert _assignment(source, "EXPECTED_TORCH") == "2.11.0+cu130"
    assert _assignment(source, "EXPECTED_CUDA") == "13.0"
    assert _assignment(source, "EXPECTED_ARCH") == "12.1a"
    assert "fr13_bf16_k64_m4_r64_u8_sm121a" in source
    assert "gemvx_m4_shuffle_r64_u8_out" in source
    assert '"--frandom-seed=fr13_bf16_k64_m4_r64_u8"' in source


def test_builder_attests_exact_b4_reused_weight_contract() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    for contract in (
        '"batch_scope": "B4_exact"',
        '"grid": [1024, 1, 1]',
        '"block": [16, 64, 1]',
        '"input": "BF16[4,5120] contiguous"',
        '"weight": "BF16[65536,5120] contiguous"',
        '"output": "BF16[4,65536] contiguous"',
        '"independent_accumulators": 4',
        '"weight_reuse_batch": 4',
    ):
        assert contract in source
    assert SOURCE.name in source
    assert '"gpu_runtime_used": False' in source
    assert '"byte_equality_claim": False' in source
