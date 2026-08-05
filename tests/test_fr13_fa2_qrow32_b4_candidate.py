from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


QROW32_B4_SO_SHA256 = (
    "77f3fb22c19d0eb2ac0ec28230cf9401221425692a505efde62aa838760d81ce"
)
QROW32_B4_SO_SIZE = 299876120
QROW32_B4_SOURCE_CLOSURE_SHA256 = (
    "3e3c18565e738f20d0a5ab5fe50d018f3d8cbd5cb94082dcd55ca730a790163c"
)
QROW32_B4_SOURCE_SHA256 = (
    "adee09b98b6f517550547bf73efa6f817c7634b2d50febb6f811e3b637fee1e4"
)
QROW32_B4_ADAPTER_SHA256 = (
    "4873f6dc368bfdbc78ebd2edf4d6d08f5033ae69cec94e0f19ae90610e1f8a6a"
)
GQA_PAIR_API_SOURCE_CLOSURE_SHA256 = (
    "f210a5ebb93930e89b0d9fe0cb6e53a76c9359873ad4268e81d3f17a7443bdf2"
)


def _patcher_module():
    path = Path("scripts/fr13_patch_fa2_tree_bias.py")
    spec = importlib.util.spec_from_file_location("fr13_patch_fa2_tree_bias", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b1_qrow32_launcher_cannot_cover_exact4_b4() -> None:
    module = _patcher_module()
    b1 = module.FIXED32_QUERY_TILE32_B1_TRANSLATION_UNIT
    b4 = module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT

    assert "static constexpr int sequences = 1;" in b1
    assert "params.b == 1" in b1
    assert "params.total_q == 32" in b1
    assert "params.k_batch_stride == 2 * 1024 * 4 * 256" in b1
    assert "params.v_batch_stride == 2 * 1024 * 4 * 256" in b1

    assert "static constexpr int sequences = 4;" in b4
    assert "params.b == 4" in b4
    assert "params.total_q == 128" in b4
    assert "params.k_batch_stride == 1024 * 4 * 256" in b4
    assert "params.v_batch_stride == 1024 * 4 * 256" in b4
    assert "FR13 qrow32 B4 launcher reached non-canonical geometry" in b4


def test_b4_qrow32_halves_ctas_without_adding_a_launch() -> None:
    module = _patcher_module()
    qrow16 = module.FIXED32_QUERY_TILE16_TRANSLATION_UNIT
    qrow32 = module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT

    assert "constexpr static int kTreeBlockM = 16;" in qrow16
    assert "dim3 grid(num_m_block, params.b, params.h);" in qrow16
    assert "constexpr static int kTreeBlockM = 32;" in qrow32
    assert "StaticLayout::query_heads_per_kv," in qrow32
    assert "StaticLayout::sequences," in qrow32
    assert "StaticLayout::kv_heads);" in qrow32
    assert "false,  // Split" in qrow32
    assert qrow32.count("kernel<<<grid") == 1

    qrow16_ctas_per_layer = 2 * 4 * 24
    qrow32_ctas_per_layer = 6 * 4 * 4
    assert qrow16_ctas_per_layer == 192
    assert qrow32_ctas_per_layer == 96
    assert qrow32_ctas_per_layer * 16 == 1536
    assert qrow16_ctas_per_layer * 16 == 3072
    assert qrow32_ctas_per_layer * 2 == qrow16_ctas_per_layer


def test_qrow32_b4_adapter_is_hidden_and_single_purpose() -> None:
    adapter = Path("csrc/fr13_fa2_qrow32_b4_launcher_adapter.cc").read_text(
        encoding="ascii"
    )

    assert "#include" not in adapter
    assert adapter.count('__attribute__((visibility("hidden")))') == 2
    assert adapter.count("fr13_run_mha_fwd_fixed32_qrow32(") == 2
    assert adapter.count("fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(") == 1
    assert "fr13_run_mha_fwd_fixed32_qrow32(params, stream);" in adapter
    assert hashlib.sha256(adapter.encode("ascii")).hexdigest() == (
        QROW32_B4_ADAPTER_SHA256
    )


def test_qrow32_b4_arm_is_pinned_at_every_selector_boundary() -> None:
    module = _patcher_module()
    namespace: dict[str, object] = {}
    exec(module.FIXED32_QUERY_TILE32_LIVE_AB_HELPERS, namespace)
    arms = namespace["_FR13_FA2_QROW32_LIVE_AB_ARMS"]
    assert isinstance(arms, dict)
    contract = arms["qrow32"]
    qrow32_source = module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT.encode("ascii")
    assert hashlib.sha256(qrow32_source).hexdigest() == QROW32_B4_SOURCE_SHA256
    closure_lines = (
        f"{GQA_PAIR_API_SOURCE_CLOSURE_SHA256}  gqa_pair_api_source_closure",
        f"{QROW32_B4_SOURCE_SHA256}  "
        "csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
        f"{QROW32_B4_ADAPTER_SHA256}  "
        "csrc/fr13_fa2_qrow32_b4_launcher_adapter.cc",
    )
    closure = ("\n".join(closure_lines) + "\n").encode("ascii")
    assert hashlib.sha256(closure).hexdigest() == QROW32_B4_SOURCE_CLOSURE_SHA256

    assert contract == {
        "sentinel": 131092,
        "num_splits": 0,
        "candidate_dispatch": "qrow32 BM32 exact B4 geometry; no fallback",
        "candidate_so_sha256": QROW32_B4_SO_SHA256,
        "candidate_so_size": QROW32_B4_SO_SIZE,
        "fa2_head": "29210221863736a08f71a866459e368ad1ac4a95",
        "fa2_source_closure_sha256": QROW32_B4_SOURCE_CLOSURE_SHA256,
    }
    assert 'candidate_arm == "gqa_pair" and' not in (
        module.FIXED32_QUERY_TILE32_LIVE_AB_HELPERS
    )

    launcher = Path("scripts/fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    runner = Path("scripts/fr13_run_b4_fa2_qrow32_live_gate.sh").read_text(
        encoding="ascii"
    )
    for selector in (launcher, runner):
        assert QROW32_B4_SO_SHA256 in selector
        assert str(QROW32_B4_SO_SIZE) in selector
        assert QROW32_B4_SOURCE_CLOSURE_SHA256 in selector
