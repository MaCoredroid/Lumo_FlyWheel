from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


QROW32_B4_SO_SHA256 = (
    "77f3fb22c19d0eb2ac0ec28230cf9401221425692a505efde62aa838760d81ce"
)
QROW32_B4_SO_SIZE = 299876120
QROW32_B4_SOURCE_CLOSURE_SHA256 = (
    "dd3bebd047b8ccc2248b0d0e75b9db1f23747c486592ec2a5c72ee96581e10dc"
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
QROW32_B4_KERNEL_HEADER_SHA256 = (
    "f93bc31541a762abe834a16bc4a85b46b7e5a8f1a4463f4abbd6526ea104bce8"
)
GQA_PAIR_KERNEL_HEADER_SHA256 = (
    "43f093e9390efbb57294c2db93c42fcd9c79b3ece2b2768991ff20c814741456"
)
ARTIFACT = Path(
    "results/fr13_fixed32_fa2_qrow32_b4_candidate_sm121a_20260805"
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
    assert "params.k_batch_stride == 2 * 1024 * 4 * 256" in b4
    assert "params.v_batch_stride == 2 * 1024 * 4 * 256" in b4
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
    # SUPERSEDED: the banked 20260805 B4 artifacts were built before the
    # interleaved-KV correction. A live B4 diagnostic (2026-08-10) observed
    # key_cache block stride 2*1024*4*256, so every B4 translation unit and the
    # shared flash_fwd_kernel.h assertion now carry the doubled page stride and
    # the generator can no longer reproduce those artifacts. Their pinned .so
    # files therefore fail closed at their own source-closure check, which is
    # the intended behaviour -- they must be rebuilt before they can be gated.
    assert hashlib.sha256(qrow32_source).hexdigest() != QROW32_B4_SOURCE_SHA256
    assert "page = 2 * 1024 * 4 * 256" in module.FIXED32_QUERY_TILE32_TRANSLATION_UNIT
    closure = json.loads((ARTIFACT / "source_closure.json").read_text("ascii"))
    canonical_closure = json.dumps(
        closure,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert hashlib.sha256(canonical_closure).hexdigest() == (
        QROW32_B4_SOURCE_CLOSURE_SHA256
    )
    assert closure["fa2_head"] == "29210221863736a08f71a866459e368ad1ac4a95"
    assert closure["patcher_flags"] == [
        "--tree-bias-tile-earlyout",
        "--fixed32-query-tile32",
    ]
    assert closure["fa2_status"] == [
        " M csrc/flash_attn/flash_api.cpp",
        " M csrc/flash_attn/flash_api_torch_lib.cpp",
        " M csrc/flash_attn/src/flash.h",
        " M csrc/flash_attn/src/flash_fwd_kernel.h",
        " M csrc/flash_attn/src/utils.h",
        "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
    ]
    assert closure["base_gqa_pair_source_closure_sha256"] == (
        GQA_PAIR_API_SOURCE_CLOSURE_SHA256
    )
    assert closure["fa2_files"][
        "csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu"
    ] == QROW32_B4_SOURCE_SHA256
    assert closure["fa2_files"]["csrc/flash_attn/src/flash_fwd_kernel.h"] == (
        QROW32_B4_KERNEL_HEADER_SHA256
    )
    assert QROW32_B4_KERNEL_HEADER_SHA256 != GQA_PAIR_KERNEL_HEADER_SHA256
    assert closure["repo_files"][
        "csrc/fr13_fa2_qrow32_b4_launcher_adapter.cc"
    ] == QROW32_B4_ADAPTER_SHA256

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


def test_qrow32_b4_artifact_is_checksum_and_provenance_complete() -> None:
    checksum_records = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text("ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksum_records[name] = digest
    assert set(checksum_records) == {
        "README.md",
        "binary_inventory.tsv",
        "build_commands.txt",
        "codegen_evidence.txt",
        "device_evidence.txt",
        "guarded_so_finalize_manifest.json",
        "guarded_static_gate.json",
        "launch_work.tsv",
        "link_input_objects.tsv",
        "manifest.json",
        "source_closure.json",
        "test_results.txt",
    }
    for name, digest in checksum_records.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest

    manifest = json.loads((ARTIFACT / "manifest.json").read_text("ascii"))
    assert manifest["candidate"]["source_closure_sha256"] == (
        QROW32_B4_SOURCE_CLOSURE_SHA256
    )
    assert manifest["device"]["static_shared_bytes"] == 1024
    assert manifest["device"]["dynamic_shared_bytes"] == 81920
    assert manifest["device"]["ptx_present"] is False
    assert manifest["launch_work"]["qrow16_threads_per_layer"] == 6144
    assert manifest["launch_work"]["qrow32_threads_per_layer"] == 6144
    assert manifest["launch_work"]["qrow16_warps_per_layer"] == 192
    assert manifest["launch_work"]["qrow32_warps_per_layer"] == 192

    static_gate = json.loads(
        (ARTIFACT / "guarded_static_gate.json").read_text("ascii")
    )
    assert static_gate["object_sha256"] == manifest["candidate"]["object_sha256"]
    assert static_gate["candidate_so_sha256"] == QROW32_B4_SO_SHA256
    assert static_gate["resources"]["dynamic_shared_bytes"] == 81920

    finalizer = json.loads(
        (ARTIFACT / "guarded_so_finalize_manifest.json").read_text("ascii")
    )
    assert finalizer["output_sha256"] == QROW32_B4_SO_SHA256
    assert finalizer["output_size"] == QROW32_B4_SO_SIZE
    assert finalizer["candidate_input_sha256"] == (
        manifest["candidate"]["raw_link_sha256"]
    )

    commands = (ARTIFACT / "build_commands.txt").read_text("ascii")
    for stage in (
        "[compile-qrow32]",
        "[pin-thrust-abi]",
        "[compile-adapter]",
        "[link]",
        "[finalize]",
        "[static-gate]",
    ):
        assert commands.count(stage) == 1
    assert QROW32_B4_SO_SHA256 in commands
    assert manifest["candidate"]["object_sha256"] in commands

    link_inputs = (ARTIFACT / "link_input_objects.tsv").read_text("ascii")
    assert link_inputs.count("\nqrow32_object\t") == 1
    assert link_inputs.count("\nadapter_object\t") == 1
    assert link_inputs.count("\ngqa_api\t") == 1
    assert manifest["candidate"]["object_sha256"] in link_inputs
