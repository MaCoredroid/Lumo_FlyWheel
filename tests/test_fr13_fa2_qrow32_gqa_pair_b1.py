from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
CONTRACT = REPO / "scripts/fr13_fixed32_contract.py"
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
BUILD_RECIPE = REPO / "scripts/fr13_build_fa2_qrow32_gqa_pair_b1_sm121a.sh"

FA2_ORIGIN = Path(
    "/home/mark/shared/lumoFlyWheel-qrow16-thin/output"
    "/fr13_fa2_qrow16_num_splits0_build_20260731/vllm-source/build"
    "/lumo_cutlass_research/_deps/vllm-flash-attn-src"
)
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
GQA_PAIR_B1_SENTINEL = 1179791670
GQA_PAIR_B1_SOURCE_CLOSURE_SHA256 = (
    "172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4"
)
GQA_PAIR_B1_SO_SHA256 = (
    "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
)
GQA_PAIR_B1_SO_SIZE = 299815552
LIVE_GATE = REPO / "scripts/fr13_run_b1_k64_qrow32_split2_live_gate.sh"
CODEGEN_FILES = (
    "csrc/flash_attn/flash_api.cpp",
    "csrc/flash_attn/flash_api_torch_lib.cpp",
    "csrc/flash_attn/src/flash.h",
    "csrc/flash_attn/src/flash_fwd_kernel.h",
    "csrc/flash_attn/src/utils.h",
    "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patcher():
    return _module(PATCHER, "fr13_b1_gqa_pair_patcher")


# --------------------------------------------------------------- sentinel


def test_b1_gqa_pair_sentinel_extends_the_b1_ascii_run_and_stays_distinct() -> None:
    module = _patcher()

    sentinel = module.FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL
    assert sentinel == 0x46523136 == GQA_PAIR_B1_SENTINEL
    # Derived the way every other B1 arm derives its tag: the next four-byte
    # ASCII "FR1x" value after qrow16 (FR13), B1 no-split (FR14) and B1 split2
    # (FR15).
    assert sentinel.to_bytes(4, "big") == b"FR16"
    assert (
        module.FIXED32_QUERY_TILE16_BATCH_STRIDE_SENTINEL.to_bytes(4, "big")
        == b"FR13"
    )
    assert (
        module.FIXED32_QUERY_TILE32_B1_SPLIT2_BATCH_STRIDE_SENTINEL + 1 == sentinel
    )
    # It must not reuse the B4 family's tags.
    assert sentinel != module.FIXED32_QUERY_GQA_PAIR32_BATCH_STRIDE_SENTINEL
    assert sentinel != module.FIXED32_QUERY_TILE32_BATCH_STRIDE_SENTINEL
    sentinels = module._FIXED32_BATCH_STRIDE_SENTINELS
    assert sentinel in sentinels
    # FR17 (the FR14 split-K arm) extends the run; the invariant that matters
    # is pairwise distinctness, since a shared tag would route one arm's
    # traffic into another arm's kernel.
    assert len(set(sentinels)) == len(sentinels) == 7


# -------------------------------------------------------- translation unit


def test_b1_gqa_pair_translation_unit_is_b1_geometry_and_never_splits() -> None:
    module = _patcher()
    unit = module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT

    assert unit.startswith("// FR13 fixed32 B1 qrow32 GQA-pair gate candidate.")
    assert "static constexpr int sequences = 1;" in unit
    assert "static_assert(StaticLayout::sequences == 1);" in unit
    assert "= 12 CTAs/layer" in unit
    # Split-K reassociates the softmax/output reduction, which is exactly why
    # the B1 qrow32 split2 candidate was byte-rejected.
    assert "false,  // Split" in unit
    assert "flash_fwd_splitkv_combine_kernel" not in unit
    assert "params.num_splits" not in unit
    assert "Split2" not in unit
    # Private symbols must not collide with the B4 sibling's.
    assert "fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel" in unit
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(" in unit
    assert '__attribute__((visibility("hidden")))' in unit
    for b4_only in (
        "B4",
        "sequences = 4",
        "Fr13Fixed32Qrow32GqaPairKernelTraits",
    ):
        assert b4_only not in unit


def test_b1_gqa_pair_translation_unit_keeps_every_live_validated_constant() -> None:
    module = _patcher()
    unit = module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    b1_qrow32 = module.FIXED32_QUERY_TILE32_B1_TRANSLATION_UNIT
    b4_pair = module.FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT

    # Paged-KV geometry: identical text to the B1 lineage that live-validates it.
    strides = (
        "    static constexpr int64_t page = 2 * 1024 * 4 * 256;\n"
        "    static constexpr int64_t row = 4 * 256;\n"
        "    static constexpr int64_t head = 256;\n"
    )
    assert strides in unit
    assert strides in b1_qrow32
    assert strides in b4_pair
    block_size = (
        "    static constexpr int value = 1024;\n"
        "    static constexpr int log2 = 10;\n"
        "    static constexpr int block_n_log2 = 6;\n"
    )
    assert block_size in unit
    assert block_size in b1_qrow32

    # Query layout: 32 rows and the 24/4/6 GQA shape, as the B1 lineage pins it.
    for shared_with_b1 in (
        "    static constexpr int query_heads = 24;\n",
        "    static constexpr int kv_heads = 4;\n",
        "    static constexpr int query_heads_per_kv = 6;\n",
    ):
        assert shared_with_b1 in unit
        assert shared_with_b1 in b1_qrow32
    assert "struct StaticQueryRows<Fr13Fixed32Qrow32GqaPairB1KernelTraits> {\n    static constexpr int value = 32;" in unit

    # Pairing traits: identical text to the B4 unit the dual byte gate proved.
    assert "256, 64, 64, 4, false, false, cutlass::bfloat16_t>" in unit
    assert "256, 64, 64, 4, false, false, cutlass::bfloat16_t>" in b4_pair
    assert "static_assert(TreeKernelTraits::kNThreads == 128);" in unit
    assert "static_assert(TreeKernelTraits::kGmemRowsPerThread == 4);" in unit
    assert "static_assert(smem_size == 96 * 1024);" in unit
    assert "__global__ __maxnreg__(254)" in unit
    assert (
        "struct StaticQueryHeadsPerCTA<Fr13Fixed32Qrow32GqaPairB1KernelTraits> {\n"
        "    static constexpr int value = 2;" in unit
    )

    # The two units may differ ONLY by the documented substitutions.
    deltas = [
        (anchor, replacement)
        for anchor, replacement, _ in (
            module._FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT_SUBSTITUTIONS
        )
    ]
    rebuilt = unit
    for anchor, replacement in deltas:
        rebuilt = rebuilt.replace(replacement, anchor)
    assert rebuilt == b4_pair


def test_b1_gqa_pair_translation_unit_derivation_fails_if_the_b4_unit_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _patcher()

    monkeypatch.setattr(
        module,
        "FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT",
        module.FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT.replace(
            "    static constexpr int sequences = 4;",
            "    static constexpr int sequences = 2;",
            1,
        ),
    )
    with pytest.raises(RuntimeError, match="B4 GQA-pair translation unit drifted"):
        module._fixed32_query_gqa_pair32_b1_translation_unit()


# ---------------------------------------------------------------- API gate


def test_b1_gqa_pair_api_gate_is_b1_shaped_and_keeps_the_paired_lse_operands() -> None:
    module = _patcher()
    gate = module.FIXED32_QUERY_GQA_PAIR32_B1_API_GATE
    declaration = module.FIXED32_QUERY_GQA_PAIR32_B1_API_DECLARATION

    assert f"    {GQA_PAIR_B1_SENTINEL};" in declaration
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" in declaration
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" in gate
    assert "&& params.b == 1\n" in gate
    assert "&& params.total_q == 32\n" in gate
    assert "non-canonical B1 geometry" in gate
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(params, stream);" in gate
    for b4_only in ("B4", "params.b == 4", "params.total_q == 128"):
        assert b4_only not in gate

    # The paired output/LSE layout addresses softmax_lse as [head, total_q] and
    # reads the fused paged-KV strides, so these operands are correctness
    # preconditions rather than decoration.
    for required in (
        "&& params.unpadded_lse\n",
        "&& params.is_seqlens_k_cumulative\n",
        "&& params.k_batch_stride == 2 * 1024 * 4 * 256\n",
        "&& params.v_batch_stride == 2 * 1024 * 4 * 256\n",
        "&& params.k_row_stride == 4 * 256\n",
        "&& params.v_row_stride == 4 * 256\n",
        "&& params.page_block_size == 1024\n",
        "&& params.seqlen_q == 32\n",
        "&& params.seqlen_q_rounded == 128\n",
        "&& params.h == 24\n",
        "&& params.h_k == 4\n",
        "&& params.h_h_k_ratio == 6\n",
        "&& params.block_table != nullptr\n",
        "&& params.num_splits == 0\n",
        "&& force_split_kernel,\n",
    ):
        assert gate.count(required) == 1, required

    # It is a gate, not a fallback: no branch returns to stock on mismatch.
    assert "TORCH_CHECK(" in gate
    assert "else" not in gate


def test_b1_gqa_pair_api_gate_operands_match_the_proven_b4_gate() -> None:
    module = _patcher()
    gate = module.FIXED32_QUERY_GQA_PAIR32_B1_API_GATE
    b4_gate = module.FIXED32_QUERY_GQA_PAIR32_API_GATE

    def operands(text: str) -> list[str]:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("&&") or line.strip().startswith("params.")
        ]

    b1_ops = operands(gate)
    b4_ops = operands(b4_gate)
    assert len(b1_ops) == len(b4_ops)
    differing = [
        (left, right)
        for left, right in zip(b4_ops, b1_ops)
        if left != right
    ]
    assert differing == [
        ("&& params.b == 4", "&& params.b == 1"),
        ("&& params.total_q == 128", "&& params.total_q == 32"),
    ]


def test_b1_gqa_pair_api_gate_derivation_fails_if_the_b4_gate_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _patcher()

    monkeypatch.setattr(
        module,
        "FIXED32_QUERY_GQA_PAIR32_API_GATE",
        module.FIXED32_QUERY_GQA_PAIR32_API_GATE.replace(
            "            && params.unpadded_lse\n", "", 1
        ),
    )
    with pytest.raises(RuntimeError, match="lost a required operand"):
        module._fixed32_query_gqa_pair32_b1_api_gate()


# --------------------------------------------------------------- geometry


def test_b1_gqa_pair_grid_covers_each_b1_query_head_exactly_once() -> None:
    # dim3(query_heads_per_kv / 2, sequences, kv_heads) = (3, 1, 4).
    observed: list[tuple[int, int]] = []
    kv_stagings: dict[tuple[int, int], int] = {}
    ctas = 0

    for kv_head in range(4):
        for batch in range(1):
            for pair_lane in range(3):
                ctas += 1
                head_base = kv_head * 6 + pair_lane * 2
                observed.extend((batch, head_base + in_pair) for in_pair in range(2))
                kv_stagings[(batch, kv_head)] = (
                    kv_stagings.get((batch, kv_head), 0) + 1
                )

    assert ctas == 12
    assert sorted(observed) == [(0, head) for head in range(24)]
    assert len(observed) == len(set(observed)) == 24
    # Each KV head's tiles are staged 3 times instead of the incumbent's 6 --
    # one CTA per head pair rather than one per query head.
    assert set(kv_stagings.values()) == {3}
    assert sum(kv_stagings.values()) == 12


# ------------------------------------------------------- arm registration


def test_b1_gqa_pair_arm_registers_in_the_b1_live_ab_registry() -> None:
    module = _patcher()
    namespace: dict[str, object] = {"os": os}
    exec(module.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS, namespace)

    arms = namespace["_FR13_FA2_QROW32_B1_ARMS"]
    assert "gqa_pair" in arms
    assert arms["gqa_pair"] == {
        "sentinel": GQA_PAIR_B1_SENTINEL,
        "num_splits": 0,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": "qrow32 B1 GQA-pair exact geometry; no fallback",
    }
    # Reduction topology must match the qrow16 reference for the byte gate to
    # mean anything.
    assert arms["gqa_pair"]["num_splits"] == arms["nosplit"]["num_splits"] == 0
    # Distinct dispatch tag from every other registered B1 arm.
    tags = [config["sentinel"] for config in arms.values()]
    assert tags.count(GQA_PAIR_B1_SENTINEL) == 1


def test_b1_gqa_pair_arm_is_selectable_and_pinned_to_the_built_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _patcher()
    namespace: dict[str, object] = {"os": os}
    exec(module.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS, namespace)

    monkeypatch.setenv("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "gqa_pair")
    assert (
        namespace["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_LIVE_AB_ARM")
        == "gqa_pair"
    )

    identity = namespace["_fr13_fa2_qrow32_b1_identity"]("gqa_pair")
    assert identity["candidate_sha256"] == GQA_PAIR_B1_SO_SHA256
    assert identity["candidate_size"] == GQA_PAIR_B1_SO_SIZE
    assert identity["source_closure_sha256"] == GQA_PAIR_B1_SOURCE_CLOSURE_SHA256
    assert identity["fa2_head"] == FA2_HEAD

    # A distinct binary from every other B1 arm.
    for arm in ("nosplit", "split2", "visibility"):
        other = namespace["_fr13_fa2_qrow32_b1_identity"](arm)
        assert other["candidate_sha256"] != identity["candidate_sha256"]
        assert other["candidate_size"] != identity["candidate_size"]


def test_b1_gqa_pair_is_a_production_arm_but_the_instruments_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gqa_pair is now servable; split2 and visibility must never be.

    This test previously asserted gqa_pair was refused here. It is admitted as
    of the production selector work: the arm is byte-qualified as a SERVED
    dispatch. split2 and visibility are not -- one varies reduction topology,
    the other only observes -- so they stay refused even though both are
    registered arms.
    """
    module = _patcher()
    namespace: dict[str, object] = {"os": os}
    exec(module.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS, namespace)

    monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "gqa_pair")
    assert (
        namespace["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_PRODUCTION_ARM")
        == "gqa_pair"
    )

    for refused in ("split2", "visibility"):
        monkeypatch.setenv("FR13_FA2_QROW32_B1_PRODUCTION_ARM", refused)
        with pytest.raises(RuntimeError, match="must be empty or one of"):
            namespace["_fr13_fa2_qrow32_b1_arm"](
                "FR13_FA2_QROW32_B1_PRODUCTION_ARM"
            )


# ------------------------------------------------------------- gate defs


def test_b1_gqa_pair_gate_defs_pin_the_source_and_refuse_the_unbuilt_binary() -> None:
    sidecar = _module(SIDECAR, "fr13_b1_gqa_pair_sidecar")

    assert sidecar.GQA_PAIR_ARM == "gqa_pair"
    assert sidecar.GQA_PAIR_SELECTOR_SENTINEL == GQA_PAIR_B1_SENTINEL
    assert sidecar.LIVE_ARMS["gqa_pair"]["num_splits"] == 0
    assert (
        sidecar.LIVE_ARMS["gqa_pair"]["candidate_dispatch"]
        == "qrow32 B1 GQA-pair exact geometry; no fallback"
    )
    assert (
        sidecar.GQA_PAIR_SOURCE_CLOSURE_SHA256 == GQA_PAIR_B1_SOURCE_CLOSURE_SHA256
    )
    assert sidecar.GQA_PAIR_CANDIDATE_SHA256 == GQA_PAIR_B1_SO_SHA256
    assert sidecar.GQA_PAIR_CANDIDATE_SIZE == GQA_PAIR_B1_SO_SIZE

    # The GQA-pair codegen emits ONE translation unit, so reusing the no-split
    # arm's modified-source set would make source validation unsatisfiable.
    assert sidecar.GQA_PAIR_SOURCE_STATUS != sidecar.SOURCE_STATUS
    assert sidecar.GQA_PAIR_SOURCE_STATUS[-1] == (
        "?? csrc/flash_attn/src/"
        "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu"
    )
    assert len(sidecar.GQA_PAIR_SOURCE_STATUS) == 6
    assert sorted(sidecar.GQA_PAIR_SOURCE_FILES) == sorted(
        line[3:] for line in sidecar.GQA_PAIR_SOURCE_STATUS
    )

    # Source qualification must not depend on the binary identity.
    contract = sidecar.gqa_pair_source_contract()
    assert contract["source_closure_sha256"] == GQA_PAIR_B1_SOURCE_CLOSURE_SHA256
    candidate = sidecar._candidate_contract("gqa_pair")
    assert candidate["sha256"] == GQA_PAIR_B1_SO_SHA256
    assert candidate["size"] == GQA_PAIR_B1_SO_SIZE
    # A live result claiming any other binary is still refused.
    with pytest.raises(ValueError, match="live candidate is not pinned"):
        sidecar.validate_live_result({}, candidate_sha256="0" * 64, arm="gqa_pair")


def test_b1_gqa_pair_gate_defs_reuse_the_proven_headers() -> None:
    sidecar = _module(SIDECAR, "fr13_b1_gqa_pair_sidecar_headers")

    # flash_fwd_kernel.h carries the paired address layout and is byte-identical
    # to the header in the B4 GQA-pair closure that passed the dual byte gate.
    assert (
        sidecar.GQA_PAIR_SOURCE_FILES["csrc/flash_attn/src/flash_fwd_kernel.h"]
        == "4f08741030c46d7e1ef1b88a10d4946f625559fedd7658c3b288e0d7a5d58d13"
    )
    # These three are byte-identical to the qualified qrow32 B1 closure.
    for shared in (
        "csrc/flash_attn/flash_api_torch_lib.cpp",
        "csrc/flash_attn/src/flash.h",
        "csrc/flash_attn/src/utils.h",
    ):
        assert sidecar.GQA_PAIR_SOURCE_FILES[shared] == sidecar.SOURCE_FILES[shared]
    # Only the dispatch shim and the new TU are genuinely new.
    assert (
        sidecar.GQA_PAIR_SOURCE_FILES["csrc/flash_attn/flash_api.cpp"]
        != sidecar.SOURCE_FILES["csrc/flash_attn/flash_api.cpp"]
    )


def test_b1_gqa_pair_is_admitted_by_the_runtime_contract_and_launcher() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import fr13_fixed32_contract as contract

    assert contract.QROW32_B1_GQA_PAIR_FA2_SHA256 == GQA_PAIR_B1_SO_SHA256
    assert contract.QROW32_B1_GQA_PAIR_FA2_SIZE == GQA_PAIR_B1_SO_SIZE

    env = {
        "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW16_PRODUCTION": "0",
        "FR13_FA2_QROW32_LIVE_PAGED_AB": "0",
        "FR13_FA2_QROW32_B1_LIVE_AB_ARM": "gqa_pair",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM": "",
        "FR13_FA2_QROW32_B1_SO_SHA256": GQA_PAIR_B1_SO_SHA256,
    }
    # The runtime contract and the gate must agree on the binary, exactly.
    assert contract._expected_runtime_fa2_identity(env) == (
        GQA_PAIR_B1_SO_SIZE,
        GQA_PAIR_B1_SO_SHA256,
    )
    env["FR13_FA2_QROW32_B1_SO_SHA256"] = "0" * 64
    with pytest.raises(contract.ContractError, match="not the pinned candidate"):
        contract._expected_runtime_fa2_identity(env)

    env["FR13_FA2_QROW32_B1_SO_SHA256"] = GQA_PAIR_B1_SO_SHA256
    env["FR13_FA2_QROW32_B1_LIVE_AB_ARM"] = "gqa_pair_typo"
    with pytest.raises(contract.ContractError, match="must be empty, nosplit"):
        contract._expected_runtime_fa2_identity(env)

    launcher = LAUNCHER.read_text()
    # The LIVE allowlist gained the FR14 tier-b arm (pass 50 in the launcher,
    # FR14 lane 4 in the contract). What this test guards is that the arms it
    # already admitted are still admitted and that gqa_pair still resolves to
    # its own binary -- not that the list stopped growing.
    assert '""|nosplit|split2|visibility|gqa_pair' in launcher
    assert "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty" in launcher
    assert "nosplit, split2" in launcher
    assert "visibility" in launcher
    # Production admits the byte-qualified GQA-pair arm alongside no-split,
    # and each arm is pinned to its own binary via the resolved pin arm.
    assert '""|nosplit|gqa_pair) ;;' in launcher
    assert 'case "$_FR13_FA2_QROW32_B1_PIN_ARM" in' in launcher


# ---------------------------------------------------------- build recipe


def test_b1_gqa_pair_build_recipe_is_pinned_cpu_only_and_reuses_reference_objects() -> None:
    recipe = BUILD_RECIPE.read_text()

    assert os.access(BUILD_RECIPE, os.X_OK)
    assert (
        "sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
        in recipe
    )
    assert f"FA2_HEAD={FA2_HEAD}" in recipe
    assert f"SOURCE_CLOSURE_SHA256={GQA_PAIR_B1_SOURCE_CLOSURE_SHA256}" in recipe
    # Explicit arming, and it refuses to run without it.
    assert 'case "${FR13_BUILD_B1_GQA_PAIR:-0}" in' in recipe
    # Hermetic and GPU-free.
    assert "--network none" in recipe
    assert "NVIDIA_VISIBLE_DEVICES=void" in recipe
    assert "CUDA_VISIBLE_DEVICES=" in recipe
    assert "CUDA_CACHE_DISABLE=1" in recipe
    assert "unexpected_gpu_device" in recipe
    assert '"$(docker ps -aq | wc -l)" -eq 0' in recipe
    # Byte-for-byte reference object reuse keeps ABI diffs attributable.
    assert "! -path '*/flash_api.cpp.o'" in recipe
    assert 'test \\"\\${#objects[@]}\\" -eq 55' in recipe
    # Attestations the gate later re-derives.
    assert "validate-source \\\n  --source-root" in recipe
    assert "--arm gqa_pair" in recipe
    assert "forbidden_sass.txt" in recipe
    assert "private_launcher_leaked" in recipe
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1" in recipe
    # Register count is recorded, never asserted: it has never been measured for
    # this translation unit. It may only appear in prose explaining that.
    executable = "\n".join(
        line for line in recipe.splitlines() if not line.lstrip().startswith("#")
    )
    assert "REG:243" not in executable
    assert "Register count is RECORDED, not asserted" in recipe


def test_b1_gqa_pair_audit_allowance_is_narrow_and_additive_only() -> None:
    recipe = BUILD_RECIPE.read_text()

    # The three diffs that establish drop-in replacement admit no allowance.
    assert "for f in defined_dynamic dt_needed runtime_path; do" in recipe
    assert 'test "$b" -eq 0 || { echo "ABI DRIFT in $f" >&2; exit 94; }' in recipe
    # undefined_dynamic: additive only, versioned libstdc++ only.
    assert "imports removed:" in recipe
    assert '[[ "$symbol" == *@GLIBCXX_* ]]' in recipe
    assert "not a versioned libstdc++ import" in recipe
    # DT_NEEDED is captured from `readelf -W -d`, whose fifth field is the
    # BRACKETED soname. The original whole-line match on the unbracketed name
    # could never succeed, which made the allowance dead on arrival; bdd2c18e5
    # fixed the recipe and this assertion was left describing the defect.
    assert "grep -qxF '[libstdc++.so.6]' \"$BUILD/candidate_dt_needed.txt\"" in recipe
    # Resolvability is proven by the mandatory load, not asserted textually.
    assert "MANDATORY" in recipe
    assert "set -o pipefail" in recipe
    assert "offline torch load did not qualify the candidate" in recipe


def test_b1_gqa_pair_live_gate_runner_carries_the_pinned_arm() -> None:
    runner = LIVE_GATE.read_text()

    assert "  gqa_pair)" in runner
    assert f"CANDIDATE_SHA256={GQA_PAIR_B1_SO_SHA256}" in runner
    assert f"CANDIDATE_BYTES={GQA_PAIR_B1_SO_SIZE}" in runner
    assert f"SOURCE_CLOSURE_SHA256={GQA_PAIR_B1_SOURCE_CLOSURE_SHA256}" in runner
    assert (
        "FR13_QROW32_B1_LIVE_ARM must be split2, visibility, or gqa_pair" in runner
    )
    # The runner's pins must equal the gate's own contract, or the gate would
    # verify one binary while the server loaded another.
    sidecar = _module(SIDECAR, "fr13_b1_gqa_pair_sidecar_runner")
    contract = sidecar._candidate_contract("gqa_pair")
    block = runner.split("  gqa_pair)")[1].split(";;")[0]
    assert contract["sha256"] in block
    assert str(contract["size"]) in block
    assert contract["source_closure_sha256"] in block
    # Every artifact the runner verifies is arm-parameterised, so the GQA-pair
    # arm demands nothing the split2/visibility arms do not already produce.
    assert (
        'LIVE_RESULT="$ARMDIR/logs/fr13_fa2_qrow32_b1_${LIVE_ARM}'
        '_live_paged_ab.json"' in runner
    )
    for shared in (
        'DIAGNOSTIC="$ARMDIR/fixed32_b1_diagnostic.json"',
        'HEALTH="$ARMDIR/health.json"',
        'TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"',
    ):
        assert shared in runner


@pytest.mark.parametrize("armed", ["0", "bogus"])
def test_b1_gqa_pair_build_recipe_refuses_unless_armed(armed: str) -> None:
    result = subprocess.run(
        ["bash", str(BUILD_RECIPE)],
        cwd=REPO,
        env={**os.environ, "FR13_BUILD_B1_GQA_PAIR": armed},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "FR13_BUILD_B1_GQA_PAIR" in result.stderr


# ------------------------------------------------------------- codegen


@pytest.mark.skipif(
    not FA2_ORIGIN.is_dir(), reason="pinned FA2 source checkout is unavailable"
)
def test_b1_gqa_pair_codegen_reproduces_the_pinned_source_closure(
    tmp_path: Path,
) -> None:
    module = _patcher()
    sidecar = _module(SIDECAR, "fr13_b1_gqa_pair_sidecar_codegen")

    # Build a minimal pristine tree from the pinned checkout rather than copying
    # 487 MiB: patch_fa2_source only touches these six files.
    root = tmp_path / "fa2"
    subprocess.run(
        ["git", "init", "-q", str(root)], check=True, capture_output=True
    )
    for relative in CODEGEN_FILES:
        blob = subprocess.run(
            ["git", "-C", str(FA2_ORIGIN), "show", f"{FA2_HEAD}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    changed = module.patch_fa2_source(
        root,
        tree_bias_tile_earlyout=True,
        fixed32_query_gqa_pair32_b1=True,
    )
    assert changed["flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu"] is True
    assert changed["flash_api.cpp"] is True
    assert changed["flash_fwd_kernel.h"] is True
    # The B4 sibling and the qrow32 B1 units must NOT be emitted.
    assert changed["flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"] is False
    assert changed["flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu"] is False

    written = root / "csrc/flash_attn/src" / (
        "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu"
    )
    assert (
        written.read_text() == module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    )

    api = (root / "csrc/flash_attn/flash_api.cpp").read_text()
    # The candidate gate AND the qrow16 reference dispatch the byte A/B compares
    # against must both resolve in the same binary.
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" in api
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(params, stream);" in api
    assert "kFr13Qrow16BatchStrideSentinel" in api
    assert "fr13_run_mha_fwd_fixed32_qrow16(params, stream);" in api
    # No split-K scratch patch: this arm keeps num_splits=0.
    assert "fr13_qrow32_b1_split2" not in api
    # The stock dispatch body survives.
    assert "FP16_SWITCH" in api

    header = (root / "csrc/flash_attn/src/flash_fwd_kernel.h").read_text()
    assert header.count(
        "// FR13_FA2_QROW32_GQA_PAIR_LAYOUT: two heads share the M tile."
    ) == 3

    # Per-file hashes and the closure digest must equal what the gate pins.
    expected = sidecar.GQA_PAIR_SOURCE_FILES
    for relative, digest in expected.items():
        assert sidecar.sha256_file(root / relative) == digest, relative
    closure = {"fa2_head": FA2_HEAD, "files": dict(expected)}
    assert (
        sidecar._digest(sidecar.canonical_bytes(closure))
        == GQA_PAIR_B1_SOURCE_CLOSURE_SHA256
    )

    # Idempotent: a second pass writes nothing.
    again = module.patch_fa2_source(
        root,
        tree_bias_tile_earlyout=True,
        fixed32_query_gqa_pair32_b1=True,
    )
    assert not any(again.values())


def test_b1_gqa_pair_source_build_is_exclusive_and_needs_the_tile_earlyout() -> None:
    module = _patcher()

    with pytest.raises(ValueError, match="mutually exclusive"):
        module.patch_fa2_source(
            Path("/nonexistent"),
            tree_bias_tile_earlyout=True,
            fixed32_query_gqa_pair32=True,
            fixed32_query_gqa_pair32_b1=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.patch_fa2_source(
            Path("/nonexistent"),
            tree_bias_tile_earlyout=True,
            fixed32_query_tile32_b1=True,
            fixed32_query_gqa_pair32_b1=True,
        )
    with pytest.raises(ValueError, match="requires --tree-bias-tile-earlyout"):
        module.patch_fa2_source(
            Path("/nonexistent"),
            fixed32_query_gqa_pair32_b1=True,
        )


def test_b1_gqa_pair_codegen_flag_is_wired_into_the_patcher_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(PATCHER), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--fixed32-query-gqa-pair32-b1" in result.stdout

    # The B1 production selector must refuse to ride on the GQA-pair source.
    result = subprocess.run(
        [
            sys.executable,
            str(PATCHER),
            "--skip-python",
            "--tree-bias-tile-earlyout",
            "--fixed32-query-gqa-pair32-b1",
            "--fixed32-query-tile32-b1-production",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires --fixed32-query-tile32-b1" in result.stderr
