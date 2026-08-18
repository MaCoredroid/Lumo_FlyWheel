"""FR14 split-K FA2 (Tier-B) codegen contract.

The load-bearing test in this file is the LAST one, and it is not about the new
arm at all: it re-derives the PROMOTED gqa_pair source closure with the split-K
flag off and asserts it is still 172b5e71... byte for byte. Everything the
split-K arm adds -- a translation unit, an API gate, a split-scratch
allocation, the paired O/LSE tensors learning to address the stock split
accumulators, and the combine kernel's static-geometry specialization -- is
gated on that one flag, so the arm that actually serves cannot move because
this one was built.

The rest asserts the properties that keep a Tier-B arm from being mistaken for
a qualified one: its own sentinel, its own closure, its own binary pins, and an
absence from the production arm set that no single edit can undo.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SIDECAR = REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py"
BUILD_RECIPE = REPO / "scripts/fr14_build_fa2_qrow32_gqa_pair_splitk_b1_sm121a.sh"
PROMOTED_RECIPE = REPO / "scripts/fr13_build_fa2_qrow32_gqa_pair_b1_sm121a.sh"

FA2_ORIGIN = Path(
    "/home/mark/shared/lumoFlyWheel-qrow16-thin/output"
    "/fr13_fa2_qrow16_num_splits0_build_20260731/vllm-source/build"
    "/lumo_cutlass_research/_deps/vllm-flash-attn-src"
)
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SPLITK_SENTINEL = 1179791671
SPLITK_CONTEXT_SPLITS = 4
GQA_PAIR_B1_SOURCE_CLOSURE_SHA256 = (
    "172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4"
)
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
    return _module(PATCHER, "fr14_splitk_patcher")


def _pristine_tree(root: Path) -> None:
    """A minimal pristine FA2 tree: patch_fa2_source touches only six files."""
    subprocess.run(["git", "init", "-q", str(root)], check=True,
                   capture_output=True)
    for relative in CODEGEN_FILES:
        blob = subprocess.run(
            ["git", "-C", str(FA2_ORIGIN), "show", f"{FA2_HEAD}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)


# ---------------------------------------------------------------- sentinel


def test_splitk_sentinel_extends_the_b1_ascii_run_and_stays_distinct() -> None:
    module = _patcher()

    sentinel = module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_BATCH_STRIDE_SENTINEL
    assert sentinel == 0x46523137 == SPLITK_SENTINEL
    assert sentinel.to_bytes(4, "big") == b"FR17"
    assert (
        module.FIXED32_QUERY_GQA_PAIR32_B1_BATCH_STRIDE_SENTINEL + 1 == sentinel
    )
    sentinels = module._FIXED32_BATCH_STRIDE_SENTINELS
    assert sentinel in sentinels
    assert len(set(sentinels)) == len(sentinels) == 7
    assert module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_CONTEXT_SPLITS == (
        SPLITK_CONTEXT_SPLITS
    )


# -------------------------------------------------------- translation unit


def test_splitk_translation_unit_is_the_promoted_unit_plus_split_k() -> None:
    module = _patcher()

    promoted = module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    unit = module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT

    # Same traits as the promoted arm. If any of these moved, the arm would no
    # longer be "gqa_pair with a split context walk" and the whole comparison
    # this lane rests on would be measuring two changes at once.
    for shared in (
        "Flash_fwd_kernel_traits<\n    256, 64, 64, 4, false, false, "
        "cutlass::bfloat16_t>;",
        "static_assert(smem_size == 96 * 1024);",
        "constexpr static int kHeadsPerCTA = 2;",
        "static_assert(TreeKernelTraits::kNThreads == 128);",
        "static_assert(TreeKernelTraits::kGmemRowsPerThread == 4);",
        "__maxnreg__(254)",
    ):
        assert shared in promoted, shared
        assert shared in unit, shared

    # What split-K adds.
    assert "true,   // Split: blockIdx.y partitions the context walk" in unit
    assert f"constexpr static int kContextSplits = {SPLITK_CONTEXT_SPLITS};" in unit
    assert "flash_fwd_splitkv_combine_kernel<" in unit
    assert "constexpr int kLogMaxSplits = 2;" in unit
    assert "static_assert((1 << kLogMaxSplits) == kContextSplits);" in unit
    # The grid's middle dimension is the split count, not the sequence count.
    assert "        kContextSplits,\n        StaticLayout::kv_heads);" in unit
    # It refuses to launch without the sentinel, the split count AND the two
    # stock accumulators -- a sentinel alone would let a num_splits=0 caller
    # reach a kernel that writes partial results into a null pointer.
    assert f"params.tree_bias_batch_stride == {SPLITK_SENTINEL}" in unit
    assert "&& params.num_splits == kContextSplits" in unit
    assert "&& params.oaccum_ptr != nullptr" in unit
    assert "&& params.softmax_lseaccum_ptr != nullptr" in unit

    # None of the promoted arm's private symbols survive: they would collide at
    # link time, and its sentinel would route byte-gated traffic here.
    for forbidden in (
        "Fr13Fixed32Qrow32GqaPairB1KernelTraits",
        "fr13_flash_fwd_fixed32_qrow32_gqa_pair_b1_kernel",
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_b1(",
        "false,  // Split",
    ):
        assert forbidden not in unit, forbidden
    assert "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_splitk_b1(" in unit


def test_splitk_translation_unit_derivation_is_anchored_and_counted() -> None:
    """A drift in the promoted unit must fail HERE, not fork the two kernels."""
    module = _patcher()

    subs = module._FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT_SUBSTITUTIONS
    promoted = module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    seen = promoted
    for anchor, replacement, expected in subs:
        assert seen.count(anchor) == expected, anchor[:60]
        seen = seen.replace(anchor, replacement)
    assert seen == module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT


# ------------------------------------------------------------- API gate


def test_splitk_api_gate_is_the_promoted_gate_plus_the_split_operands() -> None:
    module = _patcher()

    gate = module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_API_GATE
    promoted = module.FIXED32_QUERY_GQA_PAIR32_B1_API_GATE

    # Everything the promoted gate asserted about geometry is inherited.
    for inherited in (
        "&& params.b == 1\n",
        "&& params.total_q == 32\n",
        "&& params.seqlen_q == 32\n",
        "&& params.h == 24\n",
        "&& params.h_k == 4\n",
        "&& params.unpadded_lse\n",
        "&& params.is_seqlens_k_cumulative\n",
        "&& params.page_block_size == 1024\n",
        "&& params.k_batch_stride == 2 * 1024 * 4 * 256\n",
    ):
        assert inherited in promoted, inherited
        assert inherited in gate, inherited

    assert f"&& params.num_splits == {SPLITK_CONTEXT_SPLITS}\n" in gate
    assert "&& params.oaccum_ptr != nullptr\n" in gate
    assert "&& params.softmax_lseaccum_ptr != nullptr\n" in gate
    assert "&& params.num_splits == 0" not in gate
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" not in gate
    assert "kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel" in gate


def test_splitk_scratch_allocation_pins_the_split_count() -> None:
    module = _patcher()

    alloc = module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_ALLOCATION
    assert "kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel" in alloc
    assert f"|| num_splits == {SPLITK_CONTEXT_SPLITS}" in alloc
    # Stock only allocates split scratch for the ngroups-swapped decode path;
    # the private route has to opt in explicitly or the accumulators are null.
    assert "seqlenq_ngroups_swapped || fr14_qrow32_gqa_pair_splitk_b1" in alloc


# ------------------------------------------------------------ arm registry


def _selectors():
    """The B1 selector helpers as they are injected into the served vLLM.

    They are emitted as source, not imported, so the test evaluates the exact
    text that ships rather than a module-level copy that could drift from it.
    """
    module = _patcher()
    source = module.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS
    namespace: dict = {}
    exec(compile(source, "<b1_selectors>", "exec"), namespace)  # noqa: S102
    return namespace


def test_splitk_arm_is_registered_and_cannot_serve() -> None:
    namespace = _selectors()

    arms = namespace["_FR13_FA2_QROW32_B1_ARMS"]
    assert "gqa_pair_splitk" in arms
    arm = arms["gqa_pair_splitk"]
    assert arm["sentinel"] == SPLITK_SENTINEL
    assert arm["num_splits"] == SPLITK_CONTEXT_SPLITS
    # THE line that keeps a Tier-B arm out of production traffic.
    production = namespace["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"]
    assert "gqa_pair_splitk" not in production
    assert production == ("nosplit", "gqa_pair")


def test_splitk_cannot_be_byte_gated_against_the_reference() -> None:
    """The refusal is structural, not a convention this lane agreed to.

    _fr13_fa2_qrow32_b1_require_same_reduction exists so a raw-byte gate can
    only ever compare arms of identical reduction topology. Split-K's topology
    differs from the qrow16 reference's by construction, so the gate refuses
    before it can produce a green credential for a Tier-B arm.
    """
    require = _selectors()["_fr13_fa2_qrow32_b1_require_same_reduction"]

    # The reference arm and the promoted arm agree, so the gate proceeds.
    require("gqa_pair", 0)
    with pytest.raises(RuntimeError, match="identical reduction topology"):
        require("gqa_pair_splitk", 0)


def test_splitk_interface_route_requires_the_sentinel_AND_the_split_count() -> None:
    """A sentinel alone is not enough: the split count selects the kernel's
    accumulator layout, so an operand tagged for split-K with the wrong count
    would read partials that were never written."""
    module = _patcher()
    helper = module.FR13_FA2_QROW32_B1_SPLIT2_INTERFACE_HELPER

    assert "_FR13_FA2_QROW32_B1_SPLIT_ROUTES = ((2, 1179791669), (4, 1179791671))" in helper
    namespace: dict = {}
    exec("import torch\n" + helper, namespace)  # noqa: S102 - the emitted source
    allowed = namespace["_fr13_fa2_qrow32_b1_split2_interface_allowed"]

    import torch

    bias = torch.zeros(32, 32, dtype=torch.float32)

    class _FakeCudaBias:
        """The guard is a shape/stride/dtype contract; CUDA is not needed."""

        def __init__(self, stride0: int) -> None:
            self._stride = (stride0, 32, 1)
            self.is_cuda = True
            self.dtype = torch.float32
            self.shape = (1, 32, 32)

        def stride(self):
            return self._stride

    assert allowed(4, _FakeCudaBias(SPLITK_SENTINEL))
    assert allowed(2, _FakeCudaBias(1179791669))
    # Right sentinel, wrong split count.
    assert not allowed(2, _FakeCudaBias(SPLITK_SENTINEL))
    assert not allowed(8, _FakeCudaBias(SPLITK_SENTINEL))
    # Right count, unregistered sentinel.
    assert not allowed(4, _FakeCudaBias(1179791670))
    assert not allowed(4, None)
    del bias


# ------------------------------------------------------------ build recipe


def test_splitk_builder_carries_both_credentials() -> None:
    recipe = BUILD_RECIPE.read_text()
    sidecar = _module(SIDECAR, "fr14_splitk_sidecar_recipe")

    # Its own env gate: the promoted builder's must not start this one.
    assert 'case "${FR14_BUILD_B1_GQA_PAIR_SPLITK:-0}" in' in recipe
    assert "FR13_BUILD_B1_GQA_PAIR" not in recipe

    # Reproducibility credential: pinned, asserted BEFORE the link, and the
    # bootstrap escape can only refuse to link -- never produce an artifact.
    assert "SASS_DIGEST_SHA256=__SASS_PIN__" not in recipe, "SASS pin is unset"
    assert 'if [[ "$SASS_DIGEST_SHA256" != "$BUILT_SASS_DIGEST" ]]' in recipe or (
        'if [[ "$BUILT_SASS_DIGEST" != "$SASS_DIGEST_SHA256" ]]' in recipe
    )
    assert "REBUILD DID NOT REPRODUCE THE PINNED KERNEL" in recipe
    assert 'FR14_SPLITK_SASS_BOOTSTRAP:-0' in recipe
    assert "BOOTSTRAP: no link performed" in recipe
    bootstrap_at = recipe.index("BOOTSTRAP: no link performed")
    link_at = recipe.index("echo \"== Link:")
    assert bootstrap_at < link_at, "bootstrap must stop before the link"

    # Staged-artifact credential: size is a hard fail, sha refuses by default.
    assert "CANDIDATE_SO_SHA256=" in recipe
    assert "CANDIDATE_SO_SIZE=" in recipe
    assert "staged .so SIZE differs from the pin" in recipe
    assert "staged .so sha256 differs from the pin." in recipe
    assert "FR14_SPLITK_ALLOW_SO_REPIN" in recipe

    # The source closure it builds is the one the sidecar pins for this arm.
    assert f"SOURCE_CLOSURE_SHA256={sidecar.SPLITK_SOURCE_CLOSURE_SHA256}" in recipe
    assert "--arm gqa_pair_splitk" in recipe
    assert "--fixed32-query-gqa-pair32-splitk-b1" in recipe

    # The baseline credential: the reference arm in this binary must be the
    # SEALED gqa_pair kernel, asserted before the link like the arm's own.
    assert (
        "REF_SASS_DIGEST_SHA256=fa01f98840420b9c0177d06297aacabb0ed5e00c674511f"
        "daa4aa618c3473470" in recipe
    )
    assert "THE BASELINE ARM IN THIS BINARY IS NOT THE SEALED KERNEL" in recipe
    assert "grep -q 'REG:243 STACK:0 SHARED:1024 LOCAL:0'" in recipe
    baseline_at = recipe.index("BASELINE_SASS_MATCHES_SEALED_KERNEL")
    assert baseline_at < recipe.index('echo "== Link:')

    # This TU emits two kernels; a single grep -q would pass one clean kernel
    # beside one that spilled, so the contract counts them.
    assert "-eq 2 ]]" in recipe
    assert "expected exactly two kernels (attention + combine)" in recipe
    assert "a kernel in this TU uses stack" in recipe
    assert "a kernel in this TU uses local memory" in recipe

    # The audits that make the candidate a drop-in replacement are inherited
    # verbatim from the promoted recipe, not re-implemented.
    promoted = PROMOTED_RECIPE.read_text()
    for inherited in (
        "for f in defined_dynamic dt_needed runtime_path; do",
        'test "$b" -eq 0 || { echo "ABI DRIFT in $f" >&2; exit 94; }',
        "private_launcher_leaked",
        "offline torch load did not qualify the candidate",
        "reference_objects_without_flash_api",
        '-eq 55',
    ):
        assert inherited in promoted, inherited
        assert inherited in recipe, inherited


def test_splitk_sidecar_pins_its_own_closure_and_status() -> None:
    sidecar = _module(SIDECAR, "fr14_splitk_sidecar_pins")

    assert sidecar.SPLITK_ARM == "gqa_pair_splitk"
    assert sidecar.SPLITK_SELECTOR_SENTINEL == SPLITK_SENTINEL
    assert sidecar.SPLITK_NUM_SPLITS == SPLITK_CONTEXT_SPLITS
    assert sidecar.LIVE_ARMS[sidecar.SPLITK_ARM]["num_splits"] == (
        SPLITK_CONTEXT_SPLITS
    )
    # Its emitted source set is its own -- one TU, not the promoted arm's.
    status = sidecar._source_status(sidecar.SPLITK_ARM)
    assert any("gqa_pair_splitk_b1_hdim256" in line for line in status)
    # The promoted unit is in this closure too -- the baseline is compiled into
    # the same binary -- but the promoted arm's own closure must NOT have
    # acquired the split-K unit in return.
    assert any("gqa_pair_b1_hdim256" in line for line in status)
    assert sidecar._source_status("gqa_pair") == sidecar.GQA_PAIR_SOURCE_STATUS
    assert not any(
        "splitk" in line for line in sidecar.GQA_PAIR_SOURCE_STATUS
    )
    assert not any("splitk" in f for f in sidecar.GQA_PAIR_SOURCE_FILES)
    # The shared unit is byte-identical in both closures.
    shared = ("csrc/flash_attn/src/"
              "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu")
    assert (sidecar.SPLITK_SOURCE_FILES[shared]
            == sidecar.GQA_PAIR_SOURCE_FILES[shared])
    # The source closure is pinnable before the binary exists, exactly as the
    # promoted arm's was; the binary pins fail closed until they are filled.
    contract = sidecar.splitk_source_contract()
    assert contract["source_closure_sha256"] == sidecar.SPLITK_SOURCE_CLOSURE_SHA256
    if not sidecar.SPLITK_CANDIDATE_SHA256:
        with pytest.raises(ValueError, match="is not pinned"):
            sidecar._candidate_contract(sidecar.SPLITK_ARM)
    else:
        assert len(sidecar.SPLITK_CANDIDATE_SHA256) == 64
        assert sidecar.SPLITK_CANDIDATE_SIZE > 0


# ---------------------------------------------------------------- codegen


@pytest.mark.skipif(
    not FA2_ORIGIN.is_dir(), reason="pinned FA2 source checkout is unavailable"
)
def test_splitk_codegen_does_not_move_the_promoted_closure(tmp_path: Path) -> None:
    """THE regression that matters: the arm that serves must not move.

    Every split-K patch is gated on one flag. With it off, the promoted
    gqa_pair closure must still be byte-for-byte what the sealed 2026-08-10
    build produced.
    """
    module = _patcher()
    sidecar = _module(SIDECAR, "fr14_splitk_sidecar_regression")

    root = tmp_path / "fa2_promoted"
    _pristine_tree(root)
    module.patch_fa2_source(
        root, tree_bias_tile_earlyout=True, fixed32_query_gqa_pair32_b1=True
    )
    for relative, digest in sidecar.GQA_PAIR_SOURCE_FILES.items():
        assert sidecar.sha256_file(root / relative) == digest, relative
    closure = {"fa2_head": FA2_HEAD, "files": dict(sidecar.GQA_PAIR_SOURCE_FILES)}
    assert (
        sidecar._digest(sidecar.canonical_bytes(closure))
        == GQA_PAIR_B1_SOURCE_CLOSURE_SHA256
        == sidecar.GQA_PAIR_SOURCE_CLOSURE_SHA256
    )
    header = (root / "csrc/flash_attn/src/flash_fwd_kernel.h").read_text()
    # The promoted arm keeps the !Split assertions and gains no combine
    # specialization: the split-K patches simply did not run.
    assert header.count("static_assert(!Split);") == 4
    assert "FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT" not in header
    assert "FR14_FA2_COMBINE_STATIC_GEOMETRY" not in header


@pytest.mark.skipif(
    not FA2_ORIGIN.is_dir(), reason="pinned FA2 source checkout is unavailable"
)
def test_splitk_codegen_reproduces_its_pinned_source_closure(tmp_path: Path) -> None:
    module = _patcher()
    sidecar = _module(SIDECAR, "fr14_splitk_sidecar_codegen")

    root = tmp_path / "fa2_splitk"
    _pristine_tree(root)
    changed = module.patch_fa2_source(
        root,
        tree_bias_tile_earlyout=True,
        fixed32_query_gqa_pair32_splitk_b1=True,
    )
    tu = "flash_fwd_fr13_qrow32_gqa_pair_splitk_b1_hdim256_bf16_sm80.cu"
    assert changed[tu] is True
    assert changed["flash_api.cpp"] is True
    assert changed["flash_fwd_kernel.h"] is True
    # The PROMOTED unit is emitted too, and deliberately: the baseline every
    # number is measured against must be the served kernel compiled into the
    # same binary, not a rebuild that resembles it.
    assert changed["flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu"] is True
    assert (root / "csrc/flash_attn/src"
            / "flash_fwd_fr13_qrow32_gqa_pair_b1_hdim256_bf16_sm80.cu").read_text() == (
        module.FIXED32_QUERY_GQA_PAIR32_B1_TRANSLATION_UNIT
    )
    # But no OTHER arm's unit is.
    assert changed["flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu"] is False
    assert changed["flash_fwd_fr13_qrow32_gqa_pair_hdim256_bf16_sm80.cu"] is False
    assert (root / "csrc/flash_attn/src" / tu).read_text() == (
        module.FIXED32_QUERY_GQA_PAIR32_SPLITK_B1_TRANSLATION_UNIT
    )

    header = (root / "csrc/flash_attn/src/flash_fwd_kernel.h").read_text()
    # Both output sites -- the empty-split early-out and the epilogue -- learned
    # to address the stock accumulators, and NOTHING is left asserting !Split.
    assert header.count(
        "// FR14_FA2_QROW32_GQA_PAIR_SPLITK_LAYOUT: the same paired"
    ) == 4
    assert "static_assert(!Split);" not in header
    assert "row_offset_oaccum" in header
    # The combine's static geometry, which is what removed the 64-bit division
    # the SASS contract caught.
    assert header.count("// FR14_FA2_COMBINE_STATIC_GEOMETRY") == 2
    assert "kStaticCombineBatch" in header
    assert "row = idx & (kStaticCombineRows - 1);" in header

    api = (root / "csrc/flash_attn/flash_api.cpp").read_text()
    assert "kFr14Qrow32GqaPairSplitKB1BatchStrideSentinel" in api
    assert (
        "fr13_run_mha_fwd_fixed32_qrow32_gqa_pair_splitk_b1(params, stream);" in api
    )
    # The promoted arm and the qrow16 reference must resolve in the same binary
    # so the characterization has something to measure against.
    assert "kFr13Qrow32GqaPairB1BatchStrideSentinel" in api
    assert "kFr13Qrow16BatchStrideSentinel" in api
    assert "fr14_qrow32_gqa_pair_splitk_b1" in api
    assert "FP16_SWITCH" in api

    for relative, digest in sidecar.SPLITK_SOURCE_FILES.items():
        assert sidecar.sha256_file(root / relative) == digest, relative
    closure = {"fa2_head": FA2_HEAD, "files": dict(sidecar.SPLITK_SOURCE_FILES)}
    assert (
        sidecar._digest(sidecar.canonical_bytes(closure))
        == sidecar.SPLITK_SOURCE_CLOSURE_SHA256
    )
    # The pinned modified-source set must name exactly the files the codegen
    # touched. The fixture tree is freshly git-init'd, so its own status marks
    # everything untracked; what is checkable here is the SET, which the build
    # script re-derives against a real checkout.
    status = sidecar._source_status(sidecar.SPLITK_ARM)
    assert {line.split(None, 1)[-1] for line in status} == set(
        sidecar.SPLITK_SOURCE_FILES
    )

    again = module.patch_fa2_source(
        root,
        tree_bias_tile_earlyout=True,
        fixed32_query_gqa_pair32_splitk_b1=True,
    )
    assert not any(again.values())


def test_splitk_source_build_stays_mutually_exclusive() -> None:
    module = _patcher()

    with pytest.raises(ValueError, match="mutually exclusive"):
        module.patch_fa2_source(
            Path("/nonexistent"),
            tree_bias_tile_earlyout=True,
            fixed32_query_gqa_pair32_b1=True,
            fixed32_query_gqa_pair32_splitk_b1=True,
        )
    with pytest.raises(ValueError, match="tree-bias-tile-earlyout"):
        module.patch_fa2_source(
            Path("/nonexistent"),
            fixed32_query_gqa_pair32_splitk_b1=True,
        )
