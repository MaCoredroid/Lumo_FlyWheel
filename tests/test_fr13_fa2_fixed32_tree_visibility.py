from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_patch_fa2_tree_bias.py")
    spec = importlib.util.spec_from_file_location(
        "fr13_fa2_fixed32_tree_visibility", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _visible(parent: tuple[int, ...], query: int, key: int) -> bool:
    cursor = query
    while cursor >= 0:
        if cursor == key:
            return True
        cursor = parent[cursor]
    return False


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def test_fixed32_visibility_masks_cover_the_full_physical_tree() -> None:
    module = _module()
    parent = module.FIXED32_PHYSICAL_PARENT
    masks = module.FIXED32_TREE_VISIBILITY_MASKS

    assert len(parent) == len(masks) == 32
    assert parent[0] == -1
    for query, row_mask in enumerate(masks):
        for key in range(32):
            assert bool(row_mask & (1 << key)) == _visible(parent, query, key)

    # Tail23 and Hydra27 differ only in downstream valid-node ownership. The
    # attention launch retains all 32 physical slots and one shared topology.
    for valid_mask in (0x7A9CE7FF, 0x7ABDFFFF):
        for query in range(32):
            assert masks[query] & (1 << query)
            if not (valid_mask & (1 << query)):
                assert masks[query] != 0


def test_visibility_parent_is_bound_to_the_served_physical32_tree() -> None:
    module = _module()
    phase4 = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
    served_parent = tuple(_literal_assignment(phase4, "_FR13_FIXED32_PARENT"))
    choices = tuple(_literal_assignment(phase4, "_FR13_FIXED32_CHOICES"))

    index = {choice: position + 1 for position, choice in enumerate(choices)}
    reconstructed = [-1]
    for choice in choices:
        reconstructed.append(0 if len(choice) == 1 else index[choice[:-1]])

    assert tuple(reconstructed) == served_parent
    assert served_parent == module.FIXED32_PHYSICAL_PARENT


def test_visibility_helper_is_opt_in_and_keeps_dense_fallback() -> None:
    module = _module()
    baseline = module._tree_bias_helper(tile_earlyout=True)
    candidate = module._tree_bias_helper(
        tile_earlyout=True,
        fixed32_tree_visibility_mask=True,
    )

    assert "StaticTreeVisibility" not in baseline
    assert "StaticTreeVisibility" in candidate
    assert "kStaticTreeVisibility" in candidate
    assert "tree_visibility & (1U << k_rel)" in candidate
    assert "? 0.0f : -INFINITY" in candidate
    assert "tree_bias[" in candidate
    assert "if constexpr (!kStaticTreeVisibility)" in candidate
    assert module.TREE_BIAS_TILE_OVERLAP_GUARD in candidate


@pytest.mark.parametrize(
    ("source_name", "trait", "symbol"),
    (
        (
            "FIXED32_QUERY_TILE32_B1_TRANSLATION_UNIT",
            "Fr13Fixed32Qrow32B1KernelTraits",
            "fr13_fixed32_qrow32_b1_tree_visibility",
        ),
        (
            "FIXED32_QUERY_TILE32_TRANSLATION_UNIT",
            "Fr13Fixed32Qrow32KernelTraits",
            "fr13_fixed32_qrow32_tree_visibility",
        ),
        (
            "FIXED32_QUERY_GQA_PAIR32_TRANSLATION_UNIT",
            "Fr13Fixed32Qrow32GqaPairKernelTraits",
            "fr13_fixed32_qrow32_gqa_pair_tree_visibility",
        ),
    ),
)
def test_private_query_kernels_receive_one_source_bound_visibility_table(
    source_name: str,
    trait: str,
    symbol: str,
) -> None:
    module = _module()
    source = getattr(module, source_name)
    candidate = module._with_fixed32_tree_visibility(
        source,
        trait=trait,
        symbol=symbol,
        max_registers=252,
    )

    assert "FR13_FA2_FIXED32_TREE_VISIBILITY_MASK" not in source
    assert candidate.count("FR13_FA2_FIXED32_TREE_VISIBILITY_MASK") == 1
    assert candidate.count(f"StaticTreeVisibility<{trait}>") == 1
    assert candidate.count(f"{symbol}[32]") == 1
    assert candidate.count(f"return {symbol}[row]") == 1
    assert candidate.count("__global__ __maxnreg__(252)") == 1
    assert "__global__ __maxnreg__(254)" not in candidate
    for row_mask in module.FIXED32_TREE_VISIBILITY_MASKS:
        assert candidate.count(f"0x{row_mask:08x}U") == 1
    assert candidate.index("StaticTreeVisibility<") < candidate.index("__global__")


def test_rejected_b1_split2_source_is_not_reworked() -> None:
    module = _module()
    stock = "\n".join(
        (
            '#include "namespace_config.h"',
            '#include "flash_fwd_launch_template.h"',
            "namespace FLASH_NAMESPACE {",
            module.STOCK_FIXED32_QUERY_INSTANTIATION,
            "} // namespace FLASH_NAMESPACE",
        )
    )

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        source = Path(directory) / "flash_fwd_split_hdim256_bf16_sm80.cu"
        source.write_text(stock)
        assert module._patch_fixed32_query_tile32_b1_translation_unit(
            source,
            fixed32_query_tile32_b1=True,
            fixed32_tree_visibility_mask=True,
        )
        nosplit = source.with_name(
            "flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu"
        ).read_text()
        split2 = source.with_name(
            "flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu"
        ).read_text()

    assert "FR13_FA2_FIXED32_TREE_VISIBILITY_MASK" in nosplit
    assert split2 == module.FIXED32_QUERY_TILE32_B1_SPLIT2_TRANSLATION_UNIT
    assert "FR13_FA2_FIXED32_TREE_VISIBILITY_MASK" not in split2


def test_visibility_source_traffic_model_is_bounded_for_b1_and_b4() -> None:
    layers = 16
    rows = 32
    cols = 32
    heads = 24

    dense_b1 = layers * heads * rows * cols * 4
    mask_b1_min = layers * heads * rows * 4
    mask_b1_max = 2 * mask_b1_min
    dense_b4 = 4 * dense_b1
    mask_b4_min = 4 * mask_b1_min
    mask_b4_max = 2 * mask_b4_min

    assert dense_b1 == 1_572_864
    assert (mask_b1_min, mask_b1_max) == (49_152, 98_304)
    assert dense_b4 == 6_291_456
    assert (mask_b4_min, mask_b4_max) == (196_608, 393_216)
    assert dense_b1 // mask_b1_max == dense_b4 // mask_b4_max == 16


def test_visibility_flag_requires_a_private_qrow32_kernel(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="requires a private qrow32"):
        module.patch_fa2_source(
            tmp_path,
            fixed32_query_tile16=True,
            fixed32_tree_visibility_mask=True,
        )
