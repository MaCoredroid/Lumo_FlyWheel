from __future__ import annotations

import ast
from pathlib import Path

import pytest


DEPLOYED_SHAPE = (638, 10_240, 34)
DEPLOYED_ROW_ELEMS = 10_240 * 34
DEPLOYED_PAGE_ELEMS = 2_097_152
KERNEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "fr10_gdn_tree_kernel.py"
)


def _load_page_safe_row_span():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fixed32_conv_page_safe_row_span"
    ]
    assert len(definitions) == 1
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            definitions[0],
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    return namespace["_fixed32_conv_page_safe_row_span"]


PAGE_SAFE_ROW_SPAN = _load_page_safe_row_span()


@pytest.mark.parametrize(
    "stride",
    (
        (DEPLOYED_PAGE_ELEMS, 34, 1),
        (DEPLOYED_PAGE_ELEMS, 1, 10_240),
    ),
)
def test_fixed32_conv_layout_accepts_both_dense_inner_orders(
    stride: tuple[int, int, int],
) -> None:
    assert PAGE_SAFE_ROW_SPAN(DEPLOYED_SHAPE, stride) == DEPLOYED_ROW_ELEMS


@pytest.mark.parametrize(
    "stride",
    (
        (DEPLOYED_ROW_ELEMS - 1, 1, 10_240),
        (DEPLOYED_PAGE_ELEMS, 1, 1),
        (DEPLOYED_PAGE_ELEMS, 2, 20_480),
        (DEPLOYED_PAGE_ELEMS, 35, 1),
        (DEPLOYED_PAGE_ELEMS, 1, 10_241),
    ),
)
def test_fixed32_conv_layout_rejects_cross_page_aliased_or_holey_rows(
    stride: tuple[int, int, int],
) -> None:
    assert PAGE_SAFE_ROW_SPAN(DEPLOYED_SHAPE, stride) is None
