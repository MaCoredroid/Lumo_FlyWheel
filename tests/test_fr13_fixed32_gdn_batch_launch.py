from __future__ import annotations

import ast
import hashlib
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)
PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _kernel_tree_and_source() -> tuple[ast.Module, str]:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    return ast.parse(source), source


def _load_fixed32_constants() -> dict[str, object]:
    wanted = {
        "_FR13_FIXED32_PARENT",
        "_FR13_FIXED32_SUBTREE_LEVELS",
        "_FR13_FIXED32_EXPORT_NODES",
        "_FR13_FIXED32_EXPORT_SLOTS",
        "_FR13_FIXED32_MAX_BATCH",
    }
    tree, _source = _kernel_tree_and_source()
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in wanted
            for target in node.targets
        )
    ]
    module = ast.Module(body=assignments, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    assert wanted <= namespace.keys()
    return namespace


def _function_source(name: str) -> str:
    tree, source = _kernel_tree_and_source()
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_fixed32_batched_programs_match_request_local_reference() -> None:
    namespace = _load_fixed32_constants()
    parent = tuple(namespace["_FR13_FIXED32_PARENT"])
    levels = tuple(namespace["_FR13_FIXED32_SUBTREE_LEVELS"])
    export_nodes = tuple(namespace["_FR13_FIXED32_EXPORT_NODES"])
    export_slots = int(namespace["_FR13_FIXED32_EXPORT_SLOTS"])
    max_batch = int(namespace["_FR13_FIXED32_MAX_BATCH"])

    assert len(parent) == 32
    assert tuple(levels[0][0][0]) == export_nodes == (0, 1, 4, 9, 14)
    assert tuple(len(level) for level in levels) == (1, 11)
    assert export_slots == 5
    slot_by_parent = {node: slot for slot, node in enumerate(export_nodes)}

    for batch in range(1, max_batch + 1):
        assert len(levels) == 2
        assert tuple(batch * len(level) for level in levels) == (
            batch,
            11 * batch,
        )

        by_request: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {
            request: [] for request in range(batch)
        }
        destinations: set[tuple[int, int]] = set()
        for level_index, level in enumerate(levels):
            num_paths = len(level)
            for global_path in range(batch * num_paths):
                request, local_path = divmod(global_path, num_paths)
                path, parent_node = level[local_path]
                path_tuple = tuple(path)
                by_request[request].append(
                    (level_index, int(parent_node), path_tuple)
                )
                for node in path_tuple:
                    global_row = request * len(parent) + int(node)
                    legacy_sliced_row = request * len(parent) + int(node)
                    assert global_row == legacy_sliced_row
                    destinations.add((request, int(node)))

        expected_request = [
            (level_index, int(parent_node), tuple(path))
            for level_index, level in enumerate(levels)
            for path, parent_node in level
        ]
        assert all(plan == expected_request for plan in by_request.values())
        assert destinations == {
            (request, node)
            for request in range(batch)
            for node in range(len(parent))
        }

        compact_rows: set[int] = set()
        for request in range(batch):
            for slot, _node in enumerate(export_nodes):
                compact_rows.add(request * export_slots + slot)
            for _path, parent_node in levels[1]:
                load_row = (
                    request * export_slots + slot_by_parent[int(parent_node)]
                )
                assert load_row in compact_rows
                assert load_row // export_slots == request
        assert len(compact_rows) == batch * export_slots
        assert max(compact_rows) < 32


def test_batched_kernel_keeps_two_launches_and_b1_legacy_route() -> None:
    launcher = _function_source("launch_tree_gdn_prepared_fixed32_batch")
    batched_kernel = _function_source("_tree_gdn_path_kernel_fixed32_batch")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert "for level_index" in launcher
    assert "batch * num_paths" in launcher
    assert "int(contract.get(\"launches\", -1)) != 2" in launcher
    assert "pid_batch = pid_global_path // NUM_PATHS" in batched_kernel
    assert "pid_batch < BATCH_SIZE" in batched_kernel
    assert "global_node = pid_batch * N_ACTUAL + node_c" in batched_kernel
    assert "H0_INDEX_ROW + pid_batch * H0_INDEX_BATCH_STRIDE" in batched_kernel
    assert "H0_BATCH_INDEX + pid_batch * H0_ACCEPTED_BATCH_STRIDE" in batched_kernel
    assert "pid_batch.to(tl.int64) * EXPORT_SLOTS + i" in batched_kernel
    assert "pid_batch.to(tl.int64) * EXPORT_SLOTS" in batched_kernel
    assert "(pid_batch == 0)" in batched_kernel
    assert "batch_size must be in [2, 4]" in launcher
    assert "launch_tree_gdn_prepared_fixed32_batch" in patcher
    assert "fixed32_batch_gdn_selector(" in patcher
    assert "_fr13_fixed32_batch_gdn_selector is not None" in patcher
    assert "else:\n                        tree_out, _ = launch_tree_gdn_prepared(" in patcher
    assert "FR13_FIXED32_BATCH_GDN is default-off" in launcher

    patcher_tree = ast.parse(patcher)
    generated_fragments = [
        node.value
        for node in ast.walk(patcher_tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "_fr13_fixed32_batch_gdn_selector = (" in node.value
    ]
    assert len(generated_fragments) == 1
    ast.parse(textwrap.dedent(generated_fragments[0]))


def test_path_kernel_default_off_candidate_ast_is_pinned() -> None:
    source = _function_source("_tree_gdn_path_kernel")
    assert "PRECOMPUTE_LEVEL1: tl.constexpr = False" in source
    assert "LOAD_PRECOMPUTED: tl.constexpr = False" in source
    canonical = ast.dump(ast.parse(source), include_attributes=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == (
        "5f020ab18bf0b4cfd2c9d73d7dbf4228be5566a186f121c74d2d4bde0e8ccd6a"
    )
