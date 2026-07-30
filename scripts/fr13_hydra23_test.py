#!/usr/bin/env python3
"""Focused CPU/static gate for the FR13 hydra23 traded-tail topology.

Run:
    .venv/bin/python scripts/fr13_hydra23_test.py

This test intentionally covers both the CPU-testable Arctic orchestration and
the thin static wiring in the vLLM patcher/launcher. A full-path Hydra child
cannot be represented by the older ``(parent_pos, child_rank)`` alias alone:
``(0, 0)``, ``(1, 0)``, and ``(2, 0)`` all have rank zero at parent position
one, but they are three different parent-conditioned candidates.
"""
from __future__ import annotations

import ast
import builtins
import os
from pathlib import Path
import re
import sys
from unittest import mock

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import fr13_merged_drafter as md  # noqa: E402
from fr13_merged_fill import (  # noqa: E402
    build_hydra_path_columns,
    get_oob_stats,
)
from fr13_mtp_suffix_assembly import (  # noqa: E402
    HYDRA23_BRANCH_CHAINS,
    HYDRA23_ORDER,
    hydra23_branch_chain_paths,
    tail_tree_order,
)


EXPECTED_HYDRA23_ORDER = [
    (0,),
    (1,),
    (2,),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (1, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
]
EXPECTED_CHOICE_PARENTS = [
    -1, -1, -1, 0, 0, 0, 1, 3, 3, 3, 6, 7, 7, 7, 10, 11, 14,
    15, 17, 18, 19, 20, 21,
]
EXPECTED_PATCHER_PARENTS = [
    -1, 0, 0, 0, 1, 1, 1, 2, 4, 4, 4, 7, 8, 8, 8, 11, 12, 15,
    16, 18, 19, 20, 21, 22,
]
EXPECTED_REMOVED_FROM_TAIL6 = {
    (0, 0, 0, 0, 1),
    (0, 0, 0, 0, 2),
}
EXPECTED_ADDED_TO_HYDRA23 = {
    (1, 0),
    (1, 0, 0),
    (1, 0, 0, 0),
    (1, 0, 0, 0, 0),
}


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def choice_parents(order):
    index = {path: i for i, path in enumerate(order)}
    return [-1 if len(path) == 1 else index[path[:-1]] for path in order]


def patcher_parents(order):
    actual = [()] + list(order)
    index = {path: i for i, path in enumerate(actual)}
    return [-1] + [index[path[:-1]] for path in order]


def extract_python_list_assignment(source: str, name: str):
    """Extract a literal list assignment while tolerating comments/newlines."""
    assignment = source.index(name)
    start = source.index("[", assignment)
    depth = 0
    for pos in range(start, len(source)):
        if source[pos] == "[":
            depth += 1
        elif source[pos] == "]":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(source[start:pos + 1])
    raise AssertionError(f"unterminated list assignment for {name}")


def extract_literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def extract_function(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            return module
    raise AssertionError(f"missing function {name}")


def extract_shell_tree(source: str, name: str):
    match = re.search(rf'^{re.escape(name)}="(.*)"$', source, re.MULTILINE)
    assert match, f"missing shell tree assignment {name}"
    return ast.literal_eval(match.group(1))


def extract_case_clause(source: str, kind: str):
    match = re.search(
        rf"^\s*{re.escape(kind)}\)(.*?)\s*;;\s*$",
        source,
        re.MULTILINE,
    )
    assert match, f"missing case clause for {kind}"
    return match.group(1)


def clause_flags(clause: str):
    match = re.search(r"declare -a XFLAGS=\(([^)]*)\)", clause)
    assert match, f"missing XFLAGS in clause: {clause}"
    return match.group(1).split()


def test_exact_topology():
    assert HYDRA23_ORDER == EXPECTED_HYDRA23_ORDER
    assert HYDRA23_BRANCH_CHAINS == ((1, 4),)
    assert hydra23_branch_chain_paths() == [
        (1, 0),
        (1, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]
    assert len(HYDRA23_ORDER) == 23
    assert max(map(len, HYDRA23_ORDER)) == 11
    assert len(HYDRA23_ORDER) + 1 == 24
    assert next_power_of_two(len(HYDRA23_ORDER) + 1) == 32
    assert choice_parents(HYDRA23_ORDER) == EXPECTED_CHOICE_PARENTS
    assert patcher_parents(HYDRA23_ORDER) == EXPECTED_PATCHER_PARENTS

    tail6 = tail_tree_order(tail_len=6)
    assert len(tail6) == 21
    assert set(tail6) - set(HYDRA23_ORDER) == EXPECTED_REMOVED_FROM_TAIL6
    assert set(HYDRA23_ORDER) - set(tail6) == EXPECTED_ADDED_TO_HYDRA23
    assert [
        path for path in HYDRA23_ORDER if all(rank == 0 for rank in path)
    ] == [
        path for path in tail6 if all(rank == 0 for rank in path)
    ], "the complete depth-11 main spine must be unchanged"


class MockDraft:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class PatternCache:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def speculate(self, req_id, pattern, **kwargs):
        key = (str(req_id), tuple(int(token) for token in pattern))
        self.calls.append((key[0], list(key[1]), dict(kwargs)))
        return MockDraft(self.responses.get(key, []))


def test_two_walks_and_path_local_carry():
    md.reset_for_test()
    md._COMMITTED["ra"] = [9, 8]
    md._COMMITTED["rb"] = [7]
    head = [
        [100, 200],
        [101, 201],
        [102, 202],
        [103, 203],
        [104, 204],
    ]
    seeds = {1: [301, 302]}
    patterns = {
        ("ra", (9, 8, 100, 101, 102, 103, 104)): [500, 501, 502, 503, 504, 505],
        ("ra", (9, 8, 301)): [310, 311],
        ("rb", (7, 200, 201, 202, 203, 204)): [600, 601, 602, 603, 604, 605],
        ("rb", (7, 302)): [],
    }
    cache = PatternCache(patterns)

    real_open = builtins.open

    def no_tail_branch_sidecar(file, *args, **kwargs):
        if os.fspath(file) == "/logs/fr13_tail_branches.cfg":
            raise FileNotFoundError(file)
        return real_open(file, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=no_tail_branch_sidecar):
        tail = md.decide_tail(
            cache,
            ["ra", "rb"],
            head,
            head_depth=5,
            tail_len=6,
            device=torch.device("cpu"),
            pad_token=0,
            max_spec_tokens=6,
            vocab_size=1000,
            hydra_seed_per_rank=seeds,
            hydra_branch_chains=((1, 4),),
        )

    assert tail is not None and len(tail) == 6
    assert [column.tolist() for column in tail] == [
        [500, 600],
        [501, 601],
        [502, 602],
        [503, 603],
        [504, 604],
        [505, 605],
    ]

    expected_patterns = [
        ("ra", [9, 8, 100, 101, 102, 103, 104]),
        ("ra", [9, 8, 301]),
        ("rb", [7, 200, 201, 202, 203, 204]),
        ("rb", [7, 302]),
    ]
    assert [(req_id, pattern) for req_id, pattern, _kw in cache.calls] == expected_patterns
    assert len(cache.calls) == 2 * 2
    assert all(
        sum(call[0] == req_id for call in cache.calls) == 2
        for req_id in ("ra", "rb")
    )
    assert [call[2]["max_spec_tokens"] for call in cache.calls] == [6, 4, 6, 4]
    assert all(call[2]["use_tree_spec"] is False for call in cache.calls)

    paths = md.get_tail_path_tokens()
    assert set(paths) == EXPECTED_ADDED_TO_HYDRA23
    assert [path for path in paths if path[0] == 1] == [
        (1, 0),
        (1, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]
    assert paths[(1, 0)].tolist() == [310, 302]
    assert paths[(1, 0, 0)].tolist() == [311, 302]
    assert paths[(1, 0, 0, 0)].tolist() == [311, 302]
    assert paths[(1, 0, 0, 0, 0)].tolist() == [311, 302]
    assert all(column.dtype == torch.int64 for column in paths.values())
    assert all(column.device.type == "cpu" for column in paths.values())
    assert all(column.is_contiguous() for column in paths.values())
    assert len({
        column.untyped_storage().data_ptr()
        for column in [*tail, *paths.values()]
    }) == 1
    assert md.STATS["tail_speculate_fired"] == 2
    assert md.STATS["hydra_speculate_fired"] == 2
    assert md.STATS["hydra_real"] == 2
    assert md.STATS["hydra_rank2_hit"] == 1
    assert md.STATS["hydra_rank3_hit"] == 0

    # Mirror the patcher's full-path dispatch for the two colliding rank-zero
    # paths at parent_pos=1. Only the main path may use the main spine column.
    main_spine = {
        (0, 0): torch.tensor([101, 201], dtype=torch.int64),
    }
    packed = {
        path: paths[path] if path in paths else main_spine[path]
        for path in ((0, 0), (1, 0))
    }
    assert packed[(0, 0)].tolist() == [101, 201]
    assert packed[(1, 0)].tolist() == [310, 302]
    assert len({tuple(column.tolist()) for column in packed.values()}) == 2


def test_hydra_oob_is_lossless_pad():
    before, _last = get_oob_stats()
    paths = build_hydra_path_columns(
        [
            {
                1: [10, 9999, 12, 13],
            }
        ],
        torch.device("cpu"),
        pad_token=7,
        vocab_size=100,
    )
    after, last = get_oob_stats()
    assert paths[(1, 0)].tolist() == [10]
    assert paths[(1, 0, 0)].tolist() == [7]
    assert paths[(1, 0, 0, 0)].tolist() == [12]
    assert after == before + 1
    assert last == ("hydra", 1, 1, 9999)


def test_static_patcher_contract():
    patcher = (SCRIPT_DIR / "fr10_phase4_patch_vllm_tree_gdn.py").read_text()
    patcher_choices = extract_python_list_assignment(
        patcher, "_fr13_hydra23_choices ="
    )
    patcher_paths = extract_python_list_assignment(
        patcher, "_fr13_hydra23_tail_paths ="
    )
    assert patcher_choices == EXPECTED_HYDRA23_ORDER
    assert patcher_paths == [
        (1, 0),
        (1, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]

    guard = patcher[
        patcher.index("_fr13_hydra23_armed ="):
        patcher.index("_fr10_is_caterpillar =", patcher.index("_fr13_hydra23_armed ="))
    ]
    assert '_fr13_hydra23_armed = os.path.exists("/logs/fr13_hydra23.arm")' in guard
    assert '_fr10_active_decode_mode == "tree_mtp"' in guard
    assert "int(self.num_speculative_tokens) == 23" in guard
    assert "_fr10_tree_choices_current == _fr13_hydra23_choices" in guard
    assert "if _fr13_hydra23_armed != _fr13_is_hydra23:" in guard
    assert "raise RuntimeError(" in guard
    assert 'os.path.exists("/logs/fr13_tail_mode.arm")' in guard
    assert '"/logs/fr13_draft_source_merged.arm"' in guard
    assert (
        "(_fr10_wide_choices_ok or _fr13_is_hydra23 or _fr13_is_fixed32)"
        in patcher
    )

    host_start = patcher.index("# FR13_TAIL_HOSTCOPY_BATCHED")
    host_end = patcher.index(
        "_fr13_t_cols = _fr13_t.decide_tail", host_start
    )
    host_copy = patcher[host_start:host_end]
    executable_host_copy = "\n".join(
        line for line in host_copy.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert len(re.findall(r"\.cpu\s*\(", executable_host_copy)) == 1
    assert "_fr13_t_host_cols.append(" in host_copy
    assert "_fr13_t_root_wtk[:, 1].detach().reshape(-1)" in host_copy
    hydra_host_copy = host_copy[
        host_copy.index("if _fr13_is_hydra23"):
        host_copy.index("elif _fr13_is_fixed32")
    ]
    assert "_fr13_t_root_wtk[:, 2].detach().reshape(-1)" not in hydra_host_copy
    assert "_fr13_t_stack = torch.stack(_fr13_t_host_cols).cpu()" in host_copy
    assert host_copy.index("_fr13_t_host_cols.append(") < host_copy.index(
        "_fr13_t_stack = torch.stack"
    )
    assert "if _fr13_is_hydra23" in host_copy
    assert "else None" in host_copy
    assert "_fr13_hydra_contract_error = True" in patcher
    assert "if _fr13_is_hydra23 and _fr13_hydra_contract_error:" in patcher
    assert "_fr13_sp_required = (" in patcher
    assert patcher.count("if _fr13_sp_required:") == 3
    assert "os.path.exists('/logs/fr13_subtree_parallel.arm')" in patcher
    assert "os.path.exists('/tmp/fr13_subtree_parallel.arm')" in patcher
    assert "'requires a 3-dim ssm shape'" in patcher

    pack_start = patcher.index("# FR13_RESHAPE_WIDE general packer")
    pack_end = patcher.index(
        "_fr10_packed = torch.stack(_fr10_wide_cols, dim=1)", pack_start
    )
    pack = patcher[pack_start:pack_end]
    path_dispatch = pack.index("_fr10_path in _fr13_hydra_path_tokens")
    rank_dispatch = pack.index("elif _fr10_rk == 0")
    assert "zip(\n                    _fr10_wide_paths, _fr10_wide_plan" in pack
    assert "_fr13_hydra_path_tokens[_fr10_path]" in pack
    assert path_dispatch < rank_dispatch


def test_hydra_subtree_floor_schedule():
    kernel_path = (
        SCRIPT_DIR.parent
        / "src"
        / "lumo_flywheel_serving"
        / "fr10_gdn_tree_kernel.py"
    )
    kernel = kernel_path.read_text()
    parent = tuple(
        extract_literal_assignment(kernel, "_FR13_HYDRA23_PARENT")
    )
    literal_levels = extract_literal_assignment(
        kernel, "_FR13_HYDRA23_SUBTREE_LEVELS"
    )
    assert parent == tuple(EXPECTED_PATCHER_PARENTS)

    namespace = {
        "_FR13_HYDRA23_PARENT": parent,
        "_FR13_HYDRA23_SUBTREE_LEVELS": literal_levels,
    }
    exec(
        compile(
            extract_function(kernel, "_subtree_decompose"),
            str(kernel_path),
            "exec",
        ),
        namespace,
    )
    decompose = namespace["_subtree_decompose"]
    levels = decompose(parent)
    assert levels == [
        [([0, 1, 4, 8], -1)],
        [
            ([12, 16, 18, 19, 20, 21, 22, 23], 8),
            ([2, 7, 11, 15, 17], 0),
            ([3], 0),
            ([5], 1),
            ([6], 1),
            ([9], 4),
            ([10], 4),
            ([13], 8),
            ([14], 8),
        ],
    ]

    seen = set()
    earlier = set()
    export_parents = set()
    for level in levels:
        current = set()
        for path, path_parent in level:
            assert path
            assert parent[path[0]] == path_parent
            assert path_parent < 0 or path_parent in earlier
            export_parents.update([path_parent] if path_parent >= 0 else [])
            for prev, node in zip(path, path[1:]):
                assert parent[node] == prev
            assert not seen.intersection(path)
            assert not current.intersection(path)
            current.update(path)
        seen.update(current)
        earlier.update(current)
    assert seen == set(range(24))
    assert export_parents == {0, 1, 4, 8}
    max_lens = [max(len(path) for path, _par in level) for level in levels]
    assert [len(level) for level in levels] == [1, 9]
    assert max_lens == [4, 8]
    assert sum(max_lens) == 12
    assert sum(
        len(level) * max_len
        for level, max_len in zip(levels, max_lens)
    ) == 76
    assert sum(len(path) for level in levels for path, _par in level) == 24

    main = (0, 1, 4, 8, 12, 16, 18, 19, 20, 21, 22, 23)
    sides = (
        ((2, 7, 11, 15, 17), 0),
        ((3,), 0),
        ((5,), 1),
        ((6,), 1),
        ((9,), 4),
        ((10,), 4),
        ((13,), 8),
        ((14,), 8),
    )
    expected_static_slots = {4: 76, 5: 68, 6: 60, 7: 52}
    for prefix_len in range(4, 8):
        candidate = [
            [(list(main[:prefix_len]), -1)],
            [
                (list(main[prefix_len:]), main[prefix_len - 1]),
                *[(list(path), par) for path, par in sides],
            ],
        ]
        candidate_max = [
            max(len(path) for path, _par in level)
            for level in candidate
        ]
        assert sum(candidate_max) == 12
        assert sum(
            len(path) for level in candidate for path, _par in level
        ) == 24
        assert sum(
            len(level) * max_len
            for level, max_len in zip(candidate, candidate_max)
        ) == expected_static_slots[prefix_len]

    preseed = kernel[
        kernel.index("def subtree_preseed"):
        kernel.index("def subtree_get")
    ]
    path_kernel = kernel[
        kernel.index("def _tree_gdn_path_kernel"):
        kernel.index("def _tree_gdn_replay_kernel")
    ]
    launcher = kernel[
        kernel.index("def _launch_paths"):
        kernel.index("if parent_gather_selfcheck_on()")
    ]
    assert "lengths = torch.empty(len(lvl), dtype=torch.int32)" in preseed
    assert "lengths[i] = len(p)" in preseed
    assert "lengths.to(device)" in preseed
    assert "path_len = tl.load(path_lengths + pid_path)" in path_kernel
    assert "for i in tl.range(0, path_len):" in path_kernel
    assert "tl.static_range(0, MAX_PATH_LEN)" not in path_kernel
    assert "_lengths" in launcher
    assert "COUNT_INVOCATION=_count and (_li == 0)" in launcher
    assert "_subtree_state = _FR13_SUBTREE_CACHE.get(" in kernel
    assert "_subtree_cache_key(" in kernel
    assert "external_state_equal=1 counter_once=1" in kernel

    # A non-Hydra tree still takes the original deterministic heavy-path path.
    assert decompose((-1, 0, 0, 1, 1)) == [
        [([0, 1, 3], -1)],
        [([4], 1), ([2], 0)],
    ]


def test_subtree_worker_propagation_contract():
    kernel_path = (
        SCRIPT_DIR.parent
        / "src"
        / "lumo_flywheel_serving"
        / "fr10_gdn_tree_kernel.py"
    )
    kernel = kernel_path.read_text()
    launcher = (
        SCRIPT_DIR / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text()
    variant = (
        SCRIPT_DIR / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text()

    route_gate = ast.unparse(extract_function(kernel, "subtree_parallel_on"))
    selfcheck_gate = ast.unparse(
        extract_function(kernel, "subtree_parallel_selfcheck_on")
    )
    assert "/logs/fr13_subtree_parallel.arm" in route_gate
    assert "/tmp/fr13_subtree_parallel.arm" in route_gate
    assert "/logs/fr13_subtree_parallel_selfcheck.arm" in selfcheck_gate
    assert "/tmp/fr13_subtree_parallel_selfcheck.arm" in selfcheck_gate

    fake_os = mock.Mock()
    fake_os.environ = {}
    fake_os.path.exists.side_effect = (
        lambda path: path == "/logs/fr13_subtree_parallel.arm"
    )
    namespace = {"os": fake_os}
    exec(compile(extract_function(kernel, "subtree_parallel_on"), "<gate>", "exec"),
         namespace)
    assert namespace["subtree_parallel_on"]()
    fake_os.path.exists.side_effect = (
        lambda path: path == "/logs/fr13_subtree_parallel_selfcheck.arm"
    )
    exec(
        compile(
            extract_function(kernel, "subtree_parallel_selfcheck_on"),
            "<selfcheck-gate>",
            "exec",
        ),
        namespace,
    )
    assert namespace["subtree_parallel_selfcheck_on"]()

    assert ': > "$LOG_DIR/fr13_subtree_parallel.arm"' in launcher
    assert 'rm -f "$LOG_DIR/fr13_subtree_parallel.arm"' in launcher
    assert ': > "$LOG_DIR/fr13_subtree_parallel_selfcheck.arm"' in launcher
    assert 'rm -f "$LOG_DIR/fr13_subtree_parallel_selfcheck.arm"' in launcher
    assert (
        'FR13_SUBTREE_PARALLEL_SELFCHECK=1 requires '
        'FR13_SUBTREE_PARALLEL=1'
    ) in launcher
    assert (
        "FR13_SUBTREE_PARALLEL_SELFCHECK=1 requires ENFORCE_EAGER=1"
        in launcher
    )

    assert "[FR13_SUBTREE_PARALLEL ENGAGED]" in kernel
    assert "[FR13_SUBTREE_PARALLEL SELFCHECK PASS]" in kernel
    assert "route_armed={int(route_armed)}" in kernel
    assert "_subtree_route_armed" in kernel
    assert "_subtree_selfcheck_armed" in kernel
    assert "[FR13_SUBTREE_PARALLEL ENGAGED]" in variant
    assert "[FR13_SUBTREE_PARALLEL SELFCHECK PASS]" in variant
    assert "FR13_HYDRA23 exact topology engaged" in variant
    assert '"levels=[1, 9]"' in variant
    assert '"lens=[4, 8]"' in variant
    assert "Hydra23 floor schedule OK: critical=12" in variant

    campaign = (SCRIPT_DIR / "fr13_b4_campaign_driver.sh").read_text()
    assert campaign.count(
        "FAIL: serving/gate rc=$rc; terminating campaign"
    ) == 2
    assert campaign.count(
        'echo "[$arm] FAIL: serving/gate rc=$rc; terminating campaign"\n'
        '    exit "$rc"'
    ) == 2


def test_variant_and_tail6_wiring():
    variant = (SCRIPT_DIR / "fr13_bigdenom_swe_serve_variant.sh").read_text()
    launcher = (
        SCRIPT_DIR / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text()
    assert extract_shell_tree(variant, "HYDRA23_TREE") == EXPECTED_HYDRA23_ORDER

    hydra_clause = extract_case_clause(variant, "hydra23")
    assert "LAUNCHER=forked" in hydra_clause
    assert 'TREEARG="$HYDRA23_TREE"' in hydra_clause
    assert "EXPECT_RATIO=23" in hydra_clause
    assert clause_flags(hydra_clause) == [
        "FR13_TAIL_MODE=1",
        "FR13_DRAFT_SOURCE=merged",
        "FR13_TREE_GDN_GEOM_OVERRIDE=BV=8",
        "FR13_HYDRA23=1",
    ]

    expected_tail6 = tail_tree_order(tail_len=6)
    assert extract_shell_tree(variant, "TAIL6_TREE") == expected_tail6
    tail6_clause = extract_case_clause(variant, "tail6")
    assert "LAUNCHER=forked" in tail6_clause
    assert 'TREEARG="$TAIL6_TREE"' in tail6_clause
    assert "EXPECT_RATIO=21" in tail6_clause
    assert clause_flags(tail6_clause) == [
        "FR13_TAIL_MODE=1",
        "FR13_DRAFT_SOURCE=merged",
        "FR13_TREE_GDN_GEOM_OVERRIDE=BV=8",
    ]
    assert "FR13_HYDRA23" not in tail6_clause

    assert 'if [[ "${FR13_HYDRA23:-0}" == "1" ]]; then' in launcher
    assert ': > "$LOG_DIR/fr13_hydra23.arm"' in launcher
    assert 'rm -f "$LOG_DIR/fr13_hydra23.arm"' in launcher
    assert "root-rank1 rescue chain length 4" in launcher
    assert "root rescue chains 4+2" not in launcher


if __name__ == "__main__":
    tests = [
        test_exact_topology,
        test_two_walks_and_path_local_carry,
        test_hydra_oob_is_lossless_pad,
        test_static_patcher_contract,
        test_hydra_subtree_floor_schedule,
        test_subtree_worker_propagation_contract,
        test_variant_and_tail6_wiring,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("ALL HYDRA23 CPU/STATIC TESTS PASS")
