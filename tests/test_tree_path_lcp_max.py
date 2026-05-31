from scripts.swe_x86_helpers import relaunch_qwen36_round as round_relaunch

_tree_path_lcp_max_reference = round_relaunch._tree_path_lcp_max_reference


def test_path_lcp_max_prefers_alt_longer_spine() -> None:
    # Tree: path0 = 0 -> 2, path1 = 1 -> 3. The target argmax rejects path0 at
    # depth 1 but accepts path1 through depth 2, so the winner must be path1.
    result = _tree_path_lcp_max_reference(
        parents=[-1, -1, 0, 1],
        draft_tokens=[10, 20, 11, 21],
        parent_target_tokens=[99, 20, 99, 21],
        self_target_tokens=[100, 200, 101, 201],
    )

    assert result["accepted_len"] == 2
    assert result["path0_lcp"] == 0
    assert result["superset_violation"] is False
    assert result["accepted_row"] == 4
    assert result["accepted_node_ids"] == [1, 3]
    assert result["output_tokens"] == [20, 21, 201]


def test_path_lcp_max_is_superset_of_path0() -> None:
    # path0 is fully accepted, alternate roots are not. The verifier must not
    # choose a shorter branch over the native chain.
    result = _tree_path_lcp_max_reference(
        parents=[-1, -1, -1, 0, 1, 2],
        draft_tokens=[10, 20, 30, 11, 21, 31],
        parent_target_tokens=[10, 99, 99, 11, 99, 99],
        self_target_tokens=[100, 200, 300, 101, 201, 301],
    )

    assert result["accepted_len"] == 2
    assert result["path0_lcp"] == 2
    assert result["superset_violation"] is False
    assert result["accepted_row"] == 4
    assert result["accepted_node_ids"] == [0, 3]
    assert result["output_tokens"] == [10, 11, 101]


def test_path_lcp_max_tie_preserves_path0_order() -> None:
    # Both root-to-leaf paths accept one token. Stable flattened-leaf tie-break
    # keeps path0, which is needed for byte-identical chain parity.
    result = _tree_path_lcp_max_reference(
        parents=[-1, -1, 0, 1],
        draft_tokens=[10, 20, 11, 21],
        parent_target_tokens=[10, 20, 99, 99],
        self_target_tokens=[100, 200, 101, 201],
    )

    assert result["accepted_len"] == 1
    assert result["path0_lcp"] == 1
    assert result["superset_violation"] is False
    assert result["accepted_row"] == 1
    assert result["accepted_node_ids"] == [0]
    assert result["output_tokens"] == [10, 99]


def test_path_lcp_max_handles_ten_spines() -> None:
    parents = []
    drafts = []
    targets = []
    self_targets = []
    for root in range(10):
        root_idx = len(parents)
        child_idx = root_idx + 10
        parents.append(-1)
        drafts.append(100 + root)
        targets.append(100 + root if root == 7 else -1)
        self_targets.append(1000 + root)
    for root in range(10):
        parents.append(root)
        drafts.append(200 + root)
        targets.append(200 + root if root == 7 else -1)
        self_targets.append(2000 + root)

    result = _tree_path_lcp_max_reference(
        parents=parents,
        draft_tokens=drafts,
        parent_target_tokens=targets,
        self_target_tokens=self_targets,
    )

    assert result["accepted_len"] == 2
    assert result["path0_lcp"] == 0
    assert result["superset_violation"] is False
    assert result["accepted_row"] == 18
    assert result["accepted_node_ids"] == [7, 17]
    assert result["output_tokens"] == [107, 207, 2007]


def test_default_tree_width_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("LUMO_TREE_SPINES", "4")

    assert round_relaunch._default_tree(3) == str(
        [
            (0,),
            (1,),
            (2,),
            (3,),
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
        ]
    )
