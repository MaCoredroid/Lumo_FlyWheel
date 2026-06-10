"""FR13 conv fix: committed-path conv window + forced-spine diag flag.

S1/S2/S3 discriminator bind (FR13_S1S2S3_DISCRIMINATE_BIND.md): the S2
deterministic verify corruption follows an acc=2 BRANCH commit, and the m1
locus is the verify/state-advance side. The legacy prior conv window read
gathers spec_state_indices[:, accepted_len-1] AFTER the in-place
launch_tree_state_linear_remap — linear-column arithmetic that is only
spine-valid by construction.

FR13_CONV_COMMITTED_PATH (default ON): read the committed conv window from
the accepted path's LEAF NODE column (node-indexed, BEFORE the remap mutates
the bank). The per-node tree-conv write-back stores each node's window as the
last (width-1) taps of (prior ++ its root-path tokens), so the accepted
leaf's bank row IS the committed-token-path window — valid for BRANCH
winners; byte-identical to legacy for spine winners (the license).

FR13_FORCE_SPINE_COMMIT (default OFF) — DIAGNOSTIC ONLY, never a serving
config: the greedy committer scores all paths (alts verified) but always
commits the spine path prefix. Decisive S3/m1 A/B vs the chain-only boot.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_TEXT = PATCHER_PATH.read_text()
KERNEL_PATH = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
KERNEL_TEXT = KERNEL_PATH.read_text()
LAUNCHER_PATH = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _extract_function(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    ends = [
        text.find("\ndef ", start + 1),
        text.find("\n'''", start + 1),
        text.find("\n@", start + 1),
    ]
    end = min(e for e in ends if e != -1)
    return text[start:end]


def _load_gather_helper():
    # The kernel module imports triton at top, which is container-only;
    # extract just the torch-only helper (same harness style as the
    # committer-extraction tests).
    src = _extract_function(KERNEL_TEXT, "gather_committed_path_conv_prior")
    ns: dict = {"torch": torch}
    exec(src, ns)
    return ns["gather_committed_path_conv_prior"]


# Live 9-draft caterpillar (FR13 acceptance-ladder bind): committer/draft ids
# 0..8 with parents [-1,0,0,1,1,3,3,5,5]; spine drafts [0,1,3,5,7], alt
# leaves [2,4,6,8]. GDN node ids = draft id + 1, root = 0:
COMMITTER_PARENTS = [-1, 0, 0, 1, 1, 3, 3, 5, 5]
GDN_PARENTS = [-1] + [p + 1 for p in COMMITTER_PARENTS]
TREE_N = len(GDN_PARENTS)  # 10 rows per verify forward (root + 9 drafts)
WIDTH = 4  # conv kernel width -> window of WIDTH-1 prior taps
STATE_LEN = 8  # conv_state columns (native rolled-tail bank size)
DIM = 6


def _gdn_root_path(node: int) -> list[int]:
    path = []
    cur = node
    while cur >= 0:
        path.append(cur)
        cur = GDN_PARENTS[cur]
    path.reverse()
    return path


def _node_window_row(prior: torch.Tensor, x: torch.Tensor, node: int) -> torch.Tensor:
    """Mirror the patch-string per-node conv write-back exactly.

    stored row = rows [len(path) : len(path)+STATE_LEN] of
    cat(prior_taps (WIDTH-1, dim), x[path], zeros(STATE_LEN, dim)), transposed
    to (dim, STATE_LEN) — i.e. compact-head: cols [0..WIDTH-2] hold the last
    WIDTH-1 taps of (prior ++ root-path tokens), the rest are zeros.
    """
    path = _gdn_root_path(node)
    source = torch.cat(
        (
            prior.transpose(0, 1),
            x.index_select(0, torch.tensor(path, dtype=torch.long)),
            x.new_zeros((STATE_LEN, x.size(1))),
        ),
        dim=0,
    )
    idx = len(path) + torch.arange(STATE_LEN, dtype=torch.long)
    return source.index_select(0, idx).transpose(0, 1)


def _build_bank(seed: int = 1313):
    torch.manual_seed(seed)
    prior = torch.randn(DIM, WIDTH - 1, dtype=torch.float32)
    x = torch.randn(TREE_N, DIM, dtype=torch.float32)
    # Node-indexed layout: spec_state_indices[0, node] = bank row for node,
    # deliberately non-contiguous/non-monotone bank rows.
    ssi = torch.tensor(
        [[17, 3, 41, 9, 28, 55, 7, 33, 21, 46]], dtype=torch.int32
    )
    conv_state = torch.randn(64, DIM, STATE_LEN, dtype=torch.float32)
    for node in range(TREE_N):
        conv_state[int(ssi[0, node])] = _node_window_row(prior, x, node)
    return prior, x, ssi, conv_state


def _paths_buffer(gdn_path: list[int], cols: int = TREE_N) -> torch.Tensor:
    row = [0] * cols
    for i, node in enumerate(gdn_path):
        row[i] = int(node)
    return torch.tensor([row], dtype=torch.int32)


def _replay_window(prior: torch.Tensor, x: torch.Tensor, committed_gdn_nodes: list[int]) -> torch.Tensor:
    """Pure committed-token replay: last WIDTH-1 taps of (prior ++ committed)."""
    seq = torch.cat(
        (
            prior.transpose(0, 1),
            x.index_select(0, torch.tensor(committed_gdn_nodes, dtype=torch.long)),
        ),
        dim=0,
    )
    return seq[-(WIDTH - 1):].transpose(0, 1)  # (dim, WIDTH-1)


def _legacy_read(conv_state, ssi, gdn_path, acc):
    """Legacy semantics: exact gather-then-scatter remap (FR13_TREE_REMAP_SEQ)
    then the post-remap linear-column read at accepted_len - 1."""
    state = conv_state.clone()
    srcs = [conv_state[int(ssi[0, int(gdn_path[k])])].clone() for k in range(acc)]
    for k in range(acc):
        state[int(ssi[0, k])] = srcs[k]
    read_col = max(0, acc - 1)
    return state[int(ssi[0, read_col])]


def test_branch_winner_window_matches_committed_token_replay() -> None:
    gather = _load_gather_helper()
    prior, x, ssi, conv_state = _build_bank()

    # Branch winners: draft [0,2] (alt leaf, gdn [1,3]) and draft [0,1,4]
    # fully accepted (gdn [1,2,5]) — the S2 trigger class.
    for gdn_path in ([1, 3], [1, 2, 5], [1, 2]):
        acc = len(gdn_path)
        cols, rows, bank = gather(
            conv_state=conv_state,
            spec_state_indices=ssi,
            accepted_paths=_paths_buffer(gdn_path),
            num_accepted_tokens=torch.tensor([acc], dtype=torch.int32),
            num_spec_decodes=1,
        )
        leaf = gdn_path[-1]
        assert cols.tolist() == [[leaf]]
        assert rows.tolist() == [[int(ssi[0, leaf])]]
        committed = [0] + gdn_path  # root token + accepted draft tokens
        replay = _replay_window(prior, x, committed)
        # Window taps == pure committed-token replay; remaining cols zero.
        assert torch.equal(bank[0][:, : WIDTH - 1], replay)
        assert torch.equal(
            bank[0][:, WIDTH - 1 :],
            torch.zeros(DIM, STATE_LEN - (WIDTH - 1)),
        )


def test_zero_accept_reads_root_node_window() -> None:
    gather = _load_gather_helper()
    prior, x, ssi, conv_state = _build_bank()
    # Stale path buffer must not redirect the acc=0 read away from node 0.
    cols, rows, bank = gather(
        conv_state=conv_state,
        spec_state_indices=ssi,
        accepted_paths=_paths_buffer([8, 9]),
        num_accepted_tokens=torch.tensor([0], dtype=torch.int32),
        num_spec_decodes=1,
    )
    assert cols.tolist() == [[0]]
    assert rows.tolist() == [[int(ssi[0, 0])]]
    assert torch.equal(bank[0][:, : WIDTH - 1], _replay_window(prior, x, [0]))


def test_spine_winner_byte_identical_to_legacy_read() -> None:
    gather = _load_gather_helper()
    _, _, ssi, conv_state = _build_bank()
    spine_gdn = [1, 2, 4, 6, 8]  # spine drafts [0,1,3,5,7] + 1
    for acc in range(0, len(spine_gdn) + 1):
        gdn_path = spine_gdn[:acc]
        cols, rows, bank = gather(
            conv_state=conv_state,
            spec_state_indices=ssi,
            accepted_paths=_paths_buffer(gdn_path if gdn_path else [0]),
            num_accepted_tokens=torch.tensor([acc], dtype=torch.int32),
            num_spec_decodes=1,
        )
        legacy = _legacy_read(conv_state, ssi, gdn_path if gdn_path else [0], acc)
        # The semantics-preserving license: spine winners produce the SAME
        # bytes as the legacy post-remap linear read.
        assert torch.equal(bank[0], legacy), f"spine acc={acc} not byte-identical"


def test_branch_winner_equals_legacy_only_under_exact_remap() -> None:
    # Under an EXACT remap the two reads coincide for branch paths too (the
    # fix removes the dependence on the in-place remap, it does not change
    # the intended value). Documented here so any future legacy/committed
    # divergence on live captures localizes to the remap/aliasing machinery.
    gather = _load_gather_helper()
    _, _, ssi, conv_state = _build_bank()
    for gdn_path in ([1, 3], [1, 2, 5]):
        acc = len(gdn_path)
        _, _, bank = gather(
            conv_state=conv_state,
            spec_state_indices=ssi,
            accepted_paths=_paths_buffer(gdn_path),
            num_accepted_tokens=torch.tensor([acc], dtype=torch.int32),
            num_spec_decodes=1,
        )
        assert torch.equal(bank[0], _legacy_read(conv_state, ssi, gdn_path, acc))


def test_gather_is_batched_and_clamped() -> None:
    gather = _load_gather_helper()
    _, _, ssi, conv_state = _build_bank()
    ssi2 = torch.cat((ssi, ssi + 1), dim=0)
    conv_state2 = torch.randn(64, DIM, STATE_LEN, dtype=torch.float32)
    paths = torch.cat(
        (_paths_buffer([1, 3]), _paths_buffer([1, 2, 5])), dim=0
    )
    lens = torch.tensor([2, 99], dtype=torch.int32)  # over-length clamps
    cols, rows, bank = gather(
        conv_state=conv_state2,
        spec_state_indices=ssi2,
        accepted_paths=paths,
        num_accepted_tokens=lens,
        num_spec_decodes=2,
    )
    assert cols.shape == (2, 1) and rows.shape == (2, 1)
    assert cols[0].item() == 3  # leaf of [1,3]
    assert cols[1].item() == 0  # clamped len-1 -> last path col (0-padded)
    assert torch.equal(bank[0], conv_state2[int(ssi2[0, 3])])
    assert torch.equal(bank[1], conv_state2[int(ssi2[1, 0])])


# ---------------------------------------------------------------------------
# Patch-text wiring: flag default ON, gather BEFORE the in-place remap,
# legacy branch intact for FR13_CONV_COMMITTED_PATH=0.
# ---------------------------------------------------------------------------

def test_patch_text_conv_committed_path_wiring() -> None:
    text = PATCHER_TEXT
    assert 'os.environ.get("FR13_CONV_COMMITTED_PATH", "1") == "1"' in text
    assert "gather_committed_path_conv_prior, launch_tree_gdn_prepared" in text
    gather_call = text.index(") = gather_committed_path_conv_prior(")
    remap_call = text.index("launch_tree_state_linear_remap(\n                        ssm_state=ssm_state,")
    # The committed-path snapshot MUST happen before the in-place remap
    # mutates the bank (node-indexed layout is destroyed for cols < acc).
    assert gather_call < remap_call
    # Legacy read intact behind the flag (revertibility license).
    assert "_fr10_conv_read_cols = torch.clamp(" in text
    assert '"compact_head"' in text
    assert '"committed_path_node"' in text
    assert '"native_tail_pre_remap"' in text
    assert '"prior_read_mode": _fr10_prior_read_mode,' in text
    # The ssm/h0 handoff machinery is untouched (m1 seam 2 belongs to the
    # replay route): the remap still covers ssm_state and the h0 read keeps
    # the accepted-column convention.
    assert "h0_num_accepted_tokens=_fr10_accepted_lens_tensor" in text
    assert "h0_use_accepted_column=True" in text


def test_kernel_helper_docstring_states_contract() -> None:
    src = _extract_function(KERNEL_TEXT, "gather_committed_path_conv_prior")
    assert "BEFORE" in src and "launch_tree_state_linear_remap" in src
    assert "BRANCH" in src
    assert "byte-identical" in src


# ---------------------------------------------------------------------------
# FR13_FORCE_SPINE_COMMIT — greedy committer diagnostic flag.
# ---------------------------------------------------------------------------

def _install_gdn_stub(monkeypatch, *, row_req_ids):
    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.mamba",
    ]
    for name in names:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    gdn = types.ModuleType("vllm.model_executor.layers.mamba.gdn_linear_attn")
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR = torch.zeros((8, 10), dtype=torch.int32)
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR = torch.zeros((8,), dtype=torch.int32)
    if row_req_ids is not None:
        gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS = list(row_req_ids)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.gdn_linear_attn", gdn
    )
    setattr(sys.modules["vllm.model_executor.layers.mamba"], "gdn_linear_attn", gdn)
    return gdn


def _greedy_namespace() -> dict:
    ns: dict = {"torch": torch}
    exec(_extract_function(PATCHER_TEXT, "_lumo_tree_path_lcp_max_greedy_sample"), ns)
    return ns


DRAFTS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
SELF_TARGETS = [100, 101, 102, 103, 104, 105, 106, 107, 108]
NATIVE_BONUS = 5555
REJ = 999


def _run_greedy(ns, parent_targets):
    output_token_ids = torch.full((1, 10), -1, dtype=torch.long)
    accepted_tree_rows = torch.zeros((1,), dtype=torch.int32)
    out = ns["_lumo_tree_path_lcp_max_greedy_sample"](
        output_token_ids,
        accepted_tree_rows,
        [len(COMMITTER_PARENTS)],
        torch.tensor(DRAFTS),
        torch.tensor(COMMITTER_PARENTS),
        torch.tensor(parent_targets),
        torch.tensor(SELF_TARGETS),
        torch.tensor([[NATIVE_BONUS]]),
        9,
    )
    row = [int(x) for x in out[0].tolist()]
    return [x for x in row if x != -1]


def _alt_winner_targets():
    # Root + d1-alt fully accepted ([0,2], lcp 2); spine rejected at d1.
    pt = [REJ] * 9
    pt[0] = DRAFTS[0]
    pt[2] = DRAFTS[2]
    return pt


def _full_spine_targets():
    pt = [REJ] * 9
    for node in (0, 1, 3, 5, 7):
        pt[node] = DRAFTS[node]
    return pt


def test_force_spine_commit_commits_spine_prefix_alts_still_scored(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_BONUS_SELF", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    monkeypatch.setenv("FR13_FORCE_SPINE_COMMIT", "1")
    log_path = tmp_path / "tree_path_lcp.jsonl"
    monkeypatch.setenv("LUMO_TREE_PATH_LCP_LOG", str(log_path))
    ns = _greedy_namespace()
    gdn = _install_gdn_stub(monkeypatch, row_req_ids=["req-fs"])

    served = _run_greedy(ns, _alt_winner_targets())

    # The [0,2] alt path is fully accepted (lcp 2) but must NOT win: the
    # forced commit is the spine prefix (spine lcp 1, rejected at d1), so the
    # bonus is the spine d1 reject row's parent target.
    assert served == [DRAFTS[0], REJ]
    # Published state advance follows the forced spine prefix (gdn ids).
    assert gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0, :2].tolist() == [1, 0]
    assert int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0]) == 1
    assert gdn._LUMO_FA_TREE_ACCEPT_BY_REQ["req-fs"] == ([1], 1)

    rows = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    row = next(r for r in rows if r.get("event") == "tree_path_lcp_max")
    assert row["forced_spine_commit"] is True
    assert row["winner_path"] == [0, 1, 3, 5, 7]
    assert row["accepted_len"] == 1
    # Alts remain VERIFIED: the [0,2] path is still scored with its lcp 2.
    alt_score = next(s for s in row["path_scores"] if s["path"] == [0, 2])
    assert alt_score["lcp"] == 2


def test_force_spine_commit_full_spine_matches_unforced(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_BONUS_SELF", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    expected = [
        DRAFTS[0], DRAFTS[1], DRAFTS[3], DRAFTS[5], DRAFTS[7], SELF_TARGETS[7]
    ]
    for flag in ("1", "0"):
        monkeypatch.setenv("FR13_FORCE_SPINE_COMMIT", flag)
        ns = _greedy_namespace()
        _install_gdn_stub(monkeypatch, row_req_ids=["req-fs"])
        # When the spine wins anyway, forcing it is a no-op.
        assert _run_greedy(ns, _full_spine_targets()) == expected


def test_force_spine_flag_off_default_legacy_winner_intact(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_BONUS_SELF", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    monkeypatch.delenv("FR13_FORCE_SPINE_COMMIT", raising=False)  # default OFF
    ns = _greedy_namespace()
    gdn = _install_gdn_stub(monkeypatch, row_req_ids=["req-fs"])

    served = _run_greedy(ns, _alt_winner_targets())

    # Default behavior unchanged: the [0,2] alt winner commits + serves the
    # accepted leaf's self-target (S1 semantics).
    assert served == [DRAFTS[0], DRAFTS[2], SELF_TARGETS[2]]
    assert int(gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR[0]) == 2
    assert gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR[0, :2].tolist() == [1, 3]


def test_sampled_committer_fails_loud_on_force_spine_flag(monkeypatch) -> None:
    # Diagnostic-only flag: the sampled committer must refuse to run rather
    # than silently produce a non-forced (or wrongly-forced) t>0 stream.
    monkeypatch.setenv("FR13_FORCE_SPINE_COMMIT", "1")
    ns: dict = {"torch": torch}
    exec(_extract_function(PATCHER_TEXT, "_lumo_tree_canonical_multidraft_sample"), ns)
    with pytest.raises(RuntimeError, match="FR13_FORCE_SPINE_COMMIT"):
        ns["_lumo_tree_canonical_multidraft_sample"](
            None, None, [0], None, None, None, None, None, None, 0
        )


def test_committer_text_marks_flag_diagnostic_only() -> None:
    helper = _extract_function(PATCHER_TEXT, "_lumo_tree_path_lcp_max_greedy_sample")
    assert "FR13_FORCE_SPINE_COMMIT" in helper
    assert "DIAGNOSTIC ONLY" in helper
    assert "NEVER bind" in helper
    assert "'0') == '1'" in helper.split("FR13_FORCE_SPINE_COMMIT', ")[1][:12]


def test_launcher_forwards_flags_with_diagnostic_warning() -> None:
    launcher = LAUNCHER_PATH.read_text()
    assert "FR13_CONV_COMMITTED_PATH=${FR13_CONV_COMMITTED_PATH:-1}" in launcher
    assert "FR13_FORCE_SPINE_COMMIT=${FR13_FORCE_SPINE_COMMIT:-0}" in launcher
    assert '-e FR13_CONV_COMMITTED_PATH="$FR13_CONV_COMMITTED_PATH"' in launcher
    assert '-e FR13_FORCE_SPINE_COMMIT="$FR13_FORCE_SPINE_COMMIT"' in launcher
    # Never-bind-as-serving warning must live next to the flag default.
    assert "DIAGNOSTIC ONLY" in launcher
    assert "NEVER" in launcher


# ---------------------------------------------------------------------------
# Injected-fragment syntax gates.
# ---------------------------------------------------------------------------

def _extract_string_assign(func_name: str, var_name: str) -> str:
    """Pull the literal value of a string assignment inside a patcher func."""
    tree = ast.parse(PATCHER_TEXT)
    func = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == func_name
    )
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == var_name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError(f"{var_name} string assignment not found in {func_name}")


def test_conv_replacement_fragment_is_valid_python_and_ordered() -> None:
    fragment = _extract_string_assign("_patch_gdn_linear", "conv_replacement")
    # The fragment sits at method-body depth (8 spaces); scaffold two levels.
    compile("def _f(self):\n    if True:\n" + fragment, "<conv_replacement>", "exec")
    assert 'os.environ.get("FR13_CONV_COMMITTED_PATH", "1") == "1"' in fragment
    gather_call = fragment.index(") = gather_committed_path_conv_prior(")
    remap_call = fragment.index("launch_tree_state_linear_remap(")
    assert gather_call < remap_call
    # Legacy read intact (flag-off revert path) + truthful capture mode.
    assert "_fr10_conv_read_cols = torch.clamp(" in fragment
    assert '"committed_path_node"' in fragment
    assert '"compact_head"' in fragment


# ---------------------------------------------------------------------------
# Pristine-tree compile gate (rejection sampler patch target). The pristine
# gdn_linear_attn snapshot predates the live container needles (HEAD's
# _patch_gdn_linear does not apply to it either), so the conv fragment is
# covered by the scaffold-compile gate above instead.
# ---------------------------------------------------------------------------

PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")
def test_patched_rejection_sampler_compiles_with_force_spine(tmp_path) -> None:
    import importlib.util
    import py_compile

    spec = importlib.util.spec_from_file_location("fr10_phase4_patcher", PATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sampler_target = tmp_path / "rejection_sampler.py"
    sampler_target.write_text(
        (PRISTINE / "v1" / "sample" / "rejection_sampler.py").read_text()
    )
    module.REJECTION_SAMPLER_PATH = sampler_target
    assert module._patch_rejection_sampler_tree_lcp() is True
    py_compile.compile(str(sampler_target), doraise=True)
    sampler_text = sampler_target.read_text()
    assert "FR13_FORCE_SPINE_COMMIT" in sampler_text
    assert "forced_spine_commit" in sampler_text
    assert "diagnostic-only for the GREEDY" in sampler_text
