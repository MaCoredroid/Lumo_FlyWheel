"""FR13 S1: committer bonus row = accepted leaf self-target.

The greedy tree committer enumerated leaves in NODE order, so path_idx 0 is
the SHALLOW ALT leaf ([0, 2] in the 9-node caterpillar), not the spine
([0, 1, 3, 5, 7] = path_idx 3). The 'path0_native_bonus' special case
therefore fired on [0, 2]-winners and committed vLLM's precomputed
bonus_token_ids — sampled from the LAST verify row (node 8) — instead of the
accepted leaf's self-target row st[2]. FR13 acceptance-ladder bind: 14/163
greedy events served a wrong token (8.6%); prompt-3's fork at pos 14.

Fix (FR13_TREE_BONUS_SELF, default ON): fully-accepted paths ALWAYS commit
the accepted leaf's self-target row — at greedy that row IS the verify argmax
bonus for every path, including the spine. =0 restores the legacy behavior.
Also: path0_lcp / superset_violation now use the TRUE spine path index
(first-child chain from the root) instead of path_scores[0].
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_TEXT = PATCHER_PATH.read_text()


@pytest.fixture(autouse=True)
def _legacy_route_escape_hatch(monkeypatch):
    # FR13_REPLAY_ROUTE now defaults ON; these CPU tests exec the committer
    # fragments, whose replay branch imports the triton kernel and requires
    # registered GDN replay layers (GPU/serving-only). Pin the explicit
    # legacy escape hatch =0 -- the bonus-row logic under test is
    # route-independent; default-ON wiring is asserted in
    # test_fr13_replay_route_wiring.py.
    monkeypatch.setenv("FR13_REPLAY_ROUTE", "0")


def _extract_function(name: str) -> str:
    start = PATCHER_TEXT.index(f"def {name}(")
    ends = [
        PATCHER_TEXT.find("\ndef ", start + 1),
        PATCHER_TEXT.find("\n'''", start + 1),
    ]
    end = min(e for e in ends if e != -1)
    return PATCHER_TEXT[start:end]


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
    gdn._LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR = torch.zeros((8, 6), dtype=torch.int32)
    gdn._LUMO_FA_ACCEPTED_TREE_LENS_TENSOR = torch.zeros((8,), dtype=torch.int32)
    if row_req_ids is not None:
        gdn._LUMO_FA_SAMPLER_ROW_REQ_IDS = list(row_req_ids)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.gdn_linear_attn", gdn
    )
    setattr(sys.modules["vllm.model_executor.layers.mamba"], "gdn_linear_attn", gdn)
    return gdn


def _greedy_namespace() -> dict:
    ns: dict = {
        "torch": torch,
        # FR13_EAGER_PACK (FIX-2) module-globals: these CPU rigs exec the
        # bare committer function; in the injected module the flag/needle
        # are defined at module scope. Flag OFF = exact legacy transports
        # (the value logic under test is transport-independent).
        "_FR13_EAGER_PACK": False,
        "_fr13_eager_pack_needle": lambda *a, **k: None,
    }
    exec(_extract_function("_lumo_tree_path_lcp_max_greedy_sample"), ns)
    return ns


# 9-node caterpillar (committer/BFS node ids, FR13 acceptance-ladder bind):
# n0=d0; (n1 spine, n2 alt)=d1; (n3,n4)=d2; (n5,n6)=d3; (n7,n8)=d4.
# Leaves enumerate in node order = [2, 4, 6, 7, 8], so the paths are
# [0,2] [0,1,4] [0,1,3,6] [0,1,3,5,7] [0,1,3,5,8]; the spine is path_idx 3.
CATERPILLAR_PARENTS = [-1, 0, 0, 1, 1, 3, 3, 5, 5]
DRAFTS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
SELF_TARGETS = [100, 101, 102, 103, 104, 105, 106, 107, 108]
NATIVE_BONUS = 5555
REJ = 999  # parent-target that rejects the node's draft


def _run_greedy(ns, parent_targets):
    output_token_ids = torch.full((1, 10), -1, dtype=torch.long)
    accepted_tree_rows = torch.zeros((1,), dtype=torch.int32)
    out = ns["_lumo_tree_path_lcp_max_greedy_sample"](
        output_token_ids,
        accepted_tree_rows,
        [len(CATERPILLAR_PARENTS)],
        torch.tensor(DRAFTS),
        torch.tensor(CATERPILLAR_PARENTS),
        torch.tensor(parent_targets),
        torch.tensor(SELF_TARGETS),
        torch.tensor([[NATIVE_BONUS]]),
        9,
    )
    row = [int(x) for x in out[0].tolist()]
    return [x for x in row if x != -1]


def _alt_winner_targets():
    # Root + d1-alt accepted ([0, 2] fully accepted, lcp 2); spine rejected
    # at d1, so every other path stops at lcp 1.
    pt = [REJ] * 9
    pt[0] = DRAFTS[0]
    pt[2] = DRAFTS[2]
    return pt


def _full_spine_targets():
    # Whole spine [0, 1, 3, 5, 7] accepted (lcp 5); alts rejected.
    pt = [REJ] * 9
    for node in (0, 1, 3, 5, 7):
        pt[node] = DRAFTS[node]
    return pt


def test_alt_leaf_winner_serves_accepted_leaf_self_target(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_BONUS_SELF", raising=False)  # default ON
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)  # default ON
    ns = _greedy_namespace()
    _install_gdn_stub(monkeypatch, row_req_ids=["req-s1"])

    served = _run_greedy(ns, _alt_winner_targets())

    # The exact S1 bug class: a [0, 2]-winner must serve st[2], NOT the
    # native bonus sampled from node 8's row.
    assert served == [DRAFTS[0], DRAFTS[2], SELF_TARGETS[2]]


def test_full_spine_winner_serves_spine_leaf_self_target(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    expected = [
        DRAFTS[0], DRAFTS[1], DRAFTS[3], DRAFTS[5], DRAFTS[7], SELF_TARGETS[7]
    ]
    for flag in ("1", "0"):
        monkeypatch.setenv("FR13_TREE_BONUS_SELF", flag)
        ns = _greedy_namespace()
        _install_gdn_stub(monkeypatch, row_req_ids=["req-s1"])

        # The spine is path_idx 3 (not 0), so legacy and fixed behavior agree
        # here: st[7], the accepted leaf's own verify row.
        assert _run_greedy(ns, _full_spine_targets()) == expected


def test_flag_off_restores_legacy_path0_native_bonus(monkeypatch) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setenv("FR13_TREE_BONUS_SELF", "0")
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    ns = _greedy_namespace()
    _install_gdn_stub(monkeypatch, row_req_ids=["req-s1"])

    served = _run_greedy(ns, _alt_winner_targets())

    # Legacy text: the wrong-row native bonus (node 8's sampled token).
    assert served == [DRAFTS[0], DRAFTS[2], NATIVE_BONUS]


def test_log_uses_true_spine_path_index(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FR10_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.delenv("FR13_TREE_BONUS_SELF", raising=False)
    monkeypatch.delenv("FR13_TREE_REQKEY", raising=False)
    log_path = tmp_path / "tree_path_lcp.jsonl"
    monkeypatch.setenv("LUMO_TREE_PATH_LCP_LOG", str(log_path))
    ns = _greedy_namespace()
    _install_gdn_stub(monkeypatch, row_req_ids=["req-s1"])

    _run_greedy(ns, _alt_winner_targets())

    rows = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    row = next(r for r in rows if r.get("event") == "tree_path_lcp_max")
    assert row["winner_path"] == [0, 2]
    assert row["accepted_len"] == 2
    assert row["bonus_source"] == "tree_self_target"
    assert row["emitted_tokens"] == [DRAFTS[0], DRAFTS[2], SELF_TARGETS[2]]
    # TRUE spine diagnostics: spine = [0,1,3,5,7] at path_idx 3, leaf 7,
    # lcp 1 (rejected at d1) — NOT path_scores[0] (= [0,2], lcp 2).
    assert row["spine_path_idx"] == 3
    assert row["spine_leaf"] == 7
    assert row["path0_lcp"] == 1
    assert row["superset_violation"] is False
    assert row["superset_delta"] == 1


def test_sampled_committer_has_no_wrong_row_bonus_reuse() -> None:
    # S1 audit pin: the sampled committer draws a full-accept bonus from the
    # accepted leaf's OWN self-probs row; bonus_token_ids only appears in the
    # degenerate no-tree-nodes branch (where it is the correct, only row).
    sampled = _extract_function("_lumo_tree_canonical_multidraft_sample")
    assert "path0_native_bonus" not in sampled
    assert "self_probs_cpu[start + current_parent]" in sampled
    assert "elif req_i < int(bonus_token_ids.numel()):" in sampled


def test_launcher_forwards_bonus_self_flag() -> None:
    launcher = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text()
    assert '-e FR13_TREE_BONUS_SELF="${FR13_TREE_BONUS_SELF:-1}"' in launcher


PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")


@pytest.mark.skipif(not PRISTINE.exists(), reason="pristine vLLM tree not present")
def test_patched_rejection_sampler_compiles_with_s1_fix(tmp_path) -> None:
    import importlib.util
    import py_compile

    spec = importlib.util.spec_from_file_location("fr10_phase4_patcher", PATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "rejection_sampler.py"
    target.write_text(
        (PRISTINE / "v1" / "sample" / "rejection_sampler.py").read_text()
    )
    module.REJECTION_SAMPLER_PATH = target
    assert module._patch_rejection_sampler_tree_lcp() is True
    py_compile.compile(str(target), doraise=True)
    text = target.read_text()
    assert "FR13_TREE_BONUS_SELF" in text
    assert "spine_path_idx" in text
