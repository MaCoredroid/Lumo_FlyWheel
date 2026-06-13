"""FR13_COMMIT_ARGMAX_GATE wiring gate (in-process committer-row argmax gate).

ONE diagnostic flag, default OFF, inert; the FIX1-SELFCHECK / FIX-2 / chase-diag
wiring pattern (playbook class 9): text-level asserts that the flag exists
end-to-end with default OFF, an engagement needle in BOTH states, launcher
docker -e passthrough, an eager-only fail-loud where the tap syncs over the
captured forward's logits, and that the injected committer fragment + the
call-site publish fragment still parse. When the pristine vLLM 0.19 tree is
present, the rejection-sampler patch is applied end-to-end and the patched file
py_compiles.

Instrument under test (FR13_B1_SWE_GOLD_BIND.md localization):
  - call site (sample method): publishes the RAW post-constraint verify logit
    tensors (target_logits = parent-edge dist; lumo_tree_self_logits = self/node
    dist) to module globals ONLY when armed; OFF leaves them None.
  - greedy committer (_lumo_tree_path_lcp_max_greedy_sample): for each
    committed/served token, dumps to jsonl the verify logit row it ACTUALLY
    indexed and:
      CHANNEL 1 (committer row-mapping): committed_token_id vs
        argmax(verify_logits[committed_row]) + ch1_margin.
      CHANNEL 2 (verify-forward losslessness): verify argmax id + top-2 margin
        for the reduce's clean-forward comparison.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"

TEXT = PATCHER.read_text()
LAUNCHER_TEXT = LAUNCHER.read_text()

PRISTINE = Path("/tmp/vllm_pristine_019/extracted/vllm")


def _helper_snippet() -> str:
    """The committer helper r''' string injected into rejection_sampler."""
    start = TEXT.index("    helper = r'''")
    body_start = start + len("    helper = r'''")
    return TEXT[body_start:TEXT.index("'''", body_start)]


def _stock_call_new_snippet() -> str:
    """The patched `sample` method body (contains the call-site publish)."""
    anchor = (
        'stock_call_new = """        lumo_tree_parent_indices = '
        'getattr(metadata'
    )
    start = TEXT.index(anchor) + len('stock_call_new = """')
    return TEXT[start:TEXT.index('"""', start)]


def test_flag_default_off_everywhere() -> None:
    # Every read defaults to "0"; no site defaults ON; no defaultless reads.
    assert 'environ.get("FR13_COMMIT_ARGMAX_GATE", "0")' in TEXT
    assert "environ.get('FR13_COMMIT_ARGMAX_GATE', '0')" in TEXT
    assert '"FR13_COMMIT_ARGMAX_GATE", "1"' not in TEXT
    assert "'FR13_COMMIT_ARGMAX_GATE', '1'" not in TEXT
    assert 'environ.get("FR13_COMMIT_ARGMAX_GATE")' not in TEXT
    assert "environ.get('FR13_COMMIT_ARGMAX_GATE')" not in TEXT


def test_launcher_default_off_and_passthrough() -> None:
    assert (
        "FR13_COMMIT_ARGMAX_GATE=${FR13_COMMIT_ARGMAX_GATE:-0}"
        in LAUNCHER_TEXT
    )
    assert (
        'FR13_COMMIT_ARGMAX_GATE_DUMP=${FR13_COMMIT_ARGMAX_GATE_DUMP:-'
        in LAUNCHER_TEXT
    )
    assert (
        '-e FR13_COMMIT_ARGMAX_GATE="$FR13_COMMIT_ARGMAX_GATE" \\'
        in LAUNCHER_TEXT
    )
    assert (
        '-e FR13_COMMIT_ARGMAX_GATE_DUMP="$FR13_COMMIT_ARGMAX_GATE_DUMP" \\'
        in LAUNCHER_TEXT
    )
    assert "FR13_COMMIT_ARGMAX_GATE:-1" not in LAUNCHER_TEXT
    # Diagnostic discipline written next to the flag.
    assert "NEVER bind =1 into a serving config" in LAUNCHER_TEXT


def test_needle_fires_in_both_states() -> None:
    helper = _helper_snippet()
    # Single needle helper, called unconditionally (BOTH states) before the
    # gate body; message records armed=0/1.
    assert "def _fr13_commit_argmax_gate_needle(armed):" in helper
    assert (
        "FR13_COMMIT_ARGMAX_GATE committer-row gate: armed=%d (%s)" in helper
    )
    assert "'armed-eager-only' if armed else 'inert'" in helper
    # The needle is invoked once per committer call, OUTSIDE the active gate.
    needle_call = helper.index("_fr13_commit_argmax_gate_needle(_fr13_cag_active)")
    gate_body = helper.index("if _fr13_cag_active and row:")
    assert needle_call < gate_body


def test_call_site_publishes_only_when_armed() -> None:
    body = _stock_call_new_snippet()
    # ON publishes the raw logit tensors; OFF sets them None (inert committer).
    arm_idx = body.index(
        'if _fr13_cag_os.environ.get("FR13_COMMIT_ARGMAX_GATE", "0") == "1":'
    )
    pub_idx = body.index('globals()["_FR13_CAG_TARGET_LOGITS"] = target_logits')
    self_idx = body.index(
        'globals()["_FR13_CAG_SELF_LOGITS"] = lumo_tree_self_logits'
    )
    off_idx = body.index('globals()["_FR13_CAG_TARGET_LOGITS"] = None')
    assert arm_idx < pub_idx < self_idx < off_idx
    # Publish happens BEFORE the committer call so the committer can read it.
    assert pub_idx < body.index("output_token_ids = rejection_sample(")


def test_committer_gate_active_requires_published_tensor() -> None:
    helper = _helper_snippet()
    # Active iff flag ON AND the call-site actually published (OFF call-site
    # can never arm a captured boot).
    assert (
        "_FR13_COMMIT_ARGMAX_GATE and _fr13_cag_target_logits is not None"
        in helper
    )


def test_eager_only_fail_loud_before_sync() -> None:
    helper = _helper_snippet()
    # The row-stats helper fail-louds on CUDA-graph capture BEFORE topk/.item.
    rs = helper[helper.index("def _fr13_commit_argmax_gate_row_stats("):]
    rs = rs[: rs.index("\n\n\n")] if "\n\n\n" in rs else rs
    guard = rs.index("is_current_stream_capturing()")
    topk = rs.index("torch.topk(")
    assert guard < topk
    assert "FR13_COMMIT_ARGMAX_GATE is eager-only" in rs


def test_channel1_fields_present() -> None:
    helper = _helper_snippet()
    # Channel 1 = committed_id vs argmax(verify_logits[the row USED]).
    for needle in (
        "'committed_token_id': _cag_tok,",
        "'committed_row': int(_cag_flat),",
        "'verify_argmax_id': int(_cag_amax_id),",
        "'committed_token_logit': float(_cag_tok_lp),",
        "'ch1_margin': float(_cag_amax_lp - _cag_tok_lp),",
        "_cag_ch1_match = bool(_cag_tok == _cag_amax_id)",
        "'ch1_match': _cag_ch1_match,",
    ):
        assert needle in helper, needle


def test_channel2_split_fields_present() -> None:
    helper = _helper_snippet()
    # Channel 2 = verify argmax + top-2 margin for the clean-forward reduce.
    for needle in (
        "'verify_runnerup_id': int(_cag_ru_id),",
        "'verify_top2_margin': float(",
    ):
        assert needle in helper, needle


def test_capture_metadata_names_the_seam() -> None:
    helper = _helper_snippet()
    # The capture must carry the seam-naming metadata (so the reduce can label
    # which boundary/row class the divergence lands on).
    for needle in (
        "'num_accepted_this_step': int(best_lcp),",
        "'node_type': _cag_ntype,",
        "'accept_run_index': int(_cag_run_idx),",
        "'bonus_source': str(bonus_source),",
        "'logit_kind': _cag_logit_kind,",
        "'step': int(_fr13_cag_step),",
    ):
        assert needle in helper, needle


def test_row_map_covers_all_bonus_sources() -> None:
    helper = _helper_snippet()
    # The committer-row derivation must handle every bonus_source the committer
    # can emit (class 5: re-derived from this committer's own branches).
    for needle in (
        "if bonus_source == 'reject_parent_target':",
        "elif bonus_source == 'root_parent_target':",
        "elif bonus_source == 'tree_self_target':",
        "elif bonus_source == 'path0_native_bonus':",
    ):
        assert needle in helper, needle
    # The legacy native-bonus path indexed no per-node row -> row_known False.
    assert "_cag_logit_kind = None  # no per-node row indexed" in helper
    # Accepted prefix uses the parent-edge target dist; bonus_self uses the
    # node's own self dist.
    assert "_cag_logit_kind = 'target'" in helper
    assert "_cag_logit_kind = 'self'" in helper


def test_fragments_still_parse() -> None:
    helper = _helper_snippet()
    compile(helper, "<fr13_cag_helper>", "exec")
    body = _stock_call_new_snippet()
    src = (
        "def _fr13_cag_probe(self, metadata, logits, target_logits,"
        " draft_probs, bonus_token_ids, sampling_metadata,"
        " draft_token_ids):\n"
        + "\n".join(
            ("    " + ln) if ln.strip() else ln for ln in body.splitlines()
        )
    )
    compile(src, "<fr13_cag_stock_call_new>", "exec")


def _load_patcher_module():
    spec = importlib.util.spec_from_file_location(
        "fr10_phase4_patcher_cag", PATCHER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not PRISTINE.exists(), reason="pristine vLLM tree not present"
)
def test_committer_patch_applies_to_pristine_and_compiles(tmp_path) -> None:
    module = _load_patcher_module()
    rs_target = tmp_path / "rejection_sampler.py"
    rs_target.write_text(
        (PRISTINE / "v1" / "sample" / "rejection_sampler.py").read_text()
    )
    module.REJECTION_SAMPLER_PATH = rs_target
    assert module._patch_rejection_sampler_tree_lcp() is True
    patched = rs_target.read_text()
    for needle in (
        "_FR13_COMMIT_ARGMAX_GATE = (",
        "def _fr13_commit_argmax_gate_needle(",
        "def _fr13_commit_argmax_gate_row_stats(",
        '"FR13_COMMIT_ARGMAX_GATE", "0"',
        "_FR13_CAG_TARGET_LOGITS",
        "'ch1_match'",
        "'verify_top2_margin'",
        "FR13_COMMIT_ARGMAX_GATE is eager-only",
    ):
        assert needle in patched, needle
    py_compile.compile(str(rs_target), doraise=True)
