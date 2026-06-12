"""FR13_CHASE_DIAG wiring gate (superset-chase Step-1 instruments).

ONE diagnostic flag, default OFF, inert; the FIX1-SELFCHECK / FIX-2 wiring
pattern (playbook class 9): text-level asserts that the flag exists
end-to-end with default OFF, engagement needles in BOTH states, launcher
docker -e passthrough, eager-only fail-louds where the taps touch captured
GDN paths, and that every injected fragment still parses. When the pristine
vLLM tree is present, the gdn/eagle/committer patches are applied end-to-end
and the patched files py_compiled.

Instruments under test (plan wf_c43b084f, chasePlan.ladder STEP 1):
  (i)   per-event INTEGER record {token_indices_to_sample - base, prev
        accepted_len, published leaf flat row} -> fr13_chase_rowtrace.jsonl
        (eagle propose python; capture-safe, syncs => diagnostics-only boot)
  (ii)  state-parity producer/consumer byte verdict: tap A as-written
        (accepted_path + rowbug now stored) joined IN-PROCESS at tap B
        (B_JOIN verdict record) -- playbook class-8 boundary instrument;
        EAGER-ONLY via the shared boundary emit fail-loud
  (iii) drafter root-logit top-K fp32 dump -> fr13_chase_rootlogits.jsonl
  (iv)  drafter-layer KV row hashes -> fr13_chase_drafter_kv.jsonl
  (v)   H6 conv-prior CV tap (as-read row/col/bytes; byte VERDICT stays the
        dual-path FR13_TCF_SELFCHECK=1) -- EAGER-ONLY, explicit fail-loud
"""

from __future__ import annotations

import importlib.util
import py_compile
import textwrap
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


def _eagle_new_snippet() -> str:
    """The caterpillar propose replacement (a plain triple-quote string)."""
    anchor = 'new = """        _fr10_active_decode_mode'
    start = TEXT.index(anchor) + len('new = """')
    return TEXT[start:TEXT.index('"""', start)]


def _conv_replacement_snippet() -> str:
    start = TEXT.index("    conv_replacement = '''")
    body_start = start + len("    conv_replacement = '''")
    return TEXT[body_start:TEXT.index("'''", body_start)]


def test_flag_default_off_everywhere() -> None:
    # gdn module header ("..."-escaped form) + eagle """ sites read with
    # default "0"; no site may default ON; no defaultless reads.
    assert 'os.environ.get(\\"FR13_CHASE_DIAG\\", \\"0\\")' in TEXT
    assert 'os.environ.get("FR13_CHASE_DIAG", "0") == "1"' in TEXT
    assert '"FR13_CHASE_DIAG", "1"' not in TEXT
    assert "'FR13_CHASE_DIAG', '1'" not in TEXT
    assert '\\"FR13_CHASE_DIAG\\", \\"1\\"' not in TEXT
    assert 'environ.get("FR13_CHASE_DIAG")' not in TEXT
    assert "environ.get('FR13_CHASE_DIAG')" not in TEXT
    assert 'environ.get(\\"FR13_CHASE_DIAG\\")' not in TEXT


def test_launcher_default_off_and_passthrough() -> None:
    assert "FR13_CHASE_DIAG=${FR13_CHASE_DIAG:-0}" in LAUNCHER_TEXT
    assert '-e FR13_CHASE_DIAG="$FR13_CHASE_DIAG" \\' in LAUNCHER_TEXT
    assert "FR13_CHASE_DIAG:-1" not in LAUNCHER_TEXT
    for aux in ("FR13_CHASE_DIAG_DIR", "FR13_CHASE_TOPK",
                "FR13_CHASE_KV_WINDOW"):
        assert f'-e {aux}="${aux}" \\' in LAUNCHER_TEXT, aux
    # Diagnostic discipline stays written next to the flag.
    assert "NEVER bind =1 into a serving config" in LAUNCHER_TEXT


def test_needles_fire_in_both_states() -> None:
    # gdn side: import-time print needle, BOTH states.
    assert 'FR13_CHASE_DIAG gdn taps: chase=%s' in TEXT
    assert '\\"armed-eager-only\\" if _fr13_chase_on() else \\"inert\\"' in TEXT
    # eagle side: info_once needle, BOTH states (inside the caterpillar
    # branch, before the flag gate).
    eagle = _eagle_new_snippet()
    needle_idx = eagle.index(
        "FR13_CHASE_DIAG drafter instruments: chase=%s"
    )
    gate_idx = eagle.index("if _fr13_chase_diag:")
    assert needle_idx < gate_idx


def test_gdn_chase_implies_boundary_instrument_eager_only() -> None:
    # _fr13_chase_on is OR'd into _fr13_boundary_on => every tap inherits
    # the emit() CUDA-capture fail-loud (eager-only, class 6/9). The gdn
    # header lives in "..."-escaped patch strings: un-escape + join the
    # concatenated pieces before matching.
    joined = TEXT.replace('\\"', '"').replace('"\n            "', '')
    assert 'def _fr13_chase_on():' in joined
    assert 'or _fr13_chase_on()' in joined
    assert 'FR13_REPLAY_BOUNDARY_LOG is eager-only' in joined
    # The chase needle must precede _fr13_boundary_on (definition order).
    assert joined.index('def _fr13_chase_on():') < joined.index(
        'def _fr13_boundary_on():'
    )
    # Header flag list records the chase flag state in every artifact.
    assert '"FR13_CHASE_DIAG",' in joined


def test_tap_a_stores_path_and_rowbug_for_join() -> None:
    helper = _helper_snippet()
    post = helper[helper.index("def _fr13_boundary_replay_post("):]
    post = post[:post.index("\ndef ")] if "\ndef " in post else post
    assert "'accepted_path': [int(_x) for _x in _path]," in post
    assert "'rowbug': bool(_path) and int(_path[-1]) != _alen," in post


def test_tap_b_join_verdict_in_process() -> None:
    # Instrument (ii): the consumer-side join emits a per-event verdict.
    for needle in (
        '"tap": "B_JOIN"',
        '_fr13_ch_verdict = "NO_PRIOR_WRITE"',
        '_fr13_ch_verdict = "READ_ROW_NOT_WRITTEN"',
        '_fr13_ch_verdict = "BYTE_EQUAL"',
        '_fr13_ch_verdict = "BYTE_DIFF"',
        '"prev_rowbug"',
        '_FR13_BOUNDARY_LAST_WRITTEN_BY_REQ',
    ):
        assert needle in TEXT, needle


def test_conv_cv_tap_eager_only_and_h6_verdict_delegation() -> None:
    conv = _conv_replacement_snippet()
    assert '"tap": "CV"' in conv
    # Explicit fail-loud BEFORE any sync (class 6: never inside capture).
    cv_idx = conv.index('"tap": "CV"')
    guard_idx = conv.index(
        "FR13_CHASE_DIAG conv tap is eager-only"
    )
    assert guard_idx < cv_idx
    assert "torch.cuda.is_current_stream_capturing()" in conv
    # The H6 byte VERDICT remains the dual-path TCF selfcheck (mandatory on
    # the chase boot per the plan-verify caveat), not this raw tap.
    assert "FR13_TCF_SELFCHECK" in conv


def test_eagle_instruments_present_and_consume_reqkey_dict() -> None:
    eagle = _eagle_new_snippet()
    for needle in (
        "fr13_chase_rowtrace.jsonl",
        "fr13_chase_rootlogits.jsonl",
        "fr13_chase_drafter_kv.jsonl",
        "_LUMO_FA_TREE_ACCEPT_BY_REQ",
        "_LUMO_FA_SAMPLER_ROW_REQ_IDS",
        '"sampled_row_offset"',
        '"published_leaf_flat_row"',
        '"h1_row_mismatch"',
        '"prev_accepted_len"',
        'os.environ.get("FR13_CHASE_TOPK", "32")',
        'os.environ.get("FR13_CHASE_KV_WINDOW", "16")',
        "_fr10_logits.float()",
    ):
        assert needle in eagle, needle
    # Integer record before the drafter loop; KV hash after it (this
    # event's writes included), before the return.
    assert eagle.index('"sampled_row_offset"') < eagle.index(
        "for token_index in range(4):"
    )
    assert eagle.index("fr13_chase_drafter_kv.jsonl") > eagle.index(
        "for token_index in range(4):"
    )
    assert eagle.index("fr13_chase_drafter_kv.jsonl") < eagle.index(
        "return _fr10_packed"
    )


def test_fragments_still_parse() -> None:
    helper = _helper_snippet()
    compile(helper, "<fr13_chase_helper>", "exec")
    # The eagle replacement is method-body code at 8-space indentation with
    # return/global statements: wrap in a def for a syntax-only compile.
    eagle = _eagle_new_snippet()
    compile(
        "def _fr13_chase_syntax_probe(self, token_indices_to_sample,"
        " common_attn_metadata, positions, hidden_states, batch_size,"
        " sample_hidden_states, num_rejected_tokens_gpu, slot_mappings,"
        " per_group_attn_metadata):\n" + eagle,
        "<fr13_chase_eagle_block>",
        "exec",
    )
    # conv_replacement (CV tap) and the scan-branch replacement (tap B +
    # B_JOIN) are forward-body code at 8-space indentation.
    conv = _conv_replacement_snippet()
    compile(
        "def _fr13_chase_conv_probe(self, attn_metadata):\n" + conv,
        "<fr13_chase_conv_block>",
        "exec",
    )
    scan_start = TEXT.index("    replacement = '''")
    scan_body = scan_start + len("    replacement = '''")
    scan = TEXT[scan_body:TEXT.index("'''", scan_body)]
    assert '"tap": "B_JOIN"' in scan
    compile(
        "def _fr13_chase_scan_probe(self, attn_metadata):\n" + scan,
        "<fr13_chase_scan_block>",
        "exec",
    )
    # The gdn module-header injection (chase_on + print needle + boundary
    # helpers) is an implicitly-concatenated string literal: the parser
    # folds it into one Constant of module-level code -> compile directly.
    import ast

    header_consts = [
        node.value
        for node in ast.walk(ast.parse(TEXT))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "def _fr13_chase_on():" in node.value
    ]
    assert header_consts, "gdn header constant with _fr13_chase_on not found"
    for const in header_consts:
        compile(const, "<fr13_chase_gdn_header>", "exec")


def _eagle_synthetic_fixture() -> str:
    """A minimal eagle.py carrying the EXACT anchors the eagle patch needs,
    inside a class/method so the patched output py_compiles end-to-end
    (the pristine 0.19 tree predates the cu130 anchors)."""
    tree_parse = (
        "        # Parse the speculative token tree.\n"
        "        spec_token_tree = self.speculative_config.speculative_token_tree\n"
        "        assert spec_token_tree is not None\n"
        "        self.tree_choices: list[tuple[int, ...]] = "
        "ast.literal_eval(spec_token_tree)\n"
    )
    old_start = TEXT.index(
        'old = """        if any(isinstance(md, TreeAttentionMetadata)'
    ) + len('old = """')
    dispatch = TEXT[old_start:TEXT.index('"""', old_start)]
    return (
        "import ast\n"
        "\n"
        "\n"
        "class EagleProposer:\n"
        "    def __init__(self):\n"
        + tree_parse
        + "\n"
        "    def set_inputs_first_pass(self, target_token_ids, next_token_ids,\n"
        "                              target_positions, target_hidden_states,\n"
        "                              token_indices_to_sample, cad,\n"
        "                              num_rejected_tokens_gpu):\n"
        "        return target_token_ids.shape[0], token_indices_to_sample, cad\n"
        "\n"
        "    def propose(self, target_token_ids, target_positions,\n"
        "                target_hidden_states, next_token_ids,\n"
        "                token_indices_to_sample, common_attn_metadata,\n"
        "                sampling_metadata, mm_embed_inputs=None,\n"
        "                num_rejected_tokens_gpu=None, slot_mappings=None):\n"
        "        batch_size = next_token_ids.shape[0]\n"
        "        sample_hidden_states = target_hidden_states\n"
        "        hidden_states = target_hidden_states\n"
        "        positions = target_positions\n"
        "        per_group_attn_metadata = []\n"
        # The FIX-A1 (FR13_TREE_SAMPLE_ROW) injection anchors on this exact
        # call shape (live eagle.py propose); the patch inserts BEFORE it.
        "        num_tokens, token_indices_to_sample, common_attn_metadata = (\n"
        "            self.set_inputs_first_pass(\n"
        "                target_token_ids=target_token_ids,\n"
        "                next_token_ids=next_token_ids,\n"
        "                target_positions=target_positions,\n"
        "                target_hidden_states=target_hidden_states,\n"
        "                token_indices_to_sample=token_indices_to_sample,\n"
        "                cad=common_attn_metadata,\n"
        "                num_rejected_tokens_gpu=num_rejected_tokens_gpu,\n"
        "            )\n"
        "        )\n"
        + dispatch
        + "\n"
        "    def propose_tree(self, batch_size, logits, positions,\n"
        "                     hidden_states, common_attn_metadata,\n"
        "                     slot_mappings=None):\n"
        "        draft_token_ids = logits.argmax(dim=-1)\n"
        "        num_children = 1\n"
        "        draft_token_ids_list = [draft_token_ids]\n"
        "        draft_hidden_states = hidden_states.view(batch_size, 1, -1)\n"
        "        for _level in range(1):\n"
        "            draft_token_ids_list.append(draft_token_ids)\n"
        "        return draft_token_ids_list\n"
    )


def test_eagle_patch_applies_to_synthetic_fixture_and_compiles(
    tmp_path,
) -> None:
    module = _load_patcher_module()
    target = tmp_path / "eagle.py"
    target.write_text(_eagle_synthetic_fixture())
    module.EAGLE_PATH = target
    assert module._patch_eagle_tree_consumption_verify() is True
    patched = target.read_text()
    for needle in (
        "fr13_chase_rowtrace.jsonl",
        "fr13_chase_rootlogits.jsonl",
        "fr13_chase_drafter_kv.jsonl",
        '"h1_row_mismatch"',
    ):
        assert needle in patched, needle
    py_compile.compile(str(target), doraise=True)


def _load_patcher_module():
    spec = importlib.util.spec_from_file_location(
        "fr10_phase4_patcher_chase", PATCHER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not PRISTINE.exists(),
                    reason="pristine vLLM tree not present")
def test_committer_patch_applies_to_pristine_and_compiles(tmp_path) -> None:
    # NOTE: only the rejection-sampler patch anchors on the 0.19 pristine
    # tree; the gdn-linear/eagle patches target the live cu130-nightly
    # source (their generated code is covered by the synthetic fixture +
    # fragment compiles above and by the live boot needles).
    module = _load_patcher_module()
    rs_target = tmp_path / "rejection_sampler.py"
    rs_target.write_text(
        (PRISTINE / "v1" / "sample" / "rejection_sampler.py").read_text()
    )
    module.REJECTION_SAMPLER_PATH = rs_target
    assert module._patch_rejection_sampler_tree_lcp() is True
    rs_patched = rs_target.read_text()
    assert "'rowbug': bool(_path) and int(_path[-1]) != _alen," in rs_patched
    py_compile.compile(str(rs_target), doraise=True)


def test_drafter_kv_locator_accepts_bare_tensor_and_fail_louds() -> None:
    # Instrument (iv) repair (wf_a71e2a24 FAIL-1): cu130-nightly binds
    # layer.kv_cache as a BARE tensor (bind_kv_cache), the old locator only
    # matched the legacy list/tuple form => 0 modules, 168 vacuous records.
    eagle = _eagle_new_snippet()
    bare = eagle.index("torch.is_tensor(_fr13_ch_kvc)")
    legacy = eagle.index("isinstance(_fr13_ch_kvc, (list, tuple))")
    assert bare < legacy, "bare-tensor form must be checked first"
    # Use-site resolver mirrors the locator (no more .kv_cache[0] blind read).
    assert "torch.is_tensor(_fr13_ch_kvr)" in eagle
    assert "_fr13_ch_mod.kv_cache[0]" not in eagle
    # Class-9 fail-loud: a 0-module harvest can never bank silently again.
    assert "FR13_CHASE_KV_ALLOW_EMPTY" in eagle
    assert "instrument " in eagle and "VACUOUS" in eagle
    raise_idx = eagle.index("FR13_CHASE_KV_ALLOW_EMPTY")
    needle_idx = eagle.index(
        "FR13_CHASE_DIAG drafter-KV tap: %d kv-cache "
    )
    assert raise_idx < needle_idx, "fail-loud precedes the count needle"


def test_h3_probe_present_with_recorded_deviation() -> None:
    eagle = _eagle_new_snippet()
    for needle in (
        'os.environ.get("FR13_CHASE_H3", "1") == "1"',
        '"tap": "h3_target_fullattn"',
        "FR13_CHASE_H3_LAYER",
        "static_forward_context",
        '"depth_last_writer_row"',
        '"foreign_slot"',
        '"accepted_flat_row"',
        '"kv_cache_gid"',
        "the H3 probe would be VACUOUS",
        # The minimal-version deviation is recorded IN-BAND, per task.
        "served-slot only: accepted-row pre-overwrite ",
    ):
        assert needle in eagle, needle
    # H3 rides the chase-diag tail: after the drafter_kv record, before the
    # packed return; drafter modules are excluded from the target walk.
    assert eagle.index('"tap": "h3_target_fullattn"') > eagle.index(
        '"tap": "drafter_kv"'
    )
    assert eagle.index('"tap": "h3_target_fullattn"') < eagle.index(
        "return _fr10_packed"
    )
    assert eagle.index("_fr13_h3_drafter_ids") < eagle.index(
        "_fr13_h3_cands.append"
    )


def test_tap_a_store_keeps_rows_for_tap_c() -> None:
    # wf_a71e2a24 FAIL-2: the chase edit dropped 'rows' from the
    # _FR13_BOUNDARY_LAST_WRITTEN_BY_REQ store while the PRE-EXISTING tap-C
    # consumer reads lastw.get("rows") for its stale verdict (always-True
    # without it). The store must keep 'rows' as long as tap-C reads it.
    helper = _helper_snippet()
    post = helper[helper.index("def _fr13_boundary_replay_post("):]
    post = post[:post.index("\ndef ")] if "\ndef " in post else post
    assert "'rows': [int(_x) for _x in _rows_written]," in post
    assert 'lastw.get(\\"rows\\", [])' in TEXT or \
        '_fr13_bnd_lastw.get(\\"rows\\", [])' in TEXT


def test_launcher_h3_and_kv_empty_passthrough() -> None:
    for aux in ("FR13_CHASE_H3", "FR13_CHASE_H3_LAYER",
                "FR13_CHASE_KV_ALLOW_EMPTY"):
        assert f'-e {aux}="${aux}" \\' in LAUNCHER_TEXT, aux
    assert "FR13_CHASE_H3=${FR13_CHASE_H3:-1}" in LAUNCHER_TEXT
    assert (
        "FR13_CHASE_KV_ALLOW_EMPTY=${FR13_CHASE_KV_ALLOW_EMPTY:-0}"
        in LAUNCHER_TEXT
    )


def test_rescued_reducers_importable_and_compile() -> None:
    # Single-disk-risk rescue (plan Step 0b): the lockstep reducers are
    # tracked copies under scripts/fr13_chase/ with co-located lib import.
    chase_dir = REPO / "scripts" / "fr13_chase"
    expected = {
        "analyze_s1.py", "analyze_s2.py", "analyze_s3.py",
        "build_align_map.py", "fr13_disc_lib.py",
        "analyze_bootA_forced_spine.py", "analyze_bootB_convfix.py",
    }
    present = {p.name for p in chase_dir.glob("*.py")}
    assert expected <= present, expected - present
    for name in sorted(expected):
        py_compile.compile(str(chase_dir / name), doraise=True)
    for name in ("analyze_bootA_forced_spine.py", "analyze_bootB_convfix.py"):
        text = (chase_dir / name).read_text()
        assert "sys.path.insert(0, str(Path(__file__).resolve().parent))" \
            in text, name
