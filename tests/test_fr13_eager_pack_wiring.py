"""FIX-2 wiring gate: FR13_EAGER_PACK (FR13_B1_SPEED_ATTRIBUTION_BIND.md).

ONE flag, default OFF until the lossless gate passes; OFF = byte-verbatim
legacy transports (the A/B instrument, FR13_DRAFTER_SINGLE_LOGITS pattern).
ON changes NO computed value — only WHERE/HOW the same values move:

  2a packed DtoH (6 committer tolists + 48x2 replay-flag matrix -> ONE
     pinned readback; ALL-LAYER batched flag validation — representative-
     layer validation is BANNED, failed safety twice)
  2b ONE batched all-layer replay launch (init-time stacked rings with
     per-layer views so the captured forward is unchanged; int64 bank
     pointer table re-asserted per commit; int-view byte A/B gate)
  2c pinned staged HtoD for committer outputs/publishes
  2d exact-key cache for the constant runner tree-metadata tensors
  2e/2f pinned transports for depth positions and the REQKEY rewrite
  2g sampler: provably-redundant clone dropped + persistent argmax buffer
  2h verified trivials (dead cast, dead self-copy, arange hoist,
     _fr10_path0_x gated-move only — NEVER deleted)

Playbook class 9: text-level asserts that the flag exists end-to-end with
default OFF, exact legacy paths preserved under OFF, an engagement needle
firing in BOTH states, and docker -e passthrough in BOTH launchers.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
KERNEL = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
LAUNCHERS = [
    REPO / "scripts" / "fr10_launch_speed_server.sh",
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
]

TEXT = PATCHER.read_text()
KTEXT = KERNEL.read_text()


def _helper_snippet() -> str:
    """The committer helper r''' string injected into rejection_sampler."""
    start = TEXT.index("    helper = r'''")
    body_start = start + len("    helper = r'''")
    return TEXT[body_start:TEXT.index("'''", body_start)]


def _extract_function(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.find("\ndef ", start + 1)
    if end == -1:
        end = len(source)
    return source[start:end]


def test_flag_default_off_everywhere() -> None:
    # Every read site defaults OFF ("0"); no site may default ON. The
    # builder-init/gdn-linear sites live in "..."-escaped patch strings; the
    # committer helper is an r''' string; the runner sites are """ strings.
    assert (
        'os.environ.get(\\"FR13_EAGER_PACK\\", \\"0\\") == \\"1\\"' in TEXT
    )  # gdn_attn builder init + gdn_linear module header (escaped form)
    assert (
        "__import__('os').environ.get('FR13_EAGER_PACK', '0') == '1'" in TEXT
    )  # committer helper
    assert '"FR13_EAGER_PACK", "0"' in TEXT  # runner """ sites
    assert '"FR13_EAGER_PACK", "1"' not in TEXT
    assert "'FR13_EAGER_PACK', '1'" not in TEXT
    assert '\\"FR13_EAGER_PACK\\", \\"1\\"' not in TEXT
    # No defaultless reads (unset env must mean OFF, the deployed default
    # until the lossless gate passes).
    assert 'environ.get("FR13_EAGER_PACK")' not in TEXT
    assert "environ.get('FR13_EAGER_PACK')" not in TEXT
    assert 'environ.get(\\"FR13_EAGER_PACK\\")' not in TEXT
    # The kernel module reads no flag: route selection is wiring-level
    # (docstrings/comments may mention the flag name).
    assert "environ" not in KTEXT or "FR13_EAGER_PACK" not in [
        line
        for line in KTEXT.splitlines()
        if "environ" in line
    ]
    # Env is read ONCE at module scope in the gdn_linear header (flag plan:
    # no per-step env reads on the capturable conv path).
    assert (
        '_FR13_EAGER_PACK = os.environ.get(\\"FR13_EAGER_PACK\\", \\"0\\") == \\"1\\"'
        in TEXT
    )


def test_both_launchers_default_off_and_pass_the_flag() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        assert "FR13_EAGER_PACK=${FR13_EAGER_PACK:-0}" in text, launcher
        assert '-e FR13_EAGER_PACK="$FR13_EAGER_PACK" \\' in text, launcher
        assert "FR13_EAGER_PACK:-1" not in text, launcher


def test_engagement_needle_fires_in_both_states() -> None:
    helper = _helper_snippet()
    assert "FR13_EAGER_PACK committer path engaged: pack=" in helper
    assert "def _fr13_eager_pack_needle(" in helper
    # Exactly one call site (greedy committer), reached UNCONDITIONALLY after
    # the pack/legacy branch -- the needle reports pack=0 in the OFF arm too
    # (inverse engagement comes from no-spec arms having no committer call).
    assert helper.count("_fr13_eager_pack_needle(") == 2  # def + call
    committer = _extract_function(
        helper, "_lumo_tree_path_lcp_max_greedy_sample"
    )
    call_idx = committer.index("_fr13_eager_pack_needle(")
    else_idx = committer.index("    else:\n        parents_cpu = ")
    assert call_idx > else_idx  # needle sits after the if/else, both states


def test_committer_packed_dtoh_and_verbatim_legacy_off_block() -> None:
    helper = _helper_snippet()
    committer = _extract_function(
        helper, "_lumo_tree_path_lcp_max_greedy_sample"
    )

    # ON path: single packed DtoH (one pinned readback + ONE stream sync).
    assert "_fr13_eager_pack_stage(" in committer
    assert "'committer_dtoh'" in committer
    assert (
        "_ep_cpu[:_ep_total].copy_(_ep_dev[:_ep_total], non_blocking=True)"
        in committer
    )
    assert committer.count("torch.cuda.current_stream(") == 1
    # Flag matrix rides the SAME packed readback.
    assert "_ep_stacks['flags'].detach().reshape(-1)" in committer
    # counts stays on its existing CPU branch when already a list.
    assert "counts = [int(x) for x in num_draft_tokens]" in committer

    # OFF path: the exact legacy 6-tolist block, contiguous and verbatim.
    legacy_tolist = (
        "        parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]\n"
        "        drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]\n"
        "        parent_targets_cpu = [int(x) for x in parent_token_ids.detach().cpu().tolist()]\n"
        "        self_targets_cpu = [int(x) for x in self_token_ids.detach().cpu().tolist()]\n"
        "        bonus_targets_raw = bonus_token_ids.detach().cpu().tolist()\n"
    )
    assert legacy_tolist in committer

    # OFF path: legacy output writes verbatim.
    legacy_out = (
        "        output_token_ids.fill_(-1)\n"
        "        for req_i, row in enumerate(out_rows):\n"
        "            for pos, token_id in enumerate(row):\n"
        "                output_token_ids[req_i, pos] = int(token_id)\n"
    )
    assert legacy_out in committer

    # CPU tensors (offline oracles / unit rigs) always take legacy.
    assert "getattr(tree_parent_indices, 'is_cuda', False)" in committer


def test_committer_all_layer_flag_validation_no_representative_shortcut() -> None:
    helper = _helper_snippet()
    committer = _extract_function(
        helper, "_lumo_tree_path_lcp_max_greedy_sample"
    )
    # EVERY layer validated individually, with per-layer error messages.
    assert "for _ep_i, _ep_prefix in enumerate(_ep_order):" in committer
    assert "_ep_flag_rows[_ep_i][0] != 1" in committer
    assert "_ep_flag_rows[_ep_i][1] != _fr13_replay_rows" in committer
    assert "REPRESENTATIVE-LAYER VALIDATION IS BANNED" in committer
    # Layer-order identity is asserted against sorted(_fr13_replay_layers).
    assert "_ep_order != _ep_sorted_prefixes" in committer
    # Boundary diagnostics route through the VERBATIM legacy per-layer loop.
    assert "if _ep_active and not _fr13_bnd_on:" in committer
    # Bank pointer table: built once, re-asserted every commit, fail-loud.
    assert "_FR13_EAGER_PACK_BANK_TBL" in committer
    assert "_ep_tbl[0] != _ep_ptrs_now" in committer
    assert "GDN bank data_ptr changed since" in committer
    # ONE batched launch + ONE all-layer clear in the ON route.
    assert "launch_tree_gdn_replay_all_layers as _ep_launch_all" in committer
    assert "_ep_stacks['flags'][:, 0].fill_(0)" in committer
    # Legacy per-layer launch + per-layer clear intact in the else branch
    # (counts across BOTH committers are asserted by
    # test_fr13_replay_route_wiring.py: 2x _fr13_replay_launch( and
    # 2x _fr13_flags[0].fill_(0)).
    assert "_fr13_replay_launch(" in committer
    assert "_fr13_flags[0].fill_(0)" in committer


def test_committer_staged_htod_publishes_keep_legacy_order_and_values() -> None:
    helper = _helper_snippet()
    committer = _extract_function(
        helper, "_lumo_tree_path_lcp_max_greedy_sample"
    )
    # Three staged pinned transports (out tokens, publish-1, publish-2):
    # legacy ORDER is preserved exactly (publish-1 before REQKEY dict before
    # publish-2 before launch), which is why this deviates from the design's
    # literal "one HtoD" (a single transport would have to move the publish
    # values across the try-block's raise sites = an ordering change).
    assert "'committer_htod_out'" in committer
    assert "'committer_htod_pub1'" in committer
    assert "'committer_htod_pub2'" in committer
    assert committer.index("'committer_htod_pub1'") < committer.index(
        "_LUMO_FA_TREE_ACCEPT_BY_REQ"
    )
    assert committer.index("'committer_htod_pub2'") < committer.index(
        "launch_tree_gdn_replay_all_layers as _ep_launch_all"
    )
    # Pinned-reuse guard around every pinned write (event sync-then-record).
    assert committer.count(".synchronize()") >= 4  # stream + 3 event guards
    assert committer.count(".record()") == 3
    # Legacy publish text verbatim in the else branches.
    assert "dtype=_accepted_path_buf.dtype," in committer
    assert "dtype=_accepted_lens_buf.dtype," in committer
    # No skip-if-unchanged (class 8): the staged publishes are gated ONLY on
    # the same conditions as legacy (_accepted_path_rows / _fr13_replay_rows
    # truthiness + the flag), never on a value comparison with the buffers.
    assert "KEPT even when identical" in committer


def test_sampler_clone_drop_and_persistent_argmax_buffer() -> None:
    # 2g lives in the stock_call_new replacement string.
    start = TEXT.index('stock_call_new = """')
    sampler = TEXT[start:TEXT.index('"""', start + len('stock_call_new = "'))]
    # (i) clone dropped ONLY under the flag; legacy clone verbatim under OFF.
    assert (
        "            if not _FR13_EAGER_PACK:\n"
        "                if not self.is_processed_logprobs_mode:\n"
        "                    tree_self_logits = tree_self_logits.clone()\n"
    ) in sampler
    # (ii) persistent [2, T] int32 buffer; argmax inputs untouched.
    assert "_fr13_eager_pack_tree_ids_buf(" in sampler
    assert "lumo_tree_token_ids[0].copy_(target_logits.argmax(dim=-1))" in sampler
    # Legacy stack/contiguous block verbatim in the else branch.
    assert (
        "                else:\n"
        "                    lumo_tree_token_ids = torch.stack(\n"
    ) in sampler
    # The non-greedy branch and constraints/processors are untouched (no
    # _FR13_EAGER_PACK reference after the greedy block's else).
    after_greedy = sampler[sampler.index("                else:\n"):]
    assert "_FR13_EAGER_PACK" not in after_greedy


def test_runner_meta_cache_depth_positions_and_reqkey_staging() -> None:
    # 2d: exact-equality cache key spans the FULL value domain.
    assert "_fr13_eager_pack_meta_cache" in TEXT
    assert "tuple(int(_x) for _x in num_draft_tokens.tolist())" in TEXT
    assert "_ep_meta_hit[0].copy()" in TEXT  # private numpy copy on hit
    assert "target_logits_indices.copy()," in TEXT  # private copy on store
    # Legacy metadata build verbatim under miss/OFF.
    assert "_path_to_idx = {tuple(_p): _i for _i, _p in enumerate(_choices)}" in TEXT
    # 2e: persistent pinned depth-pos buffers; legacy np.empty under OFF.
    assert "_fr13_eager_pack_depth_pos" in TEXT
    assert "pin_memory=True," in TEXT
    assert "_fr10_depth_pos = np.empty(" in TEXT
    assert "_fr10_depth_pos_cpu = torch.from_numpy(_fr10_depth_pos)" in TEXT
    # NO caching of depth positions (values change every step — excluded).
    assert "depth_pos_cache" not in TEXT
    # 2f: REQKEY staging with legacy torch.tensor copies under OFF.
    assert "_fr13_eager_pack_rk_stage" in TEXT
    assert TEXT.count("_fr13_rk_paths[:_fr13_rk_n].copy_(") == 2  # ON + legacy


def test_conv_trivials_gated_and_path0_never_deleted() -> None:
    # :1192 dead cast — deleted under the flag, legacy verbatim under OFF.
    assert (
        "                    if not _FR13_EAGER_PACK:\n" in TEXT
    )
    assert "_fr10_weight_f = conv_weights.to(torch.float32)" in TEXT
    # :3159 self-copy — verified dead, deleted under the flag only.
    assert "core_attn_out_spec[0, start:end] = tree_out[:tree_n]" in TEXT
    # :1283 _fr10_path0_x — gated-MOVE only (census caution): legacy site +
    # one recompute per gated diagnostic branch = exactly 3 assignments.
    assert (
        TEXT.count("_fr10_path0_x = _fr10_x.index_select(") == 3
    )
    # arange hoist: cache keyed by the full value domain, never cached
    # during capture (no capture-pool tensor outlives its graph).
    assert "_fr13_eager_pack_conv_arange" in TEXT
    assert "torch.cuda.is_current_stream_capturing()" in TEXT


def test_builder_init_stacked_views_and_fail_loud() -> None:
    gdn_linear_start = TEXT.index("def _patch_gdn_linear(")
    # Stacked rings allocated at METADATA-BUILDER INIT (class 6), before the
    # forward patch; per-layer attributes become layer-slice VIEWS so the
    # captured forward's writes are unchanged.
    for needle in (
        "_fr13_ep_ring_k = torch.zeros(",
        "_fr13_layer._fr13_replay_ring_k = _fr13_ep_ring_k[_fr13_row]",
        "_fr13_layer._fr13_replay_flags = _fr13_ep_flags[_fr13_row]",
        '_fr13_gdn_mod._FR13_EAGER_PACK_STACKS = {',
    ):
        assert TEXT.index(needle) < gdn_linear_start, needle
    # Rows follow sorted(layer_names) = committer iteration order.
    assert "sorted(str(_n) for _n in layer_names)" in TEXT
    # FAIL-LOUD preconditions (class 9): non-uniform layers, multi-group.
    assert "FR13_EAGER_PACK stacking precondition failed" in TEXT
    assert "requires a single GDN replay" in TEXT
    # Legacy per-layer allocation intact for the OFF arm.
    assert "_fr13_layer._fr13_replay_ring_k = torch.zeros(" in TEXT


def test_batched_kernel_shape_and_byte_ab_gate_documented() -> None:
    assert "def _tree_gdn_replay_all_layers_kernel(" in KTEXT
    assert "def launch_tree_gdn_replay_all_layers(" in KTEXT
    assert "def build_replay_bank_pointer_table(" in KTEXT
    # Layer-0 anchor + int64 offset table (per-layer banks are distinct
    # KV-pool tensors). NOT a raw pointer load + tt.int_to_ptr cast: that
    # form loses AxisInfo divisibility, scalarizes the layout and changes
    # fp32 rounding vs the legacy per-layer kernel (byte-A/B fix,
    # 2026-06-12 GPU gate).
    assert "state_bank = bank_anchor + tl.load(bank_off16 + pid_l) * 4" in KTEXT
    _batched_code = "\n".join(
        line
        for line in KTEXT.split("def _tree_gdn_replay_all_layers_kernel(")[1]
        .split("def build_replay_bank_pointer_table(")[0]
        .splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "tl.pointer_type" not in _batched_code
    # Same warp shape + raw-gating basis as the single-layer replay.
    batched = KTEXT[KTEXT.index("_tree_gdn_replay_all_layers_kernel[grid]"):]
    assert "RAW_GATING=True," in batched
    assert "num_warps=8," in batched
    # grid packs (layer, spec) on pid0.
    assert (
        "grid = (int(num_layers) * int(num_spec_decodes), num_vh, triton.cdiv(dim_v, BV))"
        in KTEXT
    )
    # Class-10 gate is named at the kernel: int-view byte A/B, never atol.
    assert "int-view byte" in KTEXT
    # The shared node-step body is the bit-exactness basis (same inline).
    batched_def = KTEXT[
        KTEXT.index("def _tree_gdn_replay_all_layers_kernel("):
        KTEXT.index("def build_replay_bank_pointer_table(")
    ]
    assert "_gdn_node_step(" in batched_def
    assert "state = state + 0.0" in batched_def  # -0.0 handoff parity kept


def test_fragments_still_parse_and_committer_compiles() -> None:
    helper = _helper_snippet()
    compile(helper, "<fr13_eager_pack_helper>", "exec")
    committer = _extract_function(
        helper, "_lumo_tree_path_lcp_max_greedy_sample"
    )
    compile(committer, "<fr13_eager_pack_committer>", "exec")
    # The named string fragments carrying FIX-2 code must parse as Python
    # (method-body code at fixed indentation; the REQKEY block's compile is
    # exercised end-to-end by test_fr13_nondet_chase_fixes.py).
    tree = ast.parse(TEXT)
    checked = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in ("meta_inject", "inject", "stock_call_new", "conv_replacement")
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and "FR13_EAGER_PACK" in node.value.value
        ):
            ast.parse(textwrap.dedent(node.value.value))
            checked += 1
    # meta_inject (2d), depth-positions inject (2e), stock_call_new (2g).
    assert checked >= 3, checked
