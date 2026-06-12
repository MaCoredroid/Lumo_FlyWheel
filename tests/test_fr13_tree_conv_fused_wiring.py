"""FIX-3 wiring gate: FR13_TREE_CONV_FUSED (FR13_B1_SPEED_ATTRIBUTION_BIND).

ONE flag, default OFF until the byte A/B + live gate pass; OFF = byte-
verbatim legacy tree-conv emulation (the A/B instrument, FIX-2 pattern). ON
changes NO computed value — the same per-element ops in the same order, only
HOW they are issued:

  (1) per-node state write-back loop (7 device nodes x tree_n, the census
      tree_n multiplier) -> ONE init-time static-index gather + the existing
      index_copy_ (class 3 gather-then-scatter)
  (2) per-col tap/bias accumulation -> one bf16 mul + one fp32 cast + bias
      broadcast + EXPLICIT ordered adds (reduction ops BANNED); the silu is
      the SAME triton kernel object on shared lines
  (3) page-safe remap + (4) committed-prior gather -> once-per-kv-cache-
      group prepared row math into init-time persistent buffers + 1-2
      per-layer bank ops; the frozen library fns stay the OFF arm

Playbook class 9: text-level asserts that the flag exists end-to-end with
default OFF, exact legacy paths preserved under OFF, fail-loud engagement
preconditions (never a silent fall-through), an engagement needle firing in
BOTH states, docker -e passthrough in BOTH launchers, and tree-only scope
(no fused reference reachable from the native conv path).
"""

from __future__ import annotations

import ast
import py_compile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
FUSED_MODULE = REPO / "src" / "lumo_flywheel_serving" / "fr13_tree_conv_fused.py"
REMAP_MODULE = REPO / "src" / "lumo_flywheel_serving" / "fr13_replay_conv_remap.py"
KERNEL = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
LAUNCHERS = [
    REPO / "scripts" / "fr10_launch_speed_server.sh",
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
]

TEXT = PATCHER.read_text()
FTEXT = FUSED_MODULE.read_text()
RTEXT = REMAP_MODULE.read_text()
KTEXT = KERNEL.read_text()


def _conv_replacement() -> str:
    tree = ast.parse(TEXT)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "conv_replacement"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise AssertionError("conv_replacement string not found")


CONV = _conv_replacement()


def test_flag_default_off_everywhere() -> None:
    # Module-scope read-once (escaped patch-string form), default OFF.
    assert (
        '_FR13_TREE_CONV_FUSED = os.environ.get(\\"FR13_TREE_CONV_FUSED\\", \\"0\\") == \\"1\\"'
        in TEXT
    )
    # Builder-init read (escaped form), default OFF; two escaped read sites
    # total (module header + builder init).
    assert (
        TEXT.count('os.environ.get(\\"FR13_TREE_CONV_FUSED\\", \\"0\\") == \\"1\\"')
        == 2
    )
    # No site may default ON; no defaultless reads.
    assert '\\"FR13_TREE_CONV_FUSED\\", \\"1\\"' not in TEXT
    assert '"FR13_TREE_CONV_FUSED", "1"' not in TEXT
    assert "'FR13_TREE_CONV_FUSED', '1'" not in TEXT
    assert 'environ.get(\\"FR13_TREE_CONV_FUSED\\")' not in TEXT
    assert 'environ.get("FR13_TREE_CONV_FUSED")' not in TEXT
    # The conv forward reads ONLY the module-scope flag (no per-step env
    # read on the capturable path).
    assert 'os.environ.get("FR13_TREE_CONV_FUSED"' not in CONV
    assert "_FR13_TREE_CONV_FUSED" in CONV
    # The fused library module reads no env at all (wiring-level routing).
    assert "environ" not in FTEXT
    # The frozen OFF-arm libraries are untouched by the flag.
    assert "FR13_TREE_CONV_FUSED" not in RTEXT
    assert "FR13_TREE_CONV_FUSED" not in KTEXT


def test_both_launchers_default_off_and_pass_the_flag() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        assert "FR13_TREE_CONV_FUSED=${FR13_TREE_CONV_FUSED:-0}" in text, launcher
        assert '-e FR13_TREE_CONV_FUSED="$FR13_TREE_CONV_FUSED" \\' in text, launcher
        assert "FR13_TREE_CONV_FUSED:-1" not in text, launcher


def test_engagement_needle_fires_in_both_states() -> None:
    assert "def _fr13_tree_conv_fused_needle(" in TEXT
    assert (
        "FR13_TREE_CONV_FUSED conv emulation engaged: fused=%d" in TEXT
    )
    assert "tree_n=%d width=%d state_len=%d prepared_rows=%d" in TEXT
    assert "static_tables=%d zero_row_cached=%d" in TEXT
    # Exactly one call site in the conv forward, guarded only by the
    # one-shot latch — NOT inside a fused-only branch (fires with fused=0
    # too; the flag value is the first argument).
    assert CONV.count("_fr13_tree_conv_fused_needle(") == 1
    call_at = CONV.index("_fr13_tree_conv_fused_needle(")
    guard_at = CONV.index("if not _FR13_TREE_CONV_FUSED_NEEDLE_DONE:")
    assert guard_at < call_at
    assert (
        "_fr13_tree_conv_fused_needle(\n                            _FR13_TREE_CONV_FUSED,"
        in CONV
    )


def test_fail_loud_engagement_preconditions() -> None:
    # One-time env precondition check, called at the top of the tree-conv
    # branch under the flag; per-engagement dtype uniformity assert.
    assert "def _fr13_tree_conv_fused_check(" in TEXT
    assert "_fr13_tree_conv_fused_check()" in CONV
    assert "FR13_TREE_CONV_FUSED=1 requires FR13_CONV_COMMITTED_PATH=1" in TEXT
    assert "FR13_TREE_CONV_FUSED=1 requires FR13_REPLAY_ROUTE=1" in TEXT
    assert "requires native bf16 taps" in TEXT
    assert "FR13_TREE_CONV_FUSED=1 excludes diagnostic env" in TEXT
    for banned in (
        "FR12_NATIVE_SPINE_ORACLE",
        "FR12_TREE_CONV_NATIVE_PRIOR_READ",
        "FR12_TREE_CONV_STATE_FULL_CAPTURE",
        "FR13_CONV_REPLAY_NODES",
        "FR12_SUBKERNEL_CAPTURE",
    ):
        check_fn = TEXT[TEXT.index("def _fr13_tree_conv_fused_check(") :]
        check_fn = check_fn[: check_fn.index("def _fr13_tree_conv_fused_needle(")]
        assert banned in check_fn, banned
    assert "FR13_TREE_CONV_FUSED dtype uniformity violated" in CONV
    # Missing builder-init buffers/owners = RuntimeError, never silent legacy.
    assert "FR13_TREE_CONV_FUSED engaged without" in CONV
    assert "builder-init prep buffers/owners" in CONV
    # Prepared-rows geometry disagreement = RuntimeError.
    assert "prepared remap" in CONV and "disagree with the static" in CONV


def test_verbatim_legacy_surfaces_under_off() -> None:
    # (1) write-back loop: every legacy op survives in the else arm.
    for needle in (
        "_fr10_node_state_rows = []",
        "_fr10_node_state_rows.append(",
        "_fr10_node_x = _fr10_x.index_select(0, _fr10_node_path)",
        "_fr10_x.new_zeros(",
        "_fr10_node_store_idx = (",
        "_fr10_new_state = torch.stack(",
    ):
        assert needle in CONV, needle
    # (2) taps: legacy per-col loop + bias expand/clone seed + the tap fn.
    assert "def _fr11_conv_tap_product" in CONV
    assert "_fr10_acc = _fr10_acc + _fr11_conv_tap_product(" in CONV
    assert "expand_as(_fr10_x.float()).clone()" in CONV
    # (3) legacy two-operand window cat in the else arm.
    assert (
        "_fr10_source = torch.cat(\n"
        "                                (_fr10_prior_window.transpose(0, 1), _fr10_x),\n"
        "                                dim=0,\n"
        "                            )"
    ) in CONV
    # (4) legacy library calls intact (frozen OFF arms).
    assert ") = gather_committed_path_conv_prior(" in CONV
    assert "replay_conv_state_linear_remap(" in CONV
    assert "launch_tree_state_linear_remap(" in CONV
    # Every fused divergence is an if/else on the module-scope flag: one
    # branch site per census contributor + precheck (gather, remap, static
    # cache, window cat, taps, write-back, precheck = 7).
    assert CONV.count("if _FR13_TREE_CONV_FUSED:") == 7
    # The shared silu lines are NOT branched (same kernel object both arms).
    assert CONV.count("_fr10_out = triton_ex2_silu_bf16(") == 1
    # The final index_copy_ publish is shared/unchanged.
    assert "conv_state.index_copy_(" in CONV


def test_fused_arm_calls_the_importable_library() -> None:
    # The live fused branch runs the SAME functions the byte A/B imports.
    for fn in (
        "fused_tree_conv_source(",
        "fused_tree_conv_taps_acc(",
        "fused_tree_conv_state_rows(",
        "gather_committed_path_conv_prior_prepared(",
        "prepare_committed_path_conv_rows(",
        "prepare_replay_conv_remap_rows(",
        "replay_conv_state_linear_remap_prepared(",
        "build_tree_conv_state_src_indices(",
    ):
        assert fn in CONV, fn
    # Imported in the gdn_linear module header (escaped patch string).
    assert (
        "from lumo_flywheel_serving.fr13_tree_conv_fused import "
        "build_tree_conv_state_src_indices, fused_tree_conv_source, "
        "fused_tree_conv_state_rows, fused_tree_conv_taps_acc, "
        "gather_committed_path_conv_prior_prepared, "
        "prepare_committed_path_conv_rows, prepare_replay_conv_remap_rows, "
        "replay_conv_state_linear_remap_prepared" in TEXT
    )


def test_reduction_ops_banned_in_fused_module() -> None:
    # Bit-exactness rests on EXPLICIT ordered adds: any reduction op has
    # unspecified order. Pin the textual ban (risk 1 in the FIX-3 design).
    code_only = "\n".join(
        line
        for line in FTEXT.splitlines()
        if not line.lstrip().startswith("#")
    )
    for banned in ("torch.sum(", ".sum(", "einsum", "matmul", "tensordot", ".bmm("):
        assert banned not in code_only, banned
    # The explicit ordered-add sequence (legacy operand order: acc left).
    assert "acc = acc + prod_f32[:, col, :]" in FTEXT
    assert "acc = bias.to(torch.float32).view(1, -1) + prod_f32[:, 0, :]" in FTEXT
    # bf16 mul + single fp32 cast boundary (the legacy .to(f32) seam).
    assert "prod = window * conv_weights.t().unsqueeze(0)" in FTEXT
    assert "prod_f32 = prod.to(torch.float32)" in FTEXT


def test_legacy_replicas_pinned_to_patcher_and_frozen_libraries() -> None:
    # Replica <-> patcher legacy text (write-back loop ops).
    legacy_fn = FTEXT[FTEXT.index("def legacy_tree_conv_state_rows_reference(") :]
    legacy_fn = legacy_fn[: legacy_fn.index("def legacy_gather_committed_path")]
    for shared in (
        ".index_select(0, node_path)" if False else "node_state_source = torch.cat(",
        "new_zeros(",
        ".transpose(0, 1)",
        "torch.stack(node_state_rows, dim=0)",
    ):
        assert shared in legacy_fn, shared
    # Replica taps <-> patcher _fr11_conv_tap_product (bf16-taps arm).
    taps_fn = FTEXT[FTEXT.index("def legacy_tree_conv_taps_acc_reference(") :]
    taps_fn = taps_fn[: taps_fn.index("def legacy_tree_conv_state_rows_reference(")]
    assert ".to(tap_dtype)\n            .to(torch.float32)" in taps_fn
    assert "expand_as(x.float()).clone()" in taps_fn
    # Prepared remap row math <-> frozen fr13_replay_conv_remap lines.
    for shared in (
        "src_col = torch.where(valid, src_col, dst_col)",
        "src_rows = window.gather(1, src_col).reshape(-1)",
        "dst_rows = window.gather(1, dst_col).reshape(-1)",
        "vals = conv_state.index_select(0, src_rows)",
        "conv_state.index_copy_(0, dst_rows, vals)",
    ):
        assert shared in FTEXT, shared
        assert shared in RTEXT, shared
    # Prepared committed-prior row math <-> frozen fr10_gdn_tree_kernel fn.
    for shared in (
        "read_node_cols = accepted_paths[:b].to(torch.long).gather(1, path_cols)",
        "lens > 0, read_node_cols, torch.zeros_like(read_node_cols)",
        "bank_rows = spec_state_indices[:b].to(torch.long).gather(1, read_node_cols)",
    ):
        assert shared in FTEXT, shared
        assert shared in KTEXT, shared


def test_builder_init_prep_buffers_group_union_and_boot_needle() -> None:
    gdn_linear_start = TEXT.index("def _patch_gdn_linear(")
    # Prep buffers + owners allocated/exported at METADATA-BUILDER INIT
    # (class 6), inside the gdn_attn builder patch (before _patch_gdn_linear).
    for needle in (
        "_fr13_tcf_mod._FR13_TCF_PREP = _fr13_tcf_prep",
        "_fr13_tcf_mod._FR13_TCF_PREP_OWNERS = frozenset(",
        "_fr13_tcf_mod._FR13_TCF_GROUP_OWNERS = _fr13_tcf_groups",
    ):
        assert TEXT.index(needle) < gdn_linear_start, needle
    # Group-first ownership ranked by NUMERIC layer index (model execution
    # order), fail-loud when unrankable.
    assert "for group-first prep ownership (fail-loud)" in TEXT
    assert "_fr13_tcf_owner = min(_fr13_tcf_ranked)[1]" in TEXT
    # Union/re-init invariants (the FIX-2 cu130 3-group/5x-re-init lesson):
    # rebuild + re-export per init, owner map keyed per group, geometry
    # pinned across inits, boot-log needle for the watched first ON boot.
    assert "FR13_TREE_CONV_FUSED owner-union invariant" in TEXT
    assert "FR13_TREE_CONV_FUSED prep-buffer geometry changed" in TEXT
    assert "FR13_TREE_CONV_FUSED prep buffers rebuilt" in TEXT
    # Forward-side: owner writes go through .copy_ into the persistent
    # buffers (fixed addresses across capture/replay); consumers slice-read.
    assert '_fr13_tcf_prep["read_cols"][' in CONV
    assert '_fr13_tcf_prep["src_rows"][' in CONV
    assert ".copy_(_fr13_tcf_new_src)" in CONV
    assert 'str(self.prefix) in _fr13_tcf_owners' in CONV


def test_static_tables_capture_safe_cache() -> None:
    # FIX-2 layer-keyed cache pattern: full value-domain key, populate on
    # first NON-capturing forward, capture-time miss builds without
    # retaining.
    assert "_fr13_tree_conv_fused_static" in CONV
    cache_block = CONV[CONV.index("_fr13_tcf_key = (") :]
    cache_block = cache_block[: cache_block.index("_fr12_bf16_tap_env")]
    assert "tuple(_fr10_parent)" in cache_block
    assert "int(conv_state.size(2))" in cache_block
    assert "str(mixed_qkv_spec.device)" in cache_block
    assert "if not torch.cuda.is_current_stream_capturing():" in cache_block
    # The zero row is part of the same cached tuple (init-cached persistent
    # [1, dim] source row).
    assert "_fr13_tcf_zero_row = torch.zeros(" in cache_block


def test_tree_only_scope_native_path_untouched() -> None:
    # Census trivial: _fr10_path0_x gated-move semantics unchanged from
    # FIX-2 (legacy site + 2 gated recomputes; never deleted, never added).
    assert TEXT.count("_fr10_path0_x = _fr10_x.index_select(") == 3
    # Tree-only: no fused reference after the tree branch's native fallback
    # (the eligible-row raise marks the start of the non-tree else arm).
    native_idx = CONV.index("eligible_tree_spec_row_flat_fallback")
    assert "_FR13_TREE_CONV_FUSED" not in CONV[native_idx:]
    assert "fused_tree_conv" not in CONV[native_idx:]
    # The native causal_conv1d_update calls are byte-verbatim (both the
    # in-branch fallback and the non-tree spec path).
    assert CONV.count("mixed_qkv_spec = causal_conv1d_update(") == 2
    # Every fused site sits inside the use_fr10_tree_conv branch: the first
    # flag reference comes after the branch opens.
    branch_at = CONV.index("if use_fr10_tree_conv:")
    assert CONV.index("_FR13_TREE_CONV_FUSED") > branch_at


def test_fragments_parse_and_modules_compile() -> None:
    ast.parse(textwrap.dedent(CONV))
    # The builder-init fragment (a .replace() arg, folded string constant).
    tree = ast.parse(TEXT)
    builder_fragments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "_FR13_TCF_PREP" in node.value
        and "fr10_tree_accepted_path_bs" in node.value
    ]
    assert builder_fragments, "builder-init prep fragment not found"
    for frag in builder_fragments:
        ast.parse(textwrap.dedent(frag))
    # The header fragment (flag + check + needle defs).
    header_fragments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "_fr13_tree_conv_fused_check" in node.value
        and "_FR13_TREE_CONV_FUSED = os.environ.get(" in node.value
    ]
    assert header_fragments, "gdn_linear header fragment not found"
    for frag in header_fragments:
        ast.parse(textwrap.dedent(frag))
    for path in (PATCHER, FUSED_MODULE, REMAP_MODULE):
        py_compile.compile(str(path), doraise=True)
