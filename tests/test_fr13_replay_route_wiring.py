"""FR13 replay-route wiring tests (CPU, text-level, style of
test_fr10_phase4_sampled_committer_wiring.py).

Flag ON wiring: replay launch present in BOTH committers, scratch alloc and
all-rows publish flag-gated off, scan-time prev-lens snapshot precedes the
scan launch, ssm remap half gated off (conv half kept), persistent staging
only -- allocated at METADATA-BUILDER INIT, never in the (capturable)
forward, with a capture-safe flag-tensor handshake instead of a per-step
dict. Flag OFF: the legacy path is textually intact and every new behavior
sits behind an FR13_REPLAY_ROUTE check that defaults to "0" -- i.e. flag-OFF
is SOURCE-INERT; compile-identity is pending the refactored-scan byte A/B
(GPU TODO #2 in FR13_REPLAY_ROUTE_BUILD.md), because the shared
_gdn_node_step body refactor recompiles the default scan.
"""

from __future__ import annotations

import ast
import py_compile
import textwrap
from pathlib import Path

PATCHER = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
KERNEL = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
REFERENCE = Path("src/lumo_flywheel_serving/fr13_replay_reference.py")
GATE = Path("scripts/fr10_serving_wiring_gate.py")


def _flagged_forward_fragment() -> str:
    """The forward's flagged replay block: ring/snapshot writes up to the
    scan launch. This region executes inside the (potentially CUDA-graph
    captured) forward, so it must only WRITE preallocated buffers."""
    text = PATCHER.read_text()
    start = text.index("# FR13_REPLAY_ROUTE activation ring + scan-time")
    end = text.index("tree_out, _ = launch_tree_gdn_prepared(", start)
    return text[start:end]


def test_replay_route_flag_defaults_off_everywhere() -> None:
    text = PATCHER.read_text()
    ktext = KERNEL.read_text()

    assert 'os.environ.get("FR13_REPLAY_ROUTE", "0") == "1"' in text
    assert "__import__('os').environ.get('FR13_REPLAY_ROUTE', '0') == '1'" in text
    # No defaultless reads: unset env must mean OFF.
    assert 'environ.get("FR13_REPLAY_ROUTE")' not in text
    assert "environ.get('FR13_REPLAY_ROUTE')" not in text
    # The kernel module itself reads no flag: route selection is wiring-level
    # (the docstrings may mention the flag name).
    assert 'environ.get("FR13_REPLAY_ROUTE"' not in ktext
    assert "environ.get('FR13_REPLAY_ROUTE'" not in ktext


def test_kernel_store_node_states_is_pure_export_gate() -> None:
    ktext = KERNEL.read_text()

    assert "STORE_NODE_STATES: tl.constexpr" in ktext
    assert "if STORE_NODE_STATES:" in ktext
    assert "STORE_NODE_STATES=store_node_states" in ktext
    # store_node_states=False must not allocate the scratch and must not
    # accept a caller-provided state buffer.
    assert "state = strict_mask  # dummy pointer; no store reaches it" in ktext
    assert "return out, None" in ktext
    assert "store_node_states: bool = True" in ktext
    # Default-ON call sites keep legacy behavior SOURCE-INERT (the legacy
    # text and default arguments are intact). NOT claimed byte-identical:
    # the shared _gdn_node_step body refactor RECOMPILES the default scan,
    # so flag-OFF compile-identity is pending the refactored-scan byte A/B
    # (GPU TODO #2 in FR13_REPLAY_ROUTE_BUILD.md).
    assert ktext.index("def launch_tree_gdn_prepared") < ktext.index(
        "state = strict_mask"
    )


def test_kernel_shared_node_step_body_used_by_scan_and_replay() -> None:
    ktext = KERNEL.read_text()

    assert ktext.count("def _gdn_node_step(") == 1
    # def + scan call + replay call = 3 mentions of the open-paren form.
    assert ktext.count("_gdn_node_step(") == 3
    assert "def _tree_gdn_replay_kernel(" in ktext
    assert "def launch_tree_gdn_replay(" in ktext
    # Identical constexpr plumbing at both call sites.
    assert ktext.count("USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,") == 2
    assert ktext.count("RAW_GATING=RAW_GATING,") == 2
    # The replay launch pins the scan's warp shape and raw-gating basis.
    replay_launch = ktext[ktext.index("_tree_gdn_replay_kernel[grid]"):]
    assert "RAW_GATING=True," in replay_launch
    assert "num_warps=8," in replay_launch
    # Handoff normalization + zero-accept root publish are in the kernel.
    assert "state = state + 0.0" in ktext
    assert "flips -0.0 to +0.0" in ktext
    assert "zero-accept" in ktext.lower()
    # Replay is chain-shaped: no h_cache USE (comments may reference the
    # scan's h_cache semantics), static range over the path.
    replay_body = ktext[
        ktext.index("def _tree_gdn_replay_kernel(") : ktext.index(
            "def launch_tree_gdn_replay("
        )
    ]
    replay_code = "\n".join(
        line for line in replay_body.splitlines() if not line.lstrip().startswith("#")
    )
    assert "h_cache" not in replay_code
    assert "tl.static_range(0, PATH_COLS + 1)" in replay_body
    # Native gate-folding basis only: no rescaled-exp reconstruction.
    assert "rescal" not in replay_code.lower()


def test_patcher_flag_on_skips_scratch_alloc_and_publish() -> None:
    text = PATCHER.read_text()

    # Scratch alloc gated; legacy alloc intact for flag OFF.
    assert "tree_state_all = None" in text
    assert "tree_state_all = torch.empty(" in text
    assert (
        "tree_state = (\n"
        "                        None if _fr13_replay_route_on else tree_state_all[fr10_b]\n"
        "                    )"
    ) in text
    # Scan launches with the export compiled out under the flag.
    assert "store_node_states=not _fr13_replay_route_on," in text
    # All-rows publish gated off under the flag, intact otherwise.
    assert (
        "if not _fr13_replay_route_on:\n"
        "                        ssm_state.index_copy_("
    ) in text
    # The staged-publish diagnostic is explicitly skipped (vacuous) on-flag.
    assert "and not _fr13_replay_route_on" in text
    # tree_state-consuming diagnostics refuse loudly under the flag.
    assert "FR13_REPLAY_ROUTE is incompatible with tree_state" in text


def test_patcher_snapshots_prev_lens_at_scan_time_before_launch() -> None:
    text = PATCHER.read_text()

    snapshot = "# SNAPSHOT prev accepted lens + spec indices AT"
    launch = "tree_out, _ = launch_tree_gdn_prepared("
    assert snapshot in text
    assert text.index(snapshot) < text.index(launch)
    # The snapshot copies from the live lens buffer that the committer
    # refills BEFORE its publish block (the verify-rider Option-1 gap).
    assert "self._fr13_replay_prev_lens[" in text
    assert "self._fr13_replay_spec_idx[" in text
    assert "_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR" in text
    # Committer refill precedes the replay launch in committer text order.
    refill = "_accepted_lens_buf[: len(accepted_lens)].copy_("
    assert text.index(refill) < text.index("_fr13_replay_launch(")


def test_patcher_staging_allocated_at_builder_init_not_in_forward() -> None:
    text = PATCHER.read_text()
    gdn_linear_start = text.index("def _patch_gdn_linear(")

    # Preallocated persistent per-layer ring + snapshot + handshake buffers,
    # allocated at METADATA-BUILDER INIT (the gdn_attn builder patch comes
    # before _patch_gdn_linear in the patcher), NOT lazily on the first
    # flagged forward: a first-flagged-forward alloc inside a CUDA FULL
    # captured region is stale-pointer aliasing (gate-4 root cause #2).
    for buf in (
        "_fr13_layer._fr13_replay_ring_k = torch.zeros(",
        "_fr13_layer._fr13_replay_ring_v = torch.zeros(",
        "_fr13_layer._fr13_replay_ring_a = torch.zeros(",
        "_fr13_layer._fr13_replay_ring_b = torch.zeros(",
        "_fr13_layer._fr13_replay_prev_lens = torch.zeros(",
        "_fr13_layer._fr13_replay_spec_idx = torch.zeros(",
        "_fr13_layer._fr13_replay_flags = torch.zeros(",
    ):
        idx = text.index(buf)
        assert idx < gdn_linear_start, buf
    # Sized B_max x N_PAD following the persistent accepted-paths/lens
    # buffer pattern of the same builder __init__ (B_max =
    # max(decode_cudagraph_max_bs, max_num_seqs)).
    assert "_fr13_ring_bs = int(self.fr10_tree_accepted_path_bs)" in text
    assert text.index("self.fr10_tree_accepted_paths = torch.zeros(") < text.index(
        "_fr13_layer._fr13_replay_ring_k = torch.zeros("
    )
    # Flag ON without a speculative token tree refuses at init (fail-loud).
    assert "FR13_REPLAY_ROUTE=1 requires a speculative token tree" in text
    # Layer registration happens ONCE at builder init, not per step.
    assert "_fr13_gdn_mod._FR13_REPLAY_LAYERS = _fr13_replay_layers" in text
    assert 'setdefault("_FR13_REPLAY_LAYERS"' not in text
    # The gate-4 dict mechanism must NOT be reused.
    assert "_FR10_PENDING_TREE_STATE_PUBLISH" not in text
    # Freshness handshake checks + clears in both committers.
    assert text.count("stale or missing scan-time") == 2
    assert text.count("_fr13_flags[0].fill_(0)") == 2


def test_forward_flagged_path_only_writes_never_allocates() -> None:
    frag = _flagged_forward_fragment()
    code = "\n".join(
        line for line in frag.splitlines() if not line.lstrip().startswith("#")
    )
    # No allocation of any kind in the (capturable) flagged forward path.
    for banned in (
        "torch.zeros(",
        "torch.empty(",
        "torch.ones(",
        "torch.tensor(",
        "torch.full(",
        ".clone(",
        ".new_",
    ):
        assert banned not in code, banned
    # Missing init-time buffers must RAISE, never fall back to allocating.
    assert "FR13_REPLAY_ROUTE: replay staging buffers " in frag
    assert "missing for layer" in frag
    assert "capture-unsafe lazy allocation" in frag
    # Byte-copy contract guard: shape/dtype mismatch raises.
    assert "staging ring shape/dtype" in frag
    # The forward only WRITES the persistent buffers.
    assert "self._fr13_replay_ring_k[fr10_b, :tree_n].copy_(" in frag
    assert "self._fr13_replay_prev_lens[" in frag
    assert "self._fr13_replay_spec_idx[" in frag


def test_no_per_step_object_creation_in_flagged_replay_path() -> None:
    # Broadened gate-4 ban (verify V5): not just the old
    # _FR10_PENDING_TREE_STATE_PUBLISH dict by name -- NO per-step
    # dict/list/set/tuple-builder or staging-object creation anywhere in the
    # flagged forward replay path; the handshake must be the persistent
    # preallocated mechanism.
    frag = _flagged_forward_fragment()
    code = "\n".join(
        line for line in frag.splitlines() if not line.lstrip().startswith("#")
    )
    for banned in (
        "= {",
        "{}",
        "dict(",
        ".setdefault(",
        "= []",
        "list(",
        "set(",
        "tuple(",
        "globals()",
        "_fr13_replay_meta",
    ):
        assert banned not in code, banned
    # The capture-safe handshake: preallocated int32 flag tensor written by
    # CAPTURED device ops (so CUDA-graph replay re-arms freshness), plus a
    # fixed attribute write of the existing persistent bank tensor.
    assert "self._fr13_replay_flags[0].fill_(1)" in code
    assert "self._fr13_replay_flags[1].fill_(" in code
    assert "self._fr13_replay_ssm_state = ssm_state" in code

    # The per-step meta dict is gone from the patcher entirely (comments may
    # mention it as the banned mechanism it replaced).
    text = PATCHER.read_text()
    assert "_fr13_replay_meta =" not in text
    assert "'_fr13_replay_meta'" not in text
    assert '"_fr13_replay_meta"' not in text
    # Committers consume the persistent handshake, per layer.
    assert text.count("_fr13_layer, '_fr13_replay_flags', None") == 2
    assert text.count("_fr13_layer, '_fr13_replay_ssm_state', None") == 2
    assert text.count("_fr13_layer._fr13_replay_output_scale") >= 2


def test_serving_wiring_gate_fails_loud_under_replay_route_flag() -> None:
    # Verify V3 remediation: fr10_serving_wiring_gate.py must FAIL LOUDLY
    # when FR13_REPLAY_ROUTE=1 instead of silently passing the vacuous
    # scan_state_staging (diag[12]/[13]) check.
    gtext = GATE.read_text()
    assert gtext.count("_refuse_if_replay_route()") >= 2  # evaluate_wiring + main
    assert 'os.environ.get("FR13_REPLAY_ROUTE", "0") == "1"' in gtext
    assert "REFUSES to run with FR13_REPLAY_ROUTE=1" in gtext
    # The refusal happens before any matrix row is computed.
    assert gtext.index("def _refuse_if_replay_route") < gtext.index(
        "def evaluate_wiring"
    )


def test_patcher_committer_replay_launch_in_both_committers() -> None:
    text = PATCHER.read_text()

    assert text.count("launch_tree_gdn_replay as _fr13_replay_launch") == 2
    assert text.count("_fr13_replay_launch(") == 2
    # Launch sits after the REQKEY publish inside each committer try block:
    # greedy launch -> greedy except (_fr10_tree_lcp_log_exc), then the
    # sampled launch -> the NEXT sampled except (_fr10_commit_globals_exc).
    first_launch = text.index("_fr13_replay_launch(")
    greedy_anchor = text.index("except Exception as _fr10_tree_lcp_log_exc:")
    assert first_launch < greedy_anchor
    second_launch = text.index("_fr13_replay_launch(", first_launch + 1)
    assert second_launch > greedy_anchor
    sampled_anchor = text.index(
        "except Exception as _fr10_commit_globals_exc:", second_launch
    )
    assert sampled_anchor > second_launch
    assert text.index("_LUMO_FA_TREE_ACCEPT_BY_REQ") < first_launch
    # Inputs are the persistent device buffers + per-layer staging.
    assert "accepted_paths=_accepted_path_buf," in text
    assert "accepted_lens=_accepted_lens_buf," in text
    assert "prev_lens=_fr13_layer._fr13_replay_prev_lens," in text
    assert "state_bank=_fr13_ssm_bank," in text
    assert "'FR13_REPLAY_ROUTE: no registered GDN replay layers'" in text


def test_patcher_remap_keeps_conv_half_and_drops_ssm_half_on_flag() -> None:
    text = PATCHER.read_text()

    needle = (
        "                    launch_tree_state_linear_remap(\n"
        "                        ssm_state=(\n"
        "                            None\n"
        '                            if os.environ.get("FR13_REPLAY_ROUTE", "0") == "1"\n'
        "                            else ssm_state\n"
        "                        ),\n"
        "                        conv_state=conv_state,"
    )
    assert needle in text
    # The conv carrier rationale is documented at the call site.
    assert "conv-prior-window carrier" in text


def test_replay_route_fragments_parse_and_everything_compiles() -> None:
    text = PATCHER.read_text()
    tree = ast.parse(text)
    checked = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and "FR13_REPLAY_ROUTE" in node.value.value
        ):
            ast.parse(textwrap.dedent(node.value.value))
            checked += 1
    # forward replacement, conv replacement, committer helper.
    assert checked >= 3
    # The builder-init staging-allocation fragment is a .replace() call arg
    # (not an Assign): find the folded string constant and parse it too.
    builder_fragments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "_fr13_replay_ring_k = torch.zeros(" in node.value
        and "FR13_REPLAY_ROUTE" in node.value
    ]
    assert len(builder_fragments) == 1
    ast.parse(textwrap.dedent(builder_fragments[0]))
    for path in (PATCHER, KERNEL, REFERENCE, GATE):
        py_compile.compile(str(path), doraise=True)


def test_flag_off_legacy_surfaces_intact() -> None:
    text = PATCHER.read_text()

    # The legacy publish, remap, lens plumbing and committer publishes are
    # all still present (flag unset => SOURCE-INERT default: the legacy text
    # is intact and every new behavior is flag-gated; compile-identity is
    # NOT claimed -- the shared _gdn_node_step body refactor recompiles the
    # default scan, so it is pending the refactored-scan byte A/B, GPU TODO
    # #2 in FR13_REPLAY_ROUTE_BUILD.md).
    assert "ssm_state.index_copy_(" in text
    assert "launch_tree_state_linear_remap(" in text
    assert "h0_num_accepted_tokens=_fr10_accepted_lens_tensor" in text
    assert "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR" in text
    assert "FR13_TREE_REQKEY" in text
    assert "FR13_TREE_PER_REQ_GEN" in text
