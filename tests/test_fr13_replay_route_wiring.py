"""FR13 replay-route wiring tests (CPU, text-level, style of
test_fr10_phase4_sampled_committer_wiring.py).

Flag ON wiring: replay launch present in BOTH committers, scratch alloc and
all-rows publish flag-gated off, scan-time prev-lens snapshot precedes the
scan launch, ssm remap half gated off (conv half kept), persistent staging
only. Flag OFF: the legacy path is textually intact and every new behavior
sits behind an FR13_REPLAY_ROUTE check that defaults to "0".
"""

from __future__ import annotations

import ast
import py_compile
import textwrap
from pathlib import Path

PATCHER = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
KERNEL = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
REFERENCE = Path("src/lumo_flywheel_serving/fr13_replay_reference.py")


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
    # Default-ON call sites keep legacy behavior byte-identical.
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


def test_patcher_staging_is_persistent_not_per_step_dict() -> None:
    text = PATCHER.read_text()

    # Preallocated persistent per-layer ring + snapshot buffers.
    for buf in (
        "self._fr13_replay_ring_k = torch.zeros(",
        "self._fr13_replay_ring_v = torch.zeros(",
        "self._fr13_replay_ring_a = torch.zeros(",
        "self._fr13_replay_ring_b = torch.zeros(",
        "self._fr13_replay_prev_lens = torch.zeros(",
        "self._fr13_replay_spec_idx = torch.zeros(",
    ):
        assert buf in text, buf
    # Sized to the persistent accepted-lens buffer batch (request-keyed,
    # decode_cudagraph_max_bs/max_num_seqs), not the per-step batch.
    assert "_fr13_ring_bs = int(_fr10_accepted_lens_tensor.size(0))" in text
    # The gate-4 dict mechanism must NOT be reused.
    assert "_FR10_PENDING_TREE_STATE_PUBLISH" not in text
    # Freshness assertions on the staged payload.
    assert text.count("stale or missing scan-time") == 2
    assert text.count("_fr13_meta['fresh'] = False") == 2


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
    assert "state_bank=_fr13_meta['ssm_state']," in text
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
    for path in (PATCHER, KERNEL, REFERENCE):
        py_compile.compile(str(path), doraise=True)


def test_flag_off_legacy_surfaces_intact() -> None:
    text = PATCHER.read_text()

    # The legacy publish, remap, lens plumbing and committer publishes are
    # all still present (flag unset => byte-identical default behavior).
    assert "ssm_state.index_copy_(" in text
    assert "launch_tree_state_linear_remap(" in text
    assert "h0_num_accepted_tokens=_fr10_accepted_lens_tensor" in text
    assert "_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR" in text
    assert "FR13_TREE_REQKEY" in text
    assert "FR13_TREE_PER_REQ_GEN" in text
