"""FR13 replay-route byte-A/B vehicle port (gate-transfer matrix ADDENDUM
rider 2; CPU, text-level, style of test_fr13_replay_route_wiring.py).

Gate A's own vehicle, scripts/fr10_tree_kernel_h0_ab_replay.py, reads
``serving_tree_state`` from the COMMIT_HANDOFF capture -- a field sourced
from the per-node scratch (``tree_state_all``) that FR13_REPLAY_ROUTE=1
deletes. The byte A/B therefore runs in the STORE_NODE_STATES=True
diagnostic mode (FR13_REPLAY_ROUTE unset/0), and that mode MUST retain:

1. the legacy per-node scratch alloc + export-enabled scan launch,
2. the COMMIT_HANDOFF capture path that publishes ``serving_tree_state``,
3. the FR10_TREE_GDN_CAPTURE_PAYLOAD harness fields that double as the
   replay activation store (k pre-l2norm / v / raw_a / raw_b / h0 --
   gate-transfer matrix instrument-breaks #5).

These tests pin all three so a later cleanup cannot silently break the
one-time byte A/B (the validity switch for every CONDITIONAL gate item).
"""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

PATCHER = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
KERNEL = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
VEHICLE = Path("scripts/fr10_tree_kernel_h0_ab_replay.py")


def _src_native_save_block() -> str:
    """The torch.save dict of the fr10.src_native_handoff_payload.v1 payload
    (the COMMIT_HANDOFF capture the A/B vehicle consumes)."""
    text = PATCHER.read_text()
    start = text.index('"schema": "fr10.src_native_handoff_payload.v1"')
    end = text.index("_fr10_src_native_path,", start)
    return text[start:end]


def _capture_payload_save_block() -> str:
    """The torch.save dict of the fr10.tree_gdn_scan_capture.v1 payload
    (the FR10_TREE_GDN_CAPTURE_PAYLOAD harness)."""
    text = PATCHER.read_text()
    start = text.index('"schema": "fr10.tree_gdn_scan_capture.v1"')
    end = text.index("_fr10_payload_path,", start)
    return text[start:end]


def test_store_node_states_true_mode_retains_commit_handoff_capture() -> None:
    text = PATCHER.read_text()

    # Flag-OFF keeps the export-enabled scan: store_node_states is wired to
    # the inverse of the route flag, so the STORE_NODE_STATES=True
    # diagnostic mode is exactly FR13_REPLAY_ROUTE unset/0.
    assert "store_node_states=not _fr13_replay_route_on" in text
    # Flag-OFF allocates the per-node scratch the capture is sourced from.
    assert "tree_state_all = torch.empty(" in text
    assert (
        "None if _fr13_replay_route_on else tree_state_all[fr10_b]" in text
    )
    # The COMMIT_HANDOFF capture stages the scratch rows (tree_state_cpu)...
    assert '"tree_state_cpu": tree_state[' in text
    # ...and the payload publishes them as serving_tree_state, the exact
    # field fr10_tree_kernel_h0_ab_replay.py reads.
    block = _src_native_save_block()
    assert '"serving_tree_state": _fr10_prev_read[' in block
    assert '"tree_state_cpu"' in block


def test_replay_route_refuses_every_scratch_consuming_capture_env() -> None:
    """Fail-loud: under FR13_REPLAY_ROUTE=1 the scratch does not exist, so
    every env that captures/splices it must raise (never silently write
    garbage payloads). The A/B captures are taken with the flag OFF."""
    text = PATCHER.read_text()
    start = text.index("FR13_REPLAY_ROUTE is incompatible with tree_state")
    # The guard block sits just above the raise message.
    guard = text[max(0, start - 1500):start]
    for env in (
        "FR10_TREE_GDN_CAPTURE_PAYLOAD",
        "FR10_TREE_GDN_COMMIT_HANDOFF_LOG",
        "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD",
        "FR12_TREE_SCAN_NATIVE_SPINE",
    ):
        assert env in guard, f"replay-route capture guard must cover {env}"


def test_capture_payload_harness_saves_the_replay_inputs() -> None:
    """Matrix instrument-breaks #5: the capture payload's INPUT fields are
    exactly the replay activation store, so the harness doubles as the
    activation-store byte-check and the byte-A/B vehicle."""
    block = _capture_payload_save_block()
    # k pre-l2norm / v / raw_a / raw_b / h0 -- the replay kernel's inputs.
    assert '"key_spec": key_spec[0, start:end]' in block
    assert '"value_tree": value_tree[start:end]' in block
    assert '"a": a[start:end].detach().cpu().clone()' in block
    assert '"b": b[start:end].detach().cpu().clone()' in block
    assert '"h0": _fr10_capture_h0' in block
    # The scan halves of Gate A (old-vs-new binary out equality + the
    # exported per-node states).
    assert '"serving_out": tree_out[:tree_n]' in block
    assert '"serving_state": tree_state[:tree_n]' in block


def test_ab_vehicle_payload_keys_are_all_captured() -> None:
    """Every required payload key the vehicle reads must be written by the
    COMMIT_HANDOFF (src-native handoff) capture."""
    vtext = VEHICLE.read_text()
    required = set(re.findall(r'payload\["([a-z_A-Z0-9]+)"\]', vtext))
    # Keys read through the node-major helper.
    required |= set(
        re.findall(r'_node_major_spec_tensor\(payload, "([a-z_A-Z0-9]+)"\)', vtext)
    )
    # payload.get(...) reads are optional by construction (guarded by an
    # is-not-None check before any indexed read) -- not required.
    optional = set(re.findall(r'payload\.get\("([a-z_A-Z0-9]+)"\)', vtext))
    required -= optional
    assert "serving_tree_state" in required  # the rider-2 load-bearing field
    block = _src_native_save_block()
    saved = set(re.findall(r'"([a-z_A-Z0-9]+)":', block))
    missing = sorted(required - saved)
    assert not missing, (
        "COMMIT_HANDOFF capture is missing payload keys the byte-A/B "
        f"vehicle reads: {missing}"
    )


def test_ab_vehicle_compiles_and_uses_export_enabled_launcher() -> None:
    py_compile.compile(str(VEHICLE), doraise=True)
    vtext = VEHICLE.read_text()
    # The vehicle relies on the default export-enabled launch returning the
    # per-node states (store_node_states defaults True).
    assert "launch_tree_gdn_prepared(" in vtext
    assert "store_node_states=False" not in vtext
    assert 'payload["serving_tree_state"]' in vtext
    ktext = KERNEL.read_text()
    assert "store_node_states: bool = True" in ktext
