from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


_FR13_COMMITTER_NATIVE_ANNOUNCED = False
_FR13_FIXED32_COMMITTER_LAYER_BATCH_REAL_EVENT = (
    "/logs/fr13_fixed32_committer_layer_batch.real_event.arm"
)
# Both fixed32 logical modes retain the full depth-11 Tail spine.  The
# accepted_lens value counts accepted drafts only; the committer adds the root
# internally, while columns 12..15 are storage padding and are unreachable.
_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH = 11
_FR13_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK = (
    1 << (_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH + 1)
) - 1
_FR13_FIXED32_COMMITTER_TASK_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)
_FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_ARMS = (
    "/logs/fr13_fixed32_conv_commit_zero_tail.arm",
)
_FR13_FIXED32_PHYSICAL_PARENT = (
    -1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9,
    9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29, 30,
)
_FR13_FIXED32_TREECONV_MODE_IDENTITY = {
    "tail6_fixed32": ("Tail23", 0x7A9CE7FF),
    "hydra27_fixed32": ("Hydra27", 0x7ABDFFFF),
}
_FR13_FIXED32_TREECONV_RECORD_SCHEMA = (
    "fr13.fixed32.treeconv_zero_tail.byte_ab.v2"
)
_FR13_FIXED32_TREECONV_TERMINAL_SCHEMA = (
    "fr13.fixed32.treeconv_zero_tail.byte_ab_terminal.v2"
)


def _fr13_fixed32_treeconv_canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _fr13_fixed32_treeconv_expected_state_src() -> tuple[int, ...]:
    paths: list[list[int]] = []
    for node in range(len(_FR13_FIXED32_PHYSICAL_PARENT)):
        path: list[int] = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = _FR13_FIXED32_PHYSICAL_PARENT[cursor]
        paths.append(list(reversed(path)))
    values: list[int] = []
    width = 4
    state_length = 34
    zero_row = 36 - 1
    for path in paths:
        path_length = len(path)
        for state_col in range(state_length):
            position = path_length + state_col
            if position < width - 1:
                values.append(position)
            elif state_col < width - 1:
                values.append(width - 1 + path[position - (width - 1)])
            else:
                values.append(zero_row)
    return tuple(values)


_FR13_FIXED32_TREECONV_STATE_SRC = (
    _fr13_fixed32_treeconv_expected_state_src()
)
_FR13_FIXED32_TREECONV_STATE_SRC_SHA256 = hashlib.sha256(
    _fr13_fixed32_treeconv_canonical_json(_FR13_FIXED32_TREECONV_STATE_SRC)
).hexdigest()


def _fr13_fixed32_treeconv_topology_descriptor(mode: str) -> dict[str, object]:
    try:
        logical_topology, valid_mask = _FR13_FIXED32_TREECONV_MODE_IDENTITY[mode]
    except KeyError as error:
        raise RuntimeError(f"unsupported fixed32 tree-conv mode {mode!r}") from error
    return {
        "schema": "fr13.fixed32.treeconv_state_descriptor.v1",
        "mode": mode,
        "logical_topology": logical_topology,
        "valid_mask": valid_mask,
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "conv_width": 4,
        "conv_state_length": 34,
        "source_rows_per_request": 36,
        "live_state_columns": 3,
        "physical_parent_sha256": hashlib.sha256(
            _fr13_fixed32_treeconv_canonical_json(
                _FR13_FIXED32_PHYSICAL_PARENT
            )
        ).hexdigest(),
        "state_src_sha256": _FR13_FIXED32_TREECONV_STATE_SRC_SHA256,
    }


def _fr13_fixed32_conv_commit_zero_tail_requested() -> bool:
    """Resolve the default-off fixed32 conv zero-tail specialization."""
    raw = os.environ.get("FR13_FIXED32_CONV_COMMIT_ZERO_TAIL", "0")
    if raw not in ("0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL must be exactly 0 or 1"
        )
    return raw == "1" or any(
        os.path.exists(path)
        for path in _FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_ARMS
    )


def _fr13_fixed32_conv_commit_zero_tail_byte_ab_requested() -> bool:
    """Resolve the eager stock-serving zero-tail byte diagnostic."""
    raw = os.environ.get(
        "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB", "0"
    )
    if raw not in ("0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB must be exactly 0 or 1"
        )
    return raw == "1"


def _fr13_fixed32_treeconv_comparison_limit() -> int:
    text = os.environ.get(
        "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB_LIMIT", "320"
    )
    try:
        limit = int(text)
    except ValueError as error:
        raise RuntimeError(
            "FR13 fixed32 conv zero-tail byte A/B limit must be an integer"
        ) from error
    if not 1 <= limit <= 320:
        raise RuntimeError(
            "FR13 fixed32 conv zero-tail byte A/B limit must be in [1, 320]"
        )
    return limit


def _fr13_fixed32_treeconv_scalar(
    state: dict[str, object], key: str
) -> torch.Tensor:
    value = state.get(key)
    anchor = state.get("anchor")
    if (
        not torch.is_tensor(value)
        or not torch.is_tensor(anchor)
        or value.device != anchor.device
        or value.dtype != torch.int64
        or value.ndim != 0
        or not value.is_contiguous()
    ):
        raise RuntimeError(f"FR13 fixed32 tree-conv scalar {key!r} drifted")
    return value


def fixed32_conv_zero_tail_live_prepare_replay(
    *, mode: str, batch_size: int, enabled: bool
) -> None:
    """Order the graph-resident comparator enable around one measured replay."""
    requested = _fr13_fixed32_conv_commit_zero_tail_byte_ab_requested()
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if not requested:
        if isinstance(state, dict) and state.get("commit_zero_tail_byte_ab"):
            raise RuntimeError("FR13 fixed32 tree-conv diagnostic state leaked")
        return
    if type(enabled) is not bool or not isinstance(state, dict):
        raise RuntimeError("FR13 fixed32 tree-conv replay state is unavailable")
    batch = int(batch_size)
    active = bool(state.get("treeconv_zero_tail_replay_active", False))
    if (
        state.get("commit_zero_tail_byte_ab") is not True
        or state.get("commit_zero_tail") is not False
        or state.get("mode") != mode
        or mode != _FR13_FIXED32_MODE
        or batch not in tuple(state.get("preseeded_batches", ()))
        or state.get("treeconv_topology_descriptor")
        != _fr13_fixed32_treeconv_topology_descriptor(mode)
        or enabled == active
    ):
        raise RuntimeError(
            "FR13 fixed32 tree-conv replay identity/lifecycle drift"
        )
    _fr13_fixed32_treeconv_scalar(
        state, "treeconv_zero_tail_count_enable"
    ).fill_(1 if enabled else 0)
    state["treeconv_zero_tail_replay_active"] = enabled


def fixed32_conv_zero_tail_live_finalize(
    events: object, flush_binding: object
) -> None:
    """Persist the graph comparator only after the authenticated final flush."""
    requested = _fr13_fixed32_conv_commit_zero_tail_byte_ab_requested()
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if not requested:
        if isinstance(state, dict) and state.get("commit_zero_tail_byte_ab"):
            raise RuntimeError("FR13 fixed32 tree-conv finalizer state leaked")
        return
    event_rows = list(events) if isinstance(events, (list, tuple)) else []
    binding = flush_binding if isinstance(flush_binding, dict) else {}
    hex_chars = frozenset("0123456789abcdef")
    if (
        not isinstance(state, dict)
        or state.get("commit_zero_tail_byte_ab") is not True
        or state.get("treeconv_zero_tail_replay_active") is not False
        or not event_rows
        or len(event_rows) > _fr13_fixed32_treeconv_comparison_limit()
        or set(binding)
        != {
            "action",
            "boundary_snapshot_sha256",
            "complete_work_census_events",
            "events_sha256",
            "generation",
            "nonce",
            "producer_pid",
        }
        or binding.get("action") != "final"
        or type(binding.get("generation")) is not int
        or int(binding["generation"]) < 1
        or type(binding.get("producer_pid")) is not int
        or int(binding["producer_pid"]) < 1
        or int(binding.get("complete_work_census_events", -1))
        != len(event_rows)
        or any(
            not isinstance(binding.get(key), str)
            or len(binding[key]) != 64
            or any(character not in hex_chars for character in binding[key])
            for key in (
                "boundary_snapshot_sha256",
                "events_sha256",
                "nonce",
            )
        )
    ):
        raise RuntimeError("FR13 fixed32 tree-conv finalization drifted")

    compared_events = int(
        _fr13_fixed32_treeconv_scalar(
            state, "treeconv_zero_tail_compared_events"
        ).item()
    )
    differing_bytes = int(
        _fr13_fixed32_treeconv_scalar(
            state, "treeconv_zero_tail_differing_bytes"
        ).item()
    )
    if compared_events != len(event_rows) or differing_bytes != 0:
        raise RuntimeError(
            "FR13 fixed32 tree-conv graph comparison failed: "
            + repr(
                {
                    "expected_events": len(event_rows),
                    "compared_events": compared_events,
                    "differing_bytes": differing_bytes,
                }
            )
        )

    descriptor = state.get("treeconv_topology_descriptor")
    mode = state.get("mode")
    output_rows: list[dict[str, object]] = []
    total_compared_bytes = 0
    for expected_index, event in enumerate(event_rows):
        runtime = event.get("drafter_runtime") if isinstance(event, dict) else None
        request_digests = (
            runtime.get("request_id_sha256s")
            if isinstance(runtime, dict)
            else None
        )
        batch = event.get("batch_size") if isinstance(event, dict) else None
        compared_bytes = 48 * int(batch or 0) * 10240 * 34 * 2
        if (
            not isinstance(event, dict)
            or event.get("schema") != "fr13-fixed32-work-census-v12"
            or event.get("event_complete") is not True
            or event.get("mode") != mode
            or event.get("event_index") != expected_index
            or event.get("producer_pid") != binding["producer_pid"]
            or type(event.get("forward_step_index")) is not int
            or event["forward_step_index"] < 0
            or batch not in (1, 2, 3, 4)
            or not isinstance(request_digests, list)
            or len(request_digests) != batch
            or len(set(request_digests)) != batch
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in hex_chars for character in value)
                for value in request_digests
            )
            or runtime.get("request_ids_sha256") is None
        ):
            raise RuntimeError(
                "FR13 fixed32 tree-conv work-event binding drifted at "
                + str(expected_index)
            )
        output_rows.append(
            {
                "schema": _FR13_FIXED32_TREECONV_RECORD_SCHEMA,
                "mode": mode,
                "event_id": event.get("event_id"),
                "event_index": expected_index,
                "forward_step_index": event["forward_step_index"],
                "producer_pid": binding["producer_pid"],
                "batch_size": batch,
                "request_ids_sha256": runtime["request_ids_sha256"],
                "request_id_sha256s": list(request_digests),
                "execution_basis": "cudagraph_full_replay",
                "topology": descriptor,
                "conv_layers": 48,
                "conv_channels": 10240,
                "conv_state_length": 34,
                "source_rows_per_request": 36,
                "candidate_zero_tail": True,
                "reference_zero_tail": False,
                "reference_restored_and_served": True,
                "raw_bf16_byte_comparison": True,
                "compared_bytes": compared_bytes,
                "differing_bytes": 0,
                "byte_equal": True,
                "timing_eligible": False,
            }
        )
        total_compared_bytes += compared_bytes
    body_sha256 = hashlib.sha256(
        _fr13_fixed32_treeconv_canonical_json(output_rows)
    ).hexdigest()
    output_rows.append(
        {
            "schema": _FR13_FIXED32_TREECONV_TERMINAL_SCHEMA,
            "status": "PASS",
            "mode": mode,
            "topology": descriptor,
            "complete_work_census_events": len(event_rows),
            "first_event_index": 0,
            "last_event_index": len(event_rows) - 1,
            "first_forward_step_index": event_rows[0]["forward_step_index"],
            "last_forward_step_index": event_rows[-1]["forward_step_index"],
            "producer_pid": binding["producer_pid"],
            "counted_graph_replays": compared_events,
            "total_compared_bytes": total_compared_bytes,
            "total_differing_bytes": differing_bytes,
            "comparison_records_sha256": body_sha256,
            "work_census_events_sha256": binding["events_sha256"],
            "flush_generation": binding["generation"],
            "flush_nonce": binding["nonce"],
            "boundary_snapshot_sha256": binding[
                "boundary_snapshot_sha256"
            ],
            "flush_action": "final",
            "finalized_by_fixed32_flush": True,
            "reference_always_served": True,
            "timing_eligible": False,
        }
    )
    path = Path(
        os.environ.get(
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB_PATH",
            "/logs/fr13_fixed32_treeconv_zero_tail.byte_ab.jsonl",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "w", encoding="ascii") as handle:
        for row in output_rows:
            handle.write(
                _fr13_fixed32_treeconv_canonical_json(row).decode("ascii")
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fr13_fixed32_committer_layer_batch_requested() -> bool:
    """Return the boot-time arm for the experimental one-launch committer.

    The EngineCore worker may receive a curated environment, so the sidecar is
    the serving-safe arm.  The value is sampled while fixed32 graphs are
    preseeded; adding or removing the arm after boot cannot change the route.
    """
    if os.environ.get("FR13_FIXED32_COMMITTER_LAYER_BATCH") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_layer_batch.arm",
            "/tmp/fr13_fixed32_committer_layer_batch.arm",
        )
    )


def _fr13_fixed32_committer_bv64_warp4_requested() -> bool:
    """Return the boot-time arm for the two-tile committer geometry."""
    if os.environ.get("FR13_FIXED32_COMMITTER_BV64_WARP4") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_bv64_warp4.arm",
            "/tmp/fr13_fixed32_committer_bv64_warp4.arm",
        )
    )


def _fr13_fixed32_committer_metadata_fusion_requested() -> bool:
    """Return the boot-time arm for conv-to-committer metadata fusion."""
    if os.environ.get("FR13_FIXED32_COMMITTER_METADATA_FUSION") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_metadata_fusion.arm",
            "/tmp/fr13_fixed32_committer_metadata_fusion.arm",
        )
    )


def _fr13_fixed32_committer_direct_metadata_requested() -> bool:
    """Return the boot-time arm for direct persistent committer metadata."""
    if os.environ.get("FR13_FIXED32_COMMITTER_DIRECT_METADATA") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_direct_metadata.arm",
            "/tmp/fr13_fixed32_committer_direct_metadata.arm",
        )
    )


def _fr13_fixed32_committer_sticky_guard_requested() -> bool:
    """Return the boot-time arm for the reduction-free committer guard."""
    if os.environ.get("FR13_FIXED32_COMMITTER_STICKY_GUARD") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_sticky_guard.arm",
            "/tmp/fr13_fixed32_committer_sticky_guard.arm",
        )
    )


def _fr13_fixed32_committer_knorm_ring_requested() -> bool:
    """Return the boot-time arm for producer-reused K normalization."""
    if os.environ.get("FR13_FIXED32_COMMITTER_KNORM_RING") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_knorm_ring.arm",
            "/tmp/fr13_fixed32_committer_knorm_ring.arm",
        )
    )


def _fr13_fixed32_committer_gate_ring_requested() -> bool:
    """Return the boot-time arm for producer-reused recurrence gates."""
    if os.environ.get("FR13_FIXED32_COMMITTER_GATE_RING") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_gate_ring.arm",
            "/tmp/fr13_fixed32_committer_gate_ring.arm",
        )
    )


def _fr13_fixed32_committer_decay_ring_requested() -> bool:
    """Return the boot-time arm for producer-reused decay multipliers."""
    if os.environ.get("FR13_FIXED32_COMMITTER_DECAY_RING") == "1":
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_fixed32_committer_decay_ring.arm",
            "/tmp/fr13_fixed32_committer_decay_ring.arm",
        )
    )


def _fr13_fixed32_committer_layer_batch_real_event_marker(
    path: str = _FR13_FIXED32_COMMITTER_LAYER_BATCH_REAL_EVENT,
) -> str | None:
    """Read the active authenticated SWE-Verified qualification arm."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(
            "FR13 fixed32 committer cannot open its real-event arm"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise RuntimeError(
                "FR13 fixed32 committer real-event arm must be a private "
                "read-only regular file"
            )
        with os.fdopen(descriptor, encoding="ascii", closefd=False) as handle:
            marker = handle.read(257)
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            "FR13 fixed32 committer cannot read its real-event arm"
        ) from error
    finally:
        os.close(descriptor)
    if len(marker) > 256:
        raise RuntimeError(
            "FR13 fixed32 committer real-event arm exceeds 256 bytes"
        )
    marker = marker.strip()
    prefix = "swe_verified:"
    task_id = marker[len(prefix):] if marker.startswith(prefix) else ""
    if not task_id or any(
        character not in _FR13_FIXED32_COMMITTER_TASK_ID_CHARACTERS
        for character in task_id
    ):
        raise RuntimeError(
            "FR13 fixed32 committer real-event arm must be "
            "swe_verified:<task_id>"
        )
    return marker


def _fr13_fixed32_committer_accepted_length_mask(
    accepted_lengths, *, batch: int,
) -> int:
    """Encode one B-row event's reachable accepted draft lengths."""
    lengths = tuple(int(value) for value in accepted_lengths)
    if len(lengths) != batch or any(
        length < 0
        or length > _FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH
        for length in lengths
    ):
        raise RuntimeError(
            "FR13 fixed32 committer accepted-length qualification input drift"
        )
    mask = 0
    for length in lengths:
        mask |= 1 << length
    return mask


def _fr13_committer_native_on() -> bool:
    """FR13_COMMITTER_NATIVE, worker-env-drop-proof: the EngineCore worker gets a curated env that drops
    FR13_* vars, so os.environ is unreliable at worker runtime. The launcher (pid-1, env present) writes a
    sidecar flag file; read that OR the env (for offline/eager use). Read fresh each call (cheap stat)."""
    if os.environ.get("FR13_COMMITTER_NATIVE") == "1":
        return True
    for _p in ("/logs/fr13_committer_native.flag", "/tmp/fr13_committer_native.flag"):
        if os.path.exists(_p):
            return True
    return False

NODE_FAMILIES = (2, 3, 6, 8, 14)
QK_HEADS = 16
V_HEADS = 48
# Legacy equal-head synthetic default used by the Phase 2 microbench.
H = V_HEADS
K = 128
V = 128
BV = 16
# Deployed launch geometry for the served tree-scan: num_warps=8, num_stages
# = Triton default (left unset). These are the locked cat9 serving values.
_DEPLOYED_NUM_WARPS = 8


def _read_tree_gdn_geom_override() -> dict | None:
    """TEST-ONLY geometry override for the tree-GDN scan launch.

    Reads ``FR13_TREE_GDN_GEOM_OVERRIDE`` (e.g. ``"BV=32,num_warps=4,
    num_stages=3"``). This DOES NOT change any value/math the kernel computes;
    it only changes the Triton launch geometry (BLOCK_V tiling, warp count,
    pipeline stages) so the BV/warps A/B arms can be measured against the
    native packed-decode SASS. When the env var is unset (the deployed
    default), this returns ``None`` and the served launch is BYTE-IDENTICAL to
    the prior locked path (num_warps=8, BLOCK_V=BV, num_stages unset). The
    override is a diagnostics lever (bug-class #10 codegen-identity A/B), never
    a served-path value change.
    """
    raw = os.environ.get("FR13_TREE_GDN_GEOM_OVERRIDE")
    if not raw:
        return None
    out: dict = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"FR13_TREE_GDN_GEOM_OVERRIDE token {token!r} must be key=value"
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in ("BV", "num_warps", "num_stages"):
            raise ValueError(
                f"FR13_TREE_GDN_GEOM_OVERRIDE key {key!r} not in "
                "{BV, num_warps, num_stages}"
            )
        out[key] = int(value)
    return out or None




def scan_align_on() -> bool:
    """Whether the FR13_SCAN_ALIGN body-seam alignment is enabled.

    Default OFF: when ``FR13_SCAN_ALIGN`` is unset/0/false the served scan is
    BYTE-IDENTICAL to the locked cat9 path -- the SCAN_ALIGN constexpr threads
    through ``_gdn_node_step`` as ``False`` and every aligned branch is dead
    code (no codegen change, bug-class #10). The flag turns on the two body
    seams (l2norm div-by-sqrt, beta bf16 round-trip) that align our rank-1
    node update to the native packed-decode SASS.
    """
    raw = os.environ.get("FR13_SCAN_ALIGN", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def parent_gather_on() -> bool:
    """Collapse the O(N^2) ancestor gather to a single per-node parent gather.

    Default OFF => BYTE-IDENTICAL to the locked forward scan. The deployed inner
    loop does one full-tile masked ``tl.sum`` reduction PER ancestor j<i, then
    ``state_i = tl.where(ancestor, h_j, state_i)`` -- overwriting in increasing j,
    so ONLY the largest-index ancestor survives. In the topological node order
    (parent stored before child; the write at h_cache row i is read by children
    i'>i) the largest-index ancestor IS the immediate parent. When this flag is
    set the scan instead locates that same parent with a cheap INTEGER mask scan
    (no full-tile reduction) and issues a SINGLE gather for its state -- the
    identical one-hot masked ``tl.sum`` primitive selecting the identical row
    (0.0 + x = x), so the bits are unchanged (bug-class #10 codegen-identity),
    while the reduction count on the serial critical path drops N(N-1)/2 -> N
    (7.5x fewer at N_PAD=16, 15.5x at N_PAD=32).
    """
    raw = os.environ.get("FR13_PARENT_GATHER", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def parent_gather_selfcheck_on() -> bool:
    """In-process byte-identity gate for FR13_PARENT_GATHER.

    When set, ``launch_tree_gdn_prepared`` runs the scan BOTH ways on the SAME
    inputs in the SAME process (old ancestor loop -> out_ref, new parent gather
    -> out) and raises on any ``out`` bit difference. Boot under enforce_eager:
    the host-side compare forces a DtoH sync that would break CUDA-graph capture.
    This is the same-boot codegen-identity gate (cross-boot byte gates fork on
    GB10 autotune, so an in-process A/B is the only valid byte gate here).
    """
    raw = os.environ.get("FR13_PARENT_GATHER_SELFCHECK", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")






# FR13_NPAD_INVARIANT: the canonical N_PAD-invariant reduction order.
# The deployed cat9 tree has N_PAD = 1 << (9-1).bit_length() = 16; smaller trees
# (e.g. the leaf-free chain5 spine, N_PAD = 8) recompile the scan/reduction loop
# spans (`tl.static_range(0, N_PAD)`, `offs_n = tl.arange(0, N_PAD)`, the parent
# `tl.sum(tl.where(offs_n == j, ...))` reduction) to a SMALLER unrolled FMA tree,
# so the SAME spine node gets a different rounding order (bug-class #10
# codegen-identity; MEASURED state gap 0.0289). The fix pins the loop bound +
# offs_n lane count to this fixed N_FIXED for ALL tree sizes so the reduction
# order is identical regardless of how many leaves co-reside. The existing
# `< N_ACTUAL` masks make the [N_ACTUAL, N_FIXED) lanes contribute exactly 0.0
# (tl.where(...) already does this), so the computed value is unchanged and only
# the codegen/order is canonicalized. This is COMPUTE-ONLY: no copy, no HBM
# staging, no geometry change (num_warps stays the deployed 8).
_FR13_N_FIXED = 16


def npad_invariant_on() -> bool:
    """Whether the FR13_NPAD_INVARIANT canonical reduction order is enabled.

    Default OFF: when ``FR13_NPAD_INVARIANT`` is unset/0/false the kernel loops
    over the per-tree ``N_PAD`` (the locked cat9 served path). When ON, the scan
    loop bound, ``offs_n`` lane count, and ``h_cache`` row span are pinned to the
    fixed ``_FR13_N_FIXED`` (= the deployed cat9 ``N_PAD`` = 16) for every tree
    size, canonicalizing the scan/reduction FMA order across tree sizes. The
    ``< N_ACTUAL`` masks are kept, so inactive lanes contribute exact 0.0 and the
    computed value is byte-unchanged for the deployed cat9 tree (whose N_PAD is
    already 16); only smaller trees see the reduction order pinned. Default OFF
    => the ``N_LOOP`` constexpr equals ``N_PAD`` and the served launch is
    byte-identical to the prior locked path (bug-class #10 constexpr-dead).
    """
    raw = os.environ.get("FR13_NPAD_INVARIANT", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def hc_internal_on() -> bool:
    """FR13_HC_INTERNAL: compact the scan's h_cache to INTERNAL-node rows only.

    The one-hot ancestor reads (``tl.sum(tl.where(offs == j, h_cache, 0.0))``)
    can only ever SELECT a row whose node appears in some strict_mask ancestry
    row -- i.e. a node with children. Leaf rows are written at :1274 but never
    re-read, so caching them is pure register/local-memory waste (the measured
    BV wall: h_cache = [N_SPAN, BLOCK_V, DIM_K] fp32 tiles). When ON, the
    launcher derives the internal-node set from the static tree descriptor
    once (host-side, outside capture), packs a trace-time slot map, and the
    kernel keeps HC_ROWS (= next-pow2 internal count) rows instead of N_SPAN.
    Values are identical: internal reads select the identical state through
    the identical one-hot primitive (0.0 + x = x), and a leaf j's skipped
    iteration only removed a select whose runtime ``ancestor`` bit is always 0.
    Default OFF => HC_MASK=0 => every compacted branch is trace-time dead and
    the served launch is byte-identical to the locked path (bug-class #10).
    Incompatible with FR13_PARENT_GATHER (runtime parent index cannot use the
    trace-time slot map) and PIGGYBACK_EXPORT (chain end may be a leaf); the
    launcher fails loud on either pairing.

    RETIRED 2026-07-27 (cleanup+bake, FR13_CLEANUP_BAKE_PLAN.md): never gated
    in, and PARENT_GATHER — with which it is incompatible — is now the BAKED
    default. Mechanism hard-disabled below (env wiring also removed from the
    launchers). The HC_MASK constexpr branches in the kernel body remain as
    trace-time-dead code; their excision is a recorded follow-up requiring its
    own bit-exact gate cycle.
    """
    return False


def hc_internal_selfcheck_on() -> bool:
    """In-process byte-identity gate for FR13_HC_INTERNAL.

    Same discipline as ``parent_gather_selfcheck_on``: run the scan BOTH ways
    on the SAME inputs in the SAME process (full-span h_cache -> out_ref,
    compacted -> out) and raise on any bit difference. Boot under
    enforce_eager: the host compare forces a DtoH sync. The graph-capture leg
    is gated separately (gate_live pattern) per the eager-vs-graph lesson.
    """
    raw = os.environ.get("FR13_HC_INTERNAL_SELFCHECK", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


_FR13_HC_DESC_CACHE: dict = {}


def _hc_pack(parents: set, n_actual: int):
    """Pack an internal-node set into (mask, rows, slots_lo, slots_hi)."""
    mask = 0
    slots_lo = 0
    slots_hi = 0
    slot = 0
    for node in sorted(parents):
        if node >= 32:
            raise RuntimeError(
                f"FR13_HC_INTERNAL: internal node index {node} >= 32 unsupported"
            )
        mask |= 1 << node
        if node < 16:
            slots_lo |= slot << (4 * node)
        else:
            slots_hi |= slot << (4 * (node - 16))
        slot += 1
    if slot > 16:
        raise RuntimeError(
            f"FR13_HC_INTERNAL: {slot} internal nodes exceed the 4-bit slot map"
        )
    rows = 1
    while rows < slot:
        rows *= 2
    out = (mask, rows, slots_lo, slots_hi) if mask else (0, 0, 0, 0)
    print(
        f"[FR13_HC_INTERNAL] derived: n_actual={n_actual} internal={slot} "
        f"mask=0x{mask:x} rows={rows}",
        flush=True,
    )
    return out


def subtree_parallel_on() -> bool:
    """FR13_SUBTREE_PARALLEL: path-decomposed tree scan (queue task #60).

    The monolithic scan serializes ALL nodes inside every program. A tree
    decomposes into vertex-disjoint PATHS (heavy-path style): level-0 = the
    heavy path from the root; level-k paths hang off earlier levels. Paths
    within a level have no data dependence -> they scan CONCURRENTLY on a
    grid axis (one launch per level; tail6 = 2 launches, critical path
    21 -> ~13 node-times). Each program is a pure chain, so the h_cache /
    one-hot ancestor machinery VANISHES entirely; cross-path handoff is an
    fp32 HBM export (bit-exact roundtrip). Per-node math = the same
    _gdn_node_step -> values identical; codegen differs (bug-class #10) so
    the in-process selfcheck + capture arm gate it. Supersedes PARENT_GATHER
    and HC_INTERNAL when ON (there is no h_cache to optimize).
    """
    raw = os.environ.get("FR13_SUBTREE_PARALLEL", "")
    if raw.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_subtree_parallel.arm",
            "/tmp/fr13_subtree_parallel.arm",
        )
    )


def subtree_parallel_selfcheck_on() -> bool:
    """In-process byte gate: monolith -> out_ref vs path route -> out."""
    raw = os.environ.get("FR13_SUBTREE_PARALLEL_SELFCHECK", "")
    if raw.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return any(
        os.path.exists(path)
        for path in (
            "/logs/fr13_subtree_parallel_selfcheck.arm",
            "/tmp/fr13_subtree_parallel_selfcheck.arm",
        )
    )


_FR13_FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
_FR13_FIXED32_MODE_SIDECARS = (
    "/logs/fr13_fixed32_mode.flag",
    "/tmp/fr13_fixed32_mode.flag",
)
_FR13_FIXED32_GDN_PATH_BV_SIDECARS = (
    "/logs/fr13_fixed32_gdn_path_bv_candidate.flag",
    "/tmp/fr13_fixed32_gdn_path_bv_candidate.flag",
)
_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_SIDECARS = (
    "/logs/fr13_fixed32_gdn_path_bv_production.flag",
    "/tmp/fr13_fixed32_gdn_path_bv_production.flag",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH_SIDECARS = (
    "/logs/fr13_fixed32_gdn_single_launch_expected_batch.flag",
    "/tmp/fr13_fixed32_gdn_single_launch_expected_batch.flag",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_SIDECARS = (
    "/logs/fr13_fixed32_gdn_single_launch_tree.arm",
    "/tmp/fr13_fixed32_gdn_single_launch_tree.arm",
)
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_SIDECARS = (
    "/logs/fr13_fixed32_gdn_gqa_group3.production.arm",
    "/tmp/fr13_fixed32_gdn_gqa_group3.production.arm",
)
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH_SIDECARS = (
    "/logs/fr13_fixed32_gdn_gqa_group3.production_batch.flag",
    "/tmp/fr13_fixed32_gdn_gqa_group3.production_batch.flag",
)
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_PASS = (
    "/logs/fr13_fixed32_gdn_gqa_group3.production_credential.json"
)
# The folded single-launch arm reaches production through the same door as its
# grouped-GQA sibling above, so it is refused at that door by the same rules and
# therefore carries the same shape of sources. The names differ only in the arm
# they belong to; the /logs entry is the in-container path the launcher binds and
# the /tmp entry is the worker-visible fallback when EngineCore is handed a
# curated environment.
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_SIDECARS = (
    "/logs/fr13_fixed32_gdn_single_launch.production.arm",
    "/tmp/fr13_fixed32_gdn_single_launch.production.arm",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH_SIDECARS = (
    "/logs/fr13_fixed32_gdn_single_launch.production_batch.flag",
    "/tmp/fr13_fixed32_gdn_single_launch.production_batch.flag",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_PASS = (
    "/logs/fr13_fixed32_gdn_single_launch.production_credential.json"
)
_FR13_FIXED32_GDN_PRESCALED_PATH_BASE_SIDECARS = (
    "/logs/fr13_fixed32_gdn_prescaled_path_base.arm",
    "/tmp/fr13_fixed32_gdn_prescaled_path_base.arm",
)
_FR13_FIXED32_GDN_PATH_BV_REAL_EVENT = (
    "/logs/fr13_fixed32_gdn_path_bv.real_event.arm"
)
_FR13_FIXED32_GDN_PATH_BV_LIVE_PASS = (
    "/logs/fr13_fixed32_gdn_path_bv.live_pass.json"
)
_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS = (
    "/logs/fr13_fixed32_gdn_path_bv.production_pass.json"
)
_FR13_FIXED32_GDN_PATH_BV_CANDIDATE_ID = "fixed32_gdn_path_bv_v1"
_FR13_FIXED32_GDN_SINGLE_LAUNCH_GATE_VALUE = "single_launch"
_FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE = "gqa_group3"
_FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE = "gqa_group3_bv16"
_FR13_FIXED32_GDN_BV_SURFACES = (
    "export",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES = (
    "output",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES = (
    "output",
    "export",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
)


def _fr13_resolve_fixed32_mode() -> str | None:
    """Resolve a single fixed32 route from env and worker-visible sidecars.

    The EngineCore worker may receive a curated environment, so the launcher
    can persist the exact mode string in a sidecar. Multiple sources are
    accepted only when they agree. A malformed, unreadable, or conflicting
    source is a configuration error, never permission to use a legacy route.
    """
    sources: list[tuple[str, str]] = []
    raw_env = os.environ.get("FR13_FIXED32_MODE")
    if raw_env is not None and raw_env.strip():
        sources.append(("env:FR13_FIXED32_MODE", raw_env.strip()))
    for path in _FR13_FIXED32_MODE_SIDECARS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                raw_sidecar = handle.read(128)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                f"FR13_FIXED32_MODE: cannot read sidecar {path}: {error}"
            ) from error
        if len(raw_sidecar) >= 128:
            raise RuntimeError(
                f"FR13_FIXED32_MODE: sidecar {path} exceeds 127 bytes"
            )
        sources.append((f"sidecar:{path}", raw_sidecar.strip()))
    if not sources:
        armed_features = [
            feature
            for feature in (
                "FR13_FIXED32_KV_REMAP16",
                "FR13_FIXED32_COMMIT_DEVICE_FILL",
            )
            if os.environ.get(feature, "").strip() == "1"
        ]
        if armed_features:
            raise RuntimeError(
                "FR13_FIXED32_MODE is missing while fixed32 feature flags "
                f"are armed: {armed_features}"
            )
        return None
    invalid = [(source, value) for source, value in sources
               if value not in _FR13_FIXED32_MODES]
    if invalid:
        raise RuntimeError(
            "FR13_FIXED32_MODE: invalid fixed32 route source(s): "
            + ", ".join(f"{source}={value!r}" for source, value in invalid)
        )
    modes = {value for _source, value in sources}
    if len(modes) != 1:
        raise RuntimeError(
            "FR13_FIXED32_MODE: conflicting route sources: "
            + ", ".join(f"{source}={value!r}" for source, value in sources)
        )
    mode = modes.pop()
    for feature in (
        "FR13_FIXED32_KV_REMAP16",
        "FR13_FIXED32_COMMIT_DEVICE_FILL",
    ):
        raw_feature = os.environ.get(feature)
        if raw_feature is not None and raw_feature.strip() not in ("", "1"):
            raise RuntimeError(
                f"{feature}={raw_feature!r} conflicts with armed fixed32 mode"
            )
    return mode


_FR13_FIXED32_MODE = _fr13_resolve_fixed32_mode()


# ---------------------------------------------------------------------------
# FR14 DRAFT-VOCABULARY QUALIFICATION PROFILES (2026-08-17)
#
# Every ordered-GDN predicate in this module used to hard-code the K64/root1
# draft-vocabulary identity into its own legality check. Mark's K0 ruling made
# full-vocab drafting the production config, which parked the ARMING PATH for
# the whole ordered-GDN lever family at once: a credential could only be earned
# in a shape production never serves.
#
# The fix is the shape the launcher, the shared gate runner, the credential
# validator and the in-container patcher already carry
# (scripts/fr13_launch_forked_fa2_tree_server.sh:547 _fr13_assert_draft_vocab_profile):
# a lever DECLARES which draft-vocabulary shape it is earned in, and every
# independent layer checks the served shape agrees. The pinning is KEPT, not
# removed -- a k64_root credential still cannot authorize a full_vocab serve and
# vice versa; the identity is simply no longer welded to one of the two shapes.
#
# Three variables, one per lever, exactly the names the launcher validates and
# forwards, so a credential is earned and spent under ONE declared identity:
#   FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE   -- the ordered live gate
#   FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE -- single_launch prod
#   FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE  -- GQA-group3 production
# All default to k64_root, so every banked arm keeps its exact previous meaning
# and an unknown profile is refused BY NAME rather than silently defaulted.
_FR13_DRAFT_VOCAB_PROFILES = {
    "k64_root": {"FR13_DRAFT_VOCAB_ROOT": "1", "FR13_DRAFT_VOCAB_K": "65536"},
    "full_vocab": {"FR13_DRAFT_VOCAB_ROOT": "0", "FR13_DRAFT_VOCAB_K": "0"},
}
# The same two shapes as they appear INSIDE a credential/live-PASS artifact.
# Kept as a separate table on purpose: the env pair is strings and the
# credential pair is ints, and a JSON "65536" must never satisfy an int check.
_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS = {
    "k64_root": {"draft_vocab_k": 65536, "draft_vocab_root": 1},
    "full_vocab": {"draft_vocab_k": 0, "draft_vocab_root": 0},
}
_FR13_GDN_ORDERED_CANDIDATES = (
    "single_launch",
    "gqa_group3",
    "gqa_group3_bv16",
)


def _fr13_draft_vocab_profile(variable: str, *, environ=None) -> str:
    """Return the DECLARED draft-vocabulary profile for one lever.

    Empty is treated as unset -> k64_root, matching the launcher's
    ``${VAR:-k64_root}`` semantics exactly, so an env sweeper that forwards the
    name with an empty value cannot turn a banked arm into an import-time raise.
    A non-empty value that is not a known profile is ALWAYS refused: silently
    defaulting an unrecognised declaration is how a lever ends up serving one
    identity while claiming another.
    """
    env = os.environ if environ is None else environ
    raw = env.get(variable)
    profile = "k64_root" if raw is None or not str(raw).strip() else str(raw).strip()
    if profile not in _FR13_DRAFT_VOCAB_PROFILES:
        raise RuntimeError(
            f"{variable} must be exactly k64_root or full_vocab; "
            f"got {profile!r}"
        )
    return profile


def _fr13_draft_vocab_env_matches(profile: str, *, environ=None) -> bool:
    """Does the SERVED drafter env match the declared profile exactly?

    Same strictness as the literals this replaced: the value must be present and
    exact, an absent variable is not a match in either profile.
    """
    env = os.environ if environ is None else environ
    return all(
        str(env.get(name, "")).strip() == value
        for name, value in _FR13_DRAFT_VOCAB_PROFILES[profile].items()
    )


def _fr13_draft_vocab_credential_matches(
    credential: dict[str, object], profile: str
) -> bool:
    """Does a credential's SELF-DESCRIBED identity match the declared profile?

    Three fields, all required to agree: the profile the credential claims it
    was earned under, plus the K/root pair that claim implies. Requiring the
    pair as well as the name is what stops either shape from masquerading as the
    other -- a credential cannot say ``full_vocab`` while carrying K64 values,
    and a pre-FR14 credential that carries the pair but names no profile is
    refused rather than assumed to be k64_root.
    """
    if credential.get("qualification_profile") != profile:
        return False
    for key, value in _FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS[profile].items():
        actual = credential.get(key)
        if type(actual) is not int or actual != value:
            return False
    return True


def _fr13_resolve_fixed32_gdn_single_launch(
    fixed32_mode: str | None,
    *,
    environ=None,
    sidecars=None,
    geom_override=None,
) -> bool:
    """Resolve the source-only K64/root1 fixed32 single-launch candidate."""
    env = os.environ if environ is None else environ
    paths = (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_SIDECARS
        if sidecars is None
        else tuple(sidecars)
    )
    sources: list[tuple[str, str]] = []
    raw_env = env.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE")
    if raw_env is not None and str(raw_env).strip():
        sources.append(
            (
                "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE",
                str(raw_env).strip(),
            )
        )
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(16)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE cannot read sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 16:
            raise RuntimeError(
                "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE sidecar exceeds "
                f"15 bytes: {path}"
            )
        sources.append((f"sidecar:{path}", value.strip()))
    if not sources:
        return False
    source_values = {value for _source, value in sources}
    if source_values == {"0"} and all(
        source == "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE"
        for source, _value in sources
    ):
        return False
    invalid = [
        (source, value)
        for source, value in sources
        if value != "1"
    ]
    if invalid or len(source_values) != 1:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE must be exactly 1 from "
            f"agreeing sources: {sources!r}"
        )
    if fixed32_mode not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE requires an exact fixed32 "
            "mode"
        )
    if geom_override != {"BV": 8}:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE requires geometry pinned "
            "exactly to FR13_TREE_GDN_GEOM_OVERRIDE=BV=8"
        )
    # FR14. This bool is the CAMPAIGN instrument the ordered live gate observes,
    # so it keys on the gate's variable -- the same one the launcher clause and
    # the in-container patcher's exact_single_launch contract read. Declared,
    # never assumed; default k64_root leaves every banked arming unchanged.
    profile = _fr13_draft_vocab_profile(
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE", environ=env
    )
    if not _fr13_draft_vocab_env_matches(profile, environ=env):
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE requires the exact "
            f"{profile} drafter contract"
        )
    return True


def _fr13_resolve_fixed32_gdn_prescaled_path_base(
    single_launch_available: bool,
    *,
    environ=None,
    sidecars=None,
) -> bool:
    """Resolve the default-off pre-scaled descriptor specialization."""
    env = os.environ if environ is None else environ
    paths = (
        _FR13_FIXED32_GDN_PRESCALED_PATH_BASE_SIDECARS
        if sidecars is None
        else tuple(sidecars)
    )
    sources: list[tuple[str, str]] = []
    raw_env = env.get("FR13_FIXED32_GDN_PRESCALED_PATH_BASE")
    if raw_env is not None and str(raw_env).strip():
        sources.append(
            (
                "env:FR13_FIXED32_GDN_PRESCALED_PATH_BASE",
                str(raw_env).strip(),
            )
        )
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(16)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PRESCALED_PATH_BASE cannot read sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 16:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PRESCALED_PATH_BASE sidecar exceeds "
                f"15 bytes: {path}"
            )
        sources.append((f"sidecar:{path}", value.strip()))
    if not sources:
        return False
    invalid = [
        (source, value)
        for source, value in sources
        if value not in ("0", "1")
    ]
    if invalid or len({value for _source, value in sources}) != 1:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PRESCALED_PATH_BASE must be exactly 0 or 1 "
            f"from agreeing sources: {sources!r}"
        )
    enabled = sources[0][1] == "1"
    if enabled and not single_launch_available:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PRESCALED_PATH_BASE requires the exact "
            "K64/root1 ordered single-launch route"
        )
    return enabled


def _fr13_resolve_fixed32_gdn_path_bv_candidate(
    fixed32_mode: str | None,
    *,
    environ=None,
    sidecars=None,
    geom_override=None,
) -> int | str | None:
    """Resolve a non-serving fixed32 GDN live-gate candidate."""
    env = os.environ if environ is None else environ
    paths = (
        _FR13_FIXED32_GDN_PATH_BV_SIDECARS
        if sidecars is None
        else tuple(sidecars)
    )
    sources: list[tuple[str, str]] = []
    raw_env = env.get("FR13_FIXED32_GDN_PATH_BV_CANDIDATE")
    if raw_env is not None and str(raw_env).strip():
        sources.append(
            (
                "env:FR13_FIXED32_GDN_PATH_BV_CANDIDATE",
                str(raw_env).strip(),
            )
        )
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                raw_sidecar = handle.read(32)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PATH_BV_CANDIDATE: cannot read sidecar "
                f"{path}: {error}"
            ) from error
        if len(raw_sidecar) >= 32:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PATH_BV_CANDIDATE: sidecar exceeds "
                f"31 bytes: {path}"
            )
        sources.append((f"sidecar:{path}", raw_sidecar.strip()))
    if not sources:
        return None
    invalid = [
        (source, value)
        for source, value in sources
        if value not in (
            "16",
            "32",
            "64",
            "128",
            "single_launch",
            "gqa_group3",
            "gqa_group3_bv16",
        )
    ]
    if invalid:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE: expected one of "
            "16, 32, 64, 128, single_launch, gqa_group3, or "
            "gqa_group3_bv16, got "
            + ", ".join(
                f"{source}={value!r}" for source, value in invalid
            )
        )
    values = {value for _source, value in sources}
    if len(values) != 1:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE: conflicting sources: "
            + ", ".join(
                f"{source}={value!r}" for source, value in sources
            )
        )
    if fixed32_mode not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE requires an exact "
            f"fixed32 mode, got {fixed32_mode!r}"
        )
    if geom_override != {"BV": 8}:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE requires the served graph "
            "to be pinned exactly to FR13_TREE_GDN_GEOM_OVERRIDE=BV=8"
        )
    value = values.pop()
    if value in _FR13_GDN_ORDERED_CANDIDATES:
        # FR14. The exact mirror of the launcher's ordered-GDN live-gate clause
        # (fr13_launch_forked_fa2_tree_server.sh:3295): ALL three ordered
        # candidates qualify under the one declared gate identity, because the
        # gate they feed is one gate. Default k64_root.
        profile = _fr13_draft_vocab_profile(
            "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE", environ=env
        )
        if not _fr13_draft_vocab_env_matches(profile, environ=env):
            raise RuntimeError(
                "FR13 fixed32 GDN ordered live gate requires the exact "
                f"{profile} drafter contract"
            )
        return value
    return int(value)


def _fr13_fixed32_gdn_gqa_group3_source_sha256() -> str:
    try:
        payload = Path(__file__).resolve().read_bytes()
        candidate_path = Path(__file__).with_name("fr13_gdn_gqa_group3.py")
        candidate_payload = candidate_path.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"FR13 fixed32 GDN GQA-group3 cannot hash its source: {error}"
        ) from error
    payload = (
        b"fr10_gdn_tree_kernel.py\0"
        + payload
        + b"\0fr13_gdn_gqa_group3.py\0"
        + candidate_payload
    )
    return hashlib.sha256(payload).hexdigest()


def _fr13_fixed32_gdn_path_bv_source_sha256() -> str:
    if globals().get("_FR13_FIXED32_GDN_PATH_BV_CANDIDATE") in (
        "gqa_group3",
        "gqa_group3_bv16",
    ):
        return _fr13_fixed32_gdn_gqa_group3_source_sha256()
    try:
        payload = Path(__file__).resolve().read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"FR13 fixed32 GDN BV cannot hash its source: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _fr13_resolve_fixed32_gdn_single_launch_expected_batch(
    candidate: int | str | None,
    *,
    environ=None,
    sidecars=None,
) -> int | None:
    """Resolve the one FULL-graph batch this worker is allowed to qualify."""
    env = os.environ if environ is None else environ
    paths = (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH_SIDECARS
        if sidecars is None
        else tuple(sidecars)
    )
    sources: list[tuple[str, str]] = []
    raw_env = env.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH")
    if raw_env is not None and str(raw_env).strip():
        sources.append(
            (
                "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH",
                str(raw_env).strip(),
            )
        )
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(4)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13 GDN single-launch expected batch cannot read sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 4:
            raise RuntimeError(
                "FR13 GDN single-launch expected-batch sidecar exceeds "
                f"3 bytes: {path}"
            )
        sources.append((f"sidecar:{path}", value.strip()))
    if candidate not in (
        "single_launch",
        "gqa_group3",
        "gqa_group3_bv16",
    ):
        if sources:
            raise RuntimeError(
                "FR13 GDN ordered expected batch is set without the "
                "single-launch candidate or GQA-group3 candidate"
            )
        return None
    values = {value for _source, value in sources}
    if not sources or len(values) != 1 or values.difference({"1", "4"}):
        raise RuntimeError(
            "FR13 GDN ordered candidate requires exactly one expected batch, 1 or "
            "4, from agreeing sources: " + repr(sources)
        )
    return int(values.pop())


def _fr13_fixed32_gdn_single_launch_diagnostic_identity(
    mode: str | None,
    batch_size: int,
    candidate: str = _FR13_FIXED32_GDN_SINGLE_LAUNCH_GATE_VALUE,
) -> str:
    topology = {
        "tail6_fixed32": "tail23",
        "hydra27_fixed32": "hydra27",
    }.get(mode)
    batch = int(batch_size)
    identity = {
        "single_launch": (
            "fixed32_gdn_single_launch_tree_v2"
        ),
        "gqa_group3": (
            "fixed32_gdn_single_launch_gqa_group3_v1"
        ),
        "gqa_group3_bv16": (
            "fixed32_gdn_single_launch_gqa_group3_bv16_v1"
        ),
    }.get(candidate)
    if topology is None or batch not in (1, 4) or identity is None:
        raise RuntimeError(
            "FR13 GDN ordered diagnostic identity is not candidate/topology/batch bound"
        )
    return f"{identity}:{topology}:b{batch}"


def _fr13_resolve_fixed32_gdn_gqa_group3_production(
    fixed32_mode: str | None,
    *,
    environ=None,
    arm_sidecars=None,
    batch_sidecars=None,
    geom_override=None,
    pass_path: str | None = None,
) -> dict[str, object] | None:
    """Resolve a source-bound grouped-GQA arm after its real-task byte gate."""
    env = os.environ if environ is None else environ
    arm_paths = (
        _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_SIDECARS
        if arm_sidecars is None
        else tuple(arm_sidecars)
    )
    batch_paths = (
        _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH_SIDECARS
        if batch_sidecars is None
        else tuple(batch_sidecars)
    )
    arm_sources: list[tuple[str, str]] = []
    raw_arm = env.get("FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION")
    if raw_arm is not None and str(raw_arm).strip():
        arm_sources.append(
            (
                "env:FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION",
                str(raw_arm).strip(),
            )
        )
    for path in arm_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(4)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13 GDN GQA-group3 production cannot read arm sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 4:
            raise RuntimeError(
                "FR13 GDN GQA-group3 production arm sidecar exceeds 3 bytes"
            )
        arm_sources.append((f"sidecar:{path}", value.strip()))
    if not arm_sources:
        return None
    arm_values = {value for _source, value in arm_sources}
    if arm_values == {"0"} and all(
        source == "env:FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION"
        for source, _value in arm_sources
    ):
        return None
    if arm_values != {"1"}:
        raise RuntimeError(
            "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION must be exactly 1 from "
            "agreeing sources"
        )

    batch_sources: list[tuple[str, str]] = []
    raw_batch = env.get("FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH")
    if raw_batch is not None and str(raw_batch).strip():
        batch_sources.append(
            (
                "env:FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH",
                str(raw_batch).strip(),
            )
        )
    for path in batch_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(4)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13 GDN GQA-group3 production cannot read batch sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 4:
            raise RuntimeError(
                "FR13 GDN GQA-group3 production batch sidecar exceeds 3 bytes"
            )
        batch_sources.append((f"sidecar:{path}", value.strip()))
    batch_values = {value for _source, value in batch_sources}
    if not batch_sources or len(batch_values) != 1 or batch_values.difference(
        {"1", "4"}
    ):
        raise RuntimeError(
            "FR13 GDN GQA-group3 production requires one exact batch, 1 or 4"
        )
    batch = int(batch_values.pop())
    # FR14. The GQA-group3 PRODUCTION arm's own declared identity -- a separate
    # pin set from the gate's, and separately declared, so a credential earned
    # under one shape can never be spent under another. Mirrors the launcher's
    # clause at fr13_launch_forked_fa2_tree_server.sh:3355. Default k64_root.
    draft_vocab_profile = _fr13_draft_vocab_profile(
        "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE", environ=env
    )
    if (
        fixed32_mode not in _FR13_FIXED32_MODES
        or geom_override != {"BV": 8}
        or not _fr13_draft_vocab_env_matches(draft_vocab_profile, environ=env)
    ):
        raise RuntimeError(
            "FR13 GDN GQA-group3 production requires exact fixed32 physical32 "
            f"BV8 {draft_vocab_profile}"
        )
    resolved_pass = pass_path or env.get(
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_PASS_PATH",
        _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_PASS,
    )
    if os.path.islink(resolved_pass) or not os.path.isfile(resolved_pass):
        raise RuntimeError(
            "FR13 GDN GQA-group3 production requires a regular live-gate "
            f"credential: {resolved_pass}"
        )
    try:
        with open(resolved_pass, encoding="ascii") as handle:
            credential = json.load(handle)
        kernel_sha256 = hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest()
        candidate_sha256 = hashlib.sha256(
            Path(__file__).with_name("fr13_gdn_gqa_group3.py").read_bytes()
        ).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"FR13 GDN GQA-group3 production credential is unreadable: {error}"
        ) from error
    source_commit = credential.get("source_commit") if isinstance(
        credential, dict
    ) else None
    if (
        not isinstance(credential, dict)
        or credential.get("schema")
        != "fr13.fixed32.gdn_single_launch.real_task_credential.v4"
        or credential.get("status") != "PASS"
        or credential.get("candidate")
        != "fixed32_gdn_single_launch_gqa_group3_v1"
        or credential.get("reference")
        != "fixed32_gdn_two_launch_reference_v1"
        or credential.get("mode") != fixed32_mode
        or credential.get("batch_size") != batch
        or credential.get("expected_batch") != batch
        or credential.get("physical_rows") != 32
        # FR14. The credential must CARRY the draft-vocabulary shape it was
        # earned in, and it must be the shape this process declared. Name and
        # K/root pair are checked together, so neither shape can masquerade as
        # the other and a pre-FR14 credential (pair present, profile absent) is
        # refused rather than assumed.
        or not _fr13_draft_vocab_credential_matches(
            credential, draft_vocab_profile
        )
        or credential.get("raw_byte_equal") is not True
        or credential.get("reference_served") is not True
        or credential.get("state_restored") is not True
        or credential.get("production_enabled") is not False
        or credential.get("kernel_source_sha256") != kernel_sha256
        or credential.get("gqa_group3_source_sha256") != candidate_sha256
        or credential.get("candidate_source_sha256")
        != _fr13_fixed32_gdn_gqa_group3_source_sha256()
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError(
            "FR13 GDN GQA-group3 production credential is invalid, stale, or "
            "belongs to another source/mode/batch"
        )
    return credential


def _fr13_resolve_fixed32_gdn_single_launch_production(
    fixed32_mode: str | None,
    *,
    environ=None,
    arm_sidecars=None,
    batch_sidecars=None,
    geom_override=None,
    pass_path: str | None = None,
) -> dict[str, object] | None:
    """Resolve the source-bound folded single-launch arm after its byte gate.

    WHY THIS EXISTS ALONGSIDE ``_FR13_FIXED32_GDN_SINGLE_LAUNCH``. That bool is a
    CAMPAIGN instrument: it puts the folded kernel on the decode path from source
    and env alone so a live gate can observe it, and it is deliberately
    credential-free because the credential is the thing the gate is producing. A
    PRODUCTION serve is the opposite situation -- the gate has already run, so
    the arm must be refused unless the exact PASS artifact that gate issued is
    present, matches this source, this mode and this batch, and was issued
    without production already enabled. Both arms end at the identical folded
    kernel and identical telemetry; only the authority to reach it differs.

    Fail-closed at every step. An absent arm is not an error -- the registry
    default ships OFF and every fixed32 campaign process in this tree would trip
    an import-time raise otherwise -- but an arm that is present and not exactly
    satisfied is. Structurally this is the twin of
    ``_fr13_resolve_fixed32_gdn_gqa_group3_production``; the one real divergence
    is the source pin, commented at its clause below.
    """
    env = os.environ if environ is None else environ
    arm_paths = (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_SIDECARS
        if arm_sidecars is None
        else tuple(arm_sidecars)
    )
    batch_paths = (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH_SIDECARS
        if batch_sidecars is None
        else tuple(batch_sidecars)
    )
    arm_sources: list[tuple[str, str]] = []
    raw_arm = env.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION")
    if raw_arm is not None and str(raw_arm).strip():
        arm_sources.append(
            (
                "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION",
                str(raw_arm).strip(),
            )
        )
    for path in arm_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(4)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13 GDN single-launch production cannot read arm sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 4:
            raise RuntimeError(
                "FR13 GDN single-launch production arm sidecar exceeds 3 bytes"
            )
        arm_sources.append((f"sidecar:{path}", value.strip()))
    if not arm_sources:
        return None
    arm_values = {value for _source, value in arm_sources}
    # A caller that NAMES the arm 0 and writes no sidecar is declining it, which
    # is the default-OFF path and must never raise. Anything else disagreeing is
    # a configuration error, never permission to serve.
    if arm_values == {"0"} and all(
        source == "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION"
        for source, _value in arm_sources
    ):
        return None
    if arm_values != {"1"}:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION must be exactly 1 from "
            "agreeing sources"
        )

    batch_sources: list[tuple[str, str]] = []
    raw_batch = env.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH")
    if raw_batch is not None and str(raw_batch).strip():
        batch_sources.append(
            (
                "env:FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH",
                str(raw_batch).strip(),
            )
        )
    for path in batch_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(4)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13 GDN single-launch production cannot read batch sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 4:
            raise RuntimeError(
                "FR13 GDN single-launch production batch sidecar exceeds "
                "3 bytes"
            )
        batch_sources.append((f"sidecar:{path}", value.strip()))
    batch_values = {value for _source, value in batch_sources}
    if not batch_sources or len(batch_values) != 1 or batch_values.difference(
        {"1", "4"}
    ):
        raise RuntimeError(
            "FR13 GDN single-launch production requires one exact batch, 1 or 4"
        )
    batch = int(batch_values.pop())
    # FR14. The single_launch PRODUCTION arm's declared identity. This is a
    # SEPARATE site from the gate's on purpose (the same split the in-container
    # patcher carries): re-pointing only the gate would earn a credential the
    # production path then refuses, post-gate. Default k64_root.
    draft_vocab_profile = _fr13_draft_vocab_profile(
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE", environ=env
    )
    if (
        fixed32_mode not in _FR13_FIXED32_MODES
        or geom_override != {"BV": 8}
        or not _fr13_draft_vocab_env_matches(draft_vocab_profile, environ=env)
    ):
        raise RuntimeError(
            "FR13 GDN single-launch production requires exact fixed32 "
            f"physical32 BV8 {draft_vocab_profile}"
        )
    resolved_pass = pass_path or env.get(
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_PASS_PATH",
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_PASS,
    )
    if os.path.islink(resolved_pass) or not os.path.isfile(resolved_pass):
        raise RuntimeError(
            "FR13 GDN single-launch production requires a regular live-gate "
            f"credential: {resolved_pass}"
        )
    try:
        with open(resolved_pass, encoding="ascii") as handle:
            credential = json.load(handle)
        # The folded candidate has no source unit of its own: it is compiled
        # from this closure, so the candidate digest and the kernel digest are
        # re-derived from the SAME bytes and must both match. Any edit to this
        # file -- including one that cannot touch the kernel -- invalidates the
        # credential and forces a re-gate. That is the intended cost.
        kernel_sha256 = hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "FR13 GDN single-launch production credential is unreadable: "
            f"{error}"
        ) from error
    source_commit = credential.get("source_commit") if isinstance(
        credential, dict
    ) else None
    if (
        not isinstance(credential, dict)
        or credential.get("schema")
        != "fr13.fixed32.gdn_single_launch.real_task_credential.v4"
        or credential.get("status") != "PASS"
        # The literal candidate id, not the module constant: this resolver runs
        # while the module body is still executing, well before
        # _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID is bound. The sibling
        # spells its candidate out for exactly the same reason.
        or credential.get("candidate") != "fixed32_gdn_single_launch_tree_v2"
        or credential.get("reference")
        != "fixed32_gdn_two_launch_reference_v1"
        or credential.get("mode") != fixed32_mode
        or credential.get("batch_size") != batch
        or credential.get("expected_batch") != batch
        or credential.get("physical_rows") != 32
        # FR14. Same clause as the GQA-group3 twin, and for the same reason: the
        # credential names the draft-vocabulary shape it was earned in and
        # carries the K/root pair that name implies, and BOTH must equal what
        # this production process declared.
        or not _fr13_draft_vocab_credential_matches(
            credential, draft_vocab_profile
        )
        or credential.get("raw_byte_equal") is not True
        or credential.get("reference_served") is not True
        or credential.get("state_restored") is not True
        # The gate is a LEGALITY instrument: it shadows the candidate while the
        # reference is served, so a credential claiming production was already
        # enabled did not come from the gate this promotion rests on.
        or credential.get("production_enabled") is not False
        or credential.get("kernel_source_sha256") != kernel_sha256
        # DIVERGENCE FROM THE GQA-GROUP3 TWIN. That arm pins a second source
        # unit (fr13_gdn_gqa_group3.py) and requires its digest; this arm has no
        # such unit and the gate emits the key as JSON null. Requiring null is
        # the mirror image of the twin's requirement, and together the two
        # clauses make the arms mutually exclusive by construction -- neither
        # credential can satisfy the other's source pin. The default sentinel
        # also refuses a credential that simply omits the key.
        or credential.get("gqa_group3_source_sha256", "absent") is not None
        or credential.get("candidate_source_sha256") != kernel_sha256
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError(
            "FR13 GDN single-launch production credential is invalid, stale, "
            "or belongs to another source/mode/batch"
        )
    return credential


def _fr13_resolve_fixed32_gdn_path_bv_production(
    fixed32_mode: str | None,
    *,
    environ=None,
    sidecars=None,
    geom_override=None,
    pass_path: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, object] | None:
    """Resolve a source-bound, prior-live-gated BV production route."""
    env = os.environ if environ is None else environ
    paths = (
        _FR13_FIXED32_GDN_PATH_BV_PRODUCTION_SIDECARS
        if sidecars is None
        else tuple(sidecars)
    )
    sources: list[tuple[str, str]] = []
    raw_env = env.get("FR13_FIXED32_GDN_PATH_BV_PRODUCTION")
    if raw_env is not None and str(raw_env).strip():
        sources.append(
            (
                "env:FR13_FIXED32_GDN_PATH_BV_PRODUCTION",
                str(raw_env).strip(),
            )
        )
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(16)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PATH_BV_PRODUCTION cannot read sidecar "
                f"{path}: {error}"
            ) from error
        if len(value) >= 16:
            raise RuntimeError(
                "FR13_FIXED32_GDN_PATH_BV_PRODUCTION sidecar exceeds "
                f"15 bytes: {path}"
            )
        sources.append((f"sidecar:{path}", value.strip()))
    if not sources:
        return None
    invalid = [
        (source, value)
        for source, value in sources
        if value not in ("16", "32", "64", "128")
    ]
    values = {value for _source, value in sources}
    if invalid or len(values) != 1:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_PRODUCTION requires one of 16, 32, "
            "64, or 128 from agreeing sources: "
            + repr(sources)
        )
    candidate = int(values.pop())
    if fixed32_mode not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_PRODUCTION requires an exact fixed32 mode"
        )
    if geom_override != {"BV": candidate}:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_PRODUCTION requires geometry pinned "
            f"exactly to BV={candidate}"
        )
    resolved_pass_path = pass_path or env.get(
        "FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS_PATH",
        _FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS,
    )
    if os.path.islink(resolved_pass_path) or not os.path.isfile(
        resolved_pass_path
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_PATH_BV_PRODUCTION requires a regular live "
            f"PASS JSON: {resolved_pass_path}"
        )
    try:
        with open(resolved_pass_path, encoding="ascii") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"FR13 fixed32 GDN BV production PASS JSON is unreadable: {error}"
        ) from error
    current_sha = source_sha256 or _fr13_fixed32_gdn_path_bv_source_sha256()
    task_marker = payload.get("task_marker") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "fr13.fixed32.gdn_path_bv.live_pass.v1"
        or payload.get("status") != "pass"
        or payload.get("candidate") != _FR13_FIXED32_GDN_PATH_BV_CANDIDATE_ID
        or payload.get("source_sha256") != current_sha
        or payload.get("mode") != fixed32_mode
        or payload.get("reference_bv") != 8
        or payload.get("candidate_bv") != candidate
        or payload.get("raw_byte_equal") is not True
        or payload.get("reference_served") is not True
        or payload.get("state_restored") is not True
        or type(payload.get("batch_size")) is not int
        or payload.get("batch_size") not in (1, 2, 3, 4)
        or payload.get("covered_batches")
        != list(range(1, payload.get("batch_size", 0) + 1))
        or payload.get("records") != 48 * payload.get("batch_size", 0)
        or payload.get("physical_rows") != 32
        or not isinstance(task_marker, str)
        or not task_marker.startswith("swe_verified:")
        or len(task_marker) == len("swe_verified:")
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV production PASS JSON is invalid or belongs "
            "to a different candidate/source"
        )
    return payload


def _fr13_fixed32_gdn_bv_real_event_marker() -> str:
    default_path = (
        "/logs/fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
        if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        else _FR13_FIXED32_GDN_PATH_BV_REAL_EVENT
    )
    path = os.environ.get(
        "FR13_FIXED32_GDN_PATH_BV_REAL_EVENT_PATH",
        default_path,
    )
    if not os.path.isfile(path):
        raise RuntimeError(
            "FR13 fixed32 GDN BV live gate requires a real SWE-Verified event arm"
        )
    try:
        with open(path, encoding="ascii") as handle:
            marker = handle.read(257)
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            f"FR13 fixed32 GDN BV cannot read real-event marker: {error}"
        ) from error
    marker = marker.strip()
    prefix = "swe_verified:"
    task_id = marker[len(prefix):] if marker.startswith(prefix) else ""
    if (
        len(marker) > 256
        or not task_id
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
            for character in task_id
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV real-event marker must be "
            "swe_verified:<task_id>"
        )
    return marker


def _fr13_fixed32_gdn_bv_live_pass_emit(
    *,
    task_marker: str,
    batch_size: int,
    graph_signature: str,
    result: dict[str, object],
) -> None:
    path = os.environ.get(
        "FR13_FIXED32_GDN_PATH_BV_LIVE_JSON",
        _FR13_FIXED32_GDN_PATH_BV_LIVE_PASS,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ordered_launch = (
        result.get("candidate")
        in (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID,
            "fixed32_gdn_single_launch_gqa_group3_v1",
            "fixed32_gdn_single_launch_gqa_group3_bv16_v1",
        )
    )
    topology = {
        "tail6_fixed32": ("Tail23", 23, 0x7A9CE7FF),
        "hydra27_fixed32": ("Hydra27", 27, 0x7ABDFFFF),
    }.get(_FR13_FIXED32_MODE)
    if ordered_launch and topology is None:
        raise RuntimeError(
            "FR13 fixed32 GDN single-launch PASS has no bound topology"
        )
    if ordered_launch and batch_size != _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH:
        raise RuntimeError(
            "FR13 fixed32 GDN single-launch PASS batch differs from the baked "
            "diagnostic identity"
        )
    draft_vocab_profile = _FR13_FIXED32_GDN_ORDERED_QUALIFICATION_PROFILE
    if ordered_launch and (
        draft_vocab_profile is None
        or not _fr13_draft_vocab_env_matches(draft_vocab_profile)
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN single-launch PASS has no served draft-vocabulary "
            "identity to declare"
        )
    payload = {
        "schema": (
            "fr13.fixed32.gdn_single_launch.live_pass.v2"
            if ordered_launch
            else "fr13.fixed32.gdn_path_bv.live_pass.v1"
        ),
        "status": "pass",
        "candidate": (
            result["candidate"]
            if ordered_launch
            else _FR13_FIXED32_GDN_PATH_BV_CANDIDATE_ID
        ),
        "source_sha256": _fr13_fixed32_gdn_path_bv_source_sha256(),
        "task_marker": task_marker,
        "mode": _FR13_FIXED32_MODE,
        "graph_signature": graph_signature,
        "batch_size": int(batch_size),
        "covered_batches": (
            [int(batch_size)]
            if ordered_launch
            else list(range(1, int(batch_size) + 1))
        ),
        "records": int(result["records"]),
        "physical_rows": 32,
        "reference_bv": int(result["reference_bv"]),
        "candidate_bv": int(result["candidate_bv"]),
        "reference_physical_launches_per_request_layer": int(
            result.get("reference_physical_launches", 2)
        ),
        "candidate_physical_launches_per_request_layer": int(
            result.get("candidate_physical_launches", 2)
        ),
        "compared_byte_surfaces": list(
            _FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES
            if ordered_launch
            else _FR13_FIXED32_GDN_BV_SURFACES
        ),
        "raw_byte_equal": True,
        "reference_served": True,
        "state_restored": True,
        "real_task_authenticated": True,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
    }
    if ordered_launch:
        assert topology is not None
        assert draft_vocab_profile is not None
        payload.update(
            logical_topology=topology[0],
            logical_drafts=topology[1],
            valid_mask=topology[2],
            qualification_profile=draft_vocab_profile,
            **_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS[draft_vocab_profile],
            gate_mode="post_first_measured_full_graph_replay",
            expected_batch=int(batch_size),
            diagnostic_identity=(
                _fr13_fixed32_gdn_single_launch_diagnostic_identity(
                    _FR13_FIXED32_MODE,
                    batch_size,
                    {
                        "fixed32_gdn_single_launch_gqa_group3_v1": (
                            "gqa_group3"
                        ),
                        "fixed32_gdn_single_launch_gqa_group3_bv16_v1": (
                            "gqa_group3_bv16"
                        ),
                    }.get(result["candidate"], "single_launch"),
                )
            ),
        )
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _fr13_fixed32_gdn_single_launch_observation_emit(
    *,
    batch_size: int,
    comparator_events: list[dict[str, object]],
) -> None:
    """Publish the exact ordered comparator events awaiting ledger join."""
    path = os.environ.get(
        "FR13_FIXED32_GDN_PATH_BV_LIVE_JSON",
        _FR13_FIXED32_GDN_PATH_BV_LIVE_PASS,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    topology = {
        "tail6_fixed32": ("Tail23", 23, 0x7A9CE7FF),
        "hydra27_fixed32": ("Hydra27", 27, 0x7ABDFFFF),
    }.get(_FR13_FIXED32_MODE)
    batch = int(batch_size)
    # FR14: the observation declares the draft-vocabulary shape it was OBSERVED
    # in. Resolved beside the candidate at import and re-checked against the
    # served env here, so the record cannot describe an identity this process
    # did not run.
    draft_vocab_profile = _FR13_FIXED32_GDN_ORDERED_QUALIFICATION_PROFILE
    if (
        topology is None
        or batch != _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
        or not comparator_events
        or any(not isinstance(event, dict) for event in comparator_events)
        or draft_vocab_profile is None
        or not _fr13_draft_vocab_env_matches(draft_vocab_profile)
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN single-launch observation scope drift"
        )
    canonical_events = json.dumps(
        comparator_events,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    payload = {
        "schema": "fr13.fixed32.gdn_single_launch.live_observation.v3",
        "status": "observed_pending_authenticated_coverage_join",
        "candidate": comparator_events[-1]["candidate"],
        "source_sha256": _fr13_fixed32_gdn_path_bv_source_sha256(),
        "mode": _FR13_FIXED32_MODE,
        "batch_size": batch,
        "expected_batch": batch,
        "covered_batches": [batch],
        "records_per_comparator_event": 48,
        "comparator_event_count": len(comparator_events),
        "comparator_events_sha256": hashlib.sha256(
            canonical_events
        ).hexdigest(),
        "comparator_events": comparator_events,
        "physical_rows": 32,
        "reference_bv": 8,
        "candidate_bv": (
            16
            if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE
            else 8
        ),
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "compared_byte_surfaces": list(
            _FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES
        ),
        "reference_served": True,
        "candidate_served": False,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "logical_topology": topology[0],
        "logical_drafts": topology[1],
        "valid_mask": topology[2],
        "qualification_profile": draft_vocab_profile,
        **_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS[draft_vocab_profile],
        "gate_mode": "post_measured_replay_distinct_request_tuple",
        "coverage_authority": "authenticated_proxy_engine_request_join",
        "diagnostic_identity": (
            _fr13_fixed32_gdn_single_launch_diagnostic_identity(
                _FR13_FIXED32_MODE,
                batch,
                _FR13_FIXED32_GDN_PATH_BV_CANDIDATE,
            )
        ),
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


_FR13_FIXED32_GDN_PATH_BV_CANDIDATE = (
    _fr13_resolve_fixed32_gdn_path_bv_candidate(
        _FR13_FIXED32_MODE,
        geom_override=_read_tree_gdn_geom_override(),
    )
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH = (
    _fr13_resolve_fixed32_gdn_single_launch_expected_batch(
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
    )
)
# FR14. The ordered live gate's declared identity, bound ONCE beside the
# candidate the resolver above already validated it against, so the PASS
# artifacts below report the shape the gate actually ran in rather than a baked
# literal. A gate that announced K64 while serving K0 is the credential
# self-misdescription that already cost the qrow32 chain a full re-run. None
# when no ordered candidate is selected -- the non-ordered BV live PASS carries
# no draft-vocabulary claim at all, and must not start making one.
_FR13_FIXED32_GDN_ORDERED_QUALIFICATION_PROFILE = (
    _fr13_draft_vocab_profile(
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
    )
    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE in _FR13_GDN_ORDERED_CANDIDATES
    else None
)
_FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION = (
    _fr13_resolve_fixed32_gdn_gqa_group3_production(
        _FR13_FIXED32_MODE,
        geom_override=_read_tree_gdn_geom_override(),
    )
)


def _fr13_fixed32_gdn_gqa_group3_production_for_batch(
    batch_size: int,
) -> bool:
    """Return the exact credentialed batch; never widen B1/B4 authority."""
    credential = _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION
    if credential is None:
        return False
    batch = int(batch_size)
    credential_batch = credential.get("batch_size")
    if type(credential_batch) is not int or credential_batch not in (1, 4):
        raise RuntimeError(
            "FR13 GDN GQA-group3 production credential batch drifted"
        )
    return credential_batch == batch


_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS = (
    _fr13_resolve_fixed32_gdn_path_bv_production(
        _FR13_FIXED32_MODE,
        geom_override=_read_tree_gdn_geom_override(),
    )
)
_FR13_FIXED32_GDN_PATH_BV_PRODUCTION = (
    int(_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS["candidate_bv"])
    if _FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS is not None
    else None
)
_FR13_FIXED32_GDN_SINGLE_LAUNCH = (
    _fr13_resolve_fixed32_gdn_single_launch(
        _FR13_FIXED32_MODE,
        geom_override=_read_tree_gdn_geom_override(),
    )
)
# Resolved AFTER the diagnostic bool rather than beside the GQA-group3 sibling
# above, because the availability predicate immediately below has to see both
# forms of the same route in one expression. Dict-or-None, never a bool: the
# credential itself is what the narrowing helper and the guards interrogate.
_FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION = (
    _fr13_resolve_fixed32_gdn_single_launch_production(
        _FR13_FIXED32_MODE,
        geom_override=_read_tree_gdn_geom_override(),
    )
)


def _fr13_fixed32_gdn_single_launch_production_for_batch(
    batch_size: int,
) -> bool:
    """Return the exact credentialed batch; never widen B1/B4 authority.

    The credential proves ONE batch. B1 and B4 fold through the same kernel, so
    nothing structural would stop a B1 credential from authorizing B4 -- this is
    what stops it.
    """
    credential = _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION
    if credential is None:
        return False
    batch = int(batch_size)
    credential_batch = credential.get("batch_size")
    if type(credential_batch) is not int or credential_batch not in (1, 4):
        raise RuntimeError(
            "FR13 GDN single-launch production credential batch drifted"
        )
    return credential_batch == batch


_FR13_FIXED32_GDN_PRESCALED_PATH_BASE = (
    _fr13_resolve_fixed32_gdn_prescaled_path_base(
        bool(
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            or _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None
            or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
            or _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        )
    )
)
if (
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
    and (
        _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
        or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
    )
):
    raise RuntimeError(
        "FR13 fixed32 GDN path-BV diagnostic and production selectors are "
        "mutually exclusive"
    )
if _FR13_FIXED32_GDN_SINGLE_LAUNCH and (
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
    or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
):
    raise RuntimeError(
        "FR13 fixed32 GDN single-launch and path-BV selectors are mutually "
        "exclusive"
    )
if _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None and (
    _FR13_FIXED32_GDN_SINGLE_LAUNCH
    or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
):
    raise RuntimeError(
        "FR13 fixed32 GDN GQA-group3 production cannot inherit another GDN "
        "production selector"
    )
# The patched GDN call site hosts exactly ONE candidate. If the credentialed
# single-launch arm could co-exist with a sibling selector the refusals further
# down would stop being reachable -- whichever selector resolved first would
# simply win -- so the arm is refused here instead. The path-BV candidate is
# included because its "single_launch" value is the LIVE-GATE form of this same
# route: serving the gate arm and the promoted arm in one process would make the
# gate observe its own promotion. The diagnostic bool is deliberately NOT
# excluded; it is the campaign instrument that produced the credential, and the
# B4 selector and B1 call site below admit either one on its own terms.
if _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None and (
    _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
    or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
    or _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
):
    raise RuntimeError(
        "FR13 fixed32 GDN single-launch production cannot inherit another GDN "
        "production or path-BV selector"
    )
_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT = None
_FR13_FIXED32_GDN_BV_CAPTURES: dict[tuple[int, int, str], dict] = {}
_FR13_FIXED32_GDN_SINGLE_LAUNCH_COMPARATOR_EVENTS: list[
    dict[str, object]
] = []
_FR13_FIXED32_GDN_SINGLE_LAUNCH_REQUEST_TUPLES: set[tuple[str, ...]] = set()
_FR13_FIXED32_GDN_BV_LIVE_STATE = {
    "status": (
        "armed"
        if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
        else "disabled"
    ),
    "candidate_bv": _FR13_FIXED32_GDN_PATH_BV_CANDIDATE,
    "expected_batch": _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH,
    "diagnostic_identity": (
        _fr13_fixed32_gdn_single_launch_diagnostic_identity(
            _FR13_FIXED32_MODE,
            _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH,
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE,
        )
        if _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH is not None
        else None
    ),
    "graph_id": None,
    "batch_size": None,
    "records": 0,
    "comparator_event_count": 0,
}


def fixed32_gdn_bv_live_capture_begin(
    graph_id: int, batch_size: int
) -> None:
    """Open record collection for one final FULL fixed32 graph capture."""
    global _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT
    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is None:
        return
    identity = int(graph_id)
    batch = int(batch_size)
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        and batch != _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
    ):
        return
    if (
        _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES
        or identity <= 0
        or batch not in (1, 2, 3, 4)
        or _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT is not None
        or any(key[:2] == (batch, identity) for key in _FR13_FIXED32_GDN_BV_CAPTURES)
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate capture begin drift: "
            + repr((identity, batch, _FR13_FIXED32_MODE))
        )
    _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "batch_size": batch,
        "records": [],
    }


def _fr13_fixed32_gdn_bv_live_capture_register(record: dict) -> None:
    """Retain the persistent operands of one captured fixed32 GDN call."""
    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is None:
        return
    context = _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT
    # Eager warmups and profile-only CUDA captures deliberately have no live
    # gate context. The final capture-end hook requires the exact record count,
    # so a missing final begin cannot pass silently.
    if context is None:
        return
    required = {
        "snapshot",
        "restore",
        "run",
        "byte_equal",
        "surface_names",
    }
    expected_surfaces = (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES
        if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        else _FR13_FIXED32_GDN_BV_SURFACES
    )
    if (
        not isinstance(context, dict)
        or set(context) != {"graph_id", "batch_size", "records"}
        or not isinstance(record, dict)
        or set(record) != required
        or tuple(record["surface_names"]) != expected_surfaces
        or not all(callable(record[name]) for name in required - {"surface_names"})
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate capture record drift"
        )
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate record was not captured by CUDA"
        )
    context["records"].append(record)


def fixed32_gdn_bv_live_capture_end(
    graph_id: int,
    graph_signature: str,
    batch_size: int,
    expected_records: int,
) -> None:
    """Freeze the launch records and bind them to the signed graph identity."""
    global _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT
    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is None:
        return
    identity = int(graph_id)
    batch = int(batch_size)
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        and batch != _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
    ):
        if _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT is not None:
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch skipped capture leaked context"
            )
        return
    context = _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT
    expected = int(expected_records)
    records = context.get("records") if isinstance(context, dict) else None
    signature = str(graph_signature)
    if (
        not isinstance(context, dict)
        or int(context.get("graph_id", -1)) != identity
        or int(context.get("batch_size", -1)) != batch
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
        or expected
        != (
            48
            if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
            else 48 * batch
        )
        or not isinstance(records, list)
        or len(records) != expected
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate capture end drift: "
            + repr(
                (
                    identity,
                    batch,
                    expected,
                    len(records) if isinstance(records, list) else None,
                )
            )
        )
    key = (batch, identity, signature)
    if key in _FR13_FIXED32_GDN_BV_CAPTURES:
        raise RuntimeError("FR13 fixed32 GDN BV capture identity was reused")
    _FR13_FIXED32_GDN_BV_CAPTURES[key] = {
        "batch_size": batch,
        "graph_id": identity,
        "graph_signature": signature,
        "records": tuple(records),
    }
    _FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT = None


def _fr13_tensor_byte_equal(left, right) -> bool:
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


def _fr13_fixed32_gdn_bv_compare_records(
    records, candidate_bv: int
) -> dict[str, int]:
    """Run BV8 then the selected candidate and restore served bytes."""
    candidate = int(candidate_bv)
    if candidate not in (16, 32, 64, 128) or candidate == 8:
        raise RuntimeError(
            "FR13 fixed32 GDN BV live gate refused stock-vs-stock: "
            f"reference=8 candidate={candidate}"
        )
    checked = 0
    for index, record in enumerate(records):
        snapshot = record["snapshot"]
        restore = record["restore"]
        run = record["run"]
        byte_equal = record["byte_equal"]
        baseline = snapshot()
        if tuple(baseline) != _FR13_FIXED32_GDN_BV_SURFACES:
            raise RuntimeError(
                "FR13 fixed32 GDN BV live-gate baseline surface drift at "
                f"record {index}: {tuple(baseline)!r}"
            )
        try:
            reference = run(8)
            reference_surfaces = snapshot()
            restore(baseline)
            candidate_result = run(candidate)
            candidate_surfaces = snapshot()
            if (
                set(reference) != {"block_v", "launch_key", "output"}
                or set(candidate_result)
                != {"block_v", "launch_key", "output"}
                or int(reference["block_v"]) != 8
                or int(candidate_result["block_v"]) != candidate
                or reference["launch_key"] == candidate_result["launch_key"]
            ):
                raise RuntimeError(
                    "FR13 fixed32 GDN BV live gate detected false "
                    "stock-vs-stock launch metadata at record "
                    f"{index}: reference={reference!r} "
                    f"candidate={candidate_result!r}"
                )
            mismatches = []
            if not byte_equal(reference["output"], candidate_result["output"]):
                mismatches.append("output")
            for name in _FR13_FIXED32_GDN_BV_SURFACES:
                if not byte_equal(
                    reference_surfaces[name], candidate_surfaces[name]
                ):
                    mismatches.append(name)
            if mismatches:
                raise RuntimeError(
                    "FR13 fixed32 GDN BV live-gate byte mismatch at record "
                    f"{index}: {mismatches}"
                )
            checked += 1
        finally:
            restore(baseline)
            restored = snapshot()
            restore_bad = [
                name
                for name in _FR13_FIXED32_GDN_BV_SURFACES
                if not byte_equal(restored[name], baseline[name])
            ]
            if restore_bad:
                raise RuntimeError(
                    "FR13 fixed32 GDN BV live gate failed to restore served "
                    f"BV8 bytes at record {index}: {restore_bad}"
                )
    return {"records": checked, "reference_bv": 8, "candidate_bv": candidate}


def _fr13_fixed32_gdn_single_launch_compare_records(
    records,
) -> dict[str, object]:
    """Run stock then ordered single-launch and restore served bytes."""
    selector = globals().get("_FR13_FIXED32_GDN_PATH_BV_CANDIDATE")
    candidate_id = {
        "gqa_group3": "fixed32_gdn_single_launch_gqa_group3_v1",
        "gqa_group3_bv16": (
            "fixed32_gdn_single_launch_gqa_group3_bv16_v1"
        ),
    }.get(selector, _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID)
    checked = 0
    for index, record in enumerate(records):
        snapshot = record["snapshot"]
        restore = record["restore"]
        run = record["run"]
        byte_equal = record["byte_equal"]
        baseline = snapshot()
        if tuple(baseline) != _FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES:
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch baseline surface drift at "
                f"record {index}: {tuple(baseline)!r}"
            )
        try:
            reference = run("reference")
            reference_surfaces = snapshot()
            restore(baseline)
            candidate = run(candidate_id)
            candidate_surfaces = snapshot()
            if (
                set(reference)
                != {"candidate", "physical_launches", "output"}
                or set(candidate)
                != {"candidate", "physical_launches", "output"}
                or reference["candidate"] != "fixed32_gdn_two_launch_reference_v1"
                or candidate["candidate"]
                != candidate_id
                or reference["physical_launches"] != 2
                or candidate["physical_launches"] != 1
            ):
                raise RuntimeError(
                    "FR13 fixed32 GDN single-launch gate detected launch "
                    f"identity drift at record {index}: reference={reference!r} "
                    f"candidate={candidate!r}"
                )
            mismatches = []
            if not byte_equal(reference["output"], candidate["output"]):
                mismatches.append("output")
            for name in _FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES[1:]:
                if not byte_equal(
                    reference_surfaces[name], candidate_surfaces[name]
                ):
                    mismatches.append(name)
            if mismatches:
                raise RuntimeError(
                    "FR13 fixed32 GDN single-launch byte mismatch at record "
                    f"{index}: {mismatches}"
                )
            checked += 1
        finally:
            restore(baseline)
            restored = snapshot()
            restore_bad = [
                name
                for name in _FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES
                if not byte_equal(restored[name], baseline[name])
            ]
            if restore_bad:
                raise RuntimeError(
                    "FR13 fixed32 GDN single-launch gate failed to restore "
                    f"served stock bytes at record {index}: {restore_bad}"
                )
    return {
        "records": checked,
        "candidate": candidate_id,
        "reference_bv": 8,
        "candidate_bv": 16 if selector == "gqa_group3_bv16" else 8,
        "reference_physical_launches": 2,
        "candidate_physical_launches": 1,
    }


def fixed32_gdn_bv_live_gate_on_replay(
    graph_id: int,
    graph_signature: str,
    census_graph_signature: str,
    batch_size: int,
    expected_records: int,
    event_index: int | None = None,
    forward_step_index: int | None = None,
    request_id_sha256s: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Compare one distinct authenticated request tuple without serving it."""
    state = _FR13_FIXED32_GDN_BV_LIVE_STATE
    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is None:
        return dict(state)
    identity = int(graph_id)
    batch = int(batch_size)
    signature = str(graph_signature)
    census_signature = str(census_graph_signature)
    expected = int(expected_records)
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        and batch != _FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH
    ):
        return {
            **state,
            "status": "not_expected_batch",
            "observed_batch": batch,
        }
    single_launch = (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
    )
    if not single_launch and state["status"] == "passed":
        return dict(state)
    if state["status"] != "armed":
        raise RuntimeError(
            "FR13 fixed32 GDN BV live gate is not runnable: " + repr(state)
        )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 GDN BV live gate cannot execute during capture"
        )
    key = (batch, identity, signature)
    capture = _FR13_FIXED32_GDN_BV_CAPTURES.get(key)
    records = capture.get("records") if isinstance(capture, dict) else None
    if (
        not isinstance(capture, dict)
        or int(capture.get("batch_size", -1)) != batch
        or int(capture.get("graph_id", -1)) != identity
        or capture.get("graph_signature") != signature
        or len(census_signature) != 64
        or any(
            character not in "0123456789abcdef"
            for character in census_signature
        )
        or expected
        != (
            48
            if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
            else 48 * batch
        )
        or not isinstance(records, tuple)
        or len(records) != expected
    ):
        raise RuntimeError(
            "FR13 fixed32 GDN BV live-gate replay/capture drift: "
            + repr((identity, batch, expected, capture))
        )
    task_marker = _fr13_fixed32_gdn_bv_real_event_marker()
    request_tuple: tuple[str, ...] = ()
    containing_event_index = -1
    containing_forward_step = -1
    if single_launch:
        containing_event_index = (
            int(event_index) if event_index is not None else -1
        )
        containing_forward_step = (
            int(forward_step_index)
            if forward_step_index is not None
            else -1
        )
        request_tuple = (
            tuple(request_id_sha256s)
            if isinstance(request_id_sha256s, tuple)
            else ()
        )
        if (
            containing_event_index < 0
            or containing_forward_step < 0
            or len(request_tuple) != batch
            or len(set(request_tuple)) != batch
            or any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in request_tuple
            )
        ):
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch event/request binding drift"
            )
        if request_tuple in _FR13_FIXED32_GDN_SINGLE_LAUNCH_REQUEST_TUPLES:
            return {
                **state,
                "status": "armed",
                "comparison_status": "already_compared_request_tuple",
                "comparator": None,
            }
    state["status"] = "running"
    try:
        if single_launch:
            result = _fr13_fixed32_gdn_single_launch_compare_records(records)
        else:
            result = _fr13_fixed32_gdn_bv_compare_records(
                records, _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            )
        if single_launch:
            comparator = {
                "schema": "fr13.fixed32.gdn_single_launch.comparator_event.v1",
                "mode": _FR13_FIXED32_MODE,
                "batch_size": batch,
                "runtime_capture_manifest_sha256": signature,
                "structural_graph_signature": census_signature,
                "reference": "fixed32_gdn_two_launch_reference_v1",
                "candidate": result["candidate"],
                "reference_physical_launches_per_request_layer": int(
                    result["reference_physical_launches"]
                ),
                "candidate_physical_launches_per_request_layer": int(
                    result["candidate_physical_launches"]
                ),
                "records": int(result["records"]),
                "compared_byte_surfaces": list(
                    _FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES
                ),
                "raw_byte_equal": True,
                "state_restored": True,
                "reference_served": True,
                "candidate_served": False,
                "comparison_order": [
                    "reference",
                    "restore_baseline",
                    "candidate",
                    "restore_baseline_in_finally",
                ],
                "census_event_id": (
                    f"{_FR13_FIXED32_MODE}:{os.getpid()}:"
                    f"{containing_event_index}"
                ),
                "census_event_index": containing_event_index,
                "census_forward_step_index": containing_forward_step,
                "request_id_sha256s": list(request_tuple),
                "observed_task_marker": task_marker,
            }
            proposed_events = [
                *_FR13_FIXED32_GDN_SINGLE_LAUNCH_COMPARATOR_EVENTS,
                comparator,
            ]
            _fr13_fixed32_gdn_single_launch_observation_emit(
                batch_size=batch,
                comparator_events=proposed_events,
            )
    except Exception:
        state["status"] = "failed"
        raise
    if single_launch:
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_COMPARATOR_EVENTS.append(comparator)
        _FR13_FIXED32_GDN_SINGLE_LAUNCH_REQUEST_TUPLES.add(request_tuple)
        state.update(
            status="armed",
            graph_id=identity,
            graph_signature=signature,
            batch_size=batch,
            records=int(result["records"]),
            comparator_event_count=len(
                _FR13_FIXED32_GDN_SINGLE_LAUNCH_COMPARATOR_EVENTS
            ),
        )
        print(
            "[FR13_FIXED32_GDN_SINGLE_LAUNCH COMPARATOR] "
            f"graph_id={identity} batch={batch} records={result['records']} "
            f"event={containing_event_index} requests={len(request_tuple)} "
            "reference_launches=2 candidate_launches=1 "
            "served=reference restored=1",
            flush=True,
        )
        return {
            **state,
            "comparison_status": "compared_distinct_request_tuple",
            "comparator": dict(comparator),
        }
    state.update(
        status="passed",
        graph_id=identity,
        graph_signature=signature,
        batch_size=batch,
        records=int(result["records"]),
    )
    _fr13_fixed32_gdn_bv_live_pass_emit(
        task_marker=task_marker,
        batch_size=batch,
        graph_signature=census_signature,
        result=result,
    )
    del _FR13_FIXED32_GDN_BV_CAPTURES[key]
    print(
        "[FR13_FIXED32_GDN_BV_LIVE_GATE PASS] "
        f"graph_id={identity} batch={batch} records={result['records']} "
        f"reference_bv=8 candidate_bv={result['candidate_bv']} "
        f"candidate={result.get('candidate', _FR13_FIXED32_GDN_PATH_BV_CANDIDATE_ID)} "
        "surfaces=output,ring_k,ring_v,ring_a,ring_b,flags,counter "
        "served_bv=8 restored=1",
        flush=True,
    )
    return dict(state)


def fixed32_gdn_bv_live_gate_report() -> dict[str, object]:
    return dict(_FR13_FIXED32_GDN_BV_LIVE_STATE)


_FR13_SUBTREE_ROUTE_REQUESTED = (
    subtree_parallel_on() or _FR13_FIXED32_MODE is not None
)
_FR13_SUBTREE_SELFCHECK_REQUESTED = subtree_parallel_selfcheck_on()
if _FR13_SUBTREE_SELFCHECK_REQUESTED and not _FR13_SUBTREE_ROUTE_REQUESTED:
    raise RuntimeError(
        "FR13_SUBTREE_PARALLEL_SELFCHECK is armed while the path route "
        "is disabled"
    )
if (
    _FR13_FIXED32_GDN_SINGLE_LAUNCH
    or _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None
    or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
) and _FR13_SUBTREE_SELFCHECK_REQUESTED:
    raise RuntimeError(
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE cannot inherit the legacy "
        "subtree selfcheck"
    )


_FR13_SUBTREE_CACHE: dict = {}


def _subtree_cache_key(
    n_actual: int, vh: int, dv: int, dk: int, device
) -> tuple:
    """Shape/device-complete key for graph-stable path tensors and scratch."""
    return (
        "subtree",
        int(n_actual),
        int(vh),
        int(dv),
        int(dk),
        str(torch.device(device)),
    )


# Exact Hydra23 verifier parent vector (implicit root + 23 draft nodes).
# The device-length path kernel executes only real nodes, so all nine paths
# whose parents are produced by the root prefix can share level 1. This keeps
# the dependency length (4 + 8 = 12), exports only nodes {0, 1, 4, 8}, and
# needs two launches. Every other topology uses the generic decomposition.
_FR13_HYDRA23_PARENT = (
    -1, 0, 0, 0, 1, 1, 1, 2, 4, 4, 4, 7,
    8, 8, 8, 11, 12, 15, 16, 18, 19, 20, 21, 22,
)
_FR13_HYDRA23_SUBTREE_LEVELS = (
    (
        ((0, 1, 4, 8), -1),
    ),
    (
        ((12, 16, 18, 19, 20, 21, 22, 23), 8),
        ((2, 7, 11, 15, 17), 0),
        ((3,), 0),
        ((5,), 1),
        ((6,), 1),
        ((9,), 4),
        ((10,), 4),
        ((13,), 8),
        ((14,), 8),
    ),
)

# Exact root-inclusive fixed32 physical topology. Both logical modes use these
# rows and differ only in the sampler validity mask.
_FR13_FIXED32_PARENT = (
    -1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9,
    9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29, 30,
)
_FR13_FIXED32_SUBTREE_LEVELS = (
    (
        ((0, 1, 4, 9, 14), -1),
    ),
    (
        ((19, 24, 26, 28, 29, 30, 31), 14),
        ((2, 7, 12, 17, 22), 0),
        ((3, 8, 13, 18, 23, 25, 27), 0),
        ((5,), 1),
        ((6,), 1),
        ((10,), 4),
        ((11,), 4),
        ((15,), 9),
        ((16,), 9),
        ((20,), 14),
        ((21,), 14),
    ),
)
_FR13_FIXED32_EXPORT_NODES = (0, 1, 4, 9, 14)
_FR13_FIXED32_EXPORT_SLOTS = len(_FR13_FIXED32_EXPORT_NODES)
_FR13_FIXED32_MAX_BATCH = 4
_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID = (
    "fixed32_gdn_single_launch_tree_v2"
)
_FR13_FIXED32_GDN_GQA_GROUP3_CANDIDATE_ID = (
    "fixed32_gdn_single_launch_gqa_group3_v1"
)
_FR13_FIXED32_GDN_GQA_GROUP3_BV16_CANDIDATE_ID = (
    "fixed32_gdn_single_launch_gqa_group3_bv16_v1"
)
_FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH = None
if (
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
    in (
        _FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE,
        _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE,
    )
    or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
):
    from lumo_flywheel_serving.fr13_gdn_gqa_group3 import (
        BV16_CANDIDATE as _fr13_fixed32_gdn_gqa_group3_bv16_candidate_id,
        CANDIDATE as _fr13_fixed32_gdn_gqa_group3_candidate_id,
        launch_fixed32_gdn_gqa_group3_source_candidate,
    )

    if (
        _fr13_fixed32_gdn_gqa_group3_candidate_id
        != _FR13_FIXED32_GDN_GQA_GROUP3_CANDIDATE_ID
        or _fr13_fixed32_gdn_gqa_group3_bv16_candidate_id
        != _FR13_FIXED32_GDN_GQA_GROUP3_BV16_CANDIDATE_ID
    ):
        raise RuntimeError("FR13 fixed32 GDN GQA-group3 identity drift")
    _FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH = (
        launch_fixed32_gdn_gqa_group3_source_candidate
    )


def _fr13_fixed32_gdn_ordered_candidate_id() -> str:
    candidate = _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
    if candidate == _FR13_FIXED32_GDN_SINGLE_LAUNCH_GATE_VALUE:
        return _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID
    if candidate == _FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE:
        return _FR13_FIXED32_GDN_GQA_GROUP3_CANDIDATE_ID
    if candidate == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE:
        return _FR13_FIXED32_GDN_GQA_GROUP3_BV16_CANDIDATE_ID
    raise RuntimeError(
        "FR13 fixed32 ordered GDN candidate identity requested without its gate"
    )
# Each root-chain node is immediately followed by its terminal side paths.
# The member order is the established fixed32 level-1 descriptor order.
_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS = (
    (0, (1, 2)),
    (1, (3, 4)),
    (4, (5, 6)),
    (9, (7, 8)),
    (14, (0, 9, 10)),
)
_FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID = (
    "fixed32_sfwd_state_fusion_v1"
)
_FR13_FIXED32_SFWD_STATE_FUSION_ENABLED = (
    "/logs/fr13_fixed32_sfwd_state_fusion_byte_ab.enabled"
)
_FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT = (
    "/logs/fr13_fixed32_sfwd_state_fusion.real_event.arm"
)
_FR13_FIXED32_SFWD_STATE_FUSION_PASS = (
    "/logs/fr13_fixed32_sfwd_state_fusion.live_pass.json"
)
_FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
_FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS = tuple(
    f"swe_verified:{task_id}"
    for task_id in _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_IDS
)
_FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
_FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE = {
    "task_markers": None,
    "batch": None,
    "passed": set(),
    "attempts": {},
    "failed": False,
}
_FR13_FIXED32_BATCH_GDN_BYTE_AB_STATE = {
    "passed": set(),
    "attempts": {},
    "waiting_announced": set(),
}
_FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED = (
    "/logs/fr13_fixed32_batch_gdn_byte_ab.enabled"
)
_FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB_ENABLED = (
    "/logs/fr13_fixed32_batch_gdn_graph_byte_ab.enabled"
)
_FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT = (
    "/logs/fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
)
_FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS = (
    "/logs/fr13_fixed32_batch_gdn_byte_ab.pass.json"
)
_FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM = (
    "/logs/fr13_fixed32_batch_gdn_production.arm"
)
_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_SIDECARS = (
    "/logs/fr13_fixed32_batch_gdn_bv_candidate.flag",
    "/tmp/fr13_fixed32_batch_gdn_bv_candidate.flag",
)
_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION_SIDECARS = (
    "/logs/fr13_fixed32_batch_gdn_bv_production.flag",
    "/tmp/fr13_fixed32_batch_gdn_bv_production.flag",
)
_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_ID = "fixed32_batch_gdn_bv_v2"
_FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID = "fixed32_batch_gdn_bv8_v1"
_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_SIDECAR_SCHEMA = (
    "fr13.fixed32.batch_gdn.bv8.production_sidecar.v1"
)
_FR13_FIXED32_BATCH_GDN_EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
_FR13_FIXED32_BATCH_GDN_EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
_FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL = "per_request_tree_gdn_path"
_FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL = "fixed32_batch_tree_gdn_path"
_FR13_FIXED32_BATCH_GDN_BV_BYTE_SURFACES = (
    "out",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "state_export_compact",
    "state_export_untouched_tail",
    "flags",
    "invocation_counter",
)
_FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES = (
    "out",
    "export",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "invocation_counter",
)
# ``export`` is one device/shape cache shared by all 48 layers, so after a full
# replay it contains only the final layer's bytes. It remains covered by the
# reference/candidate compact+tail checks and the restore check, but cannot be
# used as a per-layer graph-baseline oracle.
_FR13_FIXED32_BATCH_GDN_GRAPH_STABLE_SURFACES = (
    "out",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
)
_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT = None
_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES: dict[int, dict] = {}
_FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE = {
    "status": "disabled",
    "candidate_bv": None,
    "graph_id": None,
    "graph_signature": None,
    "batch_size": None,
    "records": 0,
}
_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_ENGAGEMENT = (
    "/logs/fr13_fixed32_batch_gdn_bv64.production_engagement.json"
)
_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT = None
_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURES: dict[int, dict] = {}
_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_PUBLISHED = False
_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_ENGAGEMENT = (
    "/logs/fr13_fixed32_batch_gdn_bv8.production_engagement.json"
)
_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT = None
_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES: dict[int, dict] = {}
_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED = False
_FR13_FIXED32_PARENT_SHA256 = (
    "7abd25e38323d6c088eb627785b5c190b2e878b0a710bb349e2d690852a06ddd"
)
_FR13_FIXED32_ANCESTRY_SHA256 = (
    "90873d81e83ce1644ee4701e043b7e9d26e83b7a7ca752d538a0e6eed1946dad"
)
_FR13_FIXED32_LEVELS_SHA256 = (
    "65d91ed364a87abd50d184d902c5d045c4eebf77d172610707fc419667099311"
)
_FR13_FIXED32_COVERAGE_SHA256 = (
    "23b22df6bf551a4e788327db3b3d3d96e1eca49078d2c6bd0049da2d390eca8b"
)
_FR13_FIXED32_SFWD_CONV_STATE_LEN = 34


def fixed32_sfwd_state_fusion_contract(
    batch_size: int,
    *,
    tree_rows: int,
    conv_width: int,
    conv_state_len: int,
) -> dict[str, object]:
    """Validate the closed full-vocabulary SFWD state-fusion geometry.

    The candidate is intentionally narrower than the generic tree-conv path:
    exact fixed32 rows, the Qwen3-Next width-4 BF16 conv, and B1-B4 only.  It
    replaces the per-layer prior gather, source construction/copy, four-tap
    accumulation, activation, and persistent commit-source write with one
    B-folded launch.  The existing two-level GDN scan remains physically
    ``[1, 11]`` and continues to own byte-copy ring export and freshness flags.
    """
    batch = int(batch_size)
    rows = int(tree_rows)
    width = int(conv_width)
    state_len = int(conv_state_len)
    if batch not in (1, 2, 3, 4):
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires B1-B4, "
            f"got B={batch}"
        )
    if rows != 32:
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires exactly 32 physical "
            f"rows per request, got {rows}"
        )
    if width != 4 or state_len != _FR13_FIXED32_SFWD_CONV_STATE_LEN:
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires the exact width/state "
            "geometry "
            f"(4, {_FR13_FIXED32_SFWD_CONV_STATE_LEN}), "
            f"got ({width}, {state_len})"
        )
    source_rows = width - 1 + rows + 1
    return {
        "candidate": _FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID,
        "batch_size": batch,
        "physical_rows_per_request": rows,
        "logical_rows": batch * rows,
        "conv_width": width,
        "conv_state_len": state_len,
        "source_rows_per_request": source_rows,
        "source_rows": batch * source_rows,
        "conv_state_launches_per_layer": 1,
        "gdn_level_path_programs": (batch, 11 * batch),
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export": True,
        "gdn_flags_export": True,
        "reference_always_served": True,
    }


def _fr13_fixed32_sfwd_state_fusion_exact4_markers(
    path: str,
) -> tuple[str, ...]:
    """Read the engine-published canonical exact4 marker set fail-closed."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION exact4 marker is unavailable"
        ) from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size <= 0
            or info.st_size > 1024
        ):
            raise RuntimeError(
                "FR13_FIXED32_SFWD_STATE_FUSION exact4 marker identity is invalid"
            )
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RuntimeError(
                    "FR13_FIXED32_SFWD_STATE_FUSION exact4 marker was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(
                "FR13_FIXED32_SFWD_STATE_FUSION exact4 marker changed while reading"
            )
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    expected = (
        "\n".join(_FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS) + "\n"
    ).encode("ascii")
    if raw != expected:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires the canonical exact4 "
            "authenticated task markers"
        )
    return _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS


def fixed32_sfwd_state_fusion_gate_control(
    *,
    environ=None,
    enabled_path: str | None = None,
    event_path: str | None = None,
) -> tuple[bool, tuple[str, ...] | None]:
    """Resolve the default-off authenticated exact4 B4 byte-gate arm."""
    env = os.environ if environ is None else environ
    raw = str(env.get("FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB", ""))
    if raw not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB must be exactly 0 or 1"
        )
    enabled = enabled_path or str(
        env.get(
            "FR13_FIXED32_SFWD_STATE_FUSION_ENABLED_PATH",
            _FR13_FIXED32_SFWD_STATE_FUSION_ENABLED,
        )
    )
    event = event_path or str(
        env.get(
            "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH",
            _FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT,
        )
    )
    armed = raw == "1" or os.path.exists(enabled)
    if not armed:
        return False, None
    state = _FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE
    if bool(state.get("failed")):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION byte gate previously mismatched"
        )
    if len(state.get("passed", ())) == 48:
        return True, None
    if not os.path.exists(event):
        return armed, None
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION byte gate requires an exact "
            "fixed32 runtime"
        )
    return True, _fr13_fixed32_sfwd_state_fusion_exact4_markers(event)


def _fr13_resolve_fixed32_batch_gdn_bv(
    fixed32_mode: str | None,
    *,
    env_name: str,
    sidecars,
    environ=None,
    geom_override=None,
    allow_reference_bv: bool = False,
) -> int | None:
    """Resolve one explicit BV selector for the two-launch B2-B4 route."""
    env = os.environ if environ is None else environ
    sources: list[tuple[str, str]] = []
    raw_env = env.get(env_name)
    if raw_env is not None and str(raw_env).strip():
        sources.append((f"env:{env_name}", str(raw_env).strip()))
    for path in tuple(sidecars):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="ascii") as handle:
                value = handle.read(16)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(
                f"{env_name}: cannot read sidecar {path}: {error}"
            ) from error
        if len(value) >= 16:
            raise RuntimeError(f"{env_name}: sidecar exceeds 15 bytes: {path}")
        sources.append((f"sidecar:{path}", value.strip()))
    if not sources:
        return None
    allowed = ("8", "16", "32", "64", "128") if allow_reference_bv else (
        "16",
        "32",
        "64",
        "128",
    )
    invalid = [
        (source, value)
        for source, value in sources
        if value not in allowed
    ]
    values = {value for _source, value in sources}
    if invalid or len(values) != 1:
        allowed_text = ", ".join(allowed[:-1]) + ", or " + allowed[-1]
        raise RuntimeError(
            f"{env_name} requires one of {allowed_text} from agreeing "
            f"sources: {sources!r}"
        )
    if fixed32_mode not in _FR13_FIXED32_MODES:
        raise RuntimeError(f"{env_name} requires an exact fixed32 mode")
    if geom_override != {"BV": 8}:
        raise RuntimeError(
            f"{env_name} requires the B1/served reference geometry pinned "
            "exactly to BV=8"
        )
    return int(values.pop())


_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE = (
    _fr13_resolve_fixed32_batch_gdn_bv(
        _FR13_FIXED32_MODE,
        env_name="FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
        sidecars=_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_SIDECARS,
        geom_override=_read_tree_gdn_geom_override(),
        allow_reference_bv=True,
    )
)
_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION = (
    _fr13_resolve_fixed32_batch_gdn_bv(
        _FR13_FIXED32_MODE,
        env_name="FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
        sidecars=_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION_SIDECARS,
        geom_override=_read_tree_gdn_geom_override(),
        allow_reference_bv=True,
    )
)
if (
    _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is not None
    and _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None
):
    raise RuntimeError(
        "FR13 fixed32 batched GDN wide-BV diagnostic and production "
        "selectors are mutually exclusive"
    )
if (
    _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is not None
    or _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None
) and (
    _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
    or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
):
    raise RuntimeError(
        "FR13 fixed32 B1 path-BV and B2-B4 batched wide-BV selectors are "
        "mutually exclusive"
    )
if (
    _FR13_FIXED32_GDN_SINGLE_LAUNCH
    or _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None
    or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
) and (
    _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is not None
    or _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None
):
    raise RuntimeError(
        "FR13 fixed32 GDN single-launch and batched BV selectors are mutually "
        "exclusive"
    )


def _fr13_fixed32_batch_gdn_source_sha256() -> str:
    try:
        payload = Path(__file__).resolve().read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"FR13 fixed32 batched GDN cannot hash its source: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _fr13_fixed32_batch_gdn_byte_ab_control() -> tuple[bool, str | None]:
    """Resolve the eager real-event byte gate without worker-env assumptions.

    The launcher creates the enabled sidecar before boot, then authenticated
    engine ingress creates the event arm after admitting the first canonical
    SWE-Verified request and before executing it. This prevents graph warmup or
    a synthetic probe from satisfying the gate. The event marker is included
    in every gate record.
    """
    raw = os.environ.get("FR13_FIXED32_BATCH_GDN_BYTE_AB", "")
    if raw not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB must be exactly 0 or 1"
        )
    enabled_path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED_PATH",
        _FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED,
    )
    event_path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH",
        _FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT,
    )
    enabled = raw == "1" or os.path.exists(enabled_path)
    if not os.path.exists(event_path):
        return enabled, None
    return enabled, _fr13_fixed32_batch_gdn_real_event_marker(event_path)


def _fr13_fixed32_batch_gdn_real_event_marker(
    event_path: str | None = None,
) -> str:
    """Read the authenticated SWE-Verified event arm shared by both gates."""
    path = event_path or os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH",
        _FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT,
    )
    if not os.path.exists(path):
        raise RuntimeError(
            "FR13 fixed32 batched GDN graph byte gate requires an authenticated "
            f"real-event arm at {path}"
        )
    try:
        with open(path, encoding="ascii") as handle:
            marker = handle.read(257)
    except (OSError, UnicodeError) as error:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB cannot read real-event arm "
            f"{path}: {error}"
        ) from error
    if len(marker) > 256:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB real-event marker exceeds 256 bytes"
        )
    marker = marker.strip()
    prefix = "swe_verified:"
    task_id = marker[len(prefix):] if marker.startswith(prefix) else ""
    if not task_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        for character in task_id
    ):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB real-event marker must be "
            "swe_verified:<task_id>"
        )
    return marker


def _fr13_fixed32_batch_gdn_graph_byte_ab_control() -> bool:
    """Resolve the graph-replay B4 gate independently of the eager gate."""
    raw = os.environ.get("FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB", "")
    if raw not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB must be exactly 0 or 1"
        )
    enabled_path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB_ENABLED_PATH",
        _FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB_ENABLED,
    )
    return raw == "1" or os.path.exists(enabled_path)


def _fr13_fixed32_batch_gdn_production_control() -> dict[str, object] | None:
    """Validate the explicit post-gate production arm and PASS record."""
    raw = os.environ.get("FR13_FIXED32_BATCH_GDN_PRODUCTION", "")
    if raw not in ("", "0", "1"):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_PRODUCTION must be exactly 0 or 1"
        )
    arm_path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM_PATH",
        _FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM,
    )
    if raw != "1" and not os.path.exists(arm_path):
        return None
    pass_path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH",
        _FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS,
    )
    if os.path.islink(pass_path) or not os.path.isfile(pass_path):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_PRODUCTION requires a readable live-gate "
            f"PASS record (regular file) at {pass_path}"
        )
    try:
        with open(pass_path, encoding="ascii") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_PRODUCTION requires a readable live-gate "
            f"PASS record at {pass_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_PRODUCTION live-gate PASS record is invalid"
        )
    production_bv = _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION
    if production_bv == 8:
        credential = payload
        expected_credential_keys = {
            "schema",
            "status",
            "candidate",
            "batch",
            "subset_sha256",
            "task_ids",
            "task_marker",
            "kernel_source_sha256",
            "runtime_manifest_sha256",
            "gate_runner_sha256",
            "live_result_sha256",
            "gate_verdict_sha256",
            "reference_kernel_structure",
            "candidate_kernel_structure",
            "reference_bv",
            "candidate_bv",
            "reference_physical_launches_per_layer",
            "candidate_physical_launches_per_layer",
            "production_default_enabled",
            "live_result",
            "gate_verdict",
        }
        live_result = credential.get("live_result")
        gate_verdict = credential.get("gate_verdict")

        def _credential_digest(value: object) -> str:
            raw_value = (
                json.dumps(
                    value,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("ascii")
            return hashlib.sha256(raw_value).hexdigest()

        task_marker = credential.get("task_marker")
        ledger_sha256 = (
            gate_verdict.get("engine_ledger_chain_head_sha256")
            if isinstance(gate_verdict, dict)
            else None
        )
        runtime_manifest_sha256 = credential.get("runtime_manifest_sha256")
        gate_runner_sha256 = credential.get("gate_runner_sha256")
        expected_task_markers = {
            "swe_verified:" + task_id
            for task_id in _FR13_FIXED32_BATCH_GDN_EXACT4_TASK_IDS
        }
        credential_invalid = (
            set(credential) != expected_credential_keys
            or credential.get("schema")
            != _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_SIDECAR_SCHEMA
            or credential.get("status") != "qualified"
            or credential.get("candidate")
            != _FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID
            or credential.get("batch") != 4
            or credential.get("subset_sha256")
            != _FR13_FIXED32_BATCH_GDN_EXACT4_SUBSET_SHA256
            or credential.get("task_ids")
            != list(_FR13_FIXED32_BATCH_GDN_EXACT4_TASK_IDS)
            or task_marker not in expected_task_markers
            or credential.get("kernel_source_sha256")
            != _fr13_fixed32_batch_gdn_source_sha256()
            or not isinstance(runtime_manifest_sha256, str)
            or len(runtime_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in runtime_manifest_sha256
            )
            or not isinstance(gate_runner_sha256, str)
            or len(gate_runner_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in gate_runner_sha256
            )
            or credential.get("reference_kernel_structure")
            != _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL
            or credential.get("candidate_kernel_structure")
            != _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL
            or credential.get("reference_bv") != 8
            or credential.get("candidate_bv") != 8
            or credential.get("reference_physical_launches_per_layer") != 8
            or credential.get("candidate_physical_launches_per_layer") != 2
            or credential.get("production_default_enabled") is not False
            or not isinstance(live_result, dict)
            or not isinstance(gate_verdict, dict)
            or credential.get("live_result_sha256")
            != _credential_digest(live_result)
            or credential.get("gate_verdict_sha256")
            != _credential_digest(gate_verdict)
            or live_result.get("task_marker") != task_marker
            or gate_verdict.get("task_marker") != task_marker
            or gate_verdict.get("graph_live_pass_sha256")
            != credential.get("live_result_sha256")
            or gate_verdict.get("kernel_source_sha256")
            != credential.get("kernel_source_sha256")
            or gate_verdict.get("runtime_manifest_sha256")
            != runtime_manifest_sha256
            or gate_verdict.get("gate_runner_sha256")
            != gate_runner_sha256
            or gate_verdict.get("schema")
            != "fr13.fixed32.batch_gdn.b4_diagnostic.v1"
            or gate_verdict.get("status") != "pass"
            or gate_verdict.get("run_classification")
            != "exact4_b4_graph_byte_diagnostic"
            or gate_verdict.get("timing_eligible") is not False
            or gate_verdict.get("floor_acceptance_eligible") is not False
            or gate_verdict.get("subset_sha256")
            != _FR13_FIXED32_BATCH_GDN_EXACT4_SUBSET_SHA256
            or gate_verdict.get("task_ids")
            != list(_FR13_FIXED32_BATCH_GDN_EXACT4_TASK_IDS)
            or gate_verdict.get("gate_mode") != "post_replay_shadow"
            or gate_verdict.get("graph_id") != live_result.get("graph_id")
            or gate_verdict.get("graph_signature")
            != live_result.get("graph_signature")
            or gate_verdict.get("candidate")
            != _FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID
            or gate_verdict.get("reference_bv") != 8
            or gate_verdict.get("candidate_bv") != 8
            or gate_verdict.get("reference_kernel_structure")
            != _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL
            or gate_verdict.get("candidate_kernel_structure")
            != _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL
            or gate_verdict.get("reference_physical_launches_per_layer") != 8
            or gate_verdict.get("candidate_physical_launches_per_layer") != 2
            or gate_verdict.get("count_invocation") is not True
            or gate_verdict.get("ring_export") is not True
            or gate_verdict.get("flags_inkernel") is not True
            or gate_verdict.get("scan_align") is not False
            or gate_verdict.get("npad_invariant") is not False
            or gate_verdict.get("tree_gdn_geom_override") != "BV=8"
            or gate_verdict.get("enforce_eager") != 0
            or gate_verdict.get("cudagraph_mode")
            != "FULL_AND_PIECEWISE"
            or gate_verdict.get("production_eligible") is not True
            or gate_verdict.get("b4_layer_passes") != 48
            or gate_verdict.get("observed_pass_layers_by_batch")
            != {"2": 0, "3": 0, "4": 48}
            or not isinstance(ledger_sha256, str)
            or len(ledger_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in ledger_sha256
            )
            or gate_verdict.get("raw_byte_equal") is not True
            or gate_verdict.get("reference_always_served") is not True
            or gate_verdict.get("production_default_enabled") is not False
        )
        if credential_invalid:
            raise RuntimeError(
                "FR13 fixed32 batched BV8 production credential is invalid"
            )
        payload = dict(live_result)
        payload["_production_sidecar_sha256"] = hashlib.sha256(
            Path(pass_path).read_bytes()
        ).hexdigest()
        payload["_graph_pass_sha256"] = credential["live_result_sha256"]
        payload["_gate_verdict_sha256"] = credential["gate_verdict_sha256"]
        payload["_runtime_manifest_sha256"] = runtime_manifest_sha256
        payload["_gate_runner_sha256"] = gate_runner_sha256
    layer_keys = payload.get("layer_keys")
    task_marker = payload.get("task_marker")
    valid_layer_keys = (
        isinstance(layer_keys, list)
        and len(layer_keys) == 48
        and all(
            isinstance(key, str) and key.startswith("0x")
            for key in layer_keys
        )
        and len(set(layer_keys)) == 48
    )
    common_invalid = (
        payload.get("status") != "pass"
        or payload.get("reference_always_served") is not True
        or not isinstance(task_marker, str)
        or not task_marker.startswith("swe_verified:")
        or len(task_marker) == len("swe_verified:")
        or type(payload.get("batch")) is not int
        or not 2 <= payload["batch"] <= _FR13_FIXED32_MAX_BATCH
        or payload.get("layer_count") != 48
        or not valid_layer_keys
    )
    if production_bv is None:
        schema_invalid = (
            payload.get("schema") != "fr13.fixed32.batch_gdn.live_pass.v1"
        )
        wide_invalid = False
    else:
        graph_signature = payload.get("graph_signature")
        schema_invalid = (
            payload.get("schema")
            != "fr13.fixed32.batch_gdn.graph_live_pass.v1"
            or payload.get("gate_mode") != "post_replay_shadow"
            or type(payload.get("graph_id")) is not int
            or payload["graph_id"] <= 0
            or not isinstance(graph_signature, str)
            or len(graph_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in graph_signature
            )
            or payload.get("capture_records") != 48
            or payload.get("real_task_authenticated") is not True
            or payload.get("graph_baseline_byte_equal") is not True
        )
        candidate_id = (
            _FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID
            if production_bv == 8
            else _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_ID
        )
        wide_invalid = (
            production_bv not in (8, 64)
            or payload.get("batch") != 4
            or payload.get("candidate") != candidate_id
            or payload.get("source_sha256")
            != _fr13_fixed32_batch_gdn_source_sha256()
            or (
                _FR13_FIXED32_MODE
                not in (
                    ("tail6_fixed32", "hydra27_fixed32")
                    if production_bv == 8
                    else ("tail6_fixed32",)
                )
            )
            or payload.get("mode") != "tail6_fixed32"
            or payload.get("physical_rows_per_request") != 32
            or payload.get("reference_bv") != 8
            or payload.get("candidate_bv") != production_bv
            or payload.get("reference_physical_launches_per_layer")
            != 8
            or payload.get("candidate_physical_launches_per_layer") != 2
            or payload.get("compared_byte_surfaces")
            != list(_FR13_FIXED32_BATCH_GDN_BV_BYTE_SURFACES)
            or payload.get("raw_byte_equal") is not True
            or payload.get("state_restored") is not True
        )
        if production_bv == 8:
            wide_invalid = wide_invalid or (
                payload.get("reference_kernel_structure")
                != _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL
                or payload.get("candidate_kernel_structure")
                != _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL
                or payload.get("count_invocation") is not True
                or payload.get("ring_export") is not True
                or payload.get("flags_inkernel") is not True
                or payload.get("scan_align") is not False
                or payload.get("npad_invariant") is not False
                or payload.get("production_eligible") is not True
            )
    if common_invalid or schema_invalid or wide_invalid:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_PRODUCTION live-gate PASS record is invalid"
        )
    return payload


def fixed32_batch_gdn_selector(batch_size: int) -> str | None:
    """Resolve diagnostics or qualified batched production; default is legacy."""
    batch = int(batch_size)
    if batch <= 1:
        # B1 already has the target two launches. Keep its byte-established
        # per-request BV8 route and batch only the B2/B3 launch-count gap.
        return None
    if batch > _FR13_FIXED32_MAX_BATCH:
        raise RuntimeError(
            f"FR13 fixed32 batched GDN supports B2-B4, got B={batch}"
        )
    if (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
    ):
        # B1 is captured by launch_tree_gdn_prepared. B4 uses this folded
        # stock-serving capture route; B2/B3 remain outside qualification.
        return "single_launch_gate" if batch == 4 else None
    if _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None:
        for env_name in (
            "FR13_FIXED32_BATCH_GDN_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_PRODUCTION",
            "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
        ):
            if os.environ.get(env_name, "").strip() not in ("", "0"):
                raise RuntimeError(
                    f"{env_name} cannot authorize GDN GQA-group3 production"
                )
        diagnostic, _marker = _fr13_fixed32_batch_gdn_byte_ab_control()
        graph_diagnostic = _fr13_fixed32_batch_gdn_graph_byte_ab_control()
        production = _fr13_fixed32_batch_gdn_production_control()
        if diagnostic or graph_diagnostic or production is not None:
            raise RuntimeError(
                "FR13 GDN GQA-group3 production cannot inherit a batched "
                "GDN diagnostic or production credential"
            )
        return (
            "gqa_group3"
            if batch == 4
            and _fr13_fixed32_gdn_gqa_group3_production_for_batch(4)
            else None
        )
    if (
        _FR13_FIXED32_GDN_SINGLE_LAUNCH
        or _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None
    ):
        for env_name in (
            "FR13_FIXED32_BATCH_GDN_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_PRODUCTION",
            "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
        ):
            if os.environ.get(env_name, "").strip() not in ("", "0"):
                raise RuntimeError(
                    f"{env_name} cannot authorize the fixed32 GDN "
                    "single-launch candidate"
                )
        diagnostic, _marker = _fr13_fixed32_batch_gdn_byte_ab_control()
        graph_diagnostic = _fr13_fixed32_batch_gdn_graph_byte_ab_control()
        production = _fr13_fixed32_batch_gdn_production_control()
        if diagnostic or graph_diagnostic or production is not None:
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch cannot inherit a batched GDN "
                "diagnostic or production credential"
            )
        # Only the requested B4 folded launch is exposed. B2/B3 retain their
        # established per-request dispatch and therefore exercise the B1 arm.
        #
        # Both arms below reach the IDENTICAL folded kernel, grid and telemetry;
        # only the authority to reach it differs. The diagnostic bool is
        # source-and-env only, so it stays as unconditional at B4 as it was
        # before this arm existed -- it is the campaign instrument the live gate
        # observes, and narrowing it would invalidate the gate it feeds. The
        # production arm is credential-bound and therefore narrows to the ONE
        # batch its live gate actually proved, so a B1 credential can never
        # authorize B4.
        return (
            "single_launch"
            if batch == 4
            and (
                _FR13_FIXED32_GDN_SINGLE_LAUNCH
                or _fr13_fixed32_gdn_single_launch_production_for_batch(4)
            )
            else None
        )
    diagnostic, _marker = _fr13_fixed32_batch_gdn_byte_ab_control()
    graph_diagnostic = _fr13_fixed32_batch_gdn_graph_byte_ab_control()
    production = _fr13_fixed32_batch_gdn_production_control()
    if diagnostic and graph_diagnostic:
        raise RuntimeError(
            "FR13 fixed32 eager and graph-replay batched GDN diagnostics are "
            "mutually exclusive"
        )
    if (diagnostic or graph_diagnostic) and production is not None:
        raise RuntimeError(
            "FR13 fixed32 batched GDN diagnostic and production selectors "
            "are mutually exclusive"
        )
    if (
        _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is not None
        and not (diagnostic or graph_diagnostic)
    ):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE requires the batched GDN "
            "diagnostic selector"
        )
    if _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE == 8 and not graph_diagnostic:
        raise RuntimeError(
            "FR13 fixed32 batched BV8 structure candidate requires the exact-B4 "
            "graph diagnostic"
        )
    if (
        _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None
        and production is None
    ):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION requires the batched GDN "
            "production selector and its live PASS"
        )
    if (
        diagnostic or graph_diagnostic
    ) and _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None:
        raise RuntimeError(
            "FR13 fixed32 batched GDN diagnostic and production selectors "
            "are mutually exclusive"
        )
    if production is not None and _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is not None:
        raise RuntimeError(
            "FR13 fixed32 batched GDN diagnostic and production selectors "
            "are mutually exclusive"
        )
    if (
        (diagnostic or graph_diagnostic or production is not None)
        and (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None
            or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 batched GDN and path-BV selectors are "
            "mutually exclusive"
        )
    if diagnostic:
        return "diagnostic"
    if graph_diagnostic:
        context = _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
        if (
            batch == 4
            and isinstance(context, dict)
            and int(context.get("batch_size", -1)) == 4
        ):
            return "graph_capture"
        # B1-B3 and non-capture execution remain the established per-request
        # BV8 path. Real serving replays the graph and never re-enters Python.
        return None
    if production is not None:
        production_bv = _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION
        qualified_batch = int(production["batch"])
        if production_bv == 8:
            if qualified_batch != 4:
                raise RuntimeError(
                    "FR13 fixed32 batched BV8 production requires its exact-B4 "
                    f"live-gate credential, got B={qualified_batch}"
                )
            # Request identity only expands the same two level grids for B2-B4.
            # B1 remains on its already-two-launch legacy route.
            return "production"
        if production_bv is not None and batch < 4:
            # Preserve the pre-existing generic/BV64 production scope.
            return None
        if qualified_batch != batch:
            raise RuntimeError(
                "FR13_FIXED32_BATCH_GDN_PRODUCTION batch does not match its "
                f"live-gate PASS record: {batch} != {qualified_batch}"
            )
        return "production"
    return None


def fixed32_batch_gdn_bv64_production_capture_begin(
    graph_id: int, batch_size: int
) -> None:
    """Open one FULL capture record for source-qualified BV64 production."""
    global _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is None:
        return
    payload = _fr13_fixed32_batch_gdn_production_control()
    identity = int(graph_id)
    batch = int(batch_size)
    pass_path = Path(
        os.environ.get(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH",
            _FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS,
        )
    )
    try:
        pass_sha256 = hashlib.sha256(pass_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            "FR13 fixed32 BV64 production cannot hash its installed PASS"
        ) from error
    if (
        payload is None
        or _FR13_FIXED32_MODE != "tail6_fixed32"
        or _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION != 64
        or identity <= 0
        or batch not in (1, 2, 3, 4)
        or _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT is not None
        or batch in _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURES
    ):
        raise RuntimeError(
            "FR13 fixed32 BV64 production capture begin drift: "
            + repr((identity, batch, _FR13_FIXED32_MODE))
        )
    prior_passes = {
        record["graph_pass_sha256"]
        for record in _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURES.values()
    }
    if prior_passes and prior_passes != {pass_sha256}:
        raise RuntimeError("FR13 fixed32 BV64 production PASS changed during capture")
    _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "batch_size": batch,
        "graph_pass_sha256": pass_sha256,
        "kernel_source_sha256": payload["source_sha256"],
        "layer_keys": [],
        "candidate_bvs": [],
    }


def _fr13_fixed32_batch_gdn_bv64_production_capture_register(
    *, batch_size: int, layer_key: int, candidate_bv: int
) -> None:
    """Record each actual wide production launch while its B4 graph captures."""
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is None:
        return
    context = _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT
    batch = int(batch_size)
    key = int(layer_key)
    selected_bv = int(candidate_bv)
    if (
        not isinstance(context, dict)
        or set(context)
        != {
            "graph_id",
            "batch_size",
            "graph_pass_sha256",
            "kernel_source_sha256",
            "layer_keys",
            "candidate_bvs",
        }
        or batch != 4
        or int(context.get("batch_size", -1)) != 4
        or key <= 0
        or selected_bv != 64
    ):
        raise RuntimeError("FR13 fixed32 BV64 production launch scope drifted")
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 fixed32 BV64 production launch was not captured")
    context["layer_keys"].append(key)
    context["candidate_bvs"].append(selected_bv)


def fixed32_batch_gdn_bv64_production_capture_end(
    graph_id: int,
    batch_size: int,
    graph_signature: str,
    expected_scan_calls: int,
) -> None:
    """Bind the actual B4 BV64 launches, and B1-B3 absence, to each graph."""
    global _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is None:
        return
    context = _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT
    identity = int(graph_id)
    batch = int(batch_size)
    signature = str(graph_signature)
    layer_keys = context.get("layer_keys") if isinstance(context, dict) else None
    candidate_bvs = (
        context.get("candidate_bvs") if isinstance(context, dict) else None
    )
    expected_layers = 48 if batch == 4 else 0
    if (
        not isinstance(context, dict)
        or int(context.get("graph_id", -1)) != identity
        or int(context.get("batch_size", -1)) != batch
        or batch not in (1, 2, 3, 4)
        or int(expected_scan_calls) != 48 * batch
        or not isinstance(layer_keys, list)
        or len(layer_keys) != expected_layers
        or len(set(layer_keys)) != expected_layers
        or not isinstance(candidate_bvs, list)
        or candidate_bvs != ([64] * expected_layers)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise RuntimeError(
            "FR13 fixed32 BV64 production capture end drift: "
            + repr((identity, batch, expected_layers, layer_keys, candidate_bvs))
        )
    _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURES[batch] = {
        "graph_id": identity,
        "graph_signature": signature,
        "graph_pass_sha256": context["graph_pass_sha256"],
        "kernel_source_sha256": context["kernel_source_sha256"],
        "layer_keys": tuple(sorted(layer_keys)),
    }
    _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURE_CONTEXT = None


def fixed32_batch_gdn_bv64_production_replay_engaged(
    graph_id: int,
    batch_size: int,
    graph_signature: str,
    expected_scan_calls: int,
) -> dict[str, object]:
    """Publish once only after a measured replay of the captured B4 BV64 graph."""
    global _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_PUBLISHED
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is None:
        return {"status": "disabled"}
    identity = int(graph_id)
    batch = int(batch_size)
    signature = str(graph_signature)
    captures = _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_CAPTURES
    expected_layer_counts = {1: 0, 2: 0, 3: 0, 4: 48}
    actual_layer_counts = {
        captured_batch: len(record.get("layer_keys", ()))
        for captured_batch, record in captures.items()
    }
    record = captures.get(batch)
    if (
        set(captures) != {1, 2, 3, 4}
        or actual_layer_counts != expected_layer_counts
        or not isinstance(record, dict)
        or record.get("graph_id") != identity
        or record.get("graph_signature") != signature
        or int(expected_scan_calls) != 48 * batch
    ):
        raise RuntimeError(
            "FR13 fixed32 BV64 production replay provenance drift: "
            + repr((identity, batch, actual_layer_counts))
        )
    if batch != 4:
        return {
            "status": "legacy_lower_batch",
            "batch_size": batch,
            "wide_route_capture_layers": 0,
        }
    if _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_PUBLISHED:
        return {
            "status": "ENGAGED",
            "batch_size": 4,
            "observed_full_graph_replays_at_least": 1,
        }
    layer_keys = [f"0x{key:x}" for key in record["layer_keys"]]
    engagement = {
        "schema": "fr13.fixed32.batch_gdn.bv64.production_engagement.v1",
        "status": "ENGAGED",
        "mode": "tail6_fixed32",
        "runtime_mode": "FULL",
        "selector": "production",
        "batch_size": 4,
        "candidate": _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_ID,
        "candidate_bv": 64,
        "physical_rows_per_request": 32,
        "physical_launches_per_layer": 2,
        "layer_count": 48,
        "layer_keys": layer_keys,
        "wide_route_capture_layers_by_batch": {
            str(key): value for key, value in expected_layer_counts.items()
        },
        "graph_id": identity,
        "graph_signature": signature,
        "graph_pass_sha256": record["graph_pass_sha256"],
        "kernel_source_sha256": record["kernel_source_sha256"],
        "observed_full_graph_replays_at_least": 1,
        "fallback": 0,
        "production_default_enabled": False,
    }
    path = Path(_FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_ENGAGEMENT)
    if path.exists() or path.is_symlink():
        raise RuntimeError(
            "FR13 fixed32 BV64 production engagement path already exists"
        )
    raw = (
        json.dumps(
            engagement,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short BV64 engagement write")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    _FR13_FIXED32_BATCH_GDN_BV64_PRODUCTION_PUBLISHED = True
    return dict(engagement)


def fixed32_batch_gdn_bv8_production_capture_begin(
    graph_id: int, batch_size: int
) -> None:
    """Open one FULL capture record for source-qualified batched BV8."""
    global _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION != 8:
        return
    payload = _fr13_fixed32_batch_gdn_production_control()
    identity = int(graph_id)
    batch = int(batch_size)
    if (
        payload is None
        or _FR13_FIXED32_MODE not in ("tail6_fixed32", "hydra27_fixed32")
        or identity <= 0
        or batch not in (1, 2, 3, 4)
        or _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT is not None
        or batch in _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES
    ):
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production capture begin drift: "
            + repr((identity, batch, _FR13_FIXED32_MODE))
        )
    prior_credentials = {
        record["production_sidecar_sha256"]
        for record in _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES.values()
    }
    sidecar_sha256 = payload.get("_production_sidecar_sha256")
    if (
        not isinstance(sidecar_sha256, str)
        or len(sidecar_sha256) != 64
        or (prior_credentials and prior_credentials != {sidecar_sha256})
    ):
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production credential changed during capture"
        )
    _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "batch_size": batch,
        "production_sidecar_sha256": sidecar_sha256,
        "graph_pass_sha256": payload["_graph_pass_sha256"],
        "gate_verdict_sha256": payload["_gate_verdict_sha256"],
        "runtime_manifest_sha256": payload["_runtime_manifest_sha256"],
        "gate_runner_sha256": payload["_gate_runner_sha256"],
        "kernel_source_sha256": payload["source_sha256"],
        "runtime_mode": _FR13_FIXED32_MODE,
        "task_marker": payload["task_marker"],
        "layer_keys": [],
        "candidate_bvs": [],
    }


def _fr13_fixed32_batch_gdn_bv8_production_capture_register(
    *, batch_size: int, layer_key: int, candidate_bv: int
) -> None:
    """Record each actual two-launch BV8 dispatch in a B2-B4 graph."""
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION != 8:
        return
    context = _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT
    batch = int(batch_size)
    key = int(layer_key)
    selected_bv = int(candidate_bv)
    if (
        not isinstance(context, dict)
        or set(context)
        != {
            "graph_id",
            "batch_size",
            "production_sidecar_sha256",
            "graph_pass_sha256",
            "gate_verdict_sha256",
            "runtime_manifest_sha256",
            "gate_runner_sha256",
            "kernel_source_sha256",
            "runtime_mode",
            "task_marker",
            "layer_keys",
            "candidate_bvs",
        }
        or batch not in (2, 3, 4)
        or int(context.get("batch_size", -1)) != batch
        or key <= 0
        or selected_bv != 8
    ):
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production launch scope drifted"
        )
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production launch was not captured"
        )
    context["layer_keys"].append(key)
    context["candidate_bvs"].append(selected_bv)


def fixed32_batch_gdn_bv8_production_capture_end(
    graph_id: int,
    batch_size: int,
    graph_signature: str,
    expected_scan_calls: int,
) -> None:
    """Bind B2-B4 batched launches and B1 legacy absence to each graph."""
    global _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION != 8:
        return
    context = _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT
    identity = int(graph_id)
    batch = int(batch_size)
    signature = str(graph_signature)
    layer_keys = context.get("layer_keys") if isinstance(context, dict) else None
    candidate_bvs = (
        context.get("candidate_bvs") if isinstance(context, dict) else None
    )
    expected_layers = 0 if batch == 1 else 48
    if (
        not isinstance(context, dict)
        or int(context.get("graph_id", -1)) != identity
        or int(context.get("batch_size", -1)) != batch
        or batch not in (1, 2, 3, 4)
        or int(expected_scan_calls) != 48 * batch
        or not isinstance(layer_keys, list)
        or len(layer_keys) != expected_layers
        or len(set(layer_keys)) != expected_layers
        or not isinstance(candidate_bvs, list)
        or candidate_bvs != ([8] * expected_layers)
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production capture end drift: "
            + repr((identity, batch, expected_layers, layer_keys, candidate_bvs))
        )
    _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES[batch] = {
        "graph_id": identity,
        "graph_signature": signature,
        "production_sidecar_sha256": context["production_sidecar_sha256"],
        "graph_pass_sha256": context["graph_pass_sha256"],
        "gate_verdict_sha256": context["gate_verdict_sha256"],
        "runtime_manifest_sha256": context["runtime_manifest_sha256"],
        "gate_runner_sha256": context["gate_runner_sha256"],
        "kernel_source_sha256": context["kernel_source_sha256"],
        "runtime_mode": context["runtime_mode"],
        "task_marker": context["task_marker"],
        "layer_keys": tuple(sorted(layer_keys)),
    }
    _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT = None


def fixed32_batch_gdn_bv8_production_replay_engaged(
    graph_id: int,
    batch_size: int,
    graph_signature: str,
    expected_scan_calls: int,
) -> dict[str, object]:
    """Publish after B4 replay once every B1-B4 graph route is captured."""
    global _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED
    if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION != 8:
        return {"status": "disabled"}
    identity = int(graph_id)
    batch = int(batch_size)
    signature = str(graph_signature)
    captures = _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES
    expected_layer_counts = {1: 0, 2: 48, 3: 48, 4: 48}
    actual_layer_counts = {
        captured_batch: len(record.get("layer_keys", ()))
        for captured_batch, record in captures.items()
    }
    record = captures.get(batch)
    credential_hashes = {
        captured.get("production_sidecar_sha256")
        for captured in captures.values()
    }
    if (
        set(captures) != {1, 2, 3, 4}
        or actual_layer_counts != expected_layer_counts
        or len(credential_hashes) != 1
        or not isinstance(record, dict)
        or record.get("graph_id") != identity
        or record.get("graph_signature") != signature
        or int(expected_scan_calls) != 48 * batch
    ):
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production replay provenance drift: "
            + repr((identity, batch, actual_layer_counts, credential_hashes))
        )
    if batch != 4:
        return {
            "status": (
                "legacy_lower_batch" if batch == 1 else "batched_lower_batch"
            ),
            "batch_size": batch,
            "batched_route_capture_layers": expected_layer_counts[batch],
        }
    if _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED:
        return {
            "status": "ENGAGED",
            "batch_size": 4,
            "observed_full_graph_replays_at_least": 1,
        }
    layer_keys = [f"0x{key:x}" for key in record["layer_keys"]]
    engagement = {
        "schema": "fr13.fixed32.batch_gdn.bv8.production_engagement.v1",
        "status": "ENGAGED",
        "mode": record["runtime_mode"],
        "runtime_mode": "FULL",
        "selector": "production",
        "batch_size": 4,
        "candidate": _FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID,
        "reference_kernel_structure": _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL,
        "candidate_kernel_structure": _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
        "physical_rows_per_request": 32,
        "layer_count": 48,
        "layer_keys": layer_keys,
        "batched_route_capture_layers_by_batch": {
            str(key): value for key, value in expected_layer_counts.items()
        },
        "qualified_batch_sizes": [4],
        "lower_batch_route": "b1_legacy_b2_b3_fixed32_batched_bv8",
        "physical_launches_per_layer_by_batch": {
            "1": 2,
            "2": 2,
            "3": 2,
            "4": 2,
        },
        "all_b_le_4_launch_invariant": True,
        "graph_id": identity,
        "graph_signature": signature,
        "graph_pass_sha256": record["graph_pass_sha256"],
        "gate_verdict_sha256": record["gate_verdict_sha256"],
        "runtime_manifest_sha256": record["runtime_manifest_sha256"],
        "gate_runner_sha256": record["gate_runner_sha256"],
        "production_sidecar_sha256": record["production_sidecar_sha256"],
        "kernel_source_sha256": record["kernel_source_sha256"],
        "task_marker": record["task_marker"],
        "observed_full_graph_replays_at_least": 1,
        "fallback": 0,
        "production_default_enabled": False,
    }
    path = Path(_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_ENGAGEMENT)
    if path.exists() or path.is_symlink():
        raise RuntimeError(
            "FR13 fixed32 batched BV8 production engagement path already exists"
        )
    raw = (
        json.dumps(
            engagement,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short batched BV8 engagement write")
            offset += written
        # The worker runs as container root while the timing reducer runs as
        # the host operator through the bind mount. This record contains no
        # credentials or request data and must remain host-readable.
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    _FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED = True
    return dict(engagement)


def fixed32_batch_gdn_graph_live_capture_active(batch_size: int) -> bool:
    """Whether the final exact-B4 FULL capture is collecting layer records."""
    context = _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
    return bool(
        _fr13_fixed32_batch_gdn_graph_byte_ab_control()
        and int(batch_size) == 4
        and isinstance(context, dict)
        and int(context.get("batch_size", -1)) == 4
    )


def fixed32_batch_gdn_graph_live_capture_begin(
    graph_id: int, batch_size: int
) -> None:
    """Open the graph-shadow byte gate only for the final exact-B4 graph."""
    global _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
    if not _fr13_fixed32_batch_gdn_graph_byte_ab_control():
        return
    identity = int(graph_id)
    batch = int(batch_size)
    if batch != 4:
        return
    if (
        _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES
        or _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE is None
        or identity <= 0
        or _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT is not None
        or identity in _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES
    ):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte-gate capture begin drift: "
            + repr((identity, batch, _FR13_FIXED32_MODE))
        )
    _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT = {
        "graph_id": identity,
        "batch_size": batch,
        "records": [],
    }
    _FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE.update(
        status="armed",
        candidate_bv=int(_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE),
        graph_id=None,
        graph_signature=None,
        batch_size=None,
        records=0,
    )


def _fr13_fixed32_batch_gdn_graph_live_capture_register(record: dict) -> None:
    """Retain persistent B4 operands while the legacy BV8 graph is captured."""
    if not _fr13_fixed32_batch_gdn_graph_byte_ab_control():
        return
    context = _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
    required = {
        "layer_key",
        "snapshot",
        "restore",
        "run_reference",
        "run_candidate",
        "carrier_nonzero",
        "byte_equal",
        "surface_names",
    }
    if (
        not isinstance(context, dict)
        or set(context) != {"graph_id", "batch_size", "records"}
        or int(context.get("batch_size", -1)) != 4
        or not isinstance(record, dict)
        or set(record) != required
        or type(record.get("layer_key")) is not int
        or int(record["layer_key"]) <= 0
        or tuple(record.get("surface_names", ()))
        != _FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES
        or not all(
            callable(record[name])
            for name in required - {"layer_key", "surface_names"}
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte-gate capture record drift"
        )
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte-gate record was not captured by CUDA"
        )
    context["records"].append(record)


def fixed32_batch_gdn_graph_live_capture_end(
    graph_id: int,
    batch_size: int,
    graph_signature: str,
    expected_records: int = 48,
) -> None:
    """Freeze 48 unique layer records against one signed exact-B4 graph."""
    global _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
    if not _fr13_fixed32_batch_gdn_graph_byte_ab_control():
        return
    identity = int(graph_id)
    batch = int(batch_size)
    if batch != 4:
        return
    context = _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT
    records = context.get("records") if isinstance(context, dict) else None
    layer_keys = (
        [int(record.get("layer_key", -1)) for record in records]
        if isinstance(records, list)
        else []
    )
    signature = str(graph_signature)
    if (
        not isinstance(context, dict)
        or int(context.get("graph_id", -1)) != identity
        or int(context.get("batch_size", -1)) != 4
        or int(expected_records) != 48
        or not isinstance(records, list)
        or len(records) != 48
        or len(set(layer_keys)) != 48
        or len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte-gate capture end drift: "
            + repr(
                (
                    identity,
                    batch,
                    expected_records,
                    len(records) if isinstance(records, list) else None,
                    len(set(layer_keys)),
                    signature,
                )
            )
        )
    _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES[identity] = {
        "batch_size": 4,
        "graph_signature": signature,
        "records": tuple(records),
        "layer_keys": frozenset(layer_keys),
    }
    _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT = None


def _fr13_fixed32_batch_gdn_byte_diff(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    """Return strict byte-diff evidence including the first differing byte."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "name": name,
            "byte_equal": False,
            "shape_or_dtype_mismatch": True,
            "reference_shape": tuple(int(value) for value in reference.shape),
            "candidate_shape": tuple(int(value) for value in candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "differing_bytes": None,
            "first_nonzero_byte": None,
        }
    reference_bytes = reference.contiguous().reshape(-1).view(torch.uint8)
    candidate_bytes = candidate.contiguous().reshape(-1).view(torch.uint8)
    difference = reference_bytes != candidate_bytes
    differing_bytes = int(difference.sum().item())
    if differing_bytes == 0:
        first_nonzero = None
        reference_byte = None
        candidate_byte = None
    else:
        first_nonzero = int(
            torch.nonzero(difference, as_tuple=False)[0, 0].item()
        )
        reference_byte = int(reference_bytes[first_nonzero].item())
        candidate_byte = int(candidate_bytes[first_nonzero].item())
    return {
        "name": name,
        "byte_equal": differing_bytes == 0,
        "shape_or_dtype_mismatch": False,
        "shape": tuple(int(value) for value in reference.shape),
        "dtype": str(reference.dtype),
        "bytes": int(reference_bytes.numel()),
        "differing_bytes": differing_bytes,
        "first_nonzero_byte": first_nonzero,
        "reference_byte": reference_byte,
        "candidate_byte": candidate_byte,
    }


def _fr13_fixed32_sfwd_state_fusion_emit(record: dict[str, object]) -> None:
    path = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_PATH",
        "/logs/fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl",
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(record)
    payload["schema"] = "fr13.fixed32.sfwd_state_fusion.byte_ab.v1"
    with open(path, "a", encoding="ascii") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _fr13_fixed32_sfwd_state_fusion_pass_emit(
    *, task_markers: tuple[str, ...], batch: int, layer_keys: set[int]
) -> None:
    if (
        len(layer_keys) != 48
        or int(batch) != 4
        or task_markers
        != _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS
    ):
        return
    path = os.environ.get(
        "FR13_FIXED32_SFWD_STATE_FUSION_PASS_PATH",
        _FR13_FIXED32_SFWD_STATE_FUSION_PASS,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "schema": "fr13.fixed32.sfwd_state_fusion.b4_source_pass.v1",
        "status": "byte_pass_source_only",
        "run_classification": (
            "real_swe_verified_exact4_b4_byte_diagnostic"
        ),
        "candidate": _FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID,
        "source_sha256": _fr13_fixed32_batch_gdn_source_sha256(),
        "task_set": "canonical real SWE-Verified exact4 B4",
        "task_count": 4,
        "task_ids": list(
            _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_IDS
        ),
        "task_markers": list(task_markers),
        "subset_sha256": (
            _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_SUBSET_SHA256
        ),
        "real_task_authenticated": True,
        "batch": int(batch),
        "batch_size": 4,
        "concurrency": 4,
        "layer_count": 48,
        "layer_keys": [f"0x{key:x}" for key in sorted(layer_keys)],
        "physical_rows_per_request": 32,
        "physical_rows_total": 128,
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [int(batch), 11 * int(batch)],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "comparison_records": 48,
        "candidate_shadow_only": True,
        "served_result": "reference",
        "reference_always_served": True,
        "probe_inputs": False,
        "synthetic_inputs": False,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "production_eligible": False,
        "production_blocker": (
            "requires separately authenticated B1 and exact4 B4 byte PASS "
            "artifacts bound to the same candidate kernel"
        ),
    }
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fixed32_sfwd_state_fusion_byte_gate(
    *,
    task_markers: tuple[str, ...],
    layer_key: int,
    batch_size: int,
    reference_out: torch.Tensor,
    candidate_out: torch.Tensor,
    reference_source_stage: torch.Tensor,
    candidate_source_stage: torch.Tensor,
) -> dict[str, object]:
    """Compare candidate bytes on one authenticated real layer/event.

    The caller has already computed both arms from the same inputs. This hook
    only compares and records; it never replaces the reference tensors. A
    mismatch therefore fails closed while incumbent bytes remain served.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION byte gate is eager-only"
        )
    markers = tuple(task_markers) if isinstance(task_markers, tuple) else ()
    if markers != _FR13_FIXED32_SFWD_STATE_FUSION_EXACT4_TASK_MARKERS:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires an authenticated "
            "canonical exact4 SWE-Verified marker set"
        )
    batch = int(batch_size)
    if batch != 4:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION byte gate requires exact B4"
        )
    fixed32_sfwd_state_fusion_contract(
        batch,
        tree_rows=32,
        conv_width=4,
        conv_state_len=_FR13_FIXED32_SFWD_CONV_STATE_LEN,
    )
    state = _FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_STATE
    if state["task_markers"] is None:
        state["task_markers"] = markers
        state["batch"] = batch
    elif tuple(state["task_markers"]) != markers or int(state["batch"]) != batch:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION marker set or batch changed"
        )
    if bool(state.get("failed")):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION byte gate previously mismatched"
        )
    key = int(layer_key)
    if key in state["passed"]:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION repeated a qualified layer"
        )
    attempt = int(state["attempts"].get(key, 0)) + 1
    state["attempts"][key] = attempt
    comparisons = [
        _fr13_fixed32_batch_gdn_byte_diff(
            "conv_out", reference_out, candidate_out
        ),
        _fr13_fixed32_batch_gdn_byte_diff(
            "commit_source_stage",
            reference_source_stage,
            candidate_source_stage,
        ),
    ]
    first_nonzero = next(
        (item for item in comparisons if not bool(item["byte_equal"])), None
    )
    passed = first_nonzero is None
    record = {
        "status": "pass" if passed else "mismatch_reference_served",
        "candidate": _FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID,
        "task_markers": list(markers),
        "real_task_authenticated": True,
        "batch": batch,
        "layer_key": f"0x{key:x}",
        "attempt": attempt,
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [batch, 11 * batch],
        "gdn_physical_launches_per_layer": 2,
        "comparisons": comparisons,
        "first_nonzero": first_nonzero,
        "zero_diff": passed,
        "candidate_shadow_only": True,
        "served_result": "reference",
        "reference_always_served": True,
        "acceptance_valid": False,
        "timing_eligible": False,
        "production_eligible": False,
    }
    _fr13_fixed32_sfwd_state_fusion_emit(record)
    if passed:
        state["passed"].add(key)
        _fr13_fixed32_sfwd_state_fusion_pass_emit(
            task_markers=markers,
            batch=batch,
            layer_keys=set(state["passed"]),
        )
    else:
        state["passed"].discard(key)
        state["failed"] = True
    return record


def _fr13_fixed32_batch_gdn_byte_ab_emit(record: dict[str, object]) -> None:
    path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_PATH",
        "/logs/fr13_fixed32_batch_gdn_byte_ab.jsonl",
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(record)
    payload["schema"] = "fr13.fixed32.batch_gdn.byte_ab.v1"
    with open(path, "a", encoding="ascii") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _fr13_fixed32_batch_gdn_live_pass_emit(
    *,
    task_marker: str,
    batch: int,
    layer_keys: set[int],
    reference_bv: int | None = None,
    candidate_bv: int | None = None,
    graph_id: int | None = None,
    graph_signature: str | None = None,
    capture_records: int | None = None,
) -> None:
    """Publish the non-cryptographic production prerequisite after 48 passes."""
    if len(layer_keys) != 48:
        return
    path = os.environ.get(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH",
        _FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    structured_candidate = reference_bv is not None and candidate_bv is not None
    wide_bv = (
        structured_candidate
        and int(reference_bv) != int(candidate_bv)
    )
    # The combined wide-BV gate is formally authorized only by the exact4 B4
    # lifecycle. B2/B3 are still byte-compared and logged, but cannot publish
    # the production prerequisite.
    if wide_bv and int(batch) != 4:
        return
    graph_gate = graph_id is not None
    if graph_gate and (
        int(batch) != 4
        or type(graph_id) is not int
        or graph_id <= 0
        or not isinstance(graph_signature, str)
        or len(graph_signature) != 64
        or any(
            character not in "0123456789abcdef"
            for character in graph_signature
        )
        or capture_records != 48
    ):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN PASS identity contract drift"
        )
    payload = {
        "schema": (
            "fr13.fixed32.batch_gdn.graph_live_pass.v1"
            if graph_gate
            else "fr13.fixed32.batch_gdn.live_pass.v2"
            if wide_bv
            else "fr13.fixed32.batch_gdn.live_pass.v1"
        ),
        "status": "pass",
        "task_marker": task_marker,
        "batch": int(batch),
        "layer_count": 48,
        "layer_keys": [f"0x{key:x}" for key in sorted(layer_keys)],
        "reference_always_served": True,
    }
    if structured_candidate:
        selected_candidate = int(candidate_bv)
        payload.update(
            candidate=(
                _FR13_FIXED32_BATCH_GDN_BV8_CANDIDATE_ID
                if selected_candidate == 8
                else _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE_ID
            ),
            source_sha256=_fr13_fixed32_batch_gdn_source_sha256(),
            mode=_FR13_FIXED32_MODE,
            physical_rows_per_request=32,
            reference_bv=int(reference_bv),
            candidate_bv=selected_candidate,
            reference_physical_launches_per_layer=2 * int(batch),
            candidate_physical_launches_per_layer=2,
            compared_byte_surfaces=list(
                _FR13_FIXED32_BATCH_GDN_BV_BYTE_SURFACES
            ),
            raw_byte_equal=True,
            state_restored=True,
        )
        if selected_candidate == 8:
            if graph_gate and (scan_align_on() or npad_invariant_on()):
                raise RuntimeError(
                    "FR13 fixed32 batched BV8 PASS specialization requires "
                    "scan/npad invariant controls off"
                )
            payload.update(
                reference_kernel_structure=_FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL,
                candidate_kernel_structure=_FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL,
                count_invocation=True,
                ring_export=True,
                flags_inkernel=True,
                scan_align=False,
                npad_invariant=False,
                production_eligible=True,
            )
    if graph_gate:
        payload.update(
            gate_mode="post_replay_shadow",
            graph_id=int(graph_id),
            graph_signature=graph_signature,
            capture_records=int(capture_records),
            real_task_authenticated=True,
            graph_baseline_byte_equal=True,
        )
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _fr13_fixed32_batch_gdn_graph_compare_records(
    records,
    *,
    candidate_bv: int,
    graph_id: int,
    graph_signature: str,
    task_marker: str,
) -> dict[str, object]:
    """Shadow the replayed per-request BV8 graph with one batched candidate."""
    candidate = int(candidate_bv)
    if candidate not in (8, 16, 32, 64, 128):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte gate rejected candidate BV="
            f"{candidate}"
        )
    checked = 0
    layer_keys: set[int] = set()
    for index, record in enumerate(records):
        layer_key = int(record["layer_key"])
        if layer_key in layer_keys:
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN byte gate reused layer key "
                f"0x{layer_key:x}"
            )
        layer_keys.add(layer_key)
        snapshot = record["snapshot"]
        restore = record["restore"]
        byte_equal = record["byte_equal"]
        baseline = snapshot()
        if tuple(baseline) != _FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES:
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN baseline surface drift at record "
                f"{index}: {tuple(baseline)!r}"
            )
        if not bool(record["carrier_nonzero"]()):
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN gate refused a zero real-task "
                f"carrier at record {index}"
            )
        caught = None
        graph_comparisons = []
        comparisons = []
        try:
            reference_meta = record["run_reference"]()
            reference = snapshot()
            graph_comparisons = [
                _fr13_fixed32_batch_gdn_byte_diff(
                    "graph_baseline_" + name,
                    baseline[name],
                    reference[name],
                )
                for name in _FR13_FIXED32_BATCH_GDN_GRAPH_STABLE_SURFACES
            ]
            restore(baseline)
            candidate_meta = record["run_candidate"](candidate)
            candidate_surfaces = snapshot()
            if (
                set(reference_meta)
                != {
                    "block_v",
                    "physical_launches",
                    "kernel_structure",
                    "compact_export",
                }
                or set(candidate_meta)
                != {
                    "block_v",
                    "physical_launches",
                    "kernel_structure",
                    "compact_export",
                }
                or int(reference_meta["block_v"]) != 8
                or int(reference_meta["physical_launches"]) != 8
                or reference_meta["kernel_structure"]
                != _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL
                or int(candidate_meta["block_v"]) != candidate
                or int(candidate_meta["physical_launches"]) != 2
                or candidate_meta["kernel_structure"]
                != _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL
                or reference_meta["kernel_structure"]
                == candidate_meta["kernel_structure"]
            ):
                raise RuntimeError(
                    "FR13 fixed32 B4 graph GDN launch metadata drift at "
                    f"record {index}: reference={reference_meta!r} "
                    f"candidate={candidate_meta!r}"
                )
            compact_rows = 4 * _FR13_FIXED32_EXPORT_SLOTS
            comparison_inputs = [
                ("out", reference["out"], candidate_surfaces["out"]),
                ("ring_k", reference["ring_k"], candidate_surfaces["ring_k"]),
                ("ring_v", reference["ring_v"], candidate_surfaces["ring_v"]),
                ("ring_a", reference["ring_a"], candidate_surfaces["ring_a"]),
                ("ring_b", reference["ring_b"], candidate_surfaces["ring_b"]),
                (
                    "state_export_compact",
                    reference_meta["compact_export"],
                    candidate_meta["compact_export"],
                ),
                (
                    "state_export_untouched_tail",
                    baseline["export"][compact_rows:],
                    candidate_surfaces["export"][compact_rows:],
                ),
                ("flags", reference["flags"], candidate_surfaces["flags"]),
                (
                    "invocation_counter",
                    reference["invocation_counter"],
                    candidate_surfaces["invocation_counter"],
                ),
            ]
            comparisons = [
                _fr13_fixed32_batch_gdn_byte_diff(name, left, right)
                for name, left, right in comparison_inputs
            ]
        except Exception as error:
            caught = error
        finally:
            restore(baseline)
            restored = snapshot()
            restore_bad = [
                name
                for name in _FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES
                if not byte_equal(restored[name], baseline[name])
            ]
            if restore_bad:
                raise RuntimeError(
                    "FR13 fixed32 B4 graph GDN byte gate failed to restore "
                    f"graph-served bytes at record {index}: {restore_bad}"
                )
        if caught is not None:
            raise caught
        graph_bad = [
            item["name"]
            for item in graph_comparisons
            if not bool(item["byte_equal"])
        ]
        arm_bad = [
            item["name"]
            for item in comparisons
            if not bool(item["byte_equal"])
        ]
        zero_diff = not graph_bad and not arm_bad
        _fr13_fixed32_batch_gdn_byte_ab_emit(
            {
                "gate_mode": "post_replay_shadow",
                "task_marker": task_marker,
                "graph_id": int(graph_id),
                "graph_signature": graph_signature,
                "layer_key": f"0x{layer_key:x}",
                "batch": 4,
                "attempt": 1,
                "physical_rows_per_request": 32,
                "reference_bv": 8,
                "candidate_bv": candidate,
                "reference_kernel_structure": (
                    _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL
                ),
                "candidate_kernel_structure": (
                    _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL
                ),
                "carrier_nonzero": True,
                "legacy_physical_launches": 8,
                "candidate_physical_launches": 2,
                "graph_comparisons": graph_comparisons,
                "comparisons": comparisons,
                "graph_baseline_byte_equal": not graph_bad,
                "zero_diff": zero_diff,
                "reference_restored_and_served": True,
                "status": "pass" if zero_diff else "mismatch_reference_served",
            }
        )
        if not zero_diff:
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN byte mismatch at record "
                f"{index}: graph={graph_bad} candidate={arm_bad}"
            )
        checked += 1
    return {
        "records": checked,
        "layer_keys": layer_keys,
        "reference_bv": 8,
        "candidate_bv": candidate,
        "reference_kernel_structure": _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL,
        "candidate_kernel_structure": _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL,
    }


def fixed32_batch_gdn_graph_live_gate_on_replay(
    graph_id: int,
    graph_signature: str,
    batch_size: int,
    expected_records: int = 48,
) -> dict[str, object]:
    """Run the graph-shadow gate once after an authenticated real B4 replay."""
    state = _FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE
    if not _fr13_fixed32_batch_gdn_graph_byte_ab_control():
        return dict(state)
    if int(batch_size) != 4:
        return dict(state)
    if state["status"] == "passed":
        return dict(state)
    if state["status"] != "armed":
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte gate is not runnable: " + repr(state)
        )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN byte gate cannot run during capture"
        )
    identity = int(graph_id)
    signature = str(graph_signature)
    capture = _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES.get(identity)
    records = capture.get("records") if isinstance(capture, dict) else None
    if (
        not isinstance(capture, dict)
        or int(capture.get("batch_size", -1)) != 4
        or capture.get("graph_signature") != signature
        or int(expected_records) != 48
        or not isinstance(records, tuple)
        or len(records) != 48
        or len(capture.get("layer_keys", ())) != 48
    ):
        raise RuntimeError(
            "FR13 fixed32 B4 graph GDN replay/capture drift: "
            + repr((identity, signature, expected_records, capture))
        )
    task_marker = _fr13_fixed32_batch_gdn_real_event_marker()
    state["status"] = "running"
    try:
        result = _fr13_fixed32_batch_gdn_graph_compare_records(
            records,
            candidate_bv=int(_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE),
            graph_id=identity,
            graph_signature=signature,
            task_marker=task_marker,
        )
    except Exception:
        state["status"] = "failed"
        raise
    try:
        _fr13_fixed32_batch_gdn_live_pass_emit(
            task_marker=task_marker,
            batch=4,
            layer_keys=result["layer_keys"],
            reference_bv=int(result["reference_bv"]),
            candidate_bv=int(result["candidate_bv"]),
            graph_id=identity,
            graph_signature=signature,
            capture_records=int(result["records"]),
        )
    except Exception:
        state["status"] = "failed"
        raise
    state.update(
        status="passed",
        graph_id=identity,
        graph_signature=signature,
        batch_size=4,
        records=int(result["records"]),
    )
    _FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES.clear()
    print(
        "[FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB PASS] "
        f"task={task_marker} graph_id={identity} batch=4 records=48 "
        f"reference_bv=8 candidate_bv={result['candidate_bv']} "
        "graph_baseline_equal=1 state_restored=1 served_bv=8",
        flush=True,
    )
    return dict(state)


def fixed32_batch_gdn_graph_live_gate_report() -> dict[str, object]:
    return dict(_FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE)


def _fr13_canonical_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fr13_tree_ancestry(parent) -> tuple[tuple[int, ...], ...]:
    parent_tuple = tuple(int(value) for value in parent)
    rows = []
    for node in range(len(parent_tuple)):
        visible = [0] * len(parent_tuple)
        cursor = node
        while cursor >= 0:
            visible[cursor] = 1
            cursor = parent_tuple[cursor]
        rows.append(tuple(visible))
    return tuple(rows)


def _fr13_fixed32_schedule_contract(levels) -> dict[str, object]:
    """Validate and summarize the exact fixed32 execution schedule."""
    normalized = tuple(
        tuple((tuple(int(node) for node in path), int(parent))
              for path, parent in level)
        for level in levels
    )
    coverage = tuple(sorted(
        node
        for level in normalized
        for path, _parent in level
        for node in path
    ))
    path_counts = tuple(len(level) for level in normalized)
    max_lengths = tuple(
        max(len(path) for path, _parent in level) for level in normalized
    )
    padded_slots = sum(
        count * length
        for count, length in zip(path_counts, max_lengths, strict=True)
    )
    export_or_mask = 0
    for level in normalized[1:]:
        for _path, parent in level:
            export_or_mask |= 1 << int(parent)
    contract = {
        "path_counts": path_counts,
        "max_lengths": max_lengths,
        "launches": len(normalized),
        "programs": sum(path_counts),
        "padded_slots": padded_slots,
        "critical": sum(max_lengths),
        "export_or_mask": export_or_mask,
        "parent_sha256": _fr13_canonical_sha256(_FR13_FIXED32_PARENT),
        "ancestry_sha256": _fr13_canonical_sha256(
            _fr13_tree_ancestry(_FR13_FIXED32_PARENT)
        ),
        "levels_sha256": _fr13_canonical_sha256(normalized),
        "coverage_sha256": _fr13_canonical_sha256(coverage),
    }
    expected = {
        "path_counts": (1, 11),
        "max_lengths": (5, 7),
        "launches": 2,
        "programs": 12,
        "padded_slots": 82,
        "critical": 12,
        "export_or_mask": 16915,
        "parent_sha256": _FR13_FIXED32_PARENT_SHA256,
        "ancestry_sha256": _FR13_FIXED32_ANCESTRY_SHA256,
        "levels_sha256": _FR13_FIXED32_LEVELS_SHA256,
        "coverage_sha256": _FR13_FIXED32_COVERAGE_SHA256,
    }
    if contract != expected or coverage != tuple(range(32)):
        raise RuntimeError(
            "FR13_FIXED32: exact GDN schedule contract mismatch "
            f"actual={contract!r} coverage={coverage!r}"
        )
    return contract


def _fr13_fixed32_gdn_single_launch_contract(
    levels,
    groups=_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS,
) -> dict[str, object]:
    """Validate the exact two-tile, ordered depth-first fixed32 schedule."""
    normalized = tuple(
        tuple(
            (tuple(int(node) for node in path), int(parent))
            for path, parent in level
        )
        for level in levels
    )
    normalized_groups = tuple(
        (int(parent), tuple(int(index) for index in indices))
        for parent, indices in groups
    )
    if len(normalized) != 2 or len(normalized[0]) != 1:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH requires one root path and one "
            "terminal-path level"
        )
    root_path, root_parent = normalized[0][0]
    level1 = normalized[1]
    if root_parent != -1 or root_path != _FR13_FIXED32_EXPORT_NODES:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH root-chain drift: "
            + repr((root_path, root_parent))
        )
    if tuple(parent for parent, _indices in normalized_groups) != root_path:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH interleave order drift: "
            + repr(normalized_groups)
        )
    covered_paths = tuple(
        index for _parent, indices in normalized_groups for index in indices
    )
    if tuple(sorted(covered_paths)) != tuple(range(len(level1))):
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH branch coverage drift: "
            + repr(covered_paths)
        )
    for parent, indices in normalized_groups:
        if not indices or tuple(sorted(indices)) != indices:
            raise RuntimeError(
                "FR13_FIXED32_GDN_SINGLE_LAUNCH branch order drift: "
                + repr((parent, indices))
            )
        for index in indices:
            if level1[index][1] != parent:
                raise RuntimeError(
                    "FR13_FIXED32_GDN_SINGLE_LAUNCH parent/path mismatch: "
                    + repr((parent, index, level1[index][1]))
                )
    for previous, node in zip(root_path, root_path[1:]):
        if _FR13_FIXED32_PARENT[node] != previous:
            raise RuntimeError(
                "FR13_FIXED32_GDN_SINGLE_LAUNCH root edge drift: "
                + repr((previous, node, _FR13_FIXED32_PARENT[node]))
            )

    execution_nodes = []
    for parent, indices in normalized_groups:
        execution_nodes.append(parent)
        for index in indices:
            execution_nodes.extend(level1[index][0])
    execution = tuple(execution_nodes)
    if (
        tuple(sorted(execution)) != tuple(range(32))
        or any(execution.count(node) != 1 for node in range(32))
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH output/ring writer drift: "
            + repr(execution)
        )
    group_sizes = tuple(
        len(indices) for _parent, indices in normalized_groups
    )
    contract = {
        "candidate": _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID,
        "root_nodes": root_path,
        "branch_path_indices": tuple(
            indices for _parent, indices in normalized_groups
        ),
        "group_sizes": group_sizes,
        "groups": len(normalized_groups),
        "max_group_paths": max(group_sizes),
        "launches": 1,
        "physical_grid_z": (1,),
        "physical_programs": 1,
        "node_updates": len(execution),
        "critical_node_steps": len(execution),
        "live_state_tiles": 2,
        "state_export_writes": 0,
        "state_parent_reads": 0,
        "single_writer_nodes": len(execution),
        "outer_root_loop": "ordered_tl_range",
        "groups_sha256": _fr13_canonical_sha256(normalized_groups),
        "execution_sha256": _fr13_canonical_sha256(execution),
    }
    expected = {
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "root_nodes": (0, 1, 4, 9, 14),
        "branch_path_indices": (
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (0, 9, 10),
        ),
        "group_sizes": (2, 2, 2, 2, 3),
        "groups": 5,
        "max_group_paths": 3,
        "launches": 1,
        "physical_grid_z": (1,),
        "physical_programs": 1,
        "node_updates": 32,
        "critical_node_steps": 32,
        "live_state_tiles": 2,
        "state_export_writes": 0,
        "state_parent_reads": 0,
        "single_writer_nodes": 32,
        "outer_root_loop": "ordered_tl_range",
        "groups_sha256": (
            "cba9010f16772510ff6017e866a520552e7ada913bb786152133597cbc7c1f62"
        ),
        "execution_sha256": (
            "80aed4d1a882ee4d4cde21dbf4314ed3abaae3f7553e35b6db5cd7574fe3b7db"
        ),
    }
    if contract != expected:
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH contract drift: "
            + repr(contract)
        )
    return contract


def _fr13_fixed32_gdn_prescaled_path_descriptor(
    levels,
    single_contract: dict[str, object],
) -> dict[str, object]:
    """Derive the exact path bases while preserving the loaded loop lengths."""
    branch_level = tuple(levels[1])
    max_path_len = max(len(path) for path, _parent in branch_level)
    max_group_paths = int(single_contract["max_group_paths"])
    if len(branch_level) != 11 or max_path_len != 7 or max_group_paths != 3:
        raise RuntimeError(
            "FR13_FIXED32_GDN_PRESCALED_PATH_BASE descriptor extent drift"
        )
    path_bases = []
    for indices in single_contract["branch_path_indices"]:
        row = [int(index) * max_path_len for index in indices]
        row.extend([0] * (max_group_paths - len(row)))
        path_bases.append(tuple(row))
    path_base_lengths = [0] * (len(branch_level) * max_path_len)
    for path_index, (path, _parent) in enumerate(branch_level):
        path_base_lengths[path_index * max_path_len] = len(path)
    descriptor = {
        "schema": "fr13.fixed32.gdn_prescaled_path_base.v1",
        "max_path_len": max_path_len,
        "path_bases": tuple(path_bases),
        "path_base_lengths": tuple(path_base_lengths),
    }
    if descriptor["path_bases"] != (
        (7, 14, 0),
        (21, 28, 0),
        (35, 42, 0),
        (49, 56, 0),
        (0, 63, 70),
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_PRESCALED_PATH_BASE derived bases drift"
        )
    return descriptor


def _fr13_fixed32_gdn_single_launch_path_args(
    single_launch: dict[str, object],
    branch_lengths,
    branch_max_len: int,
):
    """Select validated incumbent or pre-scaled device descriptors."""
    if not _FR13_FIXED32_GDN_PRESCALED_PATH_BASE:
        return branch_lengths, single_launch["path_indices"], False
    contract = single_launch.get("prescaled_contract")
    path_bases = single_launch.get("prescaled_path_bases")
    path_base_lengths = single_launch.get("prescaled_path_base_lengths")
    if (
        not isinstance(contract, dict)
        or contract.get("schema")
        != "fr13.fixed32.gdn_prescaled_path_base.v1"
        or contract.get("max_path_len") != int(branch_max_len)
        or not isinstance(path_bases, torch.Tensor)
        or not isinstance(path_base_lengths, torch.Tensor)
        or tuple(path_bases.shape) != (5, 3)
        or tuple(path_base_lengths.shape) != (77,)
        or path_bases.dtype != torch.int32
        or path_base_lengths.dtype != torch.int32
        or not path_bases.is_contiguous()
        or not path_base_lengths.is_contiguous()
        or path_bases.device != branch_lengths.device
        or path_base_lengths.device != branch_lengths.device
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_PRESCALED_PATH_BASE descriptor drift; "
            "no fallback is permitted"
        )
    return path_base_lengths, path_bases, True


def _subtree_decompose(parent) -> list:
    """Heavy-path decomposition -> list of LEVELS; each level is a list of
    (path_nodes, parent_node) with parent_node = -1 for the root path.

    Heavy child = the child whose subtree is DEEPEST (ties -> lowest id, a
    deterministic choice). Every node belongs to exactly one path; a path's
    root's parent node always lives in an EARLIER level -> one launch per
    level with an fp32 state export between levels.
    """
    parent = [int(p) for p in parent]
    fixed32_parent = globals().get("_FR13_FIXED32_PARENT")
    if fixed32_parent is not None and tuple(parent) == fixed32_parent:
        return [
            [(list(path), int(par)) for path, par in level]
            for level in _FR13_FIXED32_SUBTREE_LEVELS
        ]
    if tuple(parent) == _FR13_HYDRA23_PARENT:
        return [
            [(list(path), int(par)) for path, par in level]
            for level in _FR13_HYDRA23_SUBTREE_LEVELS
        ]

    n = len(parent)
    children: list = [[] for _ in range(n)]
    for i in range(1, n):
        children[int(parent[i])].append(i)
    depth = [1] * n
    for i in range(n - 1, -1, -1):
        if children[i]:
            depth[i] = 1 + max(depth[c] for c in children[i])
    levels: list = []
    # (path start node, parent node, level index)
    stack = [(0, -1, 0)]
    while stack:
        start, par, lvl = stack.pop()
        path = []
        cur = start
        while True:
            path.append(cur)
            ch = children[cur]
            if not ch:
                break
            heavy = max(ch, key=lambda c: (depth[c], -c))
            for c in ch:
                if c != heavy:
                    stack.append((c, cur, lvl + 1))
            cur = heavy
        while len(levels) <= lvl:
            levels.append([])
        levels[lvl].append((path, par))
    return levels


def _validate_subtree_decomposition(parent, levels) -> None:
    """Fail before capture if a static path descriptor violates tree order."""
    parent = tuple(int(p) for p in parent)
    expected = set(range(len(parent)))
    seen: set[int] = set()
    earlier: set[int] = set()
    if not levels:
        raise ValueError("FR13_SUBTREE_PARALLEL: empty decomposition")

    for level_idx, level in enumerate(levels):
        if not level:
            raise ValueError(
                f"FR13_SUBTREE_PARALLEL: empty level {level_idx}"
            )
        current: set[int] = set()
        for raw_path, raw_par in level:
            path = [int(node) for node in raw_path]
            par = int(raw_par)
            if not path:
                raise ValueError(
                    f"FR13_SUBTREE_PARALLEL: empty path in level {level_idx}"
                )
            root = path[0]
            if root not in expected:
                raise ValueError(
                    f"FR13_SUBTREE_PARALLEL: node {root} out of range"
                )
            if int(parent[root]) != par:
                raise ValueError(
                    "FR13_SUBTREE_PARALLEL: path root/parent mismatch "
                    f"root={root} descriptor={par} tree={parent[root]}"
                )
            if par < 0 and (root != 0 or par != -1):
                raise ValueError(
                    "FR13_SUBTREE_PARALLEL: invalid root path "
                    f"root={root} parent={par}"
                )
            if par >= 0 and par not in earlier:
                raise ValueError(
                    "FR13_SUBTREE_PARALLEL: path parent is not in an "
                    f"earlier level root={root} parent={par} level={level_idx}"
                )
            if len(set(path)) != len(path):
                raise ValueError(
                    "FR13_SUBTREE_PARALLEL: duplicate node within path "
                    f"{path}"
                )
            for prev, node in zip(path, path[1:]):
                if node not in expected or int(parent[node]) != prev:
                    raise ValueError(
                        "FR13_SUBTREE_PARALLEL: non-parent path edge "
                        f"{prev}->{node}"
                    )
            overlap = seen.intersection(path) | current.intersection(path)
            if overlap:
                raise ValueError(
                    "FR13_SUBTREE_PARALLEL: duplicate nodes "
                    f"{sorted(overlap)}"
                )
            current.update(path)
        seen.update(current)
        earlier.update(current)

    if seen != expected:
        raise ValueError(
            "FR13_SUBTREE_PARALLEL: decomposition coverage mismatch "
            f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}"
        )


def subtree_preseed(parent, n_actual: int, vh: int, dv: int, dk: int,
                    device) -> None:
    """Preseed path tensors + the fp32 state-export buffer at builder init
    (outside capture; the tree-decode-first-call-inside-capture class)."""
    parent_tuple = tuple(int(p) for p in parent)
    if len(parent_tuple) != int(n_actual):
        raise ValueError(
            "FR13_SUBTREE_PARALLEL: parent length mismatch "
            f"len={len(parent_tuple)} n_actual={n_actual}"
        )
    if (
        _FR13_FIXED32_MODE is not None
        and parent_tuple != _FR13_FIXED32_PARENT
    ):
        raise RuntimeError(
            "FR13_FIXED32: armed runtime offered a non-fixed physical tree "
            f"mode={_FR13_FIXED32_MODE!r} n_actual={n_actual} "
            f"parent_sha256={_fr13_canonical_sha256(parent_tuple)}"
        )
    key = _subtree_cache_key(n_actual, vh, dv, dk, device)
    if key in _FR13_SUBTREE_CACHE:
        if _FR13_SUBTREE_CACHE[key].get("parent") != parent_tuple:
            raise RuntimeError(
                "FR13_SUBTREE_PARALLEL: shape/device cache collision for "
                f"different parent vectors (key={key!r})"
            )
        return
    levels = _subtree_decompose(parent_tuple)
    _validate_subtree_decomposition(parent_tuple, levels)
    fixed_contract = None
    fixed32_parent_slots = None
    fixed32_single_launch = None
    if parent_tuple == _FR13_FIXED32_PARENT:
        fixed_contract = _fr13_fixed32_schedule_contract(levels)
        schedule = "fixed32"
        root_path = tuple(int(node) for node in levels[0][0][0])
        if root_path != _FR13_FIXED32_EXPORT_NODES:
            raise RuntimeError(
                "FR13_FIXED32: root path no longer matches compact export slots "
                f"root_path={root_path!r} "
                f"export_nodes={_FR13_FIXED32_EXPORT_NODES!r}"
            )
        export_slot = {
            node: slot for slot, node in enumerate(_FR13_FIXED32_EXPORT_NODES)
        }
        fixed32_parent_slots = []
        for level in levels:
            slots = []
            for _path, parent_node in level:
                if int(parent_node) < 0:
                    slots.append(-1)
                elif int(parent_node) in export_slot:
                    slots.append(export_slot[int(parent_node)])
                else:
                    raise RuntimeError(
                        "FR13_FIXED32: path parent has no compact export slot "
                        f"parent={int(parent_node)}"
                    )
            fixed32_parent_slots.append(
                torch.tensor(slots, dtype=torch.int32, device=device)
            )
        if (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            # The credentialed arm reaches the same folded kernel as the
            # diagnostic bool, so it needs the same preseeded descriptor. Without
            # this the B1/B4 contract guards would refuse a perfectly valid
            # production serve for a missing descriptor.
            or _FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION is not None
            or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
            or _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        ):
            single_contract = _fr13_fixed32_gdn_single_launch_contract(levels)
            max_group_paths = int(single_contract["max_group_paths"])
            group_path_indices = torch.full(
                (int(single_contract["groups"]), max_group_paths),
                -1,
                dtype=torch.int32,
            )
            for group_index, indices in enumerate(
                single_contract["branch_path_indices"]
            ):
                group_path_indices[group_index, : len(indices)] = torch.tensor(
                    indices, dtype=torch.int32
                )
            prescaled_descriptor = (
                _fr13_fixed32_gdn_prescaled_path_descriptor(
                    levels, single_contract
                )
            )
            fixed32_single_launch = {
                "contract": single_contract,
                "path_indices": group_path_indices.to(device),
                "path_counts": torch.tensor(
                    single_contract["group_sizes"],
                    dtype=torch.int32,
                    device=device,
                ),
                "prescaled_contract": prescaled_descriptor,
                "prescaled_path_bases": torch.tensor(
                    prescaled_descriptor["path_bases"],
                    dtype=torch.int32,
                    device=device,
                ),
                "prescaled_path_base_lengths": torch.tensor(
                    prescaled_descriptor["path_base_lengths"],
                    dtype=torch.int32,
                    device=device,
                ),
            }
    elif parent_tuple == _FR13_HYDRA23_PARENT:
        schedule = "hydra23_floor"
    else:
        schedule = "heavy_path"
    route_armed = _FR13_SUBTREE_ROUTE_REQUESTED
    selfcheck_armed = _FR13_SUBTREE_SELFCHECK_REQUESTED
    dev_levels = []
    for lvl in levels:
        max_len = max(len(p) for p, _ in lvl)
        nodes = torch.full((len(lvl), max_len), -1, dtype=torch.int32)
        pars = torch.empty(len(lvl), dtype=torch.int32)
        lengths = torch.empty(len(lvl), dtype=torch.int32)
        for i, (p, par) in enumerate(lvl):
            nodes[i, : len(p)] = torch.tensor(p, dtype=torch.int32)
            pars[i] = par
            lengths[i] = len(p)
        dev_levels.append(
            (
                nodes.to(device),
                pars.to(device),
                max_len,
                len(lvl),
                lengths.to(device),
            )
        )
    # export mask: nodes that are some later-level path root's parent
    need = set()
    for lvl in levels[1:]:
        for _p, par in lvl:
            need.add(int(par))
    emask = torch.zeros(int(n_actual), dtype=torch.int32)
    for nd in need:
        emask[nd] = 1
    export = torch.zeros(
        int(n_actual), int(vh), int(dv), int(dk),
        dtype=torch.float32, device=device,
    )
    _FR13_SUBTREE_CACHE[key] = {
        "levels": dev_levels,
        "emask": emask.to(device),
        "export": export,
        "n_levels": len(levels),
        "critical": sum(level[2] for level in dev_levels),
        "parent": parent_tuple,
        "schedule": schedule,
        "fixed32_contract": fixed_contract,
        "fixed32_parent_slots": fixed32_parent_slots,
        "fixed32_single_launch": fixed32_single_launch,
        "fixed32_single_launch_contract": (
            fixed32_single_launch["contract"]
            if fixed32_single_launch is not None
            else None
        ),
        "last_executed_gdn": None,
        "route_armed": route_armed,
        "selfcheck_armed": selfcheck_armed,
        "engaged_announced": False,
        "selfcheck_pass_announced": False,
    }
    print(
        f"[FR13_SUBTREE_PARALLEL] preseeded: n={n_actual} "
        f"schedule={schedule} levels="
        f"{[level[3] for level in dev_levels]} "
        f"lens={[level[2] for level in dev_levels]} "
        f"critical={sum(level[2] for level in dev_levels)} "
        f"(monolith {n_actual}) "
        f"single_launch={int(fixed32_single_launch is not None)} "
        f"route_armed={int(route_armed)} "
        f"selfcheck_armed={int(selfcheck_armed)}",
        flush=True,
    )


def subtree_get(n_actual: int, vh: int, dv: int, dk: int, device):
    st = _FR13_SUBTREE_CACHE.get(
        _subtree_cache_key(n_actual, vh, dv, dk, device)
    )
    if st is None:
        raise RuntimeError(
            "FR13_SUBTREE_PARALLEL: no preseed for "
            f"n_actual={n_actual} shape=({vh},{dv},{dk}) device={device} "
            "(builder-init wiring missing; cannot derive inside capture)"
        )
    return st


def fixed32_offline_selftest() -> dict[str, object]:
    """Pure CPU validation for topology, coverage, and fixed work counts."""
    levels = _subtree_decompose(_FR13_FIXED32_PARENT)
    _validate_subtree_decomposition(_FR13_FIXED32_PARENT, levels)
    contract = _fr13_fixed32_schedule_contract(levels)
    normalized = tuple(
        tuple((tuple(path), parent) for path, parent in level)
        for level in levels
    )
    if normalized != _FR13_FIXED32_SUBTREE_LEVELS:
        raise AssertionError("fixed32 decomposition did not select exact levels")
    return {
        "mode_values": tuple(sorted(_FR13_FIXED32_MODES)),
        "rows": len(_FR13_FIXED32_PARENT),
        **contract,
    }


def hc_slot_map_get(n_actual: int, n_pad: int, device) -> torch.Tensor:
    """Device int32[n_pad] node->compacted-slot map (leaves/pad -> 0).

    Enables FR13_HC_INTERNAL + FR13_PARENT_GATHER together: the parent-gather
    branch's parent index is RUNTIME, so its one-hot select needs a runtime
    slot lookup instead of the trace-time constexpr map. A leaf can never be
    a parent (internal by definition), so the 0 filler is unreachable on the
    select path; parent_i < 0 (root) is handled by the existing where().
    Built from the preseeded/derived mask; cached per (shape, device);
    allocation is capture-guarded like every other preseed buffer.
    """
    key = ("slotmap", int(n_actual), int(n_pad), str(device))
    hit = _FR13_HC_DESC_CACHE.get(key)
    if hit is not None:
        return hit
    desc = _FR13_HC_DESC_CACHE.get(("shape", int(n_actual), int(n_pad)))
    if desc is None:
        raise RuntimeError(
            "FR13_HC_INTERNAL slot map requested before preseed "
            f"(n_actual={n_actual}, n_pad={n_pad})"
        )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_HC_INTERNAL: slot-map allocation inside graph capture; "
            "preseed must run at builder init"
        )
    mask, _rows, slots_lo, slots_hi = desc
    vals = []
    for node in range(int(n_pad)):
        if node < 32 and ((mask >> node) & 1):
            if node < 16:
                vals.append((slots_lo >> (4 * node)) & 15)
            else:
                vals.append((slots_hi >> (4 * (node - 16))) & 15)
        else:
            vals.append(0)
    t = torch.tensor(vals, dtype=torch.int32).to(device)
    _FR13_HC_DESC_CACHE[key] = t
    return t


def hc_internal_preseed(parent, n_actual: int, n_pad: int, device=None) -> None:
    """Preseed the HC descriptor from the HOST parent list at builder init.

    REQUIRED for graph boots: the tree-decode shape's FIRST scan invocation
    happens INSIDE CUDA-graph capture (vLLM's eager warmup covers prefill
    shapes only), so the strict_mask host read in ``_hc_internal_desc`` can
    never run there. The parent list gives the internal set directly
    (internal == appears as someone's immediate parent). Keyed by
    (n_actual, n_pad) — the served tree shape is locked per boot.
    """
    parents = {int(p) for p in parent if int(p) >= 0}
    _FR13_HC_DESC_CACHE[("shape", int(n_actual), int(n_pad))] = _hc_pack(
        parents, int(n_actual)
    )
    if device is not None:
        # PG-compat runtime slot map preseeded here too (outside capture).
        hc_slot_map_get(int(n_actual), int(n_pad), device)


def _hc_internal_desc(strict_mask: torch.Tensor, n_actual: int, n_pad: int):
    """One-time host derivation of (HC_MASK, HC_ROWS, HC_SLOTS_LO, HC_SLOTS_HI).

    internal set = { immediate parent of i : i in [1, n_actual) } -- and every
    strict-ancestor of any node IS some node's immediate parent (the next node
    down its path), so this set covers every row the one-hot reads can select.
    Cached by (descriptor ptr, n_actual, n_pad): the served tree descriptors
    are static buffers. The first call must happen OUTSIDE graph capture
    (vLLM's eager warmup); fail loud otherwise -- never silently fall back.
    """
    hit = _FR13_HC_DESC_CACHE.get(("shape", int(n_actual), int(n_pad)))
    if hit is not None:
        return hit
    key = (int(strict_mask.data_ptr()), int(n_actual), int(n_pad))
    hit = _FR13_HC_DESC_CACHE.get(key)
    if hit is not None:
        return hit
    if strict_mask.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_HC_INTERNAL: first-call descriptor derivation hit graph "
            "capture and no init-time preseed exists (hc_internal_preseed "
            "was not called for this shape); the builder-init wiring is broken"
        )
    sm_host = strict_mask[:n_actual, :n_actual].detach().to("cpu")
    parents: set = set()
    for i in range(1, n_actual):
        row = sm_host[i]
        p = -1
        for j in range(0, i):
            if int(row[j]) != 0:
                p = j  # largest-index ancestor == immediate parent (topo order)
        if p >= 0:
            parents.add(p)
    out = _hc_pack(parents, n_actual)
    _FR13_HC_DESC_CACHE[key] = out
    return out


@dataclass(frozen=True)
class Tree:
    parent: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.parent)

    def ancestors(self, node: int) -> tuple[int, ...]:
        out = []
        cur = self.parent[node]
        while cur >= 0:
            out.append(cur)
            cur = self.parent[cur]
        return tuple(reversed(out))

    def path(self, node: int) -> tuple[int, ...]:
        return (*self.ancestors(node), node)

    def is_single_spine(self) -> bool:
        return self.parent == tuple([-1, *range(0, self.n - 1)])

    def masks(self, device: torch.device, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        for i in range(self.n):
            visible[i, i] = 1
            for j in self.ancestors(i):
                strict[i, j] = 1
                visible[i, j] = 1
        return strict, visible


def make_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    if n == 2:
        return Tree((-1, 0))
    if n == 3:
        return Tree((-1, 0, 0))
    if n == 6:
        return Tree((-1, 0, 1, 2, 2, 1))
    if n == 8:
        return Tree((-1, 0, 1, 2, 3, 3, 2, 1))
    return Tree((-1, 0, 1, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0))


def make_spine_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    return Tree(tuple([-1, *range(0, n - 1)]))


def padded_nodes(n: int) -> int:
    n_pad = 1 << (n - 1).bit_length()
    if n_pad > 32:
        raise ValueError(f"FR10 tree kernel only warms padded node blocks up to 32, got {n}")
    return n_pad


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


@triton.jit
def _linear_remap_rows_kernel(
    state,
    spec_state_indices,
    accepted_paths,
    num_accepted_tokens,
    B: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_blk = tl.program_id(2)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    accepted_len = tl.load(num_accepted_tokens + pid_b)
    valid_path = (pid_b < B) & (pid_k < PATH_COLS) & (pid_k < SPEC_COLS) & (pid_k < accepted_len)
    src_col = tl.load(
        accepted_paths + pid_b * PATH_COLS + pid_k,
        mask=valid_path,
        other=0,
    )
    src_col = tl.maximum(0, tl.minimum(src_col, SPEC_COLS - 1))
    src_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + src_col,
        mask=valid_path,
        other=0,
    )
    dst_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + pid_k,
        mask=valid_path,
        other=0,
    )
    mask = valid_path & (offs < ROW_ELEMS)
    vals = tl.load(state + src_bank * ROW_ELEMS + offs, mask=mask)
    tl.store(state + dst_bank * ROW_ELEMS + offs, vals, mask=mask)


@triton.jit
def _linear_remap_rows_gather_kernel(
    state,
    spec_state_indices,
    accepted_paths,
    num_accepted_tokens,
    B: tl.constexpr,
    PATH_COLS: tl.constexpr,
    PATH_POW2: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    ROW_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FR13_TREE_REMAP_SEQ: race-free remap. The legacy kernel parallelizes over
    # path columns, but accepted spine paths overlap their destinations (src
    # cols [1..L] -> dst cols [0..L-1]) so the program writing column k races
    # the program reading column k as its source (nondeterministic winner and
    # corrupted state for every accepted_len >= 2). Here a single program owns
    # ALL path columns for one (batch, element-block) slice: every source row
    # slice is loaded into registers before any destination row slice is
    # stored, which makes the in-place overlapping permutation exact.
    pid_b = tl.program_id(0)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    ks = tl.arange(0, PATH_POW2)
    accepted_len = tl.load(num_accepted_tokens + pid_b)
    valid_path = (
        (pid_b < B) & (ks < PATH_COLS) & (ks < SPEC_COLS) & (ks < accepted_len)
    )
    src_col = tl.load(
        accepted_paths + pid_b * PATH_COLS + ks,
        mask=valid_path,
        other=0,
    )
    src_col = tl.maximum(0, tl.minimum(src_col, SPEC_COLS - 1))
    src_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + src_col,
        mask=valid_path,
        other=0,
    ).to(tl.int64)
    dst_bank = tl.load(
        spec_state_indices + pid_b * SPEC_COLS + ks,
        mask=valid_path,
        other=0,
    ).to(tl.int64)
    mask = valid_path[:, None] & (offs[None, :] < ROW_ELEMS)
    vals = tl.load(
        state + src_bank[:, None] * ROW_ELEMS + offs[None, :], mask=mask
    )
    tl.store(
        state + dst_bank[:, None] * ROW_ELEMS + offs[None, :], vals, mask=mask
    )


def _remap_state_rows(
    state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    num_spec_decodes: int,
    max_path_len: int,
    block: int = 256,
) -> None:
    if num_spec_decodes <= 0 or max_path_len <= 0:
        return
    if state.ndim < 2:
        raise ValueError(f"state bank must have row dimension plus payload, got {tuple(state.shape)}")
    if spec_state_indices.ndim != 2:
        raise ValueError(f"spec_state_indices must be 2D, got {tuple(spec_state_indices.shape)}")
    if accepted_paths.ndim != 2:
        raise ValueError(f"accepted_paths must be 2D, got {tuple(accepted_paths.shape)}")
    if accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            "accepted_paths batch rows must cover num_spec_decodes="
            f"{num_spec_decodes}, got {accepted_paths.shape[0]}"
        )
    if num_accepted_tokens.numel() < num_spec_decodes:
        raise ValueError(
            "num_accepted_tokens must cover num_spec_decodes="
            f"{num_spec_decodes}, got {num_accepted_tokens.numel()}"
        )
    row_elems = state.stride(0)
    path_cols = min(int(accepted_paths.shape[1]), int(max_path_len))
    spec_cols = int(spec_state_indices.shape[1])
    if path_cols <= 0 or spec_cols <= 0:
        return
    if True:  # FR13_TREE_REMAP_SEQ baked ON (gather-then-scatter remap)
        # Race-free gather-then-scatter remap (see kernel docstring). Default
        # ON: it computes the intended permutation exactly; the legacy racy
        # A/B kernel path is now dead (flag baked to constant True).
        gather_block = min(block, 128)
        path_pow2 = max(1, triton.next_power_of_2(path_cols))
        grid = (int(num_spec_decodes), triton.cdiv(row_elems, gather_block))
        _linear_remap_rows_gather_kernel[grid](
            state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            B=int(num_spec_decodes),
            PATH_COLS=path_cols,
            PATH_POW2=path_pow2,
            SPEC_COLS=spec_cols,
            ROW_ELEMS=row_elems,
            BLOCK=gather_block,
        )
        return
    grid = (int(num_spec_decodes), path_cols, triton.cdiv(row_elems, block))
    _linear_remap_rows_kernel[grid](
        state,
        spec_state_indices,
        accepted_paths,
        num_accepted_tokens,
        B=int(num_spec_decodes),
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        ROW_ELEMS=row_elems,
        BLOCK=block,
    )


# ---- generic deferred sub-span timer (observer-effect-safe: event pairs,
# query()-guarded drain, throttled atomic JSON dump; NO hot-path syncs).
# One registry entry per env prefix, e.g. FR13_KVREMAP_TIMER(_JSON). ----
_FR13_SPAN_TIMERS: dict = {}


def _fr13_span_begin(prefix: str):
    if os.environ.get(prefix) != "1":
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _fr13_span_end(prefix: str, start_ev) -> None:
    if start_ev is None:
        return
    st = _FR13_SPAN_TIMERS.setdefault(prefix, {"pend": [], "acc": [0.0, 0], "last": [0.0]})
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    st["pend"].append((start_ev, end_ev))
    while st["pend"]:
        a, b = st["pend"][0]
        try:
            if not b.query():
                break
        except Exception:
            st["pend"].pop(0)
            continue
        st["pend"].pop(0)
        try:
            st["acc"][0] += a.elapsed_time(b) / 1000.0
            st["acc"][1] += 1
        except Exception:
            pass
    import json as _sj
    import time as _st
    out = os.environ.get(prefix + "_JSON")
    if out and (_st.monotonic() - st["last"][0]) > 5.0:
        st["last"][0] = _st.monotonic()
        try:
            tmp = out + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(_sj.dumps({"span_gpu_seconds": st["acc"][0], "n": st["acc"][1], "prefix": prefix}))
            os.replace(tmp, out)
        except Exception:
            pass


def launch_tree_state_linear_remap(
    *,
    ssm_state: torch.Tensor | None,
    conv_state: torch.Tensor | None,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    max_path_len: int,
) -> None:
    """Materialize accepted tree-path rows into stock linear state columns.

    The committer publishes accepted_paths as tree node columns. vLLM's GDN and
    causal-conv consumers read recurrent state by linear accepted-token position,
    so column k must contain the state for accepted_paths[b, k].
    """
    _fr13_srt = _fr13_span_begin("FR13_STATEREMAP_TIMER")
    if ssm_state is not None:
        _remap_state_rows(
            ssm_state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            num_spec_decodes=num_spec_decodes,
            max_path_len=max_path_len,
        )
    if conv_state is not None:
        _remap_state_rows(
            conv_state,
            spec_state_indices,
            accepted_paths,
            num_accepted_tokens,
            num_spec_decodes=num_spec_decodes,
            max_path_len=max_path_len,
        )
    _fr13_span_end("FR13_STATEREMAP_TIMER", _fr13_srt)


@triton.jit
def _fr13_conv_col0_pregather_kernel(
    anchor_ptr,        # layer-0 conv bank base (elements of DTYPE)
    off16_ptr,         # [L] int64: (ptr_l - ptr_0) // 16
    ssi_ptrs,          # [L] int64: live group-local SSI base pointers
    ssi_strides,       # [L] int64: live SSI batch strides in int32 elements
    out_ptr,           # [L, B, ROW_ELEMS] staging (same dtype as banks)
    out_stride_l, out_stride_b,
    row_stride,        # conv bank ROW stride in elements (page stride)
    s1, s2,            # conv bank C/L strides in elements
    ROW_ELEMS: tl.constexpr,   # C * CONV_L (LOGICAL conv elements only)
    CONV_L: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Copies ONLY the conv-logical [C, L] block of each col0 row (page-safe:
    # conv and ssm share the mamba page; a flat page copy both wastes ~8x
    # staging and cannot be .view(C, L)'d by consumers — the r5 crash).
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return
    off16 = tl.load(off16_ptr + pid_l)
    base = anchor_ptr + off16 * (16 // ELEM_BYTES)
    # SSI is a scalar int32 load, so the raw-pointer table does not affect the
    # bank copy's vectorization/alignment contract. Each pointer names the
    # full-capacity group tensor that the metadata builder refreshes before
    # every graph replay.
    ssi_ptr = tl.load(ssi_ptrs + pid_l).to(tl.pointer_type(tl.int32))
    row = tl.load(ssi_ptr + pid_b * tl.load(ssi_strides + pid_l))
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROW_ELEMS
    c_idx = offs // CONV_L
    l_idx = offs % CONV_L
    vals = tl.load(
        base + row.to(tl.int64) * row_stride + c_idx * s1 + l_idx * s2,
        mask=mask,
    )
    tl.store(
        out_ptr + pid_l * out_stride_l + pid_b * out_stride_b + offs,
        vals,
        mask=mask,
    )


@triton.jit
def _fr13_fixed32_conv_commit_gather_kernel(
    anchor_ptr,
    off16_ptr,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    staging,
    staging_stride_l,
    staging_stride_b,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    row_stride,
    ROW_ELEMS: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Snapshot every fixed layer/request leaf row before any bank write."""
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return
    accepted_len = tl.load(accepted_lens + pid_b * lens_stride_b)
    leaf_pos = tl.maximum(accepted_len - 1, 0)
    leaf_pos = tl.minimum(leaf_pos, PATH_COLS - 1)
    leaf_node = tl.load(
        accepted_paths
        + pid_b * path_stride_b
        + leaf_pos * path_stride_s
    )
    leaf_node = tl.where(accepted_len > 0, leaf_node, 0)
    leaf_node = tl.maximum(0, tl.minimum(leaf_node, SPEC_COLS - 1))
    spec_layer = (
        spec_state_indices
        + pid_l * ssi_stride_l
        + pid_b * ssi_stride_b
    )
    src_row = tl.load(spec_layer + leaf_node * ssi_stride_s)
    off16 = tl.load(off16_ptr + pid_l)
    base = anchor_ptr + off16 * (16 // ELEM_BYTES)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROW_ELEMS
    values = tl.load(
        base + src_row.to(tl.int64) * row_stride + offs,
        mask=mask,
    )
    tl.store(
        staging
        + pid_l * staging_stride_l
        + pid_b * staging_stride_b
        + offs,
        values,
        mask=mask,
    )


@triton.jit
def _fr13_fixed32_conv_commit_scatter_kernel(
    anchor_ptr,
    off16_ptr,
    spec_state_indices,
    staging,
    staging_stride_l,
    staging_stride_b,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    row_stride,
    ROW_ELEMS: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Publish the snapshot to col0 after the gather launch completes."""
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return
    spec_layer = (
        spec_state_indices
        + pid_l * ssi_stride_l
        + pid_b * ssi_stride_b
    )
    dst_row = tl.load(spec_layer + 0 * ssi_stride_s)
    off16 = tl.load(off16_ptr + pid_l)
    base = anchor_ptr + off16 * (16 // ELEM_BYTES)
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROW_ELEMS
    values = tl.load(
        staging
        + pid_l * staging_stride_l
        + pid_b * staging_stride_b
        + offs,
        mask=mask,
    )
    tl.store(
        base + dst_row.to(tl.int64) * row_stride + offs,
        values,
        mask=mask,
    )


@triton.jit
def _fr13_fixed32_conv_direct_col0_kernel(
    anchor_ptr,
    bank_off16,
    source_anchor,
    source_off16,
    state_src,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    bank_row_stride,
    bank_c_stride,
    bank_l_stride,
    source_row_stride,
    source_c_stride,
    CONV_C: tl.constexpr,
    CONV_L: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ZERO_TAIL: tl.constexpr,
    LIVE_STATE_COLS: tl.constexpr,
):
    """Publish the accepted source-stage leaf directly to the running row."""
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return

    accepted_len = tl.load(accepted_lens + pid_b * lens_stride_b)
    leaf_pos = tl.maximum(accepted_len - 1, 0)
    leaf_pos = tl.minimum(leaf_pos, PATH_COLS - 1)
    leaf_node = tl.load(
        accepted_paths
        + pid_b * path_stride_b
        + leaf_pos * path_stride_s
    )
    leaf_node = tl.where(accepted_len > 0, leaf_node, 0)
    leaf_node = tl.maximum(0, tl.minimum(leaf_node, SPEC_COLS - 1))

    spec_layer = (
        spec_state_indices
        + pid_l * ssi_stride_l
        + pid_b * ssi_stride_b
    )
    dst_row = tl.load(spec_layer + 0 * ssi_stride_s)
    bank = anchor_ptr + tl.load(bank_off16 + pid_l) * (16 // ELEM_BYTES)
    source = (
        source_anchor
        + tl.load(source_off16 + pid_l) * (16 // ELEM_BYTES)
    )
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < CONV_C
    source_batch = pid_b.to(tl.int64) * SOURCE_ROWS
    for state_col in tl.static_range(0, CONV_L):
        if ZERO_TAIL and state_col >= LIVE_STATE_COLS:
            values = tl.zeros((BLOCK_C,), dtype=tl.bfloat16)
        else:
            source_row = tl.load(
                state_src + leaf_node * CONV_L + state_col
            ).to(tl.int64)
            values = tl.load(
                source
                + (source_batch + source_row) * source_row_stride
                + offs_c * source_c_stride,
                mask=c_mask,
            )
        tl.store(
            bank
            + dst_row.to(tl.int64) * bank_row_stride
            + offs_c * bank_c_stride
            + state_col * bank_l_stride,
            values,
            mask=c_mask,
        )


@triton.jit
def _fr13_fixed32_conv_zero_tail_compare_kernel(
    anchor_ptr,
    bank_off16,
    source_anchor,
    source_off16,
    state_src,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    count_enable,
    compared_events,
    differing_bytes,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    bank_row_stride,
    bank_c_stride,
    bank_l_stride,
    source_row_stride,
    source_c_stride,
    CONV_C: tl.constexpr,
    CONV_L: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compare candidate destination bits to the source-derived incumbent."""
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return

    enabled = tl.load(count_enable).to(tl.int64)
    accepted_len = tl.load(accepted_lens + pid_b * lens_stride_b)
    leaf_pos = tl.maximum(accepted_len - 1, 0)
    leaf_pos = tl.minimum(leaf_pos, PATH_COLS - 1)
    leaf_node = tl.load(
        accepted_paths
        + pid_b * path_stride_b
        + leaf_pos * path_stride_s
    )
    leaf_node = tl.where(accepted_len > 0, leaf_node, 0)
    leaf_node = tl.maximum(0, tl.minimum(leaf_node, SPEC_COLS - 1))

    spec_layer = (
        spec_state_indices
        + pid_l * ssi_stride_l
        + pid_b * ssi_stride_b
    )
    dst_row = tl.load(spec_layer + 0 * ssi_stride_s)
    bank = anchor_ptr + tl.load(bank_off16 + pid_l) * (16 // ELEM_BYTES)
    source = (
        source_anchor
        + tl.load(source_off16 + pid_l) * (16 // ELEM_BYTES)
    )
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < CONV_C
    source_batch = pid_b.to(tl.int64) * SOURCE_ROWS
    byte_mismatches = tl.zeros((BLOCK_C,), dtype=tl.int32)
    for state_col in tl.static_range(0, CONV_L):
        source_row = tl.load(
            state_src + leaf_node * CONV_L + state_col
        ).to(tl.int64)
        reference = tl.load(
            source
            + (source_batch + source_row) * source_row_stride
            + offs_c * source_c_stride,
            mask=c_mask,
            other=0.0,
        ).to(tl.uint16, bitcast=True)
        candidate = tl.load(
            bank
            + dst_row.to(tl.int64) * bank_row_stride
            + offs_c * bank_c_stride
            + state_col * bank_l_stride,
            mask=c_mask,
            other=0.0,
        ).to(tl.uint16, bitcast=True)
        xor = reference ^ candidate
        byte_mismatches += (
            ((xor & 0xFF) != 0).to(tl.int32)
            + ((xor >> 8) != 0).to(tl.int32)
        ) * c_mask.to(tl.int32)
    mismatch_count = tl.sum(byte_mismatches, axis=0).to(tl.int64)
    tl.atomic_add(differing_bytes, mismatch_count * enabled)
    tl.atomic_add(
        compared_events,
        enabled,
        mask=(pid_l == 0) & (pid_b == 0) & (pid_c == 0),
    )


@triton.jit
def _fr13_fixed32_conv_direct_col0_metadata_kernel(
    anchor_ptr,
    bank_off16,
    source_anchor,
    source_off16,
    state_src,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    committer_paths,
    committer_lens,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    committer_path_stride_b,
    committer_path_stride_s,
    committer_lens_stride_b,
    bank_row_stride,
    bank_c_stride,
    bank_l_stride,
    source_row_stride,
    source_c_stride,
    CONV_C: tl.constexpr,
    CONV_L: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    ELEM_BYTES: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ZERO_TAIL: tl.constexpr,
    LIVE_STATE_COLS: tl.constexpr,
):
    """Direct col0 commit with one disjoint metadata writer per request."""
    pid_l = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    if pid_b >= B:
        return

    accepted_len = tl.load(accepted_lens + pid_b * lens_stride_b)
    metadata_writer = (pid_l == 0) & (pid_c == 0)
    if metadata_writer:
        path_cols = tl.arange(0, PATH_COLS)
        path_values = tl.load(
            accepted_paths
            + pid_b * path_stride_b
            + path_cols * path_stride_s,
        )
        tl.store(
            committer_paths
            + pid_b * committer_path_stride_b
            + path_cols * committer_path_stride_s,
            path_values,
        )
        tl.store(
            committer_lens + pid_b * committer_lens_stride_b,
            accepted_len,
        )

    leaf_pos = tl.maximum(accepted_len - 1, 0)
    leaf_pos = tl.minimum(leaf_pos, PATH_COLS - 1)
    leaf_node = tl.load(
        accepted_paths
        + pid_b * path_stride_b
        + leaf_pos * path_stride_s
    )
    leaf_node = tl.where(accepted_len > 0, leaf_node, 0)
    leaf_node = tl.maximum(0, tl.minimum(leaf_node, SPEC_COLS - 1))

    spec_layer = (
        spec_state_indices
        + pid_l * ssi_stride_l
        + pid_b * ssi_stride_b
    )
    dst_row = tl.load(spec_layer + 0 * ssi_stride_s)
    bank = anchor_ptr + tl.load(bank_off16 + pid_l) * (16 // ELEM_BYTES)
    source = (
        source_anchor
        + tl.load(source_off16 + pid_l) * (16 // ELEM_BYTES)
    )
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < CONV_C
    source_batch = pid_b.to(tl.int64) * SOURCE_ROWS
    for state_col in tl.static_range(0, CONV_L):
        if ZERO_TAIL and state_col >= LIVE_STATE_COLS:
            values = tl.zeros((BLOCK_C,), dtype=tl.bfloat16)
        else:
            source_row = tl.load(
                state_src + leaf_node * CONV_L + state_col
            ).to(tl.int64)
            values = tl.load(
                source
                + (source_batch + source_row) * source_row_stride
                + offs_c * source_c_stride,
                mask=c_mask,
            )
        tl.store(
            bank
            + dst_row.to(tl.int64) * bank_row_stride
            + offs_c * bank_c_stride
            + state_col * bank_l_stride,
            values,
            mask=c_mask,
        )


@triton.jit
def _fr13_fixed32_conv_commit_row_guard_kernel(
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    bank_alias_ids,
    bank_alias_peer_layers,
    guard_flags,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    peer_stride_l,
    peer_stride_s,
    BANK_ROWS: tl.constexpr,
    B: tl.constexpr,
    LAYERS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    MAX_ACCEPTED: tl.constexpr,
    ALIAS_WIDTH: tl.constexpr,
    PEER_CAP: tl.constexpr,
):
    """Validate one fixed physical32 layer/request destination."""
    pid = tl.program_id(0)
    layer = pid // B
    request = pid - layer * B
    ssi_base = (
        spec_state_indices
        + layer * ssi_stride_l
        + request * ssi_stride_b
    )

    row_offsets = tl.arange(0, SPEC_COLS)
    rows = tl.load(ssi_base + row_offsets * ssi_stride_s).to(tl.int64)
    rows_ok = tl.sum(
        ((rows > 0) & (rows < BANK_ROWS)).to(tl.int32), axis=0
    ) == SPEC_COLS

    running_row = tl.load(ssi_base).to(tl.int64)
    peer_offsets = tl.arange(0, PEER_CAP)
    peer_slots = peer_offsets // B
    other_requests = peer_offsets - peer_slots * B
    peer_entry = peer_slots < ALIAS_WIDTH
    peer_layers = tl.load(
        bank_alias_peer_layers
        + layer * peer_stride_l
        + peer_slots * peer_stride_s,
        mask=peer_entry,
        other=-1,
    ).to(tl.int64)
    peer_layer_ok = (peer_layers >= 0) & (peer_layers < LAYERS)
    peer_table_ok = tl.sum(
        ((~peer_entry) | peer_layer_ok).to(tl.int32), axis=0
    ) == PEER_CAP
    peer_layers_safe = tl.maximum(0, tl.minimum(peer_layers, LAYERS - 1))
    other_rows = tl.load(
        spec_state_indices
        + peer_layers_safe * ssi_stride_l
        + other_requests * ssi_stride_b,
        mask=peer_entry & peer_layer_ok,
        other=-1,
    ).to(tl.int64)
    duplicate_destination = (
        peer_entry
        & peer_layer_ok
        & ((peer_layers != layer) | (other_requests != request))
        & (other_rows == running_row)
    )
    destination_unique = tl.sum(
        duplicate_destination.to(tl.int32), axis=0
    ) == 0

    contract_ok = rows_ok & peer_table_ok & destination_unique
    if layer == 0:
        accepted_len = tl.load(
            accepted_lens + request * lens_stride_b
        ).to(tl.int64)
        path_offsets = tl.arange(0, PATH_COLS)
        paths = tl.load(
            accepted_paths
            + request * path_stride_b
            + path_offsets * path_stride_s
        ).to(tl.int64)
        active_paths = path_offsets < accepted_len
        paths_ok = tl.sum(
            (
                (~active_paths)
                | ((paths >= 0) & (paths < SPEC_COLS))
            ).to(tl.int32),
            axis=0,
        ) == PATH_COLS
        lens_ok = (accepted_len >= 0) & (accepted_len <= MAX_ACCEPTED)
        contract_ok = contract_ok & paths_ok & lens_ok
        if request == 0:
            alias_offsets = tl.arange(0, 32)
            aliases_lo = tl.load(bank_alias_ids + alias_offsets).to(tl.int64)
            aliases_lo_ok = tl.sum(
                ((aliases_lo >= 0) & (aliases_lo < 16)).to(tl.int32),
                axis=0,
            ) == 32
            alias_hi_entries = alias_offsets < (LAYERS - 32)
            aliases_hi = tl.load(
                bank_alias_ids + 32 + alias_offsets,
                mask=alias_hi_entries,
                other=0,
            ).to(tl.int64)
            aliases_hi_ok = tl.sum(
                (
                    (~alias_hi_entries)
                    | ((aliases_hi >= 0) & (aliases_hi < 16))
                ).to(tl.int32),
                axis=0,
            ) == 32
            contract_ok = contract_ok & aliases_lo_ok & aliases_hi_ok

    tl.store(guard_flags + pid, contract_ok)


@triton.jit
def _fr13_fixed32_conv_commit_sticky_guard_kernel(
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    bank_alias_ids,
    bank_alias_peer_layers,
    sticky_ok,
    ssi_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    path_stride_b,
    path_stride_s,
    lens_stride_b,
    peer_stride_l,
    peer_stride_s,
    BANK_ROWS: tl.constexpr,
    B: tl.constexpr,
    LAYERS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    PATH_COLS: tl.constexpr,
    MAX_ACCEPTED: tl.constexpr,
    ALIAS_WIDTH: tl.constexpr,
    PEER_CAP: tl.constexpr,
):
    """Validate physical32 and make the first failure permanently visible."""
    pid = tl.program_id(0)
    layer = pid // B
    request = pid - layer * B
    ssi_base = (
        spec_state_indices
        + layer * ssi_stride_l
        + request * ssi_stride_b
    )

    row_offsets = tl.arange(0, SPEC_COLS)
    rows = tl.load(ssi_base + row_offsets * ssi_stride_s).to(tl.int64)
    rows_ok = tl.sum(
        ((rows > 0) & (rows < BANK_ROWS)).to(tl.int32), axis=0
    ) == SPEC_COLS

    running_row = tl.load(ssi_base).to(tl.int64)
    peer_offsets = tl.arange(0, PEER_CAP)
    peer_slots = peer_offsets // B
    other_requests = peer_offsets - peer_slots * B
    peer_entry = peer_slots < ALIAS_WIDTH
    peer_layers = tl.load(
        bank_alias_peer_layers
        + layer * peer_stride_l
        + peer_slots * peer_stride_s,
        mask=peer_entry,
        other=-1,
    ).to(tl.int64)
    peer_layer_ok = (peer_layers >= 0) & (peer_layers < LAYERS)
    peer_table_ok = tl.sum(
        ((~peer_entry) | peer_layer_ok).to(tl.int32), axis=0
    ) == PEER_CAP
    peer_layers_safe = tl.maximum(0, tl.minimum(peer_layers, LAYERS - 1))
    other_rows = tl.load(
        spec_state_indices
        + peer_layers_safe * ssi_stride_l
        + other_requests * ssi_stride_b,
        mask=peer_entry & peer_layer_ok,
        other=-1,
    ).to(tl.int64)
    duplicate_destination = (
        peer_entry
        & peer_layer_ok
        & ((peer_layers != layer) | (other_requests != request))
        & (other_rows == running_row)
    )
    destination_unique = tl.sum(
        duplicate_destination.to(tl.int32), axis=0
    ) == 0

    contract_ok = rows_ok & peer_table_ok & destination_unique
    if layer == 0:
        accepted_len = tl.load(
            accepted_lens + request * lens_stride_b
        ).to(tl.int64)
        path_offsets = tl.arange(0, PATH_COLS)
        paths = tl.load(
            accepted_paths
            + request * path_stride_b
            + path_offsets * path_stride_s
        ).to(tl.int64)
        active_paths = path_offsets < accepted_len
        paths_ok = tl.sum(
            (
                (~active_paths)
                | ((paths >= 0) & (paths < SPEC_COLS))
            ).to(tl.int32),
            axis=0,
        ) == PATH_COLS
        lens_ok = (accepted_len >= 0) & (accepted_len <= MAX_ACCEPTED)
        contract_ok = contract_ok & paths_ok & lens_ok
        if request == 0:
            alias_offsets = tl.arange(0, 32)
            aliases_lo = tl.load(bank_alias_ids + alias_offsets).to(tl.int64)
            aliases_lo_ok = tl.sum(
                ((aliases_lo >= 0) & (aliases_lo < 16)).to(tl.int32),
                axis=0,
            ) == 32
            alias_hi_entries = alias_offsets < (LAYERS - 32)
            aliases_hi = tl.load(
                bank_alias_ids + 32 + alias_offsets,
                mask=alias_hi_entries,
                other=0,
            ).to(tl.int64)
            aliases_hi_ok = tl.sum(
                (
                    (~alias_hi_entries)
                    | ((aliases_hi >= 0) & (aliases_hi < 16))
                ).to(tl.int32),
                axis=0,
            ) == 32
            contract_ok = contract_ok & aliases_lo_ok & aliases_hi_ok

    # The scalar starts at one and is never reset. A failed async assertion is
    # therefore fail-closed even if its exception is caught by an outer layer.
    tl.atomic_xchg(sticky_ok, 0, mask=~contract_ok)


@triton.jit
def _fr13_fixed32_sfwd_state_fusion_kernel(
    x,
    conv_state,
    spec_state_indices,
    source_flat,
    conv_weights,
    bias,
    out,
    source_stage,
    x_stride_row,
    conv_stride_row,
    conv_stride_c,
    conv_stride_l,
    ssi_stride_b,
    ssi_stride_s,
    weight_stride_c,
    weight_stride_w,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse exact fixed32 conv compute and both state-motion directions.

    One program owns one physical tree row and one channel tile. It reads the
    prior col-0 state directly (removing the all-layer pregather consumer),
    executes the same four BF16 tap products and ordered FP32 adds as the
    incumbent, writes the BF16 conv result, and materializes the persistent
    ``prior ++ x ++ zero`` source used by the exact post-accept col-0 commit.

    Ring K/V/A/B export and freshness flags remain fused into the following
    fixed32 GDN path scan. Keeping those stores in their natural producer
    avoids an extra state-motion launch and preserves stream ordering.
    """
    pid_row = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = pid_row // N
    pid_n = pid_row - pid_b * N
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    bank_row = tl.load(
        spec_state_indices + pid_b * ssi_stride_b + 0 * ssi_stride_s
    ).to(tl.int64)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    if HAS_BIAS:
        acc = tl.load(bias + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    for tap in tl.static_range(0, WIDTH):
        source_row = tl.load(source_flat + pid_n * WIDTH + tap).to(tl.int64)
        from_prior = source_row < (WIDTH - 1)
        prior_value = tl.load(
            conv_state
            + bank_row * conv_stride_row
            + offs_c * conv_stride_c
            + source_row * conv_stride_l,
            mask=c_mask & from_prior,
            other=0.0,
        )
        x_node = source_row - (WIDTH - 1)
        x_value = tl.load(
            x
            + (pid_b.to(tl.int64) * N + x_node) * x_stride_row
            + offs_c,
            mask=c_mask & (~from_prior) & (x_node >= 0) & (x_node < N),
            other=0.0,
        )
        value = tl.where(from_prior, prior_value, x_value).to(tl.bfloat16)
        weight = tl.load(
            conv_weights
            + offs_c * weight_stride_c
            + tap * weight_stride_w,
            mask=c_mask,
            other=0.0,
        ).to(tl.bfloat16)
        # The incumbent rounds every tap product to BF16 before converting to
        # FP32 and accumulating left-to-right. Keep both cast boundary and
        # operand order explicit; reductions are deliberately absent.
        product = (value * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + product

    activated = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(
        out + (pid_b * N + pid_n) * C + offs_c,
        activated,
        mask=c_mask,
    )

    stage_base = pid_b.to(tl.int64) * SOURCE_ROWS
    current_x = tl.load(
        x + (pid_b * N + pid_n) * x_stride_row + offs_c,
        mask=c_mask,
        other=0.0,
    )
    tl.store(
        source_stage
        + (stage_base + (WIDTH - 1) + pid_n) * C
        + offs_c,
        current_x,
        mask=c_mask,
    )
    source_edge_writer = pid_n == 0
    for prior_col in tl.static_range(0, WIDTH - 1):
        prior_value = tl.load(
            conv_state
            + bank_row * conv_stride_row
            + offs_c * conv_stride_c
            + prior_col * conv_stride_l,
            mask=c_mask & source_edge_writer,
            other=0.0,
        )
        tl.store(
            source_stage
            + (stage_base + prior_col) * C
            + offs_c,
            prior_value,
            mask=c_mask & source_edge_writer,
        )
    tl.store(
        source_stage
        + (stage_base + SOURCE_ROWS - 1) * C
        + offs_c,
        0.0,
        mask=c_mask & source_edge_writer,
    )


_FR13_CONV_PREGATHER: dict = {}
_FR13_FIXED32_CONV_PREGATHER: dict = {}
_FR13_FIXED32_CONV_SSI_GROUPS: dict[tuple[str, ...], dict[str, object]] = {}
_FR13_FIXED32_BATCHES = (1, 2, 3, 4)
_FR13_FIXED32_CONV_COMMIT_ROUTE = "fixed32_direct_source_col0"


def _fixed32_conv_page_safe_row_span(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> int | None:
    """Return the dense logical row span when it fits in one cache page."""
    if (
        len(shape) != 3
        or len(stride) != 3
        or any(value <= 0 for value in shape)
        or any(value <= 0 for value in stride)
    ):
        return None
    channels, state_length = shape[1:]
    channel_stride, state_stride = stride[1:]
    if (channel_stride, state_stride) not in (
        (state_length, 1),
        (1, channels),
    ):
        return None
    logical_row_span = (
        (channels - 1) * channel_stride
        + (state_length - 1) * state_stride
        + 1
    )
    if stride[0] < logical_row_span:
        return None
    return logical_row_span


def register_fixed32_conv_col0_ssi_group(
    *,
    layer_names,
    spec_state_indices: torch.Tensor,
    max_batch_size: int,
) -> None:
    """Register one builder-owned, replay-refreshed SSI tensor.

    Qwen3-Next exposes three GDN KV-cache groups. Each builder owns one
    full-capacity ``spec_state_indices_tensor`` and refreshes it before model
    forward/replay. Repeated builder construction replaces the same group
    registration until pregather preseed freezes the final three-group map.
    """
    if _FR13_FIXED32_MODE is None:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER SSI group registration during capture"
        )
    names = tuple(sorted(str(name) for name in layer_names))
    capacity = int(max_batch_size)
    if (
        not names
        or len(names) != len(set(names))
        or capacity not in _FR13_FIXED32_BATCHES
        or not torch.is_tensor(spec_state_indices)
        or spec_state_indices.device.type != "cuda"
        or spec_state_indices.dtype != torch.int32
        or spec_state_indices.ndim != 2
        or int(spec_state_indices.shape[0]) < capacity
        or int(spec_state_indices.shape[1]) < 1
        or any(int(value) <= 0 for value in spec_state_indices.stride())
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER invalid builder-owned SSI group: "
            f"layers={names!r} capacity={capacity} "
            f"shape={getattr(spec_state_indices, 'shape', None)!r}"
        )
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    prior = _FR13_FIXED32_CONV_SSI_GROUPS.get(names)
    if state is not None and (
        prior is None
        or int(prior["data_ptr"]) != int(spec_state_indices.data_ptr())
        or tuple(prior["stride"])
        != tuple(int(value) for value in spec_state_indices.stride())
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER builder SSI group changed after preseed"
        )
    for other_names in _FR13_FIXED32_CONV_SSI_GROUPS:
        if other_names != names and set(other_names).intersection(names):
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER layer moved between SSI groups"
            )
    _FR13_FIXED32_CONV_SSI_GROUPS[names] = {
        "layers": names,
        "tensor": spec_state_indices,
        "capacity": capacity,
        "shape": tuple(int(value) for value in spec_state_indices.shape),
        "stride": tuple(int(value) for value in spec_state_indices.stride()),
        "data_ptr": int(spec_state_indices.data_ptr()),
    }


def _validate_fixed32_conv_pregather_preseed(
    *,
    conv_banks,
    ssm_banks,
    layer_order,
    max_batch_size: int,
) -> tuple[
    torch.Tensor,
    list[int],
    int,
    int,
    int,
    tuple[torch.Tensor, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Validate and materialize the fixed32 pregather pointer contract once."""
    if _FR13_FIXED32_MODE is None:
        raise RuntimeError(
            "fixed32 conv pregather preseed requires FR13_FIXED32_MODE"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER preseed called inside capture"
        )
    if not isinstance(conv_banks, (list, tuple)) or len(conv_banks) != 48:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires exactly 48 conv banks"
        )
    if not isinstance(ssm_banks, (list, tuple)) or len(ssm_banks) != 48:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires exactly 48 companion SSM banks"
        )
    capacity = int(max_batch_size)
    if capacity not in _FR13_FIXED32_BATCHES:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER max_batch_size must be 1..4, "
            f"got {capacity}"
        )
    order = tuple(str(name) for name in layer_order)
    groups = tuple(_FR13_FIXED32_CONV_SSI_GROUPS.values())
    registered_layers = {
        name: group["tensor"]
        for group in groups
        for name in group["layers"]
    }
    registered_group_ids = {
        name: group_index
        for group_index, group in enumerate(groups)
        for name in group["layers"]
    }
    if (
        len(order) != 48
        or len(set(order)) != 48
        or len(groups) != 3
        or set(registered_layers) != set(order)
        or any(int(group["capacity"]) < capacity for group in groups)
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires exactly three builder SSI "
            f"groups covering the 48-layer order: groups={len(groups)} "
            f"registered={len(registered_layers)} order={len(order)}"
        )
    ssi_sources = tuple(registered_layers[name] for name in order)
    ssi_group_ids = tuple(registered_group_ids[name] for name in order)

    anchor = conv_banks[0]
    if not torch.is_tensor(anchor) or anchor.ndim != 3:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER bank[0] must be a 3D tensor"
        )
    if anchor.device.type != "cuda" or any(
        source.device != anchor.device for source in ssi_sources
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires one CUDA device, got "
            f"banks={anchor.device}"
        )
    element_bytes = anchor.element_size()
    if element_bytes <= 0 or 16 % element_bytes:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER element size must divide 16, got "
            f"{element_bytes}"
        )
    shape = tuple(int(value) for value in anchor.shape)
    stride = tuple(int(value) for value in anchor.stride())
    logical_row_span = _fixed32_conv_page_safe_row_span(shape, stride)
    if (
        shape[0] < capacity
        or logical_row_span is None
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires page-safe dense "
            f"logical rows, got shape={shape} stride={stride}"
        )
    pointers = []
    bank_spans = []
    ssm_pointers = []
    ssm_storage_pointers = []
    for index, bank in enumerate(conv_banks):
        if not torch.is_tensor(bank):
            raise TypeError(
                f"FR13_FIXED32_CONV_PREGATHER bank[{index}] is not a tensor"
            )
        if (
            bank.device != anchor.device
            or bank.dtype != anchor.dtype
            or tuple(int(value) for value in bank.shape) != shape
            or tuple(int(value) for value in bank.stride()) != stride
        ):
            raise ValueError(
                "FR13_FIXED32_CONV_PREGATHER bank contract mismatch at "
                f"index {index}"
            )
        pointer = int(bank.data_ptr())
        if pointer % 16:
            raise ValueError(
                "FR13_FIXED32_CONV_PREGATHER bank pointer is not 16-byte "
                f"aligned at index {index}: {pointer:#x}"
            )
        if any(int(value) <= 0 for value in bank.stride()):
            raise ValueError(
                "FR13_FIXED32_CONV_PREGATHER bank strides must be positive at "
                f"index {index}: {tuple(int(value) for value in bank.stride())}"
            )
        pointers.append(pointer)
        bank_spans.append(
            (
                pointer,
                pointer
                + (
                    (shape[0] - 1) * stride[0]
                    + logical_row_span
                )
                * element_bytes,
                index,
            )
        )
        ssm_bank = ssm_banks[index]
        if (
            not torch.is_tensor(ssm_bank)
            or ssm_bank.device != anchor.device
            or int(bank.untyped_storage().data_ptr())
            != int(ssm_bank.untyped_storage().data_ptr())
        ):
            raise ValueError(
                "FR13_FIXED32_CONV_PREGATHER conv/SSM storage binding mismatch "
                f"at index {index}"
            )
        ssm_pointers.append(int(ssm_bank.data_ptr()))
        ssm_storage_pointers.append(
            int(ssm_bank.untyped_storage().data_ptr())
        )

    indices_by_pointer: dict[int, list[int]] = {}
    span_by_pointer: dict[int, tuple[int, int, int]] = {}
    for lo, hi, index in bank_spans:
        indices_by_pointer.setdefault(lo, []).append(index)
        span_by_pointer.setdefault(lo, (lo, hi, index))
    ordered_unique_spans = sorted(span_by_pointer.values())
    for left, right in zip(
        ordered_unique_spans, ordered_unique_spans[1:]
    ):
        if right[0] < left[1]:
            raise ValueError(
                "FR13_FIXED32_CONV_PREGATHER conv bank spans partially overlap: "
                f"bank[{left[2]}]=[{left[0]:#x},{left[1]:#x}) "
                f"bank[{right[2]}]=[{right[0]:#x},{right[1]:#x})"
            )

    alias_classes = tuple(
        sorted(
            (tuple(indices) for indices in indices_by_pointer.values()),
            key=lambda indices: indices[0],
        )
    )
    if (
        len(alias_classes) != 16
        or any(len(indices) != 3 for indices in alias_classes)
        or any(
            len({ssi_group_ids[index] for index in indices}) != 3
            for indices in alias_classes
        )
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER requires 16 exact conv aliases of "
            "width 3, each spanning all three SSI groups"
        )
    ssm_indices_by_pointer: dict[int, list[int]] = {}
    for index, pointer in enumerate(ssm_pointers):
        ssm_indices_by_pointer.setdefault(pointer, []).append(index)
    ssm_alias_classes = tuple(
        sorted(
            (tuple(indices) for indices in ssm_indices_by_pointer.values()),
            key=lambda indices: indices[0],
        )
    )
    if ssm_alias_classes != alias_classes:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER conv/SSM alias topology mismatch"
        )

    alias_ids_by_layer = [-1] * 48
    alias_ranks_by_layer = [-1] * 48
    for alias_id, indices in enumerate(alias_classes):
        for alias_rank, index in enumerate(indices):
            alias_ids_by_layer[index] = alias_id
            alias_ranks_by_layer[index] = alias_rank
    alias_ids = tuple(alias_ids_by_layer)
    alias_ranks = tuple(alias_ranks_by_layer)
    if (
        set(alias_ids) != set(range(16))
        or set(alias_ranks) != {0, 1, 2}
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER alias topology materialization failed"
        )
    return (
        anchor,
        pointers,
        shape[1] * shape[2],
        shape[2],
        element_bytes,
        ssi_sources,
        alias_classes,
        alias_ids,
        alias_ranks,
        tuple(ssm_pointers),
        tuple(ssm_storage_pointers),
    )


def preseed_fixed32_conv_col0_pregather(
    *,
    conv_banks,
    ssm_banks,
    layer_order,
    max_batch_size: int,
    commit_spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    commit_source_stagings,
    commit_state_src: torch.Tensor,
    source_rows_per_batch: int,
) -> dict[str, object]:
    """Build the pointer table and warm B1 through the server capacity.

    This is the only fixed32 entry allowed to inspect conv-bank ``data_ptr``
    values. Call it after all 48 banks and the three builder-owned persistent
    SSI groups are registered, but before any measured request can run.
    """
    capacity = int(max_batch_size)
    batches = tuple(range(1, capacity + 1))
    (
        anchor,
        pointers,
        row_elems,
        conv_l,
        element_bytes,
        ssi_sources,
        alias_classes,
        alias_ids,
        alias_ranks,
        ssm_pointers,
        ssm_storage_pointers,
    ) = (
        _validate_fixed32_conv_pregather_preseed(
            conv_banks=conv_banks,
            ssm_banks=ssm_banks,
            layer_order=layer_order,
            max_batch_size=capacity,
        )
    )
    existing = _FR13_FIXED32_CONV_PREGATHER.get("state")
    bank_refs = tuple(conv_banks)
    ssm_bank_refs = tuple(ssm_banks)
    source_refs = tuple(commit_source_stagings)
    source_rows = int(source_rows_per_batch)
    conv_c = int(anchor.shape[1])
    if (
        len(source_refs) != 48
        or len({id(source) for source in source_refs}) != 48
        or len({int(source.data_ptr()) for source in source_refs}) != 48
        or source_rows <= 0
        or not torch.is_tensor(commit_state_src)
        or commit_state_src.device != anchor.device
        or commit_state_src.dtype != torch.int64
        or commit_state_src.ndim != 1
        or int(commit_state_src.numel()) != 32 * conv_l
        or not commit_state_src.is_contiguous()
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_DIRECT invalid source/state-src geometry"
        )
    for index, source in enumerate(source_refs):
        if (
            not torch.is_tensor(source)
            or source.device != anchor.device
            or source.dtype != anchor.dtype
            or source.ndim != 2
            or int(source.shape[0]) < capacity * source_rows
            or int(source.shape[1]) != conv_c
            or tuple(int(value) for value in source.stride()) != (conv_c, 1)
            or not source.is_contiguous()
            or int(source.data_ptr()) % 16
        ):
            raise ValueError(
                "FR13_FIXED32_CONV_DIRECT source contract mismatch at "
                f"index {index}"
            )
    state_src_values = tuple(
        int(value) for value in commit_state_src.detach().cpu().tolist()
    )
    if (
        min(state_src_values) < 0
        or max(state_src_values) >= source_rows
        or max(state_src_values) != source_rows - 1
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_DIRECT state-src range does not match source rows"
        )
    zero_tail = _fr13_fixed32_conv_commit_zero_tail_requested()
    zero_tail_byte_ab = (
        _fr13_fixed32_conv_commit_zero_tail_byte_ab_requested()
    )
    if zero_tail and zero_tail_byte_ab:
        raise RuntimeError(
            "FR13 fixed32 conv zero-tail production and byte A/B are exclusive"
        )
    live_state_cols = 3
    if zero_tail or zero_tail_byte_ab:
        state_src_rows = tuple(
            state_src_values[row * conv_l : (row + 1) * conv_l]
            for row in range(32)
        )
        if (
            _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES
            or anchor.dtype != torch.bfloat16
            or conv_c != 10240
            or conv_l != 34
            or source_rows != 36
            or state_src_values != _FR13_FIXED32_TREECONV_STATE_SRC
            or len(state_src_rows) != 32
            or any(
                value != source_rows - 1
                for row in state_src_rows
                for value in row[live_state_cols:]
            )
        ):
            raise RuntimeError(
                "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL exact physical32 "
                "BF16 C10240/L34/source36 contract drifted"
            )
    if (
        not torch.is_tensor(commit_spec_state_indices)
        or commit_spec_state_indices.device != anchor.device
        or commit_spec_state_indices.dtype != torch.int32
        or commit_spec_state_indices.ndim != 3
        or int(commit_spec_state_indices.shape[0]) != 48
        or int(commit_spec_state_indices.shape[1]) < capacity
        or int(commit_spec_state_indices.shape[2]) != 32
        or not commit_spec_state_indices.is_contiguous()
        or not torch.is_tensor(accepted_paths)
        or accepted_paths.device != anchor.device
        or accepted_paths.dtype != torch.int32
        or accepted_paths.ndim != 2
        or int(accepted_paths.shape[0]) < capacity
        or int(accepted_paths.shape[1]) != 16
        or not accepted_paths.is_contiguous()
        or not torch.is_tensor(accepted_lens)
        or accepted_lens.device != anchor.device
        or accepted_lens.dtype != torch.int32
        or accepted_lens.ndim != 1
        or int(accepted_lens.shape[0]) < capacity
        or not accepted_lens.is_contiguous()
    ):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER invalid fixed commit operands: "
            f"ssi={getattr(commit_spec_state_indices, 'shape', None)!r} "
            f"paths={getattr(accepted_paths, 'shape', None)!r} "
            f"lens={getattr(accepted_lens, 'shape', None)!r}"
        )
    if existing is not None:
        if (
            existing["mode"] != _FR13_FIXED32_MODE
            or existing["max_batch_size"] != capacity
            or tuple(existing["layer_order"])
            != tuple(str(name) for name in layer_order)
            or len(existing["banks"]) != 48
            or any(
                old is not new
                for old, new in zip(existing["banks"], bank_refs, strict=True)
            )
            or any(
                old is not new
                for old, new in zip(
                    existing["ssm_banks"], ssm_bank_refs, strict=True
                )
            )
            or existing.get("commit_spec_state_indices")
            is not commit_spec_state_indices
            or existing.get("accepted_paths") is not accepted_paths
            or existing.get("accepted_lens") is not accepted_lens
            or len(existing.get("source_stagings", ())) != 48
            or any(
                old is not new
                for old, new in zip(
                    existing["source_stagings"], source_refs, strict=True
                )
            )
            or existing.get("source_rows_per_batch") != source_rows
            or existing.get("state_src_values") != state_src_values
            or existing.get("commit_zero_tail") is not zero_tail
            or existing.get("commit_zero_tail_byte_ab") is not zero_tail_byte_ab
        ):
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER was already preseeded with "
                "different persistent operands"
            )
        return dict(existing["contract"])

    offsets = torch.tensor(
        [(pointer - pointers[0]) // 16 for pointer in pointers],
        dtype=torch.int64,
        device=anchor.device,
    )
    source_pointers = tuple(int(source.data_ptr()) for source in source_refs)
    source_offsets = torch.tensor(
        [
            (pointer - source_pointers[0]) // 16
            for pointer in source_pointers
        ],
        dtype=torch.int64,
        device=anchor.device,
    )
    direct_state_src = commit_state_src.detach().clone().contiguous()
    alias_ids_device = torch.tensor(
        alias_ids,
        dtype=torch.int64,
        device=anchor.device,
    )
    alias_peer_layers = tuple(
        tuple(int(peer) for peer in alias_classes[alias_id])
        for alias_id in alias_ids
    )
    alias_peer_layers_device = torch.tensor(
        alias_peer_layers,
        dtype=torch.int32,
        device=anchor.device,
    )
    ssi_ptrs = torch.tensor(
        [int(source.data_ptr()) for source in ssi_sources],
        dtype=torch.int64,
        device=anchor.device,
    )
    ssi_strides = torch.tensor(
        [int(source.stride(0)) for source in ssi_sources],
        dtype=torch.int64,
        device=anchor.device,
    )
    staging = torch.empty(
        48,
        capacity,
        row_elems,
        dtype=anchor.dtype,
        device=anchor.device,
    )
    row_guard_flags = {
        batch: torch.empty(
            48 * batch,
            dtype=torch.bool,
            device=anchor.device,
        )
        for batch in batches
    }
    zero_tail_count_enable = torch.zeros(
        (), dtype=torch.int64, device=anchor.device
    )
    zero_tail_compared_events = torch.zeros(
        (), dtype=torch.int64, device=anchor.device
    )
    zero_tail_differing_bytes = torch.zeros(
        (), dtype=torch.int64, device=anchor.device
    )
    expected_staging_stride = (capacity * row_elems, row_elems, 1)
    if (
        not staging.is_contiguous()
        or tuple(int(value) for value in staging.stride())
        != expected_staging_stride
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER staging layout drifted: "
            f"shape={tuple(staging.shape)} stride={tuple(staging.stride())}"
        )
    staging_lo = int(staging.data_ptr())
    staging_hi = staging_lo + int(staging.numel()) * staging.element_size()
    for index, bank in enumerate(bank_refs):
        bank_lo = int(bank.data_ptr())
        bank_hi = bank_lo + (
            sum(
                (int(size) - 1) * int(bank.stride(dim))
                for dim, size in enumerate(bank.shape)
            )
            + 1
        ) * bank.element_size()
        if staging_lo < bank_hi and bank_lo < staging_hi:
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER staging aliases bank "
                f"{index}: staging=[{staging_lo:#x},{staging_hi:#x}) "
                f"bank=[{bank_lo:#x},{bank_hi:#x})"
            )
    block = 1024
    commit_lease_token = object()
    state = {
        "mode": _FR13_FIXED32_MODE,
        "max_batch_size": capacity,
        "preseeded_batches": batches,
        "banks": bank_refs,
        "ssm_banks": ssm_bank_refs,
        "layer_order": tuple(str(name) for name in layer_order),
        "anchor": anchor,
        "bank_shape": tuple(int(value) for value in anchor.shape),
        "bank_stride": tuple(int(value) for value in anchor.stride()),
        "bank_data_ptrs": tuple(pointers),
        "ssm_bank_data_ptrs": ssm_pointers,
        "ssm_bank_storage_ptrs": ssm_storage_pointers,
        "bank_alias_classes": alias_classes,
        "bank_alias_ids": alias_ids,
        "bank_alias_ranks": alias_ranks,
        "bank_alias_ids_device": alias_ids_device,
        "bank_alias_peer_layers": alias_peer_layers,
        "bank_alias_peer_layers_device": alias_peer_layers_device,
        "off16": offsets,
        "ssi_sources": ssi_sources,
        "ssi_ptrs": ssi_ptrs,
        "ssi_strides": ssi_strides,
        "commit_spec_state_indices": commit_spec_state_indices,
        "accepted_paths": accepted_paths,
        "accepted_lens": accepted_lens,
        "source_stagings": source_refs,
        "source_anchor": source_refs[0],
        "source_off16": source_offsets,
        "direct_source_data_ptrs": source_pointers,
        "direct_source_storage_ptrs": tuple(
            int(source.untyped_storage().data_ptr()) for source in source_refs
        ),
        "source_shapes": tuple(
            tuple(int(value) for value in source.shape)
            for source in source_refs
        ),
        "source_strides": tuple(
            tuple(int(value) for value in source.stride())
            for source in source_refs
        ),
        "source_rows_per_batch": source_rows,
        "state_src": direct_state_src,
        "state_src_values": state_src_values,
        "commit_zero_tail": zero_tail,
        "commit_zero_tail_byte_ab": zero_tail_byte_ab,
        "commit_live_state_cols": live_state_cols,
        "state_src_digest": hashlib.sha256(
            _fr13_fixed32_treeconv_canonical_json(state_src_values)
        ).hexdigest(),
        "treeconv_topology_descriptor": (
            _fr13_fixed32_treeconv_topology_descriptor(_FR13_FIXED32_MODE)
        ),
        "treeconv_zero_tail_count_enable": zero_tail_count_enable,
        "treeconv_zero_tail_compared_events": zero_tail_compared_events,
        "treeconv_zero_tail_differing_bytes": zero_tail_differing_bytes,
        "treeconv_zero_tail_replay_active": False,
        "staging": staging,
        "row_guard_flags_by_batch": row_guard_flags,
        "row_elems": row_elems,
        "conv_c": conv_c,
        "conv_l": conv_l,
        "element_bytes": element_bytes,
        "block": block,
        "token": None,
        "n": 0,
        "stages": 0,
        "stages_by_batch": {batch: 0 for batch in batches},
        "graph_capture_stages": 0,
        "graph_capture_stages_by_batch": {
            batch: 0 for batch in batches
        },
        "profile_capture_stages": 0,
        "aux_capture_stages": 0,
        "commit_gather_launches": 0,
        "commit_scatter_launches": 0,
        "commit_direct_launches": 0,
        "commit_gather_launches_by_batch": {
            batch: 0 for batch in batches
        },
        "commit_scatter_launches_by_batch": {
            batch: 0 for batch in batches
        },
        "commit_direct_launches_by_batch": {
            batch: 0 for batch in batches
        },
        "live_selfchecked_batches": set(),
        "source_identity": (
            tuple(id(bank) for bank in bank_refs),
            tuple(id(bank) for bank in ssm_bank_refs),
            id(offsets),
            id(alias_ids_device),
            id(alias_peer_layers_device),
            tuple(id(source) for source in ssi_sources),
            id(ssi_ptrs),
            id(ssi_strides),
            id(commit_spec_state_indices),
            id(accepted_paths),
            id(accepted_lens),
            tuple(id(source) for source in source_refs),
            id(source_offsets),
            id(direct_state_src),
            id(zero_tail_count_enable),
            id(zero_tail_compared_events),
            id(zero_tail_differing_bytes),
            id(staging),
            tuple(id(row_guard_flags[batch]) for batch in batches),
        ),
        "source_data_ptrs": (
            tuple(int(bank.data_ptr()) for bank in bank_refs),
            tuple(int(bank.data_ptr()) for bank in ssm_bank_refs),
            tuple(
                int(bank.untyped_storage().data_ptr())
                for bank in ssm_bank_refs
            ),
            int(offsets.data_ptr()),
            int(alias_ids_device.data_ptr()),
            int(alias_peer_layers_device.data_ptr()),
            tuple(int(source.data_ptr()) for source in ssi_sources),
            int(ssi_ptrs.data_ptr()),
            int(ssi_strides.data_ptr()),
            int(commit_spec_state_indices.data_ptr()),
            int(accepted_paths.data_ptr()),
            int(accepted_lens.data_ptr()),
            source_pointers,
            int(source_offsets.data_ptr()),
            int(direct_state_src.data_ptr()),
            int(zero_tail_count_enable.data_ptr()),
            int(zero_tail_compared_events.data_ptr()),
            int(zero_tail_differing_bytes.data_ptr()),
            int(staging.data_ptr()),
            tuple(
                int(row_guard_flags[batch].data_ptr()) for batch in batches
            ),
        ),
        "commit_operand_meta": (
            tuple(int(value) for value in commit_spec_state_indices.shape),
            tuple(int(value) for value in commit_spec_state_indices.stride()),
            tuple(int(value) for value in accepted_paths.shape),
            tuple(int(value) for value in accepted_paths.stride()),
            tuple(int(value) for value in accepted_lens.shape),
            tuple(int(value) for value in accepted_lens.stride()),
        ),
        "commit_operand_data_ptrs": (
            int(commit_spec_state_indices.data_ptr()),
            int(accepted_paths.data_ptr()),
            int(accepted_lens.data_ptr()),
        ),
        "commit_lease_token": commit_lease_token,
        "contract": {
            "mode": _FR13_FIXED32_MODE,
            "route": "in_graph_preconsume",
            "block": block,
            "layers": 48,
            "pointer_entries": 48,
            "ssi_pointer_entries": 48,
            "ssi_groups": 3,
            "preseeded_batches": batches,
            "staging_capacity": capacity,
            "row_elems": row_elems,
            "staging_bank_nonalias": True,
            "commit_route": _FR13_FIXED32_CONV_COMMIT_ROUTE,
            "commit_launches_per_event": 1,
            "commit_direct_launches_per_event": 1,
            "commit_gather_launches_per_event": 0,
            "commit_scatter_launches_per_event": 0,
            "commit_staging_reused": False,
            "commit_source_staging_reused": True,
            "commit_source_pointer_entries": 48,
            "commit_source_rows_per_batch": source_rows,
            "commit_state_src_shape": (32, conv_l),
            "commit_zero_tail": zero_tail,
            "commit_zero_tail_byte_ab": zero_tail_byte_ab,
            "commit_source_columns_loaded_per_row": (
                live_state_cols if zero_tail else conv_l
            ),
            "commit_destination_columns_stored_per_row": conv_l,
            "commit_row_guard_route": (
                "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
            ),
            "commit_row_guard_kernel_launches_per_event": 1,
            "commit_row_guard_programs_per_request": 48,
            "commit_row_guard_physical_rows": 32,
            "commit_row_guard_path_capacity": 16,
            "commit_row_guard_alias_width": 3,
            "commit_row_guard_compare_capacity": 16,
            "commit_row_guard_path_validation_programs_per_request": 1,
            "commit_row_guard_path_vector_loads_per_request": 1,
            "commit_row_guard_alias_validation_programs_per_event": 1,
            "commit_row_guard_alias_vector_loads_per_event": 2,
            "commit_row_guard_selected_row_loads_per_program": 0,
            "commit_row_guard_peer_topology_proof": "preseed_lease_audit",
            "commit_row_guard_torch_index_transforms": 0,
            "commit_row_guard_async_scalar_reductions": 1,
            "commit_row_guard_async_assertions": 1,
            "commit_full_node_writebacks": 0,
            "commit_conv_remaps": 0,
            "commit_bank_overlap_policy": "exact_alias_only_16x3",
            "commit_bank_partial_overlap": False,
            "commit_bank_alias_groups": 16,
            "commit_bank_alias_width": 3,
            "commit_bank_destination_guard": "alias_row_unique",
            "commit_null_row_rejected": True,
            "commit_ssi_bound": True,
            "commit_paths_bound": True,
        },
    }
    if int(anchor.shape[0]) <= 3 * capacity:
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER needs one non-null warmup row per "
            f"alias/request: rows={int(anchor.shape[0])} capacity={capacity}"
        )
    # Warm every B specialization from independent, non-null physical rows.
    # Builder rows above the current eager B may be uninitialized at boot.
    safe_ssi = torch.ones(
        (capacity, 1), dtype=torch.int32, device=anchor.device
    )
    safe_ptrs = torch.full(
        (48,), int(safe_ssi.data_ptr()), dtype=torch.int64, device=anchor.device
    )
    safe_strides = torch.full(
        (48,), int(safe_ssi.stride(0)), dtype=torch.int64, device=anchor.device
    )
    safe_commit_rows = (
        torch.tensor(
            alias_ranks,
            dtype=torch.int32,
            device=anchor.device,
        ).view(48, 1)
        * capacity
        + torch.arange(
            capacity,
            dtype=torch.int32,
            device=anchor.device,
        ).view(1, capacity)
        + 1
    )
    safe_commit_ssi = (
        safe_commit_rows.view(48, capacity, 1)
        .expand(48, capacity, int(commit_spec_state_indices.shape[2]))
        .contiguous()
    )
    safe_paths = torch.zeros(
        (capacity, int(accepted_paths.shape[1])),
        dtype=torch.int32,
        device=anchor.device,
    )
    safe_lens = torch.zeros(
        (capacity,), dtype=torch.int32, device=anchor.device
    )
    state["warmup_operands"] = (safe_ssi, safe_ptrs, safe_strides)
    state["commit_warmup_operands"] = (
        safe_commit_ssi,
        safe_paths,
        safe_lens,
    )
    safe_rows_long = safe_commit_rows.to(torch.long)
    saved_warm_rows = tuple(
        bank.index_select(0, safe_rows_long[layer]).clone()
        for layer, bank in enumerate(bank_refs)
    )
    try:
        for batch in batches:
            validate_fixed32_conv_commit_rows(
                spec_state_indices=safe_commit_ssi,
                accepted_paths=safe_paths,
                accepted_lens=safe_lens,
                bank_alias_ids=alias_ids_device,
                bank_alias_peer_layers=alias_peer_layers_device,
                guard_flags=row_guard_flags[batch],
                batch=batch,
                bank_rows=int(anchor.shape[0]),
            )
            pregather_grid = (48, batch, triton.cdiv(row_elems, block))
            _fr13_conv_col0_pregather_kernel[pregather_grid](
                anchor,
                offsets,
                safe_ptrs,
                safe_strides,
                staging,
                staging.stride(0),
                staging.stride(1),
                int(anchor.stride(0)),
                int(anchor.stride(1)),
                int(anchor.stride(2)),
                ROW_ELEMS=row_elems,
                CONV_L=conv_l,
                ELEM_BYTES=element_bytes,
                B=batch,
                BLOCK=block,
            )
            direct_grid = (48, batch, triton.cdiv(conv_c, block))
            _fr13_fixed32_conv_direct_col0_kernel[direct_grid](
                anchor,
                offsets,
                source_refs[0],
                source_offsets,
                direct_state_src,
                safe_commit_ssi,
                safe_paths,
                safe_lens,
                safe_commit_ssi.stride(0),
                safe_commit_ssi.stride(1),
                safe_commit_ssi.stride(2),
                safe_paths.stride(0),
                safe_paths.stride(1),
                safe_lens.stride(0),
                int(anchor.stride(0)),
                int(anchor.stride(1)),
                int(anchor.stride(2)),
                int(source_refs[0].stride(0)),
                int(source_refs[0].stride(1)),
                CONV_C=conv_c,
                CONV_L=conv_l,
                SOURCE_ROWS=source_rows,
                ELEM_BYTES=element_bytes,
                SPEC_COLS=int(safe_commit_ssi.shape[2]),
                PATH_COLS=int(safe_paths.shape[1]),
                B=batch,
                BLOCK_C=block,
                ZERO_TAIL=zero_tail,
                LIVE_STATE_COLS=live_state_cols,
                num_warps=4,
            )
    finally:
        for layer, (bank, saved) in enumerate(
            zip(bank_refs, saved_warm_rows, strict=True)
        ):
            bank.index_copy_(0, safe_rows_long[layer], saved)
    _FR13_FIXED32_CONV_PREGATHER["state"] = state
    _FR13_FIXED32_CONV_PREGATHER["commit_lease_token"] = commit_lease_token
    print(
        "[FR13_FIXED32_CONV_PREGATHER] preseeded: "
        f"layers=48 pointer_entries=48 batches={list(batches)}",
        flush=True,
    )
    return dict(state["contract"])


def selfcheck_fixed32_conv_col0_ssi_sources(
    *, num_spec_decodes: int
) -> None:
    """Boot-only value check of the live pointer table for one batch."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER live SSI selfcheck during capture"
        )
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        return
    batch = int(num_spec_decodes)
    if not 1 <= batch <= int(state["max_batch_size"]):
        raise RuntimeError(
            f"FR13_FIXED32_CONV_PREGATHER selfcheck invalid B={batch}"
        )
    checked = state["live_selfchecked_batches"]
    if batch in checked:
        return
    sources = state["ssi_sources"]
    for pointer in dict.fromkeys(int(source.data_ptr()) for source in sources):
        layers = [
            index
            for index, source in enumerate(sources)
            if int(source.data_ptr()) == pointer
        ]
        source = sources[layers[0]]
        rows = source[:batch, 0]
        row_min = int(rows.min().item())
        row_max = int(rows.max().item())
        bank_rows = min(int(state["banks"][index].shape[0]) for index in layers)
        if row_min < 0 or row_max >= bank_rows:
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER live SSI row is out of bounds: "
                f"B={batch} layers={layers!r} min={row_min} max={row_max} "
                f"bank_rows={bank_rows}"
            )
    row_elems = int(state["row_elems"])
    block = int(state["block"])
    grid = (48, batch, triton.cdiv(row_elems, block))
    _fr13_conv_col0_pregather_kernel[grid](
        state["anchor"],
        state["off16"],
        state["ssi_ptrs"],
        state["ssi_strides"],
        state["staging"],
        state["staging"].stride(0),
        state["staging"].stride(1),
        int(state["anchor"].stride(0)),
        int(state["anchor"].stride(1)),
        int(state["anchor"].stride(2)),
        ROW_ELEMS=row_elems,
        CONV_L=int(state["conv_l"]),
        ELEM_BYTES=int(state["element_bytes"]),
        B=batch,
        BLOCK=block,
    )
    for layer, (bank, source) in enumerate(
        zip(state["banks"], sources, strict=True)
    ):
        expected = bank.index_select(
            0, source[:batch, 0].to(torch.long)
        ).reshape(batch, row_elems)
        if not torch.equal(state["staging"][layer, :batch], expected):
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER live SSI byte selfcheck failed "
                f"for B={batch} layer={layer}"
            )
    checked.add(batch)


def validate_fixed32_conv_col0_ssi_source(
    *,
    layer_name: str,
    layer_index: int,
    spec_state_indices: torch.Tensor,
    num_spec_decodes: int,
) -> None:
    """Validate the live forward view against the builder-owned source map."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    index = int(layer_index)
    batch = int(num_spec_decodes)
    if (
        state is None
        or not torch.cuda.is_current_stream_capturing()
        or batch not in state["live_selfchecked_batches"]
        or not 0 <= index < 48
        or str(layer_name) != state["layer_order"][index]
        or not 1 <= batch <= int(state["max_batch_size"])
        or not torch.is_tensor(spec_state_indices)
        or spec_state_indices.dtype != torch.int32
        or spec_state_indices.device != state["anchor"].device
        or spec_state_indices.ndim != 2
        or int(spec_state_indices.shape[0]) < batch
        or int(spec_state_indices.shape[1]) < 1
        or int(spec_state_indices.data_ptr())
        != int(state["ssi_sources"][index].data_ptr())
        or tuple(int(value) for value in spec_state_indices.stride())
        != tuple(int(value) for value in state["ssi_sources"][index].stride())
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER live SSI source mapping drift: "
            f"layer={layer_name!r} index={index} B={batch}"
        )


def _fr13_fixed32_conv_source_flat_expected(width: int = 4) -> tuple[int, ...]:
    """Return the exact fixed32 window-gather descriptor in row-major order."""
    parent = tuple(int(value) for value in _FR13_FIXED32_PARENT)
    rows: list[int] = []
    for node in range(len(parent)):
        path = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        source = list(range(width - 1)) + [
            width - 1 + path_node for path_node in path
        ]
        rows.extend(source[-width:])
    return tuple(rows)


def launch_fixed32_sfwd_state_fusion(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    source_flat: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    source_stage: torch.Tensor,
    batch_size: int,
    tree_rows: int,
) -> dict[str, object]:
    """Launch the default-off one-kernel fixed32 conv/state candidate.

    This entrypoint is deliberately byte-gate-only for the first source
    revision. It refuses CUDA graph capture, validates the topology descriptor
    on the host, and leaves selection/serving to the caller. A later production
    selector must be bound to a real-task pass artifact rather than enabling
    this function implicitly.
    """
    if _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES:
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION requires an exact fixed32 mode"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION source candidate is eager "
            "byte-gate-only; graph production is not qualified"
        )
    batch = int(batch_size)
    rows = int(tree_rows)
    if conv_state.ndim != 3:
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION conv_state must be [bank,C,L]"
        )
    channels = int(conv_state.shape[1])
    state_len = int(conv_state.shape[2])
    width = int(conv_weights.shape[1]) if conv_weights.ndim == 2 else -1
    contract = fixed32_sfwd_state_fusion_contract(
        batch,
        tree_rows=rows,
        conv_width=width,
        conv_state_len=state_len,
    )
    required_rows = batch * rows
    source_rows_per_batch = int(contract["source_rows_per_request"])
    required_source_rows = batch * source_rows_per_batch
    tensors = (x, conv_state, spec_state_indices, source_flat,
               conv_weights, out, source_stage)
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError(
            "FR13_FIXED32_SFWD_STATE_FUSION operands must all be tensors"
        )
    device = x.device
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION operands must share one CUDA device"
        )
    if x.dtype != torch.bfloat16 or any(
        tensor.dtype != torch.bfloat16
        for tensor in (conv_state, conv_weights, out, source_stage)
    ):
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION preserves the exact BF16 conv "
            "contract; dtype changes are forbidden"
        )
    if bias is not None and (
        not torch.is_tensor(bias)
        or bias.device != device
        or bias.dtype not in (torch.bfloat16, torch.float32)
        or bias.ndim != 1
        or int(bias.numel()) != channels
    ):
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION bias must be BF16/FP32 [C] or None"
        )
    geometry_failures = []
    if x.ndim != 2:
        geometry_failures.append("x_ndim")
    if tuple(int(value) for value in x.shape) != (required_rows, channels):
        geometry_failures.append("x_shape")
    if out.shape != x.shape:
        geometry_failures.append("out_shape")
    if conv_weights.shape != (channels, width):
        geometry_failures.append("conv_weights_shape")
    if spec_state_indices.ndim != 2:
        geometry_failures.append("spec_state_indices_ndim")
    if (
        spec_state_indices.ndim < 1
        or int(spec_state_indices.shape[0]) < batch
    ):
        geometry_failures.append("spec_state_indices_batch")
    if (
        spec_state_indices.ndim < 2
        or int(spec_state_indices.shape[1]) < 1
    ):
        geometry_failures.append("spec_state_indices_width")
    if spec_state_indices.dtype != torch.int32:
        geometry_failures.append("spec_state_indices_dtype")
    if source_flat.ndim != 1:
        geometry_failures.append("source_flat_ndim")
    if source_flat.numel() != rows * width:
        geometry_failures.append("source_flat_numel")
    if source_flat.dtype not in (torch.int32, torch.int64):
        geometry_failures.append("source_flat_dtype")
    if source_stage.ndim != 2:
        geometry_failures.append("source_stage_ndim")
    if source_stage.ndim < 1 or int(source_stage.shape[0]) < required_source_rows:
        geometry_failures.append("source_stage_rows")
    if source_stage.ndim < 2 or int(source_stage.shape[1]) != channels:
        geometry_failures.append("source_stage_channels")
    if x.ndim == 2 and int(x.stride(1)) != 1:
        geometry_failures.append("x_channel_stride")
    if x.ndim == 2 and int(x.stride(0)) < channels:
        geometry_failures.append("x_row_stride")
    if not out.is_contiguous():
        geometry_failures.append("out_contiguous")
    if not source_flat.is_contiguous():
        geometry_failures.append("source_flat_contiguous")
    if not source_stage.is_contiguous():
        geometry_failures.append("source_stage_contiguous")
    if geometry_failures:
        observed = {
            "batch": batch,
            "tree_rows": rows,
            "channels": channels,
            "required_rows": required_rows,
            "required_source_rows": required_source_rows,
            "x": (tuple(x.shape), tuple(x.stride()), str(x.dtype)),
            "out": (tuple(out.shape), tuple(out.stride()), str(out.dtype)),
            "conv_state": (
                tuple(conv_state.shape),
                tuple(conv_state.stride()),
                str(conv_state.dtype),
            ),
            "conv_weights": (
                tuple(conv_weights.shape),
                tuple(conv_weights.stride()),
                str(conv_weights.dtype),
            ),
            "spec_state_indices": (
                tuple(spec_state_indices.shape),
                tuple(spec_state_indices.stride()),
                str(spec_state_indices.dtype),
            ),
            "source_flat": (
                tuple(source_flat.shape),
                tuple(source_flat.stride()),
                str(source_flat.dtype),
            ),
            "source_stage": (
                tuple(source_stage.shape),
                tuple(source_stage.stride()),
                str(source_stage.dtype),
            ),
        }
        raise ValueError(
            "FR13_FIXED32_SFWD_STATE_FUSION operand geometry/layout drift: "
            f"failed={geometry_failures!r}; observed={observed!r}"
        )
    actual_source_flat = tuple(
        int(value) for value in source_flat.detach().cpu().tolist()
    )
    if actual_source_flat != _fr13_fixed32_conv_source_flat_expected(width):
        raise RuntimeError(
            "FR13_FIXED32_SFWD_STATE_FUSION fixed32 source descriptor drift"
        )

    block_c = 256
    grid = (required_rows, triton.cdiv(channels, block_c))
    bias_arg = bias if bias is not None else x
    _fr13_fixed32_sfwd_state_fusion_kernel[grid](
        x,
        conv_state,
        spec_state_indices,
        source_flat,
        conv_weights,
        bias_arg,
        out,
        source_stage,
        int(x.stride(0)),
        int(conv_state.stride(0)),
        int(conv_state.stride(1)),
        int(conv_state.stride(2)),
        int(spec_state_indices.stride(0)),
        int(spec_state_indices.stride(1)),
        int(conv_weights.stride(0)),
        int(conv_weights.stride(1)),
        B=batch,
        N=rows,
        C=channels,
        WIDTH=width,
        STATE_LEN=state_len,
        SOURCE_ROWS=source_rows_per_batch,
        HAS_BIAS=bias is not None,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return contract


def launch_fixed32_conv_col0_pregather(
    *,
    num_spec_decodes: int,
    req_ids_token=None,
    graph_capture: bool = False,
) -> None:
    """Capture the one fixed pregather launch at verifier-graph entry."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER missing all-B warmup preseed"
        )
    if state["mode"] != _FR13_FIXED32_MODE:
        raise RuntimeError("FR13_FIXED32_CONV_PREGATHER mode drift")
    if type(graph_capture) is not bool or graph_capture is not True:
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER host stages are forbidden; "
            "the fixed route is captured in the final FULL verifier graph"
        )
    if not torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER graph stage ran outside CUDA capture"
        )
    if req_ids_token is not None:
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER in-graph route is tokenless"
        )
    batch = int(num_spec_decodes)
    if not 1 <= batch <= int(state["max_batch_size"]):
        raise ValueError(
            "FR13_FIXED32_CONV_PREGATHER batch exceeds preseeded server "
            f"capacity: B={batch} capacity={state['max_batch_size']}"
        )
    if (
        state.get("source_identity")
        != (
            tuple(id(bank) for bank in state["banks"]),
            tuple(id(bank) for bank in state["ssm_banks"]),
            id(state["off16"]),
            id(state["bank_alias_ids_device"]),
            id(state["bank_alias_peer_layers_device"]),
            tuple(id(source) for source in state["ssi_sources"]),
            id(state["ssi_ptrs"]),
            id(state["ssi_strides"]),
            id(state["commit_spec_state_indices"]),
            id(state["accepted_paths"]),
            id(state["accepted_lens"]),
            tuple(id(source) for source in state["source_stagings"]),
            id(state["source_off16"]),
            id(state["state_src"]),
            id(state["treeconv_zero_tail_count_enable"]),
            id(state["treeconv_zero_tail_compared_events"]),
            id(state["treeconv_zero_tail_differing_bytes"]),
            id(state["staging"]),
            tuple(
                id(state["row_guard_flags_by_batch"][batch])
                for batch in state["preseeded_batches"]
            ),
        )
        or state.get("source_data_ptrs")
        != (
            tuple(int(bank.data_ptr()) for bank in state["banks"]),
            tuple(int(bank.data_ptr()) for bank in state["ssm_banks"]),
            tuple(
                int(bank.untyped_storage().data_ptr())
                for bank in state["ssm_banks"]
            ),
            int(state["off16"].data_ptr()),
            int(state["bank_alias_ids_device"].data_ptr()),
            int(state["bank_alias_peer_layers_device"].data_ptr()),
            tuple(int(source.data_ptr()) for source in state["ssi_sources"]),
            int(state["ssi_ptrs"].data_ptr()),
            int(state["ssi_strides"].data_ptr()),
            int(state["commit_spec_state_indices"].data_ptr()),
            int(state["accepted_paths"].data_ptr()),
            int(state["accepted_lens"].data_ptr()),
            tuple(
                int(source.data_ptr())
                for source in state["source_stagings"]
            ),
            int(state["source_off16"].data_ptr()),
            int(state["state_src"].data_ptr()),
            int(state["treeconv_zero_tail_count_enable"].data_ptr()),
            int(state["treeconv_zero_tail_compared_events"].data_ptr()),
            int(state["treeconv_zero_tail_differing_bytes"].data_ptr()),
            int(state["staging"].data_ptr()),
            tuple(
                int(state["row_guard_flags_by_batch"][batch].data_ptr())
                for batch in state["preseeded_batches"]
            ),
        )
        or not state["staging"].is_contiguous()
        or tuple(int(value) for value in state["staging"].shape)
        != (48, int(state["max_batch_size"]), int(state["row_elems"]))
        or tuple(int(value) for value in state["staging"].stride())
        != (
            int(state["max_batch_size"]) * int(state["row_elems"]),
            int(state["row_elems"]),
            1,
        )
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER persistent operand identity drift"
        )
    block = int(state["block"])
    row_elems = int(state["row_elems"])
    grid = (48, batch, triton.cdiv(row_elems, block))
    _fr13_conv_col0_pregather_kernel[grid](
        state["anchor"],
        state["off16"],
        state["ssi_ptrs"],
        state["ssi_strides"],
        state["staging"],
        state["staging"].stride(0),
        state["staging"].stride(1),
        int(state["anchor"].stride(0)),
        int(state["anchor"].stride(1)),
        int(state["anchor"].stride(2)),
        ROW_ELEMS=row_elems,
        CONV_L=int(state["conv_l"]),
        ELEM_BYTES=int(state["element_bytes"]),
        B=batch,
        BLOCK=block,
    )
    state["graph_capture_stages"] += 1
    state["graph_capture_stages_by_batch"][batch] += 1


def audit_fixed32_conv_commit_lease() -> dict[str, object]:
    """Revalidate the full fixed operand lease outside the event hot path."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT missing all-B warmup preseed"
        )
    (
        anchor,
        pointers,
        row_elems,
        conv_l,
        element_bytes,
        ssi_sources,
        alias_classes,
        alias_ids,
        alias_ranks,
        ssm_pointers,
        ssm_storage_pointers,
    ) = _validate_fixed32_conv_pregather_preseed(
        conv_banks=state["banks"],
        ssm_banks=state["ssm_banks"],
        layer_order=state["layer_order"],
        max_batch_size=int(state["max_batch_size"]),
    )
    commit_spec_state_indices = state["commit_spec_state_indices"]
    accepted_paths = state["accepted_paths"]
    accepted_lens = state["accepted_lens"]
    operand_meta = (
        tuple(int(value) for value in commit_spec_state_indices.shape),
        tuple(int(value) for value in commit_spec_state_indices.stride()),
        tuple(int(value) for value in accepted_paths.shape),
        tuple(int(value) for value in accepted_paths.stride()),
        tuple(int(value) for value in accepted_lens.shape),
        tuple(int(value) for value in accepted_lens.stride()),
    )
    operand_data_ptrs = (
        int(commit_spec_state_indices.data_ptr()),
        int(accepted_paths.data_ptr()),
        int(accepted_lens.data_ptr()),
    )
    staging = state["staging"]
    staging_lo = int(staging.data_ptr())
    staging_hi = staging_lo + int(staging.numel()) * staging.element_size()
    staging_alias = any(
        staging_lo
        < pointer
        + (
            (int(bank.shape[0]) - 1) * int(bank.stride(0))
            + int(bank.shape[1]) * int(bank.shape[2])
        )
        * bank.element_size()
        and pointer < staging_hi
        for pointer, bank in zip(pointers, state["banks"], strict=True)
    )
    direct_sources = state.get("source_stagings", ())
    direct_state_src = state.get("state_src")
    direct_source_pointers = tuple(
        int(source.data_ptr()) for source in direct_sources
    )
    direct_source_storage_pointers = tuple(
        int(source.untyped_storage().data_ptr()) for source in direct_sources
    )
    direct_source_valid = (
        len(direct_sources) == 48
        and len(set(direct_source_pointers)) == 48
        and all(
            torch.is_tensor(source)
            and source.device == anchor.device
            and source.dtype == anchor.dtype
            and source.ndim == 2
            and int(source.shape[0])
            >= int(state["max_batch_size"])
            * int(state["source_rows_per_batch"])
            and int(source.shape[1]) == int(state["conv_c"])
            and tuple(int(value) for value in source.stride())
            == (int(state["conv_c"]), 1)
            and source.is_contiguous()
            for source in direct_sources
        )
        and torch.is_tensor(direct_state_src)
        and direct_state_src.device == anchor.device
        and direct_state_src.dtype == torch.int64
        and tuple(direct_state_src.shape) == (32 * int(state["conv_l"]),)
        and direct_state_src.is_contiguous()
    )
    if (
        _FR13_FIXED32_CONV_PREGATHER.get("commit_lease_token")
        is not state.get("commit_lease_token")
        or state["mode"] != _FR13_FIXED32_MODE
        or anchor is not state["anchor"]
        or tuple(pointers) != state.get("bank_data_ptrs")
        or tuple(ssm_pointers) != state.get("ssm_bank_data_ptrs")
        or tuple(ssm_storage_pointers)
        != state.get("ssm_bank_storage_ptrs")
        or tuple(alias_classes) != state.get("bank_alias_classes")
        or tuple(alias_ids) != state.get("bank_alias_ids")
        or tuple(alias_ranks) != state.get("bank_alias_ranks")
        or tuple(
            tuple(int(peer) for peer in alias_classes[alias_id])
            for alias_id in alias_ids
        )
        != state.get("bank_alias_peer_layers")
        or not torch.is_tensor(state.get("bank_alias_ids_device"))
        or state["bank_alias_ids_device"].dtype != torch.int64
        or state["bank_alias_ids_device"].device != anchor.device
        or tuple(state["bank_alias_ids_device"].shape) != (48,)
        or not state["bank_alias_ids_device"].is_contiguous()
        or tuple(
            int(value) for value in state["bank_alias_ids_device"].tolist()
        )
        != tuple(alias_ids)
        or not torch.is_tensor(state.get("bank_alias_peer_layers_device"))
        or state["bank_alias_peer_layers_device"].dtype != torch.int32
        or state["bank_alias_peer_layers_device"].device != anchor.device
        or tuple(state["bank_alias_peer_layers_device"].shape) != (48, 3)
        or not state["bank_alias_peer_layers_device"].is_contiguous()
        or tuple(
            tuple(int(peer) for peer in row)
            for row in state["bank_alias_peer_layers_device"].tolist()
        )
        != state.get("bank_alias_peer_layers")
        or any(
            source is not expected
            for source, expected in zip(
                ssi_sources, state["ssi_sources"], strict=True
            )
        )
        or int(row_elems) != int(state["row_elems"])
        or int(conv_l) != int(state["conv_l"])
        or int(element_bytes) != int(state["element_bytes"])
        or operand_meta != state.get("commit_operand_meta")
        or operand_data_ptrs != state.get("commit_operand_data_ptrs")
        or not direct_source_valid
        or direct_sources[0] is not state.get("source_anchor")
        or direct_source_pointers != state.get("direct_source_data_ptrs")
        or direct_source_storage_pointers
        != state.get("direct_source_storage_ptrs")
        or tuple(
            tuple(int(value) for value in source.shape)
            for source in direct_sources
        )
        != state.get("source_shapes")
        or tuple(
            tuple(int(value) for value in source.stride())
            for source in direct_sources
        )
        != state.get("source_strides")
        or tuple(int(value) for value in direct_state_src.cpu().tolist())
        != state.get("state_src_values")
        or tuple(int(value) for value in state["source_off16"].cpu().tolist())
        != tuple(
            (pointer - direct_source_pointers[0]) // 16
            for pointer in direct_source_pointers
        )
        or tuple(int(value) for value in staging.shape)
        != (
            48,
            int(state["max_batch_size"]),
            int(state["row_elems"]),
        )
        or tuple(int(value) for value in staging.stride())
        != (
            int(state["max_batch_size"]) * int(state["row_elems"]),
            int(state["row_elems"]),
            1,
        )
        or not staging.is_contiguous()
        or staging_alias
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT full preseed lease audit failed"
        )
    return {
        "lease_audited": True,
        "route": state["contract"]["commit_route"],
        "layers": 48,
        "bank_overlap_policy": "exact_alias_only_16x3",
        "bank_partial_overlap": False,
        "bank_alias_groups": 16,
        "bank_alias_width": 3,
        "bank_destination_guard": "alias_row_unique",
        "null_row_rejected": True,
        "staging_bank_nonalias": True,
        "source_staging_reused": True,
        "source_pointer_entries": 48,
        "state_src_shape": (32, int(state["conv_l"])),
        "row_guard_route": state["contract"]["commit_row_guard_route"],
        "row_guard_alias_width": state["contract"][
            "commit_row_guard_alias_width"
        ],
        "row_guard_compare_capacity": state["contract"][
            "commit_row_guard_compare_capacity"
        ],
        "row_guard_path_validation_programs_per_request": state["contract"][
            "commit_row_guard_path_validation_programs_per_request"
        ],
        "row_guard_path_vector_loads_per_request": state["contract"][
            "commit_row_guard_path_vector_loads_per_request"
        ],
        "row_guard_alias_validation_programs_per_event": state["contract"][
            "commit_row_guard_alias_validation_programs_per_event"
        ],
        "row_guard_alias_vector_loads_per_event": state["contract"][
            "commit_row_guard_alias_vector_loads_per_event"
        ],
        "row_guard_selected_row_loads_per_program": state["contract"][
            "commit_row_guard_selected_row_loads_per_program"
        ],
        "row_guard_peer_topology_proof": state["contract"][
            "commit_row_guard_peer_topology_proof"
        ],
    }


def launch_fixed32_conv_commit_to_col0(
    *,
    conv_banks,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    num_spec_decodes: int,
) -> None:
    """Commit fixed32 accepted source-stage leaves directly to running rows."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT missing all-B warmup preseed"
        )
    batch = int(num_spec_decodes)
    if not 1 <= batch <= int(state["max_batch_size"]):
        raise ValueError(
            "FR13_FIXED32_CONV_COMMIT batch exceeds preseeded server "
            f"capacity: B={batch} capacity={state['max_batch_size']}"
        )
    operand_meta = (
        tuple(int(value) for value in spec_state_indices.shape),
        tuple(int(value) for value in spec_state_indices.stride()),
        tuple(int(value) for value in accepted_paths.shape),
        tuple(int(value) for value in accepted_paths.stride()),
        tuple(int(value) for value in accepted_lens.shape),
        tuple(int(value) for value in accepted_lens.stride()),
    )
    operand_data_ptrs = (
        int(spec_state_indices.data_ptr()),
        int(accepted_paths.data_ptr()),
        int(accepted_lens.data_ptr()),
    )
    if (
        state["mode"] != _FR13_FIXED32_MODE
        or _FR13_FIXED32_CONV_PREGATHER.get("commit_lease_token")
        is not state.get("commit_lease_token")
        or state.get("contract", {}).get("commit_route")
        != _FR13_FIXED32_CONV_COMMIT_ROUTE
        or not isinstance(conv_banks, tuple)
        or conv_banks is not state["banks"]
        or len(conv_banks) != 48
        or conv_banks[0] is not state["anchor"]
        or spec_state_indices is not state["commit_spec_state_indices"]
        or accepted_paths is not state["accepted_paths"]
        or accepted_lens is not state["accepted_lens"]
        or state.get("commit_operand_meta") != operand_meta
        or state.get("commit_operand_data_ptrs") != operand_data_ptrs
        or spec_state_indices.device != state["anchor"].device
        or accepted_paths.device != state["anchor"].device
        or accepted_lens.device != state["anchor"].device
        or spec_state_indices.dtype != torch.int32
        or accepted_paths.dtype != torch.int32
        or accepted_lens.dtype != torch.int32
        or state.get("source_anchor") is not state["source_stagings"][0]
        or state["source_anchor"].device != state["anchor"].device
        or state["source_anchor"].dtype != state["anchor"].dtype
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT persistent identity/geometry drift"
        )
    metadata_fusion = _fr13_fixed32_committer_metadata_fusion_state(
        batch=batch,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
    )
    direct_metadata = _fr13_fixed32_committer_direct_metadata_state(
        batch=batch,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
    )
    if metadata_fusion is not None and direct_metadata is not None:
        raise RuntimeError(
            "FR13 fixed32 committer metadata routes are not exclusive"
        )
    if metadata_fusion is None and direct_metadata is None:
        committer_state = None
        validation_bank_rows = int(state["anchor"].shape[0])
    else:
        committer_state, validation_bank_rows = (
            metadata_fusion if metadata_fusion is not None else direct_metadata
        )
    sticky_guard = bool(
        direct_metadata is not None
        and committer_state.get("sticky_guard", False)
    )
    if sticky_guard:
        validate_fixed32_conv_commit_rows_sticky(
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            bank_alias_ids=state["bank_alias_ids_device"],
            bank_alias_peer_layers=state["bank_alias_peer_layers_device"],
            sticky_ok=committer_state["sticky_guard_ok"],
            batch=batch,
            bank_rows=validation_bank_rows,
        )
    else:
        validate_fixed32_conv_commit_rows(
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            bank_alias_ids=state["bank_alias_ids_device"],
            bank_alias_peer_layers=state["bank_alias_peer_layers_device"],
            guard_flags=state["row_guard_flags_by_batch"][batch],
            batch=batch,
            bank_rows=validation_bank_rows,
        )
    conv_c = int(state["conv_c"])
    block = int(state["block"])
    grid = (48, batch, triton.cdiv(conv_c, block))
    zero_tail_byte_ab = bool(state.get("commit_zero_tail_byte_ab", False))
    if zero_tail_byte_ab and committer_state is not None:
        raise RuntimeError(
            "FR13 fixed32 conv zero-tail byte A/B requires the stock metadata route"
        )

    def _launch_direct(*, zero_tail: bool) -> None:
        _fr13_fixed32_conv_direct_col0_kernel[grid](
            state["anchor"],
            state["off16"],
            state["source_anchor"],
            state["source_off16"],
            state["state_src"],
            spec_state_indices,
            accepted_paths,
            accepted_lens,
            spec_state_indices.stride(0),
            spec_state_indices.stride(1),
            spec_state_indices.stride(2),
            accepted_paths.stride(0),
            accepted_paths.stride(1),
            accepted_lens.stride(0),
            int(state["anchor"].stride(0)),
            int(state["anchor"].stride(1)),
            int(state["anchor"].stride(2)),
            int(state["source_anchor"].stride(0)),
            int(state["source_anchor"].stride(1)),
            CONV_C=conv_c,
            CONV_L=int(state["conv_l"]),
            SOURCE_ROWS=int(state["source_rows_per_batch"]),
            ELEM_BYTES=int(state["element_bytes"]),
            SPEC_COLS=int(spec_state_indices.shape[2]),
            PATH_COLS=int(accepted_paths.shape[1]),
            B=batch,
            BLOCK_C=block,
            ZERO_TAIL=zero_tail,
            LIVE_STATE_COLS=int(state["commit_live_state_cols"]),
            num_warps=4,
        )

    if zero_tail_byte_ab:
        _fr13_fixed32_treeconv_comparison_limit()
        _launch_direct(zero_tail=True)
        _fr13_fixed32_conv_zero_tail_compare_kernel[grid](
            state["anchor"],
            state["off16"],
            state["source_anchor"],
            state["source_off16"],
            state["state_src"],
            spec_state_indices,
            accepted_paths,
            accepted_lens,
            state["treeconv_zero_tail_count_enable"],
            state["treeconv_zero_tail_compared_events"],
            state["treeconv_zero_tail_differing_bytes"],
            spec_state_indices.stride(0),
            spec_state_indices.stride(1),
            spec_state_indices.stride(2),
            accepted_paths.stride(0),
            accepted_paths.stride(1),
            accepted_lens.stride(0),
            int(state["anchor"].stride(0)),
            int(state["anchor"].stride(1)),
            int(state["anchor"].stride(2)),
            int(state["source_anchor"].stride(0)),
            int(state["source_anchor"].stride(1)),
            CONV_C=conv_c,
            CONV_L=int(state["conv_l"]),
            SOURCE_ROWS=int(state["source_rows_per_batch"]),
            ELEM_BYTES=int(state["element_bytes"]),
            SPEC_COLS=int(spec_state_indices.shape[2]),
            PATH_COLS=int(accepted_paths.shape[1]),
            B=batch,
            BLOCK_C=block,
            num_warps=4,
        )
        _launch_direct(zero_tail=False)
    elif direct_metadata is not None:
        graph_paths = committer_state["direct_accepted_paths"]
        graph_lens = committer_state["direct_accepted_lens"]
        stream_key = _fr13_fixed32_committer_stream_key(accepted_paths.device)
        lease_key = _fr13_fixed32_committer_metadata_lease_key(
            batch=batch,
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            committer_paths=graph_paths,
            committer_lens=graph_lens,
            validation_bank_rows=validation_bank_rows,
            validation_guard=(
                committer_state["sticky_guard_ok"] if sticky_guard else None
            ),
            stream_key=stream_key,
        )
        if _FR13_FIXED32_COMMITTER_METADATA_LEASE:
            raise RuntimeError(
                "FR13 fixed32 direct-metadata prior lease was not consumed"
            )
        _launch_direct(zero_tail=bool(state["commit_zero_tail"]))
        _fr13_fixed32_committer_publish_direct_metadata_lease(lease_key)
        _FR13_FIXED32_COMMITTER_COUNTERS[
            "direct_metadata_published_by_batch"
        ][batch] += 1
    elif committer_state is None:
        _launch_direct(zero_tail=bool(state["commit_zero_tail"]))
    else:
        destination_paths = committer_state["accepted_paths"]
        destination_lens = committer_state["accepted_lens"]
        stream_key = _fr13_fixed32_committer_stream_key(accepted_paths.device)
        lease_key = _fr13_fixed32_committer_metadata_lease_key(
            batch=batch,
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            committer_paths=destination_paths,
            committer_lens=destination_lens,
            validation_bank_rows=validation_bank_rows,
            stream_key=stream_key,
        )
        if _FR13_FIXED32_COMMITTER_METADATA_LEASE:
            raise RuntimeError(
                "FR13 fixed32 metadata-fusion prior lease was not consumed"
            )
        _fr13_fixed32_conv_direct_col0_metadata_kernel[grid](
            state["anchor"],
            state["off16"],
            state["source_anchor"],
            state["source_off16"],
            state["state_src"],
            spec_state_indices,
            accepted_paths,
            accepted_lens,
            destination_paths,
            destination_lens,
            spec_state_indices.stride(0),
            spec_state_indices.stride(1),
            spec_state_indices.stride(2),
            accepted_paths.stride(0),
            accepted_paths.stride(1),
            accepted_lens.stride(0),
            destination_paths.stride(0),
            destination_paths.stride(1),
            destination_lens.stride(0),
            int(state["anchor"].stride(0)),
            int(state["anchor"].stride(1)),
            int(state["anchor"].stride(2)),
            int(state["source_anchor"].stride(0)),
            int(state["source_anchor"].stride(1)),
            CONV_C=conv_c,
            CONV_L=int(state["conv_l"]),
            SOURCE_ROWS=int(state["source_rows_per_batch"]),
            ELEM_BYTES=int(state["element_bytes"]),
            SPEC_COLS=int(spec_state_indices.shape[2]),
            PATH_COLS=int(accepted_paths.shape[1]),
            B=batch,
            BLOCK_C=block,
            ZERO_TAIL=bool(state["commit_zero_tail"]),
            LIVE_STATE_COLS=int(state["commit_live_state_cols"]),
            num_warps=4,
        )
        _fr13_fixed32_committer_publish_metadata_lease(lease_key)
        _FR13_FIXED32_COMMITTER_COUNTERS[
            "metadata_fusion_published_by_batch"
        ][batch] += 1
    state["commit_direct_launches"] += 1
    state["commit_direct_launches_by_batch"][batch] += 1


def fixed32_conv_col0_commit_counters() -> dict[str, object]:
    """Return actual fixed32 direct conv commit facts."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        return {
            "preseeded": False,
            "route": _FR13_FIXED32_CONV_COMMIT_ROUTE,
            "direct_launches": 0,
            "gather_launches": 0,
            "scatter_launches": 0,
            "direct_launches_by_batch": {
                batch: 0 for batch in _FR13_FIXED32_BATCHES
            },
            "gather_launches_by_batch": {
                batch: 0 for batch in _FR13_FIXED32_BATCHES
            },
            "scatter_launches_by_batch": {
                batch: 0 for batch in _FR13_FIXED32_BATCHES
            },
        }
    return {
        "preseeded": True,
        "route": state["contract"]["commit_route"],
        "direct_launches": int(state["commit_direct_launches"]),
        "gather_launches": int(state["commit_gather_launches"]),
        "scatter_launches": int(state["commit_scatter_launches"]),
        "direct_launches_by_batch": {
            batch: int(
                state["commit_direct_launches_by_batch"].get(batch, 0)
            )
            for batch in _FR13_FIXED32_BATCHES
        },
        "gather_launches_by_batch": {
            batch: int(
                state["commit_gather_launches_by_batch"].get(batch, 0)
            )
            for batch in _FR13_FIXED32_BATCHES
        },
        "scatter_launches_by_batch": {
            batch: int(
                state["commit_scatter_launches_by_batch"].get(batch, 0)
            )
            for batch in _FR13_FIXED32_BATCHES
        },
    }


def fixed32_conv_col0_pregather_counters() -> dict[str, object]:
    """Return fixed pregather preseed and actual-launch facts."""
    state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if state is None:
        return {
            "preseeded": False,
            "pointer_entries": 0,
            "preseeded_batches": (),
            "max_batch_size": 0,
            "actual_stages": 0,
            "actual_stages_by_batch": {
                batch: 0 for batch in _FR13_FIXED32_BATCHES
            },
            "graph_capture_stages": 0,
            "graph_capture_stages_by_batch": {
                batch: 0 for batch in _FR13_FIXED32_BATCHES
            },
            "profile_capture_stages": 0,
            "aux_capture_stages": 0,
        }
    return {
        "preseeded": True,
        "pointer_entries": 48,
        "preseeded_batches": tuple(state["preseeded_batches"]),
        "max_batch_size": int(state["max_batch_size"]),
        "actual_stages": int(state["stages"]),
        "actual_stages_by_batch": {
            batch: int(state["stages_by_batch"].get(batch, 0))
            for batch in _FR13_FIXED32_BATCHES
        },
        "graph_capture_stages": int(state["graph_capture_stages"]),
        "graph_capture_stages_by_batch": {
            batch: int(state["graph_capture_stages_by_batch"].get(batch, 0))
            for batch in _FR13_FIXED32_BATCHES
        },
        "profile_capture_stages": int(state["profile_capture_stages"]),
        "aux_capture_stages": int(state["aux_capture_stages"]),
    }


def launch_conv_col0_pregather(
    *, conv_banks: list, ssi_stack: torch.Tensor, num_spec_decodes: int,
    req_ids_token,
) -> None:
    """FR13_CONV_PREGATHER: one launch stages EVERY layer's conv col0 row.

    Replaces the per-layer NPR ``torch.index_select(conv_state, 0, col0)``
    (48 launches + glue/gaps = part of the measured verify soup) with a single
    pointer-table kernel run at COMMIT time (post conv-writeback => staging is
    post-commit truth). Consumption is guarded HOST-SIDE by req_ids_token
    equality (composition-change steps fall back to the legacy gather) -- no
    device syncs, fail-safe by construction. Values byte-identical to the
    per-layer gather (pure copy).
    """
    if _FR13_FIXED32_MODE is not None:
        raise RuntimeError(
            "FR13_FIXED32_CONV_PREGATHER post-commit/host trigger is forbidden; "
            "the final FULL verifier graph owns the one pre-consume stage"
        )
    L = len(conv_banks)
    b0 = conv_banks[0]
    dev = b0.device
    # LOGICAL conv elements only (C * L_conv), NOT stride(0): the conv view is
    # as_strided over a shared mamba page (stride(0) = whole-page ~2M elems);
    # consumers .view(B, C, L_conv) the staged rows.
    conv_c = int(b0.size(1))
    conv_l = int(b0.size(2))
    row_elems = conv_c * conv_l
    st = _FR13_CONV_PREGATHER
    ptrs = [int(b.data_ptr()) for b in conv_banks]
    if st.get("ptrs") != ptrs:
        for b in conv_banks:
            if (
                b.dtype != b0.dtype
                or b.stride() != b0.stride()
                or b.shape[1:] != b0.shape[1:]
            ):
                raise RuntimeError("FR13_CONV_PREGATHER: bank dtype/stride/shape mismatch")
            if (int(b.data_ptr()) - ptrs[0]) % 16 != 0:
                raise RuntimeError("FR13_CONV_PREGATHER: bank ptr not 16B-aligned vs anchor")
        st["ptrs"] = ptrs
        st["off16"] = torch.tensor(
            [(p - ptrs[0]) // 16 for p in ptrs], dtype=torch.int64, device=dev
        )
        st["staging"] = torch.empty(
            L, int(ssi_stack.shape[1]), row_elems, dtype=b0.dtype, device=dev
        )
        st["anchor"] = b0
    ssi_ptr_values = [
        int(ssi_stack[layer].data_ptr()) for layer in range(L)
    ]
    ssi_stride_values = [int(ssi_stack.stride(1)) for _ in range(L)]
    if (
        st.get("ssi_ptr_values") != ssi_ptr_values
        or st.get("ssi_stride_values") != ssi_stride_values
    ):
        st["ssi_ptr_values"] = ssi_ptr_values
        st["ssi_stride_values"] = ssi_stride_values
        st["ssi_ptrs"] = torch.tensor(
            ssi_ptr_values, dtype=torch.int64, device=dev
        )
        st["ssi_strides"] = torch.tensor(
            ssi_stride_values, dtype=torch.int64, device=dev
        )
    ebytes = b0.element_size()
    if (16 % ebytes) != 0:
        raise RuntimeError("FR13_CONV_PREGATHER: element size must divide 16")
    B = int(num_spec_decodes)
    BLOCK = 1024
    grid = (L, B, triton.cdiv(row_elems, BLOCK))
    _fr13_conv_col0_pregather_kernel[grid](
        st["anchor"], st["off16"], st["ssi_ptrs"], st["ssi_strides"],
        st["staging"], st["staging"].stride(0), st["staging"].stride(1),
        int(b0.stride(0)),
        int(b0.stride(1)), int(b0.stride(2)),
        ROW_ELEMS=row_elems, CONV_L=conv_l, ELEM_BYTES=ebytes, B=B, BLOCK=BLOCK,
    )
    st["token"] = req_ids_token
    st["n"] = B
    st["_scnt"] = st.get("_scnt", 0) + 1
    if st["_scnt"] % 512 == 1:
        print(f"[FR13_CPG_STAGE] stages={st['_scnt']}", flush=True)


def conv_col0_staged(req_ids_token, layer_idx: int):
    """Return the staged [B, C, W]-flat view for layer_idx if fresh, else None.

    Engagement needle (observer-safe: host dict increment + 1-in-4096 print):
    a pregather arm where `served` stays 0 is VACUOUS (always falling back to
    the legacy gather) — its clean result proves nothing about the lever.
    """
    if _FR13_FIXED32_MODE is not None:
        st = _FR13_FIXED32_CONV_PREGATHER.get("state")
        if st is None:
            raise RuntimeError(
                "FR13_FIXED32_CONV_PREGATHER consumer ran before preseed"
            )
    else:
        st = _FR13_CONV_PREGATHER
    staged = st.get("token") if st else None
    ok = staged is not None and staged == req_ids_token
    c = st.setdefault("_cnt", [0, 0])  # [served, fallback]
    c[0 if ok else 1] += 1
    if (c[0] + c[1]) % 4096 == 1:
        if ok:
            why = "match"
        elif staged is None:
            why = "no-stage"
        elif not isinstance(staged, tuple) or not isinstance(req_ids_token, tuple) \
                or len(staged) != len(req_ids_token):
            why = f"shape staged={type(staged).__name__} offered={type(req_ids_token).__name__}"
        else:
            parts = []
            names = ("pairs", "seq")
            for i, (a, b) in enumerate(zip(staged, req_ids_token)):
                if a != b:
                    nm = names[i] if i < len(names) else str(i)
                    if nm == "seq":
                        parts.append(f"seq staged={a} offered={b}")
                    else:
                        parts.append(nm)
            why = "diff:" + ",".join(parts)
        print(
            f"[FR13_CPG_SERVE] served={c[0]} fallback={c[1]} last={why}",
            flush=True,
        )
    if not ok:
        return None
    return st["staging"][layer_idx, : st.get("n", 0)]


def launch_attn_kv_linear_remap(**kwargs):
    """Thin observer-effect-safe timing wrapper (FR13_KVREMAP_TIMER)."""
    _t = _fr13_span_begin("FR13_KVREMAP_TIMER")
    try:
        return _launch_attn_kv_linear_remap_impl(**kwargs)
    finally:
        _fr13_span_end("FR13_KVREMAP_TIMER", _t)


def _launch_attn_kv_linear_remap_impl(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
) -> int:
    """Re-linearize the full-attention KV cache for the committed tree path.

    FR13_ATTN_KV_REMAP. The attention analog of :func:`launch_tree_state_linear_remap`
    (which re-linearizes GDN ssm/conv state). During tree verify, node ``k`` writes
    its K/V to a flat/unique physical slot ``slot_mapping[qsl[b] + k]`` (node order).
    After accept, vLLM advances ``seq_len`` and the NEXT forward reads the accepted
    tokens at LINEAR positions ``[base .. base+acc-1]``. For a branching tree the
    accepted path (e.g. spine nodes ``[0,1,3,5,7]``) sits at NON-CONTIGUOUS flat slots,
    so linear position ``d`` would read node ``d``'s K/V (a sibling near-neighbor),
    NOT the accepted node's. This copies each accepted node's K/V from its verify slot
    -> the linear committed slot, per full-attn layer, so the next forward reads the
    true committed-path KV.

    Timing contract (caller MUST honour): call at STEP-N END, after the committer
    publishes ``accepted_paths``/``num_accepted_tokens`` and while ``slot_mapping`` is
    still THIS step's verify mapping, and BEFORE the drafter / step-(N+1) verify write
    (which would overwrite the non-accepted tail slots that hold deep accepted nodes,
    e.g. spine node 7 at flat slot base+7 with acc=5).

    Returns the number of FOREIGN (non-contiguous) rows copied per layer -- the
    engagement needle: 0 => the accepted paths were all contiguous (chain-class), so
    the remap was a no-op (a branching tree with real accepts MUST be > 0).
    """
    b = int(num_spec_decodes)
    if b <= 0 or not kv_caches:
        return 0
    device = slot_mapping.device
    path_cols = int(accepted_paths.shape[1])
    if path_cols <= 0:
        return 0
    # SAFETY: this maps spec row i -> batch request i (qsl[:b]); that is valid
    # ONLY when the first b batch requests are all full trees with a UNIFORM
    # verify span that exceeds every offset we index. A mixed prefill/decode
    # batch would put a spec row at a higher batch position and qsl[:b] would
    # index a foreign request -> wrong-slot copy: skip (safe no-op) instead.
    if int(query_start_loc.shape[0]) < b + 1:
        return 0
    qsl_full = query_start_loc[: b + 1].to(torch.long)
    spans = qsl_full[1:] - qsl_full[:-1]                                # [b]
    acc = num_accepted_tokens[:b].to(torch.long).view(-1, 1)           # [b,1]
    # m = 0-based accepted position (depth-1). accepted_paths values are the
    # +1-shifted published node ids == FLAT VERIFY ROWS (H3: node i -> flat row
    # i+1; the verify batch is [anchor@offset0, choices@offsets 1..num_spec]).
    # The next forward reads the depth-(m+1) committed token at the flat-unique
    # slot of verify offset (m+1); the accepted node's true K/V lives at verify
    # offset accepted_paths[b,m]. So copy src=offset accepted_paths[b,m] ->
    # dst=offset m+1. Contiguous (no-op) exactly when accepted_paths[b,m]==m+1.
    m_idx = torch.arange(path_cols, device=device).view(1, -1)          # [1,path]
    ap = accepted_paths[:b, :path_cols].to(torch.long)                  # [b,path] flat rows
    dst_off = m_idx + 1                                                 # [1,path]
    # FR13_SLOT_REORDER (edit 3/5): under the spine-first slot permutation the
    # verify slot_mapping row j holds the slot of canonical COLUMN pi_inv[j]
    # (sm_perm[j] == sm_flat[pi_inv[j]], so sm_perm[pi[k]] == sm_flat[k]).
    # SOURCE auto-threads: sm_perm[qsl + node] is still the accepted node's true
    # slot. DESTINATION must stay the FLAT/linear committed slot: index the
    # permuted map at pi[dst_off] to recover sm_flat[dst_off]. dst_pi=None (flag
    # OFF) preserves the shipped behavior bit-for-bit.
    if dst_pi is not None:
        dst_off = dst_pi.to(device=device, dtype=torch.long)[dst_off]   # [1,path]
    src_off = ap                                                        # [b,path]
    # Guard: the span is the VERIFY TOKEN COUNT (num_spec+1), NOT path_cols
    # (= max accepted path length). Require uniform spans AND span > the largest
    # offset indexed (max flat verify row and the max linear dst depth acc), so
    # qsl[i]+offset never crosses into request i+1.
    _max_off = torch.maximum(ap.max(), torch.maximum(acc.max(), dst_off.max()))
    if not (bool((spans == spans[0]).all()) and bool(spans[0] > _max_off)):
        return 0
    qsl = qsl_full[:b].view(-1, 1)                                     # [b,1]
    foreign = (m_idx < acc) & (ap != dst_off)                          # [b,path]
    if not bool(foreign.any()):
        return 0
    n_slots = int(slot_mapping.shape[0])
    dst_flat = (qsl + dst_off).clamp_(0, n_slots - 1)                   # [b,path]
    src_flat = (qsl + src_off).clamp_(0, n_slots - 1)                   # [b,path]
    sel = foreign.reshape(-1)
    dst_slot = slot_mapping.reshape(-1)[dst_flat.reshape(-1)][sel].to(torch.long)
    src_slot = slot_mapping.reshape(-1)[src_flat.reshape(-1)][sel].to(torch.long)
    n_foreign = int(dst_slot.numel())
    if n_foreign == 0:
        return 0
    for kv in kv_caches:
        if not torch.is_tensor(kv) or kv.dim() < 3 or int(kv.shape[0]) != 2:
            continue
        bs = int(kv.shape[2])
        # Advanced-indexing on the (block, offset) slot dims is stride-safe (works
        # for both NHD/HND cache layouts) and in-place. Gather ALL sources into a
        # temp first so overlapping src/dst (spine paths chain src cols [1..L] ->
        # dst cols [0..L-1]) permute exactly (mirrors the GDN gather-then-scatter).
        gathered = kv[:, src_slot // bs, src_slot % bs].clone()
        kv[:, dst_slot // bs, dst_slot % bs] = gathered
    return n_foreign


def launch_attn_kv_linear_remap_syncfree(**kwargs):
    """Thin observer-effect-safe timing wrapper (FR13_KVREMAP_TIMER)."""
    _t = _fr13_span_begin("FR13_KVREMAP_TIMER")
    try:
        return _launch_attn_kv_linear_remap_syncfree_impl(**kwargs)
    finally:
        _fr13_span_end("FR13_KVREMAP_TIMER", _t)


def _fr13_fixed32_device_assert(condition: torch.Tensor, message: str) -> None:
    """Enqueue a fail-closed scalar device assertion without a host readback."""
    if condition.numel() != 1:
        raise ValueError(
            "FR13_FIXED32: device assertion condition must be scalar, got "
            f"{tuple(condition.shape)}"
        )
    if not hasattr(torch, "_assert_async"):
        raise RuntimeError(
            "FR13_FIXED32 requires torch._assert_async for sync-free guards"
        )
    torch._assert_async(condition, message)


def _validate_fixed32_kv16_contract(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None,
    batch_indices: torch.Tensor | None,
    expected_cache_tensors: int = 16,
) -> tuple[int, int]:
    """Validate one fixed32 Bx16 cache-remap route."""
    b = int(num_spec_decodes)
    if b not in (1, 2, 3, 4):
        raise ValueError(f"FR13_FIXED32_KV_REMAP16 requires B=1..4, got {b}")
    if (
        not isinstance(kv_caches, (list, tuple))
        or len(kv_caches) != expected_cache_tensors
    ):
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 requires exactly "
            f"{expected_cache_tensors} cache tensors, "
            f"got {type(kv_caches).__name__} len="
            f"{len(kv_caches) if isinstance(kv_caches, (list, tuple)) else 'n/a'}"
        )
    if accepted_paths.ndim != 2 or tuple(accepted_paths.shape) != (b, 16):
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 requires accepted_paths shape "
            f"{(b, 16)}, got {tuple(accepted_paths.shape)}"
        )
    if num_accepted_tokens.ndim != 1 or num_accepted_tokens.numel() != b:
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 requires accepted lengths shape "
            f"{(b,)}, got {tuple(num_accepted_tokens.shape)}"
        )
    if query_start_loc.ndim != 1:
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 query_start_loc must be 1D, "
            f"got {tuple(query_start_loc.shape)}"
        )
    required_qsl = 2 if batch_indices is not None else b + 1
    if int(query_start_loc.shape[0]) < required_qsl:
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 query_start_loc is too short: "
            f"shape={tuple(query_start_loc.shape)} required={required_qsl}"
        )
    if batch_indices is not None and (
        batch_indices.ndim != 1
        or tuple(batch_indices.shape) != (b,)
    ):
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 batch_indices must have shape "
            f"{(b,)}, got {tuple(batch_indices.shape)}"
        )
    if slot_mapping.numel() <= 0:
        raise ValueError("FR13_FIXED32_KV_REMAP16 requires nonempty slot_mapping")
    if dst_pi is not None and dst_pi.numel() < 17:
        raise ValueError(
            "FR13_FIXED32_KV_REMAP16 dst_pi must cover offsets 0..16, "
            f"got {dst_pi.numel()}"
        )
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if batch_indices is not None and batch_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "FR13_FIXED32_KV_REMAP16 batch_indices must be int32/int64, "
            f"got {batch_indices.dtype}"
        )
    contract_tensors = [
        ("slot_mapping", slot_mapping),
        ("query_start_loc", query_start_loc),
        ("accepted_paths", accepted_paths),
        ("num_accepted_tokens", num_accepted_tokens),
    ]
    if batch_indices is not None:
        contract_tensors.append(("batch_indices", batch_indices))
    if dst_pi is not None:
        contract_tensors.append(("dst_pi", dst_pi))
    for label, tensor in contract_tensors:
        if tensor.dtype not in integer_dtypes:
            raise TypeError(
                f"FR13_FIXED32_KV_REMAP16 {label} must be integer, "
                f"got {tensor.dtype}"
            )
        if tensor.device != slot_mapping.device:
            raise ValueError(
                f"FR13_FIXED32_KV_REMAP16 {label} device "
                f"{tensor.device} != {slot_mapping.device}"
            )
    cache_capacity = None
    for index, kv in enumerate(kv_caches):
        if not torch.is_tensor(kv):
            raise TypeError(
                f"FR13_FIXED32_KV_REMAP16 cache[{index}] is not a tensor"
            )
        if kv.dim() < 3 or int(kv.shape[0]) != 2:
            raise ValueError(
                "FR13_FIXED32_KV_REMAP16 cache must have exactly two K/V "
                f"planes: cache[{index}] shape={tuple(kv.shape)}"
            )
        if int(kv.shape[1]) <= 0 or int(kv.shape[2]) <= 0:
            raise ValueError(
                f"FR13_FIXED32_KV_REMAP16 cache[{index}] has empty slot axes"
            )
        capacity = int(kv.shape[1]) * int(kv.shape[2])
        if cache_capacity is None:
            cache_capacity = capacity
        elif capacity != cache_capacity:
            raise ValueError(
                "FR13_FIXED32_KV_REMAP16 cache slot capacities differ: "
                f"cache[0]={cache_capacity} cache[{index}]={capacity}"
            )
        if kv.device != slot_mapping.device:
            raise ValueError(
                "FR13_FIXED32_KV_REMAP16 cache/slot device mismatch: "
                f"cache[{index}]={kv.device} slots={slot_mapping.device}"
            )
    assert cache_capacity is not None
    return b, cache_capacity


def _launch_attn_kv_linear_remap_syncfree_fixed16_impl(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
    batch_indices: torch.Tensor | None = None,
    expected_cache_tensors: int = 16,
) -> None:
    """Strict fixed32 cache walk over Bx16 source/destination pairs.

    Pure batches use the original prefix-contiguous qsl route. Mixed batches
    provide the ordered full-batch row index for each compact spec row. The
    public target and drafter routes fix the cache count at 16 and 1.
    """
    b, cache_capacity = _validate_fixed32_kv16_contract(
        kv_caches=kv_caches,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        accepted_paths=accepted_paths,
        num_accepted_tokens=num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
        dst_pi=dst_pi,
        batch_indices=batch_indices,
        expected_cache_tensors=expected_cache_tensors,
    )
    device = slot_mapping.device
    if batch_indices is None:
        qsl_full = query_start_loc[: b + 1].to(torch.long)
        qsl_starts = qsl_full[:-1]
        qsl_ends = qsl_full[1:]
        batch_indices_ok = None
    else:
        full_indices = batch_indices.to(torch.long)
        safe_indices = full_indices.clamp(0, query_start_loc.numel() - 2)
        qsl_starts = query_start_loc[safe_indices].to(torch.long)
        qsl_ends = query_start_loc[safe_indices + 1].to(torch.long)
        batch_indices_ok = (
            (full_indices >= 0)
            & (full_indices + 1 < query_start_loc.numel())
        ).all()
        if b > 1:
            batch_indices_ok = batch_indices_ok & (
                full_indices[1:] > full_indices[:-1]
            ).all()
    spans = qsl_ends - qsl_starts
    acc = num_accepted_tokens.to(torch.long).view(b, 1)
    m_idx = torch.arange(16, device=device).view(1, 16)
    ap = accepted_paths.to(device=device, dtype=torch.long)
    dst_off = m_idx + 1
    if dst_pi is not None:
        dst_off = dst_pi.to(device=device, dtype=torch.long)[dst_off]

    lens_ok = ((acc >= 0) & (acc <= 16)).all()
    active_pos = m_idx < acc
    paths_ok = ((~active_pos) | ((ap >= 1) & (ap < 32))).all()
    spans_ok = (spans == 32).all()
    destination_ok = ((dst_off >= 1) & (dst_off < 32)).all()
    mapping_shape_ok = (
        (qsl_starts >= 0)
        & (qsl_ends <= int(slot_mapping.numel()))
    ).all()
    contract_ok = (
        lens_ok
        & paths_ok
        & spans_ok
        & destination_ok
        & mapping_shape_ok
    )
    if batch_indices_ok is not None:
        contract_ok = contract_ok & batch_indices_ok
    _fr13_fixed32_device_assert(
        contract_ok,
        "FR13_FIXED32_KV_REMAP16 dynamic contract violation",
    )

    qsl = qsl_starts.view(b, 1)
    active = active_pos & (ap != dst_off)
    n_slots = int(slot_mapping.shape[0])
    dst_flat = (qsl + dst_off).clamp_(0, n_slots - 1).reshape(b * 16)
    src_flat = (qsl + ap).clamp_(0, n_slots - 1).reshape(b * 16)
    sm = slot_mapping.reshape(-1)
    dst_slot = sm[dst_flat].to(torch.long)
    src_slot = sm[src_flat].to(torch.long)
    active_flat = active.reshape(b * 16)
    _fr13_fixed32_device_assert(
        (
            (dst_slot >= 0)
            & (dst_slot < cache_capacity)
            & (src_slot >= 0)
            & (src_slot < cache_capacity)
        ).all(),
        "FR13_FIXED32_KV_REMAP16 slot_mapping exceeds cache capacity",
    )
    for kv in kv_caches:
        block_size = int(kv.shape[2])
        src_vals = kv[
            :, src_slot // block_size, src_slot % block_size
        ].clone()
        dst_prior = kv[
            :, dst_slot // block_size, dst_slot % block_size
        ].clone()
        mask = active_flat.view(
            1, b * 16, *([1] * (src_vals.dim() - 2))
        )
        kv[:, dst_slot // block_size, dst_slot % block_size] = torch.where(
            mask, src_vals, dst_prior
        )


def launch_attn_kv_linear_remap_syncfree_fixed16(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
    batch_indices: torch.Tensor | None = None,
) -> None:
    """Public strict KV16 entry point for fixed32 patcher wiring."""
    return _launch_attn_kv_linear_remap_syncfree_fixed16_impl(
        kv_caches=kv_caches,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        accepted_paths=accepted_paths,
        num_accepted_tokens=num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
        dst_pi=dst_pi,
        batch_indices=batch_indices,
    )


def launch_attn_kv_linear_remap_syncfree_fixed1_drafter(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    batch_indices: torch.Tensor | None = None,
) -> None:
    """Compact fresh MTP KV after its first-pass write through the flat map."""
    return _launch_attn_kv_linear_remap_syncfree_fixed16_impl(
        kv_caches=kv_caches,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        accepted_paths=accepted_paths,
        num_accepted_tokens=num_accepted_tokens,
        num_spec_decodes=num_spec_decodes,
        dst_pi=None,
        batch_indices=batch_indices,
        expected_cache_tensors=1,
    )


def _launch_attn_kv_linear_remap_syncfree_impl(
    *,
    kv_caches,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
) -> None:
    """Zero-host-sync variant of :func:`launch_attn_kv_linear_remap`.

    FR13_KV_REMAP_SYNCFREE (committer-overlap prerequisite, 2026-07-23). The
    legacy fn stalls the stream 3-4x per step (two ``bool()`` span guards, a
    ``bool(foreign.any())``, and a data-dependent boolean-mask compaction) --
    fatal for enqueueing the remap on a side stream under the drafter. This
    variant is enqueue-only with FIXED shapes:

    - every (request, path-position) pair writes its DESTINATION slot exactly
      once; the value is ``where(active, src_value, dst_prior_value)`` with
      both gathered BEFORE any write. Inert pairs (non-foreign, m>=acc, or a
      failed whole-call validity guard) become value-identical writes, so the
      final cache bytes equal the legacy fn's for every case, including its
      early-return cases. Destinations are distinct by construction (linear
      committed offsets per request) => no write races, deterministic.
    - the span-uniformity/overflow guard and the b+1 qsl-length guard fold
      into a device-side ``valid`` scalar that deactivates ALL pairs.
    - no engagement-needle return (the legacy fn stays the diagnostic arm).
    """
    if _FR13_FIXED32_MODE is not None:
        return launch_attn_kv_linear_remap_syncfree_fixed16(
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            accepted_paths=accepted_paths,
            num_accepted_tokens=num_accepted_tokens,
            num_spec_decodes=num_spec_decodes,
            dst_pi=dst_pi,
        )
    b = int(num_spec_decodes)
    if b <= 0 or not kv_caches:
        return
    if int(query_start_loc.shape[0]) < b + 1:
        return
    device = slot_mapping.device
    path_cols = int(accepted_paths.shape[1])
    if path_cols <= 0:
        return
    qsl_full = query_start_loc[: b + 1].to(torch.long)
    spans = qsl_full[1:] - qsl_full[:-1]                                # [b]
    acc = num_accepted_tokens[:b].to(torch.long).view(-1, 1)           # [b,1]
    m_idx = torch.arange(path_cols, device=device).view(1, -1)          # [1,path]
    ap = accepted_paths[:b, :path_cols].to(torch.long)                  # [b,path]
    dst_off = m_idx + 1                                                 # [1,path]
    if dst_pi is not None:
        dst_off = dst_pi.to(device=device, dtype=torch.long)[dst_off]   # [1,path]
    src_off = ap                                                        # [b,path]
    # whole-call validity as a DEVICE scalar (no host bool()): uniform spans
    # AND span > every indexed offset. Broadcast into the per-pair mask.
    _max_off = torch.maximum(ap.max(), torch.maximum(acc.max(), dst_off.max()))
    valid = (spans == spans[0]).all() & (spans[0] > _max_off)           # 0-dim bool
    qsl = qsl_full[:b].view(-1, 1)                                     # [b,1]
    active = valid & (m_idx < acc) & (ap != dst_off)                   # [b,path]
    n_slots = int(slot_mapping.shape[0])
    dst_flat = (qsl + dst_off).clamp_(0, n_slots - 1).reshape(-1)       # [b*path]
    src_flat = (qsl + src_off).clamp_(0, n_slots - 1).reshape(-1)       # [b*path]
    sm = slot_mapping.reshape(-1)
    dst_slot = sm[dst_flat].to(torch.long)
    src_slot = sm[src_flat].to(torch.long)
    active_flat = active.reshape(-1)
    apply_kv_remap_slots(
        kv_caches=kv_caches,
        dst_slot=dst_slot,
        src_slot=src_slot,
        active_flat=active_flat,
    )


def prepare_kv_remap_slots(
    *,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    dst_pi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Slot/index math of the sync-free remap, MAIN-STREAM half.

    FR13_COMMIT_OVERLAP split: this must be enqueued on the MAIN stream so its
    slot_mapping gathers are stream-ordered BEFORE the SLOT_REORDER restore
    mutates the mapping in-place at drafter start; the heavy per-layer cache
    walk (:func:`apply_kv_remap_slots`) can then run on a side stream reading
    only kv caches + these snapshot tensors. Zero host syncs. Returns None on
    the host-shape guards (b/path/qsl) where the legacy fn early-returns.
    """
    b = int(num_spec_decodes)
    if b <= 0:
        return None
    if int(query_start_loc.shape[0]) < b + 1:
        return None
    device = slot_mapping.device
    path_cols = int(accepted_paths.shape[1])
    if path_cols <= 0:
        return None
    qsl_full = query_start_loc[: b + 1].to(torch.long)
    spans = qsl_full[1:] - qsl_full[:-1]
    acc = num_accepted_tokens[:b].to(torch.long).view(-1, 1)
    m_idx = torch.arange(path_cols, device=device).view(1, -1)
    ap = accepted_paths[:b, :path_cols].to(torch.long)
    dst_off = m_idx + 1
    if dst_pi is not None:
        dst_off = dst_pi.to(device=device, dtype=torch.long)[dst_off]
    src_off = ap
    _max_off = torch.maximum(ap.max(), torch.maximum(acc.max(), dst_off.max()))
    valid = (spans == spans[0]).all() & (spans[0] > _max_off)
    qsl = qsl_full[:b].view(-1, 1)
    active = valid & (m_idx < acc) & (ap != dst_off)
    n_slots = int(slot_mapping.shape[0])
    dst_flat = (qsl + dst_off).clamp_(0, n_slots - 1).reshape(-1)
    src_flat = (qsl + src_off).clamp_(0, n_slots - 1).reshape(-1)
    sm = slot_mapping.reshape(-1)
    return sm[dst_flat].to(torch.long), sm[src_flat].to(torch.long), active.reshape(-1)


def apply_kv_remap_slots(
    *,
    kv_caches,
    dst_slot: torch.Tensor,
    src_slot: torch.Tensor,
    active_flat: torch.Tensor,
) -> None:
    """Cache-walk half of the sync-free remap (side-stream-safe: reads only
    kv caches + the prepared slot tensors; identity-safe distinct-destination
    writes; zero host syncs)."""
    for kv in kv_caches:
        if not torch.is_tensor(kv) or kv.dim() < 3 or int(kv.shape[0]) != 2:
            continue
        bs = int(kv.shape[2])
        src_vals = kv[:, src_slot // bs, src_slot % bs].clone()
        dst_prior = kv[:, dst_slot // bs, dst_slot % bs].clone()
        mask = active_flat.view(1, -1, *([1] * (src_vals.dim() - 2)))
        kv[:, dst_slot // bs, dst_slot % bs] = torch.where(
            mask, src_vals, dst_prior
        )


def gather_committed_path_conv_prior(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    num_spec_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the prior conv window from the COMMITTED path's accepted leaf.

    FR13_CONV_COMMITTED_PATH: the legacy prior-window read gathers
    ``spec_state_indices[:, accepted_len - 1]`` AFTER
    :func:`launch_tree_state_linear_remap` has permuted the bank in place —
    linear-column arithmetic that is only spine-valid by construction and
    depends on the remap having executed exactly. This helper instead reads
    the accepted path's LEAF NODE column (``accepted_paths[b, len - 1]``,
    node-indexed pre-remap layout). The per-node tree-conv write-back stores
    each node's window as the last (width - 1) taps of
    ``(prior ++ that node's root-path tokens)``, so the accepted leaf's bank
    row IS the committed token path's window by construction — valid for
    BRANCH winners ([0,2], [0,1,4]); for spine winners it is byte-identical
    to the legacy post-remap linear read (the leaf's source row is never a
    remap destination), which is the semantics-preserving license.

    Must be called BEFORE ``launch_tree_state_linear_remap`` mutates the
    bank. ``accepted_len == 0`` (no draft accepted) reads node column 0 (the
    committed root token's window), matching legacy. All ops are tensor ops
    with no host sync, so the gather is CUDA-graph safe.

    Returns ``(read_node_cols [B,1], bank_rows [B,1], prior_state_bank)``.
    """
    b = int(num_spec_decodes)
    spec_cols = int(spec_state_indices.size(-1))
    device = spec_state_indices.device
    if accepted_paths is None or num_accepted_tokens is None:
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    else:
        lens = num_accepted_tokens[:b].to(torch.long).view(-1, 1)
        path_cols = torch.clamp(
            lens - 1, min=0, max=int(accepted_paths.size(-1)) - 1
        )
        read_node_cols = accepted_paths[:b].to(torch.long).gather(1, path_cols)
        # len == 0 commits no draft node: the prior window is node 0's (the
        # committed root token's). The committer zero-fills path rows, but
        # enforce explicitly so a stale buffer cannot redirect the read.
        read_node_cols = torch.where(
            lens > 0, read_node_cols, torch.zeros_like(read_node_cols)
        )
        read_node_cols = torch.clamp(read_node_cols, min=0, max=spec_cols - 1)
    if os.environ.get("FR13_TREE_RUNROW_INIT", "1") == "1":
        # STATELESS-TREE: seed the next-step conv prior from col 0 (the running
        # row holding this-step's committed leaf, deposited by the post-accept
        # conv committer), not the accepted-leaf node column. Mirrors SSM h0_col=0.
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    bank_rows = spec_state_indices[:b].to(torch.long).gather(1, read_node_cols)
    prior_state_bank = torch.index_select(
        conv_state, 0, bank_rows.reshape(-1)
    )
    return read_node_cols, bank_rows, prior_state_bank


@triton.jit
def _gdn_node_step(
    state_i,
    b_q,
    b_k,
    b_v,
    b_b,
    b_g,
    b_raw_a,
    b_raw_b,
    b_a_log,
    b_dt_bias,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr = False,
):
    # FR13_REPLAY_ROUTE shared per-node update body. This is the SINGLE
    # source of the GDN rank-1 node update used by BOTH the tree scan kernel
    # (_tree_gdn_kernel) and the accepted-path replay kernel
    # (_tree_gdn_replay_kernel). Replay bit-exactness is by re-execution of
    # the identical fp32 instruction sequence on bit-identical inputs, so the
    # two kernels MUST inline this one body with identical constexprs
    # (DIM_K/BLOCK_V via operand shapes, OUTPUT_SCALE, USE_QK_L2NORM_IN_KERNEL,
    # RAW_GATING) and identical num_warps=8. Codegen identity across the two
    # compilations (FMA contraction/scheduling per unrolled instance) is NOT
    # spec-guaranteed: it is gated by the one-time byte A/B on captured
    # payloads (GPU-gated obligation; see FR13_REPLAY_ROUTE_BUILD.md).
    b_beta = b_b
    if RAW_GATING:
        x = b_raw_a + b_dt_bias
        softplus_x = tl.where(
            x <= 20.0,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        b_g = -tl.exp(b_a_log) * softplus_x
        # SEAM e (beta): native packed-decode is
        #   tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
        # (fused_recurrent.py:325 on the pinned image) -- a bf16 round-trip,
        # since b.dtype is bf16. Ours omits the round-trip (fp32-carry, MORE
        # precise). SCAN_ALIGN reproduces the native bf16 cast so the served
        # scan's beta matches the incumbent decode bit-for-bit.
        if SCAN_ALIGN:
            b_beta = tl.sigmoid(b_raw_b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        else:
            b_beta = tl.sigmoid(b_raw_b.to(tl.float32))
    if USE_QK_L2NORM_IN_KERNEL:
        # SEAM d (l2norm): native packed-decode is
        #   b_q = b_q / tl.sqrt(tl.sum(b_q*b_q) + 1e-6)
        # (fused_recurrent.py:314-315). Ours uses tl.rsqrt(...) -- the same
        # math but a DIFFERENT opcode (rcp+sqrt fused vs div), so the last bit
        # can differ. eps (1e-6) already matches. SCAN_ALIGN swaps to the
        # native div-by-sqrt form.
        if SCAN_ALIGN:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        else:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * OUTPUT_SCALE
    state_i *= tl.exp(b_g)
    b_v -= tl.sum(state_i * b_k[None, :], axis=1)
    b_v *= b_beta
    state_i += b_v[:, None] * b_k[None, :]
    out_i = tl.sum(state_i * b_q[None, :], axis=1)
    # SEAM K1 (per-node bf16 state round-trip): the native packed-decode ORACLE
    # (2) processes one token per kernel program -- it computes b_o from the
    # FP32 post-update b_h (fused_recurrent.py:331), then STORES the state as
    # bf16 to ht (fused_recurrent.py:336 -- ht.dtype is bf16) and the NEXT
    # token RELOADS that bf16 state -> fp32 (fused_recurrent.py:303). So the
    # state CARRIED forward to the next token is bf16-rounded while THIS token's
    # output is the precise fp32 value. Our default fp32-carry tree scan is MORE
    # precise but DISAGREES with (2); since we are scored against (2), SCAN_ALIGN
    # reproduces the store-boundary round-trip on the carried state ONLY (after
    # out_i is taken from the precise fp32 state_i, before state_i is returned to
    # be cached / fed to the next node). This is the single op with depth-growth
    # (per-node, fed recurrently) -- the diffuse-carrier lever. K1 is gated on
    # the SAME SCAN_ALIGN constexpr (MODE=body implies all body seams d/e/K1):
    # when SCAN_ALIGN is False the cast is dead code (no codegen change,
    # bug-class #10), so the default served path stays byte-identical.
    if SCAN_ALIGN:
        state_i = state_i.to(tl.bfloat16).to(tl.float32)
    return state_i, out_i


@triton.jit
def _gdn_node_step_precomputed_decay(
    state_i,
    b_q,
    b_k,
    b_v,
    b_beta,
    b_decay,
    OUTPUT_SCALE: tl.constexpr,
):
    """Run the exact raw-rsqrt recurrence from producer-owned scalars."""
    b_q = b_q * OUTPUT_SCALE
    state_i *= b_decay
    b_v -= tl.sum(state_i * b_k[None, :], axis=1)
    b_v *= b_beta
    state_i += b_v[:, None] * b_k[None, :]
    out_i = tl.sum(state_i * b_q[None, :], axis=1)
    return state_i, out_i


@triton.jit
def _tree_gdn_kernel(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    strict_mask,
    visible_mask,
    out,
    state,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    flags_ptr,
    N_ACTUAL: tl.constexpr,
    N_PAD: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr = False,
    N_LOOP: tl.constexpr = 0,
    PARENT_GATHER: tl.constexpr = False,
    PIGGYBACK_EXPORT: tl.constexpr = False,
    CHAIN_END_IDX: tl.constexpr = 0,
    RING_EXPORT: tl.constexpr = False,
    FLAGS_EXPORT: tl.constexpr = False,
    FLAGS_ROWS: tl.constexpr = 0,
    HC_MASK: tl.constexpr = 0,
    HC_ROWS: tl.constexpr = 0,
    HC_SLOTS_LO: tl.constexpr = 0,
    HC_SLOTS_HI: tl.constexpr = 0,
    hc_slot_map=None,
):
    # N_LOOP is the span (loop bound, offs_n lane count, h_cache rows) of the
    # scan/reduction. Default (0) means "use N_PAD" -- the per-tree padded span
    # of the locked served path. When FR13_NPAD_INVARIANT pins it to the fixed
    # N_FIXED (>= N_PAD), the reduction FMA order is canonical across tree sizes
    # while the strict_mask buffer keeps its true N_PAD width (mask reads are
    # guarded so lanes in [N_PAD, N_LOOP) read no ancestry and contribute 0.0;
    # they are also >= N_ACTUAL so they never load/store real data).
    N_SPAN: tl.constexpr = N_PAD if N_LOOP == 0 else N_LOOP
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_n = tl.arange(0, N_SPAN)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0),
        )

    h0_base = h0
    if H0_IS_BANK:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + H0_BATCH_INDEX).to(tl.int64) - 1,
                0,
            )
        h0_index = tl.load(h0_indices + H0_INDEX_ROW + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
    b_h0 = tl.load(
        h0_base + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Sequential rank-1 tree scan. Each row caches the post-token state for one
    # tree node, so children start from their parent's fp32 checkpoint without
    # reloading h0 or replaying ancestors from HBM.
    # FR13_HC_INTERNAL (HC_MASK != 0): keep rows ONLY for INTERNAL nodes -- an
    # ancestor is by definition a node with children, so the one-hot selects
    # below can never hit a leaf row. Slot map is trace-time (HC_SLOTS_LO/HI,
    # 4 bits per node); the footprint drops N_SPAN -> HC_ROWS fp32 tiles.
    # Default HC_MASK=0 => trace-time dead => the alloc is the exact locked
    # form (bug-class #10 constexpr-dead).
    if HC_MASK == 0:
        h_cache = tl.zeros((N_SPAN, BLOCK_V, DIM_K), dtype=tl.float32)
    else:
        h_cache = tl.zeros((HC_ROWS, BLOCK_V, DIM_K), dtype=tl.float32)
        offs_h = tl.arange(0, HC_ROWS)
    for i in tl.static_range(0, N_SPAN):
        state_i = b_h0
        if PARENT_GATHER:
            # FR13_PARENT_GATHER: the deployed else-branch below overwrites
            # state_i in increasing j, so its result is exactly the state of the
            # LARGEST-index ancestor = the immediate parent (topological order).
            # Locate that parent with the SAME guarded strict_mask reads but only
            # cheap INTEGER selects (no full-tile reduction), then do ONE gather.
            # Byte-identical (same one-hot masked tl.sum, same row, 0.0+x=x);
            # reductions/node drop from i to 1. i==0 (root) has no ancestor ->
            # state_i stays b_h0, matching the empty else-loop.
            if i > 0:
                parent_i = -1
                for j in tl.static_range(0, i):
                    if N_LOOP == 0:
                        anc_bit = tl.load(strict_mask + i * N_PAD + j)
                    else:
                        anc_bit = tl.load(
                            strict_mask + i * N_PAD + j,
                            mask=(i < N_PAD) & (j < N_PAD),
                            other=0,
                        )
                    is_anc = (anc_bit != 0) & (j < N_ACTUAL)
                    parent_i = tl.where(is_anc, j, parent_i)
                if HC_MASK == 0:
                    h_par = tl.sum(
                        tl.where((offs_n == parent_i)[:, None, None], h_cache, 0.0),
                        axis=0,
                    )
                else:
                    # FR13_HC_INTERNAL + PARENT_GATHER: runtime slot lookup
                    # (a parent is internal by definition, so the map entry
                    # is always a real slot; parent_i < 0 is masked to 0 and
                    # the result discarded by the where() below).
                    _par_slot = tl.load(
                        hc_slot_map + tl.maximum(parent_i, 0)
                    )
                    h_par = tl.sum(
                        tl.where((offs_h == _par_slot)[:, None, None], h_cache, 0.0),
                        axis=0,
                    )
                state_i = tl.where(parent_i >= 0, h_par, b_h0)
        else:
            for j in tl.static_range(0, i):
                # The strict_mask buffer is N_PAD x N_PAD. When the canonical-order
                # span exceeds N_PAD (N_LOOP != 0), lanes i,j in [N_PAD, N_SPAN)
                # must NOT read OOB: a guarded load returns 0 (no ancestry); those
                # lanes are also >= N_ACTUAL so they contribute exactly 0.0. When
                # N_LOOP == 0 (default served path) the span equals N_PAD, every
                # i,j < N_PAD, and the load is the EXACT prior unguarded form --
                # constexpr-dead guard => byte-identical codegen (bug-class #10).
                if HC_MASK == 0:
                    if N_LOOP == 0:
                        anc_bit = tl.load(strict_mask + i * N_PAD + j)
                    else:
                        anc_bit = tl.load(
                            strict_mask + i * N_PAD + j,
                            mask=(i < N_PAD) & (j < N_PAD),
                            other=0,
                        )
                    ancestor = (anc_bit != 0) & (j < N_ACTUAL)
                    h_j = tl.sum(
                        tl.where((offs_n == j)[:, None, None], h_cache, 0.0),
                        axis=0,
                    )
                    state_i = tl.where(ancestor, h_j, state_i)
                else:
                    # FR13_HC_INTERNAL: a leaf j can never satisfy `ancestor`
                    # (only nodes with children appear in strict ancestry), so
                    # its iteration is skipped at trace time -- the select it
                    # would compute is discarded at runtime in the locked form.
                    # Internal j reads its COMPACTED row through the identical
                    # one-hot primitive selecting the identical value.
                    if ((HC_MASK >> j) & 1) == 1:
                        if N_LOOP == 0:
                            anc_bit = tl.load(strict_mask + i * N_PAD + j)
                        else:
                            anc_bit = tl.load(
                                strict_mask + i * N_PAD + j,
                                mask=(i < N_PAD) & (j < N_PAD),
                                other=0,
                            )
                        ancestor = (anc_bit != 0) & (j < N_ACTUAL)
                        if j < 16:
                            h_j = tl.sum(
                                tl.where(
                                    (offs_h == ((HC_SLOTS_LO >> (4 * j)) & 15))[:, None, None],
                                    h_cache,
                                    0.0,
                                ),
                                axis=0,
                            )
                        else:
                            h_j = tl.sum(
                                tl.where(
                                    (offs_h == ((HC_SLOTS_HI >> (4 * (j - 16))) & 15))[:, None, None],
                                    h_cache,
                                    0.0,
                                ),
                                axis=0,
                            )
                        state_i = tl.where(ancestor, h_j, state_i)

        b_q = tl.load(
            q + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_k_raw = tl.load(
            k + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=i < N_ACTUAL,
            other=0.0,
        )
        b_k = b_k_raw.to(tl.float32)
        b_v_raw = tl.load(
            v + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            mask=(i < N_ACTUAL) & v_mask,
            other=0.0,
        )
        b_v = b_v_raw.to(tl.float32)
        # FR13_RING_EXPORT: stage the replay ring IN-KERNEL from the very values
        # just loaded (byte-copy contract: k pre-l2norm, v, raw_a, raw_b at input
        # precision), replacing the 4 per-layer aten .copy_() launches that the
        # nsys differential measured as part of the +20ms/draft tree-only
        # elementwise soup. Write-once partition: v is tiled exactly by the
        # (pid_vh, pid_v) grid; k is per-KH so only the first program of each
        # head group stores; a/b are per-VH scalars stored by the pid_v==0
        # column. Default OFF => constexpr-dead => byte-identical codegen.
        if RING_EXPORT:
            tl.store(
                ring_k + (i * NUM_KH + pid_kh) * DIM_K + offs_k,
                b_k_raw,
                mask=(i < N_ACTUAL)
                & (pid_v == 0)
                & (pid_vh % head_group == 0),
            )
            tl.store(
                ring_v + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
                b_v_raw,
                mask=(i < N_ACTUAL) & v_mask,
            )
        b_b = tl.load(
            beta + i * NUM_VH + pid_vh,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            g + i * NUM_VH + pid_vh,
            mask=i < N_ACTUAL,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = b_g
        b_raw_b = b_b
        b_a_log = b_g
        b_dt_bias = b_b
        if RAW_GATING:
            b_raw_b_in = tl.load(
                raw_b + i * NUM_VH + pid_vh,
                mask=i < N_ACTUAL,
                other=0.0,
            )
            b_raw_b = b_raw_b_in.to(tl.float32)
            b_raw_a_in = tl.load(
                raw_a + i * NUM_VH + pid_vh,
                mask=i < N_ACTUAL,
                other=0.0,
            )
            b_raw_a = b_raw_a_in.to(tl.float32)
            # FR13_RING_EXPORT (a/b half): per-(node, value-head) scalars, one
            # store column (pid_v==0). Requires RAW_GATING (the served path);
            # the launcher asserts that pairing.
            if RING_EXPORT:
                tl.store(
                    ring_a + i * NUM_VH + pid_vh,
                    b_raw_a_in,
                    mask=(i < N_ACTUAL) & (pid_v == 0),
                )
                tl.store(
                    ring_b + i * NUM_VH + pid_vh,
                    b_raw_b_in,
                    mask=(i < N_ACTUAL) & (pid_v == 0),
                )
            b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
            b_a_log = tl.load(A_log + pid_vh).to(tl.float32)

        state_i, out_i = _gdn_node_step(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_g,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
        )
        if HC_MASK == 0:
            h_cache = tl.where((offs_n == i)[:, None, None], state_i[None, :, :], h_cache)
        else:
            # FR13_HC_INTERNAL: leaf states are never re-read -> only internal
            # nodes keep a compacted row (identical one-hot store form).
            if ((HC_MASK >> i) & 1) == 1:
                if i < 16:
                    h_cache = tl.where(
                        (offs_h == ((HC_SLOTS_LO >> (4 * i)) & 15))[:, None, None],
                        state_i[None, :, :],
                        h_cache,
                    )
                else:
                    h_cache = tl.where(
                        (offs_h == ((HC_SLOTS_HI >> (4 * (i - 16))) & 15))[:, None, None],
                        state_i[None, :, :],
                        h_cache,
                    )
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=(i < N_ACTUAL) & v_mask,
        )
    # FR13_PIGGYBACK: after the scan, export the FIXED chain-end node's committed state -> col 0 (the running
    # row), so the NEXT step's h0=col0 carries the committed state WITHOUT the 48-kernel replay. CHAIN_END_IDX
    # is a fixed known column (static mask). Constexpr-gated (default off => constexpr-dead => byte-identical
    # codegen). Same one-hot h_cache reduction (:808-811/:831-834) + col-0 store form as the h0 seed (:764-778)
    # and the replay RUNROW_COMMIT (:1058-72).
    if FLAGS_EXPORT:
        # FR13_FLAGS_INKERNEL: staging-freshness flags written by the scan
        # itself (RING_EXPORT precedent) — replaces 2 aten fills per layer per
        # step (96 launches/step). Values identical: flags[0]=1 staged,
        # flags[1]=rows. One program stores (pid_vh==0, pid_v==0 guard).
        tl.store(flags_ptr + 0, 1, mask=(pid_vh == 0) & (pid_v == 0))
        tl.store(
            flags_ptr + 1, FLAGS_ROWS, mask=(pid_vh == 0) & (pid_v == 0)
        )
    if PIGGYBACK_EXPORT:
        _pb_state = tl.sum(
            tl.where((offs_n == CHAIN_END_IDX)[:, None, None], h_cache, 0.0),
            axis=0,
        )
        _pb_col0_index = tl.load(h0_indices + H0_INDEX_ROW + 0)
        _pb_col0_base = h0 + _pb_col0_index * H0_BANK_STRIDE
        tl.store(
            _pb_col0_base + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
            _pb_state,
            mask=v_mask[:, None],
        )


@triton.jit
def _tree_gdn_path_kernel(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    path_nodes,     # [n_paths, MAX_PATH_LEN] int32 node ids, -1 padded
    path_parent,    # [n_paths] int32 parent NODE id of the path root (-1 => h0)
    path_lengths,   # [n_paths] int32 active nodes; device-loaded graph-stable bound
    state_export,   # [N_TOTAL, NUM_VH, DIM_V, DIM_K] fp32 handoff buffer
    export_mask,    # [N_TOTAL] int32 1 => export this node's post-state
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    flags_ptr,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    MAX_PATH_LEN: tl.constexpr,
    STATE_SOURCE: tl.constexpr = 0,
    EXPORT_MODE: tl.constexpr = 0,
    RING_EXPORT: tl.constexpr = False,
    FLAGS_EXPORT: tl.constexpr = False,
    FLAGS_ROWS: tl.constexpr = 0,
):
    # FR13_SUBTREE_PARALLEL path scan: one program = one PATH (pure chain).
    # State is carried in registers node-to-node -- NO h_cache, NO one-hot
    # ancestor machinery. The path root's state comes from h0 (par < 0) or
    # the fp32 export of its parent node (written by an earlier level's
    # launch; fp32 store->load is bit-exact). Per-node math is the identical
    # _gdn_node_step; per-path node order equals the monolith's path order.
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_path = tl.program_id(2)
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0) & (pid_path == 0),
        )
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    # STATE_SOURCE is dynamic for the generic route. The exact fixed32
    # schedule specializes root level to h0 and level 1 to the exported
    # parent. This removes eleven dead h0 tile reads per (VH, V-block) without
    # changing the state bytes entering _gdn_node_step.
    if STATE_SOURCE != 2:
        h0_base = h0
        if H0_IS_BANK:
            h0_column = 0
            if H0_USE_ACCEPTED_COLUMN:
                h0_column = tl.maximum(
                    tl.load(h0_num_accepted_tokens + H0_BATCH_INDEX).to(tl.int64) - 1,
                    0,
                )
            h0_index = tl.load(h0_indices + H0_INDEX_ROW + h0_column)
            h0_base = h0 + h0_index * H0_BANK_STRIDE
        b_h0 = tl.load(
            h0_base
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            mask=v_mask[:, None],
            other=0.0,
        ).to(tl.float32)
    if STATE_SOURCE != 1:
        par = tl.load(path_parent + pid_path)
        par_state = tl.load(
            state_export
            + ((tl.maximum(par, 0).to(tl.int64) * NUM_VH + pid_vh) * DIM_V
               + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            mask=(par >= 0) & v_mask[:, None],
            other=0.0,
        )
    if STATE_SOURCE == 1:
        state_i = b_h0
    elif STATE_SOURCE == 2:
        state_i = par_state
    else:
        state_i = tl.where(par >= 0, par_state, b_h0)

    path_len = tl.load(path_lengths + pid_path)
    for i in tl.range(0, path_len):
        node = tl.load(path_nodes + pid_path * MAX_PATH_LEN + i)
        n_ok = (node >= 0) & (node < N_ACTUAL)
        node_c = tl.maximum(node, 0)
        b_q = tl.load(
            q + (node_c * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=n_ok,
            other=0.0,
        ).to(tl.float32)
        b_k_raw = tl.load(
            k + (node_c * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=n_ok,
            other=0.0,
        )
        b_k = b_k_raw.to(tl.float32)
        b_v_raw = tl.load(
            v + (node_c * NUM_VH + pid_vh) * DIM_V + offs_v,
            mask=n_ok & v_mask,
            other=0.0,
        )
        b_v = b_v_raw.to(tl.float32)
        if RING_EXPORT:
            tl.store(
                ring_k + (node_c * NUM_KH + pid_kh) * DIM_K + offs_k,
                b_k_raw,
                mask=n_ok
                & (pid_v == 0)
                & (pid_vh % head_group == 0),
            )
            tl.store(
                ring_v + (node_c * NUM_VH + pid_vh) * DIM_V + offs_v,
                b_v_raw,
                mask=n_ok & v_mask,
            )
        b_b = tl.load(
            beta + node_c * NUM_VH + pid_vh, mask=n_ok, other=0.0
        ).to(tl.float32)
        b_g = tl.load(
            g + node_c * NUM_VH + pid_vh, mask=n_ok, other=0.0
        ).to(tl.float32)
        b_raw_a = b_g
        b_raw_b = b_b
        b_a_log = b_g
        b_dt_bias = b_b
        if RAW_GATING:
            b_raw_b_in = tl.load(
                raw_b + node_c * NUM_VH + pid_vh, mask=n_ok, other=0.0
            )
            b_raw_b = b_raw_b_in.to(tl.float32)
            b_raw_a_in = tl.load(
                raw_a + node_c * NUM_VH + pid_vh, mask=n_ok, other=0.0
            )
            b_raw_a = b_raw_a_in.to(tl.float32)
            if RING_EXPORT:
                tl.store(
                    ring_a + node_c * NUM_VH + pid_vh,
                    b_raw_a_in,
                    mask=n_ok & (pid_v == 0),
                )
                tl.store(
                    ring_b + node_c * NUM_VH + pid_vh,
                    b_raw_b_in,
                    mask=n_ok & (pid_v == 0),
                )
            b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
            b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_g,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
        )
        # Keep the descriptor guard value-neutral if a corrupt node slips past
        # preseed validation; valid dynamic-loop iterations always have n_ok.
        state_i = tl.where(n_ok, new_state, state_i)
        tl.store(
            out + (node_c * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=n_ok & v_mask,
        )
        if EXPORT_MODE != 2:
            if EXPORT_MODE == 0:
                do_exp = n_ok & (tl.load(export_mask + node_c) != 0)
            else:
                do_exp = n_ok
            tl.store(
                state_export
                + ((node_c.to(tl.int64) * NUM_VH + pid_vh) * DIM_V
                   + offs_v[:, None]) * DIM_K
                + offs_k[None, :],
                state_i,
                mask=do_exp & v_mask[:, None],
            )
    if FLAGS_EXPORT:
        # FR13_FLAGS_INKERNEL under the subtree route: same staging-freshness
        # store as the monolith tail (values identical: flags[0]=1 staged,
        # flags[1]=rows). Exactly ONE program stores — the launcher passes
        # FLAGS_EXPORT=True only on the level-0 launch; pid_path==0 guards
        # within it. Stream-ordered before any later consumer.
        _fl_mask = (pid_vh == 0) & (pid_v == 0) & (pid_path == 0)
        tl.store(flags_ptr + 0, 1, mask=_fl_mask)
        tl.store(flags_ptr + 1, FLAGS_ROWS, mask=_fl_mask)


@triton.jit
def _tree_gdn_path_kernel_fixed32_batch(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    path_nodes,
    path_parent_slots,
    path_lengths,
    state_export,
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    flags_ptr,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_INDEX_BATCH_STRIDE: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_ACCEPTED_BATCH_STRIDE: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    MAX_PATH_LEN: tl.constexpr,
    NUM_PATHS: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    EXPORT_SLOTS: tl.constexpr,
    STATE_SOURCE: tl.constexpr,
    EXPORT_MODE: tl.constexpr,
    RING_EXPORT: tl.constexpr = False,
    FLAGS_EXPORT: tl.constexpr = False,
    FLAGS_ROWS: tl.constexpr = 0,
):
    """Fixed32 path scan with request folded into path-grid axis 2.

    This is separate from ``_tree_gdn_path_kernel`` so the deployed B1 and
    generic Triton source/codegen stay untouched. Each request still executes
    the same path-local ``_gdn_node_step`` sequence; only independent request
    programs share the two level launches.
    """
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_global_path = tl.program_id(2)
    pid_batch = pid_global_path // NUM_PATHS
    pid_path = pid_global_path - pid_batch * NUM_PATHS
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0) & (pid_path == 0),
        )
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    if STATE_SOURCE != 2:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            accepted_index = (
                H0_BATCH_INDEX + pid_batch * H0_ACCEPTED_BATCH_STRIDE
            )
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + accepted_index).to(tl.int64)
                - 1,
                0,
            )
        h0_index_row = H0_INDEX_ROW + pid_batch * H0_INDEX_BATCH_STRIDE
        h0_index = tl.load(h0_indices + h0_index_row + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
        b_h0 = tl.load(
            h0_base
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            mask=v_mask[:, None],
            other=0.0,
        ).to(tl.float32)
    if STATE_SOURCE != 1:
        parent_slot = tl.load(path_parent_slots + pid_path)
        export_row = (
            pid_batch.to(tl.int64) * EXPORT_SLOTS
            + tl.maximum(parent_slot, 0).to(tl.int64)
        )
        par_state = tl.load(
            state_export
            + ((export_row * NUM_VH + pid_vh) * DIM_V
               + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            mask=(parent_slot >= 0) & v_mask[:, None],
            other=0.0,
        )
    if STATE_SOURCE == 1:
        state_i = b_h0
    elif STATE_SOURCE == 2:
        state_i = par_state
    else:
        state_i = tl.where(parent_slot >= 0, par_state, b_h0)

    path_len = tl.load(path_lengths + pid_path)
    for i in tl.range(0, path_len):
        node = tl.load(path_nodes + pid_path * MAX_PATH_LEN + i)
        n_ok = (node >= 0) & (node < N_ACTUAL) & (pid_batch < BATCH_SIZE)
        node_c = tl.maximum(node, 0)
        global_node = pid_batch * N_ACTUAL + node_c
        b_q = tl.load(
            q + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=n_ok,
            other=0.0,
        ).to(tl.float32)
        b_k_raw = tl.load(
            k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
            mask=n_ok,
            other=0.0,
        )
        b_k = b_k_raw.to(tl.float32)
        b_v_raw = tl.load(
            v + (global_node * NUM_VH + pid_vh) * DIM_V + offs_v,
            mask=n_ok & v_mask,
            other=0.0,
        )
        b_v = b_v_raw.to(tl.float32)
        if RING_EXPORT:
            tl.store(
                ring_k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
                b_k_raw,
                mask=n_ok
                & (pid_v == 0)
                & (pid_vh % head_group == 0),
            )
            tl.store(
                ring_v
                + (global_node * NUM_VH + pid_vh) * DIM_V
                + offs_v,
                b_v_raw,
                mask=n_ok & v_mask,
            )
        b_b = tl.load(
            beta + global_node * NUM_VH + pid_vh, mask=n_ok, other=0.0
        ).to(tl.float32)
        b_g = tl.load(
            g + global_node * NUM_VH + pid_vh, mask=n_ok, other=0.0
        ).to(tl.float32)
        b_raw_a = b_g
        b_raw_b = b_b
        b_a_log = b_g
        b_dt_bias = b_b
        if RAW_GATING:
            b_raw_b_in = tl.load(
                raw_b + global_node * NUM_VH + pid_vh,
                mask=n_ok,
                other=0.0,
            )
            b_raw_b = b_raw_b_in.to(tl.float32)
            b_raw_a_in = tl.load(
                raw_a + global_node * NUM_VH + pid_vh,
                mask=n_ok,
                other=0.0,
            )
            b_raw_a = b_raw_a_in.to(tl.float32)
            if RING_EXPORT:
                tl.store(
                    ring_a + global_node * NUM_VH + pid_vh,
                    b_raw_a_in,
                    mask=n_ok & (pid_v == 0),
                )
                tl.store(
                    ring_b + global_node * NUM_VH + pid_vh,
                    b_raw_b_in,
                    mask=n_ok & (pid_v == 0),
                )
            b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
            b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_g,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
        )
        state_i = tl.where(n_ok, new_state, state_i)
        tl.store(
            out + (global_node * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=n_ok & v_mask,
        )
        if EXPORT_MODE == 1:
            compact_export_row = (
                pid_batch.to(tl.int64) * EXPORT_SLOTS + i
            )
            tl.store(
                state_export
                + ((compact_export_row * NUM_VH + pid_vh) * DIM_V
                   + offs_v[:, None]) * DIM_K
                + offs_k[None, :],
                state_i,
                mask=n_ok & v_mask[:, None],
            )
    if FLAGS_EXPORT:
        flag_writer = (
            (pid_vh == 0)
            & (pid_v == 0)
            & (pid_batch == 0)
            & (pid_path == 0)
        )
        tl.store(flags_ptr + 0, 1, mask=flag_writer)
        tl.store(flags_ptr + 1, FLAGS_ROWS, mask=flag_writer)


@triton.jit
def _tree_gdn_fixed32_single_launch_node(
    state_i,
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    b_a_log,
    b_dt_bias,
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    ring_k_norm,
    ring_gate,
    pid_batch,
    pid_vh,
    pid_v,
    pid_kh,
    offs_k,
    offs_v,
    v_mask,
    node,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    RING_EXPORT: tl.constexpr,
    K_NORM_EXPORT: tl.constexpr,
    GATE_EXPORT: tl.constexpr,
    DECAY_EXPORT: tl.constexpr,
):
    """Run one unchanged GDN recurrence and its single-writer stores."""
    n_ok = (node >= 0) & (node < N_ACTUAL)
    node_c = tl.maximum(node, 0)
    global_node = pid_batch * N_ACTUAL + node_c
    head_group = NUM_VH // NUM_KH
    b_q = tl.load(
        q + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
        mask=n_ok,
        other=0.0,
    ).to(tl.float32)
    b_k_raw = tl.load(
        k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
        mask=n_ok,
        other=0.0,
    )
    b_k = b_k_raw.to(tl.float32)
    b_v_raw = tl.load(
        v + (global_node * NUM_VH + pid_vh) * DIM_V + offs_v,
        mask=n_ok & v_mask,
        other=0.0,
    )
    b_v = b_v_raw.to(tl.float32)
    if RING_EXPORT:
        tl.store(
            ring_k + (global_node * NUM_KH + pid_kh) * DIM_K + offs_k,
            b_k_raw,
            mask=n_ok & (pid_v == 0) & (pid_vh % head_group == 0),
        )
        tl.store(
            ring_v + (global_node * NUM_VH + pid_vh) * DIM_V + offs_v,
            b_v_raw,
            mask=n_ok & v_mask,
        )
    b_b = tl.load(
        beta + global_node * NUM_VH + pid_vh,
        mask=n_ok,
        other=0.0,
    ).to(tl.float32)
    b_g = tl.load(
        g + global_node * NUM_VH + pid_vh,
        mask=n_ok,
        other=0.0,
    ).to(tl.float32)
    b_raw_a = b_g
    b_raw_b = b_b
    if RAW_GATING:
        b_raw_b_in = tl.load(
            raw_b + global_node * NUM_VH + pid_vh,
            mask=n_ok,
            other=0.0,
        )
        b_raw_b = b_raw_b_in.to(tl.float32)
        b_raw_a_in = tl.load(
            raw_a + global_node * NUM_VH + pid_vh,
            mask=n_ok,
            other=0.0,
        )
        b_raw_a = b_raw_a_in.to(tl.float32)
        if RING_EXPORT:
            tl.store(
                ring_a + global_node * NUM_VH + pid_vh,
                b_raw_a_in,
                mask=n_ok & (pid_v == 0),
            )
            tl.store(
                ring_b + global_node * NUM_VH + pid_vh,
                b_raw_b_in,
                mask=n_ok & (pid_v == 0),
            )
    b_k_inv_norm = 1.0
    if K_NORM_EXPORT:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k_inv_norm = tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_k = b_k * b_k_inv_norm
        tl.store(
            ring_k_norm + global_node * NUM_KH + pid_kh,
            b_k_inv_norm,
            mask=n_ok & (pid_v == 0) & (pid_vh % head_group == 0),
        )
    if GATE_EXPORT:
        x = b_raw_a + b_dt_bias
        softplus_x = tl.where(
            x <= 20.0,
            tl.log(1.0 + tl.exp(x)),
            x,
        )
        b_g = -tl.exp(b_a_log) * softplus_x
        b_b = tl.sigmoid(b_raw_b.to(tl.float32))
        b_decay = tl.exp(b_g)
        gate_offset = (global_node * NUM_VH + pid_vh) * 2
        tl.store(
            ring_gate + gate_offset,
            b_decay if DECAY_EXPORT else b_g,
            mask=n_ok & (pid_v == 0),
        )
        tl.store(
            ring_gate + gate_offset + 1,
            b_b,
            mask=n_ok & (pid_v == 0),
        )
    if DECAY_EXPORT:
        new_state, out_i = _gdn_node_step_precomputed_decay(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_decay,
            OUTPUT_SCALE=OUTPUT_SCALE,
        )
    else:
        new_state, out_i = _gdn_node_step(
            state_i,
            b_q,
            b_k,
            b_v,
            b_b,
            b_g,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=(
                USE_QK_L2NORM_IN_KERNEL and not K_NORM_EXPORT
            ),
            RAW_GATING=RAW_GATING and not GATE_EXPORT,
            SCAN_ALIGN=SCAN_ALIGN and not K_NORM_EXPORT and not GATE_EXPORT,
        )
    tl.store(
        out + (global_node * NUM_VH + pid_vh) * DIM_V + offs_v,
        out_i,
        mask=n_ok & v_mask,
    )
    return tl.where(n_ok, new_state, state_i)


@triton.jit
def _tree_gdn_kernel_fixed32_single_launch(
    q,
    k,
    v,
    g,
    beta,
    raw_a,
    raw_b,
    A_log,
    dt_bias,
    h0,
    h0_indices,
    h0_num_accepted_tokens,
    invocation_counter,
    root_nodes,
    branch_nodes,
    branch_lengths,
    group_path_indices,
    group_path_counts,
    out,
    ring_k,
    ring_v,
    ring_a,
    ring_b,
    flags_ptr,
    ring_k_norm,
    ring_gate,
    N_ACTUAL: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    H0_IS_BANK: tl.constexpr,
    H0_INDEX_ROW: tl.constexpr,
    H0_INDEX_BATCH_STRIDE: tl.constexpr,
    H0_BATCH_INDEX: tl.constexpr,
    H0_ACCEPTED_BATCH_STRIDE: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    H0_USE_ACCEPTED_COLUMN: tl.constexpr,
    RAW_GATING: tl.constexpr,
    COUNT_INVOCATION: tl.constexpr,
    SCAN_ALIGN: tl.constexpr,
    ROOT_STEPS: tl.constexpr,
    MAX_PATH_LEN: tl.constexpr,
    MAX_GROUP_PATHS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PRESCALED_PATH_BASE: tl.constexpr = False,
    RING_EXPORT: tl.constexpr = False,
    K_NORM_EXPORT: tl.constexpr = False,
    GATE_EXPORT: tl.constexpr = False,
    DECAY_EXPORT: tl.constexpr = False,
    FLAGS_EXPORT: tl.constexpr = False,
    FLAGS_ROWS: tl.constexpr = 0,
):
    """Run the exact fixed32 tree in one launch without an HBM state handoff."""
    pid_vh = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_batch = tl.program_id(2)
    if COUNT_INVOCATION:
        tl.atomic_add(
            invocation_counter,
            1,
            sem="relaxed",
            mask=(pid_vh == 0) & (pid_v == 0),
        )
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    h0_base = h0
    if H0_IS_BANK:
        h0_column = 0
        if H0_USE_ACCEPTED_COLUMN:
            accepted_index = (
                H0_BATCH_INDEX + pid_batch * H0_ACCEPTED_BATCH_STRIDE
            )
            h0_column = tl.maximum(
                tl.load(h0_num_accepted_tokens + accepted_index).to(tl.int64)
                - 1,
                0,
            )
        h0_index_row = H0_INDEX_ROW + pid_batch * H0_INDEX_BATCH_STRIDE
        h0_index = tl.load(h0_indices + h0_index_row + h0_column)
        h0_base = h0 + h0_index * H0_BANK_STRIDE
    root_state = tl.load(
        h0_base
        + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
        + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)

    # This loop is deliberately ordered and not statically expanded. The
    # inner fixed-cardinality member loop stays static to preserve its clean
    # zero-stack SM121 codegen.
    for root_index in tl.range(0, ROOT_STEPS):
        root_node = tl.load(root_nodes + root_index)
        root_state = _tree_gdn_fixed32_single_launch_node(
            root_state,
            q,
            k,
            v,
            g,
            beta,
            raw_a,
            raw_b,
            b_a_log,
            b_dt_bias,
            out,
            ring_k,
            ring_v,
            ring_a,
            ring_b,
            ring_k_norm,
            ring_gate,
            pid_batch,
            pid_vh,
            pid_v,
            pid_kh,
            offs_k,
            offs_v,
            v_mask,
            root_node,
            N_ACTUAL=N_ACTUAL,
            NUM_KH=NUM_KH,
            NUM_VH=NUM_VH,
            DIM_K=DIM_K,
            DIM_V=DIM_V,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
            RING_EXPORT=RING_EXPORT,
            K_NORM_EXPORT=K_NORM_EXPORT,
            GATE_EXPORT=GATE_EXPORT,
            DECAY_EXPORT=DECAY_EXPORT,
        )
        group_path_count = tl.load(group_path_counts + root_index)
        for member in tl.static_range(0, MAX_GROUP_PATHS):
            member_ok = member < group_path_count
            path_index = tl.load(
                group_path_indices
                + root_index * MAX_GROUP_PATHS
                + member,
                mask=member_ok,
                other=0,
            )
            if PRESCALED_PATH_BASE:
                path_base = path_index
            else:
                path_base = path_index * MAX_PATH_LEN
            path_len = tl.load(
                branch_lengths
                + (path_base if PRESCALED_PATH_BASE else path_index),
                mask=member_ok,
                other=0,
            )
            branch_state = root_state
            for path_offset in tl.range(0, path_len):
                branch_node = tl.load(
                    branch_nodes + path_base + path_offset
                )
                branch_state = _tree_gdn_fixed32_single_launch_node(
                    branch_state,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    raw_a,
                    raw_b,
                    b_a_log,
                    b_dt_bias,
                    out,
                    ring_k,
                    ring_v,
                    ring_a,
                    ring_b,
                    ring_k_norm,
                    ring_gate,
                    pid_batch,
                    pid_vh,
                    pid_v,
                    pid_kh,
                    offs_k,
                    offs_v,
                    v_mask,
                    branch_node,
                    N_ACTUAL=N_ACTUAL,
                    NUM_KH=NUM_KH,
                    NUM_VH=NUM_VH,
                    DIM_K=DIM_K,
                    DIM_V=DIM_V,
                    OUTPUT_SCALE=OUTPUT_SCALE,
                    USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
                    RAW_GATING=RAW_GATING,
                    SCAN_ALIGN=SCAN_ALIGN,
                    RING_EXPORT=RING_EXPORT,
                    K_NORM_EXPORT=K_NORM_EXPORT,
                    GATE_EXPORT=GATE_EXPORT,
                    DECAY_EXPORT=DECAY_EXPORT,
                )
    if FLAGS_EXPORT:
        flag_writer = (pid_vh == 0) & (pid_v == 0) & (pid_batch == 0)
        tl.store(flags_ptr + 0, 1, mask=flag_writer)
        tl.store(flags_ptr + 1, FLAGS_ROWS, mask=flag_writer)


@triton.jit
def _tree_gdn_replay_kernel(
    k_ring,
    v_ring,
    a_ring,
    b_ring,
    A_log,
    dt_bias,
    state_bank,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    prev_lens,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    N_PAD: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    RING_B_STRIDE_K: tl.constexpr,
    RING_N_STRIDE_K: tl.constexpr,
    RING_B_STRIDE_V: tl.constexpr,
    RING_N_STRIDE_V: tl.constexpr,
    RING_B_STRIDE_AB: tl.constexpr,
    RING_N_STRIDE_AB: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr = False,
    RUNROW_COMMIT: tl.constexpr = False,
    RUNROW_INIT: tl.constexpr = False,
    BURN_NODE_BANK: tl.constexpr = False,
    ROOT_NODE: tl.constexpr = 0,
):
    # FR13_REPLAY_ROUTE accepted-path chain replay (sibling of the scan).
    #
    # FR13 STATELESS-TREE (default all-False -> byte-identical):
    #   RUNROW_INIT   -> seed h0 from col 0 (the req running row) instead of the
    #                    positional accepted-leaf col nacc-1.
    #   RUNROW_COMMIT -> after the accepted-path loop, also store the committed
    #                    leaf `state` into col 0, so col 0 becomes native's
    #                    authoritative running row (snapshot/restore/next-init).
    #   BURN_NODE_BANK-> zero the ephemeral spec cols 1..num_spec after the
    #                    col-0 commit (col 0 preserved) = tree keeps zero lifespan.
    # The three form one lifecycle and are gated together by the committer.
    #
    # This kernel does not export per-node states to HBM; instead it re-executes the
    # committed accepted path from the activation ring (k pre-l2norm, v,
    # raw_a, raw_b at consumed precision) on the IDENTICAL shared
    # _gdn_node_step body, in the NATIVE gate-folding basis (no rescaled-exp
    # reconstruction), and publishes the post-step states directly to the
    # bank's LINEAR columns (column t = t-th accepted token), which removes
    # the ssm half of the next-step remap under the flag.
    #
    # No h_cache: one (BLOCK_V, DIM_K) register tile per program, so the
    # replay is spill-free at any tree size.
    pid_b = tl.program_id(0)
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    # h0 = same column convention as the scan's in-kernel h0 gather:
    # column clamp(prev_accepted_len - 1, 0). prev_lens is the SCAN-TIME
    # snapshot of the accepted-lens buffer (the committer refills the live
    # buffer with the NEW lens before this kernel launches).
    prev_len = tl.load(prev_lens + pid_b).to(tl.int64)
    if RUNROW_INIT:
        # STATELESS-TREE: prev step's RUNROW_COMMIT wrote its committed leaf into
        # col 0 (the req running row); read init from there = native semantics.
        h0_col = tl.zeros([], dtype=tl.int64)
    else:
        h0_col = tl.maximum(prev_len - 1, 0)
    h0_row = tl.load(spec_state_indices + pid_b * SPEC_COLS + h0_col).to(tl.int64)
    # Read the whole h0 tile into registers BEFORE any store: a later
    # publish in this same program may target the h0 bank row itself
    # (publish-overwrites-h0-row case).
    state = tl.load(
        state_bank
        + h0_row * BANK_STRIDE
        + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
        + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc_len = tl.load(accepted_lens + pid_b)
    # q is not stored: the q-side ops never touch state, out was already
    # emitted by the scan. A zero q keeps the shared body's signature and
    # constexprs identical; out_i is discarded.
    b_q = tl.zeros((DIM_K,), dtype=tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
    for t in tl.static_range(0, PATH_COLS + 1):
        if t == 0:
            # Root (gdn node ROOT_NODE; stock 0 -- FR13_PIGGYBACK catch-up
            # passes 8 = the LIVE-8 bonus twin's ring row, since ring row 0
            # is the diverged pos-0 copy) replays unconditionally: row 0 must
            # be refreshed even on a ZERO-ACCEPT event (the next h0 read
            # clamps accepted_len-1 to column 0), and the scan applies NO
            # handoff normalization to h0 before the root update.
            active = acc_len >= 0
            node = ROOT_NODE
        else:
            active = (t - 1) < acc_len
            node = tl.load(
                accepted_paths + pid_b * PATH_COLS + (t - 1),
                mask=active,
                other=0,
            ).to(tl.int64)
            node = tl.maximum(node, 0)
            node = tl.minimum(node, N_PAD - 1)
            # Parent-handoff normalization: the scan reads the parent state
            # through tl.sum(tl.where(offs_n == j, h_cache, 0.0), axis=0),
            # which flips -0.0 to +0.0 exactly once per edge. `+ 0.0`
            # reproduces that bit behavior; the root above gets none.
            state = state + 0.0
        b_k = tl.load(
            k_ring
            + pid_b * RING_B_STRIDE_K
            + node * RING_N_STRIDE_K
            + pid_kh * DIM_K
            + offs_k,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_ring
            + pid_b * RING_B_STRIDE_V
            + node * RING_N_STRIDE_V
            + pid_vh * DIM_V
            + offs_v,
            mask=active & v_mask,
            other=0.0,
        ).to(tl.float32)
        b_raw_b = tl.load(
            b_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = tl.load(
            a_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state,
            b_q,
            b_k,
            b_v,
            0.0,
            0.0,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
        )
        state = tl.where(active, new_state, state)
        if t == 0:
            dst_col = 0
        else:
            dst_col = t - 1
        dst_row = tl.load(
            spec_state_indices + pid_b * SPEC_COLS + dst_col,
            mask=active,
            other=0,
        ).to(tl.int64)
        tl.store(
            state_bank
            + dst_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=active & v_mask[:, None],
        )
    if RUNROW_COMMIT:
        # STATELESS-TREE: at loop exit `state` holds the committed leaf (last
        # active t; inactive t preserved it via tl.where at :1199) -- the SAME
        # bytes stored to col nacc-1 above, so this is byte-identical on the
        # no-cache path. Deposit into col 0 (the req running row) so col 0 is
        # native's authoritative source for snapshot / restore / next-step init.
        rr_row = tl.load(spec_state_indices + pid_b * SPEC_COLS + 0).to(tl.int64)
        tl.store(
            state_bank
            + rr_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=v_mask[:, None],
        )
    if BURN_NODE_BANK:
        # Zero the ephemeral spec cols 1..num_spec (col 0 preserved). Nothing
        # downstream reads them post-fix (h0 + snapshot both read col 0; the next
        # step's scan writes its own nodes fresh) => tree keeps zero lifespan.
        for _bc in tl.static_range(1, SPEC_COLS):
            z_row = tl.load(
                spec_state_indices + pid_b * SPEC_COLS + _bc
            ).to(tl.int64)
            tl.store(
                state_bank
                + z_row * BANK_STRIDE
                + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
                + offs_k[None, :],
                tl.zeros([BLOCK_V, DIM_K], dtype=tl.float32),
                mask=v_mask[:, None],
            )


def _fr13_prepare_committer_layout(*, accepted_paths, accepted_lens, spec_state_indices, num_spec_decodes, root_node=0):
    """Compute the per-request committed-path node layout ONCE (it depends only on accepted_paths/
    accepted_lens/spec_state_indices, all SHARED across the ~48 GDN layers). This is the only host sync;
    hoisting it out of the per-layer loop eliminates B*47 syncs that made the eager committer crawl.
    Returns (nodes_list, cu, ssi, T, max_T)."""
    dev = accepted_paths.device
    B = int(num_spec_decodes)
    acc = accepted_lens[:B].tolist()  # ONE host sync for all requests
    nodes_list, seg_len = [], []
    for b in range(B):
        L = int(acc[b])
        nodes = torch.cat([
            torch.full((1,), int(root_node), dtype=torch.long, device=dev),
            accepted_paths[b, :L].to(torch.long),
        ])
        nodes_list.append(nodes)
        seg_len.append(int(nodes.numel()))
    T = int(sum(seg_len))
    max_T = int(max(seg_len))
    cu = torch.tensor(
        [0] + list(torch.tensor(seg_len).cumsum(0).tolist()),
        device=dev, dtype=torch.int32,
    )
    ssi = torch.zeros(B, max_T, device=dev, dtype=torch.int32)
    col0 = spec_state_indices[:B, 0].to(torch.int32)
    for b in range(B):
        ssi[b, :] = col0[b]
    return nodes_list, cu, ssi, T, max_T


_FR13_SG_PENDING = []
_FR13_SG_ACCUM = [0.0, 0]  # [gpu_seconds, n_calls]
_FR13_SG_LAST_DUMP = [0.0]


def _fr13_sg_drain_dump():
    """FR13_COMMITTER_SG_TIMER (diagnostic, default OFF): accumulate ONLY the
    fused_sigmoid GPU time inside the native committer replay, via deferred cuda
    events (no host sync). Subtract from the fr13_committer_gpu span (~88ms) to
    split fused_sigmoid-GPU vs host gathers+dispatch-gaps -- settles whether the
    replay is host-bound (batch/graph reducible) or fused_sigmoid-bound."""
    import json as _j
    import os as _o
    import time as _tm
    _p = _FR13_SG_PENDING
    while _p:
        _a, _b = _p[0]
        try:
            if not _b.query():
                break
        except Exception:
            _p.pop(0)
            continue
        _p.pop(0)
        try:
            _FR13_SG_ACCUM[0] += _a.elapsed_time(_b) / 1000.0
            _FR13_SG_ACCUM[1] += 1
        except Exception:
            pass
    _out = _o.environ.get("FR13_COMMITTER_SG_TIMER_JSON")
    if _out and (_tm.monotonic() - _FR13_SG_LAST_DUMP[0]) > 5.0:
        _FR13_SG_LAST_DUMP[0] = _tm.monotonic()
        try:
            _tmp = _out + ".tmp"
            with open(_tmp, "w") as _fh:
                _fh.write(_j.dumps({"sg_gpu_seconds": _FR13_SG_ACCUM[0],
                                    "n_sg": _FR13_SG_ACCUM[1]}))
            _o.replace(_tmp, _out)
        except Exception:
            pass


_FR13_REPLAY_ONLY_PENDING = []
_FR13_REPLAY_ONLY_ACCUM = [0.0, 0]  # [gpu_seconds, n_calls]
_FR13_REPLAY_ONLY_LAST_DUMP = [0.0]


def _fr13_replay_only_teardown():
    try:
        _fr13_replay_only_drain(True)
        _FR13_REPLAY_ONLY_LAST_DUMP[0] = 0.0  # force the final dump past the throttle
        _fr13_replay_only_drain(True)
    except Exception:
        pass


if os.environ.get("FR13_REPLAY_ONLY_GPU_TIMER") == "1":
    import atexit as _fr13_ro_atexit
    _fr13_ro_atexit.register(_fr13_replay_only_teardown)


def _fr13_replay_only_drain(blocking):
    """FR13_REPLAY_ONLY_GPU_TIMER (diagnostic, default OFF): times ONLY the deployed
    per-layer committer-replay loop (the exact scope the isolated micro-bench measured),
    via deferred cuda events -- separate from the patcher's FR13_CFWD_GPU_TIMER, which
    wraps the whole self.rejection_sampler dispatch (accept/LCP/bonus decision INCLUDED,
    a broader/different quantity). Answers: what does the pure state-commit replay cost,
    live, apples-to-apples against the micro-bench's 14.97ms and native's 7ms floor."""
    import json as _j
    import os as _o
    import time as _tm
    _p = _FR13_REPLAY_ONLY_PENDING
    while _p:
        _a, _b = _p[0]
        try:
            done = _b.query()
        except Exception:
            _p.pop(0)
            continue
        if not done and not blocking:
            break
        _p.pop(0)
        try:
            if blocking and not done:
                _b.synchronize()
            _FR13_REPLAY_ONLY_ACCUM[0] += _a.elapsed_time(_b) / 1000.0
            _FR13_REPLAY_ONLY_ACCUM[1] += 1
        except Exception:
            pass
    _out = _o.environ.get("FR13_REPLAY_ONLY_GPU_TIMER_JSON")
    if _out and (_tm.monotonic() - _FR13_REPLAY_ONLY_LAST_DUMP[0]) > 5.0:
        _FR13_REPLAY_ONLY_LAST_DUMP[0] = _tm.monotonic()
        try:
            _tmp = _out + ".tmp"
            with open(_tmp, "w") as _fh:
                _fh.write(_j.dumps({
                    "schema": "fr13.replay_only_gpu_timer.v1",
                    "replay_only_gpu_seconds": _FR13_REPLAY_ONLY_ACCUM[0],
                    "n_calls": _FR13_REPLAY_ONLY_ACCUM[1],
                }))
            _o.replace(_tmp, _out)
        except Exception:
            pass


def _fr13_native_committer_replay(
    *, state_bank, spec_state_indices, accepted_paths, accepted_lens,
    k_ring, v_ring, a_ring, b_ring, A_log, dt_bias, num_spec_decodes,
    output_scale, use_qk_l2norm_in_kernel, burn_node_bank, spec_cols,
    root_node=0, layout=None,
) -> None:
    """FR13_COMMITTER_NATIVE: rebuild col-0 (the running row) by running each request's committed path
    [node 0] ++ accepted_paths[:acc_len] through NATIVE fused_sigmoid_gating (bit-exact to no-spec), instead
    of the custom _tree_gdn_replay_kernel whose gross state-carry corruption at num_accepted>1 x branches is
    the garble root (output prob probe: near-impossible token, 15-nat gap). Validated offline vs pytorch-fp32
    ground truth at 1.19e-7 (scripts/fr13_native_committer_validate.py). REQUIRES runrow_init (col-0 h0);
    EAGER only (dynamic gather shapes break CUDA-graph capture). q is zeros (state-only; q never touches state)."""
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update as _sg,
    )
    global _FR13_COMMITTER_NATIVE_ANNOUNCED
    if not _FR13_COMMITTER_NATIVE_ANNOUNCED:
        _FR13_COMMITTER_NATIVE_ANNOUNCED = True
        print(f"[FR13_COMMITTER_NATIVE ENGAGED] native committed-path replay via fused_sigmoid_gating "
              f"(num_spec_decodes={int(num_spec_decodes)})", flush=True)
    # DIRECTION-2 narrow measurement (default OFF): times THIS single per-layer call (gather +
    # fused_sigmoid + burn) -- the exact scope the isolated micro-bench measured. Separate from
    # the patcher's FR13_CFWD_GPU_TIMER (wraps the whole self.rejection_sampler dispatch,
    # accept/LCP/bonus decision INCLUDED -- a broader, different quantity). This function is the
    # SHARED committer body called both from launch_tree_gdn_replay (singular, the deployed
    # per-layer dispatch loop in the patcher) and launch_tree_gdn_replay_all_layers -- placing the
    # timer here (not in either wrapper) is correct regardless of which one is actually live.
    _rt_on = os.environ.get("FR13_REPLAY_ONLY_GPU_TIMER") == "1"
    if _rt_on:
        _rt_start = torch.cuda.Event(enable_timing=True)
        _rt_start.record()
    dev = state_bank.device
    B = int(num_spec_decodes)
    num_kh, dim_k = int(k_ring.shape[2]), int(k_ring.shape[3])
    num_vh, dim_v = int(v_ring.shape[2]), int(v_ring.shape[3])
    # ssm_state_indices [B, max_T] every col = the col-0 running row: init reads col 0 = h0; the write-back
    # is PER-TOKEN, so the final token (col T-1) deposits the committed-path final state to that row = col-0.
    if layout is None:
        layout = _fr13_prepare_committer_layout(
            accepted_paths=accepted_paths, accepted_lens=accepted_lens,
            spec_state_indices=spec_state_indices, num_spec_decodes=num_spec_decodes,
            root_node=root_node,
        )
    nodes_list, cu, ssi, T, max_T = layout
    # Per-layer: ring GATHER only (device index_select on precomputed node tensors -> NO host sync).
    q = torch.zeros(1, T, num_kh, dim_k, device=dev, dtype=k_ring.dtype)
    k = torch.cat([k_ring[b, nodes_list[b]] for b in range(B)], 0).reshape(1, T, num_kh, dim_k).contiguous()
    v = torch.cat([v_ring[b, nodes_list[b]] for b in range(B)], 0).reshape(1, T, num_vh, dim_v).contiguous()
    aa = torch.cat([a_ring[b, nodes_list[b]] for b in range(B)], 0).reshape(1, T, num_vh).contiguous()
    bb = torch.cat([b_ring[b, nodes_list[b]] for b in range(B)], 0).reshape(1, T, num_vh).contiguous()
    _fr13_sg_on = os.environ.get("FR13_COMMITTER_SG_TIMER") == "1"
    if _fr13_sg_on:
        _fr13_sg_s = torch.cuda.Event(enable_timing=True)
        _fr13_sg_e = torch.cuda.Event(enable_timing=True)
        _fr13_sg_s.record()
    _sg(
        A_log=A_log, a=aa, b=bb, dt_bias=dt_bias, q=q, k=k, v=v, scale=output_scale,
        initial_state=state_bank, inplace_final_state=True, cu_seqlens=cu,
        ssm_state_indices=ssi, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if _fr13_sg_on:
        _fr13_sg_e.record()
        _FR13_SG_PENDING.append((_fr13_sg_s, _fr13_sg_e))
        _fr13_sg_drain_dump()
    if burn_node_bank:
        for b in range(B):
            rows = spec_state_indices[b, 1:spec_cols].to(torch.long)
            state_bank[rows] = 0.0
    if _rt_on:
        _rt_stop = torch.cuda.Event(enable_timing=True)
        _rt_stop.record()
        _FR13_REPLAY_ONLY_PENDING.append((_rt_start, _rt_stop))
        _fr13_replay_only_drain(False)


_FR13_COMMITTER_BATCHED_ANNOUNCED = False


def _fr13_native_committer_all_layers_batched(
    *, banks_list, spec_state_indices, accepted_paths, accepted_lens,
    k_rings, v_rings, a_rings, b_rings, A_logs, dt_biases, num_layers,
    num_spec_decodes, output_scale, use_qk_l2norm_in_kernel, burn_node_bank,
    root_node=0,
):
    """FR13_COMMITTER_NATIVE_BATCHED (default OFF): native committer replay with the SHARED
    accepted-path layout HOISTED (nodes_list/cu computed ONCE for all ~48 layers, not per-layer
    -- the '_fr13_prepare_committer_layout' docstring names this as the B*47 syncs that made the
    eager committer crawl) AND the 4 ring gathers BATCHED over the STACKED 5D rings
    (k_rings[:, b, nodes] once, not 48x). The per-layer fused_sigmoid + per-layer ssi (running
    row) + burn are the SAME ops in the SAME order => committed state BYTE-IDENTICAL to the
    per-layer loop; only the host layout+gather overhead (~73ms) is removed. Keeps the 48
    fused_sigmoid launches (graph-capture is the next lever). Shapes: k_rings[L,B,N,KH,DK],
    v_rings[L,B,N,VH,DV], a_rings/b_rings[L,B,N,VH]; spec_state_indices STACKED [L,B,SPEC_COLS]."""
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update as _sg,
    )
    L = int(num_layers)
    B = int(num_spec_decodes)
    dev = k_rings.device
    num_kh, dim_k = int(k_rings.shape[3]), int(k_rings.shape[4])
    num_vh, dim_v = int(v_rings.shape[3]), int(v_rings.shape[4])
    _spec_cols = int(spec_state_indices.shape[2])
    # ---- SHARED layout computed ONCE (the single .tolist() host sync for all L layers) ----
    # narrow-timer hooks (same accumulators/drains as the per-layer body; the
    # batched body previously had NONE -> arming the flags here was vacuous):
    # REPLAY_ONLY = whole batched replay (gathers + 48x _sg); SG = the 48x _sg
    # loop alone. Both deferred cuda events, no host sync, default OFF.
    _rt_on = os.environ.get("FR13_REPLAY_ONLY_GPU_TIMER") == "1"
    _sg_on = os.environ.get("FR13_COMMITTER_SG_TIMER") == "1"
    if _rt_on:
        _rt_start = torch.cuda.Event(enable_timing=True)
        _rt_start.record()
    acc = accepted_lens[:B].tolist()
    nodes_list, seg = [], []
    for b in range(B):
        _nl = int(acc[b])
        nodes = torch.cat([
            torch.full((1,), int(root_node), dtype=torch.long, device=dev),
            accepted_paths[b, :_nl].to(torch.long),
        ])
        nodes_list.append(nodes)
        seg.append(int(nodes.numel()))
    T = int(sum(seg))
    max_T = int(max(seg))
    cu = torch.tensor(
        [0] + list(torch.tensor(seg).cumsum(0).tolist()),
        device=dev, dtype=torch.int32,
    )
    # ---- BATCHED gather over the STACKED rings: one cat over B, ALL L layers ----
    k_all = torch.cat([k_rings[:, b, nodes_list[b]] for b in range(B)], dim=1)
    v_all = torch.cat([v_rings[:, b, nodes_list[b]] for b in range(B)], dim=1)
    a_all = torch.cat([a_rings[:, b, nodes_list[b]] for b in range(B)], dim=1)
    b_all = torch.cat([b_rings[:, b, nodes_list[b]] for b in range(B)], dim=1)
    q = torch.zeros(1, T, num_kh, dim_k, device=dev, dtype=k_rings.dtype)
    # FR13_SSI_PREBUILD (2026-07-24, committer-gap kill): the old loop built
    # ssi per layer on the HOST (48 x [zeros + index + B row-copies] aten
    # dispatches = the measured ~78%-of-cfwd gap bucket). One broadcast over
    # the stacked [L,B,spec_cols] tensor yields byte-identical ssi content
    # (col0 broadcast across max_T) in 3 launches total.
    ssi_all = (
        spec_state_indices[:, :B, 0:1]
        .to(torch.int32)
        .expand(L, B, max_T)
        .contiguous()
    )
    if _sg_on:
        _sg_s = torch.cuda.Event(enable_timing=True)
        _sg_e = torch.cuda.Event(enable_timing=True)
        _sg_s.record()
    for _L in range(L):
        ssi = ssi_all[_L]
        _sg(
            A_log=A_logs[_L],
            a=a_all[_L].reshape(1, T, num_vh).contiguous(),
            b=b_all[_L].reshape(1, T, num_vh).contiguous(),
            dt_bias=dt_biases[_L], q=q,
            k=k_all[_L].reshape(1, T, num_kh, dim_k).contiguous(),
            v=v_all[_L].reshape(1, T, num_vh, dim_v).contiguous(),
            scale=output_scale, initial_state=banks_list[_L],
            inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=ssi,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if burn_node_bank:
            for b in range(B):
                rows = spec_state_indices[_L][b, 1:_spec_cols].to(torch.long)
                banks_list[_L][rows] = 0.0
    if _sg_on:
        _sg_e.record()
        _FR13_SG_PENDING.append((_sg_s, _sg_e))
        _fr13_sg_drain_dump()
    if _rt_on:
        _rt_stop = torch.cuda.Event(enable_timing=True)
        _rt_stop.record()
        _FR13_REPLAY_ONLY_PENDING.append((_rt_start, _rt_stop))
        _fr13_replay_only_drain(False)
    global _FR13_COMMITTER_BATCHED_ANNOUNCED
    if not _FR13_COMMITTER_BATCHED_ANNOUNCED:
        _FR13_COMMITTER_BATCHED_ANNOUNCED = True
        print(
            "[FR13_COMMITTER_NATIVE_BATCHED ENGAGED] hoisted layout (1 .tolist) + "
            "batched gather (4 cat) over " + str(L) + " layers", flush=True,
        )


# ---- FR13_COMMITTER_GRAPH: CUDA-graph the 48-layer fused_sigmoid committer loop ----
# Proven in scripts/fr13_committer_graph_microbench.py + fr13_committer_graph_varying.py:
# capturing the loop at FIXED shapes (state-neutral padding a=-1e4=>decay1, k=v=b=0=>no write => state
# EXACTLY unchanged) is BYTE-IDENTICAL to the varlen committer (max_diff=0.0) and 5.4x faster (kills the
# ~24ms 48-launch dispatch). One graph (MAX_B x MAX_PATH) replays for every (B<=MAX_B, accept<=MAX_PATH).
_FR13_GRAPH_COMMITTER = {}          # shape-sig -> dict(buffers..., graph)
_FR13_GT_PENDING = []               # deferred (e0,e1,e2,e3) quads (FR13_GRAPH_TIMER)
_FR13_GT_ACCUM = [0.0, 0.0, 0.0, 0]  # fill_s, replay_s, burn_s, n
_FR13_GT_LAST_DUMP = [0.0]
_FR13_GRAPH_COMMITTER_ANNOUNCED = False
_FR13_CFWD_CAPTURE_REMAINDER_ANNOUNCED = False


def _fr13_cfwd_capture_remainder_on() -> bool:
    """FR13_CFWD_CAPTURE_REMAINDER (default OFF).

    Extends the FR13_COMMITTER_GRAPH capture region from `_loop()` alone to
    `_fill(); _loop()`, so the per-step buffer fill becomes graph nodes instead
    of eager launches. Env is read fresh (the EngineCore worker drops bare
    FR13_* vars, so the launcher also drops a sidecar)."""
    if os.environ.get("FR13_CFWD_CAPTURE_REMAINDER") == "1":
        return True
    for _p in (
        "/logs/fr13_cfwd_capture_remainder.arm",
        "/tmp/fr13_cfwd_capture_remainder.arm",
    ):
        if os.path.exists(_p):
            return True
    return False


def _fr13_native_committer_all_layers_graph(
    *, banks_list, spec_state_indices, accepted_paths, accepted_lens,
    k_rings, v_rings, a_rings, b_rings, A_logs, dt_biases, num_layers,
    num_spec_decodes, output_scale, use_qk_l2norm_in_kernel, burn_node_bank,
    root_node=0, max_path=16, max_b=None,
):
    """FR13_COMMITTER_GRAPH (default OFF): same committed state as the batched/per-layer committer, but the
    48 fused_sigmoid launches are captured into ONE cuda graph and replayed. Enabler = state-neutral padding
    to FIXED (MAX_B x MAX_PATH) shapes. Burn stays eager (Stage 1). Falls back to the batched committer
    (lossless) when accept+1 > MAX_PATH. Requires DISTINCT running rows per request (true in reality)."""
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update as _sg,
    )
    L = int(num_layers)
    B = int(num_spec_decodes)
    dev = k_rings.device
    num_kh, dim_k = int(k_rings.shape[3]), int(k_rings.shape[4])
    num_vh, dim_v = int(v_rings.shape[3]), int(v_rings.shape[4])
    _spec_cols = int(spec_state_indices.shape[2])
    MAX_B = int(max_b) if max_b else B
    MAX_PATH = int(max_path)
    MAXT = MAX_B * MAX_PATH
    SCRATCH = int(banks_list[0].shape[0]) - 1     # reserved throwaway row for dummy segments

    # ---- host layout (the single .tolist sync) ----
    acc = accepted_lens[:B].tolist()
    seg = [1 + int(acc[b]) for b in range(B)]
    if max(seg) > MAX_PATH or B > MAX_B:
        # overflow -> eager batched committer (lossless); never truncate the accepted path
        return _fr13_native_committer_all_layers_batched(
            banks_list=banks_list, spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths, accepted_lens=accepted_lens,
            k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
            A_logs=A_logs, dt_biases=dt_biases, num_layers=L, num_spec_decodes=B,
            output_scale=output_scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            burn_node_bank=burn_node_bank, root_node=root_node,
        )

    # ---- lazy-init persistent buffers (graph-stable addresses) ----
    # The remainder arm is part of the signature: an armed and an unarmed graph
    # have different captured bodies, so they must never share a cache entry.
    _remainder = _fr13_cfwd_capture_remainder_on()
    sig = (L, MAX_B, MAX_PATH, num_kh, dim_k, num_vh, dim_v, id(banks_list[0]),
           _remainder)
    st = _FR13_GRAPH_COMMITTER.get(sig)
    if st is None:
        _dt = k_rings.dtype
        st = dict(
            kbuf=torch.zeros(L, MAXT, num_kh, dim_k, device=dev, dtype=_dt),
            vbuf=torch.zeros(L, MAXT, num_vh, dim_v, device=dev, dtype=_dt),
            abuf=torch.full((L, MAXT, num_vh), -1e4, device=dev, dtype=_dt),
            bbuf=torch.zeros(L, MAXT, num_vh, device=dev, dtype=_dt),
            qbuf=torch.zeros(1, MAXT, num_kh, dim_k, device=dev, dtype=_dt),
            cu=torch.tensor([i * MAX_PATH for i in range(MAX_B + 1)], device=dev, dtype=torch.int32),
            ssi=torch.zeros(L, MAX_B, MAX_PATH, device=dev, dtype=torch.int32),
            graph=None,
        )
        _FR13_GRAPH_COMMITTER[sig] = st
    kbuf, vbuf, abuf, bbuf = st["kbuf"], st["vbuf"], st["abuf"], st["bbuf"]
    qbuf, cu_fixed, ssi_buf = st["qbuf"], st["cu"], st["ssi"]

    # ---- FR13_CFWD_CAPTURE_REMAINDER: capture-legal staging buffers ----
    if _remainder:
        if burn_node_bank:
            raise RuntimeError(
                "FR13_CFWD_CAPTURE_REMAINDER requires burn OFF "
                "(burn is eager and mutates the banks outside the graph)"
            )
        RING = int(k_rings.shape[2])
        if "node_mat" not in st:
            # Every one of these is a device tensor at a fixed address, so the
            # captured fill re-reads them on each replay. Staging rows for
            # b >= B are neutral: acc_stage = -1 makes `valid` all-False, which
            # reproduces the untouched (memset) padding of the eager fill.
            st["acc_stage"] = torch.full(
                (MAX_B,), -1, device=dev, dtype=torch.long)
            st["path_stage"] = torch.zeros(
                (MAX_B, MAX_PATH - 1), device=dev, dtype=torch.long)
            st["ssi_stage"] = torch.full(
                (L, MAX_B, 1), SCRATCH, device=dev, dtype=torch.int32)
            st["node_mat"] = torch.zeros(
                (MAX_B, MAX_PATH), device=dev, dtype=torch.long)
            st["ar"] = torch.arange(MAX_PATH, device=dev)
            st["arb"] = torch.arange(MAX_B, device=dev)
            st["ring_ptrs"] = None

    _tm = os.environ.get("FR13_GRAPH_TIMER") == "1"
    if _tm:
        _e0, _e1, _e2, _e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        _e0.record()
    if not _remainder:
        # ---- fill fixed buffers: full re-neutralize (cheap memset) then overwrite real slots ----
        abuf.fill_(-1e4)
        bbuf.zero_()
        kbuf.zero_()
        vbuf.zero_()
        ssi_buf.fill_(SCRATCH)
        for b in range(B):
            _nl = seg[b]
            nodes = torch.cat([
                torch.full((1,), int(root_node), dtype=torch.long, device=dev),
                accepted_paths[b, :int(acc[b])].to(torch.long),
            ])
            s0 = b * MAX_PATH
            kbuf[:, s0:s0 + _nl] = k_rings[:, b, nodes]
            vbuf[:, s0:s0 + _nl] = v_rings[:, b, nodes]
            abuf[:, s0:s0 + _nl] = a_rings[:, b, nodes]
            bbuf[:, s0:s0 + _nl] = b_rings[:, b, nodes]
        # per-layer running row broadcast across all MAX_PATH cols, ALL layers at once (1 op, not L*B)
        ssi_buf[:, :B, :] = spec_state_indices[:, :B, 0:1].to(torch.int32)
    else:
        # FR13_CFWD_CAPTURE_REMAINDER: the ONLY eager work left is staging the
        # three varying device tensors into fixed addresses. Everything the old
        # eager fill did (5 memsets + a Python loop over B issuing 4 gathers and
        # 4 slice copies each + the ssi broadcast) moves inside the graph.
        st["acc_stage"].fill_(-1)
        st["acc_stage"][:B].copy_(accepted_lens[:B])
        st["path_stage"].zero_()
        st["path_stage"][:B].copy_(accepted_paths[:B, :MAX_PATH - 1])
        st["ssi_stage"].fill_(SCRATCH)
        st["ssi_stage"][:, :B].copy_(spec_state_indices[:, :B, 0:1])
    if _tm:
        _e1.record()

    def _fill():
        """Device-only restatement of the eager fill, byte-equivalent to it.

        Position j of request b is real iff j <= accepted_lens[b]; node 0 is the
        root and node j>0 is accepted_paths[b, j-1], which is exactly the
        `cat([root, accepted_paths[b, :acc]])` the eager path builds. Padding
        slots get the state-neutral values (a=-1e4 => decay 1, k=v=b=0 => no
        write), so the `where` covers every element and the five memsets the
        eager path needed are subsumed."""
        nm = st["node_mat"]
        nm[:, 0] = int(root_node)
        nm[:, 1:] = st["path_stage"].clamp(min=0)
        valid = st["ar"].unsqueeze(0) <= st["acc_stage"].unsqueeze(1)
        safe = torch.where(
            valid, nm, torch.zeros_like(nm)).clamp(0, RING - 1)
        bidx = st["arb"].view(MAX_B, 1)
        k_sel = k_rings[:, bidx, safe]
        v_sel = v_rings[:, bidx, safe]
        a_sel = a_rings[:, bidx, safe]
        b_sel = b_rings[:, bidx, safe]
        m4 = valid.view(1, MAX_B, MAX_PATH, 1, 1)
        m3 = valid.view(1, MAX_B, MAX_PATH, 1)
        kbuf.view(L, MAX_B, MAX_PATH, num_kh, dim_k)[:] = torch.where(
            m4, k_sel, torch.zeros_like(k_sel))
        vbuf.view(L, MAX_B, MAX_PATH, num_vh, dim_v)[:] = torch.where(
            m4, v_sel, torch.zeros_like(v_sel))
        abuf.view(L, MAX_B, MAX_PATH, num_vh)[:] = torch.where(
            m3, a_sel, torch.full_like(a_sel, -1e4))
        bbuf.view(L, MAX_B, MAX_PATH, num_vh)[:] = torch.where(
            m3, b_sel, torch.zeros_like(b_sel))
        ssi_buf[:] = st["ssi_stage"]

    def _loop():
        for _L in range(L):
            _sg(
                A_log=A_logs[_L],
                a=abuf[_L].reshape(1, MAXT, num_vh),
                b=bbuf[_L].reshape(1, MAXT, num_vh),
                dt_bias=dt_biases[_L], q=qbuf,
                k=kbuf[_L].reshape(1, MAXT, num_kh, dim_k),
                v=vbuf[_L].reshape(1, MAXT, num_vh, dim_v),
                scale=output_scale, initial_state=banks_list[_L],
                inplace_final_state=True, cu_seqlens=cu_fixed,
                ssm_state_indices=ssi_buf[_L],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

    def _body():
        # FR13_CFWD_CAPTURE_REMAINDER folds the fill into the captured region.
        if _remainder:
            _fill()
        _loop()

    def _ring_ptrs():
        return (
            k_rings.data_ptr(), v_rings.data_ptr(),
            a_rings.data_ptr(), b_rings.data_ptr(),
        )

    if st["graph"] is None:
        # capture ONCE. Warmup(3x)+capture(1x) mutate the running rows 4x with THIS step's data, so save
        # the running rows first, capture, RESTORE them, then replay once for the real (single) commit.
        saved = []
        for _L in range(L):
            col0 = spec_state_indices[_L][:B, 0].to(torch.long)
            saved.append((col0, banks_list[_L][col0].clone()))
        _s = torch.cuda.Stream()
        _s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_s):
            for _ in range(3):
                _body()
        torch.cuda.current_stream().wait_stream(_s)
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            _body()
        for _L in range(L):
            col0, val = saved[_L]
            banks_list[_L][col0] = val
        st["graph"] = gph
        if _remainder:
            # The captured fill dereferences the ring tensors directly, so their
            # addresses are baked into the graph. Record them and fail loud on
            # replay if they ever move -- silently reading a stale ring would
            # commit wrong state with no other symptom.
            st["ring_ptrs"] = _ring_ptrs()
        gph.replay()
        global _FR13_GRAPH_COMMITTER_ANNOUNCED
        if not _FR13_GRAPH_COMMITTER_ANNOUNCED:
            _FR13_GRAPH_COMMITTER_ANNOUNCED = True
            print(
                "[FR13_COMMITTER_GRAPH ENGAGED] captured 48-layer fused_sigmoid loop "
                "(MAX_B=" + str(MAX_B) + " MAX_PATH=" + str(MAX_PATH) + "); replaying per commit",
                flush=True,
            )
        global _FR13_CFWD_CAPTURE_REMAINDER_ANNOUNCED
        if _remainder and not _FR13_CFWD_CAPTURE_REMAINDER_ANNOUNCED:
            _FR13_CFWD_CAPTURE_REMAINDER_ANNOUNCED = True
            print(
                "[FR13_CFWD_CAPTURE_REMAINDER ENGAGED] committer fill folded into "
                "the captured region (MAX_B=" + str(MAX_B)
                + " MAX_PATH=" + str(MAX_PATH) + "); 3 staging copies remain eager",
                flush=True,
            )
    else:
        if _remainder:
            if st.get("ring_ptrs") is None:
                raise RuntimeError(
                    "FR13_CFWD_CAPTURE_REMAINDER: graph was captured without the "
                    "fill; the arm was toggled on after capture. Restart the "
                    "engine with the flag set before the first commit."
                )
            if st["ring_ptrs"] != _ring_ptrs():
                raise RuntimeError(
                    "FR13_CFWD_CAPTURE_REMAINDER: k/v/a/b ring addresses moved "
                    "after capture; the captured fill would read stale memory"
                )
        st["graph"].replay()
    if _tm:
        _e2.record()

    # ---- burn spec rows (eager, Stage 1). fused_sigmoid touched col0 only; burn touches cols 1: ----
    # ALL B requests' spec rows for a layer zeroed in one scatter (48 ops, not L*B).
    if burn_node_bank:
        for _L in range(L):
            rows = spec_state_indices[_L][:B, 1:_spec_cols].reshape(-1).to(torch.long)
            banks_list[_L][rows] = 0.0
    if _tm:
        # DEFERRED drain (2026-07-23): the original per-event
        # torch.cuda.synchronize() was a HOT-PATH SYNC that serialized the
        # pipeline at every commit and contaminated the arm's cfwd/wall
        # numbers. Same pattern as the SG timer: queue the event quad, read
        # elapsed only for query()-complete quads, throttled JSON dump.
        _e3.record()
        _FR13_GT_PENDING.append((_e0, _e1, _e2, _e3))
        while _FR13_GT_PENDING:
            _q = _FR13_GT_PENDING[0]
            try:
                if not _q[3].query():
                    break
            except Exception:
                _FR13_GT_PENDING.pop(0)
                continue
            _FR13_GT_PENDING.pop(0)
            try:
                _FR13_GT_ACCUM[0] += _q[0].elapsed_time(_q[1]) / 1000.0
                _FR13_GT_ACCUM[1] += _q[1].elapsed_time(_q[2]) / 1000.0
                _FR13_GT_ACCUM[2] += _q[2].elapsed_time(_q[3]) / 1000.0
                _FR13_GT_ACCUM[3] += 1
            except Exception:
                pass
        import json as _gt_j
        import time as _gt_tm
        _gt_out = os.environ.get("FR13_GRAPH_TIMER_JSON")
        if _gt_out and (_gt_tm.monotonic() - _FR13_GT_LAST_DUMP[0]) > 5.0:
            _FR13_GT_LAST_DUMP[0] = _gt_tm.monotonic()
            try:
                _gt_tmp = _gt_out + ".tmp"
                with open(_gt_tmp, "w") as _gt_fh:
                    _gt_fh.write(_gt_j.dumps({
                        "fill_s": _FR13_GT_ACCUM[0],
                        "replay_s": _FR13_GT_ACCUM[1],
                        "burn_s": _FR13_GT_ACCUM[2],
                        "n": _FR13_GT_ACCUM[3],
                    }))
                os.replace(_gt_tmp, _gt_out)
            except Exception:
                pass


def launch_tree_gdn_replay(
    *,
    state_bank: torch.Tensor,
    spec_state_indices: torch.Tensor,
    prev_lens: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    a_ring: torch.Tensor,
    b_ring: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_spec_decodes: int,
    output_scale: float,
    use_qk_l2norm_in_kernel: bool = True,
    runrow_commit: bool = False,
    runrow_init: bool = False,
    burn_node_bank: bool = False,
    root_node: int = 0,
) -> None:
    """Launch the FR13 accepted-path replay (the durable-state publish).

    Replaces the legacy all-rows publish + next-step ssm remap under
    FR13_REPLAY_ROUTE: replays root + accepted path from the activation ring
    and writes post-step states to bank LINEAR columns (and always column 0,
    covering the zero-accept path). All inputs must be persistent
    preallocated buffers (graph-stable addresses); per-step pinned scratch is
    the gate-4 failure mode and is banned here.
    """
    if num_spec_decodes <= 0:
        return
    if state_bank.ndim != 4:
        raise ValueError(
            f"state bank must be (rows, num_vh, dim_v, dim_k), got {tuple(state_bank.shape)}"
        )
    bank_rows, num_vh, dim_v, dim_k = state_bank.shape
    if state_bank.dtype != torch.float32:
        raise ValueError(
            f"FR13 replay requires an fp32 GDN state bank, got {state_bank.dtype}"
        )
    if (
        state_bank.stride(3) != 1
        or state_bank.stride(2) != dim_k
        or state_bank.stride(1) != dim_v * dim_k
    ):
        raise ValueError("state bank payload must be row-contiguous")
    if k_ring.ndim != 4 or v_ring.ndim != 4 or a_ring.ndim != 3 or b_ring.ndim != 3:
        raise ValueError(
            "activation ring shapes must be k(B,N,KH,DK)/v(B,N,VH,DV)/a,b(B,N,VH), got "
            f"k={tuple(k_ring.shape)} v={tuple(v_ring.shape)} "
            f"a={tuple(a_ring.shape)} b={tuple(b_ring.shape)}"
        )
    ring_bs, n_pad, num_kh, ring_dim_k = k_ring.shape
    # n_pad<=32: the replay kernels carry NO h_cache=[N_PAD,BV,DIM_K] tile (one
    # [BLOCK_V,DIM_K] tile per program), so their register budget is n_pad-independent
    # and safe at n_pad=32 even at the deployed BV=16 (accept>5 32-node horizon).
    if n_pad > 32 or n_pad & (n_pad - 1):
        raise ValueError(f"ring n_pad must be a power of two <=32, got {n_pad}")
    if not (0 <= int(root_node) < n_pad):
        raise ValueError(f"root_node {root_node} outside ring n_pad {n_pad}")
    if ring_dim_k != dim_k:
        raise ValueError(f"ring k dim {ring_dim_k} != bank dim_k {dim_k}")
    if v_ring.shape != (ring_bs, n_pad, num_vh, dim_v):
        raise ValueError(
            f"v ring shape {tuple(v_ring.shape)} != {(ring_bs, n_pad, num_vh, dim_v)}"
        )
    if a_ring.shape != (ring_bs, n_pad, num_vh) or b_ring.shape != (ring_bs, n_pad, num_vh):
        raise ValueError(
            f"a/b ring shapes must be {(ring_bs, n_pad, num_vh)}, got "
            f"{tuple(a_ring.shape)}/{tuple(b_ring.shape)}"
        )
    if not (
        k_ring.is_contiguous()
        and v_ring.is_contiguous()
        and a_ring.is_contiguous()
        and b_ring.is_contiguous()
    ):
        raise ValueError("activation rings must be contiguous")
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of k heads, got {num_vh}/{num_kh}")
    if ring_bs < num_spec_decodes:
        raise ValueError(
            f"ring batch {ring_bs} < num_spec_decodes {num_spec_decodes}"
        )
    if spec_state_indices.ndim != 2 or spec_state_indices.shape[0] < num_spec_decodes:
        raise ValueError(
            f"spec_state_indices must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(spec_state_indices.shape)}"
        )
    spec_cols = int(spec_state_indices.shape[1])
    if accepted_paths.ndim != 2 or accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            f"accepted_paths must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(accepted_paths.shape)}"
        )
    path_cols = int(accepted_paths.shape[1])
    if path_cols > spec_cols:
        # Defensive width clamp. It was introduced for FR13_SPEC_BLOCKS_CAP,
        # which narrowed the ssi (22 -> 13) while the accepted-paths BUFFER
        # stayed tree-wide; that cap was DELETED 2026-07-25 (dce60d18c), so
        # the two widths now agree and this branch is not expected to fire.
        # It is kept because it is value-identical either way: content is
        # bounded by accepted_len <= MAX_PATH (12) < spec_cols, and every
        # path-col read below is masked by accepted_len, so cols >= spec_cols
        # were only ever masked lanes.
        path_cols = spec_cols
    if prev_lens.numel() < num_spec_decodes or accepted_lens.numel() < num_spec_decodes:
        raise ValueError(
            "prev_lens/accepted_lens must cover num_spec_decodes="
            f"{num_spec_decodes}, got {prev_lens.numel()}/{accepted_lens.numel()}"
        )
    if A_log.numel() < num_vh or dt_bias.numel() < num_vh:
        raise ValueError(
            f"A_log/dt_bias must cover {num_vh} value heads, got "
            f"{A_log.numel()}/{dt_bias.numel()}"
        )
    if _fr13_committer_native_on() and runrow_init:
        # Route the committed-path state rebuild through NATIVE fused_sigmoid_gating (bit-exact to no-spec)
        # instead of the custom replay kernel. Tests whether the gross state-carry corruption (garble root)
        # is in this committer. EAGER only (dynamic gather). Falls through to custom if not runrow_init.
        _fr13_native_committer_replay(
            state_bank=state_bank, spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths, accepted_lens=accepted_lens,
            k_ring=k_ring, v_ring=v_ring, a_ring=a_ring, b_ring=b_ring,
            A_log=A_log, dt_bias=dt_bias, num_spec_decodes=num_spec_decodes,
            output_scale=output_scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            burn_node_bank=burn_node_bank, spec_cols=spec_cols,
            root_node=root_node,
        )
        return
    grid = (int(num_spec_decodes), num_vh, triton.cdiv(dim_v, BV))
    _tree_gdn_replay_kernel[grid](
        k_ring,
        v_ring,
        a_ring,
        b_ring,
        A_log,
        dt_bias,
        state_bank,
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        prev_lens,
        NUM_KH=num_kh,
        NUM_VH=num_vh,
        DIM_K=dim_k,
        DIM_V=dim_v,
        BLOCK_V=BV,
        N_PAD=n_pad,
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        BANK_STRIDE=state_bank.stride(0),
        RING_B_STRIDE_K=k_ring.stride(0),
        RING_N_STRIDE_K=k_ring.stride(1),
        RING_B_STRIDE_V=v_ring.stride(0),
        RING_N_STRIDE_V=v_ring.stride(1),
        RING_B_STRIDE_AB=a_ring.stride(0),
        RING_N_STRIDE_AB=a_ring.stride(1),
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        RAW_GATING=True,
        SCAN_ALIGN=scan_align_on(),
        RUNROW_COMMIT=runrow_commit,
        RUNROW_INIT=runrow_init,
        BURN_NODE_BANK=burn_node_bank,
        ROOT_NODE=int(root_node),
        num_warps=8,
    )


# FR13_REPLAY_GPU_TIMER (diagnostic, default OFF => byte-identical): coarse GPU-time of the accepted-path
# replay per committer step, to settle the 94ms-committer decomposition (replay vs sync-wait/DtoH). A
# non-invasive wrapper -- the timed function body is UNCHANGED; only wall-events are recorded when the
# flag is on. NOTE: uses a synchronize() so it inflates OTHER spans (diagnostic run only, never deploy).
_FR13_REPLAY_GPU_SECONDS = 0.0
_FR13_REPLAY_N = 0


def _fr13_replay_gpu_timed(_orig):
    import functools

    @functools.wraps(_orig)
    def _w(*a, **k):
        import os
        if os.environ.get("FR13_REPLAY_GPU_TIMER", "0") != "1":
            return _orig(*a, **k)
        import json
        _s = torch.cuda.Event(enable_timing=True)
        _e = torch.cuda.Event(enable_timing=True)
        _s.record()
        _r = _orig(*a, **k)
        _e.record()
        _e.synchronize()
        global _FR13_REPLAY_GPU_SECONDS, _FR13_REPLAY_N
        _FR13_REPLAY_GPU_SECONDS += _s.elapsed_time(_e) / 1000.0
        _FR13_REPLAY_N += 1
        if _FR13_REPLAY_N % 50 == 0:
            try:
                json.dump(
                    {"gpu_seconds": _FR13_REPLAY_GPU_SECONDS, "n_spans": _FR13_REPLAY_N},
                    open(os.environ.get("FR13_REPLAY_GPU_TIMER_JSON", "/logs/fr13_replay_gpu.json"), "w"),
                )
            except Exception:
                pass
        return _r

    return _w


launch_tree_gdn_replay = _fr13_replay_gpu_timed(launch_tree_gdn_replay)


def _tree_gdn_replay_all_layers_kernel(
    k_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    bank_anchor,
    bank_off16,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    prev_lens,
    NUM_SPEC: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    N_PAD: tl.constexpr,
    PATH_COLS: tl.constexpr,
    SPEC_COLS: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    RING_L_STRIDE_K: tl.constexpr,
    RING_B_STRIDE_K: tl.constexpr,
    RING_N_STRIDE_K: tl.constexpr,
    RING_L_STRIDE_V: tl.constexpr,
    RING_B_STRIDE_V: tl.constexpr,
    RING_N_STRIDE_V: tl.constexpr,
    RING_L_STRIDE_AB: tl.constexpr,
    RING_B_STRIDE_AB: tl.constexpr,
    RING_N_STRIDE_AB: tl.constexpr,
    SPEC_L_STRIDE: tl.constexpr,
    PREV_L_STRIDE: tl.constexpr,
    GATE_L_STRIDE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    RAW_GATING: tl.constexpr,
    SCAN_ALIGN: tl.constexpr = False,
    RUNROW_COMMIT: tl.constexpr = False,
    RUNROW_INIT: tl.constexpr = False,
    BURN_NODE_BANK: tl.constexpr = False,
):
    # FR13_EAGER_PACK (FIX-2 item 2b): all-layer batched sibling of
    # _tree_gdn_replay_kernel. ONE launch covers every GDN layer; pid0 packs
    # (layer, spec) as layer * NUM_SPEC + spec. Each program's instruction
    # sequence is source-identical to the single-layer kernel (same inlined
    # _gdn_node_step, same constexprs, same num_warps=8); only the base
    # addresses gain a per-layer offset (stacked rings / gates / snapshots,
    # plus an int64 bank OFFSET table relative to the layer-0 bank anchor
    # because each layer's ssm bank is a distinct KV-pool tensor -- see the
    # state_bank addptr note below for why offsets, not raw pointers).
    # Layers are independent: a program reads only
    # its own layer's ring/spec/prev rows and writes only its own layer's
    # bank rows (accepted_paths/lens are shared READ-ONLY), so inter-program
    # concurrency reorders nothing within any program's sequential replay
    # (playbook class 3: no overlapping writes across programs).
    # Class-10 caveat: codegen identity vs the legacy per-layer launch loop
    # is NOT assumed from source identity; it is gated by the int-view byte
    # A/B of bank bytes (never atol) before any live boot.
    pid_lb = tl.program_id(0)
    pid_l = pid_lb // NUM_SPEC
    pid_b = pid_lb % NUM_SPEC
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    # Byte-A/B fix (class 10, GPU gate 2026-06-12): the original
    # `tl.load(bank_ptrs + pid_l).to(tl.pointer_type(tl.float32))` form loses
    # ALL alignment info -- this Triton's AxisInfo does not propagate
    # divisibility through tt.int_to_ptr (verified by container microbench;
    # tl.multiple_of/shift hints on the loaded integer do NOT survive the
    # cast either). The whole kernel then compiles with a scalarized layout
    # (sizePerThread=[1,1], st.global.b32) instead of the legacy kernel's
    # vectorized layout (sizePerThread=[1,4], st.global.v4.b32), which
    # reshapes every tl.sum reduction tree in _gdn_node_step and changes
    # fp32 rounding (~1-2 ULP on published rows; measured, both arms
    # deterministic). Fix: address banks as ANCHOR + ELEMENT OFFSET through
    # tt.addptr, whose AxisInfo math is exact: bank_anchor is the layer-0
    # bank ARG (divisibility 16 from arg specialization) and bank_off16
    # holds (data_ptr - anchor_ptr) // 16 per layer, so `off * 4` fp32
    # elements is structurally 16-byte divisible. Host-side data_ptr()%16
    # fail-loud checks in build_replay_bank_pointer_table keep this exact.
    state_bank = bank_anchor + tl.load(bank_off16 + pid_l) * 4
    k_ring = k_rings + pid_l * RING_L_STRIDE_K
    v_ring = v_rings + pid_l * RING_L_STRIDE_V
    a_ring = a_rings + pid_l * RING_L_STRIDE_AB
    b_ring = b_rings + pid_l * RING_L_STRIDE_AB
    A_log = A_logs + pid_l * GATE_L_STRIDE
    dt_bias = dt_biases + pid_l * GATE_L_STRIDE
    spec_layer = spec_state_indices + pid_l * SPEC_L_STRIDE
    prev_layer = prev_lens + pid_l * PREV_L_STRIDE

    # h0 = same column convention as the scan's in-kernel h0 gather:
    # column clamp(prev_accepted_len - 1, 0). prev_lens is the SCAN-TIME
    # snapshot of the accepted-lens buffer (the committer refills the live
    # buffer with the NEW lens before this kernel launches).
    prev_len = tl.load(prev_layer + pid_b).to(tl.int64)
    if RUNROW_INIT:
        # STATELESS-TREE: prev step's RUNROW_COMMIT wrote the committed leaf into
        # col 0 (the req running row); seed init from there = native semantics.
        h0_col = tl.zeros([], dtype=tl.int64)
    else:
        h0_col = tl.maximum(prev_len - 1, 0)
    h0_row = tl.load(spec_layer + pid_b * SPEC_COLS + h0_col).to(tl.int64)
    # Read the whole h0 tile into registers BEFORE any store: a later
    # publish in this same program may target the h0 bank row itself
    # (publish-overwrites-h0-row case).
    state = tl.load(
        state_bank
        + h0_row * BANK_STRIDE
        + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
        + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    acc_len = tl.load(accepted_lens + pid_b)
    # q is not stored: the q-side ops never touch state, out was already
    # emitted by the scan. A zero q keeps the shared body's signature and
    # constexprs identical; out_i is discarded.
    b_q = tl.zeros((DIM_K,), dtype=tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
    for t in tl.static_range(0, PATH_COLS + 1):
        if t == 0:
            # Root (gdn node 0) replays unconditionally: row 0 must be
            # refreshed even on a ZERO-ACCEPT event (the next h0 read clamps
            # accepted_len-1 to column 0), and the scan applies NO handoff
            # normalization to h0 before the root update.
            active = acc_len >= 0
            node = 0
        else:
            active = (t - 1) < acc_len
            node = tl.load(
                accepted_paths + pid_b * PATH_COLS + (t - 1),
                mask=active,
                other=0,
            ).to(tl.int64)
            node = tl.maximum(node, 0)
            node = tl.minimum(node, N_PAD - 1)
            # Parent-handoff normalization: the scan reads the parent state
            # through tl.sum(tl.where(offs_n == j, h_cache, 0.0), axis=0),
            # which flips -0.0 to +0.0 exactly once per edge. `+ 0.0`
            # reproduces that bit behavior; the root above gets none.
            state = state + 0.0
        b_k = tl.load(
            k_ring
            + pid_b * RING_B_STRIDE_K
            + node * RING_N_STRIDE_K
            + pid_kh * DIM_K
            + offs_k,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            v_ring
            + pid_b * RING_B_STRIDE_V
            + node * RING_N_STRIDE_V
            + pid_vh * DIM_V
            + offs_v,
            mask=active & v_mask,
            other=0.0,
        ).to(tl.float32)
        b_raw_b = tl.load(
            b_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        b_raw_a = tl.load(
            a_ring + pid_b * RING_B_STRIDE_AB + node * RING_N_STRIDE_AB + pid_vh,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        new_state, out_i = _gdn_node_step(
            state,
            b_q,
            b_k,
            b_v,
            0.0,
            0.0,
            b_raw_a,
            b_raw_b,
            b_a_log,
            b_dt_bias,
            OUTPUT_SCALE=OUTPUT_SCALE,
            USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM_IN_KERNEL,
            RAW_GATING=RAW_GATING,
            SCAN_ALIGN=SCAN_ALIGN,
        )
        state = tl.where(active, new_state, state)
        if t == 0:
            dst_col = 0
        else:
            dst_col = t - 1
        dst_row = tl.load(
            spec_layer + pid_b * SPEC_COLS + dst_col,
            mask=active,
            other=0,
        ).to(tl.int64)
        tl.store(
            state_bank
            + dst_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=active & v_mask[:, None],
        )
    if RUNROW_COMMIT:
        # STATELESS-TREE (see single-layer sibling): `state` == committed leaf at
        # loop exit; deposit into col 0 = the req running row (byte-identical bytes
        # to col nacc-1) so col 0 is native's authoritative source.
        rr_row = tl.load(spec_layer + pid_b * SPEC_COLS + 0).to(tl.int64)
        tl.store(
            state_bank
            + rr_row * BANK_STRIDE
            + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state,
            mask=v_mask[:, None],
        )
    if BURN_NODE_BANK:
        # Zero ephemeral spec cols 1..num_spec (col 0 preserved) => zero lifespan.
        for _bc in tl.static_range(1, SPEC_COLS):
            z_row = tl.load(
                spec_layer + pid_b * SPEC_COLS + _bc
            ).to(tl.int64)
            tl.store(
                state_bank
                + z_row * BANK_STRIDE
                + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K
                + offs_k[None, :],
                tl.zeros([BLOCK_V, DIM_K], dtype=tl.float32),
                mask=v_mask[:, None],
            )


def build_replay_bank_pointer_table(
    banks: list[torch.Tensor],
) -> tuple[list[int], tuple[int, int, int, int], int]:
    """Validate per-layer GDN state banks for the batched all-layer replay.

    FR13_EAGER_PACK (FIX-2 item 2b): each layer's ssm bank is a distinct
    KV-pool tensor, so the batched kernel addresses them as the layer-0
    bank ANCHOR plus an int64 device OFFSET table ((ptr - ptr0) // 16,
    derived from this host pointer list; offsets-not-pointers is the
    byte-A/B alignment fix, see the kernel). This helper validates every
    bank exactly like
    launch_tree_gdn_replay (fp32, 4D, row-contiguous payload) plus the
    stacking preconditions (identical shape and stride across layers) and
    returns (host pointer list, bank shape, bank row stride). FAIL-LOUD on
    any precondition miss -- no silent per-layer fallback (playbook class 9).
    The caller must re-assert the host pointer list against the live banks'
    data_ptr() on every commit (cheap Python int compares) before launching.
    """
    if not banks:
        raise ValueError("FR13_EAGER_PACK bank table requires at least one bank")
    shape0 = tuple(banks[0].shape)
    stride0 = banks[0].stride(0)
    ptrs: list[int] = []
    for i, bank in enumerate(banks):
        if bank.ndim != 4:
            raise ValueError(
                f"bank[{i}] must be (rows, num_vh, dim_v, dim_k), got {tuple(bank.shape)}"
            )
        if bank.dtype != torch.float32:
            raise ValueError(
                f"FR13 replay requires fp32 GDN state banks, bank[{i}] is {bank.dtype}"
            )
        rows_i, num_vh_i, dim_v_i, dim_k_i = bank.shape
        if (
            bank.stride(3) != 1
            or bank.stride(2) != dim_k_i
            or bank.stride(1) != dim_v_i * dim_k_i
        ):
            raise ValueError(f"bank[{i}] payload must be row-contiguous")
        if tuple(bank.shape) != shape0 or bank.stride(0) != stride0:
            raise ValueError(
                "FR13_EAGER_PACK stacking precondition failed: bank["
                f"{i}] shape/stride {tuple(bank.shape)}/{bank.stride(0)} != "
                f"bank[0] {shape0}/{stride0}"
            )
        ptr_i = int(bank.data_ptr())
        if ptr_i % 16 != 0:
            # The batched kernel asserts tl.multiple_of(bank_ptr, 16) -- the
            # divisibility a kernel pointer ARG gets from Triton arg
            # specialization. An unaligned bank would make that hint UNSOUND
            # (silent wrong codegen), so fail loud here instead (class 9).
            raise ValueError(
                f"bank[{i}] data_ptr {ptr_i:#x} is not 16-byte aligned; the "
                "batched replay kernel's tl.multiple_of(16) hint would be "
                "unsound"
            )
        ptrs.append(ptr_i)
    return ptrs, (int(shape0[0]), int(shape0[1]), int(shape0[2]), int(shape0[3])), int(stride0)


def _fr13_native_committer_all_layers_device(
    banks_list, spec_state_indices, accepted_paths, accepted_lens,
    k_rings, v_rings, a_rings, b_rings, A_logs, dt_biases,
    num_layers, num_spec_decodes, output_scale, use_qk_l2norm_in_kernel,
    burn_node_bank, root_node=0, max_path=16,
):
    """S1 (=2) IN-CAPTURE committer: the CG body's fixed-shape neutral-padded
    fused_sigmoid sequence with the segment layout built ON DEVICE — no host
    .tolist()/torch.tensor(list) (every other committer flavor is host-layout
    and capture-illegal). Neutral padding semantics (a=-1e4 => decay 1,
    k=v=b=0 => no write) byte-proven by the CG gates. Persistent per-(sig)
    buffers => graph-stable addresses; every fill op is a device op recorded
    into the outer S1 graph, so replays recompute the layout from the live
    accepted_paths/lens device tensors. burn must be OFF (stateless-tree
    default; the S1 route asserts it) — fused_sigmoid touches col0 only."""
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update as _sg,
    )
    if burn_node_bank:
        raise RuntimeError("S1 device committer requires burn OFF")
    L = int(num_layers)
    B = int(num_spec_decodes)
    dev = k_rings.device
    num_kh, dim_k = int(k_rings.shape[-2]), int(k_rings.shape[-1])
    num_vh, dim_v = int(v_rings.shape[-2]), int(v_rings.shape[-1])
    MAX_PATH = int(max_path)
    MAXT = B * MAX_PATH
    SCRATCH = int(banks_list[0].shape[0]) - 1
    RING = int(k_rings.shape[2])
    sig = ("s1dev", L, B, MAX_PATH, num_kh, dim_k, num_vh, dim_v, id(banks_list[0]))
    st = _FR13_GRAPH_COMMITTER.get(sig)
    if st is None:
        # device-op-only init (capture-legal on first use): no tensor-from-list
        _dt = k_rings.dtype
        st = dict(
            kbuf=torch.zeros(L, MAXT, num_kh, dim_k, device=dev, dtype=_dt),
            vbuf=torch.zeros(L, MAXT, num_vh, dim_v, device=dev, dtype=_dt),
            abuf=torch.full((L, MAXT, num_vh), -1e4, device=dev, dtype=_dt),
            bbuf=torch.zeros(L, MAXT, num_vh, device=dev, dtype=_dt),
            qbuf=torch.zeros(1, MAXT, num_kh, dim_k, device=dev, dtype=_dt),
            cu=(torch.arange(B + 1, device=dev, dtype=torch.int64) * MAX_PATH).to(torch.int32),
            ssi=torch.zeros(L, B, MAX_PATH, device=dev, dtype=torch.int32),
            ar=torch.arange(MAX_PATH, device=dev),
            arb=torch.arange(B, device=dev),
        )
        _FR13_GRAPH_COMMITTER[sig] = st
    # ---- neutralize, then masked-overwrite real slots (all device ops) ----
    st["abuf"].fill_(-1e4)
    st["bbuf"].zero_()
    st["kbuf"].zero_()
    st["vbuf"].zero_()
    st["ssi"].fill_(SCRATCH)
    acc = accepted_lens[:B].to(torch.long)
    node_mat = torch.empty(B, MAX_PATH, dtype=torch.long, device=dev)
    node_mat[:, 0] = int(root_node)
    node_mat[:, 1:] = accepted_paths[:B, : MAX_PATH - 1].to(torch.long).clamp(min=0)
    valid = st["ar"].unsqueeze(0) <= acc.unsqueeze(1)          # [B, MAX_PATH]
    safe_nodes = torch.where(
        valid, node_mat, torch.zeros_like(node_mat)).clamp(0, RING - 1)
    bidx = st["arb"].view(B, 1)
    k_sel = k_rings[:, bidx, safe_nodes]                        # [L,B,MP,kh,dk]
    v_sel = v_rings[:, bidx, safe_nodes]
    a_sel = a_rings[:, bidx, safe_nodes]                        # [L,B,MP,vh]
    b_sel = b_rings[:, bidx, safe_nodes]
    m4 = valid.view(1, B, MAX_PATH, 1, 1)
    m3 = valid.view(1, B, MAX_PATH, 1)
    st["kbuf"].view(L, B, MAX_PATH, num_kh, dim_k)[:] = torch.where(
        m4, k_sel, torch.zeros_like(k_sel))
    st["vbuf"].view(L, B, MAX_PATH, num_vh, dim_v)[:] = torch.where(
        m4, v_sel, torch.zeros_like(v_sel))
    st["abuf"].view(L, B, MAX_PATH, num_vh)[:] = torch.where(
        m3, a_sel, torch.full_like(a_sel, -1e4))
    st["bbuf"].view(L, B, MAX_PATH, num_vh)[:] = torch.where(
        m3, b_sel, torch.zeros_like(b_sel))
    st["ssi"][:, :, :] = spec_state_indices[:, :B, 0:1].to(torch.int32)
    for _L in range(L):
        _sg(
            A_log=A_logs[_L],
            a=st["abuf"][_L].reshape(1, MAXT, num_vh),
            b=st["bbuf"][_L].reshape(1, MAXT, num_vh),
            dt_bias=dt_biases[_L], q=st["qbuf"],
            k=st["kbuf"][_L].reshape(1, MAXT, num_kh, dim_k),
            v=st["vbuf"][_L].reshape(1, MAXT, num_vh, dim_v),
            scale=output_scale, initial_state=banks_list[_L],
            inplace_final_state=True, cu_seqlens=st["cu"],
            ssm_state_indices=st["ssi"][_L],
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


_FR13_FIXED32_COMMITTER = {}
_FR13_FIXED32_COMMITTER_GATE_COEFFS: dict[tuple, torch.Tensor] = {}
_FR13_FIXED32_COMMITTER_GATE_PRECOMPUTE_LAUNCHES = 0
_FR13_FIXED32_COMMITTER_FAST_ROUTE: dict[str, object] = {}
_FR13_FIXED32_COMMITTER_METADATA_LEASE: dict[str, object] = {}
_FR13_FIXED32_COMMITTER_CALLBACKS = []
_FR13_FIXED32_COMMITTER_COUNTERS = {
    "captures": 0,
    "actual_replays_enqueued": 0,
    "actual_replays_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
    "metadata_fusion_published_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
    "metadata_fusion_consumed_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
    "metadata_fusion_fallbacks_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
    "direct_metadata_published_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
    "direct_metadata_consumed_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
}
_FR13_FIXED32_COMMITTER_ANNOUNCED = False
_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY = None
_FR13_FIXED32_COMMITTER_WARMUP: dict[str, object] = {}


def _fr13_fixed32_committer_stream_key(device) -> tuple[str, int]:
    stream = torch.cuda.current_stream(device)
    return str(device), int(stream.cuda_stream)


def _fr13_fixed32_committer_metadata_lease_key(
    *,
    batch: int,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    committer_paths: torch.Tensor,
    committer_lens: torch.Tensor,
    validation_bank_rows: int,
    validation_guard: torch.Tensor | None = None,
    stream_key: tuple[str, int] | None = None,
) -> tuple:
    """Bind one conv validation/copy to one exact committer replay."""
    if stream_key is None:
        stream_key = _fr13_fixed32_committer_stream_key(accepted_paths.device)
    tensors = (
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        committer_paths,
        committer_lens,
    )
    key = (
        "fixed32_committer_metadata_fusion_v1",
        int(batch),
        int(validation_bank_rows),
        tuple(stream_key),
        tuple(int(tensor.data_ptr()) for tensor in tensors),
        (
            tuple(int(value) for value in spec_state_indices.shape),
            (int(batch), 16),
            (int(batch),),
            tuple(int(value) for value in committer_paths.shape),
            tuple(int(value) for value in committer_lens.shape),
        ),
        tuple(tuple(int(value) for value in tensor.stride()) for tensor in tensors),
        tuple(str(tensor.dtype) for tensor in tensors),
        tuple(str(tensor.device) for tensor in tensors),
    )
    if validation_guard is None:
        return key
    return key + (
        (
            "sticky_guard_v1",
            int(validation_guard.data_ptr()),
            tuple(int(value) for value in validation_guard.shape),
            tuple(int(value) for value in validation_guard.stride()),
            str(validation_guard.dtype),
            str(validation_guard.device),
        ),
    )


def _fr13_fixed32_committer_publish_metadata_lease(key: tuple) -> None:
    if _FR13_FIXED32_COMMITTER_METADATA_LEASE:
        raise RuntimeError(
            "FR13 fixed32 metadata-fusion prior lease was not consumed"
        )
    _FR13_FIXED32_COMMITTER_METADATA_LEASE["key"] = key


def _fr13_fixed32_committer_consume_metadata_lease(key: tuple) -> bool:
    if not _FR13_FIXED32_COMMITTER_METADATA_LEASE:
        return False
    prior = _FR13_FIXED32_COMMITTER_METADATA_LEASE.get("key")
    if prior is None or prior != key:
        raise RuntimeError(
            "FR13 fixed32 metadata-fusion lease mismatch; refusing fallback"
        )
    _FR13_FIXED32_COMMITTER_METADATA_LEASE.clear()
    return True


def _fr13_fixed32_committer_publish_direct_metadata_lease(key: tuple) -> None:
    """Publish validation ownership for the direct-input graph route."""
    _fr13_fixed32_committer_publish_metadata_lease(key)


def _fr13_fixed32_committer_consume_direct_metadata_lease(key: tuple) -> bool:
    """Consume validation ownership for the direct-input graph route."""
    return _fr13_fixed32_committer_consume_metadata_lease(key)


def _fr13_fixed32_committer_metadata_fusion_state(
    *,
    batch: int,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
) -> tuple[dict, int] | None:
    """Resolve the armed B-specific destinations for the preceding conv launch."""
    route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    if not isinstance(route, dict):
        return None
    state = route.get("states_by_batch", {}).get(int(batch))
    if not isinstance(state, dict) or not state.get("metadata_copy_fusion", False):
        return None
    destination_paths = state.get("accepted_paths")
    destination_lens = state.get("accepted_lens")
    if (
        route.get("spec_state_indices") is not spec_state_indices
        or int(route.get("accepted_paths_data_ptr", -1))
        != int(accepted_paths.data_ptr())
        or int(route.get("accepted_lens_data_ptr", -1))
        != int(accepted_lens.data_ptr())
        or tuple(int(value) for value in accepted_paths.shape)
        != (int(route.get("capacity", -1)), 16)
        or tuple(int(value) for value in accepted_lens.shape)
        != (int(route.get("capacity", -1)),)
        or accepted_paths.dtype != torch.int32
        or accepted_lens.dtype != torch.int32
        or not accepted_paths.is_contiguous()
        or not accepted_lens.is_contiguous()
        or not torch.is_tensor(destination_paths)
        or not torch.is_tensor(destination_lens)
        or tuple(destination_paths.shape) != (int(batch), 16)
        or tuple(destination_lens.shape) != (int(batch),)
        or destination_paths.dtype != torch.int32
        or destination_lens.dtype != torch.int32
        or destination_paths.device != accepted_paths.device
        or destination_lens.device != accepted_lens.device
        or int(destination_paths.data_ptr()) == int(accepted_paths.data_ptr())
        or int(destination_lens.data_ptr()) == int(accepted_lens.data_ptr())
    ):
        raise RuntimeError(
            "FR13 fixed32 metadata-fusion persistent operand drift"
        )
    validation_bank_rows = min(
        int(state["bank_rows"]),
        int(_FR13_FIXED32_CONV_PREGATHER["state"]["anchor"].shape[0]),
    )
    if validation_bank_rows <= 0:
        raise RuntimeError("FR13 fixed32 metadata-fusion has no valid bank rows")
    return state, validation_bank_rows


def _fr13_fixed32_committer_direct_metadata_state(
    *,
    batch: int,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
) -> tuple[dict, int] | None:
    """Resolve a graph captured directly against persistent TAW metadata."""
    route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    if not isinstance(route, dict):
        return None
    state = route.get("states_by_batch", {}).get(int(batch))
    if not isinstance(state, dict) or not state.get("direct_metadata", False):
        return None
    graph_paths = state.get("direct_accepted_paths")
    graph_lens = state.get("direct_accepted_lens")
    sticky_ok = state.get("sticky_guard_ok")
    sticky_guard_invalid = state.get("sticky_guard", False) and (
        not torch.is_tensor(sticky_ok)
        or sticky_ok.dtype != torch.int32
        or tuple(sticky_ok.shape) != ()
        or not sticky_ok.is_contiguous()
        or sticky_ok.device != accepted_paths.device
        or int(sticky_ok.data_ptr())
        != int(state.get("sticky_guard_ok_data_ptr", -1))
    )
    if (
        route.get("spec_state_indices") is not spec_state_indices
        or int(route.get("accepted_paths_data_ptr", -1))
        != int(accepted_paths.data_ptr())
        or int(route.get("accepted_lens_data_ptr", -1))
        != int(accepted_lens.data_ptr())
        or tuple(int(value) for value in accepted_paths.shape)
        != (int(route.get("capacity", -1)), 16)
        or tuple(int(value) for value in accepted_lens.shape)
        != (int(route.get("capacity", -1)),)
        or accepted_paths.dtype != torch.int32
        or accepted_lens.dtype != torch.int32
        or not accepted_paths.is_contiguous()
        or not accepted_lens.is_contiguous()
        or not torch.is_tensor(graph_paths)
        or not torch.is_tensor(graph_lens)
        or tuple(graph_paths.shape) != (int(batch), 16)
        or tuple(graph_lens.shape) != (int(batch),)
        or graph_paths.dtype != torch.int32
        or graph_lens.dtype != torch.int32
        or graph_paths.device != accepted_paths.device
        or graph_lens.device != accepted_lens.device
        or int(graph_paths.data_ptr()) != int(accepted_paths.data_ptr())
        or int(graph_lens.data_ptr()) != int(accepted_lens.data_ptr())
        or sticky_guard_invalid
    ):
        raise RuntimeError(
            "FR13 fixed32 direct-metadata persistent operand drift"
        )
    validation_bank_rows = min(
        int(state["bank_rows"]),
        int(_FR13_FIXED32_CONV_PREGATHER["state"]["anchor"].shape[0]),
    )
    if validation_bank_rows <= 0:
        raise RuntimeError("FR13 fixed32 direct-metadata has no valid bank rows")
    return state, validation_bank_rows


def _fr13_fixed32_tensor_bits_equal(
    left: torch.Tensor, right: torch.Tensor
) -> bool:
    return bool(
        torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def fixed32_committer_counters() -> dict[str, object]:
    """Return enqueue/capture facts recorded by the fixed committer route."""
    preseeded_batches = tuple(sorted({
        int(state["batch"]) for state in _FR13_FIXED32_COMMITTER.values()
    }))
    ready_capacities = {
        int(state["batch"]): max(
            int(state.get("preseed_capacity", 0)),
            0,
        )
        for state in _FR13_FIXED32_COMMITTER.values()
    }
    maximum_ready_capacity = max(ready_capacities.values(), default=0)
    required_capacity = _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY
    fast_route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    states_by_batch = (
        fast_route.get("states_by_batch", {})
        if isinstance(fast_route, dict)
        else {}
    )
    layer_batch_gate_passed_by_batch = {
        int(batch): int(bool(state.get("layer_batch_byte_gate_passed", False)))
        for batch, state in states_by_batch.items()
    }
    layer_batch_gate_attempts_by_batch = {
        int(batch): int(state.get("layer_batch_byte_gate_attempts", -1))
        for batch, state in states_by_batch.items()
    }
    layer_batch_gate_coverage_mask_by_batch = {
        int(batch): int(state.get("layer_batch_byte_gate_coverage_mask", -1))
        for batch, state in states_by_batch.items()
    }
    fast_route_ready = (
        fast_route is not None
        and fast_route["mode"] == _FR13_FIXED32_MODE
        and int(fast_route["capacity"]) == required_capacity
    )
    return {
        "captures": int(_FR13_FIXED32_COMMITTER_COUNTERS["captures"]),
        "actual_replays_enqueued": int(
            _FR13_FIXED32_COMMITTER_COUNTERS["actual_replays_enqueued"]
        ),
        "actual_replays_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS["actual_replays_by_batch"]
        ),
        "metadata_fusion_published_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS[
                "metadata_fusion_published_by_batch"
            ]
        ),
        "metadata_fusion_consumed_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS[
                "metadata_fusion_consumed_by_batch"
            ]
        ),
        "metadata_fusion_fallbacks_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS[
                "metadata_fusion_fallbacks_by_batch"
            ]
        ),
        "direct_metadata_published_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS[
                "direct_metadata_published_by_batch"
            ]
        ),
        "direct_metadata_consumed_by_batch": dict(
            _FR13_FIXED32_COMMITTER_COUNTERS[
                "direct_metadata_consumed_by_batch"
            ]
        ),
        "gate_precompute_launches": int(
            _FR13_FIXED32_COMMITTER_GATE_PRECOMPUTE_LAUNCHES
        ),
        "layer_batch_gate_passed_by_batch": (
            layer_batch_gate_passed_by_batch
        ),
        "layer_batch_gate_attempts_by_batch": (
            layer_batch_gate_attempts_by_batch
        ),
        "layer_batch_gate_coverage_mask_by_batch": (
            layer_batch_gate_coverage_mask_by_batch
        ),
        "preseeded_graphs": len(_FR13_FIXED32_COMMITTER),
        "preseeded_batches": preseeded_batches,
        "ready_capacities": ready_capacities,
        "maximum_ready_capacity": maximum_ready_capacity,
        "required_capacity": required_capacity,
        "fast_route_ready": fast_route_ready,
        "all_batches_ready": (
            fast_route_ready
            and tuple(range(1, required_capacity + 1))
            == tuple(sorted(
                batch
                for batch, capacity in ready_capacities.items()
                if capacity == required_capacity
            ))
            and all(
                ready_capacities[batch] == required_capacity
                for batch in range(1, required_capacity + 1)
            )
        ),
    }


def fixed32_committer_warmup_counters() -> dict[str, object]:
    """Return unmeasured boot-replay evidence for the current fast route."""
    route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    conv_state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    evidence = _FR13_FIXED32_COMMITTER_WARMUP.get("evidence")
    if not isinstance(evidence, dict):
        return {
            "ready": False,
            "classification": "unmeasured_boot",
            "mode": _FR13_FIXED32_MODE,
            "max_batch_size": _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY,
            "batches": (),
            "replays": 0,
            "conv_commit_direct_launches": 0,
            "conv_commit_gather_launches": 0,
            "conv_commit_scatter_launches": 0,
            "route_lease_current": False,
            "bank_state_restored": False,
            "conv_bank_state_restored": False,
            "conv_staging_state_restored": False,
            "alias_destination_contract": None,
            "input_state_restored": False,
            "measured_state_restored": False,
        }
    result = dict(evidence)
    result["route_lease_current"] = (
        route is not None
        and route is _FR13_FIXED32_COMMITTER_WARMUP.get("route")
        and conv_state is _FR13_FIXED32_COMMITTER_WARMUP.get("conv_state")
    )
    result["ready"] = bool(
        result.get("ready")
        and result.get("classification") == "unmeasured_boot"
        and result.get("mode") == _FR13_FIXED32_MODE
        and int(result.get("max_batch_size", -1))
        == int(_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY or 0)
        and tuple(result.get("batches", ()))
        == tuple(
            range(
                1,
                int(_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY or 0) + 1,
            )
        )
        and int(result.get("replays", -1))
        == int(_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY or 0)
        and int(result.get("conv_commit_direct_launches", -1))
        == int(_FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY or 0)
        and int(result.get("conv_commit_gather_launches", -1)) == 0
        and int(result.get("conv_commit_scatter_launches", -1)) == 0
        and result["route_lease_current"]
        and result.get("bank_state_restored") is True
        and result.get("conv_bank_state_restored") is True
        and result.get("conv_staging_state_restored") is True
        and result.get("alias_destination_contract")
        == "exact_alias_only_16x3"
        and result.get("input_state_restored") is True
        and result.get("measured_state_restored") is True
        and result.get("scratch_overwrite_proven") is True
        and tuple(result.get("scratch_restored", ()))
        == ("accepted_paths", "accepted_lens", "node_mat", "qbuf")
        and tuple(result.get("scratch_fully_overwritten", ()))
        == ("abuf", "bbuf", "kbuf", "vbuf", "ssi")
        and tuple(result.get("scratch_immutable", ()))
        == ("cu", "path_offsets", "batch_offsets", "graph", "scratch")
    )
    return result


def register_fixed32_committer_replay_callback(callback) -> None:
    """Register a hook invoked only after an actual graph replay is enqueued."""
    if not callable(callback):
        raise TypeError("fixed32 committer replay callback must be callable")
    if callback not in _FR13_FIXED32_COMMITTER_CALLBACKS:
        _FR13_FIXED32_COMMITTER_CALLBACKS.append(callback)


def unregister_fixed32_committer_replay_callback(callback) -> None:
    """Remove a previously registered replay hook."""
    try:
        _FR13_FIXED32_COMMITTER_CALLBACKS.remove(callback)
    except ValueError:
        pass


def _fr13_fixed32_committer_signature(
    *,
    banks_list,
    spec_state_indices,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_spec_decodes,
    output_scale,
    use_qk_l2norm_in_kernel,
) -> tuple:
    """Preseed-only pointer-complete signature for captured operands."""
    return (
        "fixed32_committer_v1",
        str(k_rings.device),
        int(num_spec_decodes),
        tuple(int(bank.data_ptr()) for bank in banks_list),
        int(spec_state_indices.data_ptr()),
        int(k_rings.data_ptr()),
        int(k_norm_rings.data_ptr()) if k_norm_rings is not None else 0,
        int(gate_rings.data_ptr()) if gate_rings is not None else 0,
        int(v_rings.data_ptr()),
        int(a_rings.data_ptr()),
        int(b_rings.data_ptr()),
        int(A_logs.data_ptr()),
        int(dt_biases.data_ptr()),
        tuple(int(value) for value in k_rings.shape),
        (
            tuple(int(value) for value in k_norm_rings.shape)
            if k_norm_rings is not None
            else None
        ),
        (
            tuple(int(value) for value in gate_rings.shape)
            if gate_rings is not None
            else None
        ),
        tuple(int(value) for value in v_rings.shape),
        tuple(int(value) for value in a_rings.shape),
        tuple(int(value) for value in b_rings.shape),
        tuple(
            (
                tuple(int(value) for value in bank.shape),
                tuple(int(value) for value in bank.stride()),
                str(bank.dtype),
            )
            for bank in banks_list
        ),
        str(k_rings.dtype),
        str(k_norm_rings.dtype) if k_norm_rings is not None else None,
        str(gate_rings.dtype) if gate_rings is not None else None,
        str(v_rings.dtype),
        str(a_rings.dtype),
        str(b_rings.dtype),
        str(A_logs.dtype),
        str(dt_biases.dtype),
        float(output_scale),
        bool(use_qk_l2norm_in_kernel),
    )


def _fr13_fixed32_committer_identity_key(
    *,
    banks_list,
    spec_state_indices,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    output_scale,
    use_qk_l2norm_in_kernel,
) -> tuple:
    """Constant-size identity key for the preseeded measured route."""
    return (
        "fixed32_committer_fast_v1",
        _FR13_FIXED32_MODE,
        id(banks_list),
        id(banks_list[0]),
        id(banks_list[-1]),
        id(spec_state_indices),
        id(k_rings),
        id(k_norm_rings),
        id(gate_rings),
        id(v_rings),
        id(a_rings),
        id(b_rings),
        id(A_logs),
        id(dt_biases),
        float(output_scale),
        bool(use_qk_l2norm_in_kernel),
    )


def _validate_fixed32_committer_contract(
    *,
    banks_list,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_layers,
    num_spec_decodes,
    burn_node_bank,
    root_node=0,
    max_path=16,
) -> tuple[int, int, int, int]:
    """Validate fixed32 graph shapes without reading dynamic device values."""
    layers = int(num_layers)
    batch = int(num_spec_decodes)
    if _FR13_FIXED32_MODE is None:
        raise RuntimeError(
            "fixed32 committer requested without FR13_FIXED32_MODE"
        )
    if layers != 48:
        raise ValueError(
            f"FR13_FIXED32_COMMIT_DEVICE_FILL requires 48 layers, got {layers}"
        )
    if batch not in (1, 2, 3, 4):
        raise ValueError(
            f"FR13_FIXED32_COMMIT_DEVICE_FILL requires B=1..4, got {batch}"
        )
    if int(max_path) != 16 or int(root_node) != 0:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires path cap 16/root 0, "
            f"got cap={max_path} root={root_node}"
        )
    if burn_node_bank:
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires burn_node_bank=False"
        )
    if not isinstance(banks_list, (list, tuple)) or len(banks_list) != 48:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires exactly 48 state banks"
        )
    _bank_ptrs, bank_shape, _bank_stride = build_replay_bank_pointer_table(
        list(banks_list)
    )
    device = k_rings.device
    if device.type != "cuda":
        raise ValueError(
            f"FR13_FIXED32_COMMIT_DEVICE_FILL requires CUDA, got {device}"
        )
    for label, tensor in (
        ("spec_state_indices", spec_state_indices),
        ("accepted_paths", accepted_paths),
        ("accepted_lens", accepted_lens),
        ("k_rings", k_rings),
        ("v_rings", v_rings),
        ("a_rings", a_rings),
        ("b_rings", b_rings),
        ("A_logs", A_logs),
        ("dt_biases", dt_biases),
    ):
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"FR13_FIXED32_COMMIT_DEVICE_FILL {label} is not a tensor"
            )
        if tensor.device != device:
            raise ValueError(
                "FR13_FIXED32_COMMIT_DEVICE_FILL device mismatch: "
                f"{label}={tensor.device} rings={device}"
            )
    for index, bank in enumerate(banks_list):
        if bank.device != device:
            raise ValueError(
                "FR13_FIXED32_COMMIT_DEVICE_FILL bank device mismatch: "
                f"bank[{index}]={bank.device} rings={device}"
            )
    if accepted_paths.ndim != 2 or tuple(accepted_paths.shape) != (
        batch,
        16,
    ):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires accepted_paths "
            f"shape={(batch, 16)}, got {tuple(accepted_paths.shape)}"
        )
    if accepted_lens.ndim != 1 or accepted_lens.numel() != batch:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires accepted_lens "
            f"shape={(batch,)}, got {tuple(accepted_lens.shape)}"
        )
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if accepted_paths.dtype not in integer_dtypes or (
        accepted_lens.dtype not in integer_dtypes
    ):
        raise TypeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL accepted paths/lens must be "
            f"integer, got {accepted_paths.dtype}/{accepted_lens.dtype}"
        )
    if spec_state_indices.dtype not in integer_dtypes:
        raise TypeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL spec_state_indices must be "
            f"integer, got {spec_state_indices.dtype}"
        )
    if not accepted_paths.is_contiguous() or not accepted_lens.is_contiguous():
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL accepted inputs must be contiguous"
        )
    if k_rings.ndim != 5:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL k_rings must be "
            f"(48,B,32,KH,DK), got {tuple(k_rings.shape)}"
        )
    ring_layers, ring_batch, ring_rows, num_kh, dim_k = (
        int(value) for value in k_rings.shape
    )
    if ring_layers != 48 or ring_batch < batch or ring_rows != 32:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires ring prefix "
            f"(48,>={batch},32), got {tuple(k_rings.shape[:3])}"
        )
    k_norm_reuse = _fr13_fixed32_committer_knorm_ring_requested()
    if k_norm_reuse:
        if (
            not torch.is_tensor(k_norm_rings)
            or k_norm_rings.device != device
            or tuple(k_norm_rings.shape)
            != (48, ring_batch, 32, num_kh)
            or k_norm_rings.dtype != torch.float32
            or not k_norm_rings.is_contiguous()
        ):
            raise ValueError(
                "FR13 fixed32 committer K-norm ring must be contiguous "
                "FP32 (48,B,32,KH) on the activation-ring device"
            )
    elif k_norm_rings is not None:
        raise RuntimeError(
            "FR13 fixed32 committer received a K-norm ring while its arm "
            "is disabled"
        )
    gate_reuse = _fr13_fixed32_committer_gate_ring_requested()
    if gate_reuse and not k_norm_reuse:
        raise RuntimeError(
            "FR13 fixed32 committer gate reuse requires K-norm reuse"
        )
    if spec_state_indices.ndim != 3 or (
        int(spec_state_indices.shape[0]) != 48
        or int(spec_state_indices.shape[1]) < batch
        or int(spec_state_indices.shape[2]) < 1
    ):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL invalid spec_state_indices "
            f"shape={tuple(spec_state_indices.shape)}"
        )
    num_vh = int(v_rings.shape[-2]) if v_rings.ndim == 5 else -1
    dim_v = int(v_rings.shape[-1]) if v_rings.ndim == 5 else -1
    expected_v = (48, ring_batch, 32, num_vh, dim_v)
    expected_ab = (48, ring_batch, 32, num_vh)
    expected_gate = (48, ring_batch, 32, num_vh, 2)
    if tuple(v_rings.shape) != expected_v:
        raise ValueError(
            f"FR13_FIXED32_COMMIT_DEVICE_FILL v shape must be {expected_v}, "
            f"got {tuple(v_rings.shape)}"
        )
    if tuple(a_rings.shape) != expected_ab or tuple(b_rings.shape) != expected_ab:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL a/b shapes must be "
            f"{expected_ab}, got {tuple(a_rings.shape)}/{tuple(b_rings.shape)}"
        )
    if gate_reuse:
        if (
            not torch.is_tensor(gate_rings)
            or gate_rings.device != device
            or tuple(gate_rings.shape) != expected_gate
            or gate_rings.dtype != torch.float32
            or not gate_rings.is_contiguous()
        ):
            raise ValueError(
                "FR13 fixed32 committer gate ring must be contiguous FP32 "
                f"{expected_gate} on the activation-ring device"
            )
    elif gate_rings is not None:
        raise RuntimeError(
            "FR13 fixed32 committer received a gate ring while its arm is "
            "disabled"
        )
    _bank_rows, bank_vh, bank_dim_v, bank_dim_k = bank_shape
    if (
        num_vh != bank_vh
        or dim_v != bank_dim_v
        or dim_k != bank_dim_k
    ):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL ring/bank payload mismatch: "
            f"ring={(num_vh, dim_v, dim_k)} "
            f"bank={(bank_vh, bank_dim_v, bank_dim_k)}"
        )
    if not (
        k_rings.dtype
        == v_rings.dtype
        == a_rings.dtype
        == b_rings.dtype
    ):
        raise TypeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL ring dtypes differ: "
            f"{k_rings.dtype}/{v_rings.dtype}/{a_rings.dtype}/"
            f"{b_rings.dtype}"
        )
    if num_vh <= 0 or num_kh <= 0 or num_vh % num_kh:
        raise ValueError(
            f"FR13_FIXED32_COMMIT_DEVICE_FILL invalid heads {num_vh}/{num_kh}"
        )
    if tuple(A_logs.shape) != (48, num_vh) or (
        tuple(dt_biases.shape) != (48, num_vh)
    ):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL gates must have shape "
            f"{(48, num_vh)}, got {tuple(A_logs.shape)}/"
            f"{tuple(dt_biases.shape)}"
        )
    if not all(tensor.is_contiguous() for tensor in (
        spec_state_indices,
        k_rings,
        v_rings,
        a_rings,
        b_rings,
        A_logs,
        dt_biases,
    )):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL captured tensors must be contiguous"
        )
    return num_kh, dim_k, num_vh, dim_v


def _fr13_fixed32_committer_fast_state(
    *,
    banks_list,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_layers,
    num_spec_decodes,
    output_scale,
    use_qk_l2norm_in_kernel,
    burn_node_bank,
) -> tuple[dict, int]:
    """Resolve one preseeded graph without persistent pointer/shape walks."""
    route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    if route is None:
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL missing all-B fast-route preseed"
        )
    batch = int(num_spec_decodes)
    capacity = int(route["capacity"])
    required_capacity = _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY
    if (
        _FR13_FIXED32_MODE != route["mode"]
        or required_capacity != capacity
    ):
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL fixed route mode/capacity drift: "
            f"mode={_FR13_FIXED32_MODE}/{route['mode']} "
            f"capacity={required_capacity}/{capacity}"
        )
    if int(num_layers) != 48 or burn_node_bank:
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL fixed route requires "
            "num_layers=48 and burn_node_bank=False"
        )
    if not 1 <= batch <= capacity:
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL batch exceeds preseeded server "
            f"capacity: B={batch} capacity={capacity}"
        )
    if (
        not isinstance(banks_list, tuple)
        or len(banks_list) != 48
        or banks_list is not route["banks"]
        or banks_list[0] is not route["bank_anchor"]
        or banks_list[-1] is not route["bank_tail"]
    ):
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL persistent bank-container "
            "identity drift; reuse the exact preseeded 48-bank tuple"
        )
    identity_key = _fr13_fixed32_committer_identity_key(
        banks_list=banks_list,
        spec_state_indices=spec_state_indices,
        k_rings=k_rings,
        k_norm_rings=k_norm_rings,
        gate_rings=gate_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    if identity_key != route["identity_key"]:
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL persistent operand identity "
            "drift; reuse the exact preseeded spec/ring/gate tensors and "
            "fixed scalar options"
        )
    state = route["states_by_batch"].get(batch)
    if (
        state is None
        or state.get("graph") is None
        or int(state.get("preseed_capacity", 0)) != capacity
    ):
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL missing preseeded graph for "
            f"B={batch} at server capacity {capacity}"
        )
    if not torch.is_tensor(accepted_paths) or not torch.is_tensor(accepted_lens):
        raise TypeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL accepted paths/lens must be tensors"
        )
    if (
        tuple(accepted_paths.shape) != (batch, 16)
        or tuple(accepted_lens.shape) != (batch,)
        or not accepted_paths.is_contiguous()
        or not accepted_lens.is_contiguous()
        or accepted_paths.device != route["device"]
        or accepted_lens.device != route["device"]
        or accepted_paths.dtype != state["accepted_paths"].dtype
        or accepted_lens.dtype != state["accepted_lens"].dtype
        or (
            state.get("direct_metadata", False)
            and (
                int(accepted_paths.data_ptr())
                != int(state["direct_accepted_paths"].data_ptr())
                or int(accepted_lens.data_ptr())
                != int(state["direct_accepted_lens"].data_ptr())
            )
        )
        or (
            state.get("sticky_guard", False)
            and (
                not torch.is_tensor(state.get("sticky_guard_ok"))
                or state["sticky_guard_ok"].dtype != torch.int32
                or tuple(state["sticky_guard_ok"].shape) != ()
                or not state["sticky_guard_ok"].is_contiguous()
                or state["sticky_guard_ok"].device != route["device"]
                or int(state["sticky_guard_ok"].data_ptr())
                != int(state.get("sticky_guard_ok_data_ptr", -1))
            )
        )
    ):
        raise ValueError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL dynamic accepted-input "
            f"contract drift for B={batch}"
        )
    return state, batch


@triton.jit
def _fr13_fixed32_committer_gate_precompute_kernel(
    A_logs,
    dt_biases,
    gate_coeffs,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a_scale = -tl.exp(
        tl.load(A_logs + offsets, mask=mask, other=0).to(tl.float32)
    )
    dt_bias = tl.load(
        dt_biases + offsets, mask=mask, other=0
    ).to(tl.float32)
    tl.store(gate_coeffs + offsets * 2, a_scale, mask=mask)
    tl.store(gate_coeffs + offsets * 2 + 1, dt_bias, mask=mask)


def _fr13_fixed32_committer_gate_precompute(
    *, A_logs: torch.Tensor, dt_biases: torch.Tensor
) -> torch.Tensor:
    """Materialize event-invariant FP32 gate coefficients once per process."""
    global _FR13_FIXED32_COMMITTER_GATE_PRECOMPUTE_LAUNCHES
    if (
        tuple(A_logs.shape) != tuple(dt_biases.shape)
        or A_logs.ndim != 2
        or not A_logs.is_contiguous()
        or not dt_biases.is_contiguous()
        or A_logs.device != dt_biases.device
    ):
        raise RuntimeError("FR13 fixed32 gate-precompute input contract drift")
    key = (
        str(A_logs.device),
        int(A_logs.data_ptr()),
        int(dt_biases.data_ptr()),
        tuple(int(value) for value in A_logs.shape),
        tuple(int(value) for value in A_logs.stride()),
        tuple(int(value) for value in dt_biases.stride()),
        str(A_logs.dtype),
        str(dt_biases.dtype),
    )
    existing = _FR13_FIXED32_COMMITTER_GATE_COEFFS.get(key)
    if existing is not None:
        return existing
    if _FR13_FIXED32_COMMITTER_GATE_COEFFS:
        raise RuntimeError(
            "FR13 fixed32 gate-precompute operands changed after preseed"
        )
    total = int(A_logs.numel())
    block = 256
    gate_coeffs = torch.empty(
        (*A_logs.shape, 2),
        dtype=torch.float32,
        device=A_logs.device,
    )
    _fr13_fixed32_committer_gate_precompute_kernel[
        (triton.cdiv(total, block),)
    ](
        A_logs,
        dt_biases,
        gate_coeffs,
        N=total,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    _FR13_FIXED32_COMMITTER_GATE_PRECOMPUTE_LAUNCHES += 1
    _FR13_FIXED32_COMMITTER_GATE_COEFFS[key] = gate_coeffs
    return gate_coeffs


@triton.jit
def _fr13_fixed32_committer_native_layer_batch_kernel(
    a_rings,
    b_rings,
    gate_coeffs,
    k_rings,
    v_rings,
    bank_anchor,
    bank_off16,
    accepted_paths,
    accepted_lens,
    spec_state_indices,
    k_norm_rings,
    gate_rings,
    B: tl.constexpr,
    PATH_CAP: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    GATE_L_STRIDE: tl.constexpr,
    RING_K_L_STRIDE: tl.constexpr,
    RING_K_B_STRIDE: tl.constexpr,
    RING_K_N_STRIDE: tl.constexpr,
    RING_KN_L_STRIDE: tl.constexpr,
    RING_KN_B_STRIDE: tl.constexpr,
    RING_KN_N_STRIDE: tl.constexpr,
    RING_GATE_L_STRIDE: tl.constexpr,
    RING_GATE_B_STRIDE: tl.constexpr,
    RING_GATE_N_STRIDE: tl.constexpr,
    RING_V_L_STRIDE: tl.constexpr,
    RING_V_B_STRIDE: tl.constexpr,
    RING_V_N_STRIDE: tl.constexpr,
    RING_AB_L_STRIDE: tl.constexpr,
    RING_AB_B_STRIDE: tl.constexpr,
    RING_AB_N_STRIDE: tl.constexpr,
    SPEC_L_STRIDE: tl.constexpr,
    SPEC_B_STRIDE: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    K_NORM_REUSE: tl.constexpr,
    GATE_REUSE: tl.constexpr,
    DECAY_REUSE: tl.constexpr,
):
    """Native fused-sigmoid state recurrence batched across all layers.

    Each program retains the native kernel's ordered loop through the root and
    accepted drafts. The fixed suffix is fully neutral, so it is omitted. The
    caller discards the operator output, so this commit-only realization also
    omits the independent q projection, output store, and staging buffers. It
    gathers each live root/path row directly from the fixed32 activation rings.
    Layers share read-only path metadata and write disjoint physical ``(alias,
    row)`` destinations.
    """
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_lnh = tl.program_id(2)
    layer_span = B * HV
    i_l = i_lnh // layer_span
    i_nh = i_lnh % layer_span
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    accepted = tl.load(accepted_lens + i_n).to(tl.int64)
    # Replay enqueues a device assertion for accepted in [0, 11] and every
    # active node in [0, 31] before this graph. Consume that validated domain
    # directly so the hot scan does not repeat bounds clamps in every program.
    T = accepted + 1

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_gate = gate_coeffs + i_l * GATE_L_STRIDE + i_hv * 2
    b_a_scale = tl.load(p_gate)
    b_dt_bias = tl.load(p_gate + 1)
    # Preserve the anchor+offset form used by the byte-gated all-layer replay.
    # A raw pointer-table load loses 16-byte AxisInfo and changes reductions.
    state_bank = bank_anchor + tl.load(bank_off16 + i_l) * 4
    state_idx = tl.load(
        spec_state_indices + i_l * SPEC_L_STRIDE + i_n * SPEC_B_STRIDE
    ).to(tl.int64)
    if state_idx <= 0:
        return
    p_h0 = (
        state_bank
        + state_idx * BANK_STRIDE
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T):
        path_offset = tl.maximum(i_t - 1, 0)
        path_node = tl.load(
            accepted_paths + i_n * PATH_CAP + path_offset,
            mask=i_t > 0,
            other=0,
        ).to(tl.int64)
        node = path_node
        p_k = (
            k_rings
            + i_l * RING_K_L_STRIDE
            + i_n * RING_K_B_STRIDE
            + node * RING_K_N_STRIDE
            + i_h * K
            + o_k
        )
        p_v = (
            v_rings
            + i_l * RING_V_L_STRIDE
            + i_n * RING_V_B_STRIDE
            + node * RING_V_N_STRIDE
            + i_hv * V
            + o_v
        )
        p_a = (
            a_rings
            + i_l * RING_AB_L_STRIDE
            + i_n * RING_AB_B_STRIDE
            + node * RING_AB_N_STRIDE
            + i_hv
        )
        p_b = (
            b_rings
            + i_l * RING_AB_L_STRIDE
            + i_n * RING_AB_B_STRIDE
            + node * RING_AB_N_STRIDE
            + i_hv
        )
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if GATE_REUSE:
            p_live_gate = (
                gate_rings
                + i_l * RING_GATE_L_STRIDE
                + i_n * RING_GATE_B_STRIDE
                + node * RING_GATE_N_STRIDE
                + i_hv * 2
            )
            if not DECAY_REUSE:
                b_g_or_decay = tl.load(p_live_gate)
                b_beta = tl.load(p_live_gate + 1)
        else:
            b_b = tl.load(p_b).to(tl.float32)
            x = tl.load(p_a).to(tl.float32) + b_dt_bias
            softplus_x = tl.where(
                BETA * x <= THRESHOLD,
                (1 / BETA) * tl.log(1 + tl.exp(BETA * x)),
                x,
            )
            b_g_or_decay = b_a_scale * softplus_x
            b_beta = tl.sigmoid(b_b.to(tl.float32))

        if K_NORM_REUSE:
            b_k *= tl.load(
                k_norm_rings
                + i_l * RING_KN_L_STRIDE
                + i_n * RING_KN_B_STRIDE
                + node * RING_KN_N_STRIDE
                + i_h
            )
        elif USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        if DECAY_REUSE:
            b_h *= tl.load(p_live_gate)
        else:
            b_h *= tl.exp(b_g_or_decay)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if DECAY_REUSE:
            b_beta = tl.load(p_live_gate + 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]

    # Fixed32 expands one running-row index across every active path slot.
    # Intermediate state stores are not observable; publish only the final row.
    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def _fr13_fixed32_committer_native_layer_batch(
    *,
    state,
    banks_list,
    spec_state_indices,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    gate_coeffs,
    use_qk_l2norm_in_kernel,
    k_norm_reuse,
    gate_reuse,
    decay_reuse,
    bv64_warp4,
) -> None:
    """Launch all 48 native-realization scans to disjoint destinations once."""
    layers = 48
    batch = int(state["batch"])
    num_kh = int(state["num_kh"])
    dim_k = int(state["dim_k"])
    num_vh = int(state["num_vh"])
    dim_v = int(state["dim_v"])
    if (
        len(banks_list) != layers
        or dim_k != 128
        or dim_v != 128
        or int(state["bank_off16"].numel()) != layers
        or (
            not gate_reuse
            and (
                not torch.is_tensor(gate_coeffs)
                or tuple(gate_coeffs.shape) != (layers, num_vh, 2)
                or gate_coeffs.dtype != torch.float32
                or not gate_coeffs.is_contiguous()
            )
        )
        or (
            k_norm_reuse
            and (
                not torch.is_tensor(k_norm_rings)
                or tuple(k_norm_rings.shape)
                != (
                    layers,
                    int(k_rings.shape[1]),
                    int(k_rings.shape[2]),
                    num_kh,
                )
                or k_norm_rings.dtype != torch.float32
                or k_norm_rings.device != k_rings.device
                or not k_norm_rings.is_contiguous()
            )
        )
        or (
            gate_reuse
            and (
                not k_norm_reuse
                or not torch.is_tensor(gate_rings)
                or tuple(gate_rings.shape)
                != (
                    layers,
                    int(k_rings.shape[1]),
                    int(k_rings.shape[2]),
                    num_vh,
                    2,
                )
                or gate_rings.dtype != torch.float32
                or gate_rings.device != k_rings.device
                or not gate_rings.is_contiguous()
            )
        )
        or (decay_reuse and not gate_reuse)
    ):
        raise RuntimeError(
            "FR13 fixed32 committer layer-batch requires the pinned "
            "48-layer K=V=128 geometry"
        )
    block_k = triton.next_power_of_2(dim_k)
    # The candidate splits only independent value rows. Each program retains
    # the same ordered K reduction and recurrence, then writes disjoint rows.
    block_v = 64 if bv64_warp4 else triton.next_power_of_2(dim_v)
    kernel_warps = 4 if bv64_warp4 else 8
    grid = (1, triton.cdiv(dim_v, block_v), layers * batch * num_vh)
    accepted_paths = (
        state["direct_accepted_paths"]
        if state.get("direct_metadata", False)
        else state["accepted_paths"]
    )
    accepted_lens = (
        state["direct_accepted_lens"]
        if state.get("direct_metadata", False)
        else state["accepted_lens"]
    )
    extra_launch_kwargs = (
        {"maxnreg": 167} if decay_reuse and batch != 1 else {}
    )
    _fr13_fixed32_committer_native_layer_batch_kernel[grid](
        a_rings,
        b_rings,
        gate_coeffs if not gate_reuse else k_rings,
        k_rings,
        v_rings,
        banks_list[0],
        state["bank_off16"],
        accepted_paths,
        accepted_lens,
        spec_state_indices,
        k_norm_rings if k_norm_reuse else k_rings,
        gate_rings if gate_reuse else k_rings,
        B=batch,
        PATH_CAP=16,
        H=num_kh,
        HV=num_vh,
        K=dim_k,
        V=dim_v,
        BK=block_k,
        BV=block_v,
        BANK_STRIDE=int(banks_list[0].stride(0)),
        GATE_L_STRIDE=int(gate_coeffs.stride(0)),
        RING_K_L_STRIDE=int(k_rings.stride(0)),
        RING_K_B_STRIDE=int(k_rings.stride(1)),
        RING_K_N_STRIDE=int(k_rings.stride(2)),
        RING_KN_L_STRIDE=(
            int(k_norm_rings.stride(0)) if k_norm_reuse else 0
        ),
        RING_KN_B_STRIDE=(
            int(k_norm_rings.stride(1)) if k_norm_reuse else 0
        ),
        RING_KN_N_STRIDE=(
            int(k_norm_rings.stride(2)) if k_norm_reuse else 0
        ),
        RING_GATE_L_STRIDE=(
            int(gate_rings.stride(0)) if gate_reuse else 0
        ),
        RING_GATE_B_STRIDE=(
            int(gate_rings.stride(1)) if gate_reuse else 0
        ),
        RING_GATE_N_STRIDE=(
            int(gate_rings.stride(2)) if gate_reuse else 0
        ),
        RING_V_L_STRIDE=int(v_rings.stride(0)),
        RING_V_B_STRIDE=int(v_rings.stride(1)),
        RING_V_N_STRIDE=int(v_rings.stride(2)),
        RING_AB_L_STRIDE=int(a_rings.stride(0)),
        RING_AB_B_STRIDE=int(a_rings.stride(1)),
        RING_AB_N_STRIDE=int(a_rings.stride(2)),
        SPEC_L_STRIDE=int(spec_state_indices.stride(0)),
        SPEC_B_STRIDE=int(spec_state_indices.stride(1)),
        BETA=1.0,
        THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=bool(use_qk_l2norm_in_kernel),
        K_NORM_REUSE=bool(k_norm_reuse),
        GATE_REUSE=bool(gate_reuse),
        DECAY_REUSE=bool(decay_reuse),
        num_warps=kernel_warps,
        num_stages=3,
        **extra_launch_kwargs,
    )


def _fr13_fixed32_committer_graph_body(
    *,
    state,
    banks_list,
    spec_state_indices,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    output_scale,
    use_qk_l2norm_in_kernel,
    k_norm_reuse,
    gate_reuse,
    decay_reuse,
    bv64_warp4,
    layer_batch=None,
) -> None:
    """Record the fixed committer graph body with an optional one-call scan."""
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update as _sg,
    )

    layers = 48
    batch = state["batch"]
    path_cap = 16
    total = batch * path_cap
    num_kh = state["num_kh"]
    dim_k = state["dim_k"]
    num_vh = state["num_vh"]
    dim_v = state["dim_v"]

    use_layer_batch = (
        state.get("layer_batch", False)
        if layer_batch is None
        else bool(layer_batch)
    )
    graph_accepted_paths = (
        state["direct_accepted_paths"]
        if state.get("direct_metadata", False)
        else state["accepted_paths"]
    )
    graph_accepted_lens = (
        state["direct_accepted_lens"]
        if state.get("direct_metadata", False)
        else state["accepted_lens"]
    )
    if not use_layer_batch:
        # Preserve the exact native-reference preprocessing graph.
        state["abuf"].fill_(-1e4)
        state["bbuf"].zero_()
        state["kbuf"].zero_()
        state["vbuf"].zero_()
        state["ssi"].fill_(state["scratch"])

        accepted_lens = graph_accepted_lens.to(torch.long)
        node_mat = state["node_mat"]
        node_mat[:, 0] = 0
        node_mat[:, 1:] = (
            graph_accepted_paths[:, :15].to(torch.long).clamp(min=0)
        )
        valid = state["path_offsets"].unsqueeze(0) <= accepted_lens.unsqueeze(1)
        safe_nodes = torch.where(
            valid, node_mat, torch.zeros_like(node_mat)
        ).clamp(0, 31)
        batch_index = state["batch_offsets"].view(batch, 1)

        # Keep the incumbent full fixed16 staging graph byte-for-byte intact.
        k_selected = k_rings[:, batch_index, safe_nodes]
        v_selected = v_rings[:, batch_index, safe_nodes]
        a_selected = a_rings[:, batch_index, safe_nodes]
        b_selected = b_rings[:, batch_index, safe_nodes]
        mask4 = valid.view(1, batch, path_cap, 1, 1)
        mask3 = valid.view(1, batch, path_cap, 1)
        k_destination = state["kbuf"].view(
            layers, batch, path_cap, num_kh, dim_k
        )
        v_destination = state["vbuf"].view(
            layers, batch, path_cap, num_vh, dim_v
        )
        a_destination = state["abuf"].view(
            layers, batch, path_cap, num_vh
        )
        b_destination = state["bbuf"].view(
            layers, batch, path_cap, num_vh
        )
        k_destination.copy_(
            torch.where(mask4, k_selected, torch.zeros_like(k_selected))
        )
        v_destination.copy_(
            torch.where(mask4, v_selected, torch.zeros_like(v_selected))
        )
        a_destination.copy_(torch.where(
            mask3, a_selected, torch.full_like(a_selected, -1e4)
        ))
        b_destination.copy_(
            torch.where(mask3, b_selected, torch.zeros_like(b_selected))
        )
        state["ssi"].copy_(
            spec_state_indices[:, :batch, 0:1]
            .to(torch.int32)
            .expand(layers, batch, path_cap)
        )

    if use_layer_batch:
        _fr13_fixed32_committer_native_layer_batch(
            state=state,
            banks_list=banks_list,
            spec_state_indices=spec_state_indices,
            k_rings=k_rings,
            k_norm_rings=k_norm_rings,
            gate_rings=gate_rings,
            v_rings=v_rings,
            a_rings=a_rings,
            b_rings=b_rings,
            gate_coeffs=state["gate_coeffs"],
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            k_norm_reuse=k_norm_reuse,
            gate_reuse=gate_reuse,
            decay_reuse=decay_reuse,
            bv64_warp4=bv64_warp4,
        )
    else:
        for layer in range(layers):
            _sg(
                A_log=A_logs[layer],
                a=state["abuf"][layer].reshape(1, total, num_vh),
                b=state["bbuf"][layer].reshape(1, total, num_vh),
                dt_bias=dt_biases[layer],
                q=state["qbuf"],
                k=state["kbuf"][layer].reshape(1, total, num_kh, dim_k),
                v=state["vbuf"][layer].reshape(1, total, num_vh, dim_v),
                scale=output_scale,
                initial_state=banks_list[layer],
                inplace_final_state=True,
                cu_seqlens=state["cu"],
                ssm_state_indices=state["ssi"][layer],
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )


def _fr13_fixed32_validate_running_rows(
    *, spec_state_indices: torch.Tensor, batch: int, bank_rows: int
) -> torch.Tensor:
    """Assert in-range, per-layer-distinct running rows without host readback."""
    running_rows = spec_state_indices[:, :batch, 0].to(torch.long)
    sorted_rows = torch.sort(running_rows, dim=1).values
    distinct = (
        (sorted_rows[:, 1:] != sorted_rows[:, :-1]).all()
        if batch > 1
        else torch.ones((), dtype=torch.bool, device=running_rows.device)
    )
    _fr13_fixed32_device_assert(
        (
            (running_rows >= 0).all()
            & (running_rows < int(bank_rows)).all()
            & distinct
        ),
        "FR13_FIXED32_COMMIT_DEVICE_FILL running-row contract violation",
    )
    return running_rows


def validate_fixed32_conv_commit_rows(
    *,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    bank_alias_ids: torch.Tensor,
    bank_alias_peer_layers: torch.Tensor,
    guard_flags: torch.Tensor,
    batch: int,
    bank_rows: int,
) -> None:
    """Enqueue one fixed-grid row/path guard before raw-pointer conv access."""
    if (
        type(batch) is not int
        or not 1 <= batch <= 4
        or type(bank_rows) is not int
        or bank_rows <= 0
        or not torch.is_tensor(spec_state_indices)
        or spec_state_indices.dtype != torch.int32
        or spec_state_indices.ndim != 3
        or int(spec_state_indices.shape[0]) != 48
        or int(spec_state_indices.shape[1]) < batch
        or int(spec_state_indices.shape[2]) != 32
        or not torch.is_tensor(accepted_paths)
        or accepted_paths.dtype != torch.int32
        or accepted_paths.ndim != 2
        or int(accepted_paths.shape[0]) < batch
        or int(accepted_paths.shape[1]) != 16
        or not torch.is_tensor(accepted_lens)
        or accepted_lens.dtype != torch.int32
        or accepted_lens.ndim != 1
        or int(accepted_lens.shape[0]) < batch
        or accepted_paths.device != spec_state_indices.device
        or accepted_lens.device != spec_state_indices.device
        or not torch.is_tensor(bank_alias_ids)
        or bank_alias_ids.dtype != torch.int64
        or bank_alias_ids.device != spec_state_indices.device
        or tuple(bank_alias_ids.shape) != (48,)
        or not bank_alias_ids.is_contiguous()
        or not torch.is_tensor(bank_alias_peer_layers)
        or bank_alias_peer_layers.dtype != torch.int32
        or bank_alias_peer_layers.device != spec_state_indices.device
        or tuple(bank_alias_peer_layers.shape) != (48, 3)
        or not bank_alias_peer_layers.is_contiguous()
        or not torch.is_tensor(guard_flags)
        or guard_flags.dtype != torch.bool
        or guard_flags.device != spec_state_indices.device
        or tuple(guard_flags.shape) != (48 * batch,)
        or not guard_flags.is_contiguous()
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT row-guard geometry/dtype/device drift"
        )

    _fr13_fixed32_conv_commit_row_guard_kernel[(48 * batch,)](
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
        bank_alias_peer_layers,
        guard_flags,
        int(spec_state_indices.stride(0)),
        int(spec_state_indices.stride(1)),
        int(spec_state_indices.stride(2)),
        int(accepted_paths.stride(0)),
        int(accepted_paths.stride(1)),
        int(accepted_lens.stride(0)),
        int(bank_alias_peer_layers.stride(0)),
        int(bank_alias_peer_layers.stride(1)),
        BANK_ROWS=bank_rows,
        B=batch,
        LAYERS=48,
        SPEC_COLS=32,
        PATH_COLS=16,
        MAX_ACCEPTED=_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH,
        ALIAS_WIDTH=3,
        PEER_CAP=16,
        num_warps=4,
        num_stages=1,
    )
    _fr13_fixed32_device_assert(
        guard_flags.all(),
        "FR13_FIXED32_CONV_COMMIT precommit row/path contract violation",
    )


def validate_fixed32_conv_commit_rows_sticky(
    *,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    bank_alias_ids: torch.Tensor,
    bank_alias_peer_layers: torch.Tensor,
    sticky_ok: torch.Tensor,
    batch: int,
    bank_rows: int,
) -> None:
    """Enqueue the fixed-grid guard without a per-event scalar reduction."""
    if (
        type(batch) is not int
        or not 1 <= batch <= 4
        or type(bank_rows) is not int
        or bank_rows <= 0
        or not torch.is_tensor(spec_state_indices)
        or spec_state_indices.dtype != torch.int32
        or spec_state_indices.ndim != 3
        or int(spec_state_indices.shape[0]) != 48
        or int(spec_state_indices.shape[1]) < batch
        or int(spec_state_indices.shape[2]) != 32
        or not torch.is_tensor(accepted_paths)
        or accepted_paths.dtype != torch.int32
        or accepted_paths.ndim != 2
        or int(accepted_paths.shape[0]) < batch
        or int(accepted_paths.shape[1]) != 16
        or not torch.is_tensor(accepted_lens)
        or accepted_lens.dtype != torch.int32
        or accepted_lens.ndim != 1
        or int(accepted_lens.shape[0]) < batch
        or accepted_paths.device != spec_state_indices.device
        or accepted_lens.device != spec_state_indices.device
        or not torch.is_tensor(bank_alias_ids)
        or bank_alias_ids.dtype != torch.int64
        or bank_alias_ids.device != spec_state_indices.device
        or tuple(bank_alias_ids.shape) != (48,)
        or not bank_alias_ids.is_contiguous()
        or not torch.is_tensor(bank_alias_peer_layers)
        or bank_alias_peer_layers.dtype != torch.int32
        or bank_alias_peer_layers.device != spec_state_indices.device
        or tuple(bank_alias_peer_layers.shape) != (48, 3)
        or not bank_alias_peer_layers.is_contiguous()
        or not torch.is_tensor(sticky_ok)
        or sticky_ok.dtype != torch.int32
        or sticky_ok.device != spec_state_indices.device
        or tuple(sticky_ok.shape) != ()
        or not sticky_ok.is_contiguous()
    ):
        raise RuntimeError(
            "FR13_FIXED32_CONV_COMMIT sticky row-guard "
            "geometry/dtype/device drift"
        )

    _fr13_fixed32_conv_commit_sticky_guard_kernel[(48 * batch,)](
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        bank_alias_ids,
        bank_alias_peer_layers,
        sticky_ok,
        int(spec_state_indices.stride(0)),
        int(spec_state_indices.stride(1)),
        int(spec_state_indices.stride(2)),
        int(accepted_paths.stride(0)),
        int(accepted_paths.stride(1)),
        int(accepted_lens.stride(0)),
        int(bank_alias_peer_layers.stride(0)),
        int(bank_alias_peer_layers.stride(1)),
        BANK_ROWS=bank_rows,
        B=batch,
        LAYERS=48,
        SPEC_COLS=32,
        PATH_COLS=16,
        MAX_ACCEPTED=_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH,
        ALIAS_WIDTH=3,
        PEER_CAP=16,
        num_warps=4,
        num_stages=1,
    )
    _fr13_fixed32_device_assert(
        sticky_ok,
        "FR13_FIXED32_CONV_COMMIT precommit row/path contract violation",
    )


def preseed_fixed32_committer_graph(
    *,
    banks_list,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    k_rings,
    k_norm_rings,
    gate_rings=None,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_layers,
    num_spec_decodes,
    output_scale,
    use_qk_l2norm_in_kernel=True,
    burn_node_bank=False,
    root_node=0,
    max_path=16,
) -> dict[str, object]:
    """Warm and capture one fixed16 graph; call before measured events."""
    num_kh, dim_k, num_vh, dim_v = _validate_fixed32_committer_contract(
        banks_list=banks_list,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        k_rings=k_rings,
        k_norm_rings=k_norm_rings,
        gate_rings=gate_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        num_layers=num_layers,
        num_spec_decodes=num_spec_decodes,
        burn_node_bank=burn_node_bank,
        root_node=root_node,
        max_path=max_path,
    )
    batch = int(num_spec_decodes)
    signature = _fr13_fixed32_committer_signature(
        banks_list=banks_list,
        spec_state_indices=spec_state_indices,
        k_rings=k_rings,
        k_norm_rings=k_norm_rings,
        gate_rings=gate_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        num_spec_decodes=batch,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    existing = _FR13_FIXED32_COMMITTER.get(signature)
    if existing is not None:
        return dict(existing["contract"])
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL preseed called inside capture"
        )

    device = k_rings.device
    dtype = k_rings.dtype
    total = batch * 16
    scratch = int(banks_list[0].shape[0]) - 1
    layer_batch = _fr13_fixed32_committer_layer_batch_requested()
    metadata_copy_fusion = (
        _fr13_fixed32_committer_metadata_fusion_requested()
    )
    direct_metadata = _fr13_fixed32_committer_direct_metadata_requested()
    sticky_guard = _fr13_fixed32_committer_sticky_guard_requested()
    k_norm_reuse = _fr13_fixed32_committer_knorm_ring_requested()
    gate_reuse = _fr13_fixed32_committer_gate_ring_requested()
    decay_reuse = _fr13_fixed32_committer_decay_ring_requested()
    bv64_warp4 = _fr13_fixed32_committer_bv64_warp4_requested()
    if bv64_warp4 and (
        not layer_batch
        or _FR13_FIXED32_MODE != "hydra27_fixed32"
        or batch not in (1, 4)
        or int(k_rings.shape[2]) != 32
        or os.environ.get("FR13_DRAFT_VOCAB_ROOT") != "1"
        or os.environ.get("FR13_DRAFT_VOCAB_K") != "65536"
    ):
        raise RuntimeError(
            "FR13 fixed32 committer BV64/4-warp requires exact Hydra27 "
            "physical32 K64/root1 layer batching at B1 or B4"
        )
    if metadata_copy_fusion and not layer_batch:
        raise RuntimeError(
            "FR13 fixed32 metadata fusion requires committer layer batching"
        )
    if direct_metadata and not layer_batch:
        raise RuntimeError(
            "FR13 fixed32 direct metadata requires committer layer batching"
        )
    if direct_metadata and metadata_copy_fusion:
        raise RuntimeError(
            "FR13 fixed32 direct metadata and metadata-copy fusion are "
            "mutually exclusive"
        )
    if sticky_guard and not direct_metadata:
        raise RuntimeError(
            "FR13 fixed32 sticky guard requires direct persistent metadata"
        )
    if k_norm_reuse and (
        not layer_batch
        or not direct_metadata
        or not (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
        )
        or scan_align_on()
        or not use_qk_l2norm_in_kernel
        or k_norm_rings is None
    ):
        raise RuntimeError(
            "FR13 fixed32 K-norm reuse requires layer batching, direct "
            "metadata, in-kernel L2 normalization, and its persistent ring"
        )
    if gate_reuse and (
        not k_norm_reuse
        or not layer_batch
        or not direct_metadata
        or not (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
        )
        or scan_align_on()
        or not use_qk_l2norm_in_kernel
        or gate_rings is None
    ):
        raise RuntimeError(
            "FR13 fixed32 gate reuse requires K-norm reuse, layer batching, "
            "direct metadata, raw rsqrt gating, and its persistent ring"
        )
    if decay_reuse and (
        not gate_reuse
        or not k_norm_reuse
        or not layer_batch
        or not direct_metadata
        or not (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
        )
        or scan_align_on()
        or not use_qk_l2norm_in_kernel
        or gate_rings is None
    ):
        raise RuntimeError(
            "FR13 fixed32 decay reuse requires gate/K-norm reuse, layer "
            "batching, direct metadata, raw rsqrt gating, and the persistent "
            "gate ring"
        )
    gate_coeffs = (
        _fr13_fixed32_committer_gate_precompute(
            A_logs=A_logs,
            dt_biases=dt_biases,
        )
        if layer_batch and not gate_reuse
        else None
    )
    bank_ptrs, _bank_shape, _bank_stride = build_replay_bank_pointer_table(
        list(banks_list)
    )
    anchor_ptr = bank_ptrs[0]
    sticky_guard_ok = (
        torch.ones((), dtype=torch.int32, device=device)
        if sticky_guard
        else None
    )
    state = {
        "batch": batch,
        "bank_rows": int(banks_list[0].shape[0]),
        "num_kh": num_kh,
        "dim_k": dim_k,
        "num_vh": num_vh,
        "dim_v": dim_v,
        "scratch": scratch,
        "accepted_paths": torch.zeros(
            batch, 16, dtype=accepted_paths.dtype, device=device
        ),
        "accepted_lens": torch.zeros(
            batch, dtype=accepted_lens.dtype, device=device
        ),
        "node_mat": torch.zeros(
            batch, 16, dtype=torch.long, device=device
        ),
        "path_offsets": torch.arange(16, device=device),
        "batch_offsets": torch.arange(batch, device=device),
        "kbuf": torch.zeros(
            48, total, num_kh, dim_k, device=device, dtype=dtype
        ),
        "vbuf": torch.zeros(
            48, total, num_vh, dim_v, device=device, dtype=dtype
        ),
        "abuf": torch.full(
            (48, total, num_vh), -1e4, device=device, dtype=dtype
        ),
        "bbuf": torch.zeros(
            48, total, num_vh, device=device, dtype=dtype
        ),
        "qbuf": torch.zeros(
            1, total, num_kh, dim_k, device=device, dtype=dtype
        ),
        "cu": (
            torch.arange(batch + 1, device=device, dtype=torch.int64) * 16
        ).to(torch.int32),
        "ssi": torch.full(
            (48, batch, 16),
            scratch,
            dtype=torch.int32,
            device=device,
        ),
        "bank_off16": (
            torch.tensor(
                [(pointer - anchor_ptr) // 16 for pointer in bank_ptrs],
                dtype=torch.int64,
                device=device,
            )
            if layer_batch
            else None
        ),
        "gate_coeffs": gate_coeffs,
        "layer_batch": layer_batch,
        "metadata_copy_fusion": metadata_copy_fusion,
        "direct_metadata": direct_metadata,
        "direct_accepted_paths": accepted_paths if direct_metadata else None,
        "direct_accepted_lens": accepted_lens if direct_metadata else None,
        "sticky_guard": sticky_guard,
        "k_norm_reuse": k_norm_reuse,
        "gate_reuse": gate_reuse,
        "decay_reuse": decay_reuse,
        "bv64_warp4": bv64_warp4,
        "sticky_guard_ok": sticky_guard_ok,
        "sticky_guard_ok_data_ptr": (
            int(sticky_guard_ok.data_ptr()) if sticky_guard_ok is not None else None
        ),
        "layer_batch_byte_gate_passed": not layer_batch,
        "layer_batch_byte_gate_attempts": 0,
        "layer_batch_byte_gate_coverage_mask": (
            0
            if layer_batch
            else _FR13_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
        ),
        "reference_graph": None,
        "graph": None,
        "preseed_capacity": 0,
        "contract": {
            "mode": _FR13_FIXED32_MODE,
            "batch": batch,
            "path_cap": 16,
            "neutralizations": 5,
            "ring_gathers": 4,
            "fused_calls": 48,
            "graph_replays_per_event": 1,
            "preseed_capacity": 0,
        },
    }
    if layer_batch:
        state["contract"].update(
            {
                "fused_calls": 1,
                "neutralizations": 0,
                "ring_gathers": 0,
                "native_reference_fused_calls": 48,
                "layer_batch": True,
                "state_only_output_elided": True,
                "active_length_recurrence": True,
                "pre_replay_dynamic_bound_guard": True,
                "hot_scan_bound_clamps": 0,
                "physical_node_domain": 32,
                "accepted_steps_max": 12,
                "final_state_store_once": True,
                "direct_ring_loads": True,
                "direct_ring_inputs": 4,
                "direct_scalar_ring_inputs": 1 if k_norm_reuse else 0,
                "direct_gate_ring_inputs": 1 if gate_reuse else 0,
                "candidate_staging_launches": 0,
                "gate_coefficients_hoisted": True,
                "committer_gate_coefficients_elided": gate_reuse,
                "event_independent_gate_precompute": True,
                "committer_gate_precompute_elided": gate_reuse,
                "gate_precompute_launches_per_process": int(
                    _FR13_FIXED32_COMMITTER_GATE_PRECOMPUTE_LAUNCHES
                ),
                "gate_exp_per_event": 0,
                "gate_reuse": gate_reuse,
                "decay_reuse": decay_reuse,
                "gate_source": (
                    "fixed32_sfwd_existing_decay_multiplier"
                    if decay_reuse
                    else "fixed32_sfwd_existing_raw_gate_math"
                    if gate_reuse
                    else "committer_raw_gate_math"
                ),
                "gate_nonlinear_evaluations_per_value_head_step": (
                    0 if gate_reuse else 3
                ),
                "gate_scalar_loads_per_value_head_step": (
                    2 if gate_reuse else 0
                ),
                "producer_extra_gate_nonlinear_evaluations": 0,
                "committer_decay_exponentials_per_value_head_step": (
                    0 if decay_reuse else 1
                ),
                "raw_ab_ring_stores_per_physical_value_head": 2,
                "raw_ab_ring_store_elision": False,
                "full_value_tile": not bv64_warp4,
                "value_tile": 64 if bv64_warp4 else 128,
                "kernel_warps": 4 if bv64_warp4 else 8,
                "programs_per_layer_request_value_head": (
                    2 if bv64_warp4 else 1
                ),
                "duplicate_value_tile_k_loads_per_step": (
                    1 if bv64_warp4 else 0
                ),
                "state_elements_per_thread_before_compiler_effects": 64,
                "bv64_warp4": bv64_warp4,
                "metadata_copy_fusion": metadata_copy_fusion,
                "direct_metadata": direct_metadata,
                "sticky_guard": sticky_guard,
                "k_norm_reuse": k_norm_reuse,
                "k_norm_source": (
                    "fixed32_sfwd_existing_rsqrt"
                    if k_norm_reuse
                    else "committer_reduction"
                ),
                "k_norm_reductions_per_value_head_step": (
                    0 if k_norm_reuse else 1
                ),
                "k_norm_scalar_loads_per_value_head_step": (
                    1 if k_norm_reuse else 0
                ),
                "producer_extra_k_norm_reductions": 0,
                "sticky_guard_route": (
                    "fixed32_ownerpath_warp32_sticky_scalar_v5"
                    if sticky_guard
                    else "stock_bool_vector_v4"
                ),
                "sticky_guard_kernel_launches_per_event": (
                    1 if sticky_guard else 0
                ),
                "sticky_guard_scalar_reduction_launches_per_event": 0,
                "sticky_guard_valid_event_global_stores": 0,
                "sticky_guard_failure_atomic": (
                    "atomic_xchg_zero" if sticky_guard else "not_applicable"
                ),
                "sticky_guard_failure_state": (
                    "process_lifetime_fail_closed"
                    if sticky_guard
                    else "not_applicable"
                ),
                "metadata_copy_launches_per_event": (
                    0 if metadata_copy_fusion or direct_metadata else 2
                ),
                "metadata_copy_elements_per_request": (
                    17 if metadata_copy_fusion else 0
                ),
                "metadata_roundtrip_elements_per_request": (
                    0 if direct_metadata else 17
                ),
                "metadata_source": (
                    "persistent_taw_publish_buffers"
                    if direct_metadata
                    else "committer_graph_staging_buffers"
                ),
                "metadata_validation_lease": (
                    "conv_direct_exact_pointer_batch_stream_one_shot"
                    if metadata_copy_fusion or direct_metadata
                    else "disabled"
                ),
                "metadata_guarded_fallback": True,
                "direct_metadata_missing_lease": (
                    "fail_closed" if direct_metadata else "not_applicable"
                ),
                "duplicate_committer_metadata_guard": (
                    not (metadata_copy_fusion or direct_metadata)
                ),
                "physical_alias_row_uniqueness_guard": (
                    "validate_fixed32_conv_commit_rows"
                ),
                "byte_gate": "real_swe_all_reachable_accepted_lengths_0_11",
                "byte_gate_raw_compare": "torch_equal_uint8",
                "unseen_length_route": "shadow_then_reference",
                "accepted_length_max": (
                    _FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH
                ),
                "accepted_length_full_mask": (
                    _FR13_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
                ),
            }
        )

    def graph_body(*, use_layer_batch=None) -> None:
        _fr13_fixed32_committer_graph_body(
            state=state,
            banks_list=banks_list,
            spec_state_indices=spec_state_indices,
            k_rings=k_rings,
            k_norm_rings=k_norm_rings,
            gate_rings=gate_rings,
            v_rings=v_rings,
            a_rings=a_rings,
            b_rings=b_rings,
            A_logs=A_logs,
            dt_biases=dt_biases,
            output_scale=output_scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            k_norm_reuse=k_norm_reuse,
            gate_reuse=gate_reuse,
            decay_reuse=decay_reuse,
            bv64_warp4=bv64_warp4,
            layer_batch=use_layer_batch,
        )

    running_rows = _fr13_fixed32_validate_running_rows(
        spec_state_indices=spec_state_indices,
        batch=batch,
        bank_rows=int(banks_list[0].shape[0]),
    )
    saved_rows = [
        bank[running_rows[layer]].clone()
        for layer, bank in enumerate(banks_list)
    ]

    def restore_running_rows() -> None:
        for layer, bank in enumerate(banks_list):
            bank[running_rows[layer]] = saved_rows[layer]

    def capture_graph(*, use_layer_batch: bool):
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                graph_body(use_layer_batch=use_layer_batch)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=capture_stream):
            graph_body(use_layer_batch=use_layer_batch)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        return captured

    reference_graph = capture_graph(use_layer_batch=False)
    if layer_batch:
        restore_running_rows()
        graph = capture_graph(use_layer_batch=True)
        state["reference_graph"] = reference_graph
    else:
        graph = reference_graph
    restore_running_rows()
    state["graph"] = graph
    _FR13_FIXED32_COMMITTER[signature] = state
    _FR13_FIXED32_COMMITTER_COUNTERS["captures"] += 1
    print(
        "[FR13_FIXED32_COMMIT_DEVICE_FILL] preseeded: "
        f"mode={_FR13_FIXED32_MODE} B={batch} path_cap=16 "
        f"neutralizations={state['contract']['neutralizations']} "
        f"gathers={state['contract']['ring_gathers']} "
        f"fused_calls={state['contract']['fused_calls']} "
        f"layer_batch={int(layer_batch)} "
        f"bv64_warp4={int(bv64_warp4)} replays=1",
        flush=True,
    )
    return dict(state["contract"])


def _fr13_fixed32_committer_layer_batch_byte_gate(
    *, state, banks_list, spec_state_indices,
) -> bool:
    """Qualify accepted depths using authenticated, unmeasured real events.

    Each previously unseen accepted length shadows the native and candidate
    graphs from identical state. Every touched fp32 state byte, including signed
    zero and NaN payloads, must match. The qualifying event is always served by
    the native graph; the candidate can serve only depths covered earlier.
    """
    if not state.get("layer_batch", False):
        return True
    full_mask = _FR13_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK
    coverage_mask = int(
        state.get("layer_batch_byte_gate_coverage_mask", -1)
    )
    if not 0 <= coverage_mask <= full_mask:
        raise RuntimeError(
            "FR13 fixed32 committer layer-batch coverage mask is invalid"
        )
    if coverage_mask == full_mask:
        state["layer_batch_byte_gate_passed"] = True
        return True
    if _fr13_fixed32_committer_layer_batch_real_event_marker() is None:
        return False
    event_mask = _fr13_fixed32_committer_accepted_length_mask(
        state["accepted_lens"].tolist(),
        batch=int(state["batch"]),
    )
    unseen_mask = event_mask & ~coverage_mask
    if unseen_mask == 0:
        return True
    reference_graph = state.get("reference_graph")
    candidate_graph = state.get("graph")
    if reference_graph is None or candidate_graph is None:
        raise RuntimeError(
            "FR13 fixed32 committer layer-batch missing paired byte-gate graphs"
        )

    batch = int(state["batch"])
    running_rows = _fr13_fixed32_validate_running_rows(
        spec_state_indices=spec_state_indices,
        batch=batch,
        bank_rows=int(state["bank_rows"]),
    )
    state["layer_batch_byte_gate_attempts"] += 1
    saved_rows = tuple(
        bank.index_select(0, running_rows[layer]).clone()
        for layer, bank in enumerate(banks_list)
    )

    def restore() -> None:
        for layer, bank in enumerate(banks_list):
            bank.index_copy_(0, running_rows[layer], saved_rows[layer])

    try:
        reference_graph.replay()
        torch.cuda.synchronize(state["accepted_lens"].device)
        reference_rows = tuple(
            bank.index_select(0, running_rows[layer]).clone()
            for layer, bank in enumerate(banks_list)
        )
        restore()
        candidate_graph.replay()
        torch.cuda.synchronize(state["accepted_lens"].device)
        mismatched_layers = tuple(
            layer
            for layer, (bank, reference) in enumerate(
                zip(banks_list, reference_rows, strict=True)
            )
            if not _fr13_fixed32_tensor_bits_equal(
                bank.index_select(0, running_rows[layer]), reference
            )
        )
        if mismatched_layers:
            raise RuntimeError(
                "FR13 fixed32 committer layer-batch byte gate failed: "
                f"B={batch} mismatched_layers={mismatched_layers}"
            )
        coverage_mask |= event_mask
        state["layer_batch_byte_gate_coverage_mask"] = coverage_mask
        state["layer_batch_byte_gate_passed"] = coverage_mask == full_mask
    finally:
        restore()
        torch.cuda.synchronize(state["accepted_lens"].device)

    print(
        "[FR13_FIXED32_COMMITTER_LAYER_BATCH BYTE-GATE COVERAGE] "
        f"B={batch} new_mask={unseen_mask:#06x} "
        f"coverage={coverage_mask:#06x} complete={int(coverage_mask == full_mask)} "
        "layers=48 state_bytes=exact reference_served=1",
        flush=True,
    )
    return False


def preseed_fixed32_committer_graphs_all_batches(
    *,
    banks_list,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    k_rings,
    k_norm_rings,
    gate_rings=None,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_layers,
    max_batch_size,
    output_scale,
    use_qk_l2norm_in_kernel=True,
    burn_node_bank=False,
    root_node=0,
    max_path=16,
) -> dict[str, object]:
    """Capture fixed committer graphs for every supported batch before serving.

    ``accepted_paths`` and ``accepted_lens`` have the exact server capacity.
    The API temporarily seeds distinct, in-range running rows so every
    occupancy up to that capacity can be captured before those slots have
    carried live requests, then restores the state-index tensor before
    returning. ``banks_list`` must be the persistent tuple reused by measured
    replay, and the captured tensor objects and their storage must not be
    rebound after this call.
    """
    capacity = int(max_batch_size)
    global _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY
    prior_capacity = _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY
    if prior_capacity is not None and prior_capacity != capacity:
        raise RuntimeError(
            "FR13 fixed32 committer server capacity changed after preseed: "
            f"{prior_capacity} -> {capacity}"
        )
    if capacity not in _FR13_FIXED32_BATCHES:
        raise ValueError(
            "FR13 fixed32 all-B preseed max_batch_size must be 1..4, "
            f"got {capacity}"
        )
    batches = tuple(range(1, capacity + 1))
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL all-B preseed called inside capture"
        )
    if (
        not torch.is_tensor(accepted_paths)
        or tuple(accepted_paths.shape) != (capacity, 16)
        or not accepted_paths.is_contiguous()
    ):
        raise ValueError(
            "FR13 fixed32 all-B preseed requires contiguous accepted_paths "
            f"shape={(capacity, 16)}, got "
            f"{getattr(accepted_paths, 'shape', None)}"
        )
    if (
        not torch.is_tensor(accepted_lens)
        or tuple(accepted_lens.shape) != (capacity,)
        or not accepted_lens.is_contiguous()
    ):
        raise ValueError(
            "FR13 fixed32 all-B preseed requires contiguous accepted_lens "
            f"shape={(capacity,)}, got "
            f"{getattr(accepted_lens, 'shape', None)}"
        )
    if (
        not torch.is_tensor(spec_state_indices)
        or spec_state_indices.ndim != 3
        or int(spec_state_indices.shape[0]) != 48
        or int(spec_state_indices.shape[1]) < capacity
        or int(spec_state_indices.shape[2]) < 1
    ):
        raise ValueError(
            "FR13 fixed32 all-B preseed requires spec_state_indices "
            f"shape=(48,>={capacity},>=1), got "
            f"{getattr(spec_state_indices, 'shape', None)}"
        )
    if not isinstance(banks_list, tuple) or len(banks_list) != 48:
        raise ValueError(
            "FR13 fixed32 all-B preseed requires a persistent 48-bank tuple"
        )
    bank_rows = int(banks_list[0].shape[0])
    if bank_rows < capacity + 1:
        raise ValueError(
            "FR13 fixed32 all-B preseed has too few bank rows for distinct "
            f"warmup slots: rows={bank_rows} capacity={capacity}"
        )
    if not torch.is_tensor(k_rings) or (
        k_rings.ndim != 5 or int(k_rings.shape[1]) < capacity
    ):
        raise ValueError(
            "FR13 fixed32 all-B preseed ring batch must cover server "
            f"capacity {capacity}, got "
            f"{getattr(k_rings, 'shape', None)}"
        )

    saved_roots = spec_state_indices[:, :capacity, 0].clone()
    safe_roots = torch.arange(
        bank_rows - capacity - 1,
        bank_rows - 1,
        dtype=spec_state_indices.dtype,
        device=spec_state_indices.device,
    )
    states = {}
    try:
        spec_state_indices[:, :capacity, 0].copy_(
            safe_roots.view(1, capacity).expand(48, capacity)
        )
        for batch in batches:
            preseed_fixed32_committer_graph(
                banks_list=banks_list,
                spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths[:batch],
                accepted_lens=accepted_lens[:batch],
                k_rings=k_rings,
                k_norm_rings=k_norm_rings,
                gate_rings=gate_rings,
                v_rings=v_rings,
                a_rings=a_rings,
                b_rings=b_rings,
                A_logs=A_logs,
                dt_biases=dt_biases,
                num_layers=num_layers,
                num_spec_decodes=batch,
                output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                burn_node_bank=burn_node_bank,
                root_node=root_node,
                max_path=max_path,
            )
            signature = _fr13_fixed32_committer_signature(
                banks_list=banks_list,
                spec_state_indices=spec_state_indices,
                k_rings=k_rings,
                k_norm_rings=k_norm_rings,
                gate_rings=gate_rings,
                v_rings=v_rings,
                a_rings=a_rings,
                b_rings=b_rings,
                A_logs=A_logs,
                dt_biases=dt_biases,
                num_spec_decodes=batch,
                output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            states[batch] = _FR13_FIXED32_COMMITTER[signature]
    finally:
        spec_state_indices[:, :capacity, 0].copy_(saved_roots)

    if tuple(sorted(states)) != batches:
        raise RuntimeError(
            "FR13 fixed32 all-B preseed did not cover every occupancy through "
            f"server capacity {capacity}"
        )
    for state in states.values():
        state["preseed_capacity"] = max(
            int(state.get("preseed_capacity", 0)),
            capacity,
        )
        state["contract"]["preseed_capacity"] = state["preseed_capacity"]
    identity_key = _fr13_fixed32_committer_identity_key(
        banks_list=banks_list,
        spec_state_indices=spec_state_indices,
        k_rings=k_rings,
        k_norm_rings=k_norm_rings,
        gate_rings=gate_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    prior_route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    if (
        prior_route is not None
        and prior_route["identity_key"] != identity_key
    ):
        raise RuntimeError(
            "FR13 fixed32 committer persistent operands changed after "
            "all-B preseed"
        )
    _FR13_FIXED32_COMMITTER_FAST_ROUTE["state"] = {
        "mode": _FR13_FIXED32_MODE,
        "capacity": capacity,
        "identity_key": identity_key,
        "banks": banks_list,
        "bank_anchor": banks_list[0],
        "bank_tail": banks_list[-1],
        "spec_state_indices": spec_state_indices,
        "accepted_paths_data_ptr": int(accepted_paths.data_ptr()),
        "accepted_lens_data_ptr": int(accepted_lens.data_ptr()),
        "k_rings": k_rings,
        "k_norm_rings": k_norm_rings,
        "gate_rings": gate_rings,
        "v_rings": v_rings,
        "a_rings": a_rings,
        "b_rings": b_rings,
        "A_logs": A_logs,
        "dt_biases": dt_biases,
        "device": k_rings.device,
        "output_scale": float(output_scale),
        "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
        "states_by_batch": dict(states),
    }
    _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY = capacity
    print(
        "[FR13_FIXED32_COMMIT_DEVICE_FILL] all-B ready: "
        f"batches={list(batches)} graphs_ready={len(batches)}",
        flush=True,
    )
    return {
        "mode": _FR13_FIXED32_MODE,
        "max_batch_size": capacity,
        "batches": batches,
        "graphs": {
            batch: dict(states[batch]["contract"])
            for batch in batches
        },
        "all_batches_ready": True,
    }


def warm_fixed32_committer_graphs_all_batches() -> dict[str, object]:
    """Replay every preseeded occupancy once without changing serving state."""
    route = _FR13_FIXED32_COMMITTER_FAST_ROUTE.get("state")
    if route is None:
        raise RuntimeError(
            "FR13 fixed32 committer boot warm requires all-B preseed"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "FR13 fixed32 committer boot warm is forbidden during capture"
        )
    prior = fixed32_committer_warmup_counters()
    if prior["ready"]:
        return prior

    capacity = int(route["capacity"])
    batches = tuple(range(1, capacity + 1))
    if (
        capacity not in _FR13_FIXED32_BATCHES
        or _FR13_FIXED32_COMMITTER_REQUIRED_CAPACITY != capacity
        or tuple(sorted(route["states_by_batch"])) != batches
    ):
        raise RuntimeError(
            "FR13 fixed32 committer boot-warm route is incomplete"
        )
    counters = _FR13_FIXED32_COMMITTER_COUNTERS
    if (
        int(counters["actual_replays_enqueued"]) != 0
        or any(
            int(value) != 0
            for value in counters["actual_replays_by_batch"].values()
        )
        or any(
            int(value) != 0
            for key in (
                "metadata_fusion_published_by_batch",
                "metadata_fusion_consumed_by_batch",
                "metadata_fusion_fallbacks_by_batch",
                "direct_metadata_published_by_batch",
                "direct_metadata_consumed_by_batch",
            )
            for value in counters[key].values()
        )
        or bool(_FR13_FIXED32_COMMITTER_METADATA_LEASE)
    ):
        raise RuntimeError(
            "FR13 fixed32 committer boot warm started after a measured replay"
        )

    banks = route["banks"]
    spec_state_indices = route["spec_state_indices"]
    conv_state = _FR13_FIXED32_CONV_PREGATHER.get("state")
    if (
        not isinstance(conv_state, dict)
        or int(conv_state.get("max_batch_size", 0)) != capacity
        or conv_state.get("ssm_banks") is not banks
        or conv_state.get("commit_spec_state_indices")
        is not spec_state_indices
        or not isinstance(conv_state.get("banks"), tuple)
        or len(conv_state["banks"]) != 48
        or not torch.is_tensor(conv_state.get("accepted_paths"))
        or not torch.is_tensor(conv_state.get("accepted_lens"))
        or int(conv_state.get("commit_direct_launches", -1)) != 0
        or int(conv_state.get("commit_gather_launches", -1)) != 0
        or int(conv_state.get("commit_scatter_launches", -1)) != 0
        or any(
            int(value) != 0
            for value in conv_state.get(
                "commit_direct_launches_by_batch", {}
            ).values()
        )
        or any(
            int(value) != 0
            for value in conv_state.get(
                "commit_gather_launches_by_batch", {}
            ).values()
        )
        or any(
            int(value) != 0
            for value in conv_state.get(
                "commit_scatter_launches_by_batch", {}
            ).values()
        )
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess boot warm has no clean conv lease"
        )
    conv_banks = conv_state["banks"]
    accepted_paths = conv_state["accepted_paths"]
    accepted_lens = conv_state["accepted_lens"]
    alias_ranks = tuple(int(value) for value in conv_state["bank_alias_ranks"])
    if (
        len(alias_ranks) != 48
        or min(alias_ranks) != 0
        or max(alias_ranks) != 2
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm alias-rank contract drift"
        )
    safe_rows = (
        torch.tensor(
            alias_ranks,
            dtype=spec_state_indices.dtype,
            device=spec_state_indices.device,
        ).view(48, 1)
        * capacity
        + torch.arange(
            capacity,
            dtype=spec_state_indices.dtype,
            device=spec_state_indices.device,
        ).view(1, capacity)
        + 1
    )
    safe_rows_long = safe_rows.to(torch.long)
    maximum_safe_row = 3 * capacity
    if any(
        int(bank.shape[0]) <= maximum_safe_row
        for bank in (*banks, *conv_banks)
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm has no isolated alias rows"
        )
    saved_spec_state_indices = spec_state_indices[
        :, :capacity, :
    ].clone()
    saved_accepted_paths = accepted_paths.clone()
    saved_accepted_lens = accepted_lens.clone()
    saved_staging = conv_state["staging"].clone()
    saved_bank_rows = tuple(
        bank.index_select(0, safe_rows_long[layer]).clone()
        for layer, bank in enumerate(banks)
    )
    saved_conv_rows = tuple(
        bank.index_select(0, safe_rows_long[layer]).clone()
        for layer, bank in enumerate(conv_banks)
    )
    states = route["states_by_batch"]
    saved_state_inputs = {
        batch: (
            states[batch]["accepted_paths"].clone(),
            states[batch]["accepted_lens"].clone(),
            states[batch]["node_mat"].clone(),
            states[batch]["qbuf"].clone(),
        )
        for batch in batches
    }
    saved_conv_gathers = int(conv_state["commit_gather_launches"])
    saved_conv_scatters = int(conv_state["commit_scatter_launches"])
    saved_conv_direct = int(conv_state["commit_direct_launches"])
    saved_conv_gathers_by_batch = dict(
        conv_state["commit_gather_launches_by_batch"]
    )
    saved_conv_scatters_by_batch = dict(
        conv_state["commit_scatter_launches_by_batch"]
    )
    saved_conv_direct_by_batch = dict(
        conv_state["commit_direct_launches_by_batch"]
    )
    saved_actual = int(counters["actual_replays_enqueued"])
    saved_by_batch = dict(counters["actual_replays_by_batch"])
    metadata_counter_keys = (
        "metadata_fusion_published_by_batch",
        "metadata_fusion_consumed_by_batch",
        "metadata_fusion_fallbacks_by_batch",
        "direct_metadata_published_by_batch",
        "direct_metadata_consumed_by_batch",
    )
    saved_metadata_counters = {
        key: dict(counters[key]) for key in metadata_counter_keys
    }
    saved_callbacks = tuple(_FR13_FIXED32_COMMITTER_CALLBACKS)
    global _FR13_FIXED32_COMMITTER_ANNOUNCED
    saved_announced = _FR13_FIXED32_COMMITTER_ANNOUNCED
    _FR13_FIXED32_COMMITTER_CALLBACKS.clear()
    _FR13_FIXED32_COMMITTER_ANNOUNCED = True
    replays = 0
    conv_gathers = 0
    conv_scatters = 0
    conv_direct = 0
    try:
        spec_state_indices[:, :capacity, :].copy_(
            safe_rows.view(48, capacity, 1).expand(
                48,
                capacity,
                int(spec_state_indices.shape[2]),
            )
        )
        accepted_paths.zero_()
        accepted_lens.zero_()
        for batch in batches:
            launch_fixed32_conv_commit_to_col0(
                conv_banks=conv_banks,
                spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths,
                accepted_lens=accepted_lens,
                num_spec_decodes=batch,
            )
            conv_direct += 1
            _fr13_fixed32_committer_replay(
                banks_list=banks,
                spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths[:batch],
                accepted_lens=accepted_lens[:batch],
                k_rings=route["k_rings"],
                k_norm_rings=route["k_norm_rings"],
                gate_rings=route["gate_rings"],
                v_rings=route["v_rings"],
                a_rings=route["a_rings"],
                b_rings=route["b_rings"],
                A_logs=route["A_logs"],
                dt_biases=route["dt_biases"],
                num_layers=48,
                num_spec_decodes=batch,
                output_scale=route["output_scale"],
                use_qk_l2norm_in_kernel=route[
                    "use_qk_l2norm_in_kernel"
                ],
                runrow_init=True,
                burn_node_bank=False,
            )
            replays += 1
        torch.cuda.synchronize(route["device"])
    finally:
        try:
            spec_state_indices[:, :capacity, :].copy_(
                saved_spec_state_indices
            )
            accepted_paths.copy_(saved_accepted_paths)
            accepted_lens.copy_(saved_accepted_lens)
            conv_state["staging"].copy_(saved_staging)
            for layer, (bank, saved) in enumerate(
                zip(banks, saved_bank_rows, strict=True)
            ):
                bank.index_copy_(0, safe_rows_long[layer], saved)
            for layer, (bank, saved) in enumerate(
                zip(conv_banks, saved_conv_rows, strict=True)
            ):
                bank.index_copy_(0, safe_rows_long[layer], saved)
            for batch in batches:
                (
                    saved_state_paths,
                    saved_state_lens,
                    saved_node_mat,
                    saved_qbuf,
                ) = saved_state_inputs[batch]
                states[batch]["accepted_paths"].copy_(saved_state_paths)
                states[batch]["accepted_lens"].copy_(saved_state_lens)
                states[batch]["node_mat"].copy_(saved_node_mat)
                states[batch]["qbuf"].copy_(saved_qbuf)
        finally:
            conv_state["commit_gather_launches"] = saved_conv_gathers
            conv_state["commit_scatter_launches"] = saved_conv_scatters
            conv_state["commit_direct_launches"] = saved_conv_direct
            conv_state["commit_gather_launches_by_batch"].clear()
            conv_state["commit_gather_launches_by_batch"].update(
                saved_conv_gathers_by_batch
            )
            conv_state["commit_scatter_launches_by_batch"].clear()
            conv_state["commit_scatter_launches_by_batch"].update(
                saved_conv_scatters_by_batch
            )
            conv_state["commit_direct_launches_by_batch"].clear()
            conv_state["commit_direct_launches_by_batch"].update(
                saved_conv_direct_by_batch
            )
            counters["actual_replays_enqueued"] = saved_actual
            counters["actual_replays_by_batch"].clear()
            counters["actual_replays_by_batch"].update(saved_by_batch)
            for key in metadata_counter_keys:
                counters[key].clear()
                counters[key].update(saved_metadata_counters[key])
            _FR13_FIXED32_COMMITTER_METADATA_LEASE.clear()
            _FR13_FIXED32_COMMITTER_CALLBACKS[:] = saved_callbacks
            _FR13_FIXED32_COMMITTER_ANNOUNCED = saved_announced
            torch.cuda.synchronize(route["device"])

    bank_state_restored = all(
        _fr13_fixed32_tensor_bits_equal(
            bank.index_select(0, safe_rows_long[layer]),
            saved,
        )
        for layer, (bank, saved) in enumerate(
            zip(banks, saved_bank_rows, strict=True)
        )
    )
    conv_bank_state_restored = all(
        _fr13_fixed32_tensor_bits_equal(
            bank.index_select(0, safe_rows_long[layer]),
            saved,
        )
        for layer, (bank, saved) in enumerate(
            zip(conv_banks, saved_conv_rows, strict=True)
        )
    )
    conv_staging_state_restored = _fr13_fixed32_tensor_bits_equal(
        conv_state["staging"],
        saved_staging,
    )
    persistent_inputs_restored = (
        _fr13_fixed32_tensor_bits_equal(
            spec_state_indices[:, :capacity, :],
            saved_spec_state_indices,
        )
        and _fr13_fixed32_tensor_bits_equal(
            accepted_paths, saved_accepted_paths
        )
        and _fr13_fixed32_tensor_bits_equal(
            accepted_lens, saved_accepted_lens
        )
    )
    graph_inputs_restored = all(
        _fr13_fixed32_tensor_bits_equal(
            states[batch]["accepted_paths"], saved_state_inputs[batch][0]
        )
        and _fr13_fixed32_tensor_bits_equal(
            states[batch]["accepted_lens"], saved_state_inputs[batch][1]
        )
        and _fr13_fixed32_tensor_bits_equal(
            states[batch]["node_mat"], saved_state_inputs[batch][2]
        )
        and _fr13_fixed32_tensor_bits_equal(
            states[batch]["qbuf"], saved_state_inputs[batch][3]
        )
        for batch in batches
    )
    measured_state_restored = (
        int(counters["actual_replays_enqueued"]) == saved_actual
        and dict(counters["actual_replays_by_batch"]) == saved_by_batch
        and all(
            dict(counters[key]) == saved_metadata_counters[key]
            for key in metadata_counter_keys
        )
        and not _FR13_FIXED32_COMMITTER_METADATA_LEASE
        and int(conv_state["commit_gather_launches"])
        == saved_conv_gathers
        and int(conv_state["commit_scatter_launches"])
        == saved_conv_scatters
        and int(conv_state["commit_direct_launches"])
        == saved_conv_direct
        and dict(conv_state["commit_gather_launches_by_batch"])
        == saved_conv_gathers_by_batch
        and dict(conv_state["commit_scatter_launches_by_batch"])
        == saved_conv_scatters_by_batch
        and dict(conv_state["commit_direct_launches_by_batch"])
        == saved_conv_direct_by_batch
        and tuple(_FR13_FIXED32_COMMITTER_CALLBACKS) == saved_callbacks
        and _FR13_FIXED32_COMMITTER_ANNOUNCED is saved_announced
    )
    if (
        replays != capacity
        or conv_direct != capacity
        or conv_gathers != 0
        or conv_scatters != 0
        or not bank_state_restored
        or not conv_bank_state_restored
        or not conv_staging_state_restored
        or not persistent_inputs_restored
        or not graph_inputs_restored
        or not measured_state_restored
    ):
        raise RuntimeError(
            "FR13 fixed32 postprocess warm missed a specialization or "
            "failed state restoration"
        )
    evidence = {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": _FR13_FIXED32_MODE,
        "max_batch_size": capacity,
        "batches": batches,
        "replays": replays,
        "conv_commit_direct_launches": conv_direct,
        "conv_commit_gather_launches": conv_gathers,
        "conv_commit_scatter_launches": conv_scatters,
        "route_lease_current": True,
        "bank_state_restored": bank_state_restored,
        "conv_bank_state_restored": conv_bank_state_restored,
        "conv_staging_state_restored": conv_staging_state_restored,
        "alias_destination_contract": "exact_alias_only_16x3",
        "input_state_restored": (
            persistent_inputs_restored and graph_inputs_restored
        ),
        "measured_state_restored": measured_state_restored,
        # Every other mutable graph scratch is fully overwritten at the top
        # of graph_body before any live value is consumed.
        "scratch_overwrite_proven": True,
        "scratch_restored": (
            "accepted_paths",
            "accepted_lens",
            "node_mat",
            "qbuf",
        ),
        "scratch_fully_overwritten": (
            "abuf",
            "bbuf",
            "kbuf",
            "vbuf",
            "ssi",
        ),
        "scratch_immutable": (
            "cu",
            "path_offsets",
            "batch_offsets",
            "graph",
            "scratch",
        ),
    }
    _FR13_FIXED32_COMMITTER_WARMUP.clear()
    _FR13_FIXED32_COMMITTER_WARMUP.update(
        {
            "route": route,
            "conv_state": conv_state,
            "evidence": evidence,
        }
    )
    return fixed32_committer_warmup_counters()


def _fr13_fixed32_committer_replay(
    *,
    banks_list,
    spec_state_indices,
    accepted_paths,
    accepted_lens,
    k_rings,
    k_norm_rings,
    gate_rings,
    v_rings,
    a_rings,
    b_rings,
    A_logs,
    dt_biases,
    num_layers,
    num_spec_decodes,
    output_scale,
    use_qk_l2norm_in_kernel,
    runrow_init,
    burn_node_bank,
) -> None:
    """Copy live inputs and enqueue exactly one preseeded fixed16 replay."""
    if not runrow_init:
        raise RuntimeError(
            "FR13_FIXED32_COMMIT_DEVICE_FILL requires runrow_init=True"
        )
    state, batch = _fr13_fixed32_committer_fast_state(
        banks_list=banks_list,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        k_rings=k_rings,
        k_norm_rings=k_norm_rings,
        gate_rings=gate_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        num_layers=num_layers,
        num_spec_decodes=num_spec_decodes,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        burn_node_bank=burn_node_bank,
    )

    metadata_fused = False
    if state.get("direct_metadata", False):
        validation_bank_rows = min(
            int(state["bank_rows"]),
            int(_FR13_FIXED32_CONV_PREGATHER["state"]["anchor"].shape[0]),
        )
        lease_key = _fr13_fixed32_committer_metadata_lease_key(
            batch=batch,
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            committer_paths=state["direct_accepted_paths"],
            committer_lens=state["direct_accepted_lens"],
            validation_bank_rows=validation_bank_rows,
            validation_guard=(
                state["sticky_guard_ok"]
                if state.get("sticky_guard", False)
                else None
            ),
        )
        if not _fr13_fixed32_committer_consume_direct_metadata_lease(lease_key):
            raise RuntimeError(
                "FR13 fixed32 direct metadata requires the preceding guarded "
                "conv lease"
            )
        metadata_fused = True
        _FR13_FIXED32_COMMITTER_COUNTERS[
            "direct_metadata_consumed_by_batch"
        ][batch] += 1
    elif state.get("metadata_copy_fusion", False):
        validation_bank_rows = min(
            int(state["bank_rows"]),
            int(_FR13_FIXED32_CONV_PREGATHER["state"]["anchor"].shape[0]),
        )
        lease_key = _fr13_fixed32_committer_metadata_lease_key(
            batch=batch,
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            committer_paths=state["accepted_paths"],
            committer_lens=state["accepted_lens"],
            validation_bank_rows=validation_bank_rows,
        )
        metadata_fused = _fr13_fixed32_committer_consume_metadata_lease(
            lease_key
        )
        counter_key = (
            "metadata_fusion_consumed_by_batch"
            if metadata_fused
            else "metadata_fusion_fallbacks_by_batch"
        )
        _FR13_FIXED32_COMMITTER_COUNTERS[counter_key][batch] += 1
    if not metadata_fused:
        state["accepted_paths"].copy_(accepted_paths)
        state["accepted_lens"].copy_(accepted_lens)
        _fr13_fixed32_validate_running_rows(
            spec_state_indices=spec_state_indices,
            batch=batch,
            bank_rows=int(state["bank_rows"]),
        )
        lens = state["accepted_lens"].to(torch.long).view(batch, 1)
        positions = torch.arange(16, device=accepted_paths.device).view(1, 16)
        active = positions < lens
        paths = state["accepted_paths"].to(torch.long)
        dynamic_ok = (
            (lens >= 0).all()
            & (lens <= _FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH).all()
            & ((~active) | ((paths >= 0) & (paths < 32))).all()
        )
        _fr13_fixed32_device_assert(
            dynamic_ok,
            "FR13_FIXED32_COMMIT_DEVICE_FILL dynamic contract violation",
        )
    graph = state["graph"]
    if state.get("layer_batch", False):
        gate_passed = _fr13_fixed32_committer_layer_batch_byte_gate(
            state=state,
            banks_list=banks_list,
            spec_state_indices=spec_state_indices,
        )
        if not gate_passed:
            # Boot/probe traffic cannot power qualification. A real event that
            # introduces a depth also stays on the incumbent after its shadow.
            graph = state["reference_graph"]
    graph.replay()

    counters = _FR13_FIXED32_COMMITTER_COUNTERS
    counters["actual_replays_enqueued"] += 1
    counters["actual_replays_by_batch"][batch] += 1
    event = {
        "mode": _FR13_FIXED32_MODE,
        "batch": batch,
        "replay_index": counters["actual_replays_enqueued"],
        "status": "enqueued",
    }
    for callback in tuple(_FR13_FIXED32_COMMITTER_CALLBACKS):
        callback(dict(event))
    global _FR13_FIXED32_COMMITTER_ANNOUNCED
    if not _FR13_FIXED32_COMMITTER_ANNOUNCED:
        _FR13_FIXED32_COMMITTER_ANNOUNCED = True
        print(
            "[FR13_FIXED32_COMMIT_DEVICE_FILL ENGAGED] "
            f"mode={_FR13_FIXED32_MODE} B={batch} fixed16 one-replay",
            flush=True,
        )


def launch_tree_gdn_replay_all_layers(
    *,
    bank_anchor: torch.Tensor,
    bank_off16: torch.Tensor,
    bank_shape: tuple[int, int, int, int],
    bank_stride: int,
    spec_state_indices: torch.Tensor,
    prev_lens: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    k_rings: torch.Tensor,
    v_rings: torch.Tensor,
    a_rings: torch.Tensor,
    b_rings: torch.Tensor,
    A_logs: torch.Tensor,
    dt_biases: torch.Tensor,
    num_layers: int,
    num_spec_decodes: int,
    output_scale: float,
    use_qk_l2norm_in_kernel: bool = True,
    runrow_commit: bool = False,
    runrow_init: bool = False,
    burn_node_bank: bool = False,
    banks_list=None,
    k_norm_rings: torch.Tensor | None = None,
    gate_rings: torch.Tensor | None = None,
) -> None:
    """Launch the FR13_EAGER_PACK batched all-layer accepted-path replay.

    Semantics-preserving sibling of launch_tree_gdn_replay: one launch
    replaces the legacy per-layer loop (48 launches + 48 flag clears). Every
    input must be a persistent preallocated stacked buffer (graph-stable
    addresses) allocated at GDN metadata-builder init; per-step scratch is
    the gate-4 failure mode and is banned here. bank_anchor is the LAYER-0
    bank tensor (pointer arg = alignment anchor; byte-A/B fix, see the
    kernel) and bank_off16 is the int64 device table of
    (bank[i].data_ptr() - bank[0].data_ptr()) // 16 derived from
    build_replay_bank_pointer_table's pointer list. Fixed32 mode binds a
    persistent bank tuple and all captured operands during all-B preseed;
    measured replay uses only its constant-size identity key. Legacy mode
    retains the pointer and shape validation below.
    """
    state_already_committed = globals().pop("_FR13_S1_STATE_DONE", False)
    if _FR13_FIXED32_MODE is not None:
        if state_already_committed:
            raise RuntimeError(
                "FR13_FIXED32_COMMIT_DEVICE_FILL cannot bypass the fixed16 "
                "graph via _FR13_S1_STATE_DONE"
            )
        _fr13_fixed32_committer_replay(
            banks_list=banks_list,
            spec_state_indices=spec_state_indices,
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            k_rings=k_rings,
            k_norm_rings=k_norm_rings,
            gate_rings=gate_rings,
            v_rings=v_rings,
            a_rings=a_rings,
            b_rings=b_rings,
            A_logs=A_logs,
            dt_biases=dt_biases,
            num_layers=num_layers,
            num_spec_decodes=num_spec_decodes,
            output_scale=output_scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            runrow_init=runrow_init,
            burn_node_bank=burn_node_bank,
        )
        return
    if state_already_committed:
        # =3: scan state already committed inside the S1 graph this step —
        # the staged tail must not double-commit (fused update is not
        # idempotent). Consume-once, set by the dm wrapper post-replay.
        return
    if num_spec_decodes <= 0:
        return
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if bank_off16.dtype != torch.int64 or bank_off16.numel() < num_layers:
        raise ValueError(
            f"bank_off16 must be int64 covering {num_layers} layers, got "
            f"{bank_off16.dtype} numel={bank_off16.numel()}"
        )
    if bank_anchor.dtype != torch.float32 or bank_anchor.data_ptr() % 16 != 0:
        raise ValueError(
            "bank_anchor must be the fp32 layer-0 GDN state bank with a "
            f"16-byte-aligned data_ptr, got {bank_anchor.dtype} "
            f"ptr={int(bank_anchor.data_ptr()):#x}"
        )
    bank_rows, num_vh, dim_v, dim_k = (int(x) for x in bank_shape)
    if k_rings.ndim != 5 or v_rings.ndim != 5 or a_rings.ndim != 4 or b_rings.ndim != 4:
        raise ValueError(
            "stacked ring shapes must be k(L,B,N,KH,DK)/v(L,B,N,VH,DV)/a,b(L,B,N,VH), got "
            f"k={tuple(k_rings.shape)} v={tuple(v_rings.shape)} "
            f"a={tuple(a_rings.shape)} b={tuple(b_rings.shape)}"
        )
    ring_layers, ring_bs, n_pad, num_kh, ring_dim_k = k_rings.shape
    if ring_layers < num_layers:
        raise ValueError(f"ring layers {ring_layers} < num_layers {num_layers}")
    # n_pad<=32: the replay kernels carry NO h_cache=[N_PAD,BV,DIM_K] tile (one
    # [BLOCK_V,DIM_K] tile per program), so their register budget is n_pad-independent
    # and safe at n_pad=32 even at the deployed BV=16 (accept>5 32-node horizon).
    if n_pad > 32 or n_pad & (n_pad - 1):
        raise ValueError(f"ring n_pad must be a power of two <=32, got {n_pad}")
    if ring_dim_k != dim_k:
        raise ValueError(f"ring k dim {ring_dim_k} != bank dim_k {dim_k}")
    if v_rings.shape != (ring_layers, ring_bs, n_pad, num_vh, dim_v):
        raise ValueError(
            f"v rings shape {tuple(v_rings.shape)} != "
            f"{(ring_layers, ring_bs, n_pad, num_vh, dim_v)}"
        )
    if a_rings.shape != (ring_layers, ring_bs, n_pad, num_vh) or b_rings.shape != (
        ring_layers,
        ring_bs,
        n_pad,
        num_vh,
    ):
        raise ValueError(
            f"a/b rings must be {(ring_layers, ring_bs, n_pad, num_vh)}, got "
            f"{tuple(a_rings.shape)}/{tuple(b_rings.shape)}"
        )
    if not (
        k_rings.is_contiguous()
        and v_rings.is_contiguous()
        and a_rings.is_contiguous()
        and b_rings.is_contiguous()
    ):
        raise ValueError("stacked activation rings must be contiguous")
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of k heads, got {num_vh}/{num_kh}")
    if ring_bs < num_spec_decodes:
        raise ValueError(f"ring batch {ring_bs} < num_spec_decodes {num_spec_decodes}")
    if spec_state_indices.ndim != 3 or spec_state_indices.shape[0] < num_layers or (
        spec_state_indices.shape[1] < num_spec_decodes
    ):
        raise ValueError(
            "stacked spec_state_indices must be (L, B, SPEC_COLS) covering "
            f"{num_layers}x{num_spec_decodes}, got {tuple(spec_state_indices.shape)}"
        )
    spec_cols = int(spec_state_indices.shape[2])
    if accepted_paths.ndim != 2 or accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            f"accepted_paths must be 2D covering {num_spec_decodes} rows, "
            f"got {tuple(accepted_paths.shape)}"
        )
    path_cols = int(accepted_paths.shape[1])
    if path_cols > spec_cols:
        # Defensive width clamp. It was introduced for FR13_SPEC_BLOCKS_CAP,
        # which narrowed the ssi (22 -> 13) while the accepted-paths BUFFER
        # stayed tree-wide; that cap was DELETED 2026-07-25 (dce60d18c), so
        # the two widths now agree and this branch is not expected to fire.
        # It is kept because it is value-identical either way: content is
        # bounded by accepted_len <= MAX_PATH (12) < spec_cols, and every
        # path-col read below is masked by accepted_len, so cols >= spec_cols
        # were only ever masked lanes.
        path_cols = spec_cols
    if prev_lens.ndim != 2 or prev_lens.shape[0] < num_layers or (
        prev_lens.shape[1] < num_spec_decodes
    ):
        raise ValueError(
            "stacked prev_lens must be (L, B) covering "
            f"{num_layers}x{num_spec_decodes}, got {tuple(prev_lens.shape)}"
        )
    if accepted_lens.numel() < num_spec_decodes:
        raise ValueError(
            f"accepted_lens must cover num_spec_decodes={num_spec_decodes}, "
            f"got {accepted_lens.numel()}"
        )
    if A_logs.ndim != 2 or A_logs.shape[0] < num_layers or A_logs.shape[1] < num_vh:
        raise ValueError(
            f"stacked A_logs must be (L, VH) covering {num_layers}x{num_vh}, "
            f"got {tuple(A_logs.shape)}"
        )
    if dt_biases.shape != A_logs.shape:
        raise ValueError(
            f"stacked dt_biases shape {tuple(dt_biases.shape)} != A_logs "
            f"{tuple(A_logs.shape)}"
        )
    if not (A_logs.is_contiguous() and dt_biases.is_contiguous()):
        raise ValueError("stacked A_logs/dt_biases must be contiguous")
    if not (spec_state_indices.is_contiguous() and prev_lens.is_contiguous()):
        raise ValueError("stacked spec_state_indices/prev_lens must be contiguous")
    if _fr13_committer_native_on() and runrow_init and banks_list is not None:
        # SHIP path (EAGER_PACK on): route each layer's committed-path state rebuild through NATIVE
        # fused_sigmoid_gating (validated 1.19e-7) instead of the batched custom replay kernel, to test
        # whether the gross state-carry corruption (garble root) is here. banks_list[L] is the per-layer
        # fp32 bank; k_rings[L]/... are per-layer ring slices; A_logs[L]/dt_biases[L] per-layer params.
        # spec_state_indices is STACKED (L, B, SPEC_COLS); _fr13_prepare_committer_layout + the native
        # replay both want a 2D [B, SPEC_COLS] per-layer view. Physical row maps are layer-invariant so
        # the nodes/cu layout could be shared, but pass each layer's own 2D slice (spec_state_indices[_L])
        # so the col-0 init read + node-bank burn address the correct per-layer rows. (was: 3D passed as
        # 2D -> col0[b] became a [SPEC_COLS] vector -> "expanded size 6 vs 10" crash.)
        _spec_cols = int(spec_state_indices.shape[2])
        # env is dropped by EngineCore worker curation -> sidecar files are the
        # deployment-armable path (same pattern as _fr13_committer_native_on).
        if (torch.cuda.is_current_stream_capturing()
                or globals().get("_FR13_S1_FORCE_DEVICE", False)):
            # S1 (=2) outer capture (or its eager side-stream warmup, forced
            # via _FR13_S1_FORCE_DEVICE): EVERY other committer flavor is
            # host-layout (.tolist/tensor-from-list) or a nested graph replay
            # — both capture-illegal. The device-layout variant computes the
            # same fixed-shape neutral-padded layout with device ops.
            _fr13_native_committer_all_layers_device(
                banks_list=banks_list, spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths, accepted_lens=accepted_lens,
                k_rings=k_rings, v_rings=v_rings, a_rings=a_rings,
                b_rings=b_rings, A_logs=A_logs, dt_biases=dt_biases,
                num_layers=num_layers, num_spec_decodes=num_spec_decodes,
                output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                burn_node_bank=burn_node_bank,
            )
            return
        if (os.environ.get("FR13_COMMITTER_GRAPH") == "1"
                or os.path.exists("/logs/fr13_committer_graph.arm")
                or os.path.exists("/tmp/fr13_committer_graph.arm")):
            # DIRECTION-2: CUDA-graph the 48-layer fused_sigmoid loop (byte-identical, gates 1+2 pass).
            # per-B graph (MAX_B=B, no dummy pad); overflow (accept+1 > max_path) falls back to batched.
            _fr13_native_committer_all_layers_graph(
                banks_list=banks_list, spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths, accepted_lens=accepted_lens,
                k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
                A_logs=A_logs, dt_biases=dt_biases, num_layers=num_layers,
                num_spec_decodes=num_spec_decodes, output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                burn_node_bank=burn_node_bank, max_path=16,
            )
            return
        if (os.environ.get("FR13_COMMITTER_NATIVE_BATCHED") == "1"
                or os.path.exists("/logs/fr13_committer_batched.arm")
                or os.path.exists("/tmp/fr13_committer_batched.arm")):
            _fr13_native_committer_all_layers_batched(
                banks_list=banks_list, spec_state_indices=spec_state_indices,
                accepted_paths=accepted_paths, accepted_lens=accepted_lens,
                k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
                A_logs=A_logs, dt_biases=dt_biases, num_layers=num_layers,
                num_spec_decodes=num_spec_decodes, output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                burn_node_bank=burn_node_bank,
            )
            return
        for _L in range(int(num_layers)):
            _fr13_native_committer_replay(
                state_bank=banks_list[_L], spec_state_indices=spec_state_indices[_L],
                accepted_paths=accepted_paths, accepted_lens=accepted_lens,
                k_ring=k_rings[_L], v_ring=v_rings[_L], a_ring=a_rings[_L], b_ring=b_rings[_L],
                A_log=A_logs[_L], dt_bias=dt_biases[_L], num_spec_decodes=num_spec_decodes,
                output_scale=output_scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                burn_node_bank=burn_node_bank, spec_cols=_spec_cols,
            )
        return
    grid = (int(num_layers) * int(num_spec_decodes), num_vh, triton.cdiv(dim_v, BV))
    _tree_gdn_replay_all_layers_kernel[grid](
        k_rings,
        v_rings,
        a_rings,
        b_rings,
        A_logs,
        dt_biases,
        bank_anchor,
        bank_off16,
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        prev_lens,
        NUM_SPEC=int(num_spec_decodes),
        NUM_KH=num_kh,
        NUM_VH=num_vh,
        DIM_K=dim_k,
        DIM_V=dim_v,
        BLOCK_V=BV,
        N_PAD=n_pad,
        PATH_COLS=path_cols,
        SPEC_COLS=spec_cols,
        BANK_STRIDE=int(bank_stride),
        RING_L_STRIDE_K=k_rings.stride(0),
        RING_B_STRIDE_K=k_rings.stride(1),
        RING_N_STRIDE_K=k_rings.stride(2),
        RING_L_STRIDE_V=v_rings.stride(0),
        RING_B_STRIDE_V=v_rings.stride(1),
        RING_N_STRIDE_V=v_rings.stride(2),
        RING_L_STRIDE_AB=a_rings.stride(0),
        RING_B_STRIDE_AB=a_rings.stride(1),
        RING_N_STRIDE_AB=a_rings.stride(2),
        SPEC_L_STRIDE=spec_state_indices.stride(0),
        PREV_L_STRIDE=prev_lens.stride(0),
        GATE_L_STRIDE=A_logs.stride(0),
        OUTPUT_SCALE=output_scale,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        RAW_GATING=True,
        SCAN_ALIGN=scan_align_on(),
        RUNROW_COMMIT=runrow_commit,
        RUNROW_INIT=runrow_init,
        BURN_NODE_BANK=burn_node_bank,
        num_warps=8,
    )


def launch_tree_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    tree: Tree,
    *,
    strict_mask: torch.Tensor | None = None,
    visible_mask: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    output_scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    invocation_counter: torch.Tensor | None = None,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the FR10 dense tree verifier.

    For CUDA graph capture, pass preallocated masks, output, and state buffers.
    The allocation path is only for probes and offline validation.
    """
    n = tree.n
    n_pad = padded_nodes(n)
    if strict_mask is None or visible_mask is None:
        strict_mask, visible_mask = tree.masks(q.device, n_pad)
    return launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        h0=h0,
        n_actual=n,
        n_pad=n_pad,
        strict_mask=strict_mask,
        visible_mask=visible_mask,
        out=out,
        state=state,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        invocation_counter=invocation_counter,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def launch_tree_gdn_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    *,
    n_actual: int,
    n_pad: int,
    strict_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    out: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    output_scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    h0_indices: torch.Tensor | None = None,
    h0_num_accepted_tokens: torch.Tensor | None = None,
    h0_is_bank: bool = False,
    h0_index_row: int = 0,
    h0_batch_index: int = 0,
    h0_use_accepted_column: bool = False,
    invocation_counter: torch.Tensor | None = None,
    raw_a: torch.Tensor | None = None,
    raw_b: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    piggyback_export: bool = False,
    chain_end_idx: int = 0,
    ring_k: torch.Tensor | None = None,
    ring_v: torch.Tensor | None = None,
    ring_a: torch.Tensor | None = None,
    ring_b: torch.Tensor | None = None,
    ring_k_norm: torch.Tensor | None = None,
    ring_gate: torch.Tensor | None = None,
    staging_flags: torch.Tensor | None = None,
    staging_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch with precomputed graph-safe tree descriptors.

    The tree replay route: no per-node HBM state export, no scratch state
    allocation; the durable accepted states are produced by
    launch_tree_gdn_replay at the committer instead. Returns (out, None).
    """
    if n_actual <= 0 or n_actual > n_pad:
        raise ValueError(f"invalid tree node counts n_actual={n_actual}, n_pad={n_pad}")
    # The monolithic scan needs BV<=8 at n_pad=32 to bound h_cache residency.
    # Source-gated fixed32 BV production uses the path scan, which has no h_cache;
    # let that exact route reach the stronger production checks below.
    _ov_cap = _read_tree_gdn_geom_override()
    _bv_cap = int(_ov_cap.get("BV", BV)) if _ov_cap else BV
    _fixed32_path_bv_production = (
        _FR13_FIXED32_MODE in _FR13_FIXED32_MODES
        and _FR13_FIXED32_GDN_PATH_BV_PRODUCTION == _bv_cap
    )
    _npad_cap = 32 if _bv_cap <= 8 or _fixed32_path_bv_production else 16
    if n_pad > _npad_cap or n_pad & (n_pad - 1):
        raise ValueError(
            f"n_pad must be a power of two <={_npad_cap} (effective BV={_bv_cap}; "
            f"n_pad>16 needs shrink-BV<=8), got {n_pad}")
    if q.shape[0] < n_actual:
        raise ValueError(f"q has {q.shape[0]} rows but n_actual={n_actual}")
    if k.shape[0] < n_actual:
        raise ValueError(f"k has {k.shape[0]} rows but n_actual={n_actual}")
    if v.shape[0] < n_actual:
        raise ValueError(f"v has {v.shape[0]} rows but n_actual={n_actual}")
    if g.shape[0] < n_actual or beta.shape[0] < n_actual:
        raise ValueError(
            f"g/beta rows must cover n_actual={n_actual}, got {g.shape[0]}/{beta.shape[0]}"
        )
    if strict_mask.shape[0] < n_pad or strict_mask.shape[1] < n_pad:
        raise ValueError(f"strict_mask must cover {n_pad}x{n_pad}, got {tuple(strict_mask.shape)}")
    if visible_mask.shape[0] < n_pad or visible_mask.shape[1] < n_pad:
        raise ValueError(f"visible_mask must cover {n_pad}x{n_pad}, got {tuple(visible_mask.shape)}")
    num_kh = q.shape[1]
    num_vh = v.shape[1]
    dim_k = q.shape[2]
    dim_v = v.shape[2]
    if k.shape[1] != num_kh or k.shape[2] != dim_k:
        raise ValueError(f"q/k shape mismatch: q={tuple(q.shape)} k={tuple(k.shape)}")
    if g.shape[1] != num_vh or beta.shape[1] != num_vh:
        raise ValueError(f"g/beta must use value-head count {num_vh}")
    if h0_is_bank:
        if h0.ndim != 4 or h0.shape[1:] != (num_vh, dim_v, dim_k):
            raise ValueError(
                f"h0 bank shape must be (*, {num_vh}, {dim_v}, {dim_k}), got {tuple(h0.shape)}"
            )
        if h0_indices is None:
            raise ValueError("h0_indices is required when h0_is_bank=True")
        if h0_use_accepted_column and h0_num_accepted_tokens is None:
            raise ValueError(
                "h0_num_accepted_tokens is required when h0_use_accepted_column=True"
            )
        if h0_index_row < 0 or h0_index_row >= h0_indices.numel():
            raise ValueError(
                f"h0_index_row {h0_index_row} outside h0_indices numel {h0_indices.numel()}"
            )
        if h0_use_accepted_column:
            if h0_batch_index < 0 or h0_batch_index >= h0_num_accepted_tokens.numel():
                raise ValueError(
                    "h0_batch_index "
                    f"{h0_batch_index} outside num_accepted_tokens numel "
                    f"{h0_num_accepted_tokens.numel()}"
                )
        if h0_indices.is_cuda:
            # Avoid GPU->CPU sync during capture. This range check is for eager
            # launches and debug repros; graph-captured serving relies on the
            # row-count guard above and prevalidated metadata.
            pass
        else:
            idx = int(h0_indices.reshape(-1)[h0_index_row].item())
            if idx < 0 or idx >= h0.shape[0]:
                raise ValueError(f"h0 bank index {idx} outside bank rows {h0.shape[0]}")
        h0_bank_stride = h0.stride(0)
    elif h0.shape != (num_vh, dim_v, dim_k):
        raise ValueError(f"h0 shape must be {(num_vh, dim_v, dim_k)}, got {tuple(h0.shape)}")
    else:
        h0_bank_stride = 0
    if h0_indices is None:
        h0_indices = strict_mask
    if h0_num_accepted_tokens is None:
        h0_num_accepted_tokens = strict_mask
    count_invocation = invocation_counter is not None
    if invocation_counter is None:
        invocation_counter = strict_mask
    raw_gating = (
        raw_a is not None
        or raw_b is not None
        or A_log is not None
        or dt_bias is not None
    )
    if raw_gating:
        if raw_a is None or raw_b is None or A_log is None or dt_bias is None:
            raise ValueError("raw_a, raw_b, A_log, and dt_bias must be provided together")
        if raw_a.shape[0] < n_actual or raw_a.shape[1] != num_vh:
            raise ValueError(
                f"raw_a must cover ({n_actual}, {num_vh}), got {tuple(raw_a.shape)}"
            )
        if raw_b.shape[0] < n_actual or raw_b.shape[1] != num_vh:
            raise ValueError(
                f"raw_b must cover ({n_actual}, {num_vh}), got {tuple(raw_b.shape)}"
            )
        if A_log.numel() < num_vh or dt_bias.numel() < num_vh:
            raise ValueError(
                f"A_log/dt_bias must cover {num_vh} value heads, got {A_log.numel()}/{dt_bias.numel()}"
            )
    else:
        raw_a = g
        raw_b = beta
        A_log = g
        dt_bias = beta
    if num_vh % num_kh != 0:
        raise ValueError(f"value heads must be a multiple of q/k heads, got {num_vh}/{num_kh}")
    if out is None:
        out = torch.empty((n_pad, num_vh, dim_v), device=q.device, dtype=q.dtype)
    elif out.shape[0] < n_actual or out.shape[1:] != (num_vh, dim_v):
        raise ValueError(
            f"out must be at least ({n_actual}, {num_vh}, {dim_v}), got {tuple(out.shape)}"
        )
    # STATELESS-TREE (replay-only): no per-node state export -> do NOT allocate
    # per-node scratch. A caller passing a state buffer is a wiring bug.
    if state is not None:
        raise ValueError("state buffer must not be provided; the tree replay route does not stage per-node states")
    state = strict_mask  # dummy pointer; no store reaches it
    # FR13_RING_EXPORT: in-kernel replay-ring staging (replaces the caller's 4
    # per-layer aten .copy_() launches). All-or-nothing, and only defined for the
    # RAW_GATING served path (the ring stages raw_a/raw_b, which the kernel only
    # loads under RAW_GATING).
    _ring_export = ring_k is not None
    if _ring_export:
        if ring_v is None or ring_a is None or ring_b is None:
            raise ValueError("ring export requires all four ring tensors")
        if raw_a is None or raw_b is None:
            raise ValueError("ring export requires the RAW_GATING path (raw_a/raw_b)")
        if ring_k.shape[0] < n_actual or ring_v.shape[0] < n_actual:
            raise ValueError(
                f"ring buffers must cover n_actual={n_actual} rows, got "
                f"{ring_k.shape[0]}/{ring_v.shape[0]}"
            )
        if (
            ring_k.dtype != k.dtype
            or ring_v.dtype != v.dtype
            or ring_a.dtype != raw_a.dtype
            or ring_b.dtype != raw_b.dtype
        ):
            raise ValueError("ring dtype mismatch: the ring stages byte-copies of the scan inputs")
    else:
        # dummy pointers; RING_EXPORT=False makes every ring store dead code
        ring_k = ring_v = ring_a = ring_b = strict_mask
    _k_norm_export = ring_k_norm is not None
    if _k_norm_export:
        if (
            not _ring_export
            or not _fr13_fixed32_committer_knorm_ring_requested()
            or not (
                _FR13_FIXED32_GDN_SINGLE_LAUNCH
                or _FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION is not None
            )
            or not use_qk_l2norm_in_kernel
            or ring_k_norm.dtype != torch.float32
            or ring_k_norm.device != k.device
            or ring_k_norm.ndim != 2
            or tuple(ring_k_norm.shape) != (n_pad, num_kh)
            or not ring_k_norm.is_contiguous()
        ):
            raise RuntimeError(
                "FR13 fixed32 K-norm export requires the armed B1 "
                "single-launch physical32 ring contract"
            )
    else:
        ring_k_norm = strict_mask
    _gate_export = ring_gate is not None
    if _gate_export:
        if (
            not _k_norm_export
            or not _fr13_fixed32_committer_gate_ring_requested()
            or not raw_gating
            or scan_align_on()
            or ring_gate.dtype != torch.float32
            or ring_gate.device != k.device
            or ring_gate.ndim != 3
            or tuple(ring_gate.shape) != (n_pad, num_vh, 2)
            or not ring_gate.is_contiguous()
        ):
            raise RuntimeError(
                "FR13 fixed32 gate export requires the armed B1 "
                "single-launch physical32 K-norm ring contract"
            )
    else:
        ring_gate = strict_mask
    _decay_export = _fr13_fixed32_committer_decay_ring_requested()
    if _decay_export and not _gate_export:
        raise RuntimeError(
            "FR13 fixed32 decay export requires the armed gate-ring route"
        )
    # Resolve launch geometry. The deployed/served path uses BLOCK_V=BV and
    # num_warps=8 with the Triton default num_stages (unset). The TEST-ONLY
    # override (FR13_TREE_GDN_GEOM_OVERRIDE) lets the BV/warps A/B arms run; it
    # is value-neutral (only tiling/warps/stages change). When unset, _geom is
    # None and the launch below is byte-identical to the prior locked path.
    _geom = _read_tree_gdn_geom_override()
    _bv = BV
    _num_warps = _DEPLOYED_NUM_WARPS
    _extra_launch_kwargs: dict = {}
    if _geom is not None:
        _bv = int(_geom.get("BV", _bv))
        _num_warps = int(_geom.get("num_warps", _num_warps))
        if "num_stages" in _geom:
            _extra_launch_kwargs["num_stages"] = int(_geom["num_stages"])
    if _gate_export:
        _extra_launch_kwargs["maxnreg"] = 80
    # FR13_SCAN_ALIGN body seams (d: l2norm div-by-sqrt, e: beta bf16
    # round-trip, K1: per-node carried-state bf16 round-trip = the depth-growth
    # store-boundary seam). MODE=body turns on all three (full (2)-oracle
    # alignment of the per-node update). Default OFF => SCAN_ALIGN=False => every
    # aligned branch in _gdn_node_step is dead code and the launch is
    # byte-identical to the locked path. The diagnostic geom-override above is
    # INDEPENDENT of this (it predates the unified flag and stays value-neutral).
    _scan_align = scan_align_on()
    if _k_norm_export and _scan_align:
        raise RuntimeError(
            "FR13 fixed32 K-norm export requires the incumbent rsqrt "
            "normalization path"
        )
    # FR13_NPAD_INVARIANT: pin the scan loop bound + offs_n lane count + h_cache
    # row span to the fixed N_FIXED for ALL tree sizes so the reduction FMA order
    # is canonical regardless of how many leaves co-reside (bug-class #10
    # codegen-identity; closes the MEASURED 0.0289 leaf-co-residency state gap).
    # N_LOOP=0 (default) => the kernel's N_SPAN equals N_PAD and the launch is
    # byte-identical to the locked path. num_warps stays the deployed value (8);
    # this is COMPUTE-ONLY (no copy, no HBM staging, geometry HELD), NOT the
    # refuted recompute route (which changed geometry to native BV32/w1/s3).
    _n_loop = 0
    if npad_invariant_on():
        if n_pad > _FR13_N_FIXED:
            raise ValueError(
                f"FR13_NPAD_INVARIANT N_FIXED={_FR13_N_FIXED} < n_pad={n_pad}; "
                "the canonical span must cover the tree's padded node block"
            )
        _n_loop = _FR13_N_FIXED
    # FR13_PARENT_GATHER: default OFF => the launch below is byte-identical to the
    # locked path (PARENT_GATHER=False threads through as a dead constexpr branch).
    _parent_gather = parent_gather_on()
    # FR13_HC_INTERNAL: compact h_cache to internal-node rows (leaves are never
    # re-read). One-time host derivation from the static tree descriptor,
    # cached; the first call for a shape must be OUTSIDE graph capture (vLLM
    # eager warmup) -- _hc_internal_desc fails loud otherwise. Default OFF =>
    # HC_MASK=0 => trace-time dead => byte-identical locked launch.
    _hc_mask = 0
    _hc_rows = 0
    _hc_slots_lo = 0
    _hc_slots_hi = 0
    _hc_slot_map = strict_mask  # dummy pointer when off/PG-off (dead code)
    if hc_internal_on():
        if piggyback_export:
            raise ValueError(
                "FR13_HC_INTERNAL is incompatible with PIGGYBACK_EXPORT "
                "(the chain-end node may be a leaf with no cached row)"
            )
        _hc_mask, _hc_rows, _hc_slots_lo, _hc_slots_hi = _hc_internal_desc(
            strict_mask, n_actual, n_pad
        )
        if _parent_gather and _hc_mask != 0:
            # HCxPG compat: the parent-gather branch's runtime parent index
            # reads its compacted slot from a device map (preseeded at
            # builder init; the getter fail-louds under capture).
            _hc_slot_map = hc_slot_map_get(n_actual, n_pad, strict_mask.device)
    grid = (num_vh, triton.cdiv(dim_v, _bv))

    _flags_export = staging_flags is not None
    _flags_arg = staging_flags if _flags_export else strict_mask  # dummy ptr when off (dead code)
    _flags_rows = int(staging_rows) if _flags_export else 0
    _subtree_state = _FR13_SUBTREE_CACHE.get(
        _subtree_cache_key(
            n_actual, num_vh, dim_v, dim_k, q.device
        )
    )
    if _FR13_SUBTREE_ROUTE_REQUESTED and _subtree_state is None:
        raise RuntimeError(
            f"FR13_SUBTREE_PARALLEL: no preseed for n_actual={n_actual}; "
            "the armed route cannot fall back to the monolithic scan"
        )
    _subtree_route_armed = bool(
        _subtree_state is not None and _subtree_state["route_armed"]
    )
    _subtree_selfcheck_armed = bool(
        _subtree_state is not None
        and _subtree_state["selfcheck_armed"]
    )

    def _launch_paths(
        _out,
        _count=count_invocation,
        *,
        _path_block_v=_bv,
        _counter_arg=invocation_counter,
        _single_launch_override=None,
        _gqa_group3_override=None,
    ):
        # FR13_SUBTREE_PARALLEL route: one launch per path level; paths in a
        # level scan concurrently on grid axis 2. RING/RAW semantics match
        # the monolith per node; PIGGYBACK unsupported (asserted below).
        st = _subtree_state
        if st is None:
            raise RuntimeError(
                f"FR13_SUBTREE_PARALLEL: no preseed for n_actual={n_actual}"
            )
        _fixed32_io = (
            _FR13_FIXED32_MODE is not None
            and st.get("schedule") == "fixed32"
            and st.get("fixed32_contract") is not None
        )
        if _FR13_FIXED32_MODE is not None and not _fixed32_io:
            raise RuntimeError(
                "FR13_FIXED32: exact path I/O specialization requires the "
                "validated fixed32 schedule"
            )
        _single_launch = st.get("fixed32_single_launch")
        # Same folded kernel, two authorities. The diagnostic bool is unchanged
        # and still unconditional at B1; the production arm is credential-bound
        # and admits B1 only when the credential it holds was issued FOR B1.
        _single_launch_enabled = (
            (
                _FR13_FIXED32_GDN_SINGLE_LAUNCH
                or _fr13_fixed32_gdn_single_launch_production_for_batch(1)
            )
            if _single_launch_override is None
            else bool(_single_launch_override)
        )
        _gqa_group3_enabled = (
            _fr13_fixed32_gdn_gqa_group3_production_for_batch(1)
            if _gqa_group3_override is None
            else bool(_gqa_group3_override)
        )
        _ordered_launch_enabled = (
            _single_launch_enabled or _gqa_group3_enabled
        )
        _ordered_block_v = (
            16
            if _gqa_group3_enabled
            and _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE
            else 8
        )
        # _single_launch_enabled now folds in the credentialed arm, so this one
        # raise covers BOTH single-launch authorities against GQA-group3
        # production. The module-level exclusion already refuses that pairing at
        # import; this is the per-request backstop for an override-injected or
        # otherwise drifted call.
        if _single_launch_enabled and _gqa_group3_enabled:
            raise RuntimeError(
                "FR13 fixed32 ordered GDN launch selectors overlapped"
            )
        if _ordered_launch_enabled and (
            not _fixed32_io
            or not isinstance(_single_launch, dict)
            or not isinstance(_single_launch.get("contract"), dict)
            or _single_launch["contract"].get("candidate")
            != _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID
            or _single_launch["contract"].get("physical_grid_z") != (1,)
            or _single_launch["contract"].get("critical_node_steps") != 32
            or _single_launch["contract"].get("outer_root_loop")
            != "ordered_tl_range"
            or _path_block_v != _ordered_block_v
            or _geom != {"BV": 8}
            or _subtree_selfcheck_armed
            or n_actual != 32
            or n_pad != 32
        ):
            raise RuntimeError(
                "FR13_FIXED32_GDN_SINGLE_LAUNCH exact K64/root1 BV8 B1 "
                "or admitted GQA3 BV16 B1 contract drift; no fallback is "
                "permitted"
            )
        if _ordered_launch_enabled:
            assert isinstance(_single_launch, dict)
            (
                _root_nodes,
                _root_parents,
                _root_max_len,
                _root_paths,
                _root_lengths,
            ) = st["levels"][0]
            (
                _branch_nodes,
                _branch_parents,
                _branch_max_len,
                _branch_paths,
                _branch_lengths,
            ) = st["levels"][1]
            _single_contract = _single_launch["contract"]
            (
                _branch_lengths_arg,
                _path_indices_arg,
                _prescaled_path_base,
            ) = _fr13_fixed32_gdn_single_launch_path_args(
                _single_launch,
                _branch_lengths,
                _branch_max_len,
            )
            if _gqa_group3_enabled:
                if not callable(_FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH):
                    raise RuntimeError(
                        "FR13 fixed32 GDN GQA-group3 launch was not preloaded"
                    )
                _FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    raw_a=raw_a,
                    raw_b=raw_b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    h0=h0,
                    h0_indices=h0_indices,
                    h0_num_accepted_tokens=h0_num_accepted_tokens,
                    invocation_counter=_counter_arg,
                    root_nodes=_root_nodes,
                    branch_nodes=_branch_nodes,
                    branch_lengths=_branch_lengths_arg,
                    group_path_indices=_path_indices_arg,
                    group_path_counts=_single_launch["path_counts"],
                    out=_out,
                    ring_k=ring_k,
                    ring_v=ring_v,
                    ring_a=ring_a,
                    ring_b=ring_b,
                    flags=_flags_arg,
                    ring_k_norm=ring_k_norm,
                    ring_gate=ring_gate,
                    batch_size=1,
                    mode=_FR13_FIXED32_MODE,
                    output_scale=output_scale,
                    h0_is_bank=h0_is_bank,
                    h0_index_row=h0_index_row,
                    h0_index_batch_stride=int(h0_indices.stride(0)),
                    h0_batch_index=h0_batch_index,
                    h0_accepted_batch_stride=int(
                        h0_num_accepted_tokens.stride(0)
                    ),
                    h0_bank_stride=h0_bank_stride,
                    h0_use_accepted_column=h0_use_accepted_column,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    raw_gating=raw_gating,
                    count_invocation=_count,
                    scan_align=_scan_align,
                    root_steps=int(_single_contract["groups"]),
                    max_path_len=_branch_max_len,
                    max_group_paths=int(
                        _single_contract["max_group_paths"]
                    ),
                    prescaled_path_base=_prescaled_path_base,
                    ring_export=_ring_export,
                    k_norm_export=_k_norm_export,
                    gate_export=_gate_export,
                    decay_export=_decay_export,
                    flags_export=_flags_export,
                    flags_rows=_flags_rows,
                    descriptor_execution_sha256=str(
                        _single_contract["execution_sha256"]
                    ),
                    block_v=_ordered_block_v,
                    maxnreg=(128 if _gate_export else None),
                )
                _ordered_candidate_id = (
                    _FR13_FIXED32_GDN_GQA_GROUP3_BV16_CANDIDATE_ID
                    if _ordered_block_v == 16
                    else _FR13_FIXED32_GDN_GQA_GROUP3_CANDIDATE_ID
                )
                _route = (
                    "fixed32_single_launch_gqa_group3_bv16"
                    if _ordered_block_v == 16
                    else "fixed32_single_launch_gqa_group3"
                )
            else:
                _tree_gdn_kernel_fixed32_single_launch[
                    (num_vh, triton.cdiv(dim_v, _path_block_v), 1)
                ](
                    q,
                    k,
                    v,
                    g,
                    beta,
                    raw_a,
                    raw_b,
                    A_log,
                    dt_bias,
                    h0,
                    h0_indices,
                    h0_num_accepted_tokens,
                    _counter_arg,
                    _root_nodes,
                    _branch_nodes,
                    _branch_lengths_arg,
                    _path_indices_arg,
                    _single_launch["path_counts"],
                    _out,
                    ring_k,
                    ring_v,
                    ring_a,
                    ring_b,
                    _flags_arg,
                    ring_k_norm,
                    ring_gate,
                    N_ACTUAL=n_actual,
                    NUM_KH=num_kh,
                    NUM_VH=num_vh,
                    DIM_K=dim_k,
                    DIM_V=dim_v,
                    BLOCK_V=_path_block_v,
                    OUTPUT_SCALE=output_scale,
                    USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                    H0_IS_BANK=h0_is_bank,
                    H0_INDEX_ROW=h0_index_row,
                    H0_INDEX_BATCH_STRIDE=int(h0_indices.stride(0)),
                    H0_BATCH_INDEX=h0_batch_index,
                    H0_ACCEPTED_BATCH_STRIDE=int(
                        h0_num_accepted_tokens.stride(0)
                    ),
                    H0_BANK_STRIDE=h0_bank_stride,
                    H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
                    RAW_GATING=raw_gating,
                    COUNT_INVOCATION=_count,
                    SCAN_ALIGN=_scan_align,
                    ROOT_STEPS=int(_single_contract["groups"]),
                    MAX_PATH_LEN=_branch_max_len,
                    MAX_GROUP_PATHS=int(
                        _single_contract["max_group_paths"]
                    ),
                    NUM_GROUPS=int(_single_contract["groups"]),
                    PRESCALED_PATH_BASE=_prescaled_path_base,
                    RING_EXPORT=_ring_export,
                    K_NORM_EXPORT=_k_norm_export,
                    GATE_EXPORT=_gate_export,
                    DECAY_EXPORT=_decay_export,
                    FLAGS_EXPORT=_flags_export,
                    FLAGS_ROWS=_flags_rows,
                    num_warps=_num_warps,
                    **_extra_launch_kwargs,
                )
                _ordered_candidate_id = (
                    _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID
                )
                _route = "fixed32_single_launch_tree"
            st["last_executed_gdn"] = {
                "route": _route,
                "candidate": _ordered_candidate_id,
                "physical_launches": 1,
                "physical_programs": 1,
                "physical_grid_z": (1,),
                "physical_recurrence_critical_path": 32,
                "state_export_writes": 0,
                "state_parent_reads": 0,
                "prescaled_path_base": bool(_prescaled_path_base),
                "logical_launches": 2,
                "logical_programs": 12,
                "logical_padded_slots": 82,
                "logical_critical_path": 12,
            }
            if not st["engaged_announced"]:
                st["engaged_announced"] = True
                print(
                    "[FR13_SUBTREE_PARALLEL ENGAGED] "
                    f"n_actual={n_actual} schedule={st['schedule']} "
                    f"physical={_route} ordered_root_loop=1 "
                    "critical=32",
                    flush=True,
                )
            return {
                "block_v": int(_path_block_v),
                "launch_key": (
                    "tree_gdn_path",
                    _ordered_candidate_id,
                    int(_path_block_v),
                    int(triton.cdiv(dim_v, _path_block_v)),
                    int(_num_warps),
                    tuple(sorted(_extra_launch_kwargs.items())),
                ),
            }
        for _li, (
            _nodes,
            _pars,
            _mlen,
            _npaths,
            _lengths,
        ) in enumerate(st["levels"]):
            # Generic schedules retain the original dynamic source/export
            # decisions. The validated two-level fixed32 schedule has exactly
            # one h0-rooted path whose five nodes are all handoff parents, then
            # eleven export-rooted terminal paths.
            _state_source = 0
            _export_mode = 0
            if _fixed32_io:
                _state_source = 1 if _li == 0 else 2
                _export_mode = 1 if _li == 0 else 2
            _tree_gdn_path_kernel[
                (
                    num_vh,
                    triton.cdiv(dim_v, _path_block_v),
                    _npaths,
                )
            ](
                q,
                k,
                v,
                g,
                beta,
                raw_a,
                raw_b,
                A_log,
                dt_bias,
                h0,
                h0_indices,
                h0_num_accepted_tokens,
                _counter_arg,
                _nodes,
                _pars,
                _lengths,
                st["export"],
                st["emask"],
                _out,
                ring_k,
                ring_v,
                ring_a,
                ring_b,
                _flags_arg,
                N_ACTUAL=n_actual,
                NUM_KH=num_kh,
                NUM_VH=num_vh,
                DIM_K=dim_k,
                DIM_V=dim_v,
                BLOCK_V=_path_block_v,
                OUTPUT_SCALE=output_scale,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                H0_IS_BANK=h0_is_bank,
                H0_INDEX_ROW=h0_index_row,
                H0_BATCH_INDEX=h0_batch_index,
                H0_BANK_STRIDE=h0_bank_stride,
                H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
                RAW_GATING=raw_gating,
                COUNT_INVOCATION=_count and (_li == 0),
                SCAN_ALIGN=_scan_align,
                MAX_PATH_LEN=_mlen,
                STATE_SOURCE=_state_source,
                EXPORT_MODE=_export_mode,
                RING_EXPORT=_ring_export,
                FLAGS_EXPORT=_flags_export and (_li == 0),
                FLAGS_ROWS=_flags_rows,
                num_warps=_num_warps,
                **_extra_launch_kwargs,
            )
        if not st["engaged_announced"]:
            st["engaged_announced"] = True
            print(
                "[FR13_SUBTREE_PARALLEL ENGAGED] "
                f"n_actual={n_actual} schedule={st['schedule']} "
                f"critical={st['critical']}",
                flush=True,
            )
        return {
            "block_v": int(_path_block_v),
            "launch_key": (
                "tree_gdn_path",
                int(_path_block_v),
                int(triton.cdiv(dim_v, _path_block_v)),
                int(_num_warps),
                tuple(sorted(_extra_launch_kwargs.items())),
            ),
        }

    def _launch(
        _out,
        _pg_flag,
        _hc=(_hc_mask, _hc_rows, _hc_slots_lo, _hc_slots_hi),
        _count=count_invocation,
    ):
        _tree_gdn_kernel[grid](
            q,
            k,
            v,
            g,
            beta,
            raw_a,
            raw_b,
            A_log,
            dt_bias,
            h0,
            h0_indices,
            h0_num_accepted_tokens,
            invocation_counter,
            strict_mask,
            visible_mask,
            _out,
            state,
            ring_k,
            ring_v,
            ring_a,
            ring_b,
            _flags_arg,
            N_ACTUAL=n_actual,
            N_PAD=n_pad,
            NUM_KH=num_kh,
            NUM_VH=num_vh,
            DIM_K=dim_k,
            DIM_V=dim_v,
            BLOCK_V=_bv,
            OUTPUT_SCALE=output_scale,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            H0_IS_BANK=h0_is_bank,
            H0_INDEX_ROW=h0_index_row,
            H0_BATCH_INDEX=h0_batch_index,
            H0_BANK_STRIDE=h0_bank_stride,
            H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
            RAW_GATING=raw_gating,
            COUNT_INVOCATION=_count,
            SCAN_ALIGN=_scan_align,
            N_LOOP=_n_loop,
            PARENT_GATHER=_pg_flag,
            PIGGYBACK_EXPORT=piggyback_export,
            CHAIN_END_IDX=chain_end_idx,
            RING_EXPORT=_ring_export,
            FLAGS_EXPORT=_flags_export,
            FLAGS_ROWS=_flags_rows,
            HC_MASK=_hc[0],
            HC_ROWS=_hc[1],
            HC_SLOTS_LO=_hc[2],
            HC_SLOTS_HI=_hc[3],
            hc_slot_map=_hc_slot_map,
            num_warps=_num_warps,
            **_extra_launch_kwargs,
        )

    def _byte_equal(_left, _right):
        return _fr13_tensor_byte_equal(_left, _right)

    def _snapshot_external():
        _snapshot = {}
        if _ring_export:
            _snapshot.update(
                ring_k=ring_k.clone(),
                ring_v=ring_v.clone(),
                ring_a=ring_a.clone(),
                ring_b=ring_b.clone(),
            )
        if _flags_export:
            _snapshot["flags"] = _flags_arg.clone()
        return _snapshot

    def _restore_external(_snapshot):
        for _name, _tensor in _snapshot.items():
            if _name == "ring_k":
                ring_k.copy_(_tensor)
            elif _name == "ring_v":
                ring_v.copy_(_tensor)
            elif _name == "ring_a":
                ring_a.copy_(_tensor)
            elif _name == "ring_b":
                ring_b.copy_(_tensor)
            else:
                _flags_arg.copy_(_tensor)

    def _external_mismatches(_reference):
        _actual = _snapshot_external()
        return [
            _name
            for _name, _tensor in _reference.items()
            if not _byte_equal(_actual[_name], _tensor)
        ]

    def _counter_before_selfcheck():
        return invocation_counter.clone() if count_invocation else None

    def _assert_counter_once(_before, _label):
        if _before is not None and not torch.equal(
            invocation_counter, _before + 1
        ):
            raise RuntimeError(
                f"{_label} SELFCHECK MISMATCH: invocation counter did not "
                "advance exactly once"
            )

    if _FR13_FIXED32_GDN_PATH_BV_PRODUCTION is not None:
        production_pass = _FR13_FIXED32_GDN_PATH_BV_PRODUCTION_PASS
        if (
            not isinstance(production_pass, dict)
            or _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES
            or not _subtree_route_armed
            or _subtree_selfcheck_armed
            or _subtree_state is None
            or _subtree_state.get("schedule") != "fixed32"
            or _subtree_state.get("fixed32_contract") is None
            or n_actual != 32
            or n_pad != 32
            or _bv != _FR13_FIXED32_GDN_PATH_BV_PRODUCTION
            or _geom != {"BV": _FR13_FIXED32_GDN_PATH_BV_PRODUCTION}
            or _FR13_FIXED32_GDN_PATH_BV_PRODUCTION > dim_v
            or dim_v % _FR13_FIXED32_GDN_PATH_BV_PRODUCTION != 0
            or not _ring_export
            or not _flags_export
            or int(staging_rows) not in production_pass["covered_batches"]
        ):
            raise RuntimeError(
                "FR13 fixed32 GDN BV production geometry/PASS contract drift; "
                "no fallback is permitted"
            )
        if not _subtree_state.get("production_bv_announced", False):
            _subtree_state["production_bv_announced"] = True
            print(
                "[FR13_FIXED32_GDN_PATH_BV_PRODUCTION ENGAGED] "
                f"mode={_FR13_FIXED32_MODE} batch={staging_rows} "
                f"block_v={_FR13_FIXED32_GDN_PATH_BV_PRODUCTION} "
                "source_bound_pass=1 fallback=0",
                flush=True,
            )

    if _FR13_FIXED32_GDN_PATH_BV_CANDIDATE is not None and not (
        _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
        in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        and isinstance(_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT, dict)
        and int(_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT.get("batch_size", -1))
        != 1
    ):
        # The served FULL graph remains the exact BV8 launch above. Capture
        # only persistent operand references; the first measured replay gate
        # allocates private outputs, runs explicit BV8 then BV16/32/64/128, and
        # restores all shared state before returning the graph's BV8 output.
        if (
            _FR13_FIXED32_MODE not in _FR13_FIXED32_MODES
            or not _subtree_route_armed
            or _subtree_state is None
            or _subtree_state.get("schedule") != "fixed32"
            or _subtree_state.get("fixed32_contract") is None
            or n_actual != 32
            or n_pad != 32
            or _bv != 8
            or _geom != {"BV": 8}
            or not _ring_export
            or not _flags_export
        ):
            raise RuntimeError(
                "FR13 fixed32 GDN BV live gate requires the exact fixed32 "
                "BV8 served path with export/rings/flags/counter"
            )

        _gdn_bv_gate_counter_holder = {}

        def _gdn_bv_gate_counter():
            if count_invocation:
                return invocation_counter
            # Formal fixed32 pins FR10_METRICS=0, so the served graph has no
            # counter store. Use one private scalar for both live-gate arms to
            # exercise and byte-compare COUNT_INVOCATION without touching the
            # strict-mask dummy pointer used by the served graph.
            counter = _gdn_bv_gate_counter_holder.get("counter")
            if counter is None:
                counter = torch.zeros(
                    (), dtype=torch.int32, device=q.device
                )
                _gdn_bv_gate_counter_holder["counter"] = counter
            return counter

        _gdn_single_launch_gate = (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        )
        _gdn_gqa_group3_gate = (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in (
                _FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE,
                _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE,
            )
        )
        _gdn_gqa_group3_bv16_gate = (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE
        )
        _gdn_ordered_candidate_id = (
            _fr13_fixed32_gdn_ordered_candidate_id()
            if _gdn_single_launch_gate
            else None
        )

        def _gdn_bv_gate_snapshot():
            assert _subtree_state is not None
            snapshot = {
                "ring_k": ring_k.clone(),
                "ring_v": ring_v.clone(),
                "ring_a": ring_a.clone(),
                "ring_b": ring_b.clone(),
                "flags": _flags_arg.clone(),
                "counter": _gdn_bv_gate_counter().clone(),
            }
            if not _gdn_single_launch_gate:
                snapshot = {
                    "export": _subtree_state["export"].clone(),
                    **snapshot,
                }
            else:
                snapshot = {
                    "output": out.clone(),
                    "export": _subtree_state["export"].clone(),
                    **snapshot,
                }
            return snapshot

        def _gdn_bv_gate_restore(_snapshot):
            assert _subtree_state is not None
            if not _gdn_single_launch_gate:
                _subtree_state["export"].copy_(_snapshot["export"])
            else:
                out.copy_(_snapshot["output"])
                _subtree_state["export"].copy_(_snapshot["export"])
            ring_k.copy_(_snapshot["ring_k"])
            ring_v.copy_(_snapshot["ring_v"])
            ring_a.copy_(_snapshot["ring_a"])
            ring_b.copy_(_snapshot["ring_b"])
            _flags_arg.copy_(_snapshot["flags"])
            _gdn_bv_gate_counter().copy_(_snapshot["counter"])

        def _gdn_bv_gate_run(_path_block_v):
            if _gdn_single_launch_gate:
                if _path_block_v == "reference":
                    _single_launch = False
                    _candidate = "fixed32_gdn_two_launch_reference_v1"
                    _physical_launches = 2
                elif _path_block_v == _gdn_ordered_candidate_id:
                    _single_launch = not _gdn_gqa_group3_gate
                    _candidate = _gdn_ordered_candidate_id
                    _physical_launches = 1
                else:
                    raise RuntimeError(
                        "FR13 fixed32 GDN single-launch live gate rejected "
                        f"candidate selector: {_path_block_v!r}"
                    )
                _gate_out = torch.empty_like(out)
                _gate_block_v = (
                    16
                    if _candidate != "fixed32_gdn_two_launch_reference_v1"
                    and _gdn_gqa_group3_bv16_gate
                    else 8
                )
                _launch_paths(
                    _gate_out,
                    _count=True,
                    _path_block_v=_gate_block_v,
                    _counter_arg=_gdn_bv_gate_counter(),
                    _single_launch_override=_single_launch,
                    _gqa_group3_override=(
                        _candidate != "fixed32_gdn_two_launch_reference_v1"
                        and _gdn_gqa_group3_gate
                    ),
                )
                return {
                    "candidate": _candidate,
                    "physical_launches": _physical_launches,
                    "output": _gate_out,
                }
            _gate_bv = int(_path_block_v)
            if (
                _gate_bv not in (8, 16, 32, 64, 128)
                or _gate_bv > dim_v
                or dim_v % _gate_bv != 0
            ):
                raise RuntimeError(
                    "FR13 fixed32 GDN BV live gate rejected path geometry: "
                    f"BLOCK_V={_gate_bv} DIM_V={dim_v}"
                )
            _gate_out = torch.empty_like(out)
            _launch_meta = _launch_paths(
                _gate_out,
                _count=True,
                _path_block_v=_gate_bv,
                _counter_arg=_gdn_bv_gate_counter(),
            )
            return {
                **_launch_meta,
                "output": _gate_out,
            }

        _fr13_fixed32_gdn_bv_live_capture_register(
            {
                "snapshot": _gdn_bv_gate_snapshot,
                "restore": _gdn_bv_gate_restore,
                "run": _gdn_bv_gate_run,
                "byte_equal": _byte_equal,
                "surface_names": (
                    _FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES
                    if _gdn_single_launch_gate
                    else _FR13_FIXED32_GDN_BV_SURFACES
                ),
            }
        )

    if parent_gather_selfcheck_on():
        # In-process byte-identity gate: run BOTH scans on the SAME inputs and
        # raise on any bit difference (boot enforce_eager; the host compare syncs).
        out_ref = torch.empty_like(out)
        _external_before = _snapshot_external()
        _counter_before = _counter_before_selfcheck()
        _launch(out_ref, False, _count=False)
        _external_ref = _snapshot_external()
        _restore_external(_external_before)
        _launch(out, True)
        _n = int(n_actual)
        _external_bad = _external_mismatches(_external_ref)
        if not _byte_equal(out_ref[:_n], out[:_n]) or _external_bad:
            _diff = (out_ref[:_n].float() - out[:_n].float()).abs().max().item()
            raise RuntimeError(
                "FR13_PARENT_GATHER SELFCHECK MISMATCH: parent gather is NOT "
                f"byte-identical to the ancestor loop (max_abs={_diff:.3e}, "
                f"n_actual={_n}, n_pad={n_pad}, external={_external_bad})"
            )
        _assert_counter_once(_counter_before, "FR13_PARENT_GATHER")
    elif _subtree_route_armed:
        if piggyback_export:
            raise ValueError(
                "FR13_SUBTREE_PARALLEL does not support PIGGYBACK_EXPORT"
            )
        if _subtree_selfcheck_armed:
            # In-process byte gate: monolith -> reference, restore every
            # externally visible buffer, then path route -> candidate.
            out_ref = torch.empty_like(out)
            _external_before = _snapshot_external()
            _counter_before = _counter_before_selfcheck()
            _launch(out_ref, _parent_gather, _count=False)
            _external_ref = _snapshot_external()
            _restore_external(_external_before)
            _launch_paths(out)
            _n = int(n_actual)
            _external_bad = _external_mismatches(_external_ref)
            if not _byte_equal(out_ref[:_n], out[:_n]) or _external_bad:
                _diff = (out_ref[:_n].float() - out[:_n].float()).abs().max().item()
                raise RuntimeError(
                    "FR13_SUBTREE_PARALLEL SELFCHECK MISMATCH: path route is "
                    f"NOT byte-identical to the monolith (max_abs={_diff:.3e}, "
                    f"n_actual={_n}, external={_external_bad})"
                )
            _assert_counter_once(_counter_before, "FR13_SUBTREE_PARALLEL")
            _st = _subtree_state
            assert _st is not None
            if not _st["selfcheck_pass_announced"]:
                _st["selfcheck_pass_announced"] = True
                print(
                    "[FR13_SUBTREE_PARALLEL SELFCHECK PASS] "
                    f"byte_equal=1 external_state_equal=1 counter_once=1 "
                    f"n_actual={_n} "
                    f"schedule={_st['schedule']} critical={_st['critical']}",
                    flush=True,
                )
        else:
            _launch_paths(out)
    elif hc_internal_selfcheck_on() and _hc_mask != 0:
        # In-process byte-identity gate for FR13_HC_INTERNAL (same discipline:
        # boot enforce_eager; the host compare syncs).
        out_ref = torch.empty_like(out)
        _external_before = _snapshot_external()
        _counter_before = _counter_before_selfcheck()
        _launch(
            out_ref,
            _parent_gather,
            _hc=(0, 0, 0, 0),
            _count=False,
        )
        _external_ref = _snapshot_external()
        _restore_external(_external_before)
        _launch(out, _parent_gather)
        _n = int(n_actual)
        _external_bad = _external_mismatches(_external_ref)
        if not _byte_equal(out_ref[:_n], out[:_n]) or _external_bad:
            _diff = (out_ref[:_n].float() - out[:_n].float()).abs().max().item()
            raise RuntimeError(
                "FR13_HC_INTERNAL SELFCHECK MISMATCH: compacted h_cache is NOT "
                f"byte-identical to the full-span cache (max_abs={_diff:.3e}, "
                f"n_actual={_n}, n_pad={n_pad}, hc_mask=0x{_hc_mask:x}, "
                f"external={_external_bad})"
            )
        _assert_counter_once(_counter_before, "FR13_HC_INTERNAL")
    else:
        _launch(out, _parent_gather)
    return out, None


def fixed32_batch_gdn_launch_contract(
    batch_size: int,
    *,
    n_actual: int,
    n_pad: int,
    block_v: int,
    dim_v: int,
) -> dict[str, object]:
    """Validate the closed fixed32 geometry and expose its launch invariant."""
    batch = int(batch_size)
    rows = int(n_actual)
    padded_rows = int(n_pad)
    bv = int(block_v)
    width = int(dim_v)
    if batch < 2 or batch > _FR13_FIXED32_MAX_BATCH:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: batch_size must be in [2, 4]; "
            f"got {batch}"
        )
    if rows != len(_FR13_FIXED32_PARENT) or padded_rows != 32:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN requires the exact 32-row physical tree, "
            f"got n_actual={rows} n_pad={padded_rows}"
        )
    if bv not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: BLOCK_V must be a supported power of two "
            f"through 128, got {bv}"
        )
    if width <= 0 or bv > width or width % bv != 0:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: BLOCK_V must divide DIM_V without "
            f"exceeding it, got BLOCK_V={bv} DIM_V={width}"
        )
    return {
        "batch_size": batch,
        "physical_rows_per_request": padded_rows,
        "block_v": bv,
        "physical_launches_per_layer": 2,
        "level_grid_z": (batch, 11 * batch),
        "path_programs": 12 * batch,
    }


def launch_tree_gdn_prepared_fixed32_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    *,
    batch_size: int,
    n_actual: int,
    n_pad: int,
    strict_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    out: torch.Tensor,
    h0_indices: torch.Tensor,
    h0_num_accepted_tokens: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    output_scale: float = 1.0,
    use_qk_l2norm_in_kernel: bool = False,
    h0_index_row: int = 0,
    h0_batch_index: int = 0,
    h0_use_accepted_column: bool = False,
    invocation_counter: torch.Tensor | None = None,
    ring_k: torch.Tensor | None = None,
    ring_v: torch.Tensor | None = None,
    ring_a: torch.Tensor | None = None,
    ring_b: torch.Tensor | None = None,
    ring_k_norm: torch.Tensor | None = None,
    ring_gate: torch.Tensor | None = None,
    staging_flags: torch.Tensor | None = None,
    staging_rows: int = 0,
) -> tuple[torch.Tensor, None]:
    """Launch the exact fixed32 B2-B4 path or B4 single-launch scan.

    Request programs are folded into path-grid axis 2. The existing 32-row
    subtree export scratch is reused as ``[batch, 5]`` compact parent slots,
    so B4 consumes 20 rows and does not allocate inside graph capture.

    The eager diagnostic compares both arms inline. The graph diagnostic
    captures the legacy per-request BV8 route, retains only persistent operand
    closures, and performs the candidate comparison after a real B4 replay.
    """
    batch = int(batch_size)
    if batch < 2 or batch > _FR13_FIXED32_MAX_BATCH:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: batch_size must be in [2, 4]; B1 must "
            "use the legacy per-request route, "
            f"got {batch}"
        )
    selector = fixed32_batch_gdn_selector(batch)
    if selector is None:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN is default-off; arm either its diagnostic "
            "or post-gate production selector"
        )
    if _FR13_FIXED32_MODE is None:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN requires an armed fixed32 runtime"
        )
    if int(n_actual) != len(_FR13_FIXED32_PARENT) or int(n_pad) != 32:
        fixed32_batch_gdn_launch_contract(
            batch,
            n_actual=n_actual,
            n_pad=n_pad,
            block_v=8,
            dim_v=128,
        )
    for name, mask in (("strict_mask", strict_mask), ("visible_mask", visible_mask)):
        if mask.ndim != 2 or mask.shape[0] < n_pad or mask.shape[1] < n_pad:
            raise ValueError(
                f"FR13_FIXED32_BATCH_GDN: {name} must cover "
                f"{n_pad}x{n_pad}, got {tuple(mask.shape)}"
            )
    rows = batch * int(n_actual)
    tensors_with_rows = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "raw_a": raw_a,
        "raw_b": raw_b,
        "out": out,
    }
    for name, tensor in tensors_with_rows.items():
        if tensor.shape[0] < rows:
            raise ValueError(
                f"FR13_FIXED32_BATCH_GDN: {name} has {tensor.shape[0]} rows, "
                f"needs {rows}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"FR13_FIXED32_BATCH_GDN: {name} must be contiguous"
            )

    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: q/k must have equal [B*32, KH, DK] "
            f"shapes, got q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    if v.ndim != 3:
        raise ValueError(
            f"FR13_FIXED32_BATCH_GDN: v must be rank 3, got {tuple(v.shape)}"
        )
    num_kh = int(q.shape[1])
    num_vh = int(v.shape[1])
    dim_k = int(q.shape[2])
    dim_v = int(v.shape[2])
    if num_vh % num_kh != 0:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: value heads must be a multiple of "
            f"q/k heads, got {num_vh}/{num_kh}"
        )
    if v.shape[2] != dim_v or out.shape[1:] != (num_vh, dim_v):
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: v/out shape mismatch "
            f"v={tuple(v.shape)} out={tuple(out.shape)}"
        )
    gate_shape = (num_vh,)
    for name, tensor in (
        ("g", g),
        ("beta", beta),
        ("raw_a", raw_a),
        ("raw_b", raw_b),
    ):
        if tensor.shape[1:] != gate_shape:
            raise ValueError(
                f"FR13_FIXED32_BATCH_GDN: {name} trailing shape must be "
                f"{gate_shape}, got {tuple(tensor.shape)}"
            )
    if A_log.numel() < num_vh or dt_bias.numel() < num_vh:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: A_log/dt_bias do not cover all value "
            f"heads ({A_log.numel()}/{dt_bias.numel()} < {num_vh})"
        )
    if h0.ndim != 4 or h0.shape[1:] != (num_vh, dim_v, dim_k):
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: h0 bank shape must be "
            f"(*, {num_vh}, {dim_v}, {dim_k}), got {tuple(h0.shape)}"
        )
    if h0_indices.ndim != 2 or h0_indices.shape[0] < batch:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: h0_indices must cover [B, SPEC_COLS], "
            f"got {tuple(h0_indices.shape)} for B={batch}"
        )
    if h0_num_accepted_tokens.ndim != 1:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: accepted-token counts must be rank 1"
        )
    last_accepted_index = (
        int(h0_batch_index) + (batch - 1) * int(h0_num_accepted_tokens.stride(0))
    )
    if last_accepted_index >= h0_num_accepted_tokens.numel():
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: accepted-token counts do not cover "
            f"batch index {last_accepted_index}"
        )
    last_h0_row = int(h0_index_row) + (batch - 1) * int(h0_indices.stride(0))
    if last_h0_row < 0 or last_h0_row >= h0_indices.numel():
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: h0 index rows do not cover flattened "
            f"row {last_h0_row}"
        )

    ring_export = ring_k is not None
    if ring_export:
        if ring_v is None or ring_a is None or ring_b is None:
            raise ValueError(
                "FR13_FIXED32_BATCH_GDN: ring export requires all four rings"
            )
        expected_ring_shapes = (
            ("ring_k", ring_k, (num_kh, dim_k), k.dtype),
            ("ring_v", ring_v, (num_vh, dim_v), v.dtype),
            ("ring_a", ring_a, (num_vh,), raw_a.dtype),
            ("ring_b", ring_b, (num_vh,), raw_b.dtype),
        )
        for name, tensor, trailing_shape, dtype in expected_ring_shapes:
            assert tensor is not None
            if (
                tensor.shape[0] < rows
                or tensor.shape[1:] != trailing_shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"FR13_FIXED32_BATCH_GDN: invalid {name} shape/dtype/layout "
                    f"{tuple(tensor.shape)}/{tensor.dtype}"
                )
    else:
        ring_k = ring_v = ring_a = ring_b = strict_mask
    k_norm_export = ring_k_norm is not None
    if k_norm_export:
        if (
            not ring_export
            or not _fr13_fixed32_committer_knorm_ring_requested()
            or batch != 4
            or selector not in ("single_launch", "gqa_group3")
            or not use_qk_l2norm_in_kernel
            or scan_align_on()
            or ring_k_norm.dtype != torch.float32
            or ring_k_norm.device != k.device
            or ring_k_norm.ndim != 2
            or tuple(ring_k_norm.shape) != (rows, num_kh)
            or not ring_k_norm.is_contiguous()
        ):
            raise RuntimeError(
                "FR13 fixed32 K-norm export requires the armed B4 "
                "single-launch physical32 ring contract"
            )
    else:
        ring_k_norm = strict_mask
    gate_export = ring_gate is not None
    if gate_export:
        if (
            not k_norm_export
            or not _fr13_fixed32_committer_gate_ring_requested()
            or ring_gate.dtype != torch.float32
            or ring_gate.device != k.device
            or ring_gate.ndim != 3
            or tuple(ring_gate.shape) != (rows, num_vh, 2)
            or not ring_gate.is_contiguous()
        ):
            raise RuntimeError(
                "FR13 fixed32 gate export requires the armed B4 "
                "single-launch physical32 K-norm ring contract"
            )
    else:
        ring_gate = strict_mask
    decay_export = _fr13_fixed32_committer_decay_ring_requested()
    if decay_export and not gate_export:
        raise RuntimeError(
            "FR13 fixed32 B4 decay export requires the armed gate-ring route"
        )

    subtree_state = subtree_get(
        n_actual, num_vh, dim_v, dim_k, q.device
    )
    contract = subtree_state.get("fixed32_contract")
    parent_slots = subtree_state.get("fixed32_parent_slots")
    single_launch = subtree_state.get("fixed32_single_launch")
    if (
        subtree_state.get("schedule") != "fixed32"
        or contract is None
        or int(contract.get("launches", -1)) != 2
        or parent_slots is None
        or len(parent_slots) != 2
        or not bool(subtree_state.get("route_armed"))
        or bool(subtree_state.get("selfcheck_armed"))
    ):
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN: exact two-level armed preseed contract "
            "missing or legacy subtree selfcheck is still armed"
        )
    if selector in ("single_launch", "single_launch_gate", "gqa_group3") and (
        not (
            _FR13_FIXED32_GDN_SINGLE_LAUNCH
            # The credentialed arm is the fourth way "single_launch" can be
            # selected at B4, and it must be admitted here for the same reason
            # it was admitted in the selector: it reaches the same kernel under
            # the same descriptor, only with the live gate's PASS behind it.
            or _fr13_fixed32_gdn_single_launch_production_for_batch(4)
            or _fr13_fixed32_gdn_gqa_group3_production_for_batch(4)
            or _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in ("single_launch", "gqa_group3", "gqa_group3_bv16")
        )
        or batch != 4
        or not isinstance(single_launch, dict)
        or not isinstance(single_launch.get("contract"), dict)
        or single_launch["contract"].get("candidate")
        != _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID
        or single_launch["contract"].get("physical_grid_z") != (1,)
        or single_launch["contract"].get("critical_node_steps") != 32
        or single_launch["contract"].get("outer_root_loop")
        != "ordered_tl_range"
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH exact K64/root1 B4 descriptor "
            "drift; no fallback is permitted"
        )
    needed_export_rows = batch * _FR13_FIXED32_EXPORT_SLOTS
    if int(subtree_state["export"].shape[0]) < needed_export_rows:
        raise RuntimeError(
            "FR13_FIXED32_BATCH_GDN: compact export scratch capacity "
            f"{subtree_state['export'].shape[0]} < {needed_export_rows}"
        )

    geom = _read_tree_gdn_geom_override()
    block_v = BV
    num_warps = _DEPLOYED_NUM_WARPS
    extra_launch_kwargs: dict = {}
    if geom is not None:
        block_v = int(geom.get("BV", block_v))
        num_warps = int(geom.get("num_warps", num_warps))
        if "num_stages" in geom:
            extra_launch_kwargs["num_stages"] = int(geom["num_stages"])
    if gate_export:
        extra_launch_kwargs["maxnreg"] = 80
    configured_wide_bv = None
    if selector in ("diagnostic", "graph_capture"):
        configured_wide_bv = _FR13_FIXED32_BATCH_GDN_BV_CANDIDATE
    elif selector == "production":
        configured_wide_bv = _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION
    candidate_block_v = configured_wide_bv or block_v
    if configured_wide_bv is None and block_v > 8:
        raise RuntimeError(
            "FR13 fixed32 batched GDN refuses BLOCK_V>8 without an explicit "
            "combined wide-BV selector and live gate"
        )
    if configured_wide_bv is not None and (
        block_v != 8 or geom != {"BV": 8}
    ):
        raise RuntimeError(
            "FR13 fixed32 batched wide-BV requires the served/reference route "
            "pinned exactly to BV=8"
        )
    if selector in ("single_launch", "single_launch_gate", "gqa_group3") and (
        candidate_block_v != 8 or geom != {"BV": 8}
    ):
        raise RuntimeError(
            "FR13_FIXED32_GDN_SINGLE_LAUNCH B4 route is pinned exactly to BV8"
        )
    launch_contract = fixed32_batch_gdn_launch_contract(
        batch,
        n_actual=n_actual,
        n_pad=n_pad,
        block_v=candidate_block_v,
        dim_v=dim_v,
    )

    count_invocation = invocation_counter is not None
    if invocation_counter is None:
        invocation_counter = strict_mask
    flags_export = staging_flags is not None
    flags_arg = staging_flags if flags_export else strict_mask
    flags_rows = int(staging_rows) if flags_export else 0
    if flags_export and flags_rows != batch:
        raise ValueError(
            "FR13_FIXED32_BATCH_GDN: staging_rows must equal batch_size, "
            f"got {flags_rows}/{batch}"
        )

    def _launch_batched(
        _block_v: int,
        *,
        _single_launch_override=None,
        _gqa_group3_override=None,
    ) -> None:
        _single_launch_enabled = (
            selector == "single_launch"
            if _single_launch_override is None
            else bool(_single_launch_override)
        )
        _gqa_group3_enabled = (
            selector == "gqa_group3"
            if _gqa_group3_override is None
            else bool(_gqa_group3_override)
        )
        _ordered_launch_enabled = (
            _single_launch_enabled or _gqa_group3_enabled
        )
        _ordered_block_v = (
            16
            if _gqa_group3_enabled
            and _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE
            else 8
        )
        if _single_launch_enabled and _gqa_group3_enabled:
            raise RuntimeError(
                "FR13 fixed32 batched ordered GDN launch selectors overlapped"
            )
        if _ordered_launch_enabled and _block_v != _ordered_block_v:
            raise RuntimeError(
                "FR13 fixed32 batched ordered GDN BLOCK_V drift"
            )
        if _ordered_launch_enabled:
            assert isinstance(single_launch, dict)
            (
                root_nodes,
                _root_parents,
                _root_max_len,
                _root_paths,
                _root_lengths,
            ) = subtree_state["levels"][0]
            (
                branch_nodes,
                _branch_parents,
                branch_max_len,
                _branch_paths,
                branch_lengths,
            ) = subtree_state["levels"][1]
            single_contract = single_launch["contract"]
            (
                branch_lengths_arg,
                path_indices_arg,
                prescaled_path_base,
            ) = _fr13_fixed32_gdn_single_launch_path_args(
                single_launch,
                branch_lengths,
                branch_max_len,
            )
            if _gqa_group3_enabled:
                if not callable(_FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH):
                    raise RuntimeError(
                        "FR13 fixed32 batched GDN GQA-group3 launch was not preloaded"
                    )
                _FR13_FIXED32_GDN_GQA_GROUP3_LAUNCH(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    raw_a=raw_a,
                    raw_b=raw_b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    h0=h0,
                    h0_indices=h0_indices,
                    h0_num_accepted_tokens=h0_num_accepted_tokens,
                    invocation_counter=invocation_counter,
                    root_nodes=root_nodes,
                    branch_nodes=branch_nodes,
                    branch_lengths=branch_lengths_arg,
                    group_path_indices=path_indices_arg,
                    group_path_counts=single_launch["path_counts"],
                    out=out,
                    ring_k=ring_k,
                    ring_v=ring_v,
                    ring_a=ring_a,
                    ring_b=ring_b,
                    flags=flags_arg,
                    ring_k_norm=ring_k_norm,
                    ring_gate=ring_gate,
                    batch_size=batch,
                    mode=_FR13_FIXED32_MODE,
                    output_scale=output_scale,
                    h0_is_bank=True,
                    h0_index_row=int(h0_index_row),
                    h0_index_batch_stride=int(h0_indices.stride(0)),
                    h0_batch_index=int(h0_batch_index),
                    h0_accepted_batch_stride=int(
                        h0_num_accepted_tokens.stride(0)
                    ),
                    h0_bank_stride=int(h0.stride(0)),
                    h0_use_accepted_column=h0_use_accepted_column,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    raw_gating=True,
                    count_invocation=count_invocation,
                    scan_align=scan_align_on(),
                    root_steps=int(single_contract["groups"]),
                    max_path_len=branch_max_len,
                    max_group_paths=int(
                        single_contract["max_group_paths"]
                    ),
                    prescaled_path_base=prescaled_path_base,
                    ring_export=ring_export,
                    k_norm_export=k_norm_export,
                    gate_export=gate_export,
                    decay_export=decay_export,
                    flags_export=flags_export,
                    flags_rows=flags_rows,
                    descriptor_execution_sha256=str(
                        single_contract["execution_sha256"]
                    ),
                    block_v=_ordered_block_v,
                    maxnreg=(128 if gate_export else None),
                )
                ordered_candidate_id = (
                    _FR13_FIXED32_GDN_GQA_GROUP3_BV16_CANDIDATE_ID
                    if _ordered_block_v == 16
                    else _FR13_FIXED32_GDN_GQA_GROUP3_CANDIDATE_ID
                )
                ordered_route = (
                    "fixed32_single_launch_gqa_group3_bv16"
                    if _ordered_block_v == 16
                    else "fixed32_single_launch_gqa_group3"
                )
            else:
                _tree_gdn_kernel_fixed32_single_launch[
                    (num_vh, triton.cdiv(dim_v, _block_v), batch)
                ](
                    q,
                    k,
                    v,
                    g,
                    beta,
                    raw_a,
                    raw_b,
                    A_log,
                    dt_bias,
                    h0,
                    h0_indices,
                    h0_num_accepted_tokens,
                    invocation_counter,
                    root_nodes,
                    branch_nodes,
                    branch_lengths_arg,
                    path_indices_arg,
                    single_launch["path_counts"],
                    out,
                    ring_k,
                    ring_v,
                    ring_a,
                    ring_b,
                    flags_arg,
                    ring_k_norm,
                    ring_gate,
                    N_ACTUAL=n_actual,
                    NUM_KH=num_kh,
                    NUM_VH=num_vh,
                    DIM_K=dim_k,
                    DIM_V=dim_v,
                    BLOCK_V=_block_v,
                    OUTPUT_SCALE=output_scale,
                    USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                    H0_IS_BANK=True,
                    H0_INDEX_ROW=int(h0_index_row),
                    H0_INDEX_BATCH_STRIDE=int(h0_indices.stride(0)),
                    H0_BATCH_INDEX=int(h0_batch_index),
                    H0_ACCEPTED_BATCH_STRIDE=int(
                        h0_num_accepted_tokens.stride(0)
                    ),
                    H0_BANK_STRIDE=int(h0.stride(0)),
                    H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
                    RAW_GATING=True,
                    COUNT_INVOCATION=count_invocation,
                    SCAN_ALIGN=scan_align_on(),
                    ROOT_STEPS=int(single_contract["groups"]),
                    MAX_PATH_LEN=branch_max_len,
                    MAX_GROUP_PATHS=int(
                        single_contract["max_group_paths"]
                    ),
                    NUM_GROUPS=int(single_contract["groups"]),
                    PRESCALED_PATH_BASE=prescaled_path_base,
                    RING_EXPORT=ring_export,
                    K_NORM_EXPORT=k_norm_export,
                    GATE_EXPORT=gate_export,
                    DECAY_EXPORT=decay_export,
                    FLAGS_EXPORT=flags_export,
                    FLAGS_ROWS=flags_rows,
                    num_warps=num_warps,
                    **extra_launch_kwargs,
                )
                ordered_candidate_id = (
                    _FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID
                )
                ordered_route = "fixed32_single_launch_tree"
            subtree_state["last_executed_gdn"] = {
                "route": ordered_route,
                "candidate": ordered_candidate_id,
                "physical_launches": 1,
                "physical_programs": batch,
                "physical_grid_z": (batch,),
                "physical_recurrence_critical_path": 32,
                "state_export_writes": 0,
                "state_parent_reads": 0,
                "prescaled_path_base": bool(prescaled_path_base),
                "logical_launches": 2 * batch,
                "logical_programs": 12 * batch,
                "logical_padded_slots": 82 * batch,
                "logical_critical_path": 12,
            }
            return
        for level_index, (
            nodes,
            _parents,
            max_path_len,
            num_paths,
            path_lengths,
        ) in enumerate(subtree_state["levels"]):
            state_source = 1 if level_index == 0 else 2
            export_mode = 1 if level_index == 0 else 2
            _tree_gdn_path_kernel_fixed32_batch[
                (num_vh, triton.cdiv(dim_v, _block_v), batch * num_paths)
            ](
                q,
                k,
                v,
                g,
                beta,
                raw_a,
                raw_b,
                A_log,
                dt_bias,
                h0,
                h0_indices,
                h0_num_accepted_tokens,
                invocation_counter,
                nodes,
                parent_slots[level_index],
                path_lengths,
                subtree_state["export"],
                out,
                ring_k,
                ring_v,
                ring_a,
                ring_b,
                flags_arg,
                N_ACTUAL=n_actual,
                NUM_KH=num_kh,
                NUM_VH=num_vh,
                DIM_K=dim_k,
                DIM_V=dim_v,
                BLOCK_V=_block_v,
                OUTPUT_SCALE=output_scale,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                H0_INDEX_ROW=int(h0_index_row),
                H0_INDEX_BATCH_STRIDE=int(h0_indices.stride(0)),
                H0_BATCH_INDEX=int(h0_batch_index),
                H0_ACCEPTED_BATCH_STRIDE=int(
                    h0_num_accepted_tokens.stride(0)
                ),
                H0_BANK_STRIDE=int(h0.stride(0)),
                H0_USE_ACCEPTED_COLUMN=h0_use_accepted_column,
                RAW_GATING=True,
                COUNT_INVOCATION=count_invocation and level_index == 0,
                SCAN_ALIGN=scan_align_on(),
                MAX_PATH_LEN=max_path_len,
                NUM_PATHS=num_paths,
                BATCH_SIZE=batch,
                EXPORT_SLOTS=_FR13_FIXED32_EXPORT_SLOTS,
                STATE_SOURCE=state_source,
                EXPORT_MODE=export_mode,
                RING_EXPORT=ring_export,
                FLAGS_EXPORT=flags_export and level_index == 0,
                FLAGS_ROWS=flags_rows,
                num_warps=num_warps,
                **extra_launch_kwargs,
            )

    def _launch_reference(*, collect_export: bool) -> torch.Tensor | None:
        reference_exports = []
        for request in range(batch):
            start = request * int(n_actual)
            end = start + int(n_actual)
            launch_tree_gdn_prepared(
                q=q[start:end],
                k=k[start:end],
                v=v[start:end],
                g=g[start:end],
                beta=beta[start:end],
                raw_a=raw_a[start:end],
                raw_b=raw_b[start:end],
                A_log=A_log,
                dt_bias=dt_bias,
                h0=h0,
                h0_indices=h0_indices,
                h0_num_accepted_tokens=h0_num_accepted_tokens,
                h0_is_bank=True,
                h0_index_row=(
                    int(h0_index_row)
                    + request * int(h0_indices.stride(0))
                ),
                h0_batch_index=(
                    int(h0_batch_index)
                    + request * int(h0_num_accepted_tokens.stride(0))
                ),
                h0_use_accepted_column=h0_use_accepted_column,
                n_actual=n_actual,
                n_pad=n_pad,
                strict_mask=strict_mask,
                visible_mask=visible_mask,
                out=out[start:end],
                output_scale=output_scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                invocation_counter=(
                    invocation_counter if count_invocation else None
                ),
                ring_k=ring_k[start:end] if ring_export else None,
                ring_v=ring_v[start:end] if ring_export else None,
                ring_a=ring_a[start:end] if ring_export else None,
                ring_b=ring_b[start:end] if ring_export else None,
                staging_flags=flags_arg if flags_export else None,
                staging_rows=flags_rows,
            )
            if collect_export:
                reference_exports.append(
                    torch.stack(
                        tuple(
                            subtree_state["export"][node]
                            for node in _FR13_FIXED32_EXPORT_NODES
                        ),
                        dim=0,
                    )
                )
        if not collect_export:
            return None
        return torch.cat(reference_exports, dim=0)

    def _snapshot_external() -> dict[str, torch.Tensor]:
        snapshot = {
            "out": out[:rows].clone(),
            "export": subtree_state["export"].clone(),
        }
        if ring_export:
            snapshot.update(
                ring_k=ring_k[:rows].clone(),
                ring_v=ring_v[:rows].clone(),
                ring_a=ring_a[:rows].clone(),
                ring_b=ring_b[:rows].clone(),
            )
        if flags_export:
            snapshot["flags"] = flags_arg.clone()
        if count_invocation:
            snapshot["invocation_counter"] = invocation_counter.clone()
        return snapshot

    def _restore_external(snapshot: dict[str, torch.Tensor]) -> None:
        out[:rows].copy_(snapshot["out"])
        subtree_state["export"].copy_(snapshot["export"])
        if ring_export:
            ring_k[:rows].copy_(snapshot["ring_k"])
            ring_v[:rows].copy_(snapshot["ring_v"])
            ring_a[:rows].copy_(snapshot["ring_a"])
            ring_b[:rows].copy_(snapshot["ring_b"])
        if flags_export:
            flags_arg.copy_(snapshot["flags"])
        if count_invocation:
            invocation_counter.copy_(snapshot["invocation_counter"])

    if selector == "single_launch_gate":
        if batch != 4:
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch gate requires exact B4"
            )
        if not ring_export or not flags_export or not count_invocation:
            raise RuntimeError(
                "FR13 fixed32 GDN single-launch B4 gate requires K/V/A/B "
                "ring export, in-kernel flags, and invocation counter"
            )

        ordered_candidate_id = _fr13_fixed32_gdn_ordered_candidate_id()
        gqa_group3_gate = (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            in (
                _FR13_FIXED32_GDN_GQA_GROUP3_GATE_VALUE,
                _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE,
            )
        )
        gqa_group3_bv16_gate = (
            _FR13_FIXED32_GDN_PATH_BV_CANDIDATE
            == _FR13_FIXED32_GDN_GQA_GROUP3_BV16_GATE_VALUE
        )

        def _single_launch_gate_snapshot():
            return {
                "output": out[:rows].clone(),
                "export": subtree_state["export"].clone(),
                "ring_k": ring_k[:rows].clone(),
                "ring_v": ring_v[:rows].clone(),
                "ring_a": ring_a[:rows].clone(),
                "ring_b": ring_b[:rows].clone(),
                "flags": flags_arg.clone(),
                "counter": invocation_counter.clone(),
            }

        def _single_launch_gate_restore(snapshot):
            out[:rows].copy_(snapshot["output"])
            subtree_state["export"].copy_(snapshot["export"])
            ring_k[:rows].copy_(snapshot["ring_k"])
            ring_v[:rows].copy_(snapshot["ring_v"])
            ring_a[:rows].copy_(snapshot["ring_a"])
            ring_b[:rows].copy_(snapshot["ring_b"])
            flags_arg.copy_(snapshot["flags"])
            invocation_counter.copy_(snapshot["counter"])

        def _single_launch_gate_run(candidate):
            if candidate == "reference":
                _launch_reference(collect_export=False)
                identity = "fixed32_gdn_two_launch_reference_v1"
                physical_launches = 2
            elif candidate == ordered_candidate_id:
                _launch_batched(
                    16 if gqa_group3_bv16_gate else 8,
                    _single_launch_override=not gqa_group3_gate,
                    _gqa_group3_override=gqa_group3_gate,
                )
                identity = ordered_candidate_id
                physical_launches = 1
            else:
                raise RuntimeError(
                    "FR13 fixed32 GDN single-launch B4 gate rejected "
                    f"candidate selector: {candidate!r}"
                )
            return {
                "candidate": identity,
                "physical_launches": physical_launches,
                "output": out[:rows].clone(),
            }

        _fr13_fixed32_gdn_bv_live_capture_register(
            {
                "snapshot": _single_launch_gate_snapshot,
                "restore": _single_launch_gate_restore,
                "run": _single_launch_gate_run,
                "byte_equal": _fr13_tensor_byte_equal,
                "surface_names": (
                    _FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES
                ),
            }
        )
        _launch_reference(collect_export=False)
        return out, None

    byte_ab_enabled = selector == "diagnostic"
    graph_byte_ab_capture = selector == "graph_capture"
    real_event_marker = None
    if byte_ab_enabled:
        _enabled, real_event_marker = (
            _fr13_fixed32_batch_gdn_byte_ab_control()
        )
        if not _enabled:
            raise RuntimeError(
                "FR13 fixed32 batched GDN diagnostic selector drifted"
            )
    layer_key = int(A_log.data_ptr())
    batch_layer_key = (batch, layer_key)
    gate_state = _FR13_FIXED32_BATCH_GDN_BYTE_AB_STATE
    if graph_byte_ab_capture:
        if batch != 4 or not fixed32_batch_gdn_graph_live_capture_active(batch):
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN capture selector drifted"
            )
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN gate requires final CUDA capture"
            )
        if not ring_export or not flags_export or not count_invocation:
            raise RuntimeError(
                "FR13 fixed32 B4 graph GDN byte gate requires K/V/A/B ring "
                "export, in-kernel flags, and the invocation counter"
            )

        def _graph_gate_run_reference() -> dict[str, object]:
            compact_export = _launch_reference(collect_export=True)
            assert compact_export is not None
            return {
                "block_v": 8,
                "physical_launches": 2 * batch,
                "kernel_structure": _FR13_FIXED32_BATCH_GDN_REFERENCE_KERNEL,
                "compact_export": compact_export,
            }

        def _graph_gate_run_candidate(_candidate_bv: int) -> dict[str, object]:
            selected_bv = int(_candidate_bv)
            if selected_bv != int(candidate_block_v):
                raise RuntimeError(
                    "FR13 fixed32 B4 graph GDN candidate selector drift: "
                    f"{selected_bv} != {candidate_block_v}"
                )
            _launch_batched(selected_bv)
            return {
                "block_v": selected_bv,
                "physical_launches": int(
                    launch_contract["physical_launches_per_layer"]
                ),
                "kernel_structure": _FR13_FIXED32_BATCH_GDN_CANDIDATE_KERNEL,
                "compact_export": subtree_state["export"][
                    :needed_export_rows
                ].clone(),
            }

        _fr13_fixed32_batch_gdn_graph_live_capture_register(
            {
                "layer_key": layer_key,
                "snapshot": _snapshot_external,
                "restore": _restore_external,
                "run_reference": _graph_gate_run_reference,
                "run_candidate": _graph_gate_run_candidate,
                "carrier_nonzero": lambda: bool(
                    int(torch.count_nonzero(q[:rows]).item())
                ),
                "byte_equal": _fr13_tensor_byte_equal,
                "surface_names": _FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES,
            }
        )
        _launch_reference(collect_export=False)
        return out, None
    if byte_ab_enabled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FR13_FIXED32_BATCH_GDN_BYTE_AB is eager-only; boot with "
                "ENFORCE_EAGER=1"
            )
        if not ring_export or not flags_export or not count_invocation:
            raise RuntimeError(
                "FR13_FIXED32_BATCH_GDN_BYTE_AB requires K/V/A/B ring export, "
                "in-kernel flags, and the invocation counter"
            )
        if batch_layer_key in gate_state["passed"]:
            _launch_reference(collect_export=False)
            return out, None
        elif real_event_marker is None:
            _launch_reference(collect_export=False)
            if batch_layer_key not in gate_state["waiting_announced"]:
                gate_state["waiting_announced"].add(batch_layer_key)
                print(
                    "[FR13_FIXED32_BATCH_GDN_BYTE_AB WAITING] "
                    f"batch={batch} layer_key=0x{layer_key:x} "
                    "reference_served=1",
                    flush=True,
                )
            return out, None
        elif int(torch.count_nonzero(q[:rows]).item()) == 0:
            _launch_reference(collect_export=False)
            _fr13_fixed32_batch_gdn_byte_ab_emit(
                {
                    "task_marker": real_event_marker,
                    "layer_key": f"0x{layer_key:x}",
                    "batch": batch,
                    "physical_rows_per_request": int(n_pad),
                    "reference_bv": int(block_v),
                    "candidate_bv": int(candidate_block_v),
                    "carrier_nonzero": False,
                    "zero_diff": None,
                    "reference_restored_and_served": True,
                    "status": "zero_carrier_reference_served",
                }
            )
            return out, None
        else:
            bound_marker = gate_state.get("task_marker")
            bound_candidate_bv = gate_state.get("candidate_bv")
            if bound_marker is None:
                gate_state["task_marker"] = real_event_marker
                gate_state["candidate_bv"] = int(candidate_block_v)
            elif (
                bound_marker != real_event_marker
                or type(bound_candidate_bv) is not int
                or int(bound_candidate_bv) != int(candidate_block_v)
            ):
                raise RuntimeError(
                    "FR13 fixed32 batched GDN live gate cannot combine task "
                    "markers or BV candidates in one process"
                )
            attempt = int(gate_state["attempts"].get(batch_layer_key, 0)) + 1
            gate_state["attempts"][batch_layer_key] = attempt
            before = _snapshot_external()
            reference_export = _launch_reference(collect_export=True)
            assert reference_export is not None
            reference = _snapshot_external()
            candidate = None
            try:
                _restore_external(before)
                _launch_batched(candidate_block_v)
                candidate = _snapshot_external()
                comparison_inputs = [
                    ("out", reference["out"], candidate["out"]),
                    ("ring_k", reference["ring_k"], candidate["ring_k"]),
                    ("ring_v", reference["ring_v"], candidate["ring_v"]),
                    ("ring_a", reference["ring_a"], candidate["ring_a"]),
                    ("ring_b", reference["ring_b"], candidate["ring_b"]),
                    (
                        "state_export_compact",
                        reference_export,
                        candidate["export"][:needed_export_rows],
                    ),
                    (
                        "state_export_untouched_tail",
                        before["export"][needed_export_rows:],
                        candidate["export"][needed_export_rows:],
                    ),
                    ("flags", reference["flags"], candidate["flags"]),
                    (
                        "invocation_counter",
                        reference["invocation_counter"],
                        candidate["invocation_counter"],
                    ),
                ]
                comparisons = [
                    _fr13_fixed32_batch_gdn_byte_diff(name, left, right)
                    for name, left, right in comparison_inputs
                ]
            finally:
                _restore_external(reference)
            first_nonzero = next(
                (
                    item
                    for item in comparisons
                    if not bool(item["byte_equal"])
                ),
                None,
            )
            zero_diff = first_nonzero is None
            record = {
                "task_marker": real_event_marker,
                "layer_key": f"0x{layer_key:x}",
                "batch": batch,
                "attempt": attempt,
                "physical_rows_per_request": int(n_pad),
                "reference_bv": int(block_v),
                "candidate_bv": int(candidate_block_v),
                "carrier_nonzero": True,
                "legacy_physical_launches": 2 * batch,
                "candidate_physical_launches": int(
                    launch_contract["physical_launches_per_layer"]
                ),
                "comparisons": comparisons,
                "first_nonzero": first_nonzero,
                "zero_diff": zero_diff,
                "reference_restored_and_served": True,
                "status": "pass" if zero_diff else "mismatch_reference_served",
            }
            _fr13_fixed32_batch_gdn_byte_ab_emit(record)
            if zero_diff:
                gate_state["passed"].add(batch_layer_key)
                batch_layer_keys = {
                    passed_layer_key
                    for passed_batch, passed_layer_key in gate_state["passed"]
                    if passed_batch == batch
                }
                _fr13_fixed32_batch_gdn_live_pass_emit(
                    task_marker=real_event_marker,
                    batch=batch,
                    layer_keys=batch_layer_keys,
                    reference_bv=block_v,
                    candidate_bv=candidate_block_v,
                )
            print(
                "[FR13_FIXED32_BATCH_GDN_BYTE_AB "
                f"{'PASS' if zero_diff else 'MISMATCH'}] "
                f"task={real_event_marker} layer_key=0x{layer_key:x} "
                f"attempt={attempt} reference_bv={block_v} "
                f"candidate_bv={candidate_block_v} reference_served=1 "
                f"first_nonzero={first_nonzero}",
                flush=True,
            )
            return out, None
    elif selector == "single_launch":
        _launch_batched(candidate_block_v)
    elif selector == "gqa_group3":
        _launch_batched(candidate_block_v)
    elif selector == "production":
        if _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION == 8:
            if (
                not count_invocation
                or not ring_export
                or not flags_export
                or scan_align_on()
                or npad_invariant_on()
            ):
                raise RuntimeError(
                    "FR13 fixed32 batched BV8 production specialization "
                    "requires metrics/ring/flags on and scan/npad off"
                )
            _fr13_fixed32_batch_gdn_bv8_production_capture_register(
                batch_size=batch,
                layer_key=layer_key,
                candidate_bv=candidate_block_v,
            )
        elif _FR13_FIXED32_BATCH_GDN_BV_PRODUCTION is not None:
            _fr13_fixed32_batch_gdn_bv64_production_capture_register(
                batch_size=batch,
                layer_key=layer_key,
                candidate_bv=candidate_block_v,
            )
        _launch_batched(candidate_block_v)
    else:
        raise RuntimeError(
            f"FR13 fixed32 batched GDN selector is invalid: {selector!r}"
        )
    if not subtree_state.get("batch_engaged_announced", False):
        subtree_state["batch_engaged_announced"] = True
        route_detail = (
            f"launches=1 path_grids=({batch},) export_rows=0 "
            "ordered_root_loop=1"
            if selector in ("single_launch", "gqa_group3")
            else (
                f"launches=2 path_grids=({batch},{11 * batch}) "
                f"export_rows={needed_export_rows}"
            )
        )
        print(
            "[FR13_FIXED32_BATCH_GDN ENGAGED] "
            f"batch={batch} {route_detail} block_v={candidate_block_v} "
            "fallback=0",
            flush=True,
        )
    return out, None
