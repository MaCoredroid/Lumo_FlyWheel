"""Default-off batched slot scatter for the fixed32 conv-commit publish block.

WHAT THIS IS
------------
The fixed32 device commit route publishes THIS event's accepted tree paths into
two persistent buffer families before the conv committer and the GDN replay
consume them:

  * ``spec_paths`` / ``spec_lens``  -- compact spec-row order, rows ``[0, B)``.
    These are the tensors ``launch_fixed32_conv_commit_to_col0`` reads.
  * ``slot_paths`` / ``slot_lens``  -- sampler-slot order, one row per FULL
    batch slot. Sparse: only the ``B`` rows named by ``slot_indices`` are
    written; every other row keeps its previous content.

The incumbent publish writes the slot family with one ``copy_`` PER ROW inside a
Python loop, so the block costs ``2 * B + 2`` ATen launches per event. Every one
of those launches moves a 16-element int32 row: pure launch latency on the
serving path. This module replaces the loop with two ``index_copy_`` calls fed
from the already-published compact rows, making the block a constant 4 launches.

  incumbent : 2*B + 2   (B=1 -> 4, B=2 -> 6, B=4 -> 10)
  batched   : 4         (B=1 -> 4, B=2 -> 4, B=4 -> 4)

The lever is therefore worth 0 launches at B1, 2 at B2 and 6 at B4. It is a B4
lever and is documented as one; see
``results/fr13_conv_plumbing_20260811/design.md`` for why the rest of the
conv-plumbing rung is already retired and this is what is left of it.

BYTE SAFETY
-----------
Three facts, each enforced rather than asserted in prose:

1. ``spec_paths[:batch]`` is, by the two lines that immediately precede this
   block, exactly ``device_paths`` cast to the slot dtype. The incumbent loop
   casts ``device_paths[compact_row]`` to that SAME dtype through ``copy_``.
   int64 -> int32 narrowing is deterministic and elementwise, so sourcing the
   slot rows from the compact rows instead of from ``device_paths`` writes the
   identical bytes.

2. The reorder (compact publish first, slot publish second) is observation-free
   only if the two families are DISJOINT storage. That is checked here, per
   event, on the untyped storage -- not assumed from the fact that they are
   different globals.

3. ``index_copy_`` requires its index to be unique or the result is
   order-dependent. ``slot_indices`` is already proven distinct upstream (the
   route rejects duplicate sampler/spec request ids), and this module re-proves
   it locally so the guarantee does not depend on a caller three frames away.

Any precondition failure RAISES. There is no silent fallback to the sequential
path: a batched publish that cannot prove itself must abort the event, because
a wrong publish here corrupts the conv commit AND the GDN replay for the step.

FLAG
----
``FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS``, strict ``"0"``/``"1"``, default
``"0"``. Fixed32-only: :func:`assert_batched_slots_requires_fixed32` refuses the
flag when ``FR13_FIXED32_MODE`` is not armed, mirroring
``_fr13_assert_mamba_spec_blocks_cdiv_requires_fixed32``. Outside fixed32 the
publish block feeding this code does not exist (the generic routes stage paths
through the host list transport), so arming the flag there would be a lie about
what is running.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


SCHEMA = "fr13.fixed32.conv_commit.batched_slots.v1"
FLAG_ENV = "FR13_FIXED32_CONV_COMMIT_BATCHED_SLOTS"
# ROUND 18 ADJUDICATION -- KEPT hydra27/tail6, deliberately not widened.
# Default-off batched slot scatter for the conv-commit publish block, tied to
# hydra27's commit-slot geometry. hydra31 arms four more drafts and a deeper
# spine, so the scatter plan is a different plan. Kept refusing.
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))

# Bounded so a pathological request-rotation pattern cannot grow the cache
# without limit. 4 slots admit 4! = 24 orderings per device; 512 is far past
# any legal working set and still small enough to notice in a heap dump.
SLOT_SCATTER_INDEX_CACHE_LIMIT = 512

_SLOT_SCATTER_INDEX: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}


def resolve_batched_slots(env: Mapping[str, str]) -> bool:
    """Strict 0/1 resolution of the batched-slot flag. Default OFF.

    Anything other than exactly ``"0"`` or ``"1"`` raises. A typo must not be
    read as OFF: the campaign has been burned by levers that silently declined
    to arm and then produced a "candidate" arm that was the stock path.
    """
    raw = str(env.get(FLAG_ENV, "0")).strip()
    if raw not in ("0", "1"):
        raise RuntimeError(
            f"{FLAG_ENV} must be exactly 0 or 1 (observed {raw!r})"
        )
    return raw == "1"


def assert_batched_slots_requires_fixed32(
    *, enabled: bool, fixed32_mode: object
) -> None:
    """FAIL-LOUD: the batched slot publish is a FIXED32-ONLY lever.

    The transformation sources the slot rows from ``spec_paths[:batch]``, which
    only carries the published compact rows on the fixed32 device commit route.
    The generic (staged / S1 step-graph) routes publish through a different
    transport -- a host list reordered per spec request id -- and never
    establish the compact-then-slot ordering this lever depends on. Arming the
    flag outside fixed32 would therefore either do nothing (the code is not
    reached) or read an unpublished buffer. Refuse instead of documenting it.
    """
    if not enabled:
        return
    if isinstance(fixed32_mode, str) and fixed32_mode in FIXED32_MODES:
        return
    raise RuntimeError(
        f"{FLAG_ENV}=1 requires fixed32 to be armed but FR13_FIXED32_MODE is "
        f"{fixed32_mode!r}. The batched slot publish sources sampler-slot rows "
        "from the compact rows this route publishes one statement earlier; "
        "outside fixed32 the commit route is the staged/S1 transport, which "
        "reorders through a host list and never publishes those compact rows, "
        "so the lever has no legal meaning there. Set FR13_FIXED32_MODE (one "
        f"of {sorted(FIXED32_MODES)}) or unset {FLAG_ENV}."
    )


def batched_slot_launch_census(*, batch: int, batched: bool) -> int:
    """ATen launches this publish block costs. Self-naming, not a comment.

    Used by the unit tests to pin the launch arithmetic and by the design note
    so the claimed saving is generated from the same expression the code obeys.
    """
    if type(batch) is not int or batch < 1:
        raise ValueError(f"batch must be a positive int (observed {batch!r})")
    # compact publish: paths + lens = 2, always.
    if batched:
        # + one index_copy_ per family. The index tensor is cached, so the
        # steady state adds nothing; a cache MISS adds one H2D and is counted
        # by batched_slot_index_build_launches.
        return 2 + 2
    return 2 + 2 * batch


def batched_slot_index_build_launches(*, cache_hit: bool) -> int:
    """H2D launches the index build costs (0 on the steady-state cache hit)."""
    return 0 if cache_hit else 1


def slot_scatter_index(
    slot_indices: tuple[int, ...] | list[int],
    device: torch.device,
) -> torch.Tensor:
    """Persistent int64 index for the batched slot scatter.

    A fresh ``torch.tensor(...)`` per event would cost an H2D copy and eat the
    launch saving outright, so the tensor is cached on the exact
    ``(slot tuple, device)`` key. The cached tensor is RE-VALIDATED on every
    hit: a cache is a place for bugs to hide, and this one addresses writes into
    the buffers the conv committer reads.

    Building during CUDA-graph capture is refused. An H2D inside a capture bakes
    THIS event's slot ordering into every replay of the graph.
    """
    if not isinstance(slot_indices, (tuple, list)):
        raise TypeError(
            "slot_indices must be a tuple/list of ints (observed "
            f"{type(slot_indices).__name__})"
        )
    key_indices = tuple(int(value) for value in slot_indices)
    if not key_indices:
        raise ValueError("slot_indices is empty")
    if any(value < 0 for value in key_indices):
        raise ValueError(f"slot_indices has a negative row: {key_indices!r}")
    if len(set(key_indices)) != len(key_indices):
        raise ValueError(
            "slot_indices must be distinct for index_copy_ (observed "
            f"{key_indices!r})"
        )
    if not isinstance(device, torch.device):
        raise TypeError(
            f"device must be a torch.device (observed {device!r})"
        )
    key = (key_indices, str(device))
    cached = _SLOT_SCATTER_INDEX.get(key)
    if cached is not None:
        if (
            cached.dtype != torch.int64
            or cached.device != device
            or int(cached.numel()) != len(key_indices)
            or not cached.is_contiguous()
            or tuple(int(v) for v in cached.tolist()) != key_indices
        ):
            raise RuntimeError(
                f"{FLAG_ENV}: slot scatter index cache drift for {key!r}"
            )
        return cached
    if _capture_active():
        raise RuntimeError(
            f"{FLAG_ENV}: slot scatter index built during CUDA-graph capture "
            f"for {key!r}; the H2D would bake one event's slot ordering into "
            "every replay. Warm the cache before capture."
        )
    if len(_SLOT_SCATTER_INDEX) >= SLOT_SCATTER_INDEX_CACHE_LIMIT:
        raise RuntimeError(
            f"{FLAG_ENV}: slot scatter index cache exceeded "
            f"{SLOT_SCATTER_INDEX_CACHE_LIMIT} entries; the slot ordering is "
            "not converging and the cache is no longer amortising anything."
        )
    built = torch.tensor(key_indices, dtype=torch.int64, device=device)
    _SLOT_SCATTER_INDEX[key] = built
    return built


def clear_slot_scatter_index_cache() -> None:
    """Drop every cached index (tests and teardown only)."""
    _SLOT_SCATTER_INDEX.clear()


def slot_scatter_index_cache_size() -> int:
    """Cached index count -- an engagement needle, not a debug print."""
    return len(_SLOT_SCATTER_INDEX)


def _capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _disjoint_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """True when the two tensors share no byte of storage."""
    if left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr():
        return True
    return False


def publish_committer_paths(
    *,
    slot_paths: torch.Tensor,
    slot_lens: torch.Tensor,
    spec_paths: torch.Tensor,
    spec_lens: torch.Tensor,
    device_paths: torch.Tensor,
    device_lens: torch.Tensor,
    slot_indices: tuple[int, ...],
    batch: int,
    batched: bool,
) -> None:
    """Publish this event's accepted paths into the compact and slot families.

    ``batched=False`` is the incumbent, statement-for-statement. ``batched=True``
    is the constant-launch form. Both write identical bytes to both families;
    :mod:`tests.test_fr13_fixed32_conv_commit_batched_slots` proves it on
    synthetic states for every batch in 1..4, both narrowing regimes and every
    slot permutation.
    """
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError(
            f"batch must be an int in [1, 4] (observed {batch!r})"
        )
    if len(slot_indices) != batch:
        raise RuntimeError(
            "FR13 fixed32 committer publish slot map width "
            f"{len(slot_indices)} != batch {batch}"
        )
    if (
        int(device_paths.shape[0]) != batch
        or int(device_lens.shape[0]) != batch
        or int(spec_paths.shape[0]) < batch
        or int(spec_lens.shape[0]) < batch
        or int(slot_paths.shape[1]) != int(spec_paths.shape[1])
        or int(device_paths.shape[1]) != int(spec_paths.shape[1])
        or slot_paths.dtype != spec_paths.dtype
        or slot_lens.dtype != spec_lens.dtype
        or any(int(value) >= int(slot_paths.shape[0]) for value in slot_indices)
        or any(int(value) >= int(slot_lens.shape[0]) for value in slot_indices)
        or any(int(value) < 0 for value in slot_indices)
        or len(set(int(value) for value in slot_indices)) != batch
    ):
        raise RuntimeError(
            "FR13 fixed32 committer publish geometry/slot-map drift"
        )
    if not batched:
        for compact_row, slot_row in enumerate(slot_indices):
            slot_paths[slot_row].copy_(device_paths[compact_row])
            slot_lens[slot_row].copy_(device_lens[compact_row])
        spec_paths[:batch].copy_(device_paths)
        spec_lens[:batch].copy_(device_lens)
        return
    # BATCHED. The compact publish MUST run first: the slot publish reads it.
    if not (
        _disjoint_storage(slot_paths, spec_paths)
        and _disjoint_storage(slot_lens, spec_lens)
        and _disjoint_storage(slot_paths, device_paths)
        and _disjoint_storage(slot_lens, device_lens)
    ):
        raise RuntimeError(
            "FR13 fixed32 batched committer publish requires disjoint slot / "
            "compact / product storage; the reorder is only observation-free "
            "when the slot family cannot alias the buffer it now reads"
        )
    index = slot_scatter_index(slot_indices, spec_paths.device)
    spec_paths[:batch].copy_(device_paths)
    spec_lens[:batch].copy_(device_lens)
    slot_paths.index_copy_(0, index, spec_paths[:batch])
    slot_lens.index_copy_(0, index, spec_lens[:batch])
