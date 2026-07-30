from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path("scripts/fr13_merged_drafter.py").resolve()


def _module():
    spec = importlib.util.spec_from_file_location(
        "fr13_fixed32_arctic_lifecycle_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reset_for_test()
    return module


class RecordingCache:
    def __init__(self) -> None:
        self.active_requests: set[str] = set()
        self.cached_requests: set[str] = set()
        self.trees: dict[str, list[int]] = {}
        self.calls: list[tuple[object, ...]] = []
        self.fail_start = False
        self.fail_add = False
        self.fail_stop = False
        self.fail_start_req: str | None = None
        self.fail_add_req: str | None = None
        self.fail_stop_req: str | None = None

    def start_request(self, req_id, prompt) -> None:
        if self.fail_start or str(req_id) == self.fail_start_req:
            raise RuntimeError("injected start failure")
        self.active_requests.add(str(req_id))
        self.cached_requests.add(str(req_id))
        self.trees[str(req_id)] = list(prompt)
        self.calls.append(("start", str(req_id), list(prompt)))

    def add_active_response(self, req_id, tokens) -> None:
        if self.fail_add or str(req_id) == self.fail_add_req:
            raise RuntimeError("injected add failure")
        self.trees[str(req_id)].extend(tokens)
        self.calls.append(("add", str(req_id), list(tokens)))

    def stop_request(self, req_id) -> None:
        if self.fail_stop or str(req_id) == self.fail_stop_req:
            raise RuntimeError("injected stop failure")
        self.active_requests.remove(str(req_id))
        self.calls.append(("stop", str(req_id)))

    def evict_cached_response(self, req_id) -> None:
        self.cached_requests.remove(str(req_id))
        self.trees.pop(str(req_id), None)
        self.calls.append(("evict", str(req_id)))


def _accepted_record(
    step_seq: int,
    full_request_ids: tuple[str, ...],
    rows: dict[str, list[int]],
) -> dict[str, object]:
    request_ids = tuple(rows)
    output_rows = tuple(
        tuple(tokens + [-1] * (32 - len(tokens)))
        for tokens in rows.values()
    )
    return {
        "step_seq": step_seq,
        "request_ids": request_ids,
        "full_request_ids": full_request_ids,
        "output_rows": output_rows,
        "output_lens": tuple(len(tokens) for tokens in rows.values()),
    }


def test_b1_context_is_contiguous_through_prefill_and_tree_accept() -> None:
    md = _module()
    cache = RecordingCache()
    prompt_len = 22_869
    row = np.arange(prompt_len + 64, dtype=np.int64)

    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (prompt_len,),
        (0,),
        (1_024,),
        (0,),
        (prompt_len,),
        (True,),
        1,
    )
    assert md._INGESTED_LEN["request-0"] == prompt_len
    assert md._COMMITTED["request-0"] == row[1_000:1_024].tolist()
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        None,
        (29_999,),
        1,
        30_000,
    )
    assert not [call for call in cache.calls if call[0] == "add"]

    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (prompt_len,),
        (22_528,),
        (341,),
        (0,),
        (prompt_len,),
        (False,),
        2,
    )
    assert md._COMMITTED["request-0"] == row[prompt_len - 24 : prompt_len].tolist()
    first_root = 20_001
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        None,
        (first_root,),
        2,
        30_000,
    )
    assert md._INGESTED_LEN["request-0"] == prompt_len + 1
    assert md._COMMITTED["request-0"][-1] == first_root
    assert cache.calls[-1] == ("add", "request-0", [first_root])

    row[prompt_len] = first_root
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (prompt_len,),
        (prompt_len,),
        (32,),
        (31,),
        (prompt_len + 1,),
        (False,),
        3,
    )
    accepted = [21_111, 21_003]
    next_root = 20_002
    record = _accepted_record(
        3,
        ("request-0",),
        {"request-0": accepted + [next_root]},
    )
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        record,
        (next_root,),
        3,
        30_000,
    )
    assert cache.calls[-1] == (
        "add",
        "request-0",
        accepted + [next_root],
    )
    assert md._INGESTED_LEN["request-0"] == prompt_len + 4
    assert md._COMMITTED["request-0"][-4:] == [
        first_root,
        *accepted,
        next_root,
    ]

    calls_before_async_stage = list(cache.calls)
    row[prompt_len + 1 : prompt_len + 32] = -1
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (prompt_len,),
        (prompt_len + 32,),
        (32,),
        (31,),
        (prompt_len + 2,),
        (False,),
        4,
    )
    assert cache.calls == calls_before_async_stage
    assert md._FIXED32_PENDING_STEP["rows"][0]["safe_end"] == prompt_len + 4
    assert md._COMMITTED["request-0"][-4:] == [
        first_root,
        *accepted,
        next_root,
    ]
    final_root = 20_003
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        _accepted_record(
            4,
            ("request-0",),
            {"request-0": [final_root]},
        ),
        (final_root,),
        4,
        30_000,
    )
    assert md._INGESTED_LEN["request-0"] == prompt_len + 5
    assert md._COMMITTED["request-0"][-1] == final_root


def test_b4_mixed_rows_use_exact_request_owners() -> None:
    md = _module()
    cache = RecordingCache()
    prompts = {
        "a": list(range(100, 150)),
        "b": list(range(200, 250)),
        "c": list(range(300, 350)),
        "old": list(range(400, 450)),
    }
    md.note_new_requests(cache, prompts, strict=True)
    cache.add_active_response("a", [1_001])
    cache.add_active_response("c", [1_003])
    md._INGESTED_LEN["a"] = 51
    md._INGESTED_LEN["c"] = 51
    md._COMMITTED["a"] = prompts["a"][-23:] + [1_001]
    md._COMMITTED["c"] = prompts["c"][-23:] + [1_003]

    request_ids = ("b", "a", "d", "c")
    rows = np.zeros((4, 96), dtype=np.int64)
    rows[0, :50] = prompts["b"]
    rows[1, :50] = prompts["a"]
    rows[1, 50] = 1_001
    rows[2, :40] = np.arange(500, 540)
    rows[3, :50] = prompts["c"]
    rows[3, 50] = 1_003
    md.stage_fixed32_step(
        cache,
        request_ids,
        rows,
        (50, 50, 40, 50),
        (10, 81, 0, 50),
        (8, 32, 40, 32),
        (0, 31, 0, 31),
        (50, 51, 40, 51),
        (True, False, False, False),
        7,
    )
    assert "old" not in cache.active_requests
    assert "old" not in md._INGESTED_LEN
    assert md._COMMITTED["b"] == rows[0, :18].tolist()
    assert md._COMMITTED["a"][-1] == 1_001
    assert md._COMMITTED["d"] == rows[2, 16:40].tolist()
    assert md._COMMITTED["c"][-1] == 1_003

    a_output = [1_101, 1_102, 1_201]
    c_output = [1_301, 1_401]
    record = _accepted_record(
        7,
        request_ids,
        {"a": a_output, "c": c_output},
    )
    md.finalize_fixed32_step(
        cache,
        request_ids,
        record,
        (9_999, a_output[-1], 1_202, c_output[-1]),
        7,
        30_000,
    )
    adds = [call for call in cache.calls if call[0] == "add"]
    assert adds[-3:] == [
        ("add", "a", [*a_output[:-1], a_output[-1]]),
        ("add", "d", [1_202]),
        ("add", "c", [*c_output[:-1], c_output[-1]]),
    ]
    assert md._INGESTED_LEN["b"] == 50
    assert md._INGESTED_LEN["a"] == 54
    assert md._INGESTED_LEN["d"] == 41
    assert md._INGESTED_LEN["c"] == 53


@pytest.mark.parametrize(
    ("accepted_drafts", "expected_boundary"),
    ((0, 12), (31, 43)),
)
def test_async_spec_stage_uses_owned_arctic_watermark(
    accepted_drafts: int,
    expected_boundary: int,
) -> None:
    md = _module()
    cache = RecordingCache()
    prompt = list(range(10))
    row = np.full(96, -1, dtype=np.int64)
    row[: len(prompt)] = prompt

    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (10,),
        (9,),
        (1,),
        (0,),
        (10,),
        (False,),
        1,
    )
    first_root = 100
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        None,
        (first_root,),
        1,
        1_000,
    )
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (10,),
        (10,),
        (32,),
        (31,),
        (11,),
        (False,),
        2,
    )
    accepted = list(range(200, 200 + accepted_drafts))
    bonus = 500
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        _accepted_record(
            2,
            ("request-0",),
            {"request-0": accepted + [bonus]},
        ),
        (bonus,),
        2,
        1_000,
    )
    assert md._INGESTED_LEN["request-0"] == expected_boundary
    committed_before = list(md._COMMITTED["request-0"])
    calls_before = list(cache.calls)

    md.stage_fixed32_step(
        cache,
        ("request-0",),
        np.asarray([row]),
        (10,),
        (42,),
        (32,),
        (31,),
        (12,),
        (False,),
        3,
    )

    assert md._FIXED32_PENDING_STEP["rows"][0]["safe_end"] == expected_boundary
    assert md._COMMITTED["request-0"] == committed_before
    assert cache.calls == calls_before
    if accepted_drafts == 31:
        accepted_again = list(range(600, 631))
        bonus_again = 700
        md.finalize_fixed32_step(
            cache,
            ("request-0",),
            _accepted_record(
                3,
                ("request-0",),
                {"request-0": accepted_again + [bonus_again]},
            ),
            (bonus_again,),
            3,
            1_000,
        )
        assert md._INGESTED_LEN["request-0"] == 75
        calls_before_repeated_stage = list(cache.calls)
        md.stage_fixed32_step(
            cache,
            ("request-0",),
            np.asarray([row]),
            (10,),
            (74,),
            (32,),
            (31,),
            (13,),
            (False,),
            4,
        )
        assert md._FIXED32_PENDING_STEP["rows"][0]["safe_end"] == 75
        assert cache.calls == calls_before_repeated_stage


@pytest.mark.parametrize("computed", (10, 43))
def test_async_spec_stage_rejects_watermark_outside_correctable_range(
    computed: int,
) -> None:
    md = _module()
    cache = RecordingCache()
    prompt = list(range(10))
    row = np.zeros(96, dtype=np.int64)
    md.note_new_requests(cache, {"request-0": prompt}, strict=True)
    cache.add_active_response("request-0", [100, 101])
    md._INGESTED_LEN["request-0"] = 12
    md._COMMITTED["request-0"] = prompt + [100, 101]
    calls_before = list(cache.calls)

    with pytest.raises(RuntimeError, match="sequence geometry drift"):
        md.stage_fixed32_step(
            cache,
            ("request-0",),
            np.asarray([row]),
            (10,),
            (computed,),
            (32,),
            (31,),
            (12,),
            (False,),
            1,
        )

    assert cache.calls == calls_before
    assert md._FIXED32_PENDING_STEP is None


def test_strict_cache_failures_do_not_advance_python_bookkeeping() -> None:
    md = _module()
    cache = RecordingCache()
    md.note_new_requests(cache, {"request-0": [1, 2]}, strict=True)
    md.ingest_from_sequence(
        cache,
        "request-0",
        [1, 2],
        2,
        strict=True,
    )
    before = (
        md._INGESTED_LEN["request-0"],
        list(md._COMMITTED["request-0"]),
        md.STATS["ingested"],
    )
    cache.fail_add = True
    with pytest.raises(RuntimeError, match="injected add failure"):
        md.ingest_from_sequence(
            cache,
            "request-0",
            [1, 2, 3],
            3,
            strict=True,
        )
    assert (
        md._INGESTED_LEN["request-0"],
        md._COMMITTED["request-0"],
        md.STATS["ingested"],
    ) == before
    assert md._FIXED32_LIFECYCLE_POISON == {
        "phase": "ingest_from_sequence",
        "step_seq": None,
    }
    with pytest.raises(RuntimeError, match="lifecycle is poisoned"):
        md.ingest_from_sequence(
            cache,
            "request-0",
            [1, 2, 3],
            3,
            strict=True,
        )

    md = _module()
    failing_start = RecordingCache()
    failing_start.fail_start = True
    with pytest.raises(RuntimeError, match="injected start failure"):
        md.note_new_requests(
            failing_start,
            {"request-1": [4, 5]},
            strict=True,
        )
    assert "request-1" not in md._INGESTED_LEN
    assert md._FIXED32_LIFECYCLE_POISON == {
        "phase": "note_new_requests",
        "step_seq": None,
    }


def test_legacy_add_failure_keeps_historical_best_effort_behavior() -> None:
    md = _module()
    cache = RecordingCache()
    md.note_new_requests(cache, {"request-0": [1, 2]})
    cache.fail_add = True
    md.ingest_from_sequence(cache, "request-0", [1, 2, 3], 3)
    assert md._INGESTED_LEN["request-0"] == 3
    assert md._COMMITTED["request-0"] == [1, 2, 3]


def _stage_two_final_prefill_rows(md, cache, step_seq=1) -> None:
    rows = np.asarray(
        [
            [101, 0, 0, 0],
            [201, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    md.stage_fixed32_step(
        cache,
        ("a", "b"),
        rows,
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
        (1, 1),
        (False, False),
        step_seq,
    )


def test_b4_finalization_preflights_all_rows_before_any_write() -> None:
    md = _module()
    cache = RecordingCache()
    _stage_two_final_prefill_rows(md, cache)
    before_calls = list(cache.calls)
    before_state = (
        dict(md._INGESTED_LEN),
        {req_id: list(tokens) for req_id, tokens in md._COMMITTED.items()},
    )

    with pytest.raises(RuntimeError, match="outside the vocabulary"):
        md.finalize_fixed32_step(
            cache,
            ("a", "b"),
            None,
            (301, 999),
            1,
            500,
        )

    assert cache.calls == before_calls
    assert (
        md._INGESTED_LEN,
        md._COMMITTED,
    ) == before_state
    assert md._FIXED32_LIFECYCLE_POISON is None


def test_b4_late_native_failure_poison_latches_without_python_advance() -> None:
    md = _module()
    cache = RecordingCache()
    _stage_two_final_prefill_rows(md, cache)
    before_state = (
        dict(md._INGESTED_LEN),
        {req_id: list(tokens) for req_id, tokens in md._COMMITTED.items()},
    )
    cache.fail_add_req = "b"

    with pytest.raises(RuntimeError, match="injected add failure"):
        md.finalize_fixed32_step(
            cache,
            ("a", "b"),
            None,
            (301, 302),
            1,
            500,
        )

    assert ("add", "a", [301]) in cache.calls
    assert (
        md._INGESTED_LEN,
        md._COMMITTED,
    ) == before_state
    assert md._FIXED32_LIFECYCLE_POISON == {
        "phase": "finalize_fixed32_step",
        "step_seq": 1,
    }
    with pytest.raises(RuntimeError, match="lifecycle is poisoned"):
        md.finalize_fixed32_step(
            cache,
            ("a", "b"),
            None,
            (301, 302),
            1,
            500,
        )


def test_b4_staging_preflights_geometry_before_any_native_write() -> None:
    md = _module()
    cache = RecordingCache()
    rows = np.asarray([[101, 0], [201, 0]], dtype=np.int64)

    with pytest.raises(RuntimeError, match="sequence geometry drift"):
        md.stage_fixed32_step(
            cache,
            ("a", "b"),
            rows,
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 1),
            (False, False),
            1,
        )

    assert cache.calls == []
    assert md._INGESTED_LEN == {}
    assert md._COMMITTED == {}
    assert md._FIXED32_LIFECYCLE_POISON is None


def test_b4_late_start_failure_poison_latches_without_python_state() -> None:
    md = _module()
    cache = RecordingCache()
    cache.fail_start_req = "b"
    rows = np.asarray([[101, 0], [201, 0]], dtype=np.int64)

    with pytest.raises(RuntimeError, match="injected start failure"):
        md.stage_fixed32_step(
            cache,
            ("a", "b"),
            rows,
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 0),
            (1, 1),
            (False, False),
            1,
        )

    assert ("start", "a", [101]) in cache.calls
    assert md._INGESTED_LEN == {}
    assert md._COMMITTED == {}
    assert md._FIXED32_LIFECYCLE_POISON == {
        "phase": "stage_fixed32_step",
        "step_seq": 1,
    }


def test_strict_owner_adoption_rejects_stale_local_state() -> None:
    md = _module()
    active = RecordingCache()
    active.active_requests.add("active")
    active.cached_requests.add("active")
    with pytest.raises(RuntimeError, match="consistent owner"):
        md.note_new_requests(active, {"active": [1, 2]}, strict=True)

    inactive = RecordingCache()
    md._INGESTED_LEN["inactive"] = 9
    with pytest.raises(RuntimeError, match="retained local state"):
        md.note_new_requests(inactive, {"inactive": [1, 2]}, strict=True)


def test_forced_resume_restarts_before_a_regressed_safe_boundary() -> None:
    md = _module()
    cache = RecordingCache()
    row = np.asarray([[10, 11, 20]], dtype=np.int64)
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        row,
        (2,),
        (0,),
        (2,),
        (0,),
        (2,),
        (False,),
        1,
    )
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        None,
        (20,),
        1,
        100,
    )
    assert md._INGESTED_LEN["request-0"] == 3

    md.stage_fixed32_step(
        cache,
        ("request-0",),
        row,
        (2,),
        (0,),
        (1,),
        (0,),
        (2,),
        (True,),
        2,
        restart_request_ids={"request-0"},
    )

    assert cache.calls[-3:] == [
        ("stop", "request-0"),
        ("evict", "request-0"),
        ("start", "request-0", [10, 11]),
    ]
    assert cache.trees["request-0"] == [10, 11]
    assert md._INGESTED_LEN["request-0"] == 2
    assert md._COMMITTED["request-0"] == [10]


def test_same_id_replacement_rebuilds_the_native_tree_from_new_prompt() -> None:
    md = _module()
    cache = RecordingCache()
    old_row = np.asarray([[10, 20, 0]], dtype=np.int64)
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        old_row,
        (1,),
        (0,),
        (1,),
        (0,),
        (1,),
        (False,),
        1,
    )
    md.finalize_fixed32_step(
        cache,
        ("request-0",),
        None,
        (20,),
        1,
        100,
    )
    assert cache.trees["request-0"] == [10, 20]

    replacement = np.asarray([[91, 92, 93]], dtype=np.int64)
    md.stage_fixed32_step(
        cache,
        ("request-0",),
        replacement,
        (2,),
        (2,),
        (1,),
        (0,),
        (3,),
        (False,),
        2,
        restart_request_ids={"request-0"},
    )

    assert cache.calls[-4:] == [
        ("stop", "request-0"),
        ("evict", "request-0"),
        ("start", "request-0", [91, 92]),
        ("add", "request-0", [93]),
    ]
    assert cache.trees["request-0"] == [91, 92, 93]
    assert md._INGESTED_LEN["request-0"] == 3
    assert md._COMMITTED["request-0"] == [91, 92, 93]
