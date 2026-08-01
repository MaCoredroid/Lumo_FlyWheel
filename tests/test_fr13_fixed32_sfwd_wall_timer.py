from __future__ import annotations

import ast
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


PATCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fr10_phase4_patch_vllm_tree_gdn.py"
)
FORMAL_SEQUENCE = (
    Path(__file__).resolve().parents[1] / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
)
LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fr13_launch_forked_fa2_tree_server.sh"
)
FLOOR_GATE = Path(__file__).resolve().parents[1] / "scripts" / "fr13_floor_gate.py"


def _load_floor_gate() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fr13_floor_gate_timer_contract",
        FLOOR_GATE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timer_namespace() -> dict[str, object]:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    patch_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_gpu_model_runner_sfwd_gpu_timer"
    )
    module_source = next(
        ast.literal_eval(node.value)
        for node in ast.walk(patch_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "module_block"
            for target in node.targets
        )
    )
    namespace: dict[str, object] = {}
    exec(compile(module_source, "<fr13-sfwd-timer>", "exec"), namespace)
    return namespace


def _timer_class() -> type[Any]:
    return _timer_namespace()["_Fr13SfwdGpuTimer"]  # type: ignore[return-value]


def _fixed_flush_source() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    sources = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fixed_flush"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(sources) == 1
    return sources[0]


def _new_timer(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("FR13_SFWD_GPU_TIMER", "0")
    monkeypatch.setenv("FR13_STEP_WALL_CAP_S", "1.5")
    monkeypatch.setenv("FR13_TORCH_PROF", "")
    return _timer_class()()


def _mark_at(
    monkeypatch: pytest.MonkeyPatch,
    timer: Any,
    ticks: tuple[float, ...],
    marks: tuple[tuple[int, tuple[str, ...]], ...],
) -> None:
    tick_iter = iter(ticks)
    with monkeypatch.context() as clock:
        clock.setattr(time, "perf_counter", lambda: next(tick_iter))
        for num_reqs, request_ids in marks:
            fwd_index = timer._fwd_next_index
            timer._fwd_next_index += 1
            timer._fwd_started += 1
            timer.wall_mark(
                num_reqs=num_reqs,
                request_ids=request_ids,
                fwd_index=fwd_index,
            )


def test_same_request_continues_wall_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0, 10.1),
        ((1, ("req-a",)), (1, ("req-a",))),
    )

    assert timer._wall_steps == 1
    assert timer._wall_drafts == 1
    assert timer._wall_accum_s == pytest.approx(0.1)
    assert timer._wall_rejected == 0
    assert timer._wall_samples_fwd_i == [0]


def test_changed_request_resets_without_rejecting_idle_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0, 20.0, 20.2),
        (
            (1, ("req-a",)),
            (1, ("req-b",)),
            (1, ("req-b",)),
        ),
    )

    assert timer._wall_steps == 1
    assert timer._wall_drafts == 1
    assert timer._wall_accum_s == pytest.approx(0.2)
    assert timer._wall_rejected == 0
    assert timer._wall_chain_resets == 1
    assert timer._wall_request_set_resets == 1
    assert timer._wall_samples_fwd_i == [1]


def test_same_request_gap_above_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0, 20.0),
        ((1, ("req-a",)), (1, ("req-a",))),
    )

    assert timer._wall_steps == 0
    assert timer._wall_drafts == 0
    assert timer._wall_accum_s == 0.0
    assert timer._wall_rejected == 1
    assert timer._wall_samples_fwd_i == []


def test_reordered_b4_request_set_continues_wall_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0, 10.25),
        (
            (4, ("req-d", "req-a", "req-c", "req-b")),
            (4, ("req-b", "req-d", "req-a", "req-c")),
        ),
    )

    assert timer._wall_steps == 1
    assert timer._wall_drafts == 4
    assert timer._wall_accum_s == pytest.approx(0.25)
    assert timer._wall_rejected == 0
    assert timer._wall_samples_fwd_i == [0]


def test_changed_b4_membership_resets_without_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0, 20.0, 20.2),
        (
            (4, ("req-a", "req-b", "req-c", "req-d")),
            (4, ("req-a", "req-b", "req-c", "req-e")),
            (4, ("req-e", "req-c", "req-b", "req-a")),
        ),
    )

    assert timer._wall_steps == 1
    assert timer._wall_drafts == 4
    assert timer._wall_accum_s == pytest.approx(0.2)
    assert timer._wall_rejected == 0
    assert timer._wall_chain_resets == 1
    assert timer._wall_request_set_resets == 1
    assert timer._wall_samples_fwd_i == [1]


def test_wall_break_clears_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)

    _mark_at(
        monkeypatch,
        timer,
        (10.0,),
        ((1, ("req-a",)),),
    )
    timer.wall_break()

    assert timer._wall_prev_t is None
    assert timer._wall_prev_n == 0
    assert timer._wall_prev_req_ids is None
    assert timer._wall_prev_fwd_index is None
    assert timer._wall_chain_resets == 1
    assert timer._wall_request_set_resets == 0

    _mark_at(
        monkeypatch,
        timer,
        (20.0, 20.1),
        ((1, ("req-a",)), (1, ("req-a",))),
    )

    assert timer._wall_steps == 1
    assert timer._wall_rejected == 0
    assert timer._wall_samples_fwd_i == [1]


@pytest.mark.parametrize(
    ("num_reqs", "request_ids"),
    (
        (2, ("req-a",)),
        (2, ("req-a", "req-a")),
        (1, ("",)),
        (1, ("   ",)),
        (1, (123,)),
        (True, ("req-a",)),
    ),
)
def test_malformed_request_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    num_reqs: Any,
    request_ids: tuple[Any, ...],
) -> None:
    timer = _new_timer(monkeypatch)
    _mark_at(
        monkeypatch,
        timer,
        (10.0,),
        ((1, ("req-a",)),),
    )
    with monkeypatch.context() as clock:
        clock.setattr(time, "perf_counter", lambda: 10.1)
        fwd_index = timer._fwd_next_index
        timer._fwd_next_index += 1
        timer._fwd_started += 1
        accepted = timer.wall_mark(
            num_reqs=num_reqs,
            request_ids=request_ids,
            fwd_index=fwd_index,
        )

    assert accepted is False
    assert timer._wall_invalid_request_ids == 1
    assert timer._wall_steps == 0
    assert timer._wall_samples_fwd_i == []
    assert timer._wall_prev_t is None
    assert timer._wall_prev_req_ids is None
    assert timer._wall_prev_fwd_index is None
    assert timer._wall_chain_resets == 1


def test_bad_forward_index_is_a_bookkeeping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timer = _new_timer(monkeypatch)
    _mark_at(
        monkeypatch,
        timer,
        (10.0,),
        ((1, ("req-a",)),),
    )
    with monkeypatch.context() as clock:
        clock.setattr(time, "perf_counter", lambda: 10.1)
        timer._fwd_started += 1
        accepted = timer.wall_mark(
            num_reqs=1,
            request_ids=("req-a",),
            fwd_index=999,
        )

    assert accepted is False
    assert timer._wall_bookkeeping_errors == 1
    assert timer._wall_prev_t is None
    assert timer._wall_chain_resets == 1


def test_nonpositive_sfwd_period_disables_live_rewrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FR13_SFWD_GPU_TIMER", "0")
    monkeypatch.setenv("FR13_SFWD_GPU_TIMER_DUMP_S", "0")
    timer = _timer_class()()
    dumps: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    timer._dump_json = lambda *args, **kwargs: dumps.append((args, kwargs))

    with monkeypatch.context() as clock:
        clock.setattr(
            time,
            "monotonic",
            lambda: pytest.fail("disabled live writer read the clock"),
        )
        timer._maybe_dump_json()

    assert dumps == []


def test_nonpositive_span_period_disables_live_rewrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FR13_TEST_SPAN_TIMER", "0")
    monkeypatch.setenv("FR13_SPAN_GPU_TIMER_DUMP_S", "-1")
    timer_class = _timer_namespace()["_Fr13SpanTimer"]
    timer = timer_class(
        "FR13_TEST_SPAN_TIMER",
        "fr13_test_span_seconds",
        "FR13_TEST_SPAN_JSON",
        "test",
    )
    dumps: list[None] = []
    timer._dump = lambda: dumps.append(None)

    with monkeypatch.context() as clock:
        clock.setattr(
            time,
            "monotonic",
            lambda: pytest.fail("disabled live writer read the clock"),
        )
        timer._maybe_dump()

    assert dumps == []


def test_formal_sequence_disables_periodic_timer_sidecars(tmp_path: Path) -> None:
    source = FORMAL_SEQUENCE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    floor_gate = _load_floor_gate()
    task_ids = list(floor_gate.EVIDENCE_SETS[4]["task_ids"])
    required_env = floor_gate.fixed32_required_env(
        tmp_path,
        mode="tail6_fixed32",
        task_ids=task_ids,
    )

    assert "export FR13_SFWD_GPU_TIMER_DUMP_S=0\n" in source
    assert "export FR13_SPAN_GPU_TIMER_DUMP_S=0\n" in source
    assert "export FR13_TIMER_EXPLICIT_FLUSH=1\n" in source
    assert (
        '"FR13_SFWD_GPU_TIMER_DUMP_S|${FR13_SFWD_GPU_TIMER_DUMP_S:-}|0"'
    ) in launcher
    assert (
        '"FR13_SPAN_GPU_TIMER_DUMP_S|${FR13_SPAN_GPU_TIMER_DUMP_S:-}|0"'
    ) in launcher
    assert required_env["FR13_SFWD_GPU_TIMER_DUMP_S"] == "0"
    assert required_env["FR13_SPAN_GPU_TIMER_DUMP_S"] == "0"


def test_flush_boundaries_break_wall_chain_after_snapshot() -> None:
    source = _fixed_flush_source()
    tree = ast.parse(source)
    break_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_f32_flush_break_wall_chain"
    )
    calls: list[str] = []
    namespace = {
        "_FR13_SFWD_GPU_TIMER": SimpleNamespace(
            wall_break=lambda: calls.append("wall_break")
        )
    }
    exec(
        compile(
            ast.Module(body=[break_definition], type_ignores=[]),
            "<fixed32-flush-break>",
            "exec",
        ),
        namespace,
    )
    namespace["_fr13_f32_flush_break_wall_chain"]()
    assert calls == ["wall_break"]

    flush_one = ast.get_source_segment(
        source,
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_fr13_f32_flush_one"
        ),
    )
    assert flush_one is not None
    boundary = flush_one.index("_fr13_f32_flush_write_boundary(request, counters)")
    wall_break = flush_one.index("_fr13_f32_flush_break_wall_chain()")
    sidecar_dump = flush_one.index("sfwd._dump_json")
    assert boundary < wall_break < sidecar_dump


@pytest.mark.parametrize(
    ("action", "is_final"),
    (("snapshot", False), ("final", True)),
)
def test_explicit_flush_emits_complete_timer_sidecars(
    action: str,
    is_final: bool,
) -> None:
    source = _fixed_flush_source()
    tree = ast.parse(source)
    flush_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fr13_f32_flush_one"
    )
    calls: list[object] = []
    events = [{"event_index": 0}]

    class _Timer:
        def __init__(self, name: str) -> None:
            self.name = name

        def _drain(self, blocking: bool) -> None:
            calls.append((self.name, "drain", blocking))

        def _dump_json(self, *, final: bool, with_samples: bool) -> None:
            calls.append((self.name, "dump_json", final, with_samples))

        def _dump(self) -> None:
            calls.append((self.name, "dump"))

    sfwd = _Timer("sfwd")
    dfwd = _Timer("dfwd")
    cfwd = _Timer("cfwd")
    counters = {"steps": 7}
    request = {"action": action, "generation": 4, "nonce": "nonce-4"}
    sample_condition = threading.Condition(threading.RLock())
    namespace: dict[str, object] = {
        "_FR13_FIXED32_FLUSH_GENERATION": 3,
        "_FR13_FIXED32_FLUSH_QUIESCING": False,
        "_FR13_FIXED32_FLUSH_TERMINAL": False,
        "_FR13_FIXED32_SAMPLE_COND": sample_condition,
        "_FR13_FIXED32_SAMPLE_PENDING": {},
        "_FR13_FIXED32_SAMPLE_FAILURE": None,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: calls.append("synchronize"))
        ),
        "_fr13_f32_flush_runtime_state": lambda: (
            None,
            events,
            7,
            None,
            None,
            sfwd,
            dfwd,
            cfwd,
        ),
        "_fr13_f32_flush_reconcile": lambda: calls.append("reconcile"),
        "_fr13_f32_flush_counters": lambda *, require_drained: (
            calls.append(("counters", require_drained)) or counters
        ),
        "_fr13_f32_flush_write_boundary": lambda req, values: calls.append(
            ("boundary", req, values)
        ),
        "_fr13_f32_flush_break_wall_chain": lambda: calls.append("wall_break"),
        "_fr13_f32_flush_write_census": (
            lambda values, *, final: calls.append(("census", values, final))
        ),
        "_fr13_f32_flush_write_ack": lambda **kwargs: calls.append(("ack", kwargs)),
    }
    exec(
        compile(
            ast.Module(body=[flush_definition], type_ignores=[]),
            "<fixed32-flush-one>",
            "exec",
        ),
        namespace,
    )

    namespace["_fr13_f32_flush_one"](request)

    assert calls == [
        "synchronize",
        ("sfwd", "drain", False),
        ("dfwd", "drain", False),
        ("cfwd", "drain", False),
        "reconcile",
        ("counters", True),
        ("boundary", request, counters),
        "wall_break",
        ("sfwd", "dump_json", is_final, True),
        ("dfwd", "dump"),
        ("cfwd", "dump"),
        ("census", events, is_final),
        (
            "ack",
            {
                "generation": 4,
                "nonce": "nonce-4",
                "action": action,
                "status": "ok",
                "counters": counters,
            },
        ),
    ]
    assert namespace["_FR13_FIXED32_FLUSH_GENERATION"] == 4
    assert namespace["_FR13_FIXED32_FLUSH_QUIESCING"] is False
    assert namespace["_FR13_FIXED32_FLUSH_TERMINAL"] is is_final


def test_flush_fails_closed_on_prior_sample_failure() -> None:
    source = _fixed_flush_source()
    tree = ast.parse(source)
    flush_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_fr13_f32_flush_one"
    )
    cause = ValueError("sample failed")
    calls: list[str] = []
    namespace: dict[str, object] = {
        "_FR13_FIXED32_FLUSH_GENERATION": 3,
        "_FR13_FIXED32_FLUSH_QUIESCING": False,
        "_FR13_FIXED32_FLUSH_TERMINAL": False,
        "_FR13_FIXED32_SAMPLE_COND": threading.Condition(threading.RLock()),
        "_FR13_FIXED32_SAMPLE_PENDING": {},
        "_FR13_FIXED32_SAMPLE_FAILURE": ("sample failed", cause),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: calls.append("synchronize"))
        ),
        "_fr13_f32_flush_write_boundary": (
            lambda _request, _counters: calls.append("boundary")
        ),
        "_fr13_f32_flush_write_ack": lambda **_kwargs: calls.append("ack"),
    }
    exec(
        compile(
            ast.Module(body=[flush_definition], type_ignores=[]),
            "<fixed32-flush-one>",
            "exec",
        ),
        namespace,
    )

    with pytest.raises(
        RuntimeError,
        match="prior sample failed before flush: sample failed",
    ) as flush_error:
        namespace["_fr13_f32_flush_one"](
            {
                "action": "snapshot",
                "generation": 4,
                "nonce": "nonce-4",
            }
        )

    assert flush_error.value.__cause__ is cause
    assert calls == []
    assert namespace["_FR13_FIXED32_FLUSH_GENERATION"] == 3
    assert namespace["_FR13_FIXED32_FLUSH_QUIESCING"] is False


def test_flush_reconcile_still_rejects_a_stranded_proposal() -> None:
    source = _fixed_flush_source()
    tree = ast.parse(source)
    reconcile_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_f32_flush_reconcile"
    )
    gdn = SimpleNamespace(_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT=object())
    namespace: dict[str, object] = {
        "_fr13_f32_flush_runtime_state": lambda: (
            gdn,
            [],
            0,
            None,
            None,
            None,
            None,
            None,
        )
    }
    exec(
        compile(
            ast.Module(body=[reconcile_definition], type_ignores=[]),
            "<fixed32-flush-reconcile>",
            "exec",
        ),
        namespace,
    )

    with pytest.raises(
        RuntimeError,
        match="fixed32 flush saw an incomplete drafter proposal",
    ):
        namespace["_fr13_f32_flush_reconcile"]()
