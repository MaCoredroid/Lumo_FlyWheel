"""FR13_HOST_TAIL_NVTX / FR13_HOST_TAIL_DEFER: post-DFWD tail.

vLLM is not installed on this host, so the patch is asserted against the
patcher's own source (the house convention -- see
tests/test_fr13_fixed32_nvtx_profile.py). The retire queue, which is real
concurrency, is AST-extracted from the injected module block and exercised
directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

PATCHER = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
REDUCER = Path("scripts/fr13_fixed32_nsys_reduce.py")
LAUNCHER = Path("scripts/fr13_launch_forked_fa2_tree_server.sh")


def _patcher_text() -> str:
    return PATCHER.read_text()


def _fn_node(name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(_patcher_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {PATCHER}")


def _module_block(*, nvtx: bool = False, defer: bool = False) -> str:
    """The source string this patch appends to gpu_model_runner.py.

    The block is an f-string: the host-residual rung bakes the resolved flags
    into it as literals at patch time, because the mp/spawn EngineCore worker
    gets a curated env that drops bare FR13_* masters. Evaluating it here with
    a stub ``flags`` mapping is what lets the test see both armings.
    """
    for node in ast.walk(_fn_node("_patch_gpu_model_runner_host_tail")):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "module_block"
                for t in node.targets
            )
        ):
            expression = ast.Expression(body=node.value)
            ast.fix_missing_locations(expression)
            return eval(  # noqa: S307 - patcher-owned literal, not user input
                compile(expression, "<fr13-host-tail-block>", "eval"),
                {"flags": {"nvtx": nvtx, "defer": defer}},
            )
    raise AssertionError("module_block assignment not found")


# --------------------------------------------------------------- flags


def test_both_flags_are_opt_in() -> None:
    """OFF is the default, and the value is BAKED, never read in the worker.

    The first cut of this patch read os.environ inside the served process.
    That process is the mp/spawn EngineCore worker, whose curated env drops
    bare FR13_* masters, so an arm could report as armed at pid 1 and run as
    stock where it mattered. The host-residual rung moved the read to patch
    time; see tests/test_fr13_host_tail_prep_bake.py for the strict-parsing
    and fixed32 preconditions that now guard it.
    """
    block = _module_block()
    assert "_FR13_HOST_TAIL_NVTX = False" in block
    assert "_FR13_HOST_TAIL_DEFER = False" in block
    armed = _module_block(nvtx=True, defer=True)
    assert "_FR13_HOST_TAIL_NVTX = True" in armed
    assert "_FR13_HOST_TAIL_DEFER = True" in armed
    for text in (block, armed):
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )
        assert "environ" not in code, "the worker must not read the flag env"
        assert 'get("FR13_HOST_TAIL' not in code


def test_flags_reach_the_server() -> None:
    launcher = LAUNCHER.read_text()
    assert '-e FR13_HOST_TAIL_NVTX="${FR13_HOST_TAIL_NVTX:-0}"' in launcher
    assert '-e FR13_HOST_TAIL_DEFER="${FR13_HOST_TAIL_DEFER:-0}"' in launcher
    assert (
        '-e FR13_HOST_TAIL_PREP_BAKE="${FR13_HOST_TAIL_PREP_BAKE:-0}"'
        in launcher
    )


# --------------------------------------------------------------- anchors


def test_patch_is_count_asserted_and_self_diagnosing() -> None:
    src = ast.unparse(_fn_node("_patch_gpu_model_runner_host_tail"))
    assert "text.count(readback_anchor) != 1" in src
    assert "text.count(seal_anchor) != 1" in src
    text = _patcher_text()
    assert "draft-token readback anchor count" in text
    assert "drafter proposal seal anchor count" in text
    # Ordering dependency must name the patch that has to run first.
    assert "_patch_gpu_model_runner_sfwd_gpu_timer " in text
    assert "must run BEFORE this patch." in text


def test_patch_is_idempotent_and_registered_after_its_dependencies() -> None:
    src = ast.unparse(_fn_node("_patch_gpu_model_runner_host_tail"))
    assert "if sentinel in text:" in src and "return False" in src
    text = _patcher_text()
    reqkey = text.index("_patch_gpu_model_runner_replay_draft_reqkey()),")
    host_tail = text.index("_patch_gpu_model_runner_host_tail()),")
    assert reqkey < host_tail, "host tail patch must be registered after reqkey"


def test_defer_off_runs_the_seal_inline_and_in_order() -> None:
    src = ast.unparse(_fn_node("_patch_gpu_model_runner_host_tail"))
    # Previous step's deferred seal is drained before this step submits.
    assert "_fr13_host_tail_drain()" in src
    assert "if _fr13_host_tail_defer_enabled():" in src
    assert "else:" in src
    assert "_fr13_fixed32_drafter_proposal_end(\\n" in src or (
        "_fr13_fixed32_drafter_proposal_end" in src
    )


def test_scheduler_range_is_named_as_the_ladder_specifies() -> None:
    src = ast.unparse(_fn_node("_patch_scheduler_host_tail_nvtx"))
    assert "fr13.fixed32.sched_next" in src
    assert "text.count(anchor) != 1" in src


def test_scheduler_range_is_actually_opened() -> None:
    """Regression: the first cut defined the helper and never called it.

    ``fr13.fixed32.sched_next`` was reserved in the reducer and could not
    appear in any capture -- an unsatisfiable measurement precondition. The
    patch now renames the stock method and wraps it.
    """
    src = ast.unparse(_fn_node("_patch_scheduler_host_tail_nvtx"))
    assert "self._fr13_sched_next_nvtx(True)" in src
    assert "self._fr13_sched_next_nvtx(False)" in src
    assert "_fr13_update_from_output_inner" in src


# --------------------------------------------------------------- reducer


def test_reducer_tolerates_but_does_not_require_the_tail_ranges() -> None:
    """A capture with the arm off has none of these ranges; it must still reduce."""
    text = REDUCER.read_text()
    for name in (
        "fr13.fixed32.sample_readback",
        "fr13.fixed32.output_proc",
        "fr13.fixed32.sched_next",
        "fr13.fixed32.kv_bookkeep",
    ):
        assert name in text
    assert "optional = set(HOST_TAIL_RANGES.values())" in text
    assert "unexpected = sorted(observed - expected - optional)" in text
    # The four phase ranges stay mandatory.
    assert "missing = sorted(expected - observed)" in text


def test_reducer_still_rejects_an_unknown_fixed32_range() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fr13_reduce_t", REDUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [{"range": name} for name in module.ATTRIBUTION_RANGES.values()]
    # Baseline: the four phases plus step reduce cleanly.
    module._require_exact_phase_ranges(
        rows, range_column="Range", report_name="t"
    )
    # Optional tail ranges are tolerated.
    module._require_exact_phase_ranges(
        rows + [{"range": "fr13.fixed32.sched_next"}],
        range_column="Range",
        report_name="t",
    )
    # Anything else under the fixed32 prefix is still fatal.
    try:
        module._require_exact_phase_ranges(
            rows + [{"range": "fr13.fixed32.bogus"}],
            range_column="Range",
            report_name="t",
        )
    except module.ReductionError as error:
        assert "fr13.fixed32.bogus" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unknown fixed32 range was not rejected")
    # A missing mandatory phase is still fatal.
    try:
        module._require_exact_phase_ranges(
            rows[:-1], range_column="Range", report_name="t"
        )
    except module.ReductionError as error:
        assert "missing=" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing phase range was not rejected")


# --------------------------------------------------------------- retire queue


def _retire_class():
    """Execute the injected module block and hand back the retire queue."""
    namespace: dict = {}
    exec(compile(_module_block(), "<fr13-host-tail>", "exec"), namespace)
    return namespace


def test_retire_queue_runs_work_and_drains() -> None:
    ns = _retire_class()
    queue = ns["_Fr13HostTailRetire"]()
    seen = []
    for i in range(64):
        queue.submit(lambda i=i: seen.append(i))
    queue.drain()
    assert sorted(seen) == list(range(64))
    assert seen == list(range(64)), "retire queue must preserve submission order"


def test_retire_queue_reraises_worker_errors_on_drain() -> None:
    """The census seal's fail-loud contract must survive being deferred."""
    ns = _retire_class()
    queue = ns["_Fr13HostTailRetire"]()

    def boom():
        raise RuntimeError("FR13 fixed32 completed drafter proposal work drift")

    queue.submit(boom)
    try:
        queue.drain()
    except RuntimeError as error:
        assert "drafter proposal work drift" in str(error)
    else:  # pragma: no cover
        raise AssertionError("deferred error was swallowed")
    # The error must not latch: a clean drain afterwards succeeds.
    queue.submit(lambda: None)
    queue.drain()


def test_drain_is_a_noop_before_any_deferral() -> None:
    ns = _retire_class()
    assert ns["_FR13_HOST_TAIL_RETIRE"] is None
    ns["_fr13_host_tail_drain"]()
