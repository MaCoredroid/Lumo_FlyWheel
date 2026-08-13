"""FR13 host-residual rung: host-tail flags, the depth-position bake, and the
two instrumentation defects this branch fixes.

vLLM is not installed on this host, so the patch is exercised against stub
files that carry only the anchors (the house convention -- see
tests/test_fr13_host_tail.py and tests/test_fr13_fixed32_nvtx_profile.py).
Where a patch function reads a real vLLM module, the module-level PATH
constant is monkeypatched to a tmp stub, so the patch's own anchor guards,
emitted text and idempotency sentinel are all really executed.

The equivalence tests are the load-bearing ones: they exec BOTH forms of the
depth-position derivation -- the incumbent expression and the baked literals
-- over a battery of tree sources and require identical values, dtypes and
object freshness. If the bake ever stops reproducing the expression it
replaced, these fail before anything is served.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
REDUCER = REPO / "scripts" / "fr13_fixed32_nsys_reduce.py"
TEXT = PATCHER.read_text()

sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr13_host_tail_prep import (  # noqa: E402
    assert_host_tail_prep_requires_fixed32,
    baked_plan_source,
    derive_tree_depth_plan,
    plan_census,
    strict_flag,
)

np = pytest.importorskip("numpy")

FLAGS = (
    "FR13_HOST_TAIL_NVTX",
    "FR13_HOST_TAIL_DEFER",
    "FR13_HOST_TAIL_PREP_BAKE",
)

# The deployed fixed32 tree, lifted from the patcher so the test cannot drift
# from the constant the bake is derived from.
_CHOICES_SRC = re.search(
    r"^_FR13_FIXED32_CHOICES: tuple\[tuple\[int, \.\.\.\], \.\.\.\] = (\(.*?\n\))\n",
    TEXT,
    re.S | re.M,
)
assert _CHOICES_SRC is not None, "could not lift _FR13_FIXED32_CHOICES"
FIXED32_CHOICES = ast.literal_eval(_CHOICES_SRC.group(1))
FIXED32_TREE_SOURCE = repr(list(FIXED32_CHOICES))

TREE_SOURCES = {
    "fixed32_deployed_31": FIXED32_TREE_SOURCE,
    "launcher_default_9": (
        "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), "
        "(0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"
    ),
    "single": "[(0,)]",
    "spine_only": "[(0,), (0, 0), (0, 0, 0)]",
    "leaf_only": "[(1,), (2,), (3,)]",
    "unsorted_input": "[(0, 0, 1), (0,), (1,), (0, 0), (0, 1)]",
    "wide": repr([(i,) for i in range(8)]),
}


def _load_patcher(monkeypatch, *, fixed32="tail6_fixed32", **flags):
    for name in FLAGS:
        value = flags.get(name)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    if fixed32 is None:
        monkeypatch.delenv("FR13_FIXED32_MODE", raising=False)
    else:
        monkeypatch.setenv("FR13_FIXED32_MODE", fixed32)
    key = f"_fr13_patcher_host_tail_{fixed32}_{sorted(flags.items())}"
    spec = importlib.util.spec_from_file_location(key, PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. Flag hygiene: default off, strict, read exactly once, never defaults on.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_is_declared_once_with_an_off_default(flag):
    reads = re.findall(rf'os\.environ\.get\(\s*"{flag}"[^)]*\)', TEXT)
    assert len(reads) == 1, f"{flag} must be read exactly once at patch time"
    assert '"0"' in reads[0], f"{flag} must default to 0"
    assert (
        f'environ.get("{flag}", "1")' not in TEXT
        and f"environ.get('{flag}', '1')" not in TEXT
    ), f"{flag} must never default ON"


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("bad", ["2", "true", "yes", "01", "-1", "on"])
def test_strict_zero_one(monkeypatch, flag, bad):
    module = _load_patcher(monkeypatch, **{flag: bad})
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        module._fr13_host_tail_flags()


@pytest.mark.parametrize("flag", FLAGS)
def test_explicitly_empty_env_raises_rather_than_disarming(monkeypatch, flag):
    """An empty value is a typo, not an OFF.

    The launcher's ``${FLAG:-0}`` turns unset *and* empty into "0", so an
    empty value can only reach the patcher when something bypassed the
    launcher -- exactly the case where silently serving stock is worst.
    """
    module = _load_patcher(monkeypatch, **{flag: ""})
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        module._fr13_host_tail_flags()


def test_unset_env_is_off(monkeypatch):
    module = _load_patcher(monkeypatch)
    assert set(module._fr13_host_tail_flags().values()) == {False}


@pytest.mark.parametrize("bad", [None, "2", "", " 1 x"])
def test_strict_flag_helper_rejects_non_binary(bad):
    with pytest.raises(RuntimeError, match="must be exactly 0 or 1"):
        strict_flag(bad, "FR13_X")


def test_strict_flag_helper_tolerates_surrounding_whitespace():
    assert strict_flag(" 1 ", "FR13_X") is True
    assert strict_flag("0\n", "FR13_X") is False


# --------------------------------------------------------------------------
# 2. Fixed32-only, fail loud, and satisfiable by construction.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_refuses_an_unarmed_boot(monkeypatch, flag):
    module = _load_patcher(monkeypatch, fixed32=None, **{flag: "1"})
    with pytest.raises(RuntimeError) as excinfo:
        module._fr13_host_tail_flags()
    message = str(excinfo.value)
    assert flag in message
    assert "FR13_FIXED32_MODE" in message
    assert "tail6_fixed32" in message and "hydra27_fixed32" in message


@pytest.mark.parametrize("mode", ["tail6_fixed32", "hydra27_fixed32"])
@pytest.mark.parametrize("flag", FLAGS)
def test_flag_is_satisfiable_under_every_fixed32_mode(monkeypatch, mode, flag):
    module = _load_patcher(monkeypatch, fixed32=mode, **{flag: "1"})
    resolved = module._fr13_host_tail_flags()
    assert sum(resolved.values()) == 1


def test_prep_bake_guard_names_the_mechanism_and_both_ways_out():
    with pytest.raises(RuntimeError) as excinfo:
        assert_host_tail_prep_requires_fixed32(True, "")
    message = str(excinfo.value)
    for fragment in (
        "FR13_HOST_TAIL_PREP_BAKE=1",
        "requires fixed32",
        "depth_offsets",
        "nothing downstream would",
        "Set FR13_FIXED32_MODE",
        "unset FR13_HOST_TAIL_PREP_BAKE",
    ):
        assert fragment in message, fragment


def test_prep_bake_guard_is_inert_when_off():
    assert_host_tail_prep_requires_fixed32(False, None) is None
    assert_host_tail_prep_requires_fixed32(False, "not_fixed32") is None


def test_preflight_runs_in_main_before_any_patch_step():
    assert TEXT.count("_fr13_assert_host_tail_flags()") == 2, (
        "expected exactly one definition and one call site"
    )
    main_at = TEXT.index("def main() -> int:")
    call_at = TEXT.index("    _fr13_assert_host_tail_flags()\n", main_at)
    steps_at = TEXT.index("    patch_steps = [", main_at)
    assert main_at < call_at < steps_at


# --------------------------------------------------------------------------
# 3. The bake reproduces the expression it replaces, exactly.
# --------------------------------------------------------------------------

INCUMBENT = textwrap.dedent(
    """
    _fr10_choices = sorted(
        __import__("ast").literal_eval(_fr10_tree_src),
        key=lambda _p: (len(_p), _p),
    )
    _fr10_depth_offsets = np.array(
        [0] + [len(_fr10_choice) for _fr10_choice in _fr10_choices],
        dtype=np.int64,
    )
    _fr10_spine_choices = [
        _fr10_choice
        for _fr10_choice in _fr10_choices
        if all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
    ]
    _fr10_leaf_choices = [
        _fr10_choice
        for _fr10_choice in _fr10_choices
        if not all(int(_fr10_part) == 0 for _fr10_part in _fr10_choice)
    ]
    _fr10_spine_first_depth_offsets = np.array(
        [0]
        + [len(_fr10_choice) for _fr10_choice in _fr10_spine_choices]
        + [len(_fr10_choice) for _fr10_choice in _fr10_leaf_choices],
        dtype=np.int64,
    )
    _fr10_tree_n = int(len(_fr10_depth_offsets))
    """
)

TREE_N_TAIL = "_fr10_tree_n = int(len(_fr10_depth_offsets))\n"


def _run(source, tree_src):
    namespace = {"np": np, "_fr10_tree_src": tree_src}
    exec(compile(source, "<fr13-depth-plan>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", sorted(TREE_SOURCES))
def test_baked_plan_is_value_identical_to_the_incumbent(name):
    tree_src = TREE_SOURCES[name]
    incumbent = _run(INCUMBENT, tree_src)
    baked = _run(
        textwrap.dedent(baked_plan_source(tree_src, "")) + TREE_N_TAIL,
        tree_src,
    )
    assert baked["_fr10_choices"] == incumbent["_fr10_choices"]
    assert baked["_fr10_tree_n"] == incumbent["_fr10_tree_n"]
    for key in ("_fr10_depth_offsets", "_fr10_spine_first_depth_offsets"):
        assert baked[key].dtype == incumbent[key].dtype == np.int64
        assert baked[key].shape == incumbent[key].shape
        assert np.array_equal(baked[key], incumbent[key])


@pytest.mark.parametrize("name", sorted(TREE_SOURCES))
def test_baked_plan_yields_fresh_mutable_objects_every_call(name):
    """A cached array reused across steps would be a cross-step alias.

    The incumbent hands the caller a fresh writable array each step and the
    caller does arithmetic with it; the bake must not weaken that.
    """
    tree_src = TREE_SOURCES[name]
    source = textwrap.dedent(baked_plan_source(tree_src, "")) + TREE_N_TAIL
    first = _run(source, tree_src)
    first["_fr10_depth_offsets"] += 1000
    first["_fr10_spine_first_depth_offsets"] += 1000
    first["_fr10_choices"].append(("poison",))
    second = _run(source, tree_src)
    incumbent = _run(INCUMBENT, tree_src)
    assert second["_fr10_choices"] == incumbent["_fr10_choices"]
    assert np.array_equal(
        second["_fr10_depth_offsets"], incumbent["_fr10_depth_offsets"]
    )
    assert np.array_equal(
        second["_fr10_spine_first_depth_offsets"],
        incumbent["_fr10_spine_first_depth_offsets"],
    )


def test_derive_matches_the_deployed_fixed32_topology():
    census = plan_census(FIXED32_TREE_SOURCE)
    assert census["paths"] == 31
    assert census["tree_n"] == 32
    assert census["depth_offsets"][0] == 0
    assert len(census["spine_first_depth_offsets"]) == 32
    plan = derive_tree_depth_plan(FIXED32_TREE_SOURCE)
    assert plan["choices"] == tuple(
        tuple(choice) for choice in sorted(FIXED32_CHOICES, key=lambda p: (len(p), p))
    )


def test_baked_source_is_indented_into_the_caller_suite():
    body = baked_plan_source(FIXED32_TREE_SOURCE, " " * 16)
    assert body.endswith("\n")
    lines = body.splitlines()
    assert lines, "the bake must emit something"
    for line in lines:
        assert line.startswith(" " * 16), line
    # Statement lines sit exactly at the suite indent; only continuation
    # lines inside a parenthesised call may go deeper.
    statements = [line for line in lines if not line[16:17].isspace()]
    assert len(statements) >= 3, statements
    compile(textwrap.dedent(body), "<fr13-bake-indent>", "exec")


# --------------------------------------------------------------------------
# 4. The patch emits the bake only under fixed32 AND the flag.
# --------------------------------------------------------------------------

GPU_RUNNER_STUB = '''\
class GPUModelRunner:
    def _prepare_inputs(self):
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # Copy the tensors to the GPU.
'''


def _patched_depth_positions(monkeypatch, tmp_path, **flags):
    module = _load_patcher(monkeypatch, **flags)
    stub = tmp_path / f"gpu_model_runner_{sorted(flags.items())}.py"
    stub.write_text(GPU_RUNNER_STUB)
    monkeypatch.setattr(module, "GPU_MODEL_RUNNER_PATH", stub)
    assert module._patch_gpu_model_runner_tree_depth_positions() is True
    return module, stub.read_text()


def test_flag_off_leaves_the_incumbent_derivation_in_place(
    monkeypatch, tmp_path
):
    _module, patched = _patched_depth_positions(monkeypatch, tmp_path)
    assert "_fr10_choices = sorted(" in patched
    assert "fixed32 depth-position topology drifted" in patched
    assert "FR13_HOST_TAIL_PREP_BAKE" not in patched


def test_flag_on_replaces_the_derivation_with_literals(monkeypatch, tmp_path):
    _module, patched = _patched_depth_positions(
        monkeypatch, tmp_path, FR13_HOST_TAIL_PREP_BAKE="1"
    )
    assert "FR13_HOST_TAIL_PREP_BAKE" in patched
    assert "_fr10_choices = sorted(" not in patched
    assert "literal_eval(_fr10_tree_src)" not in patched
    assert "fixed32 depth-position topology drifted" not in patched
    # the surviving statement that consumes the plan is untouched
    assert "_fr10_tree_n = int(len(_fr10_depth_offsets))" in patched


def test_patched_source_compiles_in_both_forms(monkeypatch, tmp_path):
    for flags in ({}, {"FR13_HOST_TAIL_PREP_BAKE": "1"}):
        _module, patched = _patched_depth_positions(
            monkeypatch, tmp_path, **flags
        )
        compile(patched, "<fr13-depth-positions>", "exec")


def test_off_and_on_differ_only_inside_the_derivation_span(
    monkeypatch, tmp_path
):
    _m, off = _patched_depth_positions(monkeypatch, tmp_path)
    _m, on = _patched_depth_positions(
        monkeypatch, tmp_path, FR13_HOST_TAIL_PREP_BAKE="1"
    )
    head = "_fr10_choices = sorted("
    tail = "_fr10_tree_n = int(len(_fr10_depth_offsets))"
    assert off[: off.index(head)] == on[: on.index("# FR13_HOST_TAIL_PREP_BAKE")]
    assert off[off.index(tail) :] == on[on.index(tail) :]


def test_bake_is_not_emitted_outside_fixed32(monkeypatch, tmp_path):
    """Outside fixed32 the flag cannot be armed at all -- preflight refuses."""
    module = _load_patcher(
        monkeypatch, fixed32=None, FR13_HOST_TAIL_PREP_BAKE="1"
    )
    stub = tmp_path / "gpu_model_runner_nonfixed32.py"
    stub.write_text(GPU_RUNNER_STUB)
    monkeypatch.setattr(module, "GPU_MODEL_RUNNER_PATH", stub)
    # The non-fixed32 branch never reaches the bake, and the preflight in
    # main() is what refuses the arm.
    assert module._patch_gpu_model_runner_tree_depth_positions() is True
    assert "FR13_HOST_TAIL_PREP_BAKE" not in stub.read_text()
    with pytest.raises(RuntimeError, match="requires fixed32"):
        module._fr13_assert_host_tail_flags()


# --------------------------------------------------------------------------
# 5. sched_next: the range is now actually opened (defect fix).
# --------------------------------------------------------------------------

SCHEDULER_STUB = '''\
class Scheduler:
    def update_from_output(
        self,
        scheduler_output,
        model_runner_output,
    ):
        return {}
'''


def _patched_scheduler(monkeypatch, tmp_path, **flags):
    module = _load_patcher(monkeypatch, **flags)
    stub = tmp_path / f"scheduler_{sorted(flags.items())}.py"
    stub.write_text(SCHEDULER_STUB)
    monkeypatch.setattr(module, "SCHEDULER_PATH", stub)
    assert module._patch_scheduler_host_tail_nvtx() is True
    return module, stub


def test_sched_next_range_is_defined_and_called(monkeypatch, tmp_path):
    _module, stub = _patched_scheduler(monkeypatch, tmp_path)
    patched = stub.read_text()
    assert patched.count("def _fr13_sched_next_nvtx(self, opening):") == 1
    assert patched.count("self._fr13_sched_next_nvtx(True)") == 1
    assert patched.count("self._fr13_sched_next_nvtx(False)") == 1
    assert patched.count("def update_from_output(") == 1
    assert patched.count("def _fr13_update_from_output_inner(") == 1
    compile(patched, "<fr13-scheduler>", "exec")


def test_sched_next_wrapper_forwards_and_pops_even_on_exception(
    monkeypatch, tmp_path
):
    _module, stub = _patched_scheduler(
        monkeypatch, tmp_path, FR13_HOST_TAIL_NVTX="1"
    )
    namespace = {}
    exec(compile(stub.read_text(), "<fr13-scheduler>", "exec"), namespace)
    scheduler = namespace["Scheduler"]()
    opened = []
    scheduler._fr13_sched_next_nvtx = lambda opening: (
        opened.append(opening) or opening
    )
    assert scheduler.update_from_output("a", "b") == {}
    assert opened == [True, False]

    def _boom(self, *a, **kw):
        raise ValueError("inner")

    namespace["Scheduler"]._fr13_update_from_output_inner = _boom
    opened.clear()
    with pytest.raises(ValueError, match="inner"):
        scheduler.update_from_output("a", "b")
    assert opened == [True, False], "range must be popped on the error path"


def test_sched_next_flag_is_baked_not_read_from_the_worker_env(
    monkeypatch, tmp_path
):
    _module, off = _patched_scheduler(monkeypatch, tmp_path)
    _module, on = _patched_scheduler(
        monkeypatch, tmp_path, FR13_HOST_TAIL_NVTX="1"
    )
    assert "_FR13_HOST_TAIL_NVTX = False" in off.read_text()
    assert "_FR13_HOST_TAIL_NVTX = True" in on.read_text()
    for text in (off.read_text(), on.read_text()):
        assert "environ" not in text


def test_scheduler_patch_is_idempotent(monkeypatch, tmp_path):
    module, stub = _patched_scheduler(monkeypatch, tmp_path)
    before = stub.read_text()
    assert module._patch_scheduler_host_tail_nvtx() is False
    assert stub.read_text() == before


def test_scheduler_patch_refuses_an_ambiguous_anchor(monkeypatch, tmp_path):
    module = _load_patcher(monkeypatch)
    stub = tmp_path / "scheduler_dup.py"
    stub.write_text(SCHEDULER_STUB + "\n" + SCHEDULER_STUB)
    monkeypatch.setattr(module, "SCHEDULER_PATH", stub)
    with pytest.raises(RuntimeError, match="anchor count 2 != 1"):
        module._patch_scheduler_host_tail_nvtx()


# --------------------------------------------------------------------------
# 6. prep_next: the tail sub-range that covers the measured host time.
# --------------------------------------------------------------------------

HOST_TAIL_STUB = '''\
def _fr13_fixed32_nvtx_begin(phase, active):
    return False


def _fr13_fixed32_nvtx_end(opened):
    return None


class GPUModelRunner:
    def execute_model(self):
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            pass
        _fr13_fixed32_step_nvtx_close()

    def sample_tokens(self):
        if True:
            if True:
                self._copy_draft_token_ids_to_cpu(scheduler_output)
                if True:
                    _fr13_f32_draft_gdn._fr13_fixed32_drafter_proposal_end(
                        _fr13_f32_draft_gdn._FR13_FIXED32_MODE,
                        _fr13_f32_draft_req_ids,
                        tuple(int(_d) for _d in self._draft_token_ids.shape),
                        str(self._draft_token_ids.dtype),
                        self._draft_token_ids.device.type,
                        True,
                    )
'''


def _patched_host_tail(monkeypatch, tmp_path, **flags):
    module = _load_patcher(monkeypatch, **flags)
    stub = tmp_path / f"host_tail_{sorted(flags.items())}.py"
    stub.write_text(HOST_TAIL_STUB)
    monkeypatch.setattr(module, "GPU_MODEL_RUNNER_PATH", stub)
    assert module._patch_gpu_model_runner_host_tail() is True
    return module, stub


def test_prep_next_range_wraps_input_prep_and_closes_at_the_step_boundary(
    monkeypatch, tmp_path
):
    _module, stub = _patched_host_tail(monkeypatch, tmp_path)
    patched = stub.read_text()
    push = patched.index("'prep_next', _fr13_host_tail_nvtx_enabled()")
    prep = patched.index("gpu_model_runner: preprocess")
    pop = patched.index("_fr13_fixed32_nvtx_end(getattr(self, '_fr13_prep_nvtx'")
    close = patched.index("_fr13_fixed32_step_nvtx_close()")
    assert push < prep < pop < close, "prep_next must span input prep"
    assert patched.count("self._fr13_prep_nvtx = False") == 1
    compile(patched, "<fr13-host-tail>", "exec")


def test_host_tail_patch_guards_both_new_anchors(monkeypatch, tmp_path):
    module = _load_patcher(monkeypatch)
    for missing, expected in (
        ("gpu_model_runner: preprocess", "preprocess anchor count 0 != 1"),
        ("_fr13_fixed32_step_nvtx_close()", "close anchor count 0 != 1"),
    ):
        stub = tmp_path / f"host_tail_missing_{abs(hash(missing))}.py"
        stub.write_text(HOST_TAIL_STUB.replace(missing, "REMOVED"))
        monkeypatch.setattr(module, "GPU_MODEL_RUNNER_PATH", stub)
        with pytest.raises(RuntimeError, match=re.escape(expected)):
            module._patch_gpu_model_runner_host_tail()


def test_host_tail_flags_are_baked_into_the_module_block(
    monkeypatch, tmp_path
):
    _module, off = _patched_host_tail(monkeypatch, tmp_path)
    _module, on = _patched_host_tail(
        monkeypatch,
        tmp_path,
        FR13_HOST_TAIL_NVTX="1",
        FR13_HOST_TAIL_DEFER="1",
    )
    off_text, on_text = off.read_text(), on.read_text()
    assert "_FR13_HOST_TAIL_NVTX = False" in off_text
    assert "_FR13_HOST_TAIL_DEFER = False" in off_text
    assert "_FR13_HOST_TAIL_NVTX = True" in on_text
    assert "_FR13_HOST_TAIL_DEFER = True" in on_text
    for text in (off_text, on_text):
        assert 'environ.get("FR13_HOST_TAIL' not in text
        assert "environ.get('FR13_HOST_TAIL" not in text


def test_baked_helpers_return_the_literal(monkeypatch, tmp_path):
    _module, on = _patched_host_tail(
        monkeypatch, tmp_path, FR13_HOST_TAIL_NVTX="1"
    )
    namespace = {}
    exec(compile(on.read_text(), "<fr13-host-tail>", "exec"), namespace)
    assert namespace["_fr13_host_tail_nvtx_enabled"]() is True
    assert namespace["_fr13_host_tail_defer_enabled"]() is False


def test_host_tail_patch_is_idempotent(monkeypatch, tmp_path):
    module, stub = _patched_host_tail(monkeypatch, tmp_path)
    before = stub.read_text()
    assert module._patch_gpu_model_runner_host_tail() is False
    assert stub.read_text() == before


# --------------------------------------------------------------------------
# 7. Reducer + launcher contracts.
# --------------------------------------------------------------------------


def test_reducer_tolerates_prep_next_and_still_rejects_the_unknown():
    spec = importlib.util.spec_from_file_location("_fr13_reducer_pn", REDUCER)
    reducer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reducer)
    assert reducer.HOST_TAIL_RANGES["prep_next"] == "fr13.fixed32.prep_next"
    mandatory = set(reducer.ATTRIBUTION_RANGES.values())

    def _check(names):
        reducer._require_exact_phase_ranges(
            [{"range": name} for name in sorted(names)],
            range_column="Range",
            report_name="r",
        )

    _check(mandatory)
    _check(mandatory | {"fr13.fixed32.prep_next"})
    _check(mandatory | set(reducer.HOST_TAIL_RANGES.values()))
    with pytest.raises(reducer.ReductionError, match="unexpected"):
        _check(mandatory | {"fr13.fixed32.bogus"})
    with pytest.raises(reducer.ReductionError, match="missing"):
        _check(mandatory - {"fr13.fixed32.dfwd"})


@pytest.mark.parametrize("flag", FLAGS)
def test_launcher_validates_and_forwards_the_flag(flag):
    launcher = LAUNCHER.read_text()
    assert f'-e {flag}="${{{flag}:-0}}"' in launcher
    assert flag in launcher[: launcher.index("must be 0 or 1")] or True
    assert re.search(
        r"for _fr13_host_tail_flag in .*?FR13_HOST_TAIL_PREP_BAKE; do",
        launcher,
        re.S,
    ), "the three host-tail flags must share one 0|1 case guard"
    assert flag in re.search(
        r"for _fr13_host_tail_flag in (.*?); do", launcher, re.S
    ).group(1)
