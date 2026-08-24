"""BOOT TEN: the serve step, and the 19 blobs no enumeration could see.

Boot ten reached health, acked generation zero, PASSED the generation-1 flush --
the wholesale close held -- and died on the FIRST REAL REQUEST:

    _fr10_mtp_trace_orig_propose -> RuntimeError:
      FR13 fixed32 mode/topology mismatch:
        mode='hydra31_fixed32' exact_shape=False nodes=31

nodes=31 was CORRECT. The predicate compared the served tree against ONE
hardcoded list -- hydra27's -- so a correct hydra31 tree could only ever read
as a mismatch.

THE REACH FAILURE BEHIND IT, and it is the honest headline of this landing.
Every blob enumeration I have run filtered planted strings through
``ast.parse(text)`` and skipped whatever raised SyntaxError. Planted fragments
that are INDENTED (they are spliced into an existing block, so they do not
parse standalone) were silently dropped. Nineteen blobs were invisible that
way, including the 278KB one that killed boot ten. Dedent-then-parse reaches
them; every scan in this file does that now.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

PATCHER = SCRIPTS / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_SOURCE = PATCHER.read_text(encoding="utf-8")


def _topology():
    import importlib

    return importlib.import_module("fr13_fixed32_topology")


def planted_blobs(path: Path) -> list[tuple[str, str]]:
    """Every planted string that is Python -- INDENTED FRAGMENTS INCLUDED.

    The dedent fallback is the whole point: without it this returns 30 of the
    49 parseable blobs in the closure and silently drops the rest.
    """
    out: list[tuple[str, str]] = []
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError):
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        if len(value.value) <= 2000:
            continue
        name = next(
            (t.id for t in node.targets if isinstance(t, ast.Name)), "?"
        )
        for candidate in (value.value, textwrap.dedent(value.value)):
            try:
                ast.parse(candidate)
            except SyntaxError:
                continue
            out.append((name, candidate))
            break
    return out


def _propose_blob() -> str:
    for name, text in planted_blobs(PATCHER):
        if "_fr13_fixed32_choices_by_mode" in text:
            return text
    raise AssertionError("the propose guard's blob is gone")


def test_indented_blobs_were_invisible_and_now_are_not() -> None:
    """THE REACH FAILURE, stated so it can fail.

    A standalone-parse-only enumeration must find strictly fewer blobs than a
    dedent-aware one; if that ever stops being true the fallback has been
    dropped.
    """
    import fr14_mode_table_parity as parity

    standalone = dedented = 0
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if (
                not isinstance(value, ast.Constant)
                or not isinstance(value.value, str)
                or len(value.value) <= 2000
            ):
                continue
            direct = True
            try:
                ast.parse(value.value)
            except SyntaxError:
                direct = False
            if direct:
                standalone += 1
                dedented += 1
                continue
            try:
                ast.parse(textwrap.dedent(value.value))
            except SyntaxError:
                continue
            dedented += 1
    assert dedented > standalone, (
        "the dedent fallback found nothing extra -- either it was dropped or "
        "the indented blobs are gone"
    )
    assert dedented - standalone >= 15, (dedented, standalone)


def test_the_propose_guard_carries_a_tree_for_every_mode() -> None:
    blob = _propose_blob()
    tree = ast.parse(blob)
    tables = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_fr13_fixed32_choices",
                    "_fr13_tail10_choices",
                ):
                    tables[target.id] = ast.literal_eval(node.value)
    assert set(tables) == {"_fr13_fixed32_choices", "_fr13_tail10_choices"}
    topology = _topology()
    assert [tuple(x) for x in tables["_fr13_fixed32_choices"]] == [
        tuple(x) for x in topology.FIXED32_CHOICES
    ]
    assert [tuple(x) for x in tables["_fr13_tail10_choices"]] == [
        tuple(x) for x in topology.TAIL10_CHOICES
    ]
    # the two profiles genuinely differ, or the guard proves nothing
    assert tables["_fr13_fixed32_choices"] != tables["_fr13_tail10_choices"]


def _run_propose_guard(mode: str, choices, nspec: int = 31, decode: str = "tree_mtp"):
    """Execute the guard block itself, exactly as planted -- no source edits.

    The block reads ``self.tree_choices`` and ``self.num_speculative_tokens``,
    so a stand-in object supplies both and the planted text runs verbatim.
    """
    import os as _os

    blob = _propose_blob()
    lines = blob.split("\n")
    start = next(
        index
        for index, line in enumerate(lines)
        if "_fr13_fixed32_choices = [" in line
    )
    end = next(
        index
        for index, line in enumerate(lines)
        if "_fr13_hydra23_armed = os.path.exists" in line
    )
    body = textwrap.dedent("\n".join(lines[start:end]))
    engine = type(
        "Engine",
        (),
        {"tree_choices": [tuple(x) for x in choices], "num_speculative_tokens": nspec},
    )()
    namespace = {
        "_FR13_FIXED32_MODE": mode,
        "_fr10_active_decode_mode": decode,
        "os": _os,
        "self": engine,
    }
    exec(body, namespace)  # noqa: S102 - our own planted source


@pytest.mark.parametrize(
    ("mode", "tree", "expected"),
    [
        ("hydra27_fixed32", "h27", "pass"),
        ("tail6_fixed32", "h27", "pass"),
        ("", "h27", "pass"),
        ("hydra31_fixed32", "h31", "pass"),
        ("hydra31_fixed32", "h27", "refuse"),
        ("hydra27_fixed32", "h31", "refuse"),
    ],
)
def test_the_propose_guard_accepts_each_mode_and_still_refuses_a_wrong_tree(
    mode: str, tree: str, expected: str
) -> None:
    """MUTATION PROOF, both directions. The guard keeps its teeth."""
    topology = _topology()
    choices = (
        topology.FIXED32_CHOICES if tree == "h27" else topology.TAIL10_CHOICES
    )
    if expected == "pass":
        _run_propose_guard(mode, choices)
        return
    with pytest.raises(RuntimeError) as refusal:
        _run_propose_guard(mode, choices)
    assert "mode/topology mismatch" in str(refusal.value)


def test_the_propose_refusal_names_which_predicate_failed() -> None:
    """`exact_shape=False nodes=31` named the symptom and hid every cause."""
    topology = _topology()
    with pytest.raises(RuntimeError) as refusal:
        _run_propose_guard("hydra31_fixed32", topology.FIXED32_CHOICES)
    message = str(refusal.value)
    for fragment in (
        "decode_mode_ok=",
        "spec_tokens_ok=",
        "tree_ok=",
        "against audited",
        "first differing path",
    ):
        assert fragment in message, fragment
    # the bare form is gone
    blob = _propose_blob()
    assert '" exact_shape=" + str(_fr13_is_fixed32)' not in blob


def test_an_unregistered_mode_refuses_rather_than_guessing_a_tree() -> None:
    topology = _topology()
    with pytest.raises(RuntimeError) as refusal:
        _run_propose_guard("hydra99_fixed32", topology.FIXED32_CHOICES)
    assert "propose guard has no tree for mode" in str(refusal.value)


# --------------------------------------------------------------------------- #
# THE SERVE-PATH ENUMERATION, and its residual                                 #
# --------------------------------------------------------------------------- #
#: work_census entry points that take a mode POSITIONALLY. A call that omits a
#: keyword `mode=` is not evidence of anything for these; the tripwire has to
#: know the difference or it reports twenty false positives.
POSITIONAL_MODE_AUTHORITIES = {"reference_event", "shape_profile"}

#: Host-side callers of forward_graph_structural_signature that pass no mode.
#: These validate BANKED artifacts, where the era default is the correct
#: reading; a hydra31 runroot needs the report-shape decision already recorded
#: against fr13_floor_gate's pair shape, not a blind mode= here.
HOST_SIDE_SIGNATURE_CALLERS = {
    "scripts/fr13_fixed32_work_census.py": "builds and self-tests its own report",
    "scripts/fr13_depth_acceptance.py": "reduce-time validator over banked arms",
    "scripts/fr13_floor_gate.py": "reduce-time gate over banked arms",
}


def test_no_engine_path_authority_call_omits_the_mode() -> None:
    """THE FOURTH TRIPWIRE over every planted blob, indented ones included.

    Scope is what the ENGINE executes: the planted blobs. Host-side reduce
    callers are classified above, not silently swept in.
    """
    import fr13_fixed32_work_census as census
    import fr14_mode_table_parity as parity

    keyword_authorities = set()
    for name in dir(census):
        target = getattr(census, name, None)
        if name.startswith("_") or not inspect.isfunction(target):
            continue
        try:
            params = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if "mode" in params and name not in POSITIONAL_MODE_AUTHORITIES:
            keyword_authorities.add(name)
    assert "forward_graph_structural_signature" in keyword_authorities
    offenders: list[tuple[str, int, str]] = []
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        for blob_name, text in planted_blobs(path):
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in keyword_authorities
                ):
                    continue
                if "mode" not in {kw.arg for kw in node.keywords}:
                    offenders.append((f"{rel}::{blob_name}", node.lineno, node.func.id))
    assert not offenders, (
        "planted code calling a mode-aware authority without a mode: "
        + repr(offenders)
    )


def test_the_host_side_signature_callers_are_classified() -> None:
    for rel, reason in HOST_SIDE_SIGNATURE_CALLERS.items():
        assert (REPO / rel).is_file(), rel
        assert len(reason) > 25, rel


def test_the_residual_un_enumerated_region_is_named_and_small() -> None:
    """ORDER 3: 'none' is a claim; this makes the residual checkable.

    After the dedent fallback, what remains un-enumerated is the set of planted
    strings that are NOT Python at all -- CUDA/C++ translation units, kernel
    replacements and prompt templates. They cannot carry a Python mode
    predicate, but they CAN carry a baked geometry constant, so they are named
    here rather than claimed clean.
    """
    import fr14_mode_table_parity as parity

    non_python: dict[str, int] = {}
    for rel in parity.serve_execution_closure():
        if not rel.endswith(".py"):
            continue
        path = REPO / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if (
                not isinstance(value, ast.Constant)
                or not isinstance(value.value, str)
                or len(value.value) <= 2000
            ):
                continue
            ok = False
            for candidate in (value.value, textwrap.dedent(value.value)):
                try:
                    ast.parse(candidate)
                    ok = True
                    break
                except SyntaxError:
                    continue
            if not ok:
                non_python[rel] = non_python.get(rel, 0) + 1
    # The residual is real and bounded; it lives in the CUDA/C++ patchers.
    assert non_python, "the residual vanished -- re-derive the claim"
    assert set(non_python) <= {
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
        "scripts/fr13_patch_cutlass_fixed32_wave.py",
        "scripts/fr13_patch_fa2_tree_bias.py",
        "scripts/fr13_patch_fp8_quant_fixed32.py",
        "scripts/fr14_patch_nvfp4_lmhead.py",
        "src/lumo_flywheel_serving/auto_research.py",
    }, non_python
    assert sum(non_python.values()) <= 30, non_python


def test_hydra27_still_resolves_to_exactly_the_tree_it_always_did() -> None:
    """PAIRING EVIDENCE for the declared blob change.

    The planted bytes had to move -- the list hydra27 compares against stopped
    being the only one -- so what must hold instead is that hydra27 and tail6
    resolve to the identical tree, and that the unset mode does too.
    """
    blob = _propose_blob()
    lines = blob.split("\n")
    start = next(
        index
        for index, line in enumerate(lines)
        if "_fr13_fixed32_choices = [" in line
    )
    end = next(
        index
        for index, line in enumerate(lines)
        if "_fr13_fixed32_expected_choices = " in line
    )
    namespace: dict = {}
    exec(  # noqa: S102 - our own planted source
        textwrap.dedent("\n".join(lines[start : end + 3])).replace(
            "_fr13_fixed32_expected_choices = _fr13_fixed32_choices_by_mode.get(\n"
            "    _FR13_FIXED32_MODE\n)",
            "",
        ),
        namespace,
    )
    by_mode = namespace["_fr13_fixed32_choices_by_mode"]
    topology = _topology()
    era = [tuple(x) for x in topology.FIXED32_CHOICES]
    for mode in ("", "tail6_fixed32", "hydra27_fixed32"):
        assert [tuple(x) for x in by_mode[mode]] == era, mode
    assert [tuple(x) for x in by_mode["hydra31_fixed32"]] == [
        tuple(x) for x in topology.TAIL10_CHOICES
    ]
    # tail6 and hydra27 share the era list BY REFERENCE, not a retyped copy
    assert by_mode["tail6_fixed32"] is by_mode["hydra27_fixed32"]
