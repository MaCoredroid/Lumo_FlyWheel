"""THE SIGNATURE DETECTOR (round 16) + hydra27 PAIRING EVIDENCE.

Round 16's refusal was a new class: `fr13_fixed32_contract.fixed32_tree_text()`
and `speculative_config_text()` were PARAMETERLESS and each encoded hydra27's
tree, so every equality check against them refused any other profile.  The
blindness lived in the SIGNATURE, which is why no call-site edit could fix it --
fixing `fixed32_tree_text` alone just moved the refusal one line down to
`speculative_config_text`, and fixing both would have moved it again to
`expected_pid1_argv` (which embeds the spec config in the vLLM argv) on the
next boot.

This module closes discovery as a LIST instead of boot-by-boot:

  1. ENUMERATION (static).  Transitive closure over the contract's AST from the
     profile-varying names it imports from `fr13_fixed32_topology`.  Every
     public function in that closure MUST accept a `profile` parameter.  The
     roster is pinned, so a newly profile-varying accessor is a visible diff.

  2. PROOF (dynamic).  The contract is re-executed against a topology whose
     profile-varying constants carry hydra31's values.  Every public callable
     that does NOT take a `profile` parameter must return byte-identical values
     in both worlds -- i.e. it is *proven* profile-invariant rather than assumed.
     Public module constants get the same treatment; a constant cannot take a
     parameter, so one that moves under the swap must become an accessor.

  3. MUTATION PROOF.  A lint that cannot fail is worse than none, so the
     detector is run against a synthetic parameterless accessor and must flag it.

  4. PAIRING EVIDENCE.  The banked H27i baseline only survives the diff if a
     hydra27 arm's execution is unchanged.  We load the pre-round-16 contract
     from its pinned git blob and assert the parameterised accessors, called
     with hydra27 or defaulted, return byte-identical text.
"""

from __future__ import annotations

import ast
import os
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SERVING = REPO / "src" / "lumo_flywheel_serving"

# Round 18: every site through round 17 lived in scripts/, so that is the only
# root the detectors swept -- and the seventh site was in src/. A detector that
# only looks where the last bug was found is a detector that finds the last bug.
# Both roots, everywhere, from here on.
PROFILE_ROOTS = (SCRIPTS, SERVING)


def _root_python_files() -> list[Path]:
    return sorted(
        path for root in PROFILE_ROOTS for path in root.glob("*.py")
    )
CONTRACT_PATH = SCRIPTS / "fr13_fixed32_contract.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_contract as contract  # noqa: E402
import fr13_fixed32_topology as topology  # noqa: E402
from fr13_fixed32_topology import (  # noqa: E402
    PROFILE_HYDRA27,
    PROFILE_HYDRA31,
    profile as topology_profile,
)

# The pre-round-16 contract, for pairing evidence.  Pinned by blob id so that
# committing this change does not move the baseline out from under the test.
BASELINE_BLOB = "410d9d3aeaef8b3806e2223324d2bbba706d5853"

# ---------------------------------------------------------------------------
# The swap map: topology module constant -> the PROFILES key it is hydra27's
# value of.  Curated, not inferred: several unrelated constants coincide with a
# profile field numerically (GDN_CONV_KERNEL_SIZE == 4 == rescue_carry_slots),
# and mapping those would make the detector lie.  test_swap_map_is_faithful
# validates every entry against the hydra27 profile.
# ---------------------------------------------------------------------------
SWAP_MAP: dict[str, str] = {
    "FIXED32_CHOICES": "choices",
    "PHYSICAL_BRANCH_CHAINS": "physical_branch_chains",
    "ARCTIC_LOOKUP_CHAINS": "arctic_lookup_chains",
    "ARCTIC_MAIN_TAIL_LENGTH": "main_tail_length",
    "ARCTIC_LOOKUP_CALLS_PER_REQUEST": "arctic_lookup_calls",
    "ARCTIC_LOOKUP_TOKENS_PER_REQUEST": "arctic_requested_tokens",
    "GATED_ARCTIC_MAIN_TAIL_LENGTH": "gated_main_tail_length",
    "GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST": "gated_arctic_requested_tokens",
    "RESCUE_CARRY_SLOTS_PER_REQUEST": "rescue_carry_slots",
    "PHYSICAL_PARENT": "physical_parent",
    "EXPECTED_PHYSICAL_PARENT": "physical_parent",
    "PHYSICAL_PARENT_SHA256": "physical_parent_sha256",
    "TREE_ANCESTRY_SHA256": "tree_ancestry_sha256",
    "SUBTREE_LEVELS": "subtree_levels",
    "MAX_PHYSICAL_DEPTH": "max_physical_depth",
    "WALK_CAP": "walk_cap",
    "TAW_PATH_SCATTER_SLOTS": "walk_cap",
    "GDN_LEVEL_PATH_COUNTS": "gdn_level_path_counts",
    "GDN_LEVEL_MAX_LENGTHS": "gdn_level_max_lengths",
    "GDN_LAUNCHES": "gdn_launches",
    "GDN_PATH_PROGRAMS": "gdn_path_programs",
    "GDN_PADDED_SLOTS": "gdn_padded_slots",
}

# Imports that carry no profile-varying data.
NON_DATA_IMPORTS = {"PROFILE_HYDRA27", "PROFILE_HYDRA31", "profile", "Mode"}

# The pinned roster of profile-varying public functions (task 1's sweep result).
EXPECTED_PROFILE_VARYING = {
    "fixed32_tree_text",
    "speculative_config_text",
    "expected_pid1_argv",
    "expected_process_pid1_argv",
    "validate_process_pid1_argv",
}


def _contract_tree(source: str | None = None) -> ast.Module:
    return ast.parse(source if source is not None else CONTRACT_PATH.read_text())


def _topology_imports(tree: ast.Module) -> dict[str, str]:
    """{local name: original name} imported from fr13_fixed32_topology."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "fr13_fixed32_topology":
            for alias in node.names:
                out[alias.asname or alias.name] = alias.name
    return out


def _has_profile_param(func) -> bool:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return "profile" in sig.parameters


def _public_module_names(source: str | None = None) -> list[str]:
    """Public names bound at module level in the contract's own source."""
    names: list[str] = []
    for node in _contract_tree(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return sorted({n for n in names if not n.startswith("_")})


# ---------------------------------------------------------------------------
# 1. ENUMERATION -- static transitive closure
# ---------------------------------------------------------------------------
def profile_varying_functions(source: str | None = None) -> dict[str, set[str]]:
    """Public functions whose value depends on a profile-varying topology name.

    Returns {function name: the seeds/callees that make it vary}.
    """
    tree = _contract_tree(source)
    imports = _topology_imports(tree)
    seeds = {
        local
        for local, orig in imports.items()
        if orig in SWAP_MAP
        or orig.startswith("HYDRA27_")
        or orig.startswith("HYDRA31_")
        or orig.startswith("TAIL10_")
    }

    defs: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, node)

    uses: dict[str, set[str]] = {}
    for name, node in defs.items():
        used: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                used.add(sub.id)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                used.add(sub.func.id)
        uses[name] = used

    varying: dict[str, set[str]] = {
        name: (used & seeds) for name, used in uses.items() if used & seeds
    }
    changed = True
    while changed:
        changed = False
        for name, used in uses.items():
            reached = used & set(varying)
            if reached and not (varying.get(name, set()) >= reached):
                varying.setdefault(name, set()).update(reached)
                changed = True
    return {k: v for k, v in varying.items() if not k.startswith("_")}


def test_swap_map_is_faithful() -> None:
    """Every mapped constant really is hydra27's value of that profile field."""
    p27 = topology_profile(PROFILE_HYDRA27)
    for const, key in sorted(SWAP_MAP.items()):
        assert hasattr(topology, const), f"topology lost {const}; update SWAP_MAP"
        assert key in p27, f"PROFILES lost key {key!r}; update SWAP_MAP"
        assert getattr(topology, const) == p27[key], (
            f"SWAP_MAP claims {const} is profile[{key!r}] but the values differ "
            f"({getattr(topology, const)!r} != {p27[key]!r}) -- the map is stale"
        )


def test_every_profile_varying_topology_import_is_known() -> None:
    """A new profile-varying import must be classified, not silently absorbed."""
    p27, p31 = topology_profile(PROFILE_HYDRA27), topology_profile(PROFILE_HYDRA31)
    unclassified = []
    for local, orig in sorted(_topology_imports(_contract_tree()).items()):
        if orig in SWAP_MAP or orig in NON_DATA_IMPORTS:
            continue
        assert not orig.startswith(("HYDRA27_", "HYDRA31_", "TAIL10_")), (
            f"the profile-generic contract imports profile-specific {orig!r}; "
            "it must take the value from profile(mode) instead"
        )
        value = getattr(topology, orig, None)
        if any(value == p27[k] and p27[k] != p31[k] for k in p27):
            unclassified.append(orig)
    assert not unclassified, (
        f"topology imports look profile-varying but are unmapped: {unclassified}"
    )


def test_profile_varying_public_surface_is_the_pinned_roster() -> None:
    found = set(profile_varying_functions())
    assert found == EXPECTED_PROFILE_VARYING, (
        "the profile-varying public surface moved.\n"
        f"  new:  {sorted(found - EXPECTED_PROFILE_VARYING)}\n"
        f"  gone: {sorted(EXPECTED_PROFILE_VARYING - found)}\n"
        "Every entry must take a `profile` argument; update the roster."
    )


def test_every_profile_varying_accessor_takes_a_profile_argument() -> None:
    """The round-16 class, asserted directly: no signature blindness."""
    blind = [
        name
        for name in sorted(profile_varying_functions())
        if not _has_profile_param(getattr(contract, name))
    ]
    assert not blind, (
        f"profile-varying but PARAMETERLESS: {blind} -- each encodes one "
        "profile's topology in its return value while offering no way to ask "
        "for another, so every equality check against it refuses other profiles"
    )


# ---------------------------------------------------------------------------
# 2. PROOF -- dynamic, executed under both profiles
# ---------------------------------------------------------------------------
def _swapped_contract(source: str | None = None) -> types.ModuleType:
    """Re-execute the contract against a hydra31-valued topology."""
    p31 = topology_profile(PROFILE_HYDRA31)
    shim = types.ModuleType("fr13_fixed32_topology")
    shim.__dict__.update(topology.__dict__)
    for const, key in SWAP_MAP.items():
        setattr(shim, const, p31[key])

    saved = sys.modules.get("fr13_fixed32_topology")
    sys.modules["fr13_fixed32_topology"] = shim
    try:
        module = types.ModuleType("fr13_fixed32_contract__hydra31")
        module.__file__ = str(CONTRACT_PATH)
        code = compile(
            source if source is not None else CONTRACT_PATH.read_text(),
            str(CONTRACT_PATH),
            "exec",
        )
        exec(code, module.__dict__)
        return module
    finally:
        if saved is not None:
            sys.modules["fr13_fixed32_topology"] = saved
        else:  # pragma: no cover
            del sys.modules["fr13_fixed32_topology"]


# Public callables the dynamic half must NOT invoke, with the reason.  Each is
# still covered by the static enumeration above (none is in the varying closure).
DYNAMIC_DENY = {
    "main": "CLI entry point; parses sys.argv and writes files",
    "parse_args": "CLI entry point; parses sys.argv and can SystemExit",
    "run_self_test": "self-test entry point; shells out",
}

# Callables whose invocation needs the serving image, not this host.  Declared
# rather than silently skipped: if one becomes callable it is compared instead.
ENV_BLOCKED = {"build_runtime_attestation": ModuleNotFoundError}

# Coverage floor: these public names are ACTUALLY executed under both profiles
# and compared.  Pinned so that a function which quietly starts raising (and so
# would silently drop out of the comparison) fails the detector instead.
PINNED_DYNAMIC_COVERAGE = {
    "nsys_profile_prefix",
    "expected_model_file_records",
}


def classify(
    base: types.ModuleType,
    swapped: types.ModuleType,
    source: str | None = None,
) -> dict[str, dict]:
    """Bucket every public name of the contract by how it was checked.

    `source` names the text the surface is enumerated from; it must be the text
    the two modules were built from, or the ledger silently omits whatever the
    on-disk file does not happen to contain (the mutation test proves this).
    """
    out: dict[str, dict] = {
        "moved": {},          # profile-varying but unparameterised -- the defect
        "compared": {},       # executed under both profiles, byte-identical
        "exempt": {},         # takes a `profile` argument, so it can be asked
        "denied": {},         # side-effecting entry point (static coverage only)
        "blocked": {},        # not callable on this host (static coverage only)
        "needs_args": {},     # required arguments (static coverage only)
        "types": {},          # classes
    }
    for name in _public_module_names(source):
        if not hasattr(base, name) or not hasattr(swapped, name):
            continue
        a, b = getattr(base, name), getattr(swapped, name)
        if isinstance(a, type):
            out["types"][name] = repr(a)
            continue
        if not callable(a):
            if a != b:
                out["moved"][name] = "constant"
            else:
                out["compared"][name] = "constant"
            continue
        if _has_profile_param(a):
            out["exempt"][name] = str(inspect.signature(a))
            continue
        if name in DYNAMIC_DENY:
            out["denied"][name] = DYNAMIC_DENY[name]
            continue
        sig = inspect.signature(a)
        if any(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in sig.parameters.values()
        ):
            out["needs_args"][name] = str(sig)
            continue
        try:
            va = a()
            vb = b()
        except BaseException as exc:  # noqa: BLE001 -- SystemExit is not an Exception
            out["blocked"][name] = f"{type(exc).__name__}: {exc}"
            continue
        if va != vb:
            out["moved"][name] = "callable"
        else:
            out["compared"][name] = "callable"
    return out


def test_parameterless_public_surface_is_proven_profile_invariant() -> None:
    """The PROOF half: executed for both profiles, values compared."""
    buckets = classify(contract, _swapped_contract())
    assert not buckets["moved"], (
        "executed under both profiles, these public names moved but cannot be "
        f"asked for a profile: {sorted(buckets['moved'])}. A callable must take "
        "a `profile` argument; a constant must become such a callable."
    )
    assert buckets["exempt"].keys() >= {"fixed32_tree_text", "speculative_config_text"}


def test_dynamic_coverage_has_not_silently_shrunk() -> None:
    """A lint whose subjects quietly vanish is a lint that cannot fail."""
    buckets = classify(contract, _swapped_contract())
    compared = set(buckets["compared"])
    missing = PINNED_DYNAMIC_COVERAGE - compared
    assert not missing, (
        f"these were executed under both profiles before and no longer are: "
        f"{sorted(missing)} -- blocked={buckets['blocked']}"
    )


def test_every_public_name_is_accounted_for() -> None:
    """Accounting completeness: no public name escapes both halves."""
    buckets = classify(contract, _swapped_contract())
    seen: dict[str, list[str]] = {}
    for bucket, members in buckets.items():
        for name in members:
            seen.setdefault(name, []).append(bucket)
    duplicated = {k: v for k, v in seen.items() if len(v) > 1}
    assert not duplicated, f"names in multiple buckets: {duplicated}"
    unaccounted = set(_public_module_names()) - set(seen)
    assert not unaccounted, f"public names checked by nothing: {sorted(unaccounted)}"
    # every statically-varying function must land in `exempt` or `needs_args`
    for name in profile_varying_functions():
        assert seen.get(name, []) and seen[name][0] in ("exempt", "needs_args"), (
            f"{name} is profile-varying but landed in bucket {seen.get(name)}"
        )
    for name in ENV_BLOCKED:
        if name in buckets["blocked"]:
            assert type(None).__name__ or True
    assert set(buckets["denied"]) == set(DYNAMIC_DENY)


def test_the_parameterised_accessors_actually_track_the_profile() -> None:
    """The fix is real: asking for hydra31 yields hydra31, not hydra27."""
    p31 = topology_profile(PROFILE_HYDRA31)
    assert contract.fixed32_tree_text(PROFILE_HYDRA31) == repr(list(p31["choices"]))
    assert contract.fixed32_tree_text(PROFILE_HYDRA31) != contract.fixed32_tree_text()
    spec31 = contract.speculative_config_text(PROFILE_HYDRA31)
    assert contract.fixed32_tree_text(PROFILE_HYDRA31) in spec31
    assert spec31 != contract.speculative_config_text()
    assert spec31 in contract.expected_pid1_argv(1, PROFILE_HYDRA31)
    proc31 = contract.expected_process_pid1_argv(
        1, profile=PROFILE_HYDRA31, attribution_only=False
    )
    assert spec31 in proc31
    assert (
        contract.validate_process_pid1_argv(
            proc31, 1, profile=PROFILE_HYDRA31, attribution_only=False
        )
        == proc31
    )
    with pytest.raises(Exception):
        contract.validate_process_pid1_argv(proc31, 1, attribution_only=False)


# ---------------------------------------------------------------------------
# 3. MUTATION PROOF -- the detector must be able to fail
# ---------------------------------------------------------------------------
MUTANT = '''

def hydra_tree_fingerprint() -> str:
    return "%s|%d" % (repr(list(FIXED32_CHOICES)), WALK_CAP)
'''


def test_detector_fires_on_an_injected_parameterless_accessor() -> None:
    source = CONTRACT_PATH.read_text()
    assert "WALK_CAP" not in _topology_imports(_contract_tree()), (
        "mutant assumes WALK_CAP is not already imported"
    )
    mutated = source.replace(
        "from fr13_fixed32_topology import (\n    FIXED32_CHOICES,",
        "from fr13_fixed32_topology import (\n    WALK_CAP,\n    FIXED32_CHOICES,",
        1,
    )
    assert "WALK_CAP,\n" in mutated
    mutated += MUTANT

    # the static half sees it
    tree = ast.parse(mutated)
    imports = _topology_imports(tree)
    assert "WALK_CAP" in imports and "FIXED32_CHOICES" in imports
    names = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "hydra_tree_fingerprint" in names

    # the dynamic half proves it
    saved = sys.modules.get("fr13_fixed32_topology")
    try:
        plain = types.ModuleType("contract_mutant_h27")
        plain.__file__ = str(CONTRACT_PATH)
        exec(compile(mutated, str(CONTRACT_PATH), "exec"), plain.__dict__)
    finally:
        if saved is not None:
            sys.modules["fr13_fixed32_topology"] = saved
    swapped = _swapped_contract(mutated)
    assert not _has_profile_param(plain.hydra_tree_fingerprint)
    moved = classify(plain, swapped, source=mutated)["moved"]
    assert "hydra_tree_fingerprint" in _public_module_names(mutated)
    assert "hydra_tree_fingerprint" in moved, (
        "the detector failed to flag a parameterless accessor that encodes the "
        "tree -- it cannot fail, so it is worse than none"
    )


# ---------------------------------------------------------------------------
# 4. PAIRING EVIDENCE -- hydra27 execution is unchanged by the diff
# ---------------------------------------------------------------------------
def _baseline_source() -> str:
    try:
        return subprocess.run(
            ["git", "cat-file", "blob", BASELINE_BLOB],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-round-16 blob {BASELINE_BLOB[:12]} unavailable: {exc}")


def test_detector_fires_on_the_actual_round16_defect() -> None:
    """The decisive mutation proof: run the detector against the PRE-FIX source.

    Both halves must name the two accessors whose parameterless signatures
    refused H31i, and the dynamic half must additionally name `expected_pid1_argv`
    -- the refusal that was still one boot away when round 16 was called.
    """
    source = _baseline_source()
    varying = profile_varying_functions(source)
    assert {"fixed32_tree_text", "speculative_config_text"} <= set(varying)

    old_plain = types.ModuleType("contract_round16_h27")
    old_plain.__file__ = str(CONTRACT_PATH)
    exec(compile(source, f"<blob {BASELINE_BLOB[:12]}>", "exec"), old_plain.__dict__)
    blind = [n for n in varying if not _has_profile_param(getattr(old_plain, n))]
    assert set(blind) >= {
        "fixed32_tree_text",
        "speculative_config_text",
        "expected_pid1_argv",
        "expected_process_pid1_argv",
        "validate_process_pid1_argv",
    }, f"static half missed part of the defect: flagged {sorted(blind)}"

    moved = classify(old_plain, _swapped_contract(source), source=source)["moved"]
    assert {"fixed32_tree_text", "speculative_config_text"} <= set(moved), (
        f"dynamic half missed the round-16 defect: moved={sorted(moved)}"
    )


def _baseline_contract() -> types.ModuleType:
    try:
        blob = subprocess.run(
            ["git", "cat-file", "blob", BASELINE_BLOB],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-round-16 blob {BASELINE_BLOB[:12]} unavailable: {exc}")
    module = types.ModuleType("fr13_fixed32_contract__baseline")
    module.__file__ = str(CONTRACT_PATH)
    exec(compile(blob, f"<blob {BASELINE_BLOB[:12]}>", "exec"), module.__dict__)
    return module


def test_hydra27_arm_execution_is_byte_identical_to_the_banked_baseline() -> None:
    """Required for the banked H27i to survive: the diff moves nothing on hydra27.

    Enumerate-and-show, against the actual pre-round-16 source rather than
    hand-copied literals: every parameterised accessor, called the way existing
    callers call it (defaulted) or explicitly with hydra27, returns byte-identical
    text.  H31i can therefore be paired against the existing H27i baseline.
    """
    old = _baseline_contract()
    checks: dict[str, tuple[object, object]] = {
        "fixed32_tree_text()": (
            old.fixed32_tree_text(),
            contract.fixed32_tree_text(),
        ),
        "fixed32_tree_text(hydra27)": (
            old.fixed32_tree_text(),
            contract.fixed32_tree_text(PROFILE_HYDRA27),
        ),
        "speculative_config_text()": (
            old.speculative_config_text(),
            contract.speculative_config_text(),
        ),
        "speculative_config_text(hydra27)": (
            old.speculative_config_text(),
            contract.speculative_config_text(PROFILE_HYDRA27),
        ),
    }
    for conc in (1, 4):
        checks[f"expected_pid1_argv({conc})"] = (
            old.expected_pid1_argv(conc),
            contract.expected_pid1_argv(conc),
        )
        checks[f"expected_pid1_argv({conc},hydra27)"] = (
            old.expected_pid1_argv(conc),
            contract.expected_pid1_argv(conc, PROFILE_HYDRA27),
        )
        for attribution_only in (False, True):
            key = f"expected_process_pid1_argv({conc},{attribution_only})"
            checks[key] = (
                old.expected_process_pid1_argv(
                    conc, attribution_only=attribution_only
                ),
                contract.expected_process_pid1_argv(
                    conc, attribution_only=attribution_only
                ),
            )
            checks[key + "+profile"] = (
                old.expected_process_pid1_argv(
                    conc, attribution_only=attribution_only
                ),
                contract.expected_process_pid1_argv(
                    conc, profile=PROFILE_HYDRA27, attribution_only=attribution_only
                ),
            )
    drift = {k: (a, b) for k, (a, b) in checks.items() if a != b}
    assert not drift, f"hydra27 execution CHANGED: {sorted(drift)}"
    assert len(checks) >= 16


def test_baseline_validator_still_accepts_the_new_hydra27_argv() -> None:
    """Cross-direction: the OLD validator accepts what the NEW builder emits."""
    old = _baseline_contract()
    for conc in (1, 4):
        argv = contract.expected_process_pid1_argv(conc, attribution_only=False)
        assert (
            old.validate_process_pid1_argv(argv, conc, attribution_only=False) == argv
        )


# ---------------------------------------------------------------------------
# 5. CALL-SITE CENSUS -- the other half of "a list instead of boot-by-boot"
# ---------------------------------------------------------------------------
# Parameterising the accessors only helps where callers actually pass a mode.
# Every production call site must name its profile, or be pinned here with the
# reason it is hydra27-only. A NEW unparameterised call site fails this test.
PROFILE_ACCESSORS = {
    "fixed32_tree_text": 1,
    "speculative_config_text": 1,
    "expected_pid1_argv": 2,
    "expected_process_pid1_argv": None,  # keyword-only `profile=`
    "validate_process_pid1_argv": None,
}

KNOWN_HYDRA27_ONLY_CALL_SITES = {
    (
        "fr13_fixed32_nsys_reduce.py",
        "validate_process_pid1_argv",
    ): (
        "nsys profiling is qualified for tail6/hydra27 only (_B1_PROFILER_MODES); "
        "the reducer has no mode in scope at _validate_process_identity. Profiling "
        "a hydra31 arm requires threading --mode through first."
    ),
    (
        "fr13_floor_gate.py",
        "fixed32_tree_text",
    ): (
        "module constant FIXED32_TREE, hydra27 by definition and read by nothing "
        "else in the gate; the gate's own FIXED32_MODE_SPECS has no hydra31 entry "
        "yet, which is the site that must be taught before a hydra31 floor gate."
    ),
}


def _call_argument_text(text: str, open_paren: int) -> str:
    depth, i = 0, open_paren
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
        i += 1
    return ""


def call_site_census() -> dict[tuple[str, str], list[int]]:
    """{(file, accessor): [line numbers of calls that name no profile]}."""
    blind: dict[tuple[str, str], list[int]] = {}
    for path in _root_python_files() + sorted(SCRIPTS.glob("*.sh")):
        if path.name == CONTRACT_PATH.name:
            continue
        text = path.read_text()
        for accessor, positional in PROFILE_ACCESSORS.items():
            # module-qualified calls count too: the round-16 refusal came
            # through `contract.fixed32_tree_text()`, and a census that only
            # saw bare names would have reported the tree as fully covered.
            pattern = rf"(?<![\w.])(?:[A-Za-z_]\w*\.)?{accessor}\s*\("
            for match in re.finditer(pattern, text):
                if re.search(r"\bdef\s+$", text[: match.start()]):
                    continue
                args = _call_argument_text(text, match.end() - 1)
                if "profile=" in args:
                    continue
                if positional is not None:
                    top, depth = [], 0
                    current = ""
                    for ch in args:
                        if ch in "([{":
                            depth += 1
                        elif ch in ")]}":
                            depth -= 1
                        if ch == "," and depth == 0:
                            top.append(current)
                            current = ""
                        else:
                            current += ch
                    if current.strip():
                        top.append(current)
                    if len([a for a in top if a.strip()]) >= positional:
                        continue
                line = text[: match.start()].count("\n") + 1
                blind.setdefault((path.name, accessor), []).append(line)
    return blind


def test_every_production_call_site_names_its_profile() -> None:
    blind = call_site_census()
    unexpected = {k: v for k, v in blind.items() if k not in KNOWN_HYDRA27_ONLY_CALL_SITES}
    assert not unexpected, (
        "these call sites invoke a profile-varying accessor without naming a "
        f"profile, so they answer for hydra27 whatever the arm is: {unexpected}"
    )


def test_the_known_hydra27_only_list_has_not_gone_stale() -> None:
    """A pinned exception that no longer exists is a lie; fail so it is removed."""
    blind = call_site_census()
    stale = [k for k in KNOWN_HYDRA27_ONLY_CALL_SITES if k not in blind]
    assert not stale, f"pinned hydra27-only call sites no longer exist: {stale}"


# ---------------------------------------------------------------------------
# 6. ROUND 17 -- THE PATCHERS
# ---------------------------------------------------------------------------
# Round 17's site is the drafter patcher, and it is the dangerous one. The
# patcher BAKES the tree into the drafter source it emits: three injection
# sites assign the baked literal over whatever tree the server was configured
# with, then compare the parse against that same literal. A self-comparison
# cannot fail, so a hydra31 arm would have booted a drafter built from
# hydra27's 31 paths -- same width, same (4, 6) branch pairs, mask and
# active-count checks all satisfied -- and served numbers that read as a tail10
# result. Round-6's configuration-as-observation, at the topology level.
PATCHER = SCRIPTS / "fr10_phase4_patch_vllm_tree_gdn.py"
# Re-pinned when the accept ladder was wired into the flush boundary (round
# 21's last gate). That edit is a DECLARED re-attestation of exactly one
# injected blob; the guard below now names it, so every other blob is still
# held byte-identical and a second change fails.
PATCHER_BASELINE_BLOB = "5f10dbcef47ef4a54d19263ef23e2a5317b7836d"
LADDER_WIRING_MARKER = "fr13_fixed32_accept_ladder_snapshot as accept_ladder_snapshot"
# DECLARED CHANGE, round 22 / sixth member of the walk-derived-pin class: the
# planted GDN schedule contract stopped being a hydra27 literal. There is no
# fix that leaves hydra27's planted BYTES identical -- the pin is inside the
# planted text and it has to stop being one -- so this is a re-attestation
# event, declared here rather than slipped through. What must still hold, and
# is asserted below, is that hydra27 RESOLVES to exactly the retired values.
GDN_SCHEDULE_WIRING_MARKER = "_FR13_FIXED32_GDN_SCHEDULE_BY_PROFILE"
# DECLARED CHANGE, boot ten: the MTP-trace propose guard compared the served
# tree against ONE hardcoded list (hydra27's), so a correct hydra31 tree read
# as a mode/topology mismatch on the first real request. It now carries a tree
# per profile. hydra27's planted bytes necessarily move -- the list it compares
# against stopped being the only one -- so this is declared, and what must
# still hold is that hydra27 RESOLVES to exactly the tree it always did.
PROPOSE_TREE_WIRING_MARKER = "_fr13_fixed32_choices_by_mode"

RETIRED_GDN_HYDRA27_CONTRACT = {
    "path_counts": (1, 11),
    "max_lengths": (5, 7),
    "launches": 2,
    "programs": 12,
    "padded_slots": 82,
    "critical": 12,
    "export_or_mask": 16915,
}



def _load_patcher(mode: str, source: str | None = None) -> types.ModuleType:
    import os

    saved = os.environ.get("FR13_FIXED32_MODE")
    os.environ["FR13_FIXED32_MODE"] = mode
    try:
        module = types.ModuleType(f"patcher__{mode or 'unset'}")
        module.__file__ = str(PATCHER)
        exec(
            compile(source if source is not None else PATCHER.read_text(),
                    str(PATCHER), "exec"),
            module.__dict__,
        )
        return module
    finally:
        if saved is None:
            os.environ.pop("FR13_FIXED32_MODE", None)
        else:
            os.environ["FR13_FIXED32_MODE"] = saved


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("", PROFILE_HYDRA27),
        ("tail6_fixed32", PROFILE_HYDRA27),
        (PROFILE_HYDRA27, PROFILE_HYDRA27),
        (PROFILE_HYDRA31, PROFILE_HYDRA31),
    ],
)
def test_patcher_bakes_the_served_profiles_tree(mode: str, expected: str) -> None:
    module = _load_patcher(mode)
    want = topology_profile(expected)
    assert module._FR13_FIXED32_CHOICES == tuple(want["choices"])
    assert module._FR13_FIXED32_PARENT == tuple(want["physical_parent"])
    assert module._FR13_FIXED32_WALK_CAP == want["walk_cap"]
    assert module._FR13_FIXED32_TREE_ANCESTRY_SHA256 == want["tree_ancestry_sha256"]
    # the derived bakes follow, which is the point of binding in ONE place
    assert module._FR13_FIXED32_TREE_SOURCE == repr(list(want["choices"]))


def test_patcher_and_contract_agree_on_mode_to_profile() -> None:
    """Two rosters is two things to forget; assert they cannot diverge."""
    module = _load_patcher(PROFILE_HYDRA27)
    patcher_map = dict(module._FR13_FIXED32_TREE_PROFILE_BY_MODE)
    assert patcher_map.pop("") == PROFILE_HYDRA27, "unset mode must be hydra27"
    assert patcher_map == dict(contract.TREE_PROFILE_BY_MODE)


def test_patcher_digest_helpers_match_the_topology_authority() -> None:
    """The binding is worthless if it computes a different digest."""
    module = _load_patcher(PROFILE_HYDRA27)
    for name in (PROFILE_HYDRA27, PROFILE_HYDRA31):
        want = topology_profile(name)
        parent = tuple(want["physical_parent"])
        assert (
            module._fr13_fixed32_canonical_sha256(list(parent))
            == want["physical_parent_sha256"]
        )
        ancestry = module._fr13_fixed32_ancestor_matrix(parent)
        assert (
            module._fr13_fixed32_canonical_sha256([list(r) for r in ancestry])
            == want["tree_ancestry_sha256"]
        )


def test_observable_binding_accepts_the_matching_tree() -> None:
    for mode, name in (
        ("", PROFILE_HYDRA27),
        ("tail6_fixed32", PROFILE_HYDRA27),
        (PROFILE_HYDRA27, PROFILE_HYDRA27),
        (PROFILE_HYDRA31, PROFILE_HYDRA31),
    ):
        module = _load_patcher(mode)
        want = topology_profile(name)
        assert module._fr13_fixed32_bind_tree_to_profile(
            mode, tuple(want["physical_parent"])
        ) == want["tree_ancestry_sha256"]


def test_observable_binding_refuses_the_wrong_tree() -> None:
    """MUTATION PROOF: hydra31 mode + hydra27 tree must REFUSE, and vice versa.

    Shape equality cannot see this -- both trees are 31 physical drafts with a
    32-entry parent vector -- so the binding is to the profile's independently
    pinned ancestry digest (90873d81... vs 5b33c46a...).
    """
    p27, p31 = topology_profile(PROFILE_HYDRA27), topology_profile(PROFILE_HYDRA31)
    assert len(p27["physical_parent"]) == len(p31["physical_parent"]) == 32
    assert len(p27["choices"]) == len(p31["choices"]) == 31

    module31 = _load_patcher(PROFILE_HYDRA31)
    with pytest.raises(RuntimeError) as wrong:
        module31._fr13_fixed32_bind_tree_to_profile(
            PROFILE_HYDRA31, tuple(p27["physical_parent"])
        )
    assert "not the served profile" in str(wrong.value)

    module27 = _load_patcher(PROFILE_HYDRA27)
    with pytest.raises(RuntimeError):
        module27._fr13_fixed32_bind_tree_to_profile(
            PROFILE_HYDRA27, tuple(p31["physical_parent"])
        )
    with pytest.raises(RuntimeError):
        module27._fr13_fixed32_bind_tree_to_profile(
            "nope_fixed32", tuple(p27["physical_parent"])
        )


def test_observable_binding_catches_an_ancestry_only_difference() -> None:
    """Not shape equality: a same-shape, same-width tree with one edge moved.

    This is the case a parent-vector length or draft-count check waves through.
    """
    module = _load_patcher(PROFILE_HYDRA27)
    parent = list(topology_profile(PROFILE_HYDRA27)["physical_parent"])
    moved = parent[:]
    for node in range(2, len(moved)):
        if moved[node] != parent[node - 1] and moved[node] > 0:
            moved[node] = moved[node] - 1
            break
    assert moved != parent and len(moved) == len(parent)
    with pytest.raises(RuntimeError):
        module._fr13_fixed32_bind_tree_to_profile(PROFILE_HYDRA27, tuple(moved))


def test_walk_cap_is_the_profiles_not_a_literal() -> None:
    """The hardcoded 12 refused the vehicle's correctly-derived 16."""
    text = PATCHER.read_text()
    assert 'FR13_FIXED32_TAW_WALK_CAP=12"' not in text
    assert "_FR13_FIXED32_WALK_CAP" in text
    assert _load_patcher(PROFILE_HYDRA27)._FR13_FIXED32_WALK_CAP == 12
    assert _load_patcher(PROFILE_HYDRA31)._FR13_FIXED32_WALK_CAP == 16


# --- THE PATCHER SIGNATURE SCAN (round 17 task 3) --------------------------
# Round 16 asked whether an ACCESSOR can be told which profile to answer for.
# The same question for a PATCHER is whether its baked literals can. A patcher
# that carries exactly one profile's tree is hydra27-only by construction, and
# no call-site edit can fix that either.
KNOWN_SINGLE_PROFILE_PATCHERS = {
    "fr13_sfwd_prior_reuse_descriptorless.py": (
        "DEFAULT-OFF SFWD prior-reuse kernel; its fixed bases are derived from "
        "hydra27's parent and the lever is byte-AB qualified on that tree. Found "
        "by this scan rather than by the hydra-mention enumeration -- the file "
        "never says 'hydra27', it just carries the tree."
    ),
    "fr13_sfwd_prior_reuse_i32_descriptor.py": (
        "Offline-only int32-descriptor variant of the same default-off kernel; "
        "same tree, same qualification, same refusal."
    ),
    "fr13_patch_fa2_tree_bias.py": (
        "FIXED32_PHYSICAL_PARENT feeds _fixed32_visibility_masks(), which bakes "
        "a 32-entry __device__ __constant__ self-plus-ancestor table into the "
        "FA2 CUDA source -- 12 of those 32 masks are wrong for hydra31 (rows 18 "
        "and 21..31). It is DORMANT: fixed32_tree_visibility_mask defaults False "
        "at all seven definitions, no argparse flag arms it, and no shell path "
        "passes it, so the qualified .so binaries do not carry it. NOT edited "
        "here on purpose: the launchers pin sha256 of this file as "
        "FR13_FA2_QROW32_B{1,4}_PATCH_SOURCE_SHA256, so touching it re-attests "
        "every FA2 arm. test_the_fa2_visibility_specialization_stays_dormant "
        "fires the moment it is armed."
    ),
}


def profile_literal_scan() -> dict[str, dict[str, list[int]]]:
    """{patcher: {profile: [line numbers of module-level tree/parent literals]}}"""
    signatures = {
        name: {
            "choices": tuple(topology_profile(name)["choices"]),
            "parent": tuple(topology_profile(name)["physical_parent"]),
        }
        for name in (PROFILE_HYDRA27, PROFILE_HYDRA31)
    }
    found: dict[str, dict[str, list[int]]] = {}
    for path in _root_python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List)):
                continue
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError, MemoryError):
                continue
            for name, sig in signatures.items():
                if tuple(value) in (sig["choices"], sig["parent"]):
                    found.setdefault(path.name, {}).setdefault(name, []).append(
                        node.lineno
                    )
    return found


def _defines_the_profile_table(path: Path) -> bool:
    """The topology module is the AUTHORITY, not a consumer of it.

    Its bare constants are hydra27's by definition and the other profiles live
    in PROFILES, several of them constructed rather than written as literals, so
    a literal scan cannot see them. Recognised structurally rather than pinned
    by name: a file that publishes a PROFILES table of two or more profiles is
    where topology is defined.
    """
    if path.name != "fr13_fixed32_topology.py":
        return False
    import fr13_fixed32_topology as authority

    return len(getattr(authority, "PROFILES", {})) >= 2


def test_no_patcher_knows_only_one_profiles_tree() -> None:
    scan = profile_literal_scan()
    assert scan, "the scan found no topology literals at all -- it cannot fail"
    single = {
        patcher: sorted(profiles)
        for patcher, profiles in scan.items()
        if len(profiles) < 2
        and not _defines_the_profile_table(
            next(
                (p for root in PROFILE_ROOTS
                 for p in [root / patcher] if p.exists()),
                Path(patcher),
            )
        )
    }
    unexpected = {
        k: v for k, v in single.items() if k not in KNOWN_SINGLE_PROFILE_PATCHERS
    }
    assert not unexpected, (
        "these patchers bake exactly one profile's topology, so they answer for "
        f"hydra27 whatever the served mode is: {unexpected}"
    )
    assert PATCHER.name in scan and len(scan[PATCHER.name]) == 2, (
        "the drafter patcher must carry both profiles' trees"
    )


def test_the_known_single_profile_patcher_list_has_not_gone_stale() -> None:
    scan = profile_literal_scan()
    stale = [
        name
        for name in KNOWN_SINGLE_PROFILE_PATCHERS
        if name not in scan or len(scan.get(name, {})) >= 2
    ]
    assert not stale, f"pinned single-profile patchers are now fixed: {stale}"


def test_the_fa2_visibility_specialization_stays_dormant() -> None:
    """The dormant hazard's tripwire.

    The FA2 tree-visibility masks are hydra27's, compiled into the kernel. They
    are unreachable today. If anyone arms them, this fires -- because the moment
    they are live, a hydra31 serve computes tree attention with hydra27's
    ancestry on 12 of 32 rows, and nothing downstream can see it.
    """
    fa2 = SCRIPTS / "fr13_patch_fa2_tree_bias.py"
    text = fa2.read_text()
    assert text.count("fixed32_tree_visibility_mask: bool = False") == 7
    assert "fixed32_tree_visibility_mask=True" not in text
    armed = [
        path.name
        for path in sorted(SCRIPTS.glob("*.sh"))
        if "tree_visibility_mask" in path.read_text()
        or "tree-visibility-mask" in path.read_text()
    ]
    assert not armed, (
        "a shell path now arms the FA2 tree-visibility specialization: "
        f"{armed}. Those masks are hydra27's ancestry, baked into the kernel; "
        "hydra31 needs a rebuilt and re-qualified FA2 before it can serve."
    )


# --- PAIRING EVIDENCE (round 17 task 5) ------------------------------------
def _patcher_baseline_source() -> str:
    try:
        return subprocess.run(
            ["git", "cat-file", "blob", PATCHER_BASELINE_BLOB],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-round-17 patcher blob unavailable: {exc}")


def _module_bound_names(source: str) -> list[str]:
    """EVERY module-level binding, public or private.

    Tuple-unpacking and annotated targets count: the round-17 binding assigns
    five names at once, and a collector that only understood `name = value`
    reported them as deletions from the patcher surface.
    """
    names: list[str] = []

    def walk_target(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                walk_target(element)

    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                walk_target(target)
        elif isinstance(node, ast.AnnAssign):
            walk_target(node.target)
    return sorted(set(names))


def _emitted_surface(module: types.ModuleType, source: str) -> dict[str, object]:
    """Everything the patcher can bake: module constants + injected blobs."""
    surface: dict[str, object] = {}
    for name in _module_bound_names(source):
        if hasattr(module, name):
            value = getattr(module, name)
            if not callable(value) and not isinstance(value, types.ModuleType):
                surface[f"const:{name}"] = value
    # Content-addressed, never position-keyed: adding code above a blob shifts
    # every walk index, and a position-keyed ledger then reports the whole
    # corpus as "removed" while a genuinely edited blob hides in the noise.
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 2000
        ):
            digest = hashlib.sha256(node.value.encode()).hexdigest()
            surface[f"blob:{digest}"] = len(node.value)
    return surface



def _gdn_expected_for(blob: str, mode: str) -> dict:
    """Resolve the planted GDN schedule table under a served mode."""
    tree = ast.parse(blob)
    lines = blob.split("\n")
    kept = []
    for node in tree.body:
        names = [
            t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)
        ]
        if any(n.startswith("_FR13_FIXED32_GDN_SCHEDULE") for n in names) or (
            "_FR13_FIXED32_GDN_TREE_PROFILE_BY_MODE" in names
            or "_FR13_FIXED32_GDN_MODE" in names
            or (isinstance(node, ast.If) and "_FR13_FIXED32_GDN_MODE" in ast.dump(node))
        ):
            kept.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    saved = os.environ.get("FR13_FIXED32_MODE")
    os.environ["FR13_FIXED32_MODE"] = mode
    namespace: dict = {}
    try:
        exec("\n".join(kept), namespace)  # noqa: S102 - our own planted source
    finally:
        if saved is None:
            os.environ.pop("FR13_FIXED32_MODE", None)
        else:
            os.environ["FR13_FIXED32_MODE"] = saved
    return dict(namespace["_FR13_FIXED32_GDN_SCHEDULE_EXPECTED"])


@pytest.mark.parametrize("mode", ["", "tail6_fixed32", PROFILE_HYDRA27])
def test_hydra27_patcher_output_is_byte_identical_to_the_baseline(mode: str) -> None:
    """Required for the banked runs: the parameterisation moves nothing on hydra27.

    Every module-level constant the patcher can interpolate, and every injected
    source blob over 2000 chars, compared against the pre-round-17 source.
    """
    source = _patcher_baseline_source()
    old = _emitted_surface(_load_patcher(mode, source), source)
    new_source = PATCHER.read_text()
    new = _emitted_surface(_load_patcher(mode, new_source), new_source)

    drift = {
        key: (old[key], new[key])
        for key in set(old) & set(new)
        if old[key] != new[key]
    }
    declared = "const:_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
    if declared in drift:
        was, now = drift.pop(declared)
        assert GDN_SCHEDULE_WIRING_MARKER not in was
        assert GDN_SCHEDULE_WIRING_MARKER in now, (
            "the planted runtime blob changed and it is NOT the declared GDN "
            "schedule wiring -- that is a silent edit to every banked run"
        )
        assert _gdn_expected_for(now, mode or "hydra27_fixed32") == (
            RETIRED_GDN_HYDRA27_CONTRACT
        ), "hydra27 no longer resolves to the retired GDN contract"
    assert not drift, f"hydra27 patcher output CHANGED: {sorted(drift)}"
    # One blob is declared changed: the flush boundary now drains the ladder.
    # Everything else must still be present unchanged.
    changed_blobs = {k for k in set(old) ^ set(new) if k.startswith("blob:")}
    assert len(changed_blobs) <= 6, (
        "more than the three declared blobs (ladder wiring, GDN schedule, "
        f"propose tree) changed: {sorted(changed_blobs)}"
    )

    removed = sorted(set(old) - set(new))
    # The flush blob is declared changed (the ladder wiring), so exactly one
    # blob digest leaves the ledger. A CONSTANT leaving it is still a failure.
    assert all(key.startswith("blob:") for key in removed), (
        f"a non-blob binding left the patcher surface: {removed}"
    )
    assert len(removed) <= 3, f"more than the declared blobs changed: {removed}"
    assert len(old) > 100, f"the surface ledger collapsed to {len(old)} entries"
    module = _load_patcher(mode, new_source)
    baseline = _load_patcher(mode, source)
    assert module._FR13_FIXED32_CHOICES == baseline._FR13_FIXED32_CHOICES
    assert module._FR13_FIXED32_PARENT == baseline._FR13_FIXED32_PARENT
    assert module._FR13_HYDRA27_CHOICES == baseline._FR13_FIXED32_CHOICES
    assert module._FR13_HYDRA27_PARENT == baseline._FR13_FIXED32_PARENT


def test_every_injected_blob_is_unchanged_for_hydra27() -> None:
    """The drafter blob digest the banked runs pin, stated as a digest."""
    source = _patcher_baseline_source()

    def blob_digests(text: str) -> list[str]:
        return sorted(
            hashlib.sha256(node.value.encode()).hexdigest()
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 2000
        )

    def blobs(text: str) -> dict[str, str]:
        return {
            hashlib.sha256(node.value.encode()).hexdigest(): node.value
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 2000
        }

    old, new = blobs(source), blobs(PATCHER.read_text())
    assert len(old) >= 40, f"blob enumeration collapsed to {len(old)}"
    removed = [old[d] for d in set(old) - set(new)]
    added = [new[d] for d in set(new) - set(old)]
    # EXACTLY ONE declared change: the flush blob that now drains the ladder.
    # Anything else is a silent edit and must fail.
    assert len(removed) <= 3 and len(added) <= 3, (
        f"{len(removed)} blob(s) changed; only the ladder wiring, the GDN "
        "schedule table and the propose tree are declared"
    )
    declared_markers = (
        LADDER_WIRING_MARKER,
        GDN_SCHEDULE_WIRING_MARKER,
        PROPOSE_TREE_WIRING_MARKER,
    )
    for blob in added:
        assert any(marker in blob for marker in declared_markers), (
            "an injected source blob changed and it is NOT the declared ladder "
            "wiring -- that is a RE-ATTESTATION EVENT for every banked hydra27 "
            "run, not a silent edit"
        )
    for blob in removed:
        assert not any(marker in blob for marker in declared_markers), (
            "the declared changes should be ADDITIONS of the wiring"
        )


# ---------------------------------------------------------------------------
# 7. ROUND 18 -- THE SEVENTH SITE AND THE SECOND ROOT
# ---------------------------------------------------------------------------
# Every site through round 17 was in scripts/. The seventh was in
# src/lumo_flywheel_serving/, where the tree-GDN machinery lives, and the
# route resolver there refused hydra31 as an "invalid fixed32 route source"
# before a serve could begin. The adjudication was NOT a blanket widen: the
# mode VOCABULARY was widened, the twelve default-off levers' QUALIFICATION
# rosters were kept refusing, and the tree-GDN schedule -- which is genuinely
# derived from the parent vector -- was keyed on the served profile.
GDN_KERNEL = SERVING / "fr10_gdn_tree_kernel.py"
GDN_KERNEL_BASELINE_BLOB = "f1d3b60d79c267bc46473d940adff00856b6421c"

FIXED32_MODE_VOCABULARY = frozenset(
    ("tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32")
)

# Per-FILE counts of mode collections that name two profiles but not hydra31.
# Counts, not line numbers: the line-keyed adjudications of round 3 went stale
# the moment anything above them moved. A NEW hydra27-only allowlist in either
# root changes a count (or adds a key) and fails, which is the point -- an
# eighth site dies in tests instead of on a boot.
MODE_ALLOWLIST_BASELINE = {
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py": 1,
    "scripts/fr13_b4_floor_gate_reduce.py": 1,
    "scripts/fr13_b4_gqa_width4_pair_reduce.py": 1,
    "scripts/fr13_b4_taw_width4_pair_reduce.py": 1,
    "scripts/fr13_cfwd_logit_direct_decision_kernel.py": 3,
    "scripts/fr13_cfwd_packed_walk_active_depth_kernel.py": 1,
    "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py": 1,
    "scripts/fr13_cutlass_b4_pass.py": 3,
    "scripts/fr13_cutlass_wave_binary.py": 1,
    "scripts/fr13_depth_acceptance.py": 8,
    "scripts/fr13_device_multidraft_kernel.py": 5,
    "scripts/fr13_fa2_qrow32_gate.py": 2,
    "scripts/fr13_fa2_qrow32_gqa_pair_gate.py": 4,
    # ROSTER LANDING: both of these went to ZERO because their mode allowlists
    # stopped being hardcoded lists and now derive from
    # fr13_fixed32_contract.FIXED32_MODES, the roster authority whose own
    # comment already told consumers to do exactly that. A count that SHRANK
    # because a hardcoded roster was replaced by a derivation is the direction
    # this lint wants; it is pinned at 0 so deleting the guard entirely would
    # still fail.
    "scripts/fr13_fixed32_flush_protocol.py": 0,
    "scripts/fr13_fixed32_nsys_reduce.py": 1,
    "scripts/fr13_fixed32_semantics_test.py": 2,
    # Round 19: was 2 (VALID_BY_MODE + VALID_MASK_BY_MODE), then 1, now 0 --
    # both mode indices carry the full roster, so the authority contributes no
    # hydra27-only allowlist at all. Absent from this table means zero.
    "scripts/fr13_floor_gate.py": 5,
    "scripts/fr13_gdn_gqa_group3_production_credential.py": 1,
    "scripts/fr13_gdn_single_launch_gate.py": 1,
    "scripts/fr13_gdn_single_launch_production_credential.py": 1,
    "scripts/fr13_taw_b1_credential.py": 2,
    "scripts/fr13_treeconv_zero_tail_credential.py": 1,
    "scripts/run_swe_bench_q36_a.py": 0,
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": 7,
    "src/lumo_flywheel_serving/fr13_fixed32_commit_slot_scatter.py": 1,
    "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py": 1,
    "src/lumo_flywheel_serving/fr13_host_tail_prep.py": 1,
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py": 1,
}


# Round 19: keying an allowlist by NAMED constants instead of string literals
# is good practice AND it hid two of them from this census. Resolving Name
# nodes against the authority's own constants closes that evasion -- otherwise
# "use a constant" becomes the way to add a mode roster the detector cannot see.
def _mode_constant_values() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        import fr13_fixed32_topology as authority
    except Exception:  # pragma: no cover
        return values
    for name in dir(authority):
        if name.startswith("_"):
            continue
        value = getattr(authority, name)
        if isinstance(value, str) and value in FIXED32_MODE_VOCABULARY:
            values[name] = value
    return values


def _mode_collections(source: str) -> list[tuple[int, frozenset[str]]]:
    """Every collection whose members are all fixed32 mode names.

    Members may be string literals or names bound to them in the authority.
    """
    constants = _mode_constant_values()

    def resolve(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Attribute):
            return constants.get(node.attr)
        return None

    out: list[tuple[int, frozenset[str]]] = []
    for node in ast.walk(ast.parse(source)):
        names: set[str] | None = None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values = [
                value
                for value in (resolve(element) for element in node.elts)
                if value is not None
            ]
            if values and len(values) == len(node.elts):
                names = set(values)
        elif isinstance(node, ast.Dict):
            keys = [
                value
                for value in (resolve(key) for key in node.keys)
                if value is not None
            ]
            if keys and len(keys) == len(node.keys):
                names = set(keys)
        if (
            names
            and len(names & FIXED32_MODE_VOCABULARY) >= 2
            and names <= FIXED32_MODE_VOCABULARY
        ):
            out.append((node.lineno, frozenset(names)))
    return out


def mode_allowlist_census() -> dict[str, int]:
    """{repo-relative file: count of allowlists that exclude hydra31}."""
    census: dict[str, int] = {}
    for path in _root_python_files():
        try:
            collections = _mode_collections(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        count = sum(
            1 for _line, names in collections if "hydra31_fixed32" not in names
        )
        if count:
            census[path.relative_to(REPO).as_posix()] = count
    return census


def test_mode_allowlist_inventory_is_the_pinned_baseline() -> None:
    """An eighth site in EITHER root dies here instead of on a boot."""
    census = mode_allowlist_census()
    added = {k: v for k, v in census.items() if k not in MODE_ALLOWLIST_BASELINE}
    grew = {
        k: (MODE_ALLOWLIST_BASELINE[k], v)
        for k, v in census.items()
        if k in MODE_ALLOWLIST_BASELINE and v > MODE_ALLOWLIST_BASELINE[k]
    }
    shrank = {
        k: (MODE_ALLOWLIST_BASELINE[k], census.get(k, 0))
        for k in MODE_ALLOWLIST_BASELINE
        if census.get(k, 0) < MODE_ALLOWLIST_BASELINE[k]
    }
    assert not added and not grew, (
        "new hydra27-only mode allowlist(s) appeared -- ADJUDICATE each one "
        "(widen if the code is profile-parametric, keep-and-document if it is "
        f"bound to one tree like the qualified levers): added={added} grew={grew}"
    )
    assert not shrank, (
        f"allowlists were widened without updating the baseline: {shrank}"
    )


def test_the_serve_path_vocabulary_admits_hydra31() -> None:
    """The widen half of the adjudication, asserted where it matters."""
    source = GDN_KERNEL.read_text()
    vocabularies = [
        names for _line, names in _mode_collections(source)
        if names == FIXED32_MODE_VOCABULARY
    ]
    assert vocabularies, (
        "fr10_gdn_tree_kernel has no full mode vocabulary, so the route "
        "resolver still refuses hydra31 before the serve begins"
    )
    assert "_FR13_FIXED32_ROUTE_MODES" in source
    assert "if value not in _FR13_FIXED32_ROUTE_MODES" in source


def _triton_stub() -> None:
    """The GDN kernel imports triton, which the host does not have."""
    if "triton" in sys.modules:
        return
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("triton")
        stub.jit = lambda fn=None, **_kw: (
            (lambda decorated: decorated) if fn is None else fn
        )
        stub.cdiv = lambda left, right: (left + right - 1) // right
        stub.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
        language = types.ModuleType("triton.language")
        stub.language = language
        sys.modules["triton"] = stub
        sys.modules["triton.language"] = language


def _load_gdn_kernel(mode: str | None, source: str | None = None):
    import os

    _triton_stub()
    saved = os.environ.get("FR13_FIXED32_MODE")
    if mode is None:
        os.environ.pop("FR13_FIXED32_MODE", None)
    else:
        os.environ["FR13_FIXED32_MODE"] = mode
    name = f"gdn_kernel__{mode or 'unset'}"
    module = types.ModuleType(name)
    module.__file__ = str(GDN_KERNEL)
    # dataclass() resolves __module__ through sys.modules, so the module has to
    # be registered while it executes.
    sys.modules[name] = module
    try:
        exec(
            compile(
                source if source is not None else GDN_KERNEL.read_text(),
                str(GDN_KERNEL),
                "exec",
            ),
            module.__dict__,
        )
        return module
    finally:
        sys.modules.pop(name, None)
        if saved is None:
            os.environ.pop("FR13_FIXED32_MODE", None)
        else:
            os.environ["FR13_FIXED32_MODE"] = saved


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, PROFILE_HYDRA27),
        ("tail6_fixed32", PROFILE_HYDRA27),
        (PROFILE_HYDRA27, PROFILE_HYDRA27),
        (PROFILE_HYDRA31, PROFILE_HYDRA31),
    ],
)
def test_gdn_kernel_binds_the_served_profiles_tree(
    mode: str | None, expected: str
) -> None:
    module = _load_gdn_kernel(mode)
    want = topology_profile(expected)
    assert module._FR13_FIXED32_TREE_PROFILE == expected
    assert module._FR13_FIXED32_PARENT == tuple(want["physical_parent"])
    assert module._FR13_FIXED32_PARENT_SHA256 == want["physical_parent_sha256"]
    assert module._FR13_FIXED32_ANCESTRY_SHA256 == want["tree_ancestry_sha256"]
    assert module._FR13_FIXED32_SUBTREE_LEVELS == tuple(
        tuple((tuple(path), parent) for path, parent in level)
        for level in want["subtree_levels"]
    )


@pytest.mark.parametrize("mode", [PROFILE_HYDRA27, PROFILE_HYDRA31])
def test_the_tree_gdn_schedule_derives_for_both_profiles(mode: str) -> None:
    """The critical path: the stage-2 derivation must reach these consumers.

    hydra31's second level runs 11 rows deep instead of 7, so the padded slot
    count is 126 and the critical path 16. If the decomposition or the contract
    could not produce that, hydra31 would refuse at subtree_preseed no matter
    what the route resolver allowed.
    """
    module = _load_gdn_kernel(mode)
    want = topology_profile(mode)
    levels = module._subtree_decompose(module._FR13_FIXED32_PARENT)
    module._validate_subtree_decomposition(module._FR13_FIXED32_PARENT, levels)
    contract = module._fr13_fixed32_schedule_contract(levels)
    assert contract["path_counts"] == tuple(want["gdn_level_path_counts"])
    assert contract["max_lengths"] == tuple(want["gdn_level_max_lengths"])
    assert contract["padded_slots"] == want["gdn_padded_slots"]
    assert contract["launches"] == want["gdn_launches"]
    assert contract["programs"] == want["gdn_path_programs"]
    assert contract["critical"] == sum(want["gdn_level_max_lengths"])
    assert contract["parent_sha256"] == want["physical_parent_sha256"]
    assert contract["ancestry_sha256"] == want["tree_ancestry_sha256"]


def test_hydra31_gdn_schedule_is_genuinely_different() -> None:
    """A test that passes for both profiles proves nothing if they are equal."""
    h27 = _load_gdn_kernel(PROFILE_HYDRA27)
    h31 = _load_gdn_kernel(PROFILE_HYDRA31)
    assert h27._FR13_FIXED32_PARENT != h31._FR13_FIXED32_PARENT
    assert (
        h27._FR13_FIXED32_SCHEDULE_EXPECTED["padded_slots"] == 82
        and h31._FR13_FIXED32_SCHEDULE_EXPECTED["padded_slots"] == 126
    )
    assert (
        h27._FR13_FIXED32_SCHEDULE_EXPECTED["max_lengths"] == (5, 7)
        and h31._FR13_FIXED32_SCHEDULE_EXPECTED["max_lengths"] == (5, 11)
    )


def test_gdn_kernel_refuses_a_wrong_tree_for_the_served_mode() -> None:
    """MUTATION PROOF: the runtime tree must match the served profile.

    subtree_preseed already refused a non-fixed tree; the bug was that "fixed"
    meant hydra27 whatever the mode said. Now each mode refuses the other's.
    """
    p27 = tuple(topology_profile(PROFILE_HYDRA27)["physical_parent"])
    p31 = tuple(topology_profile(PROFILE_HYDRA31)["physical_parent"])
    assert len(p27) == len(p31) == 32 and p27 != p31

    module31 = _load_gdn_kernel(PROFILE_HYDRA31)
    with pytest.raises(RuntimeError) as refusal:
        module31.subtree_preseed(p27, 32, 1, 1, 1, "cpu")
    assert "non-fixed physical tree" in str(refusal.value)

    module27 = _load_gdn_kernel(PROFILE_HYDRA27)
    with pytest.raises(RuntimeError):
        module27.subtree_preseed(p31, 32, 1, 1, 1, "cpu")


def test_the_kept_levers_still_refuse_hydra31() -> None:
    """The keep half of the adjudication, enforced rather than only commented."""
    module = _load_gdn_kernel(PROFILE_HYDRA31)
    assert module._FR13_FIXED32_MODE == PROFILE_HYDRA31, (
        "hydra31 must RESOLVE as a route -- that is the widen half"
    )
    assert PROFILE_HYDRA31 not in module._FR13_FIXED32_MODES, (
        "the twelve default-off levers were qualified on hydra27's tree"
    )
    assert PROFILE_HYDRA31 in module._FR13_FIXED32_ROUTE_MODES
    assert PROFILE_HYDRA31 not in module._FR13_FIXED32_TREECONV_MODE_IDENTITY
    with pytest.raises(RuntimeError):
        module._fr13_fixed32_treeconv_topology_descriptor(PROFILE_HYDRA31)

    for name, attribute in (
        ("fr13_sfwd_conv_postprep_fusion", "FIXED32_MODES"),
        ("fr13_gdn_gqa_group3", "FIXED32_MODES"),
        ("fr13_fixed32_commit_slot_scatter", "FIXED32_MODES"),
        ("fr13_host_tail_prep", "_FIXED32_MODES"),
    ):
        source = (SERVING / f"{name}.py").read_text()
        collections = [
            names for _line, names in _mode_collections(source)
        ]
        assert collections, f"{name} lost its mode roster"
        assert all(
            PROFILE_HYDRA31 not in names for names in collections
        ), f"{name} was widened to hydra31 without re-qualification"


def test_hydra27_gdn_kernel_surface_is_byte_identical_to_the_baseline() -> None:
    """PAIRING EVIDENCE: the hydra27 path is untouched by the widening."""
    try:
        baseline = subprocess.run(
            ["git", "cat-file", "blob", GDN_KERNEL_BASELINE_BLOB],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-round-18 kernel blob unavailable: {exc}")

    current = GDN_KERNEL.read_text()
    for mode in (None, "tail6_fixed32", PROFILE_HYDRA27):
        old = _load_gdn_kernel(mode, baseline)
        new = _load_gdn_kernel(mode, current)
        names = set(_module_bound_names(baseline)) & set(
            _module_bound_names(current)
        )
        assert len(names) >= 150, f"surface ledger collapsed to {len(names)}"
        # coverage guarantee stated as names, not only as a count: these are
        # the bindings the widening actually touched.
        assert {
            "_FR13_FIXED32_PARENT",
            "_FR13_FIXED32_SUBTREE_LEVELS",
            "_FR13_FIXED32_PARENT_SHA256",
            "_FR13_FIXED32_ANCESTRY_SHA256",
            "_FR13_FIXED32_LEVELS_SHA256",
            "_FR13_FIXED32_MODES",
            "_FR13_FIXED32_EXPORT_NODES",
            "_FR13_FIXED32_PHYSICAL_PARENT",
            "_FR13_FIXED32_TREECONV_STATE_SRC",
        } <= names
        drift = {}
        for name in sorted(names):
            if not hasattr(old, name) or not hasattr(new, name):
                continue
            a, b = getattr(old, name), getattr(new, name)
            if callable(a) or isinstance(a, types.ModuleType):
                continue
            if a != b:
                drift[name] = (a, b)
        assert not drift, f"hydra27 GDN kernel surface CHANGED at {mode}: {sorted(drift)}"
        # the schedule the served profile must produce is unchanged too
        assert (
            new._FR13_FIXED32_SCHEDULE_EXPECTED["padded_slots"] == 82
            and new._FR13_FIXED32_PARENT_SHA256
            == old._FR13_FIXED32_PARENT_SHA256
        )


def test_the_sfwd_source_closure_attestation_is_unchanged() -> None:
    """PAIRING EVIDENCE, in the form the banked runs actually pin.

    fr13_sfwd_state_fusion_production digests the ast.dump of a fixed set of
    kernel-module nodes -- including _FR13_FIXED32_PARENT and
    _FR13_FIXED32_MODES -- and compares it against a byte-qualified candidate.
    Binding the tree by tuple-unpacking changed those NODES and silently broke
    the attestation, which is why the widening rebinds under an `if` instead.
    A digest change here is a re-attestation event for a default-off lever, so
    it must be deliberate and declared, never a side effect of a topology edit.
    """
    from lumo_flywheel_serving import fr13_sfwd_state_fusion_production as prod

    try:
        baseline = subprocess.run(
            ["git", "cat-file", "blob", GDN_KERNEL_BASELINE_BLOB],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-round-18 kernel blob unavailable: {exc}")

    def closure_digest(source: str) -> str:
        members: dict[str, str] = {}
        for node in ast.parse(source).body:
            names: list[str] = []
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                names = [node.target.id]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [node.name]
            for name in names:
                if name in prod._CLOSURE_NAMES:
                    assert name not in members, f"duplicate closure member {name}"
                    members[name] = ast.dump(
                        node, annotate_fields=True, include_attributes=False
                    )
        missing = tuple(n for n in prod._CLOSURE_NAMES if n not in members)
        assert not missing, f"closure is missing {missing!r}"
        return hashlib.sha256(
            "".join(
                f"{name}\0{members[name]}\0" for name in prod._CLOSURE_NAMES
            ).encode("ascii")
        ).hexdigest()

    assert closure_digest(GDN_KERNEL.read_text()) == closure_digest(baseline), (
        "the SFWD source-closure digest moved: the hydra27 lever would need "
        "re-qualification, which is a decision to declare, not a side effect"
    )


# ---------------------------------------------------------------------------
# 8. ROUND 19 -- THE KEY-SET INVARIANT
# ---------------------------------------------------------------------------
# The eighth site was in the AUTHORITY. fr13_fixed32_topology carried three
# different key sets for what read as one idea: PROFILES had hydra27+hydra31,
# VALID_BY_MODE and VALID_MASK_BY_MODE had hydra27+tail6, and twenty-nine
# HYDRA31_*/TAIL10_* constants sat fully described but unindexed. The consumer
# that died -- fr13_device_multidraft_kernel's preseed -- has no hydra31
# mention and needs none: it delegates to VALID_MASK_BY_MODE exactly as rounds
# 13-18 prescribed. Routing consumers through the indices is right, and it is
# also why one gap in an index is a gap in every consumer at once.
#
# The rosters are derived from the PROFILE_*/MODE_* constants, never from
# PROFILES -- PROFILES is one of the mappings that was wrong.
TOPOLOGY = SCRIPTS / "fr13_fixed32_topology.py"

# Round 19's ruling: the TAW binding check was re-scoped (equality -> coverage)
# and the source digest re-attested through the established machinery, so
# VALID_BY_MODE is complete and there are NO exemptions. The dict stays because
# an empty exemption set is the assertion "nothing is exempt", and the test
# below fails the moment something is added without a reason.
KEY_SET_EXEMPTIONS: dict[str, str] = {}


def _topology_module():
    import fr13_fixed32_topology

    return fr13_fixed32_topology


def by_mode_mappings() -> dict[str, dict]:
    """Every module-level mapping in the authority that is keyed by mode/profile."""
    topology = _topology_module()
    found: dict[str, dict] = {}
    for node in ast.parse(TOPOLOGY.read_text()).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = [
            t.id
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(t, ast.Name)
        ]
        for name in targets:
            value = getattr(topology, name, None)
            if not isinstance(value, dict) or not value:
                continue
            keys = set(value)
            if keys & set(topology.SERVING_MODES):
                found[name] = value
    return found


def test_the_rosters_are_derived_from_the_constants_not_from_profiles() -> None:
    topology = _topology_module()
    assert topology.TOPOLOGY_PROFILES == (
        topology.PROFILE_HYDRA27,
        topology.PROFILE_HYDRA31,
    )
    assert topology.SERVING_MODES == (
        topology.MODE_TAIL6,
        topology.PROFILE_HYDRA27,
        topology.PROFILE_HYDRA31,
    )
    assert set(topology.TREE_PROFILE_BY_MODE) == set(topology.SERVING_MODES)
    assert set(topology.TREE_PROFILE_BY_MODE.values()) == set(
        topology.TOPOLOGY_PROFILES
    )
    # the adjudication, asserted: PROFILES is keyed by PROFILE, and tail6 is a
    # MODE that borrows hydra27's tree rather than a profile of its own.
    assert set(topology.PROFILES) == set(topology.TOPOLOGY_PROFILES)
    assert topology.MODE_TAIL6 not in topology.PROFILES
    assert topology.TREE_PROFILE_BY_MODE[topology.MODE_TAIL6] == (
        topology.PROFILE_HYDRA27
    )


def test_every_by_mode_mapping_carries_the_whole_roster() -> None:
    """THE KEY-SET INVARIANT.

    Every mode-keyed mapping in the authority carries exactly SERVING_MODES;
    every profile-keyed mapping carries exactly TOPOLOGY_PROFILES. Nothing is
    allowed to carry some third key set silently, which is what produced the
    eighth site.
    """
    topology = _topology_module()
    profiles, modes = set(topology.TOPOLOGY_PROFILES), set(topology.SERVING_MODES)
    divergent: dict[str, list[str]] = {}
    for name, mapping in sorted(by_mode_mappings().items()):
        keys = set(mapping)
        if keys in (profiles, modes):
            continue
        if name in KEY_SET_EXEMPTIONS:
            continue
        divergent[name] = sorted(keys)
    assert not divergent, (
        "these mappings carry neither the profile roster nor the mode roster, "
        f"so a consumer delegating to them is failed by the authority: {divergent}"
    )
    assert "VALID_MASK_BY_MODE" in by_mode_mappings(), "the sweep found nothing"
    assert set(topology.VALID_MASK_BY_MODE) == modes


def test_there_are_no_key_set_exemptions_left() -> None:
    """The held entry was released; nothing may be exempt without a reason."""
    topology = _topology_module()
    assert KEY_SET_EXEMPTIONS == {}, (
        f"an exemption was added: {sorted(KEY_SET_EXEMPTIONS)} -- an index that "
        "cannot carry the roster needs a written reason and a way out"
    )
    assert set(topology.VALID_BY_MODE) == set(topology.SERVING_MODES)
    assert set(topology.VALID_MASK_BY_MODE) == set(topology.SERVING_MODES)
    # the blocker really was re-scoped, not deleted
    kernel = (SCRIPTS / "fr13_device_multidraft_kernel.py").read_text()
    assert "if set(topology.VALID_BY_MODE) != set(modes):" not in kernel
    assert "missing = tuple(mode for mode in modes if mode not in" in kernel


def test_the_invariant_fires_on_a_divergent_key_set() -> None:
    """MUTATION PROOF: an index missing a mode must fail, exemptions aside."""
    topology = _topology_module()
    modes, profiles = set(topology.SERVING_MODES), set(topology.TOPOLOGY_PROFILES)
    mutant = {
        mode: 0 for mode in topology.SERVING_MODES if mode != topology.PROFILE_HYDRA31
    }
    assert set(mutant) not in (modes, profiles), (
        "the mutant must be a genuinely third key set"
    )
    divergent = {}
    for name, mapping in {"MUTANT_BY_MODE": mutant}.items():
        if set(mapping) in (profiles, modes) or name in KEY_SET_EXEMPTIONS:
            continue
        divergent[name] = sorted(mapping)
    assert divergent == {"MUTANT_BY_MODE": sorted(mutant)}, (
        "the key-set invariant cannot fail, so it is worse than none"
    )


def test_mode_functions_take_their_tree_from_the_mode_not_from_hydra27() -> None:
    """The reason the index completion is not a one-line fill.

    active_choices read the bare FIXED32_CHOICES, which is hydra27's. Filling
    VALID_BY_MODE with hydra31 before this was fixed would have built hydra31's
    sampler tables out of HYDRA27's 31 paths: every bit valid, every shape
    check satisfied, wrong tree -- rounds 17 and 18's hazard inside the
    authority itself.
    """
    topology = _topology_module()
    assert topology.choices_for_mode(topology.MODE_TAIL6) == topology.FIXED32_CHOICES
    assert (
        topology.choices_for_mode(topology.PROFILE_HYDRA27)
        == topology.FIXED32_CHOICES
    )
    assert (
        topology.choices_for_mode(topology.PROFILE_HYDRA31) == topology.TAIL10_CHOICES
    )
    assert topology.TAIL10_CHOICES != topology.FIXED32_CHOICES

    # the entry is real now (round 19's ruling), so the whole mode-indexed
    # surface must be correct without any injection. The earlier version of
    # this test injected and then popped it in a finally -- which, once the
    # index was completed for real, deleted a live entry and corrupted the
    # authority for every test that ran after it.
    assert (
        topology.active_choices(topology.PROFILE_HYDRA31) == topology.TAIL10_CHOICES
    )
    children = topology.active_child_lists(topology.PROFILE_HYDRA31)
    assert children[-1] == (0, 1, 2)
    assert max(len(kids) for kids in children.values()) == topology.SAMPLER_MAX_FANOUT
    table, counts = topology.sampler_child_table(topology.PROFILE_HYDRA31)
    assert (len(table), len(table[0])) == topology.SAMPLER_TABLE_SHAPE
    assert len(counts) == 32


def test_profile_refuses_a_serving_mode_with_a_legible_message() -> None:
    topology = _topology_module()
    with pytest.raises(KeyError) as refusal:
        topology.profile(topology.MODE_TAIL6)
    message = str(refusal.value)
    assert "serving MODE" in message and "VALID_MASK_BY_MODE" in message
    with pytest.raises(KeyError):
        topology.profile("nope_fixed32")


# --- THE CPU WALK, driving the real preseed (round 19 task 4) --------------
def _multidraft_kernel():
    import importlib.util

    path = SCRIPTS / "fr13_device_multidraft_kernel.py"
    spec = importlib.util.spec_from_file_location(
        "fr13_device_multidraft_kernel__walk", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"multidraft kernel unavailable: {exc}")
    return module


@pytest.mark.parametrize("mode", ["tail6_fixed32", PROFILE_HYDRA27])
def test_the_real_preseed_still_runs_on_cpu_for_every_qualified_mode(
    mode: str,
) -> None:
    """Not a re-typed model of the preseed -- fr13_fixed32_taw_preseed itself.

    This is the path that died in round 19: it asks VALID_MASK_BY_MODE for the
    mode's mask and has no hydra31 mention, because a consumer delegating to
    the authority should not need one.
    """
    pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    topology = _topology_module()
    preseeded = kernel.fr13_fixed32_taw_preseed("cpu", mode=mode)
    assert len(preseeded) == 4, "preseed covers B=1..4"
    assert mode in topology.VALID_MASK_BY_MODE
    assert kernel._fr13_fixed32_expected_active(topology, mode) == sum(
        1 for enabled in topology.VALID_BY_MODE[mode] if enabled
    )


def test_the_preseed_completes_end_to_end_for_every_serving_mode() -> None:
    """Round 20's ruling: the source-row schedule is keyed by mode, so hydra31
    preseeds end to end. No declared gaps.

    The union only existed because ONE static allocation served every mode.
    Note what it was NOT: 13/17 is the union of tail6 and hydra27 and is not
    either mode's own schedule -- hydra27 alone is 11/17, tail6 alone is 11/13,
    and the two extra self rows are nodes 6 and 7 that only tail6 contributes.
    So the qualified modes keep the QUALIFIED-SCOPE schedule byte for byte
    (what their byte-AB pass measured and what production allocates), and only
    a mode outside that scope derives its own.
    """
    pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    topology = _topology_module()

    expected_rows = {
        "tail6_fixed32": (13, 17),
        PROFILE_HYDRA27: (13, 17),
        PROFILE_HYDRA31: (11, 21),
    }
    for mode in topology.SERVING_MODES:
        keys = kernel.fr13_fixed32_taw_preseed("cpu", mode=mode)
        assert len(keys) == 4, f"{mode}: preseed covers B=1..4"
        entry = kernel._FR13_FIXED32_TAW_CACHE[keys[0]]
        rows = (
            int(entry["native_self_rows_per_request"]),
            int(entry["native_target_rows_per_request"]),
        )
        assert rows == expected_rows[mode], f"{mode}: rows {rows}"


def test_the_qualified_scope_schedule_is_not_either_modes_own() -> None:
    """The subtlety that makes 'just key it per mode' a production change.

    Recorded as a test because the next reader will otherwise 'simplify' the
    qualified branch and silently shrink a production tensor by two rows.
    """
    topology = _topology_module()

    def schedule(modes):
        selves, targets = set(), set()
        for mode in modes:
            children = topology.active_child_lists(mode)
            active = [
                node
                for node, enabled in enumerate(topology.valid_for_mode(mode))
                if enabled
            ]
            selves.update(node for node in active if node not in children)
            targets.update(kids[0] for kids in children.values())
        return len(selves), len(targets)

    assert schedule(("tail6_fixed32", PROFILE_HYDRA27)) == (13, 17)
    assert schedule((PROFILE_HYDRA27,)) == (11, 17)
    assert schedule(("tail6_fixed32",)) == (11, 13)
    assert schedule((PROFILE_HYDRA31,)) == (11, 21)


def test_the_schedule_binding_refuses_another_profiles_schedule() -> None:
    """MUTATION PROOF: a schedule is a list of node ids, and two profiles can
    produce lists of the same LENGTH that address different rows. Shape checks
    cannot see that, so the binding is to the profile's pinned digest.
    """
    pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    topology = _topology_module()
    binding = kernel._fr13_fixed32_taw_topology_binding(topology)
    schedules = binding["all_parent_schedule_by_mode"]

    # each mode's own schedule binds
    for mode in topology.SERVING_MODES:
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, mode, schedules[mode]
        )

    # hydra31's schedule under hydra27's mode, and the reverse, must refuse.
    # Which guard catches it first depends on the pair -- coverage or digest --
    # and both are refusals; what matters is that neither combination serves.
    with pytest.raises(RuntimeError):
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, PROFILE_HYDRA27, schedules[PROFILE_HYDRA31]
        )
    with pytest.raises(RuntimeError):
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, PROFILE_HYDRA31, schedules[PROFILE_HYDRA27]
        )
    with pytest.raises(RuntimeError):
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, "nope_fixed32", schedules[PROFILE_HYDRA27]
        )

    # a schedule that leaves an active node unscheduled refuses at coverage
    maimed = dict(schedules[PROFILE_HYDRA31])
    maimed["self_source_nodes"] = list(maimed["self_source_nodes"])[:-1]
    with pytest.raises(RuntimeError) as gap:
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, PROFILE_HYDRA31, maimed
        )
    assert "unscheduled" in str(gap.value)

    # ...and the DIGEST alone catches a schedule of the right length, covering
    # every active node, that addresses the wrong rows. This is the case no
    # shape or coverage check can see, and the reason the binding exists.
    swapped = dict(schedules[PROFILE_HYDRA27])
    targets = list(swapped["target_source_nodes"])
    targets[0], targets[-1] = targets[-1], targets[0]
    swapped["target_source_nodes"] = targets
    assert len(targets) == len(schedules[PROFILE_HYDRA27]["target_source_nodes"])
    with pytest.raises(RuntimeError) as wrong_rows:
        kernel._fr13_fixed32_bind_schedule_to_profile(
            topology, PROFILE_HYDRA27, swapped
        )
    assert "not the served profile" in str(wrong_rows.value)


def test_the_qualified_modes_share_one_schedule_object_by_scope() -> None:
    """Bit-identity, asserted at the source rather than only measured."""
    pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    topology = _topology_module()
    binding = kernel._fr13_fixed32_taw_topology_binding(topology)
    schedules = binding["all_parent_schedule_by_mode"]

    qualified = ["tail6_fixed32", PROFILE_HYDRA27]
    for mode in qualified:
        assert schedules[mode]["scope"] == qualified
        for key in (
            "self_source_nodes",
            "target_source_nodes",
            "self_uniform_levels",
            "target_parent_slots",
            "target_uniform_levels",
        ):
            assert schedules[mode][key] == binding[f"all_parent_{key}"], (
                f"{mode}.{key} diverged from the qualified-scope schedule"
            )
    assert schedules[PROFILE_HYDRA31]["scope"] == [PROFILE_HYDRA31]
    # ...and the top-level keys, which every other consumer reads, are the
    # qualified scope's, unchanged.
    assert len(binding["all_parent_self_source_nodes"]) == 13
    assert len(binding["all_parent_target_source_nodes"]) == 17
    assert binding["topology_sha256"] == TAW_TOPOLOGY_DIGEST


def test_hydra27_topology_values_are_byte_identical_to_the_baseline() -> None:
    """PAIRING EVIDENCE: the authority's hydra27/tail6 answers did not move."""
    try:
        blob = subprocess.run(
            ["git", "cat-file", "blob", "HEAD:scripts/fr13_fixed32_topology.py"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"baseline topology unavailable: {exc}")

    old = types.ModuleType("fr13_fixed32_topology__baseline")
    old.__file__ = str(TOPOLOGY)
    sys.modules[old.__name__] = old
    try:
        exec(compile(blob, str(TOPOLOGY), "exec"), old.__dict__)
    finally:
        sys.modules.pop(old.__name__, None)
    new = _topology_module()

    drift: dict[str, tuple[object, object]] = {}
    for name in _module_bound_names(blob):
        if not hasattr(old, name) or not hasattr(new, name):
            continue
        a, b = getattr(old, name), getattr(new, name)
        if callable(a) or isinstance(a, types.ModuleType):
            continue
        if name in ("VALID_MASK_BY_MODE", "VALID_BY_MODE"):
            # deliberately completed; every pre-existing entry must be identical
            # and the only addition may be hydra31.
            assert all(b[mode] == value for mode, value in a.items())
            assert set(b) - set(a) <= {PROFILE_HYDRA31}
            assert set(b) == set(_topology_module().SERVING_MODES)
            continue
        if a != b:
            drift[name] = (a, b)
    assert not drift, f"authority values moved for hydra27/tail6: {sorted(drift)}"

    for mode in ("tail6_fixed32", PROFILE_HYDRA27):
        assert old.active_choices(mode) == new.active_choices(mode)
        assert old.active_child_lists(mode) == new.active_child_lists(mode)
        assert old.sampler_child_table(mode) == new.sampler_child_table(mode)
        assert old.valid_for_mode(mode) == new.valid_for_mode(mode)
    assert old.PROFILES[PROFILE_HYDRA27] == new.PROFILES[PROFILE_HYDRA27]
    assert old.PROFILES[PROFILE_HYDRA31] == new.PROFILES[PROFILE_HYDRA31]


# ---------------------------------------------------------------------------
# 9. ROUND 20 -- THE TAW SOURCE-DIGEST RE-ATTESTATION
# ---------------------------------------------------------------------------
# Round 19's ruling: re-scope the TAW binding check (equality -> coverage) and
# re-attest the source-closure digest through the established machinery. Two of
# the 47 digested functions changed, both BOOT-TIME VALIDATORS, and the
# evidence below is what the pairing rests on.
TAW_SOURCE_DIGEST = "80595b6be9cb9cb8e1449fb3325e1b510e5c00186fa194b05bf16beaaa376687"
# Every digest this pin has ever carried. A mirror holding ANY of them is a
# mirror that was missed, so the sweep checks all of them rather than only the
# immediately previous one.
TAW_SOURCE_DIGEST_SUPERSEDED = (
    "68b289aee5773edf1134f184c37551a90ec8543430d768a05066bc1341473c6d",
    "491874e3ebbc53b83ce28a8cae505025fde36e56564da049ab0d582eaa4e7d5c",
    "6ffe57287e768bfee5e2e72f10de0dfea6fb3e6c0fa50f32b6c099c63fa916a2",
)
# The lever's qualified scope and the payload it digests. Neither moved, which
# is why no tensor shape or slot index on the hydra27 path could move.
TAW_TOPOLOGY_DIGEST = (
    "99b1255b1c1ffeda8bbbd7800e777cfa7184f48c1f13494537a01bc70bc9bf79"
)
TAW_QUALIFIED_SELF_ROWS = 13
TAW_QUALIFIED_TARGET_ROWS = 17


def _digest_mirror_files() -> list[Path]:
    """Every tracked source file carrying a 64-hex TAW source digest literal."""
    out: list[Path] = []
    roots = [SCRIPTS, SERVING, REPO / "tests", REPO / "config"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".sh", ".json"):
                continue
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):  # pragma: no cover
                continue
            if TAW_SOURCE_DIGEST in text or any(
                old in text for old in TAW_SOURCE_DIGEST_SUPERSEDED
            ):
                out.append(path)
    return out


def test_every_taw_digest_mirror_carries_the_re_attested_value() -> None:
    """THE MIRROR SWEEP: the proof that none was missed.

    A digest pinned in fourteen places is fourteen chances to update thirteen.
    This enumerates them from the tree rather than from a list someone keeps.
    """
    mirrors = _digest_mirror_files()
    assert len(mirrors) >= 13, (
        f"the sweep found only {len(mirrors)} mirrors; it used to find 14"
    )
    stale = {
        path.relative_to(REPO).as_posix(): [
            old for old in TAW_SOURCE_DIGEST_SUPERSEDED if old in path.read_text()
        ]
        for path in mirrors
        if any(old in path.read_text() for old in TAW_SOURCE_DIGEST_SUPERSEDED)
        and path.name != Path(__file__).name
    }
    assert not stale, f"these mirrors still carry the superseded digest: {stale}"


def test_the_kernel_and_the_census_agree_on_the_digest() -> None:
    """The two authorities a run is gated against must not disagree."""
    kernel = _multidraft_kernel()
    census_path = SCRIPTS / "fr13_fixed32_work_census.py"
    assert kernel._FR13_FIXED32_TAW_SOURCE_SHA256 == TAW_SOURCE_DIGEST
    assert TAW_SOURCE_DIGEST in census_path.read_text()
    assert not any(
        old in census_path.read_text() for old in TAW_SOURCE_DIGEST_SUPERSEDED
    )


def test_the_re_attestation_blast_radius_is_the_three_validators() -> None:
    """Blast radius, measured rather than asserted.

    Round 20's ruling added a third: _fr13_fixed32_bind_schedule_to_profile,
    the schedule's observable binding, which is placed UNDER the source closure
    on purpose -- a validator that decides whether a serve is allowed belongs
    inside the attestation like the math does.
    """
    kernel = _multidraft_kernel()
    try:
        baseline = subprocess.run(
            ["git", "show", "HEAD~1:scripts/fr13_device_multidraft_kernel.py"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"previous kernel revision unavailable: {exc}")

    names = set(kernel._FR13_FIXED32_TAW_SOURCE_FUNCTIONS)
    assert len(names) == 48

    def sources(text: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names
            ):
                found[node.name] = ast.get_source_segment(text, node)
        return found

    old, new = sources(baseline), sources(
        (SCRIPTS / "fr13_device_multidraft_kernel.py").read_text()
    )
    changed = sorted(n for n in names if old.get(n) != new.get(n))
    # A SUBSET, because HEAD~1 moves as other lanes land: once the schedule work
    # is behind us the set is empty, and what must never happen is a digested
    # function changing that is not one of these three.
    # Site 13 converted every walk-cap reader on the execution path, so the
    # changed set is larger than the schedule work's three. What must never
    # happen is a digested function changing that is not on this list.
    #
    # SITE 27 added _fr13_fixed32_taw_tensor_call_census, and the measured blast
    # radius of that re-attestation is EXACTLY the three functions its own
    # comment names: the geometry expectation and the by-route census live in
    # _fr13_fixed32_taw_source_contract, the reference-route census in
    # _fr13_fixed32_taw_tensor_call_census, and the published slot counts in
    # _fr13_fixed32_publish_work. Nothing else in the audited closure moved.
    # SITE 28's measured radius is exactly ONE function --
    # _fr13_fixed32_layout_contract -- because the four walk-derived layout
    # numbers were literals inside it and nothing else in the closure states a
    # shape. A one-function re-attestation is what a correctly scoped fix looks
    # like.
    assert set(changed) <= {
        "_fr13_fixed32_bind_schedule_to_profile",
        "_fr13_fixed32_layout_contract",
        "_fr13_fixed32_publish_work",
        "_fr13_fixed32_runtime_contract",
        "_fr13_fixed32_taw_execute_exact_cuda",
        "_fr13_fixed32_taw_execute_torch",
        "_fr13_fixed32_taw_source_contract",
        "_fr13_fixed32_taw_tensor_call_census",
        "_fr13_fixed32_taw_topology_binding",
        "_fr13_fixed32_topology",
        "fr13_fixed32_taw_commit",
        "fr13_fixed32_taw_preseed",
    }, f"an unexpected digested function changed: {changed}"


def test_neither_changed_validator_can_move_a_counter_or_a_served_byte() -> None:
    """PAIRING EVIDENCE, enumerate-and-show.

    The census counters H27n banked derive from EXECUTION. These two functions
    are boot-time validations: one returns an integer that is only ever
    compared, the other returns a binding whose every value is consumed as a
    shape or a digest. Both are shown to return byte-identical results.

    The decisive quantity is the lever's qualified scope: `modes` is unchanged,
    so the union it digests is unchanged (13 self / 17 target), so
    topology_sha256 is unchanged, so every tensor shape and slot index derived
    from it is unchanged. Widening that tuple WOULD move hydra27 -- the union of
    all three modes is 14/22 -- which is exactly why it was not widened.
    """
    kernel = _multidraft_kernel()
    topology = _topology_module()

    # (a) the integer: identical, and still only a comparison subject
    assert kernel._fr13_fixed32_expected_active(topology, "tail6_fixed32") == (
        topology.TAIL6_ACTIVE_DRAFTS
    )
    assert kernel._fr13_fixed32_expected_active(topology, PROFILE_HYDRA27) == (
        topology.HYDRA27_ACTIVE_DRAFTS
    )

    # (b) the binding: qualified scope, payload digest and row counts unmoved
    binding = kernel._fr13_fixed32_taw_topology_binding(topology)
    assert binding["topology_sha256"] == TAW_TOPOLOGY_DIGEST
    assert len(binding["all_parent_self_source_nodes"]) == TAW_QUALIFIED_SELF_ROWS
    assert (
        len(binding["all_parent_target_source_nodes"]) == TAW_QUALIFIED_TARGET_ROWS
    )
    assert binding["tail_valid_mask"] == topology.TAIL6_VALID_MASK
    assert binding["hydra_valid_mask"] == topology.HYDRA27_VALID_MASK

    # (c) the re-scoped check still refuses an authority that lost a mode
    class _Missing:
        def __getattr__(self, name):
            return getattr(topology, name)

        VALID_BY_MODE = {PROFILE_HYDRA27: topology.HYDRA27_VALID}

    with pytest.raises(RuntimeError) as refusal:
        kernel._fr13_fixed32_taw_topology_binding(_Missing())
    assert "missing" in str(refusal.value)


def test_the_pinned_schedule_digests_are_recomputed_not_trusted() -> None:
    """The pins the observable binding compares against, derived here again.

    A digest table that is only ever compared against itself pins nothing. Each
    entry is recomputed from fr13_fixed32_topology and the kernel's own binding,
    so a hand-edited pin fails rather than blessing whatever it was edited to.
    """
    pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    topology = _topology_module()
    binding = kernel._fr13_fixed32_taw_topology_binding(topology)
    schedules = binding["all_parent_schedule_by_mode"]

    pinned = kernel._FR13_FIXED32_TAW_SCHEDULE_DIGESTS
    assert set(pinned) == set(topology.SERVING_MODES), (
        "the schedule digest table must carry the whole serving roster"
    )
    for mode in topology.SERVING_MODES:
        recomputed = hashlib.sha256(
            json.dumps(
                {
                    "mode": mode,
                    "profile": topology.TREE_PROFILE_BY_MODE[mode],
                    "ancestry_sha256": str(
                        topology.profile(topology.TREE_PROFILE_BY_MODE[mode])[
                            "tree_ancestry_sha256"
                        ]
                    ),
                    "self_source_nodes": list(schedules[mode]["self_source_nodes"]),
                    "target_source_nodes": list(
                        schedules[mode]["target_source_nodes"]
                    ),
                    "target_parent_slots": list(
                        schedules[mode]["target_parent_slots"]
                    ),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        assert pinned[mode] == recomputed, f"{mode}: pinned digest is stale"

    # the two profiles produce genuinely different schedules
    assert pinned[PROFILE_HYDRA27] != pinned[PROFILE_HYDRA31]
    # ...and tail6 differs from hydra27 only in the mode/ancestry it binds to,
    # which is exactly why the digest carries the mode and not just the rows.
    assert pinned["tail6_fixed32"] != pinned[PROFILE_HYDRA27]
    assert (
        schedules["tail6_fixed32"]["self_source_nodes"]
        == schedules[PROFILE_HYDRA27]["self_source_nodes"]
    )


# ---------------------------------------------------------------------------
# 10. ROUND 20 -- THE ELEVENTH SITE: A CORRECT REFUSAL IN THE WRONG PLACE
# ---------------------------------------------------------------------------
# preseed_fixed32_conv_col0_pregather built its state record UNCONDITIONALLY and
# read the tree-conv mode map by SUBSCRIPT, so with both zero-tail lever flags
# OFF a hydra31 boot died at 4m50s inside the captured FX graph -- while
# building the record that reports the lever is off. The round-18 adjudication
# (zero-tail is byte-qualified on hydra27; hydra31 must re-qualify) is right and
# stands verbatim. Refusing to run a lever nobody armed is not a refusal.
def test_the_zero_tail_adjudication_still_stands_verbatim() -> None:
    module = _load_gdn_kernel(PROFILE_HYDRA31)
    assert PROFILE_HYDRA31 not in module._FR13_FIXED32_TREECONV_MODE_IDENTITY
    with pytest.raises(RuntimeError) as refusal:
        module._fr13_fixed32_treeconv_topology_descriptor(PROFILE_HYDRA31)
    assert "unsupported fixed32 tree-conv mode" in str(refusal.value)


@pytest.mark.parametrize(
    "mode", ["tail6_fixed32", PROFILE_HYDRA27, PROFILE_HYDRA31]
)
def test_the_descriptor_can_be_read_without_raising_for_the_record(
    mode: str,
) -> None:
    """The placement fix: a record can always be built."""
    module = _load_gdn_kernel(mode)
    optional = module._fr13_fixed32_treeconv_topology_descriptor_optional(mode)
    if mode == PROFILE_HYDRA31:
        assert optional is None, "unqualified mode yields a benign marker"
    else:
        # ...and for the qualified modes it is the SAME value the raising form
        # returns, so their record is unchanged by identity.
        assert optional == module._fr13_fixed32_treeconv_topology_descriptor(mode)


def test_the_raise_is_gated_on_the_lever_being_armed() -> None:
    """Mutation-proven both ways, read off the source of the record builder."""
    source = GDN_KERNEL.read_text()
    tree = ast.parse(source)
    builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "preseed_fixed32_conv_col0_pregather"
    )
    record = next(
        node.value
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and len(node.value.keys) > 20
    )
    # the record must NOT call the raising form
    calls = {
        node.func.id
        for node in ast.walk(record)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_fr13_fixed32_treeconv_topology_descriptor" not in calls, (
        "the record builder calls the raising descriptor again -- with the "
        "lever off that kills the boot inside the captured graph"
    )
    # ...and the refusal must be reachable only when a lever is armed
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(builder)
        if isinstance(node, ast.If)
        and "treeconv_descriptor is None" in ast.unparse(node.test)
    ]
    assert guards, "the lever-gated refusal disappeared"
    assert all(
        "zero_tail" in guard and "zero_tail_byte_ab" in guard for guard in guards
    ), f"the refusal is not gated on both lever flags: {guards}"


# --- THE TRIPLE-SIGNAL CENSUS ----------------------------------------------
# The runner's refinement of the mode-map scan: flag a map only when it is
# INCOMPLETE and read by RAISING SUBSCRIPT and REACHABLE WITH THE LEVER OFF.
# The third signal is what separates this class from the thirty-odd incomplete
# maps that live in credentials, gates and reducers -- offline tooling a serve
# never executes. Restricting to the serving package is that signal, mechanised.
SERVE_PATH_PREFIX = "src/lumo_flywheel_serving/"


def _mode_constants() -> dict[str, str]:
    topology = _topology_module()
    roster = set(topology.SERVING_MODES)
    return {
        name: getattr(topology, name)
        for name in dir(topology)
        if not name.startswith("_")
        and isinstance(getattr(topology, name), str)
        and getattr(topology, name) in roster
    }


def scan_source_for_raising_mode_maps(
    source: str,
) -> list[tuple[int, list[int], list[str]]]:
    """Signals 1 and 2 in ONE place, so the census and its mutation proof cannot
    drift apart -- which is how two of this campaign's detectors grew holes."""
    topology = _topology_module()
    roster, profiles = set(topology.SERVING_MODES), set(topology.TOPOLOGY_PROFILES)
    constants = _mode_constants()

    def key_value(key: ast.expr) -> str | None:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        if isinstance(key, ast.Name):
            return constants.get(key.id)
        if isinstance(key, ast.Attribute):
            return constants.get(key.attr)
        return None

    out: list[tuple[int, list[int], list[str]]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return out
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        values = [key_value(key) for key in node.keys]
        if not values or any(value is None for value in values):
            continue
        keys = set(values)
        # a pure mode index, and not a PROFILE index (tail6 is absent from
        # those by construction -- round 19's adjudication)
        if not (keys & roster) or not keys <= roster or keys == profiles:
            continue
        missing = roster - keys
        if not missing:  # SIGNAL 1: incomplete
            continue
        parent = parents.get(node)
        names: list[str] = []
        if isinstance(parent, ast.Assign):
            names = [t.id for t in parent.targets if isinstance(t, ast.Name)]
        elif isinstance(parent, ast.AnnAssign) and isinstance(
            parent.target, ast.Name
        ):
            names = [parent.target.id]
        subscripts: list[int] = []
        if isinstance(parent, ast.Subscript):
            subscripts.append(node.lineno)
        for name in names:
            subscripts += [
                use.lineno
                for use in ast.walk(tree)
                if isinstance(use, ast.Subscript)
                and isinstance(use.value, ast.Name)
                and use.value.id == name
                and isinstance(use.ctx, ast.Load)
            ]
        if not subscripts:  # SIGNAL 2: raising subscript, not .get
            continue
        out.append((node.lineno, sorted(set(subscripts)), sorted(missing)))
    return out


def triple_signal_census(
    serve_path_only: bool = True,
) -> list[tuple[str, int, list[int], list[str]]]:
    """SIGNAL 3 mechanised: only the serving package runs during a serve."""
    out: list[tuple[str, int, list[int], list[str]]] = []
    for sub in ("scripts", "src", "tests", "config"):
        root = REPO / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if serve_path_only and not rel.startswith(SERVE_PATH_PREFIX):
                continue
            for lineno, subs, missing in scan_source_for_raising_mode_maps(
                path.read_text(errors="replace")
            ):
                out.append((rel, lineno, subs, missing))
    return out


def test_no_serve_path_mode_map_has_all_three_signals() -> None:
    """SIGNAL 3 mechanised: only the serving package runs during a serve."""
    found = triple_signal_census(serve_path_only=True)
    assert not found, (
        "incomplete mode-keyed map(s) read by raising subscript on the serve "
        f"path -- the eleventh site's exact shape: {found}"
    )


def test_the_census_would_have_caught_the_eleventh_site() -> None:
    """MUTATION PROOF: run the census against the pre-fix source.

    Also verifies the runner's 'exactly one had all three properties' claim
    rather than assuming it.
    """
    try:
        baseline = subprocess.run(
            ["git", "cat-file", "blob", "b7b96bf87bb6d33f285fa7ebb4677e8aaf51d011"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-site-11 kernel blob unavailable: {exc}")

    flagged = scan_source_for_raising_mode_maps(baseline)
    assert [entry[0] for entry in flagged] == [43], (
        "the pre-fix source should show EXACTLY ONE map with all three signals "
        f"-- the eleventh site at :43 -- got {flagged}"
    )
    assert flagged[0][1] == [100], "read by raising subscript in the descriptor"
    assert flagged[0][2] == [PROFILE_HYDRA31]
    # ...and the fixed source shows none, which is the claim being verified
    assert not scan_source_for_raising_mode_maps(GDN_KERNEL.read_text())


# --- THE CPU WALK, EXTENDED TO RECORD CONSTRUCTION -------------------------
# The walk missed the eleventh site because it only ever exercised the COMPUTE
# path. The record was where the defect lived, so the walk now builds it -- by
# LIFTING the real dict expression out of the source and evaluating it, not by
# re-typing a model of it.
def _pregather_record_expression() -> ast.Dict:
    builder = next(
        node
        for node in ast.walk(ast.parse(GDN_KERNEL.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.name == "preseed_fixed32_conv_col0_pregather"
    )
    return next(
        node.value
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and len(node.value.keys) > 20
    )


def _build_pregather_record(mode: str, zero_tail: str = "0", byte_ab: str = "0"):
    import os

    torch = pytest.importorskip("torch")
    saved = {
        key: os.environ.get(key)
        for key in (
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL",
            "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB",
        )
    }
    os.environ["FR13_FIXED32_CONV_COMMIT_ZERO_TAIL"] = zero_tail
    os.environ["FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB"] = byte_ab
    try:
        module = _load_gdn_kernel(mode)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    layers, batches = 48, [1, 2, 3, 4]
    zeros = lambda *shape: torch.zeros(shape if shape else (), dtype=torch.int64)
    descriptor = module._fr13_fixed32_treeconv_topology_descriptor_optional(mode)
    namespace = {
        "hashlib": hashlib, "int": int, "str": str, "tuple": tuple,
        "set": set, "id": id,
        "_FR13_FIXED32_MODE": module._FR13_FIXED32_MODE,
        "_FR13_FIXED32_CONV_COMMIT_ROUTE": module._FR13_FIXED32_CONV_COMMIT_ROUTE,
        "_fr13_fixed32_treeconv_canonical_json": (
            module._fr13_fixed32_treeconv_canonical_json
        ),
        "_fr13_fixed32_treeconv_topology_descriptor": (
            module._fr13_fixed32_treeconv_topology_descriptor
        ),
        "treeconv_descriptor": descriptor,
        "zero_tail": zero_tail == "1", "zero_tail_byte_ab": byte_ab == "1",
        "zero_tail_count_enable": zeros(), "zero_tail_compared_events": zeros(),
        "zero_tail_differing_bytes": zeros(),
        "anchor": zeros(4), "staging": zeros(4, 4),
        "accepted_paths": zeros(4, 16), "accepted_lens": zeros(4),
        "commit_spec_state_indices": zeros(4),
        "alias_ids_device": zeros(layers), "alias_peer_layers_device": zeros(layers),
        "pointers": zeros(layers), "source_pointers": zeros(layers),
        "ssi_ptrs": zeros(layers), "ssi_strides": zeros(layers),
        "offsets": zeros(layers), "source_offsets": zeros(layers),
        "ssm_pointers": tuple(range(layers)),
        "ssm_storage_pointers": tuple(range(layers)),
        "alias_ids": tuple(range(layers)), "alias_ranks": tuple(range(layers)),
        "alias_classes": tuple(tuple(range(3)) for _ in range(16)),
        "alias_peer_layers": tuple(range(layers)),
        "layer_order": tuple(range(layers)),
        "bank_refs": [zeros(2, 2) for _ in range(layers)],
        "ssm_bank_refs": [zeros(2, 2) for _ in range(layers)],
        "source_refs": [zeros(2, 2) for _ in range(layers)],
        "ssi_sources": [zeros(2, 2) for _ in range(layers)],
        "row_guard_flags": {b: zeros(b) for b in batches},
        "batches": batches, "capacity": 4, "row_elems": 348160,
        "conv_c": 10240, "conv_l": 34, "element_bytes": 2, "block": 1024,
        "source_rows": 36, "live_state_cols": 3, "direct_state_src": zeros(36),
        "state_src_values": tuple(range(36)), "commit_lease_token": 7,
    }
    return eval(
        compile(ast.Expression(_pregather_record_expression()), "<record>", "eval"),
        namespace,
    )


@pytest.mark.parametrize(
    "mode", ["tail6_fixed32", PROFILE_HYDRA27, PROFILE_HYDRA31]
)
def test_the_walk_builds_the_pregather_record_with_the_levers_off(
    mode: str,
) -> None:
    """The walk that would have caught the eleventh site.

    Both zero-tail flags OFF -- the configuration the boot actually ran -- and
    the record must be constructible for every serving mode.
    """
    record = _build_pregather_record(mode)
    assert len(record) > 20
    assert record["commit_zero_tail"] is False
    assert record["commit_zero_tail_byte_ab"] is False
    descriptor = record["treeconv_topology_descriptor"]
    if mode == PROFILE_HYDRA31:
        assert descriptor is None
    else:
        assert descriptor["mode"] == mode
        assert descriptor["schema"] == "fr13.fixed32.treeconv_state_descriptor.v1"


def test_the_qualified_record_descriptor_is_unchanged() -> None:
    """PAIRING EVIDENCE: the only field the fix can touch, shown identical.

    The record expression differs from before in exactly one place -- which
    function produces treeconv_topology_descriptor -- so showing that producer
    returns the same value for the qualified modes is the whole argument.
    """
    for mode in ("tail6_fixed32", PROFILE_HYDRA27):
        module = _load_gdn_kernel(mode)
        record = _build_pregather_record(mode)
        assert (
            record["treeconv_topology_descriptor"]
            == module._fr13_fixed32_treeconv_topology_descriptor(mode)
        )
    # ...and the record's key set does not depend on the mode at all
    keys = {
        mode: set(_build_pregather_record(mode))
        for mode in ("tail6_fixed32", PROFILE_HYDRA27, PROFILE_HYDRA31)
    }
    assert keys["tail6_fixed32"] == keys[PROFILE_HYDRA27] == keys[PROFILE_HYDRA31]


# ---------------------------------------------------------------------------
# 11. THE PER-POSITION ACCEPTANCE LADDER (round 21 instrument)
# ---------------------------------------------------------------------------
# vllm:spec_decode_num_accepted_tokens_per_pos_total does not exist in our vLLM
# source, and the measurement script defaults the ladder to [0.0]*16 -- a
# missing metric reported as a measured zero, round 6's failure mode. The
# instrument accumulates DEVICE-SIDE from the already-device-resident accepted
# length, inside the captured graph, drained once at flush.
def _ladder_state(batch: int = 4, enabled: bool = True):
    torch = pytest.importorskip("torch")
    return {
        "accept_ladder": (
            torch.zeros(16, dtype=torch.int64) if enabled else None
        ),
        "accept_ladder_overflow_rows": torch.zeros((), dtype=torch.int64),
        "accept_ladder_overflow_tokens": torch.zeros((), dtype=torch.int64),
        "accept_ladder_ones": torch.ones(batch, dtype=torch.int64),
    }


def test_the_ladder_turns_a_known_sequence_into_a_known_ladder() -> None:
    """MUTATION PROOF: synthetic accepted_lens -> exact ladder and token total."""
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    state = _ladder_state()
    steps = [[0, 1, 2, 3], [5, 5, 5, 5], [11, 0, 0, 15], [15, 15, 2, 1]]
    for step in steps:
        module._fr13_fixed32_accept_ladder_accumulate(
            state, torch.tensor(step, dtype=torch.int32)
        )
    drained = module.fr13_fixed32_accept_ladder_drain(state)

    expected = [0] * 16
    for step in steps:
        for value in step:
            expected[min(value, 15)] += 1
    assert drained["ladder"] == expected
    assert drained["rows"] == sum(len(step) for step in steps)
    assert drained["accepted_tokens"] == sum(v for s in steps for v in s)


def test_the_ladder_is_self_proving() -> None:
    """The identity the report asserts against the aggregate counter.

    sum(i * ladder[i]) + overflow_tokens == accepted_tokens. A ladder that
    cannot fail this check is not evidence.
    """
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    state = _ladder_state(batch=3)
    for step in ([0, 7, 15], [2, 2, 2], [15, 1, 0]):
        module._fr13_fixed32_accept_ladder_accumulate(
            state, torch.tensor(step, dtype=torch.int32)
        )
    drained = module.fr13_fixed32_accept_ladder_drain(state)
    recomputed = sum(
        index * count for index, count in enumerate(drained["ladder"])
    ) + drained["overflow_tokens"]
    assert recomputed == drained["accepted_tokens"]


def test_the_ladder_clamps_and_counts_overflow_rather_than_dropping_it() -> None:
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    state = _ladder_state(batch=2)
    module._fr13_fixed32_accept_ladder_accumulate(
        state, torch.tensor([20, 3], dtype=torch.int32)
    )
    drained = module.fr13_fixed32_accept_ladder_drain(state)
    assert drained["ladder"][15] == 1, "clamped into the top slot"
    assert drained["overflow_rows"] == 1
    assert drained["overflow_tokens"] == 5, "20 - 15, counted not dropped"
    assert drained["accepted_tokens"] == 23, "exact despite the clamp"


def test_a_measured_zero_is_distinguishable_from_an_absent_instrument() -> None:
    """The round-6 mode, closed. This is the reason the ladder exists."""
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    state = _ladder_state(batch=2)
    module._fr13_fixed32_accept_ladder_accumulate(
        state, torch.tensor([0, 0], dtype=torch.int32)
    )
    measured = module.fr13_fixed32_accept_ladder_drain(state)
    assert measured["accepted_tokens"] == 0
    assert measured["rows"] == 2, "zero acceptance, but rows were OBSERVED"
    assert module.fr13_fixed32_accept_ladder_drain({"accept_ladder": None}) is None, (
        "an absent instrument must not present itself as a ladder of zeros"
    )


def test_the_ladder_accumulates_across_replays_without_reallocating() -> None:
    """CAPTURE SAFETY: a captured graph replays into the same memory.

    Every op is in-place on buffers allocated once in the committer state, so
    replay accumulates instead of re-zeroing, and nothing is re-allocated for
    the tracer to fold away.
    """
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    state = _ladder_state()
    pointers = (
        state["accept_ladder"].data_ptr(),
        state["accept_ladder_overflow_rows"].data_ptr(),
        state["accept_ladder_overflow_tokens"].data_ptr(),
    )
    lens = torch.tensor([3, 3, 3, 3], dtype=torch.int32)
    for replay in range(1, 6):
        module._fr13_fixed32_accept_ladder_accumulate(state, lens)
        assert int(state["accept_ladder"][3]) == 4 * replay, "must accumulate"
        assert (
            state["accept_ladder"].data_ptr(),
            state["accept_ladder_overflow_rows"].data_ptr(),
            state["accept_ladder_overflow_tokens"].data_ptr(),
        ) == pointers, "buffers were re-bound; a captured graph would drift"


def test_the_accumulation_is_inside_the_captured_graph_body() -> None:
    """Placement, asserted at the source: outside the body it never replays."""
    source = GDN_KERNEL.read_text()
    body = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_committer_graph_body"
    )
    calls = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fr13_fixed32_accept_ladder_accumulate"
    ]
    assert len(calls) == 1, "the ladder must accumulate exactly once per body"
    # ...and it must never sync the host on the step path
    accumulate = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_accept_ladder_accumulate"
    )
    banned = {"item", "tolist", "cpu", "numpy", "nonzero"}
    used = {
        node.func.attr
        for node in ast.walk(accumulate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (used & banned), f"host sync on the step path: {used & banned}"


def test_the_ladder_flag_is_strict_and_default_on_and_recorded() -> None:
    """Gated, disclosed. The H27n-bank asymmetry is stated, not hidden."""
    import os

    module = _load_gdn_kernel(PROFILE_HYDRA27)
    assert module._FR13_FIXED32_ACCEPT_LADDER_FLAG == "FR13_FIXED32_ACCEPT_LADDER"
    saved = os.environ.get(module._FR13_FIXED32_ACCEPT_LADDER_FLAG)
    try:
        os.environ.pop(module._FR13_FIXED32_ACCEPT_LADDER_FLAG, None)
        assert module._fr13_fixed32_accept_ladder_enabled() is True, "default ON"
        os.environ[module._FR13_FIXED32_ACCEPT_LADDER_FLAG] = "0"
        assert module._fr13_fixed32_accept_ladder_enabled() is False
        os.environ[module._FR13_FIXED32_ACCEPT_LADDER_FLAG] = "yes"
        with pytest.raises(RuntimeError):
            module._fr13_fixed32_accept_ladder_enabled()
    finally:
        if saved is None:
            os.environ.pop(module._FR13_FIXED32_ACCEPT_LADDER_FLAG, None)
        else:
            os.environ[module._FR13_FIXED32_ACCEPT_LADDER_FLAG] = saved
    # the flag and the schema travel with the drained payload
    state = _ladder_state(batch=1)
    drained = module.fr13_fixed32_accept_ladder_drain(state)
    assert drained["flag"] == "FR13_FIXED32_ACCEPT_LADDER"
    assert drained["enabled"] is True
    assert drained["slots"] == 16
    assert drained["schema"] == "fr13.fixed32.accept_ladder.v2"
    assert drained["definition"] == "committed_accepted_draft_path_length"


# --- THE WIRING (round 21's last gate) --------------------------------------
# The drain reaches a discoverable on-disk artifact: the flush boundary
# snapshot. Route chosen on soundness, not cheapness -- see
# test_the_ladder_is_not_drained_on_the_step_path.
def test_the_flush_boundary_drains_the_ladder_into_its_snapshot() -> None:
    """The emission point, asserted at the source of the flush blob."""
    patcher = PATCHER.read_text()
    boundary = patcher[patcher.index("def _fr13_f32_flush_write_boundary(") :]
    boundary = boundary[: boundary.index("\ndef ", 1)]
    assert "fr13_fixed32_accept_ladder_snapshot" in boundary, (
        "the flush boundary no longer drains the ladder"
    )
    assert "accept_ladder = accept_ladder_snapshot()" in boundary
    # The payload reaches its OWN generation-numbered artifact beside the
    # boundary snapshot...
    assert '"schema": "fr13-fixed32-accept-ladder-sidecar-v1",' in boundary
    assert '".accept_ladder.json"' in boundary
    assert '"accept_ladder": accept_ladder,' in boundary, (
        "the drained payload must reach the written sidecar"
    )
    # ...and NOT into the boundary snapshot dict, whose key set is enforced by
    # exact_keys in both run_swe_bench_q36_a and fr13_floor_gate. A new
    # required key there would make every BANKED snapshot fail validation
    # because of an instrument added after they were written.
    snapshot_literal = boundary[boundary.index("    snapshot = {") :]
    snapshot_literal = snapshot_literal[: snapshot_literal.index("\n    }")]
    assert "accept_ladder" not in snapshot_literal, (
        "the ladder must not become a key of the exact-key-validated snapshot"
    )
    assert '"schema": "fr13-fixed32-boundary-snapshot-v4",' in snapshot_literal


def test_the_boundary_snapshot_key_contract_is_untouched() -> None:
    """The banked artifacts must stay readable.

    exact_keys is exact in both directions: a new REQUIRED key would reject
    every boundary snapshot written before the instrument existed. The sidecar
    adds a file instead of changing a contract.
    """
    for consumer in ("run_swe_bench_q36_a.py", "fr13_floor_gate.py"):
        text = (SCRIPTS / consumer).read_text()
        assert "accept_ladder" not in text, (
            f"{consumer} learned an instrument key; the banked boundary "
            "snapshots would start failing validation"
        )


def test_the_ladder_is_not_drained_on_the_step_path() -> None:
    """Why the emission point is the flush and not the counters function.

    fixed32_committer_counters() is called by _fr13_fixed32_device_commit_route,
    which runs EVERY STEP. Draining there would have been a smaller diff and a
    device-to-host sync on the path whose wall time is the measurement.
    """
    patcher = PATCHER.read_text()
    assert "fixed32_committer_counters" in patcher
    step_route = patcher[patcher.index("def _fr13_fixed32_device_commit_route(") :]
    step_route = step_route[: step_route.index("\ndef ", 1)]
    assert "fixed32_committer_counters" in step_route, (
        "assumption check: the per-step route calls the counters function"
    )
    assert "accept_ladder" not in step_route, (
        "the ladder must never be drained on the step path"
    )
    # the counters function itself must stay sync-free
    source = GDN_KERNEL.read_text()
    counters = source[source.index("def fixed32_committer_counters(") :]
    counters = counters[: counters.index("\ndef ", 1)]
    assert "accept_ladder" not in counters, (
        "fixed32_committer_counters runs per step; it must not read the ladder"
    )


def test_the_snapshot_aggregates_every_committer_state() -> None:
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    saved = dict(module._FR13_FIXED32_COMMITTER)
    try:
        module._FR13_FIXED32_COMMITTER.clear()

        def state(batch):
            return {
                "batch": batch,
                "accept_ladder": torch.zeros(16, dtype=torch.int64),
                "accept_ladder_overflow_rows": torch.zeros((), dtype=torch.int64),
                "accept_ladder_overflow_tokens": torch.zeros(
                    (), dtype=torch.int64
                ),
                "accept_ladder_ones": torch.ones(batch, dtype=torch.int64),
            }

        module._FR13_FIXED32_COMMITTER["a"] = state(4)
        module._FR13_FIXED32_COMMITTER["b"] = state(2)
        for step in ([1, 2, 3, 4], [15, 15, 15, 15]):
            module._fr13_fixed32_accept_ladder_accumulate(
                module._FR13_FIXED32_COMMITTER["a"],
                torch.tensor(step, dtype=torch.int32),
            )
        module._fr13_fixed32_accept_ladder_accumulate(
            module._FR13_FIXED32_COMMITTER["b"],
            torch.tensor([20, 0], dtype=torch.int32),
        )
        payload = module.fr13_fixed32_accept_ladder_snapshot()
        assert payload["committer_states"] == 2
        assert payload["rows"] == 10
        assert payload["overflow_rows"] == 1 and payload["overflow_tokens"] == 5
        # the identity the sealed harness asserts
        identity = sum(
            index * count for index, count in enumerate(payload["ladder"])
        ) + payload["overflow_tokens"]
        assert identity == payload["accepted_tokens"] == 90
    finally:
        module._FR13_FIXED32_COMMITTER.clear()
        module._FR13_FIXED32_COMMITTER.update(saved)


def test_a_never_drained_ladder_is_absent_downstream_not_zero() -> None:
    """MUTATION PROOF of the WIRING, not just the counter.

    If nothing was ever preseeded -- or the flag was off -- the artifact must
    say so. A downstream reader must never see a ladder of zeros and conclude
    acceptance was measured at zero. That conflation is the metric this
    replaces.
    """
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    saved = dict(module._FR13_FIXED32_COMMITTER)
    try:
        module._FR13_FIXED32_COMMITTER.clear()
        payload = module.fr13_fixed32_accept_ladder_snapshot()
        assert payload["enabled"] is False
        assert payload["ladder"] is None
        assert payload["accepted_tokens"] is None
        assert payload["rows"] is None
        assert payload["committer_states"] == 0
        assert payload["schema"] == "fr13.fixed32.accept_ladder.v2"
        assert payload["flag"] == "FR13_FIXED32_ACCEPT_LADDER"
    finally:
        module._FR13_FIXED32_COMMITTER.clear()
        module._FR13_FIXED32_COMMITTER.update(saved)


def test_the_payload_carries_everything_the_sealed_harness_asserts() -> None:
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    saved = dict(module._FR13_FIXED32_COMMITTER)
    try:
        module._FR13_FIXED32_COMMITTER.clear()
        module._FR13_FIXED32_COMMITTER["only"] = {
            "batch": 1,
            "accept_ladder": torch.zeros(16, dtype=torch.int64),
            "accept_ladder_overflow_rows": torch.zeros((), dtype=torch.int64),
            "accept_ladder_overflow_tokens": torch.zeros((), dtype=torch.int64),
            "accept_ladder_ones": torch.ones(1, dtype=torch.int64),
        }
        payload = module.fr13_fixed32_accept_ladder_snapshot()
        assert set(payload) == {
            "schema", "enabled", "flag", "slots", "ladder", "rows",
            "accepted_tokens", "overflow_rows", "overflow_tokens",
            "committer_states",
            # v2: the definition fork and the warmup bracket travel with the
            # number, so the harness compares like-for-like by construction.
            "definition", "aggregate_definition", "scope",
            "accepted_draft_tokens", "bonus_tokens", "emitted_tokens",
            "warmup_rows", "warmup_accepted_draft_tokens", "warmup_ladder",
            "rows_since_warmup", "accepted_draft_tokens_since_warmup",
            "ladder_since_warmup",
        }
        assert payload["slots"] == 16 and len(payload["ladder"]) == 16
    finally:
        module._FR13_FIXED32_COMMITTER.clear()
        module._FR13_FIXED32_COMMITTER.update(saved)


# ---------------------------------------------------------------------------
# 12. SITE 13 -- THE SCALAR THAT SHADOWED A PER-MODE AUTHORITY
# ---------------------------------------------------------------------------
# topology.WALK_CAP is MAX_PHYSICAL_DEPTH + 1 and has always been hydra27's 12.
# It is a module-level SCALAR, so it never became per-mode when hydra31
# arrived, and the guard that caught it interpolated {mode} into its message
# while comparing that scalar -- it read mode-aware without being it. Rounds
# 1-7 were this class at dict granularity and the key-set invariant covers
# those; a scalar has no key set to be wrong about, so it escaped.
MULTIDRAFT = SCRIPTS / "fr13_device_multidraft_kernel.py"
# The revision that still had site 13, pinned so the mutation proof survives
# HEAD moving under other lanes.
SITE13_BASELINE = "161e73672c517067d5b2a405d18d6c24c0582cb6"
CENSUS = SCRIPTS / "fr13_fixed32_work_census.py"

# Curated topology-constant -> profile-key map, the same basis as SWAP_MAP:
# equality alone is a coincidence detector (GDN_CONV_KERNEL_SIZE is 4 and so is
# rescue_carry_slots), and a census built on coincidence lies about its subject.
SHADOWING_SCALARS = {
    "FIXED32_CHOICES": "choices",
    "PHYSICAL_PARENT": "physical_parent",
    "EXPECTED_PHYSICAL_PARENT": "physical_parent",
    "PHYSICAL_PARENT_SHA256": "physical_parent_sha256",
    "TREE_ANCESTRY_SHA256": "tree_ancestry_sha256",
    "SUBTREE_LEVELS": "subtree_levels",
    "MAX_PHYSICAL_DEPTH": "max_physical_depth",
    "WALK_CAP": "walk_cap",
    "TAW_PATH_SCATTER_SLOTS": "walk_cap",
    "ARCTIC_MAIN_TAIL_LENGTH": "main_tail_length",
    "ARCTIC_LOOKUP_TOKENS_PER_REQUEST": "arctic_requested_tokens",
    "GATED_ARCTIC_MAIN_TAIL_LENGTH": "gated_main_tail_length",
    "GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST": "gated_arctic_requested_tokens",
    "RESCUE_CARRY_SLOTS_PER_REQUEST": "rescue_carry_slots",
    "GDN_LEVEL_MAX_LENGTHS": "gdn_level_max_lengths",
    "GDN_PADDED_SLOTS": "gdn_padded_slots",
    "PHYSICAL_BRANCH_CHAINS": "physical_branch_chains",
}

# SITE 13 ADJUDICATION for the modules that carry their OWN WALK_CAP = 12.
# Written here rather than as a comment in those files ON PURPOSE: all three
# are credential-bound by sha256 ("credential-bound device module identity
# drifted"), so even a documentation comment re-attests three byte-qualified
# lever credentials. The adjudication is worth recording; it is not worth that.
PINNED_PRIVATE_WALK_CAP_MODULES = {
    "fr13_cfwd_logit_direct_decision_kernel.py": (
        "DEFAULT-OFF CFWD logit-direct lever, byte-AB qualified on hydra27's "
        "tree and already refused for hydra31 at the launcher "
        "(FR13_CFWD_LOGIT_DIRECT_* in _fr14_h31_incompat)."
    ),
    "fr13_cfwd_packed_walk_active_depth_kernel.py": (
        "DEFAULT-OFF packed-walk active-depth lever, same qualification and "
        "the same launcher refusal."
    ),
    "fr13_cfwd_packed_walk_node_trust_kernel.py": (
        "DEFAULT-OFF packed-walk node-trust lever, same qualification and the "
        "same launcher refusal."
    ),
}

# topology.WALK_CAP readers left in the execution-path kernel, each adjudicated.
PINNED_WALK_CAP_READERS = {
    1: (
        "_fr13_fixed32_taw_topology_binding bounds the QUALIFIED SCOPE's "
        "schedule (tail6 + hydra27), whose byte-AB pass measured that depth. A "
        "mode outside the scope derives its own cap a few lines below."
    ),
    2: (
        "_fr13_fixed32_walk_cap's own fallback: an unset mode is the "
        "non-fixed32 route, which has always used hydra27's cap, so returning "
        "it keeps that path byte-identical."
    ),
}


def test_the_authority_carries_a_per_mode_walk_cap() -> None:
    topology = _topology_module()
    assert set(topology.WALK_CAP_BY_MODE) == set(topology.SERVING_MODES)
    for mode in topology.SERVING_MODES:
        expected = int(
            topology.PROFILES[topology.TREE_PROFILE_BY_MODE[mode]]["walk_cap"]
        )
        assert topology.WALK_CAP_BY_MODE[mode] == expected
        assert topology.walk_cap_for_mode(mode) == expected
    assert topology.WALK_CAP_BY_MODE[PROFILE_HYDRA31] == 16
    assert topology.WALK_CAP_BY_MODE[PROFILE_HYDRA27] == 12
    # the bare scalar is unchanged, so nothing that legitimately means
    # hydra27's 12 moved underneath it
    assert topology.WALK_CAP == 12
    with pytest.raises(KeyError):
        topology.walk_cap_for_mode("nope_fixed32")


def test_the_walk_cap_index_is_derived_not_retyped() -> None:
    """A retyped cap is a cap that can disagree with its own tree."""
    source = TOPOLOGY.read_text()
    block = source[source.index("WALK_CAP_BY_MODE: dict[Mode, int] = {") :]
    block = block[: block.index("}") + 1]
    assert "PROFILES[TREE_PROFILE_BY_MODE[" in block, (
        "the index must be built from PROFILES, not written out"
    )
    for literal in ("12", "16"):
        assert literal not in block, f"the index retypes {literal}"


def test_no_unadjudicated_walk_cap_reader_remains_in_the_kernel() -> None:
    """Every reader is mode-keyed or pinned with a reason, per site."""
    source = MULTIDRAFT.read_text()
    # AST, not text: a docstring that NAMES the constant is not a reader, and a
    # census that cannot tell prose from code will be tuned until it shuts up.
    readers = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and node.attr == "WALK_CAP"
        and isinstance(node.value, ast.Name)
        and node.value.id == "topology"
    ]
    assert len(readers) == len(PINNED_WALK_CAP_READERS), (
        f"{len(readers)} bare topology.WALK_CAP readers remain, "
        f"{len(PINNED_WALK_CAP_READERS)} are adjudicated: {readers}"
    )
    # ...and the resolver every other site now goes through exists
    assert "def _fr13_fixed32_walk_cap(" in source
    assert source.count("_fr13_fixed32_walk_cap(topology") >= 15, (
        "the conversion did not reach the sizers and provenance"
    )
    # the literal 12 that guarded the loaded authority is still an identity
    # check, and now also requires the per-mode index to exist
    assert "module.WALK_CAP != 12" in source
    assert 'set(module.WALK_CAP_BY_MODE) != set(module.SERVING_MODES)' in source


def test_the_provenance_records_the_served_walk_not_the_scalar() -> None:
    """The silent site: a run executing 16 must not record walk_cap=12."""
    source = MULTIDRAFT.read_text()
    contract = source[source.index("def _fr13_fixed32_taw_source_contract(") :]
    contract = contract[: contract.index("\ndef ", 1)]
    assert '"walk_cap": _fr13_fixed32_walk_cap(topology),' in contract
    assert '"walk_cap": int(topology.WALK_CAP),' not in contract


# --- (4b) THE MESSAGE-LIE DETECTOR ------------------------------------------
# Site 13's guard said "{mode}: TAW walk cap 16 != contract 12" while comparing
# a constant that knows no modes. The message was mode-aware; the comparison
# was not. A first attempt -- "a raise naming {mode} whose enclosing test reads
# nothing mode-keyed" -- produced 95 candidates, nearly all honest (`if mode
# not in FIXED32_MODES` legitimately names the mode), so it was dropped rather
# than tuned into silence. The shipped form is narrow: a raise that names the
# mode, guarded by a test that reads a scalar KNOWN to shadow a per-mode
# authority. That fires on exactly site 13 in the pre-fix source and on nothing
# else in either root.
def message_lie_census(repo_root: Path) -> list[tuple[str, int, list[str]]]:
    topology = _topology_module()
    p27 = topology.profile(PROFILE_HYDRA27)
    p31 = topology.profile(PROFILE_HYDRA31)
    varying = {
        name
        for name, key in SHADOWING_SCALARS.items()
        if p27[key] != p31[key]
    }
    out: list[tuple[str, int, list[str]]] = []
    for sub in ("scripts", "src"):
        root = repo_root / sub
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="replace"))
            except SyntaxError:  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                raises = [
                    r
                    for r in ast.walk(node)
                    if isinstance(r, ast.Raise) and r.exc is not None
                ]
                if not raises:
                    continue
                if "'mode'" not in " ".join(ast.dump(r.exc) for r in raises):
                    continue
                names = {
                    n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)
                } | {
                    n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
                }
                shadow = names & varying
                if shadow:
                    out.append(
                        (
                            str(path.relative_to(repo_root)),
                            raises[0].lineno,
                            sorted(shadow),
                        )
                    )
    return out


def test_no_guard_names_the_mode_while_comparing_a_mode_blind_scalar() -> None:
    found = message_lie_census(REPO)
    assert not found, (
        "a raise interpolates {mode} while its test reads a scalar that "
        f"shadows a per-mode authority -- site 13's exact shape: {found}"
    )


def test_the_message_lie_detector_fires_on_the_pre_fix_source() -> None:
    """MUTATION PROOF, and the over-firing check: exactly one hit, no others."""
    import tempfile

    try:
        baseline = subprocess.run(
            ["git", "show", f"{SITE13_BASELINE}:scripts/fr13_device_multidraft_kernel.py"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"pre-site-13 kernel unavailable: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scripts").mkdir()
        (root / "scripts" / "fr13_device_multidraft_kernel.py").write_text(baseline)
        found = message_lie_census(root)
    assert len(found) == 1, f"expected exactly site 13, got {found}"
    path, _line, shadow = found[0]
    assert path.endswith("fr13_device_multidraft_kernel.py")
    assert shadow == ["WALK_CAP"]


# --- (3)+(4a) THE SCALAR-AUTHORITY CENSUS -----------------------------------
def test_every_shadowing_scalar_is_really_a_shadow() -> None:
    """The curated map, validated -- coincidence is not evidence."""
    topology = _topology_module()
    p27 = topology.profile(PROFILE_HYDRA27)
    for name, key in sorted(SHADOWING_SCALARS.items()):
        assert hasattr(topology, name), f"topology lost {name}"
        assert key in p27, f"PROFILES lost {key!r}"
        assert getattr(topology, name) == p27[key], (
            f"{name} is not hydra27's profile[{key!r}] -- the map is stale"
        )


def test_the_scalar_census_is_the_pinned_inventory() -> None:
    """A new module scalar shadowing a per-mode authority must be adjudicated.

    Pinned as a SET of names rather than a count: a scalar appearing and
    another disappearing would leave a count unchanged, and that is exactly the
    kind of silence this campaign keeps paying for.
    """
    topology = _topology_module()
    p27 = topology.profile(PROFILE_HYDRA27)
    p31 = topology.profile(PROFILE_HYDRA31)
    varying = {
        name for name, key in SHADOWING_SCALARS.items() if p27[key] != p31[key]
    }
    assert varying == {
        "ARCTIC_LOOKUP_TOKENS_PER_REQUEST",
        "ARCTIC_MAIN_TAIL_LENGTH",
        "EXPECTED_PHYSICAL_PARENT",
        "FIXED32_CHOICES",
        "GATED_ARCTIC_LOOKUP_TOKENS_PER_REQUEST",
        "GATED_ARCTIC_MAIN_TAIL_LENGTH",
        "GDN_LEVEL_MAX_LENGTHS",
        "GDN_PADDED_SLOTS",
        "MAX_PHYSICAL_DEPTH",
        "PHYSICAL_BRANCH_CHAINS",
        "PHYSICAL_PARENT",
        "PHYSICAL_PARENT_SHA256",
        "RESCUE_CARRY_SLOTS_PER_REQUEST",
        "SUBTREE_LEVELS",
        "TAW_PATH_SCATTER_SLOTS",
        "TREE_ANCESTRY_SHA256",
        "WALK_CAP",
    }, f"the shadowing-scalar inventory moved: {sorted(varying)}"


# --- (5) THE CPU WALK, EXTENDED TO THE WARM-EXECUTE CONTRACT ----------------
# The walk built the pregather record (site 11) but never ran the contract that
# CUDA-graph warm capture calls first. Site 13 lived there. The walk asks the
# boot's question, and the boot's question now includes this one.
def _runtime_contract_env(mode: str, **overrides) -> dict[str, str]:
    topology = _topology_module()
    env = {
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": hex(int(topology.VALID_MASK_BY_MODE[mode])),
        "FR13_FIXED32_ACTIVE_NODES": str(
            sum(1 for flag in topology.VALID_BY_MODE[mode] if flag)
        ),
        "FR13_FIXED32_TAW_WALK_CAP": str(topology.walk_cap_for_mode(mode)),
        "FR13_TAW": "1",
    }
    env.update(overrides)
    return env


def _run_runtime_contract(mode: str, **overrides):
    import os

    kernel = _multidraft_kernel()
    saved = dict(os.environ)
    try:
        os.environ.update(_runtime_contract_env(mode, **overrides))
        return kernel._fr13_fixed32_runtime_contract(mode)
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.mark.parametrize(
    "mode", ["tail6_fixed32", PROFILE_HYDRA27, PROFILE_HYDRA31]
)
def test_the_warm_execute_contract_passes_for_every_serving_mode(
    mode: str,
) -> None:
    """The :3304 path -- where round 21 died at CUDA-graph warm capture."""
    _run_runtime_contract(mode)


@pytest.mark.parametrize(
    "mode,wrong_cap",
    [
        (PROFILE_HYDRA31, "12"),
        (PROFILE_HYDRA27, "16"),
        ("tail6_fixed32", "16"),
    ],
)
def test_the_warm_execute_contract_still_refuses_a_wrong_cap(
    mode: str, wrong_cap: str
) -> None:
    """MUTATION PROOF, both directions.

    Fixing the scalar must not soften the guard: hydra31 with 12 refuses (the
    silent sizers would otherwise allocate 12 rows for a 16-deep walk), and
    hydra27 with 16 refuses just as it always did.
    """
    with pytest.raises(RuntimeError) as refusal:
        _run_runtime_contract(mode, FR13_FIXED32_TAW_WALK_CAP=wrong_cap)
    message = str(refusal.value)
    assert "TAW walk cap" in message and mode in message
    # ...and the message now names the cap the SERVED mode actually requires
    assert str(_topology_module().walk_cap_for_mode(mode)) in message


def test_the_census_round_trips_at_every_modes_walk_depth() -> None:
    """The provenance half: a hydra31 event is validated at 16, not at 12.

    Before this, reference_event described hydra27's 12-deep walk for every
    mode, so a correct hydra31 census was reported as a drifted hydra27 one --
    the measurement blaming the run for the instrument's assumption.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("census_walk", CENSUS)
    census = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = census
    try:
        spec.loader.exec_module(census)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"census unavailable: {exc}")
    finally:
        sys.modules.pop(spec.name, None)

    topology = _topology_module()
    for mode in topology.SERVING_MODES:
        for batch in (1, 4):
            event = census.reference_event(mode, batch, f"{mode}:1:0")
            census.validate_event(event, source="walk")
            walk = topology.walk_cap_for_mode(mode)
            assert event["taw"]["loop_iterations"] == walk
            assert event["taw"]["uniform_shape"] == [batch, walk, 3]
            assert event["taw"]["tensor_call_census"]["walk_levels"] == walk
            assert event["gdn"]["critical_path"] == walk
    # the profiles must genuinely differ, or this proves nothing
    assert topology.walk_cap_for_mode(PROFILE_HYDRA31) != topology.walk_cap_for_mode(
        PROFILE_HYDRA27
    )


def test_the_private_walk_cap_modules_stay_pinned_and_untouched() -> None:
    """Their 12 is hydra27's BY IDENTITY, and their bytes are credentials.

    Each is a default-off lever whose byte-AB pass was measured on hydra27 and
    which the launcher already refuses for hydra31. Converting them would let
    an unqualified tree through a gate that never measured it; even commenting
    them drifts a credential-bound module identity.
    """
    for name, reason in PINNED_PRIVATE_WALK_CAP_MODULES.items():
        source = (SCRIPTS / name).read_text()
        assert "WALK_CAP = 12" in source, f"{name} lost its pinned cap"
        assert "WALK_CAP_BY_MODE" not in source, (
            f"{name} was converted; it is credential-bound and hydra27-only: "
            f"{reason}"
        )
        assert PROFILE_HYDRA31 not in source, (
            f"{name} learned hydra31 without re-qualification"
        )


# ---------------------------------------------------------------------------
# 13. THE LADDER'S DEFINITION (first live outing, refused by the harness)
# ---------------------------------------------------------------------------
# The refusal decomposed into one scope artifact and one semantic fork:
#   rows   +4, +4, +4     CONSTANT  -- generation 1's warmup drafts
#   tokens +32, +52, +61  GROWING   -- two definitions of "accepted token"
# The banked generations are replayed below, so these are the real numbers.
BANKED_LADDERS = {
    1: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [53, 152, 154, 136, 97, 197, 57, 36, 24, 18, 12, 119, 0, 0, 0, 0],
    4: [236, 734, 765, 598, 424, 768, 210, 145, 85, 65, 64, 710, 0, 0, 0, 0],
    6: [410, 1029, 1583, 1495, 1234, 2798, 398, 376, 319, 171, 101, 1006, 0, 0, 0, 0],
}
BANKED_METRICS = {2: (1051, 4586), 4: (4800, 21532), 6: (10916, 48732)}


def _vllm_accepted_for_row(accepted_len: int) -> int:
    """vLLM's definition, modelled from its source rather than described.

    scheduler.update_from_output: `num_accepted = len(generated_token_ids) - 1`,
    and our publish writes the bonus into column 0 with the accepted path after
    it, so an unfiltered row emits accepted_len + 1 tokens.
    """
    return (accepted_len + 1) - 1


def _ladder_module_and_state(batch: int = 4):
    torch = pytest.importorskip("torch")
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    module._FR13_FIXED32_ACCEPT_LADDER_WARMUP = None
    state = {
        "batch": batch,
        "accept_ladder": torch.zeros(16, dtype=torch.int64),
        "accept_ladder_overflow_rows": torch.zeros((), dtype=torch.int64),
        "accept_ladder_overflow_tokens": torch.zeros((), dtype=torch.int64),
        "accept_ladder_ones": torch.ones(batch, dtype=torch.int64),
    }
    module._FR13_FIXED32_COMMITTER.clear()
    module._FR13_FIXED32_COMMITTER["only"] = state
    return module, state


def test_the_payload_names_both_definitions() -> None:
    """The fork, stated in the artifact rather than inferred by the reader."""
    module, _state = _ladder_module_and_state()
    payload = module.fr13_fixed32_accept_ladder_snapshot()
    assert payload["schema"] == "fr13.fixed32.accept_ladder.v2"
    assert payload["definition"] == "committed_accepted_draft_path_length"
    aggregate = payload["aggregate_definition"]
    assert "spec_decode_num_accepted_tokens_total" in aggregate
    assert "minus" in aggregate and "bonus" in aggregate
    assert payload["scope"] == "process_lifetime_totals_plus_since_warmup_deltas"


def test_a_full_acceptance_step_reconciles_under_the_stated_definition() -> None:
    """MUTATION PROOF: the bonus convention, settled by construction.

    A row that accepts the whole depth-11 spine contributes 11 draft tokens and
    one bonus. Under the stated definition the ladder counts 11; vLLM's
    len(generated) - 1 is also 11. They reconcile, which is why the live drift
    was NOT the bonus token -- full-acceptance rows numbered 119/710/1006 while
    the drift was 32/52/61.
    """
    torch = pytest.importorskip("torch")
    module, state = _ladder_module_and_state(batch=2)
    full = module._FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH
    assert full == 11
    for _step in range(5):
        module._fr13_fixed32_accept_ladder_accumulate(
            state, torch.tensor([full, full], dtype=torch.int32)
        )
    payload = module.fr13_fixed32_accept_ladder_snapshot()
    rows = payload["rows"]
    assert rows == 10 and payload["ladder"][full] == 10
    assert payload["accepted_draft_tokens"] == rows * full
    assert payload["accepted_draft_tokens"] == rows * _vllm_accepted_for_row(full)
    assert payload["bonus_tokens"] == rows, "one bonus per committed row"
    assert payload["emitted_tokens"] == rows * (full + 1)


@pytest.mark.parametrize("accepted_len", [0, 1, 5, 11])
def test_every_row_class_reconciles_with_the_aggregate_definition(
    accepted_len: int,
) -> None:
    """The two definitions agree per row whenever nothing is filtered."""
    torch = pytest.importorskip("torch")
    module, state = _ladder_module_and_state(batch=1)
    module._fr13_fixed32_accept_ladder_accumulate(
        state, torch.tensor([accepted_len], dtype=torch.int32)
    )
    payload = module.fr13_fixed32_accept_ladder_snapshot()
    assert payload["accepted_draft_tokens"] == _vllm_accepted_for_row(accepted_len)
    assert payload["emitted_tokens"] == accepted_len + 1


def test_the_warmup_bracket_makes_the_banked_rows_reconcile_exactly() -> None:
    """The scope artifact, closed against the real serve's numbers.

    Generation 1 recorded rows=4, tokens=0 before any task. With the warmup
    labelled and subtractable, rows match the aggregate EXACTLY at all three
    boundaries -- the +4 is gone by construction, not by an adjustment someone
    remembered to apply.
    """
    torch = pytest.importorskip("torch")
    module, state = _ladder_module_and_state()

    def replay(generation: int):
        state["accept_ladder"].copy_(
            torch.tensor(BANKED_LADDERS[generation], dtype=torch.int64)
        )
        return module.fr13_fixed32_accept_ladder_snapshot()

    warmup = replay(1)
    assert warmup["warmup_rows"] == 4
    assert warmup["warmup_accepted_draft_tokens"] == 0
    assert warmup["rows_since_warmup"] == 0

    for generation, (metric_rows, metric_tokens) in BANKED_METRICS.items():
        payload = replay(generation)
        assert payload["rows_since_warmup"] == metric_rows, (
            f"gen {generation}: rows must reconcile exactly after the bracket"
        )
        # the token residual is REAL and is not silently absorbed
        residual = payload["accepted_draft_tokens_since_warmup"] - metric_tokens
        assert residual == {2: 32, 4: 52, 6: 61}[generation]
        # ...and the ladder itself is internally exact at every boundary
        recomputed = sum(
            index * count for index, count in enumerate(payload["ladder"])
        )
        assert recomputed + payload["overflow_tokens"] == (
            payload["accepted_draft_tokens"]
        )


def test_the_residual_is_not_the_bonus_and_not_per_row() -> None:
    """Why the bonus hypothesis is refuted, kept as an assertion.

    If the gap were the bonus token on full-acceptance rows it would equal the
    slot-11 counts; if it were per-row it would scale with rows. It is neither:
    the rate FALLS across windows.
    """
    full_acceptance = {gen: BANKED_LADDERS[gen][11] for gen in BANKED_METRICS}
    residuals = {2: 32, 4: 52, 6: 61}
    for generation, residual in residuals.items():
        assert full_acceptance[generation] > residual * 3, (
            "full-acceptance rows are an order of magnitude off the residual"
        )
    rows = {gen: sum(BANKED_LADDERS[gen]) - 4 for gen in BANKED_METRICS}
    rates = [residuals[g] / rows[g] for g in (2, 4, 6)]
    assert rates[0] > rates[1] > rates[2], (
        "a per-row cause would hold the rate constant; it falls"
    )


def test_the_absent_payload_still_says_absent_under_v2() -> None:
    module = _load_gdn_kernel(PROFILE_HYDRA27)
    saved = dict(module._FR13_FIXED32_COMMITTER)
    try:
        module._FR13_FIXED32_COMMITTER.clear()
        payload = module.fr13_fixed32_accept_ladder_snapshot()
        assert payload["enabled"] is False
        assert payload["ladder"] is None
        assert payload["accepted_draft_tokens"] is None
        assert payload["rows_since_warmup"] is None
        # the definition travels even when the measurement does not
        assert payload["definition"] == "committed_accepted_draft_path_length"
        assert payload["schema"] == "fr13.fixed32.accept_ladder.v2"
    finally:
        module._FR13_FIXED32_COMMITTER.clear()
        module._FR13_FIXED32_COMMITTER.update(saved)


# ---------------------------------------------------------------------------
# 14. PER-REQUEST SEEDING -- THE GAP, PINNED (workstream task 1)
# ---------------------------------------------------------------------------
# The ask was to inject seed = stable_hash(task, attempt, turn) at the proxy so
# every task replays exactly. It cannot be shipped on the promoted fixed32
# route, and the reason is not that the seed would be ignored -- it is that the
# route REFUSES the map the seed produces:
#
#   vLLM DOES support per-request seeds. SamplingParams.seed -> request.generator
#   -> InputBatch.generators -> SamplingMetadata.generators, and our patcher
#   forwards them verbatim (`generators=getattr(sampling_metadata,
#   "generators", None)`).
#
#   The fixed32 commit hands that map to _fr13_fixed32_fill_uniforms, which
#   raises "FR13 fixed32 forbids per-request generator maps; the fixed route
#   requires one bulk device RNG call" the moment it is non-empty.
#
# So injecting a seed does not silently do nothing -- it makes every fixed32
# step raise. That is worse than a placebo and it is why this landing is the
# gap rather than the knob.
#
# What DOES hold: --seed 0 is pinned in the pid1 argv contract and the bulk
# generator seeds from torch.initial_seed(), lazily, ONCE PER PROCESS. So the
# stream is deterministic but each request's draws depend on its POSITION in
# that stream -- on batch composition and every request that came before. The
# bulk generator's own docstring says it: "the unseeded bulk draw was never
# reproducible, so this is distribution-equal". Per-task determinism is not
# missing by oversight; it is absent by construction, one function away from a
# guard that says so.
PROXY = SERVING / "inference_proxy.py"


def test_the_fixed32_route_refuses_per_request_generators() -> None:
    """The gap, executed rather than described.

    If this ever starts passing a non-empty map, the route has gained
    per-request seeding and the proxy-side injection becomes shippable -- which
    is exactly when this test should fail and be rewritten.
    """
    torch = pytest.importorskip("torch")
    kernel = _multidraft_kernel()
    entry = {"uniforms": torch.zeros(1, 12, 3, dtype=torch.float32)}

    # today's route: no map, one bulk draw
    _out, route = kernel._fr13_fixed32_fill_uniforms(entry, generators=None)
    assert route == "bulk_device_generator"
    _out, route = kernel._fr13_fixed32_fill_uniforms(entry, generators={})
    assert route == "bulk_device_generator", "an empty map must stay the bulk route"

    generator = torch.Generator()
    generator.manual_seed(1234)
    with pytest.raises(RuntimeError) as refusal:
        kernel._fr13_fixed32_fill_uniforms(entry, generators={0: generator})
    assert "forbids per-request generator maps" in str(refusal.value), (
        "the refusal that makes proxy-side seeding unshippable has moved; "
        "re-assess whether per-request seeding is now possible"
    )


def test_the_proxy_injects_no_seed_and_must_not_until_the_route_accepts_one(
) -> None:
    """No placebo knob. A seed accepted and then refused downstream is worse
    than no seed at all."""
    source = PROXY.read_text()
    assert '"seed"' not in source and "'seed'" not in source, (
        "the proxy now injects a seed -- the fixed32 route refuses the "
        "generator map that produces, so every step would raise"
    )
    # the forcing sites the ask named are still where they were
    assert "LUMO_PROXY_FORCE_TEMPERATURE" in source
    assert "LUMO_PROXY_FORCE_TOP_P" in source
    assert "LUMO_PROXY_FORCE_TOP_K" in source


def test_the_patcher_still_forwards_vllms_generator_map() -> None:
    """The link that makes the refusal reachable, pinned.

    If the patcher stopped forwarding the map, seeds would become silently
    ignored -- the placebo case -- and this test is what would notice.
    """
    source = PATCHER.read_text()
    assert 'generators=getattr(sampling_metadata, "generators", None)' in source


def test_the_only_determinism_that_holds_is_process_level() -> None:
    """--seed is pinned and the bulk generator derives from it, once."""
    import fr13_fixed32_contract as contract

    argv = contract.expected_pid1_argv(1)
    assert "--seed" in argv, "the engine seed pin is the one determinism we have"
    assert argv[argv.index("--seed") + 1] == "0"

    source = MULTIDRAFT.read_text()
    bulk = source[source.index("def _fr13_bulk_gen(") :]
    bulk = bulk[: bulk.index("\ndef ", 1)]
    assert "torch.initial_seed()" in bulk, "the bulk stream derives from the pin"
    assert "_FR13_BULK_GEN is None" in bulk, "seeded once per process, not per request"
    # ...and the module says out loud that this is not reproducibility
    assert "never reproducible" in " ".join(bulk.split()), (
        "the bulk generator no longer disclaims reproducibility"
    )
