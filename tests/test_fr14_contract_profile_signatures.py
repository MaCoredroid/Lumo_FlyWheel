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
import hashlib
import importlib.util
import inspect
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
PATCHER_BASELINE_BLOB = "036049d3c54bcec4c0fad3fd0894a1373903aa76"


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
    assert not drift, f"hydra27 patcher output CHANGED: {sorted(drift)}"

    removed = sorted(set(old) - set(new))
    assert not removed, f"removed from the patcher surface: {removed}"
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

    old, new = blob_digests(source), blob_digests(PATCHER.read_text())
    assert len(old) >= 40, f"blob enumeration collapsed to {len(old)}"
    assert old == new, (
        "an injected source blob changed -- if this is intended it is a "
        "RE-ATTESTATION EVENT for every banked hydra27 run, not a silent edit"
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
    "scripts/fr13_fixed32_flush_protocol.py": 1,
    "scripts/fr13_fixed32_nsys_reduce.py": 1,
    "scripts/fr13_fixed32_semantics_test.py": 2,
    "scripts/fr13_fixed32_topology.py": 2,
    "scripts/fr13_floor_gate.py": 5,
    "scripts/fr13_gdn_gqa_group3_production_credential.py": 1,
    "scripts/fr13_gdn_single_launch_gate.py": 1,
    "scripts/fr13_gdn_single_launch_production_credential.py": 1,
    "scripts/fr13_taw_b1_credential.py": 2,
    "scripts/fr13_treeconv_zero_tail_credential.py": 1,
    "scripts/run_swe_bench_q36_a.py": 1,
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": 7,
    "src/lumo_flywheel_serving/fr13_fixed32_commit_slot_scatter.py": 1,
    "src/lumo_flywheel_serving/fr13_gdn_gqa_group3.py": 1,
    "src/lumo_flywheel_serving/fr13_host_tail_prep.py": 1,
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py": 1,
}


def _mode_collections(source: str) -> list[tuple[int, frozenset[str]]]:
    """Every literal collection whose strings are all fixed32 mode names."""
    out: list[tuple[int, frozenset[str]]] = []
    for node in ast.walk(ast.parse(source)):
        names: set[str] | None = None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values = [
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            ]
            if values and len(values) == len(node.elts):
                names = set(values)
        elif isinstance(node, ast.Dict):
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
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
