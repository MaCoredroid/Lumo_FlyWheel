"""The ordered-GDN single_launch gate can be earned in the shape it will SERVE.

FR14. Under Mark's K0 production ruling the served B1 config is full_vocab, but
the ordered-GDN live gate was pinned to K64/root1 in two places -- a launcher
clause and the shared runner. A credential earned in a shape production never
serves is the mislabelled-evidence class this campaign keeps closing, so both
sites are RE-PINNED to a declared identity rather than unpinned.

Both default to k64_root, so all four banked ordered-GDN entrypoints
(fr13_run_b1_/b4_hydra27_/b4_tail23_/…) reproduce byte-for-byte.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = REPO / "scripts" / "fr13_run_gdn_single_launch_live_gate.sh"
B1_SHIM = REPO / "scripts" / "fr13_run_b1_gdn_single_launch_live_gate.sh"
LAUNCHER_TEXT = LAUNCHER.read_text(encoding="utf-8")
RUNNER_TEXT = RUNNER.read_text(encoding="utf-8")

PROFILE_BLOCK_RE = re.compile(
    r"^FR13_GDN_GATE_DRAFT_VOCAB_PROFILE=.*?"
    r"^export FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE=[^\n]*\n",
    re.S | re.M,
)


def _resolve(profile: str | None) -> dict[str, str]:
    """Execute the runner's real identity block and report what it exports."""
    match = PROFILE_BLOCK_RE.search(RUNNER_TEXT)
    assert match, "gate draft-vocabulary profile block not found"
    script = (
        "set -uo pipefail\n"
        + match.group(0)
        + 'printf "K=%s\\nROOT=%s\\nBLOCKS=%s\\nNEEDS=%s\\nBYTES=%s\\nFLOOR=%s\\n" '
        '"$FR13_DRAFT_VOCAB_K" "$FR13_DRAFT_VOCAB_ROOT" "$FR13_DRAFT_VOCAB_BLOCKS" '
        '"${FR13_NEEDS_ALLOW:-}" "$GATE_EXPECTED_WEIGHT_BYTES" "$GATE_EXPECTED_FLOOR_MS"\n'
    )
    env = {"PATH": "/usr/bin:/bin"}
    if profile is not None:
        env["FR13_GDN_GATE_DRAFT_VOCAB_PROFILE"] = profile
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        return {"_rc": str(proc.returncode), "_err": proc.stderr.strip()}
    out = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    out["_rc"] = "0"
    return out


# --------------------------------------------------------------------------
# regression safety: the banked entrypoints must not move
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [None, "k64_root"])
def test_the_default_reproduces_the_banked_k64_gate_exactly(profile) -> None:
    got = _resolve(profile)
    assert got["_rc"] == "0", got
    assert got["K"] == "65536"
    assert got["ROOT"] == "1"
    assert got["NEEDS"] == ""
    # the exact constants the runner asserted before this train
    assert got["BYTES"] == "25210209416"
    assert got["FLOOR"] == "92.345089436"


def test_full_vocab_serves_the_k0_identity_and_the_arm_b_floor() -> None:
    got = _resolve("full_vocab")
    assert got["_rc"] == "0", got
    assert got["K"] == "0"
    assert got["ROOT"] == "0"
    assert got["NEEDS"] == "FR13_DRAFT_VOCAB_K=0", "sanctioned override missing"
    assert got["BYTES"] == "25430574256"
    assert got["FLOOR"] == "93.15228665201465"


def test_an_unknown_profile_is_refused() -> None:
    got = _resolve("k64")  # plausible typo
    assert got["_rc"] != "0"
    assert "must be k64_root or full_vocab" in got["_err"]


def test_the_block_map_is_canonical_in_both_profiles() -> None:
    """Production carries the canonical map; a gate that dropped it would not be
    earned in the production shape."""
    for profile in ("k64_root", "full_vocab"):
        got = _resolve(profile)
        assert got["BLOCKS"] == "/workspace/scripts/fr13_dvk_subset_blocks.json"


# --------------------------------------------------------------------------
# the launcher clause
# --------------------------------------------------------------------------


def test_the_launcher_gate_clause_no_longer_hardcodes_k64() -> None:
    assert "FR13 ordered GDN live gate requires exact K64/root1" not in LAUNCHER_TEXT


def test_the_launcher_gate_clause_asserts_the_whole_identity_via_the_helper() -> None:
    """Two of four fields is not an identity: the helper checks root, K, the
    block map and the sanctioned override, and is the same one every other lever
    uses -- so runner and launcher cannot disagree about what a profile means."""
    assert "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" in LAUNCHER_TEXT
    assert (
        '_fr13_assert_draft_vocab_profile \\\n'
        '\t      "$FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" \\\n'
        '\t      "FR13 ordered GDN live gate" || exit 2' in LAUNCHER_TEXT
    )


def test_the_launcher_gate_profile_defaults_to_k64_root() -> None:
    assert (
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE="
        "${FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE:-k64_root}"
        in LAUNCHER_TEXT
    )


# --------------------------------------------------------------------------
# the runner reports what it served
# --------------------------------------------------------------------------


def test_the_launcher_meta_reports_the_served_identity_not_a_literal() -> None:
    """A K0 gate that announced draft_vocab_k=65536 is exactly the credential
    self-misdescription the qrow32 chain already had to re-run to fix."""
    assert "draft_vocab_k=65536\\ndraft_vocab_root=1" not in RUNNER_TEXT
    assert 'draft_vocab_k=\'"$FR13_DRAFT_VOCAB_K"\'' in RUNNER_TEXT
    assert 'draft_vocab_root=\'"$FR13_DRAFT_VOCAB_ROOT"\'' in RUNNER_TEXT


def test_the_serve_env_carries_the_declared_identity_and_the_override() -> None:
    assert (
        'FR13_DRAFT_VOCAB_K="$FR13_DRAFT_VOCAB_K" '
        'FR13_DRAFT_VOCAB_ROOT="$FR13_DRAFT_VOCAB_ROOT"' in RUNNER_TEXT
    )
    assert 'FR13_NEEDS_ALLOW="${FR13_NEEDS_ALLOW:-}"' in RUNNER_TEXT


def test_the_runner_propagates_the_profile_to_both_launcher_gates() -> None:
    """The gate path and the production selector must qualify under the SAME
    declared identity, or a gate could be earned under one and spent under
    another."""
    for var in (
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE",
    ):
        assert f'export {var}="$FR13_GDN_GATE_DRAFT_VOCAB_PROFILE"' in RUNNER_TEXT


def test_the_b1_entrypoint_needs_no_fork_for_k0() -> None:
    """The K0 B1 gate is the existing B1 shim plus one env var -- no new runner,
    so there is no second copy to drift."""
    shim = B1_SHIM.read_text(encoding="utf-8")
    assert "FR13_GDN_GATE_MODE=hydra27_fixed32" in shim
    assert "FR13_GDN_GATE_BATCH=1" in shim
    assert "exec bash" in shim
    assert "FR13_GDN_GATE_DRAFT_VOCAB_PROFILE" not in shim, (
        "the shim must not pin a profile; the caller declares it"
    )


def test_both_scripts_parse() -> None:
    for script in (LAUNCHER, RUNNER, B1_SHIM):
        assert (
            subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, check=False
            ).returncode
            == 0
        ), f"{script.name} does not parse"


# --------------------------------------------------------------------------
# the fourth site: the in-container patcher's env contract
# --------------------------------------------------------------------------

PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PATCHER_TEXT = PATCHER.read_text(encoding="utf-8")


def test_the_patcher_gate_contract_no_longer_hardcodes_k64() -> None:
    """The fourth site, found by a 70-second boot refusal rather than by reading.

    The launcher clause, the gate runner and the credential validator all carried
    the declared identity, and the gate STILL refused -- inside the container, at
    the patcher's own env contract, which pinned ROOT=1/K=65536 for every
    ordered-GDN candidate. Every layer of this stack checks the identity
    independently, which is why re-pointing one is never finished until a real
    boot says so.
    """
    # Both single_launch contracts -- the GATE's and the PRODUCTION arm's -- are
    # separate sites and both had to move. Other levers' K64 pin sets
    # (gqa_group3, SFWD fusion, the draft-head U8 bridge) are deliberately left
    # alone: they are correct for their own arms.
    for marker in ("exact_single_launch = {", "exact_single_launch_production = {"):
        idx = PATCHER_TEXT.index(marker)
        block = PATCHER_TEXT[idx : idx + 400]
        assert '"FR13_DRAFT_VOCAB_K": "65536"' not in block, (
            f"{marker} still hardcodes K64"
        )
        assert "**_" in block, f"{marker} does not splice a declared profile"
    assert "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" in PATCHER_TEXT
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE" in PATCHER_TEXT


def test_the_patcher_gate_contract_offers_both_profiles_and_refuses_others() -> None:
    assert '"k64_root": {"FR13_DRAFT_VOCAB_ROOT": "1", "FR13_DRAFT_VOCAB_K": "65536"}' in (
        PATCHER_TEXT
    )
    assert '"full_vocab": {"FR13_DRAFT_VOCAB_ROOT": "0", "FR13_DRAFT_VOCAB_K": "0"}' in (
        PATCHER_TEXT
    )
    assert "must be \\nk64_root or full_vocab" in PATCHER_TEXT.replace(
        '"\n                f"', "\\n"
    ) or "k64_root or full_vocab" in PATCHER_TEXT


def test_the_patcher_gate_contract_defaults_to_k64_root() -> None:
    """Banked ordered-GDN gates must be unaffected."""
    assert (
        'os.environ.get(\n            "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE", "k64_root"\n        )'
        in PATCHER_TEXT
    )


def test_all_four_sites_are_covered() -> None:
    """One list, so the next person can see the whole train at once."""
    sites = {
        "launcher gate clause": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in LAUNCHER_TEXT,
        "shared gate runner": "FR13_GDN_GATE_DRAFT_VOCAB_PROFILE" in RUNNER_TEXT,
        "credential validator": "DRAFT_VOCAB_PROFILES"
        in (REPO / "scripts" / "fr13_gdn_single_launch_production_credential.py").read_text(
            encoding="utf-8"
        ),
        "in-container patcher": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in PATCHER_TEXT,
    }
    missing = [name for name, ok in sites.items() if not ok]
    assert not missing, f"draft-vocabulary profile missing from: {missing}"


# --------------------------------------------------------------------------
# sites 6-13: the kernel's own predicates, and the credential CONTENT
#
# The first five sites were bash and patcher env contracts. These are the ones
# that decide, inside the EngineCore worker, whether the folded kernel may run
# at all -- and the one that decides what the credential SAYS. Every check is
# kept; none is re-pointed away. Both shapes are expressible and neither can
# masquerade as the other, because the profile NAME and the K/root pair it
# implies are required to agree.
# --------------------------------------------------------------------------

import ast  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

KERNEL = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
KERNEL_TEXT = KERNEL.read_text(encoding="utf-8")
GATE = REPO / "scripts" / "fr13_gdn_single_launch_gate.py"
GATE_TEXT = GATE.read_text(encoding="utf-8")
SL_VALIDATOR = REPO / "scripts" / "fr13_gdn_single_launch_production_credential.py"
GQA3_VALIDATOR = REPO / "scripts" / "fr13_gdn_gqa_group3_production_credential.py"

# The exact values the K64 sites used to bake, and the K0 values Mark's ruling
# makes production. Regression is measured against THESE, not against a
# re-derivation of them.
K64_ENV = {"FR13_DRAFT_VOCAB_ROOT": "1", "FR13_DRAFT_VOCAB_K": "65536"}
K0_ENV = {"FR13_DRAFT_VOCAB_ROOT": "0", "FR13_DRAFT_VOCAB_K": "0"}
K64_CREDENTIAL = {"draft_vocab_k": 65536, "draft_vocab_root": 1}
K0_CREDENTIAL = {"draft_vocab_k": 0, "draft_vocab_root": 0}


def _kernel_namespace(*wanted: str) -> dict[str, object]:
    """Execute the kernel's REAL profile units, not a re-typed copy of them."""
    tree = ast.parse(KERNEL_TEXT)
    constants = {
        "_FR13_DRAFT_VOCAB_PROFILES",
        "_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS",
        "_FR13_GDN_ORDERED_CANDIDATES",
        "_FR13_FIXED32_MODES",
    }
    functions = {
        "_fr13_draft_vocab_profile",
        "_fr13_draft_vocab_env_matches",
        "_fr13_draft_vocab_credential_matches",
        *wanted,
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    namespace: dict[str, object] = {"os": os, "json": json}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL,
            "exec",
        ),
        namespace,
    )
    return namespace


def test_the_kernel_tables_are_the_banked_values_in_both_shapes() -> None:
    namespace = _kernel_namespace()
    assert namespace["_FR13_DRAFT_VOCAB_PROFILES"] == {
        "k64_root": K64_ENV,
        "full_vocab": K0_ENV,
    }
    assert namespace["_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS"] == {
        "k64_root": K64_CREDENTIAL,
        "full_vocab": K0_CREDENTIAL,
    }


@pytest.mark.parametrize(
    "variable",
    (
        "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE",
        "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE",
    ),
)
def test_every_kernel_lever_defaults_to_k64_root_and_refuses_typos(
    variable: str,
) -> None:
    """Absent and empty both mean the banked identity; a typo means refusal.

    Empty-is-default matches the launcher's `${VAR:-k64_root}` exactly, so an
    env sweeper that forwards the name with an empty value cannot turn a banked
    arm into an import-time raise. A non-empty unknown is never defaulted --
    silently defaulting a declaration is how a lever ends up serving one
    identity while claiming another.
    """
    resolve = _kernel_namespace()["_fr13_draft_vocab_profile"]
    assert resolve(variable, environ={}) == "k64_root"
    assert resolve(variable, environ={variable: ""}) == "k64_root"
    assert resolve(variable, environ={variable: "  "}) == "k64_root"
    assert resolve(variable, environ={variable: "full_vocab"}) == "full_vocab"
    for typo in ("k64", "K64_ROOT", "fullvocab", "0"):
        with pytest.raises(RuntimeError, match="k64_root or full_vocab"):
            resolve(variable, environ={variable: typo})


def test_the_env_predicate_is_no_weaker_than_the_literals_it_replaced() -> None:
    """Present and exact, in both shapes -- an absent variable is not a match."""
    matches = _kernel_namespace()["_fr13_draft_vocab_env_matches"]
    assert matches("k64_root", environ=K64_ENV)
    assert matches("full_vocab", environ=K0_ENV)
    # neither shape satisfies the other
    assert not matches("k64_root", environ=K0_ENV)
    assert not matches("full_vocab", environ=K64_ENV)
    # the pre-train literals refused these too
    assert not matches("k64_root", environ={})
    assert not matches("full_vocab", environ={})
    assert not matches("k64_root", environ={"FR13_DRAFT_VOCAB_ROOT": "1"})
    assert not matches(
        "k64_root", environ={**K64_ENV, "FR13_DRAFT_VOCAB_K": "32768"}
    )


def test_a_credential_cannot_masquerade_as_the_other_shape() -> None:
    """Name AND pair, together. Either alone is forgeable by omission."""
    matches = _kernel_namespace()["_fr13_draft_vocab_credential_matches"]
    k64 = {"qualification_profile": "k64_root", **K64_CREDENTIAL}
    k0 = {"qualification_profile": "full_vocab", **K0_CREDENTIAL}
    assert matches(k64, "k64_root")
    assert matches(k0, "full_vocab")
    assert not matches(k64, "full_vocab")
    assert not matches(k0, "k64_root")
    # a credential that NAMES full_vocab while carrying K64 values
    assert not matches({"qualification_profile": "full_vocab", **K64_CREDENTIAL},
                       "full_vocab")
    # a credential that carries the pair but declares nothing (pre-FR14 shape)
    assert not matches(dict(K64_CREDENTIAL), "k64_root")
    # JSON strings must not satisfy the int pair
    assert not matches(
        {"qualification_profile": "k64_root", "draft_vocab_k": "65536",
         "draft_vocab_root": 1},
        "k64_root",
    )
    # bool is an int subclass in Python; True must not pass as draft_vocab_root
    assert not matches(
        {"qualification_profile": "full_vocab", "draft_vocab_k": 0,
         "draft_vocab_root": False},
        "full_vocab",
    )


@pytest.mark.parametrize(
    ("resolver", "variable"),
    (
        (
            "_fr13_resolve_fixed32_gdn_single_launch",
            "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE",
        ),
        (
            "_fr13_resolve_fixed32_gdn_path_bv_candidate",
            "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE",
        ),
    ),
)
def test_the_kernel_gate_resolvers_arm_in_both_shapes(
    resolver: str, variable: str
) -> None:
    """The whole point of the port: the B1 shape production SERVES is armable."""
    namespace = _kernel_namespace(
        resolver, "_fr13_resolve_fixed32_gdn_path_bv_candidate"
    )
    arm = (
        {"FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "1"}
        if resolver == "_fr13_resolve_fixed32_gdn_single_launch"
        else {"FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "single_launch"}
    )
    resolve = namespace[resolver]
    assert resolve(
        "hydra27_fixed32",
        environ={**arm, **K64_ENV},
        sidecars=(),
        geom_override={"BV": 8},
    )
    assert resolve(
        "hydra27_fixed32",
        environ={**arm, **K0_ENV, variable: "full_vocab"},
        sidecars=(),
        geom_override={"BV": 8},
    )
    # declared one shape, served the other -> refused, in BOTH directions
    with pytest.raises(RuntimeError, match="full_vocab drafter contract"):
        resolve(
            "hydra27_fixed32",
            environ={**arm, **K64_ENV, variable: "full_vocab"},
            sidecars=(),
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="k64_root drafter contract"):
        resolve(
            "hydra27_fixed32",
            environ={**arm, **K0_ENV},
            sidecars=(),
            geom_override={"BV": 8},
        )


def test_no_kernel_ordered_predicate_still_bakes_the_k64_literal() -> None:
    """The eight enumerated sites, by absence of what they used to contain."""
    for stale in (
        'str(env.get("FR13_DRAFT_VOCAB_K", "")).strip() != "65536"',
        'credential.get("draft_vocab_k") != 65536',
        'credential.get("draft_vocab_root") != 1',
        "draft_vocab_k=65536",
        # a K64 pair standing ALONE as a record field, as opposed to the one
        # place it legitimately belongs: the k64_root row of the profile table
        '"draft_vocab_k": 65536,\n',
    ):
        assert stale not in KERNEL_TEXT, f"K64 literal survives: {stale}"
    assert (
        '"k64_root": {"draft_vocab_k": 65536, "draft_vocab_root": 1},'
        in KERNEL_TEXT
    ), "the k64_root row must still carry the banked values"
    # the committer BV64/4-warp lever is a DIFFERENT arm and keeps its own pin
    # (doctrine: a lever's identity is re-pointed only for the lever being
    # re-pointed), so exactly one K64 env literal remains and it is that one.
    assert KERNEL_TEXT.count('os.environ.get("FR13_DRAFT_VOCAB_K") != "65536"') == 1
    assert "committer BV64/4-warp requires exact Hydra27" in KERNEL_TEXT


def test_both_pass_emitters_declare_the_shape_they_ran_in() -> None:
    for marker in (
        "qualification_profile=draft_vocab_profile,",
        '"qualification_profile": draft_vocab_profile,',
    ):
        assert marker in KERNEL_TEXT
    assert (
        "**_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS[draft_vocab_profile],"
        in KERNEL_TEXT
    )
    # and the emitter re-checks the constant against the SERVED env
    assert KERNEL_TEXT.count(
        "_fr13_draft_vocab_env_matches(draft_vocab_profile)"
    ) == 2


# --------------------------------------------------------------------------
# the credential CONTENT and SCHEMA
# --------------------------------------------------------------------------


def test_the_gate_stamps_the_declared_profile_into_the_credential() -> None:
    assert '"qualification_profile": args.draft_vocab_profile,' in GATE_TEXT
    assert '"qualification_profile": draft_vocab_profile,' in GATE_TEXT
    assert '"--draft-vocab-profile",' in GATE_TEXT
    assert 'default="k64_root",' in GATE_TEXT
    assert "DRAFT_VOCAB_PROFILES" in GATE_TEXT
    # no baked literal survives in the records the credential is minted from;
    # the only surviving K64 pair is the k64_root row of the profile table
    assert '"draft_vocab_k": 65536,\n' not in GATE_TEXT
    assert '"draft_vocab_root": 1,\n' not in GATE_TEXT
    assert (
        '"k64_root": {"draft_vocab_k": 65536, "draft_vocab_root": 1},'
        in GATE_TEXT
    )


def test_the_runner_declares_the_same_profile_to_the_reducer_it_served() -> None:
    """Earned and reduced under ONE declaration, or the credential could name a
    shape the serve never ran."""
    assert (
        '--draft-vocab-profile "$FR13_GDN_GATE_DRAFT_VOCAB_PROFILE"'
        in RUNNER_TEXT
    )


def test_the_shape_change_bumped_the_schema_everywhere_at_once() -> None:
    """A record that gained a required field is a new shape; the version says
    so. Every consumer moves together or the bump is worse than useless."""
    v4 = "fr13.fixed32.gdn_single_launch.real_task_credential.v4"
    for text in (
        GATE_TEXT,
        KERNEL_TEXT,
        SL_VALIDATOR.read_text(encoding="utf-8"),
        GQA3_VALIDATOR.read_text(encoding="utf-8"),
    ):
        assert v4 in text
        assert "real_task_credential.v3" not in text
    assert "live_observation.v3" in GATE_TEXT and "live_observation.v2" not in GATE_TEXT
    assert "live_pass.v2" in KERNEL_TEXT
    # the NON-ordered BV live pass carries no draft-vocabulary claim and must
    # not be dragged along by a bump that does not apply to it
    assert "fr13.fixed32.gdn_path_bv.live_pass.v1" in KERNEL_TEXT


def test_both_validators_require_the_profile_the_caller_declared() -> None:
    for path in (SL_VALIDATOR, GQA3_VALIDATOR):
        text = path.read_text(encoding="utf-8")
        assert '"qualification_profile": draft_vocab_profile,' in text
        assert '"draft_vocab_k": draft_vocab["draft_vocab_k"],' in text
        assert '"draft_vocab_root": draft_vocab["draft_vocab_root"],' in text
        assert '"draft_vocab_k": 65536,\n' not in text, (
            f"{path.name} bakes K64 as a record field"
        )
        assert "--draft-vocab-profile" in text


def test_all_thirteen_sites_are_covered() -> None:
    """One list, so the next reader sees the whole train at once.

    Sites 1-5 were closed by 088b73c63 / f6dcc74b6 / 0eafeac4e. Sites 6-13 are
    this commit: the kernel's four resolvers (two of them credential-checking
    as well), both PASS emitters, and the credential content/schema itself.
    """
    patcher_text = PATCHER_TEXT
    gqa3_text = GQA3_VALIDATOR.read_text(encoding="utf-8")
    sl_text = SL_VALIDATOR.read_text(encoding="utf-8")
    sites = {
        "1 launcher gate clause": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in LAUNCHER_TEXT,
        "2 shared gate runner": "FR13_GDN_GATE_DRAFT_VOCAB_PROFILE" in RUNNER_TEXT,
        "3 single_launch credential validator": "DRAFT_VOCAB_PROFILES" in sl_text,
        "4 patcher gate contract": "FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE"
        in patcher_text,
        "5 patcher production contract": (
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE" in patcher_text
        ),
        "6 kernel single_launch campaign bool": (
            'FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE requires the exact "\n'
            in KERNEL_TEXT
            or "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE requires the exact" in KERNEL_TEXT
        ),
        "7 kernel ordered live-gate candidate": (
            "FR13 fixed32 GDN ordered live gate requires the exact" in KERNEL_TEXT
        ),
        "8 kernel gqa_group3 production env": (
            "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE" in KERNEL_TEXT
        ),
        "9 kernel gqa_group3 production credential": (
            KERNEL_TEXT.count("_fr13_draft_vocab_credential_matches(") >= 3
        ),
        "10 kernel single_launch production env": (
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE" in KERNEL_TEXT
        ),
        "11 kernel PASS emitters": (
            "_FR13_FIXED32_GDN_ORDERED_QUALIFICATION_PROFILE" in KERNEL_TEXT
        ),
        "12 gate credential content + schema": (
            '"qualification_profile": args.draft_vocab_profile,' in GATE_TEXT
        ),
        "13 gqa_group3 credential validator + patcher contract": (
            "DRAFT_VOCAB_PROFILES" in gqa3_text
            and "FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE" in patcher_text
        ),
    }
    missing = [name for name, ok in sites.items() if not ok]
    assert not missing, f"draft-vocabulary profile missing from: {missing}"
