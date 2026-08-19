#!/usr/bin/env python3
"""Every place that asks "which fixed32 profile is this?" must know all of them.

The same question gets asked in three kinds of place, and each one has now
refused a serve on its own:

  * the CONSUMERS (topology, drafter, census, patcher)      -- rounds 12 and back
  * the SELECTOR / vehicle dispatch                          -- round 13
  * the launcher's IN-CONTAINER PREFLIGHT                    -- round 14

Round 14 also showed that fixing a mode TABLE is not enough: three lines later
the same block compared the dispatched tree against `topology.FIXED32_CHOICES`
unconditionally, and hydra31's tree genuinely differs, so the refusal simply
moved down. A profile-parity detector therefore has to find two things:

  1. every dict keyed by fixed32 mode  -> must have a row for every profile;
  2. every comparison against a PROFILE-VARYING topology constant -> must be
     keyed on the mode, not made unconditionally.

Both are answered from `fr13_fixed32_topology.PROFILES`, so this fails on the
NEXT profile anyone adds rather than on the next boot.

Run:  python3 scripts/fr14_mode_table_parity.py [--json out.json]
Exit 1 if any site cannot admit every profile.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fr13_fixed32_topology as topology  # noqa: E402

# THE launcher-family roster, and the only place it is written down.
#
# "both launcher families" was wrong by one for six rounds: fr14_leg3 is a live
# serving path (arm B's profile-chain legs) and had none of the FR14 work --
# including the PROMOTED fused-topk default, so it would have served the unfused
# kernel silently. Everything that iterates launchers imports this, so a fourth
# family joins every detector at once.
LAUNCHER_FAMILIES = (
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
    "scripts/fr14_leg3_launch_nomiddleware.sh",
)
# the serve vehicle asks the same profile question and is scanned with them
SHELL_SITES = LAUNCHER_FAMILIES + (
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
)

# Markers that must appear in EVERY launcher family, with the same count. A
# promoted default or a safety guard present in two of three is the defect this
# roster exists to catch.
FAMILY_PARITY_MARKERS = (
    "FR14_FUSED_DRAFT_TOPK",            # promoted default ON
    "_fr14_fused_topk_sha_default",     # its pinned credential
    "FR14_SUFFIX_PASS_GATE",            # refused-final, but guarded
    "_fr14_gate_incompat",              # gate x draft-head refusals
    "FR14_GATE_SPLIT_GRAPH",            # the split-graph interlock
    "_fr14_h31_incompat",               # hydra31 x hydra27-qualified levers
    "hydra31_fixed32",                  # the tail10 profile
    "gqa_pair_splitk",                  # lane 4's arm
    # The PROMOTED DEFAULT (Mark, pass 100). Every literal the default boot
    # arms itself from is checked identical across all three families, because
    # the two-of-three lesson has now been paid for twice: fr14_leg3 drifted a
    # whole arm behind in the paired-contract family, and this file's own
    # LAUNCHER list was a 2-tuple when the universal resolver test counted its
    # answers.
    "_FR13_SPLITK_DEFAULT_ARM",         # the arm the default arms
    "_FR13_SPLITK_DEFAULT_SO_SHA256",   # its pinned binary
    "_FR13_SPLITK_DEFAULT_SO_SIZE",
    "_FR13_SPLITK_DEFAULT_CLOSURE",     # its source closure
    "_FR13_SPLITK_DEFAULT_SASS",        # the kernel digest ...
    "_FR13_SPLITK_DEFAULT_BASELINE_SASS",   # ... and the sealed baseline
    "_FR13_SPLITK_DEFAULT_CREDENTIAL",  # the staged tier-b credential
    "SPLIT-K IS THE PROMOTED DEFAULT",  # the block itself
    "staged binary missing or not the pinned kernel",   # its hard refusal
    "must not silently serve the incumbent",            # and why it is hard
    "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST",        # the host/container split
    # F1/F2 (severity-1, pass 106). Arming a selector is not surviving it: the
    # promoted default set TIER_B_ARM and none of the provenance the selector
    # gate 950 lines later reads, so every plain hydra27 B1 boot exited 2 and
    # the default had never once served. Two of three is exactly how that gets
    # re-introduced -- the fix landed in the canonical launcher first and the
    # twins were a separate edit -- so each half is pinned here by count.
    "cannot mint the selector provenance",              # F1: the mint's guard
    # Keyed on the MINT's own assignment, not on the bare git idiom: the
    # canonical launcher also runs it for the gqa_pair default's serviceability
    # probe, which the twins do not have, and a marker that counts both would
    # report a parity failure that is not one.
    "FR13_FA2_QROW32_B1_SOURCE_COMMIT:-$(git rev-parse HEAD",   # F1: the mint
    "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256:-$(sha256sum",      # ... both halves
    "_fr13_b1_commit_bound",                            # F1: the reconciliation
    "requires a credential earned at this HEAD",        # ... tier-A keeps it
    "tier-b selector requires a well-formed source commit",  # ... tier-B weaker
    "STANDS DOWN",                                      # F2: the arbitration
    # SITE 12 (pass 113). The vocab-profile conversion, keyed on each CALL's
    # own shape rather than on the bare helper name -- the forks carry a
    # comment mentioning the helper, so a name-count marker would have read 7
    # against production's 6 and reported a divergence that is not one, which
    # is how a marker gets deleted for crying wolf. These are exact call
    # sites; the structural detector below catches the ones nobody listed.
    '"$FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE" "FR13 qrow32 B1 selector" || exit 2',
    '"$FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE" "FR13 qrow32 B4 GQA-pair" || exit 2',
    '"$FR13_FIXED32_GDN_GQA_GROUP3_QUALIFICATION_PROFILE" "FR13 GDN GQA-group3 production" || exit 2',
    '"$FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE" "FR13 GDN single-launch production" || exit 2',
    '"$FR13_FIXED32_GDN_LIVE_GATE_QUALIFICATION_PROFILE" \\',
    # ... and the two profile variables the forks never gained, without which
    # the calls above would expand to the empty string and refuse everything.
    "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=${FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE:-k64_root}",
    "FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE=${FR13_FA2_QROW32_B4_QUALIFICATION_PROFILE:-k64_root}",
)


def scan_family_parity():
    """Any FR14 marker that is not identical across every launcher family."""
    bad = []
    texts = {rel: (REPO / rel).read_text() for rel in LAUNCHER_FAMILIES}
    for marker in FAMILY_PARITY_MARKERS:
        counts = {rel: text.count(marker) for rel, text in texts.items()}
        if len(set(counts.values())) != 1:
            missing = [Path(r).name for r, c in counts.items() if c == 0]
            bad.append(
                f"{marker}: counts differ across families {counts}"
                + (f" -- absent from {missing}" if missing else "")
            )
    return bad


# ===========================================================================
# SITE 12 (FR14 pass 113): the STRUCTURAL twin check.
#
# The marker roster above is ENUMERATED: it sees only the divergences it was
# told about, so a generalization that predates its marker is invisible
# forever. Site 12 is what that costs. Production converted five levers from a
# hard-coded `ROOT==1 && K==65536 && BLOCKS==<pinned>` predicate to
# `_fr13_assert_draft_vocab_profile`, which admits full_vocab as well; the
# no-middleware forks took ONE of the five and kept the hardcode on the other
# four, making K0 full-vocab structurally impossible in the forks -- and K0
# full-vocab is the only shape split-K has ever served in. The forks were
# CURRENT on that night's F1/F2 work and STALE on the earlier generalization
# at the same time. Selective staleness is not something a per-marker roster
# can be relied on to find, because the roster is written after each miss.
#
# This detector is not told any lever's name. It discovers refusal regions by
# walking the launchers -- every `[[ ... ]] || { echo "..."; exit 2; }` -- and
# compares the SHAPE of each region across the three families:
#
#   * how many inline draft-vocab identity clauses it hard-codes, and
#   * whether the profile helper is called immediately above it.
#
# Regions are keyed by a PREFIX of their refusal message, deliberately short.
# Keying on the whole message is what let site 12 hide: the fork's message
# still said "K64/root1" where production's no longer did, so the two regions
# did not look like the same region at all. The prefix is the part that
# survives a predicate change; the tail is the part that records it.
#
# Levers that exist in only one family are IGNORED (the forks are forks). Only
# a lever present in two or more families, whose shape disagrees, is reported.
# ===========================================================================

# An inline draft-vocab identity clause: the thing the helper replaced.
_VOCAB_CLAUSE = re.compile(
    r'^\s*&&\s+"\$\{?FR13_DRAFT_VOCAB_(?:K|ROOT|BLOCKS)[^"]*"?\}?"\s*==\s*"[^"]*"\s*\\$'
)
_REFUSAL = re.compile(r'^\s*echo "(FR13[^"]{12,})" >&2\s*$')
_PREDICATE = re.compile(r"^\s*\[\[ ")
# Short enough to survive the predicate change that the divergence IS.
_KEY_CHARS = 40


def _vocab_regions(text):
    """Every refusal region, keyed by (message prefix, ORDINAL) -> shape.

    The ordinal is not decoration. A 40-char prefix is short enough to survive
    the predicate change that the divergence IS, which means it is also short
    enough to collide: "FR13 GDN GQA-group3 production requires " prefixes both
    the lever's own refusal and, twenty lines later, its live-PASS-JSON
    refusal. A plain dict silently kept the LAST one, and the first version of
    this detector reported two divergences where there were four -- it found
    site 12 and missed two more of exactly the same kind, which is the failure
    mode it exists to prevent. Key on (text, ordinal), never on text.
    """
    lines = text.split("\n")
    regions = {}
    seen = {}
    for i, line in enumerate(lines):
        match = _REFUSAL.match(line)
        if not match:
            continue
        if not any(lines[j].strip() == "exit 2" for j in (i + 1, i + 2)):
            continue
        start = None
        for j in range(i, max(i - 80, -1), -1):
            if _PREDICATE.match(lines[j]):
                start = j
                break
        if start is None:
            continue
        prefix = match.group(1)[:_KEY_CHARS]
        seen[prefix] = seen.get(prefix, 0) + 1
        key = (prefix, seen[prefix])
        above = "\n".join(lines[max(start - 4, 0):start])
        regions[key] = {
            "line": start + 1,
            "hardcoded_vocab_clauses": sum(
                1 for j in range(start, i) if _VOCAB_CLAUSE.match(lines[j])
            ),
            "calls_profile_helper": "_fr13_assert_draft_vocab_profile" in above,
        }
    return regions


def scan_vocab_profile_parity():
    """Refusal regions whose draft-vocab SHAPE disagrees across families.

    Non-enumerated: nothing here names a lever. A lever converted in one
    family and not another is reported whether or not anyone remembered to
    add a marker for it.
    """
    bad = []
    per_family = {
        rel: _vocab_regions((REPO / rel).read_text())
        for rel in LAUNCHER_FAMILIES
    }
    keys = set()
    for regions in per_family.values():
        keys |= set(regions)
    for key in sorted(keys):
        present = {
            rel: regions[key]
            for rel, regions in per_family.items()
            if key in regions
        }
        if len(present) < 2:
            continue  # a fork-only or production-only lever, not a divergence
        shapes = {
            (v["calls_profile_helper"], v["hardcoded_vocab_clauses"])
            for v in present.values()
        }
        if len(shapes) == 1:
            continue
        detail = ", ".join(
            f"{Path(rel).name}:{v['line']}"
            f" helper={'Y' if v['calls_profile_helper'] else 'N'}"
            f" hardcoded={v['hardcoded_vocab_clauses']}"
            for rel, v in sorted(present.items())
        )
        prefix, ordinal = key
        bad.append(
            f"{prefix!r} (refusal #{ordinal}): draft-vocab shape differs "
            f"across families -- {detail}"
        )
    return bad


# ===========================================================================
# SITE 14 (FR14 pass 118): the LITERAL-TABLE projection.
#
# Measurement 1 died a second time, past the site-12 fix, on
#
#   fixed32 requires FR13_MANDATORY_WEIGHT_BYTES=37335563648, got 25430574256
#
# fr14_leg3's mandatory-weight-bytes / weight-floor table still carried the
# PRE-NVFP4-PORT checkpoint on all three vocabulary rows -- 37.3GB fp8-era
# against the port's 25.4GB -- while production and the armb twin were
# byte-identical and current. Same fork, same selective staleness, third boot
# lost to it.
#
# Site 12's projection asked a POLICY question (does this region delegate the
# identity or hard-code it?). This one asks a VALUE question: a constant that
# must track a shared authority, carried as a literal in three places. Same
# shape, different projection -- which is the point of projections: the next
# class gets a new one, not a marker per site.
#
# The projection is exact rather than heuristic, and that is measured, not
# assumed. Over the three families:
#
#   NAME=<number>                    65 keys, 0 single-family
#   NAME=${NAME:-<number>}           76 keys, 0 single-family
#   "$NAME" == "<number>"           594 keys, 0 single-family
#   NAME=<64 hex chars>               7 keys, 0 single-family
#
# 742 shared keys, NOT ONE of which exists in only one family. The forks are
# forks in their plumbing, not in their constants, so cross-family equality is
# the right rule here and a divergence is always a defect -- unlike the
# selector-gate predicates, where the forks legitimately differ and only a
# shape projection is meaningful.
#
# Keyed on (name, ORDINAL) for the same reason as site 12: the floor table
# assigns _fixed32_expected_mandatory_weight_bytes three times, once per
# vocabulary row, and keying on the name alone would compare row 3 with row 3
# and call the other two clean.
#
# WHAT THIS DOES NOT COVER, stated as a floor rather than implied as
# completeness: literals inside arrays, `case` patterns, heredocs, and
# multi-line `[[ ]]` continuations whose operand is not a bare quoted number.
# ===========================================================================

_LITERAL_PROJECTIONS = (
    # (label, pattern, uses_search)
    ("assignment",
     re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?)\s*$"),
     False),
    ("default",
     re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=\$\{\1:-(-?\d+(?:\.\d+)?)\}\s*$"),
     False),
    ("comparison",
     re.compile(r'"\$\{?([A-Za-z_]\w*)(?::-[^}]*)?\}?"\s*==\s*"(-?\d+(?:\.\d+)?)"'),
     True),
    ("digest",
     re.compile(r"\b([A-Za-z_]\w*)=([0-9a-f]{64})\b"),
     True),
)


def _literal_table(text):
    """(projection, name, ordinal) -> literal, for one launcher."""
    table = {}
    seen = {}
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            continue
        for label, pattern, use_search in _LITERAL_PROJECTIONS:
            match = pattern.search(line) if use_search else pattern.match(line)
            if not match:
                continue
            name, value = match.group(1), match.group(2)
            key = (label, name)
            seen[key] = seen.get(key, 0) + 1
            table[(label, name, seen[key])] = value
    return table


def scan_literal_table_parity():
    """Constants that must track a shared authority but disagree by family."""
    bad = []
    tables = {
        rel: _literal_table((REPO / rel).read_text())
        for rel in LAUNCHER_FAMILIES
    }
    keys = set()
    for table in tables.values():
        keys |= set(table)
    for key in sorted(keys):
        present = {rel: t[key] for rel, t in tables.items() if key in t}
        if len(present) < 2:
            continue
        if len(set(present.values())) == 1:
            continue
        label, name, ordinal = key
        detail = ", ".join(
            f"{Path(rel).name}={value}" for rel, value in sorted(present.items())
        )
        bad.append(
            f"{name} [{label} #{ordinal}]: literal differs across families "
            f"-- {detail}"
        )
    return bad

# Topology names whose VALUE differs between profiles. Comparing one of these
# unconditionally is the round-14 defect, whatever the mode table says.
PROFILE_VARYING = frozenset({
    "FIXED32_CHOICES", "TAIL10_CHOICES",
    "WALK_CAP", "TAIL10_WALK_CAP",
    "MAX_PHYSICAL_DEPTH", "TAIL10_MAX_PHYSICAL_DEPTH",
    "PHYSICAL_PARENT", "PHYSICAL_PARENT_SHA256",
    "TREE_ANCESTRY_SHA256",
    "ARCTIC_MAIN_TAIL_LENGTH", "ARCTIC_LOOKUP_TOKENS_PER_REQUEST",
    "RESCUE_CARRY_SLOTS_PER_REQUEST", "PHYSICAL_BRANCH_CHAINS",
    "SUBTREE_LEVELS", "GDN_LEVEL_PATH_COUNTS", "GDN_LEVEL_MAX_LENGTHS",
    "GDN_PATH_PROGRAMS", "GDN_PADDED_SLOTS",
})

# Names that are the SAME in every profile -- safe to compare unconditionally.
PROFILE_INVARIANT = frozenset({
    "PHYSICAL_DRAFTS", "PHYSICAL_ROWS", "SAMPLER_MAX_FANOUT",
    "COMMIT_PATH_CAP", "MODEL_LAYERS", "TREE_ATTENTION_LAYERS", "GDN_LAYERS",
})

MODE_NAMES = frozenset({"tail6_fixed32", "hydra27_fixed32", "hydra31_fixed32"})


def embedded_python(text: str):
    """Every heredoc'd python block in a shell file, with its start offset."""
    out = []
    for m in re.finditer(r"<<'PY'\n(.*?)\nPY\n", text, re.S):
        out.append((m.start(1), m.group(1)))
    return out


def _profiles() -> set[str]:
    return set(topology.PROFILES)


def scan_mode_tables(path: Path):
    """Dicts keyed by fixed32 mode that do not admit every profile."""
    bad = []
    for offset, block in embedded_python(path.read_text()):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if not (keys & MODE_NAMES):
                continue
            missing = _profiles() - keys
            if missing:
                bad.append(
                    f"{path.name}: mode table at block+{node.lineno} "
                    f"missing {sorted(missing)}"
                )
    return bad


def scan_unconditional_comparisons(path: Path):
    """Comparisons against a profile-varying topology constant.

    A comparison is acceptable when the other side is derived from the armed
    profile (a `profile(...)` lookup or a `_profile[...]` subscript); it is a
    defect when it names the constant directly, because that constant belongs to
    exactly one profile.
    """
    bad = []
    for offset, block in embedded_python(path.read_text()):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        lines = block.split("\n")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for side in [node.left, *node.comparators]:
                name = None
                if isinstance(side, ast.Attribute):
                    name = side.attr
                elif isinstance(side, ast.Name):
                    name = side.id
                if name is None or name not in PROFILE_VARYING:
                    continue
                src = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
                bad.append(
                    f"{path.name}: unconditional compare against "
                    f"topology.{name} -- {src[:70]}"
                )
    return bad


def scan_shell_whitelists(path: Path):
    """Bash KIND / FR13_FIXED32_MODE whitelists that omit a profile.

    Only lines that already enumerate more than one fixed32 mode are treated as
    whitelists; a single-mode line is a lever precondition (hydra27-qualified
    levers legitimately refuse hydra31) and is left alone.
    """
    bad = []
    for raw in path.read_text().split("\n"):
        if raw.lstrip().startswith("#"):
            continue
        # match a mode as a whole token however it is written: quoted
        # ("$KIND" == "hydra31_fixed32") or as a case pattern
        # (""|tail6_fixed32|hydra27_fixed32|hydra31_fixed32)).
        named = {
            m for m in MODE_NAMES
            if re.search(rf"(?<![\w-]){re.escape(m)}(?![\w-])", raw)
        }
        if len(named) < 2:
            continue
        missing = _profiles() - named
        if missing:
            bad.append(
                f"{path.name}: whitelist omits {sorted(missing)} -- "
                f"{raw.strip()[:80]}"
            )
    return bad


def sweep():
    rows = []
    for rel in SHELL_SITES:
        path = REPO / rel
        if not path.exists():
            continue
        for kind, finder in (
            ("mode-table", scan_mode_tables),
            ("unconditional-compare", scan_unconditional_comparisons),
            ("shell-whitelist", scan_shell_whitelists),
        ):
            for problem in finder(path):
                rows.append({"kind": kind, "file": rel, "detail": problem})
    for problem in scan_family_parity():
        rows.append(
            {"kind": "family-parity", "file": "<launcher families>",
             "detail": problem}
        )
    for problem in scan_vocab_profile_parity():
        rows.append(
            {"kind": "vocab-profile-parity", "file": "<launcher families>",
             "detail": problem}
        )
    for problem in scan_literal_table_parity():
        rows.append(
            {"kind": "literal-table-parity", "file": "<launcher families>",
             "detail": problem}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = sweep()
    profiles = sorted(_profiles())
    print(f"profiles: {profiles}")
    print(f"shell sites scanned: {len(SHELL_SITES)}")
    for r in rows:
        print(f"  [STALE] {r['kind']:22s} {r['detail']}")
    print(f"\n{len(rows)} site(s) cannot admit every profile")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "schema": "fr14.mode_table_parity.v1",
                    "profiles": profiles,
                    "sites": list(SHELL_SITES),
                    "stale": rows,
                },
                indent=1,
            )
            + "\n"
        )
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
