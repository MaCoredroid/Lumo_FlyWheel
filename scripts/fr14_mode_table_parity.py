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
    # SITE 16 moved this half: the digest is minted from the credential's
    # sealed identity now, not by hashing the artifact the gate re-hashes.
    '"patch_source_sha256"[[:space:]]*:[[:space:]]*"[0-9a-f]\\{64\\}"',
    "patch_source_sha256 identity to mint from",
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
    # The tier-B canonical WORKLOAD table. exact16 is the QC that verifies the
    # split-K promotion, and until this table existed the promotion's own gate
    # could not admit it. Keyed on the resolver's call and on each row's name,
    # not on the pins -- the pins are what the rows differ BY.
    '_fr13_b1_tierb_workload_pins "$FR13_FA2_QROW32_B1_TIERB_WORKLOAD" || exit 2',
    "FR13_FA2_QROW32_B1_TIERB_WORKLOAD=${FR13_FA2_QROW32_B1_TIERB_WORKLOAD:-exact4}",
    "_FR13_B1_TIERB_WORKLOADS=\"exact4 exact16 exact16_minus_13236 exact16_qc_remainder_12 random1024_calibration\"",
    "config/fr13_fixed32/subset_b4_sixteen.json",
    "config/fr13_fixed32/subset_b4_four.json",
    # The exact16 resume set: fifteen tasks, its own name and its own file,
    # because declaring exact16 while serving fifteen is the pins-as-fiction
    # move the workload table exists to prevent.
    "config/fr13_fixed32/subset_b4_sixteen_minus_13236.json",
    "exact16_minus_13236",
    "config/fr13_fixed32/subset_b4_sixteen_qc_remainder_12.json",
    "exact16_qc_remainder_12",
    # The tier-B spelling and its refusing alias. The alias is the load-bearing
    # half: it is what lets a banked vehicle keep the legacy name while making
    # a boot that carries BOTH and disagrees refuse instead of picking one.
    "FR13_FA2_QROW32_B1_TIERB_TASK_IDS=${FR13_FA2_QROW32_B1_TIERB_TASK_IDS:-}",
    "FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=${FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256:-}",
    "are both set and disagree; set one",
    '"$_fr13_tierb_declared_ids" == "$_fr13_tierb_task_ids"',
    # SITE 15: arming is not owning. The withdrawal helper and the four owned
    # binary pins, keyed on the OWNING form -- a marker keyed on the name
    # alone would still match the ':-' spelling that burned.
    "_fr13_b1_withdraw_pointer_imports",
    "FR13_FA2_QROW32_B1_SO_SHA256=$_FR13_SPLITK_DEFAULT_SO_SHA256",
    "FR13_FA2_QROW32_B1_SO_SIZE=$_FR13_SPLITK_DEFAULT_SO_SIZE",
    "FR13_FA2_QROW32_B1_FA2_HEAD=$_FR13_SPLITK_DEFAULT_FA2_HEAD",
    "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=$_FR13_SPLITK_DEFAULT_CLOSURE",
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


# ===========================================================================
# SITE 15: the IMPORT-PRECEDES-OWNER projection.
#
# _fr13_b1_load_credential_pointer auto-imports the gqa_pair credential env
# whenever the pointer FILE EXISTS -- no arm named, no opt-in -- roughly 500
# lines before the promoted split-K default block, whose ${VAR:-literal}
# fallbacks only fill EMPTY variables. So exact16 armed the split-K default,
# watched gqa_pair stand down, and then had its selector gate measure
# split-K's 300,123,792-byte binary against the incumbent's imported
# 299,815,552.
#
# The general shape: an IMPORTER runs early and fills names conditionally; an
# OWNER runs later and fills the same names conditionally; whoever ran first
# wins, which is the opposite of what "owner" means. Neither half looks wrong
# on its own, and no cross-family comparison can see it -- all three families
# would have been equally wrong.
#
# So this projection is intra-file, not cross-family: for each name an
# importer may set, does a LATER owner try to claim it with a non-empty ':-'
# default? If so the owner does not own it, and the earliest importer decides
# what the boot serves.
#
# It is exact for the class it names: it reads the pointer's own whitelist, so
# a name added to that whitelist tomorrow is covered without anyone adding a
# marker. It is a floor for the general shape -- other importers with other
# whitelists would each need naming here.
# ===========================================================================

_POINTER_WHITELIST_START = '    case "$name" in\n'
_POINTER_WHITELIST_END = '      *)\n        echo "FR13 B1 credential pointer may not set $name" >&2\n'
# Two spellings of the same thing. The second matters: site 16 turned the
# patcher-digest mint into a MULTI-LINE `${VAR:-$(` ... `)}`, which the
# single-line pattern stopped matching -- the detector quietly lost sight of
# the very name it was written for. A projection that narrows when the code is
# reformatted is worse than none, because nothing announces the loss.
_OWNER_DEFAULT = re.compile(
    r"^\s*([A-Za-z_]\w*)=\$\{\1:-(?P<filler>.*)\}\s*$"
)
_OWNER_DEFAULT_MULTILINE = re.compile(
    r"^\s*([A-Za-z_]\w*)=\$\{\1:-(?P<filler>\$\(\s*)$"
)


def _pointer_imported_names(text):
    """The names the credential pointer is allowed to set, from its own case."""
    try:
        start = text.index(_POINTER_WHITELIST_START)
        end = text.index(_POINTER_WHITELIST_END, start)
    except ValueError:
        return set(), None
    # Only the case PATTERNS, not every FR13_ token in the block: the body
    # mentions _FR13_B1_POINTER_IMPORTED, and a whitelist that quietly
    # contained the bookkeeping array would be a projection reading its own
    # implementation.
    names = set()
    for line in text[start:end].split("\n"):
        match = re.match(r"^\s*\|?\s*(FR13_[A-Z0-9_]+)\s*(?:\\|\))\s*$", line)
        if match:
            names.add(match.group(1))
    call = text.index(
        '_fr13_b1_load_credential_pointer "$FR13_B1_CREDENTIAL_POINTER"'
    )
    return names, text[:call].count("\n") + 1


_WITHDRAWAL_CALL = "_fr13_b1_withdraw_pointer_imports\n"


def scan_import_precedes_owner():
    """Names an early importer fills that a later owner only ':-' defaults.

    An owner has exactly two legal ways to actually own a name it shares with
    an earlier importer:

      1. assign it UNCONDITIONALLY (no ':-'), or
      2. run after the imports have been WITHDRAWN.

    Both are accepted; anything else means the importer decides. Recognising
    the withdrawal is not a rubber stamp -- deleting the withdrawal call makes
    this fire again, which is proved by mutation in
    tests/test_fr14_mode_table_parity.py.
    """
    bad = []
    for rel in LAUNCHER_FAMILIES:
        text = (REPO / rel).read_text()
        names, call_line = _pointer_imported_names(text)
        if not names:
            continue  # this family has no importer
        lines = text.split("\n")
        for lineno, line in enumerate(lines, 1):
            if lineno <= call_line or line.lstrip().startswith("#"):
                continue
            match = (
                _OWNER_DEFAULT.match(line)
                or _OWNER_DEFAULT_MULTILINE.match(line)
            )
            if not match or match.group(1) not in names:
                continue
            filler = match.group("filler")
            if not filler:
                continue  # `${VAR:-}` is normalisation, not ownership
            withdrawn = any(
                l.strip() == _WITHDRAWAL_CALL.strip()
                for l in lines[call_line:lineno - 1]
            )
            if withdrawn:
                continue
            bad.append(
                f"{Path(rel).name}:{lineno} {match.group(1)} is filled by the "
                f"credential pointer at line {call_line} and only ':-' "
                f"defaulted here (to {filler!r}); the importer wins and the "
                "owner does not own it"
            )
    return bad


# ===========================================================================
# SITE 16: MINT-BY-HASHING-THE-ARTIFACT.
#
# The promoted default minted FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256 by
# hashing scripts/fr13_patch_fa2_tree_bias.py -- the same file the selector
# gate then compares against disk. A value derived from an artifact cannot
# test that artifact: the gate was `x == x` and could not fail however stale
# the credential was. It is now minted from the credential's SEALED identity,
# so the comparison is sealed-vs-disk.
#
# The shape is: `VAR=${VAR:-$(hash ARTIFACT)}` followed later by a gate
# comparing "$VAR" against `$(hash ARTIFACT)` for the SAME artifact. Neither
# line is wrong alone, which is why this needs a detector rather than a
# reading.
# ===========================================================================

_MINT_FROM_HASH = re.compile(
    r"^\s*(?:export\s+)?([A-Za-z_]\w*)=\$\{\1:-\$\("
)
_HASHES = re.compile(r"\$\((?:sha256sum|sha1sum|md5sum)\s+(\S+)")

# Adjudicated: vacuous, but not load-bearing, and not fixable by re-sourcing.
#
# FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256 is minted by hashing the
# credential file and later compared against a re-hash of that same file. On
# the CALLER path (an operator supplies the digest) the check is real. On the
# promoted-default path it is `x == x` -- and it cannot be otherwise, because a
# file's own digest has no external source here short of a fourth literal pin
# that would need re-pinning at every re-seal. What actually guards the default
# path is the credential's internal binding: its canonical payload digest, and
# verify-tier-b re-deriving the whole chain in the container against pinned
# bounds. So this one is recorded, not repaired.
_MINT_HASH_ADJUDICATED = frozenset({
    "FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256",
})


def scan_mint_hashes_its_own_gate():
    """Values minted by hashing the artifact a later gate re-hashes."""
    bad = []
    for rel in LAUNCHER_FAMILIES:
        lines = (REPO / rel).read_text().split("\n")
        for i, line in enumerate(lines):
            match = _MINT_FROM_HASH.match(line)
            if not match:
                continue
            name = match.group(1)
            hashed = _HASHES.search("\n".join(lines[i:i + 5]))
            if not hashed:
                continue
            artifact = hashed.group(1).strip('"')
            leaf = artifact.rsplit("/", 1)[-1]
            gated = [
                j + 1 for j, later in enumerate(lines[i + 1:], i + 1)
                if f'"${name}"' in later
                and "sha256sum" in later
                and leaf in later
            ]
            if not gated or name in _MINT_HASH_ADJUDICATED:
                continue
            bad.append(
                f"{Path(rel).name}:{i + 1} {name} is minted by hashing "
                f"{artifact} and gated against a re-hash of it at line "
                f"{gated[0]}; the gate cannot fail"
            )
    return bad


# ===========================================================================
# SITE 18: COMMIT-BINDING SCOPE.
#
# Pass 101 made a tier-B credential's source_commit RECORDED, not BOUND: a
# credential attests numerics, and a commit that touches no kernel input
# cannot change them. That rule then had to be applied in three places -- the
# sidecar's field lists, the launcher's selector gate, the patcher's
# credential check -- and each one was found stale in turn, by a boot.
#
# Any comparison of a B1 source commit against HEAD must therefore be
# TIER-SCOPED. This finds the ones that are not. It is the launcher half; the
# patcher half is a single comparator behind one shared predicate, pinned by
# test_no_container_side_reader_compares_a_credential_commit_to_the_serve.
# ===========================================================================

_HEAD_COMPARISON = re.compile(
    r'"\$(?:\{)?FR13_FA2_QROW32_B1_SOURCE_COMMIT(?:\})?"\s*==\s*"\$\((?:git rev-parse HEAD)\)"'
)
_COMMIT_SCOPE_OPENER = "_fr13_b1_commit_bound == 1"


def scan_commit_binding_scope():
    """B1 commit-vs-HEAD comparisons that are not tier-scoped."""
    bad = []
    for rel in LAUNCHER_FAMILIES:
        lines = (REPO / rel).read_text().split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#") or not _HEAD_COMPARISON.search(line):
                continue
            # the scoping `if` must open within the preceding few lines, and
            # nothing may close it in between
            window = lines[max(i - 6, 0):i]
            if any(_COMMIT_SCOPE_OPENER in earlier for earlier in window):
                continue
            bad.append(
                f"{Path(rel).name}:{i + 1} compares the B1 source commit "
                "against HEAD without a tier scope; pass 101 retired that "
                "binding for tier-B credentials"
            )
    return bad


# ===========================================================================
# SITE 19: THE THREE WORKLOAD TABLES.
#
# The workload-identity concept lives in three places, and it has to:
#
#   1. the LAUNCHER's `_fr13_b1_tierb_workload_pins` case  -- bash, host-side,
#      running before docker, so it cannot import anything;
#   2. the PATCHER's _FR13_FA2_QROW32_B1_TIER_B_WORKLOADS -- Python, but
#      executing inside the pinned image against a patched vLLM tree, where
#      the campaign deliberately duplicates tables rather than acquiring
#      imports (see _FR13_FA2_QROW32_B1_TIER_B_ARMS);
#   3. fr13_floor_gate.EVIDENCE_SETS                       -- Python, host-side.
#
# Sites 12, 17, 18 and 19 are one disease: a concept stated N times and
# updated N-1 times. Site 19 is its purest form -- exact16_minus_13236 landed
# in tables 1 and 2 and was structurally inexpressible in table 3, which is
# keyed by task COUNT rather than by workload.
#
# DERIVE OR TIE. Derivation was considered and rejected honestly: no single
# authority is reachable at the moment all three need the answer. Table 1 is
# bash before docker; table 3 runs on the host with the repo importable; table
# 2 runs in a container whose whole design is to depend on nothing it did not
# bring. (Where a value COULD be sourced rather than copied, it now is -- site
# 16 mints the patcher digest from the credential instead of re-deriving it.)
#
# So they are TIED. This joins the three on the one identity they all carry --
# the SUBSET FILE -- and requires them to agree on its digest and its task
# ids. A new workload that reaches two tables fails here until it reaches the
# third, which is the failure mode of all four sites, made loud.
# ===========================================================================

_LAUNCHER_WORKLOAD_ARM = re.compile(r"^    ([A-Za-z0-9_]+)\)\s*$")
_LAUNCHER_WORKLOAD_FIELD = re.compile(
    r'^\s*(_fr13_tierb_task_ids|_fr13_tierb_subset_sha256|_fr13_tierb_subset_file)'
    r'="([^"]*)"\s*$'
)


def launcher_workload_table(text):
    """The launcher's keyed case, read out of the bash itself."""
    start = text.index("_fr13_b1_tierb_workload_pins() {")
    end = text.index("\n}\n", start)
    table, current = {}, None
    for line in text[start:end].split("\n"):
        arm = _LAUNCHER_WORKLOAD_ARM.match(line)
        if arm:
            current = arm.group(1)
            table[current] = {}
            continue
        field = _LAUNCHER_WORKLOAD_FIELD.match(line)
        if field and current:
            table[current][field.group(1)] = field.group(2)
    return {
        name: {
            "task_ids": tuple(
                task for task in fields.get("_fr13_tierb_task_ids", "").split(",")
                if task
            ),
            "sha256": fields.get("_fr13_tierb_subset_sha256", ""),
            "relative_path": fields.get("_fr13_tierb_subset_file", ""),
        }
        for name, fields in table.items()
        if fields
    }


def patcher_workload_table(text):
    """_FR13_FA2_QROW32_B1_TIER_B_WORKLOADS, without importing the patcher."""
    start = text.index("_FR13_FA2_QROW32_B1_TIER_B_WORKLOADS = {")
    end = text.index("\n}\n", start) + 2
    namespace = {}
    exec(compile(text[start:end], "<workloads>", "exec"), namespace)
    return {
        name: {"task_ids": tuple(t for t in ids.split(",") if t), "sha256": sha}
        for name, (ids, sha) in namespace[
            "_FR13_FA2_QROW32_B1_TIER_B_WORKLOADS"
        ].items()
    }


def evidence_sets_table():
    """fr13_floor_gate.EVIDENCE_SETS, re-keyed by subset file."""
    import fr13_floor_gate  # noqa: E402

    return {
        row["relative_path"]: {
            "task_ids": tuple(row["task_ids"]),
            "sha256": row["sha256"],
            "task_count": count,
        }
        for count, row in fr13_floor_gate.EVIDENCE_SETS.items()
    }


def scan_workload_table_agreement():
    """The three workload tables, joined on the subset file they name."""
    bad = []
    patcher = patcher_workload_table(
        (REPO / "scripts/fr13_patch_fa2_tree_bias.py").read_text()
    )
    evidence = evidence_sets_table()
    for rel in LAUNCHER_FAMILIES:
        launcher = launcher_workload_table((REPO / rel).read_text())
        name = Path(rel).name
        if set(launcher) != set(patcher):
            bad.append(
                f"{name}: launcher names {sorted(launcher)} but the patcher "
                f"names {sorted(patcher)}"
            )
        for workload, row in launcher.items():
            mirror = patcher.get(workload)
            if mirror is None:
                continue
            if mirror["task_ids"] != row["task_ids"] or (
                mirror["sha256"] != row["sha256"]
            ):
                bad.append(
                    f"{name}: {workload} differs between the launcher and the "
                    "patcher workload tables"
                )
            subset = row["relative_path"]
            if not subset:
                # a synthetic shape carries no subset; it must NOT appear in
                # EVIDENCE_SETS, which exists to bind evidence files
                if any(
                    entry["task_ids"] == row["task_ids"]
                    for entry in evidence.values()
                ):
                    bad.append(
                        f"{name}: {workload} carries no subset but matches an "
                        "evidence set"
                    )
                continue
            seen = evidence.get(subset)
            if seen is None:
                bad.append(
                    f"{name}: {workload} names {subset}, which "
                    "fr13_floor_gate.EVIDENCE_SETS does not know -- the gate "
                    "cannot express this workload"
                )
                continue
            if seen["sha256"] != row["sha256"]:
                bad.append(f"{name}: {workload} subset digest differs from "
                           "EVIDENCE_SETS")
            if seen["task_ids"] != row["task_ids"]:
                bad.append(f"{name}: {workload} task ids differ from "
                           "EVIDENCE_SETS")
            if seen["task_count"] != len(row["task_ids"]):
                bad.append(
                    f"{name}: {workload} has {len(row['task_ids'])} tasks but "
                    f"EVIDENCE_SETS files it under {seen['task_count']}"
                )
    return bad


# ===========================================================================
# SITE 20: LITERAL COUNT DISJUNCTIONS IN BASH GUARDS.
#
# The three-table tie (site 19) compares TABLES. Site 20 was not in a table:
# it was a guard literal --
#
#     [[ "${_fixed32_subset_binding[0]}" == "4" || ... == "16" ]]
#
# -- five hundred lines downstream of the Python step that had just validated
# fifteen against EVIDENCE_SETS. The authority blessed the count; a
# restatement of its old contents rejected it. Fifth statement of the rule,
# invisible to a table comparison.
#
# So this is a sibling projection, over SHAPE rather than contents: a bash
# guard that decides a task/evidence count by comparing it against literal
# numbers. Adding "15" to such a guard would have been statement 5.5; the fix
# is for the authority to print its own allowed set and for bash to check
# membership, which is what the two call sites now do.
#
# Keyed on the disjunction shape, not on the specific numbers, per the
# reformatting-decay lesson: a projection pinned to "4" and "16" would go
# blind the moment someone wrote 8 or 32.
# ===========================================================================

# Both roots. Site 21 was in the launcher family, which the site-20 roster did
# not cover -- a projection is only as wide as the files it is pointed at.
# SITE 22: the roster became a derivation. COUNT_GUARD_SITES was a hand-kept
# list, which is why site 22 lived in a root it never named. The roots are
# kept only as the closure's ENTRY POINTS; the universe is what they reach.
COUNT_GUARD_SITES = LAUNCHER_FAMILIES + (
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_b4_campaign_driver.sh",
    "scripts/swe_x86_helpers/offload_codex_proxy.sh",
)
# The shape is a VARIABLE compared to two or more bare integer literals inside
# one `[[ ... ]]`. Deliberately NOT keyed on the variable's name: site 20's was
# `${_fixed32_subset_binding[0]}`, which contains no "count" at all, and a
# projection that only found count-ish names would have missed the very site it
# was written for. (It did, in its first draft.)
# TWO subject spellings, and the second is not optional.
#
# SITE 21 hid behind `${#_fixed32_task_ids[@]}` -- an array-LENGTH expression,
# unquoted, so neither the quoted-variable pattern above nor the runner's own
# census grep saw it (that grep's character class excluded `#`, and returned a
# zero-hit it was right not to believe). A projection for "compares a COUNT to
# literals" that cannot see the canonical way bash spells a count was never
# going to find anything.
_INT_COMPARISON = re.compile(
    r'(?:"(\$\{?[^"]+?\}?)"|(\$\{#[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}))'
    r'\s*==\s*"?([0-9]+)"?'
)


# WHICH SUBJECTS THIS IS ABOUT, and why it is not "every integer disjunction".
#
# Widening the scan to the launcher families reported 54 guards, and NONE of
# them were this class: MAX_NUM_SEQS 1|4, booleans 0|1, GDN BV 16|32|64|128.
# Those own their values -- nothing computes them, no table can outgrow them,
# there is nothing for the guard to fall behind. Adjudicating 54 entries would
# be a detector switched off one line at a time.
#
# The class is narrower and sharper: a guard comparing a COUNT OF AN
# AUTHORITY-VALIDATED ARTIFACT against literals. Sites 20 and 21 are both
# exactly that, in the three spellings bash offers --
#
#     ${#_fixed32_task_ids[@]}        (array length; site 21)
#     ${_fixed32_subset_binding[0]}   (a field of the binding; site 20)
#     $FIXED32_TASK_COUNT             (the binding, named; site 20's sibling)
#
# -- and the file must have the authority in scope at all, which is what
# `references fr13_floor_gate` establishes. That is the difference between a
# closed enumeration and a restatement of a list something else owns.
_AUTHORITY_MODULE = "fr13_floor_gate"
_COUNTED_SUBJECT = re.compile(
    r"^\$\{#[A-Za-z_]\w*\[[@*]\]\}$"          # array length
    r"|subset_binding"                             # a field of the binding
    r"|(?i:task_count)"                            # the binding, named
)


def _bracket_guards(lines):
    """Every `[[ ... ]]` guard, with continuations joined, and its line."""
    joined, start = None, None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if joined is None:
            if "[[" not in stripped:
                continue
            joined, start = stripped, i + 1
        else:
            joined += " " + stripped
        if joined.rstrip().endswith("\\"):
            joined = joined.rstrip()[:-1]
            continue
        yield start, joined
        joined, start = None, None


def scan_literal_count_guards(paths=None):
    """Bash guards that decide a value by comparing it to literal integers."""
    bad = []
    for rel in (serve_execution_closure() if paths is None else paths):
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text()
        if _AUTHORITY_MODULE not in text:
            continue  # no authority in scope; nothing here can go stale
        for lineno, guard in _bracket_guards(text.split("\n")):
            seen = {}
            for quoted, length, literal in _INT_COMPARISON.findall(guard):
                seen.setdefault(quoted or length, set()).add(literal)
            for name, literals in seen.items():
                if len(literals) < 2 or not _COUNTED_SUBJECT.search(name):
                    continue
                bad.append(
                    f"{Path(rel).name}:{lineno} decides {name} by literal "
                    f"disjunction {sorted(literals)}; the authority that "
                    "validated it should print its own allowed values and "
                    "bash should check membership"
                )
    return bad



# ===========================================================================
# SITE 22: THE SERVE EXECUTION CLOSURE, and the quantifier projection.
#
# Two failures at once. The count rule was written as a REGEX REPETITION
# QUANTIFIER -- `{3}` for four ids, `{15}` for sixteen -- so the literals 4 and
# 16 never appeared and every count-scan was blind to it by construction. And
# it lived in scripts/swe_x86_helpers/, a fourth root no sweep covered.
#
# The second failure is the worse one. Site 21 reported the class "closed by
# predicate"; that was retracted, correctly, because a predicate is only closed
# over the universe it is run on, and the universe had never been enumerated --
# it was a hand-kept roster of files somebody had happened to think of.
#
# So the universe is DERIVED: the transitive closure of what a serve actually
# executes, starting from the launcher families and following source/bash/
# python invocations down. Every scan below runs over that closure, so a new
# helper joins the sweep by being executed, not by being remembered.
# ===========================================================================

CLOSURE_ROOTS = LAUNCHER_FAMILIES + (
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_b4_campaign_driver.sh",
)
# WHAT COUNTS AS AN EDGE. The first draft matched invocation SYNTAX -- `bash
# X`, `source X`, `python3 X` -- and missed two shapes immediately:
#
#   OFFLOAD_HELPER=scripts/swe_x86_helpers/offload_codex_proxy.sh   (run later
#                                                                    via "$VAR")
#   source "$SCRIPT_DIR/fr13_required_tree_flags.sh"                (resolved
#                                                                    relative)
#
# and offload_codex_proxy.sh is site 22 itself, so the closure would have
# excluded the file it was built to find. Edges are therefore any repo file a
# script NAMES, resolved against the repo root and against the naming file's
# own directory. That over-approximates -- a file merely mentioned in a
# comment-free line joins the universe -- and over-approximating is the safe
# direction for a universe definition: a scan that runs on too much reports
# something a human can dismiss, a scan that runs on too little reports
# nothing and is believed.
_NAMED_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:sh|py)")


def serve_execution_closure(roots=CLOSURE_ROOTS):
    """Every repo file a serve can reach, transitively, from the roots down.

    Returns a sorted list. This is the universe definition; a scan that does
    not run over it is closed only over whatever its author remembered.
    """
    seen, queue = set(), list(roots)
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        path = REPO / rel
        if not path.exists():
            continue
        seen.add(rel)
        here = Path(rel).parent
        for line in path.read_text(errors="replace").split("\n"):
            if line.lstrip().startswith("#"):
                continue
            for named in _NAMED_FILE.findall(line):
                for candidate in (named, str(here / Path(named).name)):
                    candidate = str(Path(candidate))
                    if candidate in seen:
                        continue
                    if (REPO / candidate).is_file():
                        queue.append(candidate)
    return sorted(seen)


# A count rule hidden inside a regex: `{N}` immediately after a group that
# looks like an id pattern. Keyed on the quantifier's ADJACENCY to an id-shaped
# group rather than on any particular id shape, so a different id alphabet
# still matches.
_QUANTIFIER_COUNT = re.compile(r"\)\{(\d+)\}")
_ID_SHAPED = re.compile(r"\[A-Za-z0-9[^\]]*\]\+__")


def scan_regex_quantifier_counts(paths=None):
    """Count rules expressed as regex repetition, where no scan can see them."""
    bad = []
    for rel in (serve_execution_closure() if paths is None else paths):
        path = REPO / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(errors="replace").split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            counts = _QUANTIFIER_COUNT.findall(line)
            if not counts or not _ID_SHAPED.search(line):
                continue
            bad.append(
                f"{rel}:{lineno} expresses a task-count rule as a regex "
                f"repetition quantifier {{{','.join(counts)}}}; the count "
                "never appears as a literal, so no count scan can see it -- "
                "validate FORMAT in the regex and the COUNT separately"
            )
    return bad


# ===========================================================================
# SITE 23: EMBEDDED PYTHON, where the bash projections cannot see.
#
# The count rule's EIGHTH statement and FOURTH encoding:
#
#     len(task_ids) not in ((1,) if diagnostic else (4, 16))
#
# inside a `<<'PY'` heredoc, in the SAME FILE whose bash check was converted at
# site 20. The conversion stopped at the language boundary, and so did every
# scanner: they all read shell.
#
# So the shell files in the closure are opened up -- heredoc bodies extracted
# and scanned as Python. The encoding is enumerated rather than the instance:
# a tuple, a set, a list, or a chain of `len(x) == N` comparisons all say the
# same thing, and all four are matched.
# ===========================================================================

_HEREDOC_OPEN = re.compile(r"<<-?\s*[\"']?([A-Za-z_]\w*)[\"']?\s*$")

# WHAT THIS IS ABOUT, narrowed the way site 21 taught.
#
# Scanning every `len(x) != N` in the closure's heredocs reported 14 sites, and
# 12 were closed shapes that own their values -- len(layers) != 48,
# len(graph_signature) != 64, len(rows) != 1. Those cannot go stale; nothing
# computes them.
#
# The class is a count whose admissible values DEPEND ON THE RUN CLASS -- the
# `(1,) if diagnostic else (...)` shape. That is the rule the authority owns,
# and both live instances wear it. Keyed on the rule's own shape rather than on
# any variable name, and a line that already consults the authority is not a
# finding: converting it is the fix, not a new offence.
_PY_RUN_CLASS_COUNT = re.compile(
    r"len\s*\([^()]*\)\s*(?:not\s+in|in)\s*\(\s*\([^)]*\)\s*if\s+\w+\s+else\s+([^)]*)\)"
)
_AUTHORITY_NAMES = ("ADMISSIBLE_TASK_COUNTS", "admissible_task_counts",
                    "EVIDENCE_SETS")


def extract_embedded_python(text):
    """(first_body_line, body) for every heredoc that looks like Python."""
    lines = text.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        opener = _HEREDOC_OPEN.search(lines[i])
        if not opener:
            i += 1
            continue
        terminator = opener.group(1)
        start = i + 1
        end = start
        while end < len(lines) and lines[end].strip() != terminator:
            end += 1
        body = "\n".join(lines[start:end])
        if re.search(r"^\s*(?:import|from)\s+\w", body, re.M):
            blocks.append((start + 1, body))
        i = end + 1
    return blocks


def scan_embedded_python_count_literals(paths=None):
    """Run-class-dependent count rules written as literals inside heredocs."""
    bad = []
    for rel in (serve_execution_closure() if paths is None else paths):
        path = REPO / rel
        if not path.exists() or path.suffix != ".sh":
            continue
        for offset, body in extract_embedded_python(path.read_text(errors="replace")):
            for lineno, line in enumerate(body.split("\n"), offset):
                if line.lstrip().startswith("#"):
                    continue
                match = _PY_RUN_CLASS_COUNT.search(line)
                if not match:
                    continue
                if any(name in match.group(1) for name in _AUTHORITY_NAMES):
                    continue  # already consults the authority
                bad.append(
                    f"{rel}:{lineno} embedded python decides a run-class "
                    f"dependent task count from the literal {match.group(1).strip()} "
                    "-- where the shell projections cannot see it; import or "
                    "receive the authority and check membership in it"
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
    for problem in scan_import_precedes_owner():
        rows.append(
            {"kind": "import-precedes-owner", "file": "<launcher families>",
             "detail": problem}
        )
    for problem in scan_mint_hashes_its_own_gate():
        rows.append(
            {"kind": "mint-hashes-its-own-gate", "file": "<launcher families>",
             "detail": problem}
        )
    for problem in scan_commit_binding_scope():
        rows.append(
            {"kind": "commit-binding-scope", "file": "<launcher families>",
             "detail": problem}
        )
    for problem in scan_workload_table_agreement():
        rows.append(
            {"kind": "workload-table-agreement", "file": "<three tables>",
             "detail": problem}
        )
    for problem in scan_literal_count_guards():
        rows.append(
            {"kind": "literal-count-guard", "file": "<serve closure>",
             "detail": problem}
        )
    for problem in scan_regex_quantifier_counts():
        rows.append(
            {"kind": "regex-quantifier-count", "file": "<serve closure>",
             "detail": problem}
        )
    for problem in scan_embedded_python_count_literals():
        rows.append(
            {"kind": "embedded-python-count", "file": "<serve closure>",
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
