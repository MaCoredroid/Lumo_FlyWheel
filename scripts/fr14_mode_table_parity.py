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

# Every shell file that can reach a fixed32 profile decision.
SHELL_SITES = (
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr14_armb_leg3_launch_nomiddleware.sh",
    "scripts/fr14_leg3_launch_nomiddleware.sh",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
)

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
