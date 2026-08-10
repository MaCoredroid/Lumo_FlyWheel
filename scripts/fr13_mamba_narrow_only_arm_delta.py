#!/usr/bin/env python3
"""Prove two arms' container environments differ in exactly one CONFIGURATION key.

This is the provenance device of the within-run lever pair: if the arms differ
anywhere but FR13_MAMBA_SPEC_BLOCKS_CDIV, the measured APC delta is attributable
to something other than the lever, and the pair is worthless.

IDENTITY vs CONFIGURATION. Every arm necessarily carries its own name in some
paths -- run dir, cidfile, nsys output, the GPU-timer sidecars -- and Docker
assigns each container a random HOSTNAME. Those differ by construction and say
nothing about what the engine computed.

The first version of this check carried a hand-listed set of name-bearing keys
and FAILED CLOSED on a complete, valid pair because the list was written from
the env block this repo's driver sets and missed the four the launcher and
Docker inject downstream (FR13_RUN_DIR, FR13_FIXED32_CIDFILE, LUMO_NSYS_OUTPUT,
HOSTNAME). A hand list is the wrong instrument: it has to be extended every time
a caller adds a path, and each extension is an unreviewed licence to ignore a
difference.

The rule here is MECHANICAL instead. A key is identity iff substituting the OFF
arm's name for the ON arm's name inside its value makes the two values equal --
i.e. the difference IS the arm name and nothing else. Anything whose value
differs for any other reason stays in the diff and fails the run. HOSTNAME is
the single explicit exemption, because Docker derives it from the container id
rather than from the arm name, and it is recorded in the artifact so the
exemption is visible rather than silent.

Exit 0 = the arms differ in exactly the expected key. Exit 1 = they do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Docker assigns this from the container id, not from the arm name, so the
# substitution rule cannot recognise it. Named explicitly so the exemption is
# auditable in the emitted artifact.
CONTAINER_ASSIGNED = ("HOSTNAME",)


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def classify(
    off: dict[str, str], on: dict[str, str], off_arm: str, on_arm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the raw difference into identity (arm-name-only) and configuration."""
    identity: dict[str, Any] = {}
    config: dict[str, Any] = {}
    for key in sorted(set(off) | set(on)):
        a, b = off.get(key), on.get(key)
        if a == b:
            continue
        entry = {"off": a, "on": b}
        if key in CONTAINER_ASSIGNED:
            identity[key] = {**entry, "reason": "container-assigned by Docker"}
            continue
        # The difference IS the arm name iff renaming makes them identical.
        if isinstance(a, str) and isinstance(b, str) and b.replace(on_arm, off_arm) == a:
            identity[key] = {**entry, "reason": "value differs only by the arm name"}
            continue
        config[key] = entry
    return identity, config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runroot", required=True, type=Path)
    ap.add_argument("--off-arm", required=True)
    ap.add_argument("--on-arm", required=True)
    ap.add_argument(
        "--expect-key",
        default="FR13_MAMBA_SPEC_BLOCKS_CDIV",
        help="the one configuration key the arms may differ in",
    )
    ap.add_argument("--expect-off", default="0")
    ap.add_argument("--expect-on", default="1")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    off_env = read_env(args.runroot / args.off_arm / "container_env.txt")
    on_env = read_env(args.runroot / args.on_arm / "container_env.txt")
    identity, config = classify(off_env, on_env, args.off_arm, args.on_arm)

    ok = (
        set(config) == {args.expect_key}
        and config.get(args.expect_key, {}).get("off") == args.expect_off
        and config.get(args.expect_key, {}).get("on") == args.expect_on
    )
    result = {
        "schema": "fr13.mamba_narrow.within_run_only_arm_delta.v2",
        "off_arm": args.off_arm,
        "on_arm": args.on_arm,
        "n_keys_compared": len(set(off_env) | set(on_env)),
        "configuration_diff": config,
        "configuration_differing_keys": sorted(config),
        "identity_diff": identity,
        "identity_differing_keys": sorted(identity),
        "container_assigned_exemptions": list(CONTAINER_ASSIGNED),
        "expected_key": args.expect_key,
        "only_arm_delta_proven": ok,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"keys compared            : {result['n_keys_compared']}")
    print(f"identity-only differences: {sorted(identity)}")
    print(f"configuration differences: {sorted(config)}")
    if not ok:
        print(
            "only-arm-delta VIOLATED: arms differ in "
            f"{sorted(config)}, expected exactly ['{args.expect_key}'] "
            f"({args.expect_off} -> {args.expect_on})",
            file=sys.stderr,
        )
        return 1
    print(
        f"only-arm-delta OK: the arms differ in exactly {args.expect_key} "
        f"({args.expect_off} -> {args.expect_on})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
