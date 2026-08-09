#!/usr/bin/env python3
"""Extract and pin the rejection-sampler patch region of the FR10 patcher.

Tier-B (bounded-drift) kernel qualification admits reduction-order changes in
VERIFIER-FORWARD kernels that cannot be byte-identical.  The hard invariant of
that policy is that the *rejection-sampling mechanism itself* stays
byte-identical, so that the served distribution remains exactly lossless with
respect to the (epsilon-perturbed) verifier it is handed.

This helper makes that invariant mechanical.  It parses
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` with `ast`, lifts the three
top-level functions that write `vllm/v1/sample/rejection_sampler.py` -- which
between them carry the injected sampler body and the
`apply_sampling_constraints` path -- and hashes their concatenated source.  The
resulting `sampler_region_sha256` is recorded in every
`fr13.tier_b.qualification.v1` artifact and re-asserted on every
re-qualification.

The region is located by AST, not by line number, so unrelated edits elsewhere
in the 41k-line patcher do not move the pin; any edit *inside* the region does.

Usage:

    fr13_tier_b_sampler_pin.py emit [--patcher PATH] [--output PATH]
    fr13_tier_b_sampler_pin.py assert --expect <sha256> [--patcher PATH]
    fr13_tier_b_sampler_pin.py assert --qualification ART.json [--patcher PATH]

`emit` writes the pin record; `assert` exits non-zero and prints a loud
diagnostic when the region has drifted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


PIN_SCHEMA = "fr13.tier_b.sampler_region_pin.v1"
QUALIFICATION_SCHEMA = "fr13.tier_b.qualification.v1"

DEFAULT_PATCHER = (
    Path(__file__).resolve().parent / "fr10_phase4_patch_vllm_tree_gdn.py"
)

# The vLLM file the pinned region writes.  Recorded so a reviewer can see, from
# the artifact alone, which serving source the pin is about.
SAMPLER_TARGET = "vllm/v1/sample/rejection_sampler.py"

# Ordered, exhaustive.  Every top-level function in the patcher that emits into
# REJECTION_SAMPLER_PATH must appear here; `extract_region` fails closed if the
# patcher grows one that does not.
SAMPLER_REGION_FUNCTIONS: tuple[str, ...] = (
    # Injects the canonical multidraft sampler body -- the rejection-sampling
    # mechanism proper -- including the `apply_sampling_constraints` path and
    # the LCP committer.
    "_patch_rejection_sampler_tree_lcp",
    "_patch_rejection_sampler_bonus_handoff",
    "_patch_rejection_sampler_target_logits_handoff",
)

# Substrings that identify a patcher function as sampler-side.  Used only by the
# completeness check below, never to select the region.
_SAMPLER_FUNCTION_MARKERS = ("_patch_rejection_sampler",)


class SamplerPinError(RuntimeError):
    """The sampler region could not be extracted, or it drifted."""


def _read(patcher: Path) -> str:
    if not patcher.is_file():
        raise SamplerPinError(f"patcher is not a regular file: {patcher}")
    return patcher.read_text(encoding="utf-8")


def _top_level_functions(text: str, patcher: Path) -> dict[str, ast.FunctionDef]:
    try:
        tree = ast.parse(text, filename=str(patcher))
    except SyntaxError as exc:  # pragma: no cover - patcher is compiled in CI
        raise SamplerPinError(f"patcher does not parse: {exc}") from exc
    found: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in found:
            raise SamplerPinError(
                f"patcher defines {node.name} more than once at top level"
            )
        found[node.name] = node
    return found


def _function_source(text: str, node: ast.FunctionDef) -> str:
    lines = text.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    if end is None or start < 0 or end > len(lines):
        raise SamplerPinError(f"cannot bound the source of {node.name}")
    return "".join(lines[start:end])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_region(text: str, patcher: Path = DEFAULT_PATCHER) -> list[dict]:
    """Return one record per pinned sampler function, in declared order.

    Fails closed when a named function is missing, when the patcher grows a
    sampler-side function that is not pinned, or when the source of a pinned
    function cannot be bounded.
    """

    functions = _top_level_functions(text, patcher)

    missing = [n for n in SAMPLER_REGION_FUNCTIONS if n not in functions]
    if missing:
        raise SamplerPinError(
            "patcher is missing pinned sampler functions: " + ", ".join(missing)
        )

    unpinned = sorted(
        name
        for name in functions
        if any(marker in name for marker in _SAMPLER_FUNCTION_MARKERS)
        and name not in SAMPLER_REGION_FUNCTIONS
    )
    if unpinned:
        raise SamplerPinError(
            "patcher grew unpinned sampler-side functions; extend "
            "SAMPLER_REGION_FUNCTIONS and re-qualify every Tier-B candidate: "
            + ", ".join(unpinned)
        )

    records: list[dict] = []
    for name in SAMPLER_REGION_FUNCTIONS:
        node = functions[name]
        source = _function_source(text, node)
        records.append(
            {
                "name": name,
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "lines": node.end_lineno - node.lineno + 1,
                "bytes": len(source.encode("utf-8")),
                "sha256": _sha256(source.encode("utf-8")),
            }
        )
    return records


def region_bytes(text: str, patcher: Path = DEFAULT_PATCHER) -> bytes:
    """Canonical byte image of the sampler region.

    Each function's source is prefixed by a name banner, so renaming, dropping
    or reordering a function changes the image even when the bodies do not.
    """

    functions = _top_level_functions(text, patcher)
    extract_region(text, patcher)  # fail-closed completeness check
    chunks: list[bytes] = []
    for name in SAMPLER_REGION_FUNCTIONS:
        chunks.append(f"### {PIN_SCHEMA} {name}\n".encode("utf-8"))
        chunks.append(_function_source(text, functions[name]).encode("utf-8"))
    return b"".join(chunks)


def region_sha256(text: str, patcher: Path = DEFAULT_PATCHER) -> str:
    return _sha256(region_bytes(text, patcher))


def pin_record(patcher: Path = DEFAULT_PATCHER) -> dict:
    text = _read(patcher)
    raw = patcher.read_bytes()
    return {
        "schema": PIN_SCHEMA,
        "patcher": patcher.name,
        "patcher_sha256": _sha256(raw),
        "patcher_bytes": len(raw),
        "sampler_target": SAMPLER_TARGET,
        "region_functions": extract_region(text, patcher),
        "region_bytes": len(region_bytes(text, patcher)),
        "sampler_region_sha256": region_sha256(text, patcher),
    }


def assert_pin(expected: str, patcher: Path = DEFAULT_PATCHER) -> dict:
    """Raise SamplerPinError unless the region hashes to `expected`."""

    expected = expected.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SamplerPinError(f"expected sha256 is malformed: {expected!r}")
    record = pin_record(patcher)
    observed = record["sampler_region_sha256"]
    if observed != expected:
        drifted = [
            f"{f['name']} @{f['lineno']}-{f['end_lineno']} {f['sha256']}"
            for f in record["region_functions"]
        ]
        raise SamplerPinError(
            "SAMPLER REGION DRIFTED -- Tier-B qualification is void.\n"
            f"  expected {expected}\n"
            f"  observed {observed}\n"
            "  per-function: " + "; ".join(drifted) + "\n"
            "  The rejection-sampling mechanism must stay byte-identical. "
            "Re-qualify every Tier-B candidate against the new pin."
        )
    record["expected_sampler_region_sha256"] = expected
    record["sampler_region_unchanged"] = True
    return record


def _expected_from_qualification(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SamplerPinError(f"cannot read qualification artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise SamplerPinError("qualification artifact is not a JSON object")
    schema = payload.get("schema")
    if schema != QUALIFICATION_SCHEMA:
        raise SamplerPinError(
            f"qualification artifact schema is {schema!r}, "
            f"expected {QUALIFICATION_SCHEMA!r}"
        )
    expected = payload.get("sampler_region_sha256")
    if not isinstance(expected, str):
        raise SamplerPinError(
            "qualification artifact has no string sampler_region_sha256"
        )
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patcher",
        type=Path,
        default=DEFAULT_PATCHER,
        help="path to fr10_phase4_patch_vllm_tree_gdn.py",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="print the sampler-region pin record")
    emit.add_argument("--output", type=Path, default=None)

    check = sub.add_parser(
        "assert", help="fail unless the sampler region matches a recorded value"
    )
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--expect", help="expected sampler_region_sha256")
    source.add_argument(
        "--qualification",
        type=Path,
        help="fr13.tier_b.qualification.v1 artifact carrying the expected sha",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "emit":
            record = pin_record(args.patcher)
        else:
            expected = (
                args.expect
                if args.expect is not None
                else _expected_from_qualification(args.qualification)
            )
            record = assert_pin(expected, args.patcher)
    except SamplerPinError as exc:
        print(f"[fr13_tier_b_sampler_pin] {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(record, indent=2, sort_keys=True)
    if args.command == "emit" and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
