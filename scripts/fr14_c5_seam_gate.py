"""THE c5 SEAM GATE -- a standing per-task degeneration-SHAPE instrument.

c5 = Delta accepted[pos=5] / Delta accepted[pos=4], from the stock counter
``vllm:spec_decode_num_accepted_tokens_per_pos_total``. Position 5 is the
MTP-head -> Arctic-tail seam, so c5 measures how much of the accepted path is
being carried by the suffix cache rather than by the heads.

WHY THIS AND NOT ACCEPT RATE. On the 100 banked task-arms, acc/step catches only
the extreme loop (8.273) and sits mid-band for the enumeration class -- which is
the campaign's own blindness statement. c5 separates 4 of the 5 corpus
degenerations from all 95 healthy arms with zero overlap.

  low  c5  = the cache lost the thread and the model is inventing (enumeration)
  high c5  = the cache is driving (n-gram loop)

CORRIDOR [0.40, 0.70], PRE-REGISTERED from the 100-arm sweep before this module
existed. Reproduced independently here: 96/100 in band, 4 outside, and all four
outside are known degenerations. The fifth (13033/Ch27, 0.6855) sits 0.017 above
the healthy ceiling at task-aggregate resolution and needs the windowed variant.

THIS FLAGS, IT DOES NOT REFUSE. A shape outside the corridor is a call for the
eyeball, not a verdict. The corridor's high side is thin by construction and a
gate that refused on it would refuse healthy arms.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Pre-registered from the 100-arm sweep. Changing these is a re-registration.
C5_CORRIDOR_LOW = 0.40
C5_CORRIDOR_HIGH = 0.70

#: The seam. Position 5 is where the MTP heads hand off to the Arctic tail.
C5_NUMERATOR_POSITION = 5
C5_DENOMINATOR_POSITION = 4

SCHEMA = "fr14.c5_seam_gate.v1"

_PER_POS = re.compile(
    r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*'
    r'position="(\d+)"\}\s+([0-9.eE+-]+)'
)


class C5Error(RuntimeError):
    """Raised only for malformed inputs -- never for a shape verdict."""


def parse_per_position(text: str) -> dict[int, float]:
    """Pull the per-position accepted counter out of a Prometheus scrape."""
    found = {
        int(match.group(1)): float(match.group(2))
        for match in _PER_POS.finditer(text)
    }
    if not found:
        raise C5Error(
            "scrape carries no vllm:spec_decode_num_accepted_tokens_per_pos_total"
        )
    return found


def per_position_from_ladder(ladder: list[int]) -> dict[int, int]:
    """Reconstruct the stock per-position counter from an accept ladder.

    The ladder is a DENSITY (rows whose accepted path had exactly length i);
    the stock counter is a SURVIVAL function (rows whose path exceeded i). One
    determines the other:  per_pos[i] = sum(ladder[j] for j > i).

    Verified against the banked exact16 run: the reconstruction matches the
    scraped counter exactly on 10,890 of 10,916 rows, and the residual is the
    same +61 that the ladder's token total carried -- decomposed per position
    (26, 17, 10, 5, 2, 1 at positions 0..5, zero beyond), which is a truncated
    emission on a handful of rows rather than a convention difference.
    """
    return {
        index: sum(ladder[index + 1 :])
        for index in range(len(ladder))
    }


def classify(c5: float) -> str:
    if c5 < C5_CORRIDOR_LOW:
        return "DEGENERATION-SHAPE:low(cache-lost-thread/enumeration)"
    if c5 > C5_CORRIDOR_HIGH:
        return "DEGENERATION-SHAPE:high(cache-driving/loop)"
    return "in-corridor"


def _c5(delta4: float, delta5: float, *, label: str) -> dict[str, object]:
    if delta4 <= 0:
        # Not a verdict: a window with no position-4 acceptance carries no
        # seam information at all, and saying "0.0" would invent one.
        return {
            "schema": SCHEMA,
            "label": label,
            "c5": None,
            "verdict": "no-signal",
            "delta_pos4": delta4,
            "delta_pos5": delta5,
            "corridor": [C5_CORRIDOR_LOW, C5_CORRIDOR_HIGH],
        }
    value = delta5 / delta4
    return {
        "schema": SCHEMA,
        "label": label,
        "c5": value,
        "verdict": classify(value),
        "delta_pos4": delta4,
        "delta_pos5": delta5,
        "corridor": [C5_CORRIDOR_LOW, C5_CORRIDOR_HIGH],
    }


def c5_from_scrapes(pre_text: str, post_text: str, *, label: str) -> dict[str, object]:
    """Task-aggregate c5 from a pre/post metrics bracket."""
    pre, post = parse_per_position(pre_text), parse_per_position(post_text)
    delta4 = post.get(C5_DENOMINATOR_POSITION, 0.0) - pre.get(
        C5_DENOMINATOR_POSITION, 0.0
    )
    delta5 = post.get(C5_NUMERATOR_POSITION, 0.0) - pre.get(
        C5_NUMERATOR_POSITION, 0.0
    )
    result = _c5(delta4, delta5, label=label)
    result["source"] = "vllm_metrics_bracket"
    return result


def c5_from_ladders(
    before: list[int], after: list[int], *, label: str
) -> dict[str, object]:
    """WINDOWED c5 (the F4 variant), from two accept-ladder snapshots.

    Free: it needs no extra scrape. The ladder is already drained at every
    flush boundary, so consecutive sidecars give within-task resolution that
    the per-task bracket cannot. On the banked exact16 run this reads 0.3498
    for the degenerating window against a scraped task-aggregate of 0.3499,
    while the whole-run cumulative is 0.4587 -- INSIDE the corridor. Aggregating
    the run hides the very thing the instrument is for.
    """
    if len(before) != len(after):
        raise C5Error("ladder snapshots differ in width")
    start = per_position_from_ladder(before)
    end = per_position_from_ladder(after)
    delta4 = end.get(C5_DENOMINATOR_POSITION, 0) - start.get(
        C5_DENOMINATOR_POSITION, 0
    )
    delta5 = end.get(C5_NUMERATOR_POSITION, 0) - start.get(
        C5_NUMERATOR_POSITION, 0
    )
    result = _c5(delta4, delta5, label=label)
    result["source"] = "accept_ladder_window"
    return result


def sweep_runroot(runroot: Path) -> list[dict[str, object]]:
    """Per-task c5 for every bracketed task under a runroot."""
    out: list[dict[str, object]] = []
    for pre in sorted(runroot.rglob("per_task/*/vllm_metrics_pre.txt")):
        post = pre.with_name("vllm_metrics_post.txt")
        if not post.exists():
            continue
        try:
            out.append(
                c5_from_scrapes(
                    pre.read_text(errors="replace"),
                    post.read_text(errors="replace"),
                    label=pre.parent.name,
                )
            )
        except C5Error:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="c5 seam gate (flags, never refuses)")
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = sweep_runroot(args.runroot)
    if args.json:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        for row in rows:
            value = row["c5"]
            shown = "  n/a " if value is None else f"{value:.4f}"
            print(f"  c5={shown}  {row['verdict']:52s} {row['label']}")
    # FLAGS ONLY. Exit 0 whatever the shapes are: the eyeball adjudicates, and
    # a gate that exits non-zero here would become a refusal by the back door.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
