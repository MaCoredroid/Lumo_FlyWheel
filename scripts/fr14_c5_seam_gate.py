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
existed. Reproduced here against the PINNED calibration set below: 95 in band,
4 outside, and all four outside are known degenerations. The fifth (13033/Ch27,
0.6855) sits 0.017 above the healthy ceiling at task-aggregate resolution and
needs the windowed variant. (The reproduction reads 99 arms where the
pre-registration note says 100; see C5_CALIBRATION_RUNROOTS -- everything that
decides the corridor agrees, and the measured number is the pinned one.)

THIS FLAGS, IT DOES NOT REFUSE. A shape outside the corridor is a call for the
eyeball, not a verdict. The corridor's high side is thin by construction and a
gate that refused on it would refuse healthy arms.

APPLICABILITY IS PART OF THE INSTRUMENT (2026-08-24). c5 is a SEAM conditional,
and an arm whose drafter has no seam has no c5 -- not a low one, none. The
first four MTP-5 chain-drafter arms entered the corpus with num_spec_tokens=5,
so their positions run 0..4, position 5 does not exist, delta_pos5 is
STRUCTURALLY zero, and each read c5 = 0/n = 0.0000: four manufactured floors
below every real degeneration in the corpus, making the healthiest arms in the
bank look like its worst. A statistic calibrated on one topology and applied
blindly to another reads the healthy case as the pathological one -- the
eyeball's inverted-metric lesson, one layer down.

So an inapplicable arm now contributes NOTHING rather than zero, which is the
measured-zero-vs-absent distinction at corpus level, and the exclusion is
DECLARED in the output rather than silent. Applicability is decided from
EVIDENCE, never from a filename: the engine's own declared num_spec_tokens, and
the arm's own c5_applicable stamp when it carries one.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

#: Pre-registered from the 100-arm sweep. Changing these is a re-registration.
C5_CORRIDOR_LOW = 0.40
C5_CORRIDOR_HIGH = 0.70

#: MINIMUM DENOMINATOR for a corridor verdict (E-A audit, pass 162).
#:
#: c5 is a ratio of counts, so its sampling error is ~sqrt(p(1-p)/n) at the
#: denominator n = delta accepted[pos=4]. On trivially short tasks n is small
#: enough that a perfectly healthy arm can read below the corridor by chance:
#: E-A found two CLEAN arms with 8-11 requests reading 0.3627 and 0.3876.
#:
#: Derived, not picked. At p = the healthy median 0.5425:
#:     n=100 -> SE 0.0497, 3-sigma band [0.394, 0.692]  CROSSES the 0.40 floor
#:     n=150 -> SE 0.0406, 3-sigma band [0.421, 0.665]  clear
#: 150 is the smallest round n at which a median-healthy arm cannot reach the
#: low bound at 3 sigma. The port's 100 exclusive-bracket arms have a minimum
#: denominator of 215, so every one of them keeps its verdict with 43% margin.
#:
#: Below this the gate reports NO-SIGNAL, never a corridor verdict. delta_pos4
#: is emitted in every payload so a consumer can re-threshold without re-running.
C5_MIN_DENOMINATOR = 150

#: The seam. Position 5 is where the MTP heads hand off to the Arctic tail.
C5_NUMERATOR_POSITION = 5
C5_DENOMINATOR_POSITION = 4

SCHEMA = "fr14.c5_seam_gate.v1"

_PER_POS = re.compile(
    r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*'
    r'position="(\d+)"\}\s+([0-9.eE+-]+)'
)

#: THE ENGINE'S OWN DECLARATION of how many positions its drafter proposes.
#: Present in every runroot's boot log snapshot. Positions are numbered
#: 0..num_spec_tokens-1, so the seam at position 5 EXISTS only when
#: num_spec_tokens > 5 -- 31 on the tree drafter, 5 on the MTP-5 chain.
_NUM_SPEC_TOKENS = re.compile(r"num_spec_tokens=(\d+)")

#: An arm may also state the answer itself. The MTP-5 probes stamp
#: ``c5_applicable=NO -- c5 is a SEAM conditional and a chain drafter has no
#: seam`` next to a note that says do not quote a c5 for this arm. A declaration
#: is stronger evidence than an inference, so it wins where both exist.
_C5_APPLICABLE_STAMP = re.compile(r"^\s*c5_applicable\s*=\s*(\w+)", re.MULTILINE)

#: Files that may carry either kind of evidence, in the order they are read.
_APPLICABILITY_EVIDENCE = ("boot_log_snapshot.txt", "*PROBE.txt", "*_PROBE.txt")


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
    def _no_signal(reason: str) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "label": label,
            "c5": None,
            "verdict": f"no-signal:{reason}",
            "delta_pos4": delta4,
            "delta_pos5": delta5,
            "min_denominator": C5_MIN_DENOMINATOR,
            "corridor": [C5_CORRIDOR_LOW, C5_CORRIDOR_HIGH],
        }

    if delta4 <= 0:
        # Not a verdict: a window with no position-4 acceptance carries no
        # seam information at all, and saying "0.0" would invent one.
        return _no_signal("no-denominator")
    if delta4 < C5_MIN_DENOMINATOR:
        # The E-A case: a real ratio, but on too few samples to distinguish
        # from the healthy median. Reporting a corridor verdict here would
        # flag short healthy tasks as degeneration shapes.
        return _no_signal("insufficient-denominator")
    value = delta5 / delta4
    return {
        "schema": SCHEMA,
        "label": label,
        "c5": value,
        "verdict": classify(value),
        "delta_pos4": delta4,
        "delta_pos5": delta5,
        "min_denominator": C5_MIN_DENOMINATOR,
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


def seam_applicability(runroot: Path) -> dict[str, object]:
    """Does this arm's drafter HAVE the seam c5 measures? From evidence.

    Never from the filename. Two independent sources, either of which settles
    it, and a declaration outranks an inference:

      * ``c5_applicable=NO`` stamped by the arm itself;
      * ``num_spec_tokens`` from the engine's boot log -- the seam at
        C5_NUMERATOR_POSITION exists only if the drafter proposes past it.

    UNDETERMINED IS NOT APPLICABLE. An arm whose drafter cannot be identified
    contributes nothing and says so, because silently admitting it is exactly
    how the four chain arms got in.
    """
    declared: str | None = None
    spec_tokens: set[int] = set()
    evidence: list[str] = []
    for pattern in _APPLICABILITY_EVIDENCE:
        for path in sorted(runroot.rglob(pattern)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            stamp = _C5_APPLICABLE_STAMP.search(text)
            found = {int(value) for value in _NUM_SPEC_TOKENS.findall(text)}
            if stamp is None and not found:
                continue
            evidence.append(str(path.relative_to(runroot)))
            if stamp is not None:
                declared = stamp.group(1).upper()
            spec_tokens |= found

    if declared == "NO":
        return {
            "applicable": False,
            "reason": "declared:c5_applicable=NO",
            "num_spec_tokens": sorted(spec_tokens) or None,
            "declared": declared,
            "evidence": evidence,
        }
    if not spec_tokens:
        return {
            "applicable": False,
            "reason": "undetermined:no-declared-drafter-width",
            "num_spec_tokens": None,
            "declared": declared,
            "evidence": evidence,
        }
    if len(spec_tokens) > 1:
        # One arm, one drafter. Two widths in one runroot means the evidence
        # does not describe a single serve, and averaging over that is how a
        # corpus statistic stops meaning anything.
        return {
            "applicable": False,
            "reason": f"ambiguous:multiple-drafter-widths={sorted(spec_tokens)}",
            "num_spec_tokens": sorted(spec_tokens),
            "declared": declared,
            "evidence": evidence,
        }
    width = next(iter(spec_tokens))
    if width <= C5_NUMERATOR_POSITION:
        return {
            "applicable": False,
            "reason": (
                f"no-seam:num_spec_tokens={width} proposes positions "
                f"0..{width - 1}, so position {C5_NUMERATOR_POSITION} "
                "does not exist"
            ),
            "num_spec_tokens": [width],
            "declared": declared,
            "evidence": evidence,
        }
    return {
        "applicable": True,
        "reason": f"seam-present:num_spec_tokens={width}",
        "num_spec_tokens": [width],
        "declared": declared,
        "evidence": evidence,
    }


def sweep_runroot(
    runroot: Path, *, include_inapplicable: bool = False
) -> list[dict[str, object]]:
    """Per-task c5 for every bracketed task under a runroot.

    An arm with no seam contributes NOTHING by default -- not a zero. Pass
    ``include_inapplicable`` to get one declared exclusion record per task
    instead, which is what the CLI prints so the exclusion is visible.
    """
    applicability = seam_applicability(runroot)
    out: list[dict[str, object]] = []
    for pre in sorted(runroot.rglob("per_task/*/vllm_metrics_pre.txt")):
        post = pre.with_name("vllm_metrics_post.txt")
        if not post.exists():
            continue
        if not applicability["applicable"]:
            if include_inapplicable:
                out.append(
                    {
                        "schema": SCHEMA,
                        "label": pre.parent.name,
                        "c5": None,
                        "verdict": f"not-applicable:{applicability['reason']}",
                        "seam_applicable": False,
                        "source": "vllm_metrics_bracket",
                    }
                )
            continue
        try:
            row = c5_from_scrapes(
                pre.read_text(errors="replace"),
                post.read_text(errors="replace"),
                label=pre.parent.name,
            )
        except C5Error:
            continue
        row["seam_applicable"] = True
        out.append(row)
    return out


def sweep_output_root(
    output_root: Path, *, include_inapplicable: bool = False
) -> list[tuple[str, dict[str, object]]]:
    """(runroot name, row) for every FR14 arm under an output root."""
    rows: list[tuple[str, dict[str, object]]] = []
    for runroot in sorted(output_root.glob("fr14_*")):
        if not runroot.is_dir():
            continue
        for row in sweep_runroot(
            runroot, include_inapplicable=include_inapplicable
        ):
            rows.append((runroot.name, row))
    return rows


# --------------------------------------------------------------------------- #
# THE PRE-REGISTERED CALIBRATION CORPUS                                        #
# --------------------------------------------------------------------------- #
# A corridor that moves when someone runs a new arm is not pre-registered. The
# corridor above was derived BEFORE this module existed; the set it was derived
# from is enumerated here so that new arms are CHECKED against it instead of
# being absorbed into it. Cqc10's ten perfectly legitimate rows shifted every
# calibration aggregate simply by existing -- that is the defect this closes,
# and it is separate from the chain-drafter one.
#
# THE COUNT. The pre-registration note records a 100-arm sweep; reproducing it
# from the bank today yields 99 applicable task-arms across these 47 runroots.
# Both readings agree on everything that decides the corridor -- 4 arms outside
# it, all four known degenerations, a healthy floor of 0.4517 and a minimum
# denominator of 215 -- so the difference is one in-band arm that is no longer
# on disk under this name. The measured number is pinned rather than the
# remembered one, and the discrepancy is recorded rather than rounded away.
C5_CALIBRATION_RUNROOTS: tuple[str, ...] = (
    "fr14_b1_stock_20260816T200746Z",
    "fr14_b1_stock_20260816T204931Z",
    "fr14_b1_stock_20260817T020534Z",
    "fr14_b1_stock_20260817T031507Z",
    "fr14_b1_stock_20260817T054447Z",
    "fr14_gqa_k0_gate_20260817T081816Z",
    "fr14_gqa_k0_gate_20260817T083701Z",
    "fr14_gqa_k0_gate_20260817T091550Z",
    "fr14_gqa_k0_gate_20260817T093130Z",
    "fr14_gqa_k0_gate_20260817T095444Z",
    "fr14_gqa_k0_gate_20260817T124914Z",
    "fr14_gqa_k0_gate_20260817T143129Z",
    "fr14_gqa_k0_gate_20260817T181354Z",
    "fr14_gqa_k0_gate_20260817T192648Z",
    "fr14_gqa_k0_gate_20260817T194827Z",
    "fr14_gqa_k0_gate_20260817T201301Z",
    "fr14_gqa_k0_gate_20260817T203855Z",
    "fr14_gqa_k0_gate_20260817T235503Z",
    "fr14_hydra27_lever_pair_20260817T101311Z",
    "fr14_hydra27_lever_pair_20260817T130251Z",
    "fr14_hydra27_lever_pair_20260817T144219Z",
    "fr14_maxstack_20260817T210423Z",
    "fr14_promoab_C_20260818T035118Z",
    "fr14_promoab_Ch27_20260819T064150Z",
    "fr14_promoab_Ch27i_20260819T085259Z",
    "fr14_promoab_Ch27n_20260819T112147Z",
    "fr14_promoab_Cp0_20260818T120217Z",
    "fr14_promoab_Cp1_20260818T081918Z",
    "fr14_promoab_Cqc16_20260819T222438Z",
    "fr14_promoab_Gp5_20260818T174541Z",
    "fr14_promoab_Gp6_20260818T205129Z",
    "fr14_promoab_Sr12_20260819T043506Z",
    "fr14_promoab_Sr6_20260819T005801Z",
    "fr14_promoab_gate_20260818T033233Z",
    "fr14_promoab_gate_20260818T080612Z",
    "fr14_promoab_gate_20260818T151743Z",
    "fr14_promoab_gate_20260818T155348Z",
    "fr14_promoab_gate_20260818T170922Z",
    "fr14_promoab_gate_20260818T172912Z",
    "fr14_promoab_gate_20260818T203938Z",
    "fr14_promoab_gate_20260819T013426Z",
    "fr14_promoab_gate_20260819T024021Z",
    "fr14_promoab_gate_20260819T030908Z",
    "fr14_promoab_gate_20260819T034114Z",
    "fr14_promoab_gate_20260819T042359Z",
    "fr14_promoab_gate_20260819T062951Z",
    "fr14_promoab_gate_20260819T083216Z",
)
#: Rows and digest of the pinned set, so absorption is LOUD rather than silent.
#: The digest covers sorted [runroot, task, round(c5, 6)] triples: a new arm
#: entering the calibration set, or an existing one changing value, moves it.
C5_CALIBRATION_ROWS = 99
C5_CALIBRATION_SHA256 = (
    "362e39712fc96d6b8f668112667e9f4fc8a578cab4fad68505a95553bfb130b4"
)


def calibration_digest(rows: list[tuple[str, dict[str, object]]]) -> str:
    """Canonical digest of a calibration set's own numbers."""
    triples = sorted(
        [name, str(row["label"]), round(float(row["c5"]), 6)]
        for name, row in rows
        if row.get("c5") is not None
    )
    return hashlib.sha256(
        json.dumps(triples, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def calibration_corpus(output_root: Path) -> list[tuple[str, dict[str, object]]]:
    """The pinned set only. New arms cannot enter it by being run."""
    rows: list[tuple[str, dict[str, object]]] = []
    for name in C5_CALIBRATION_RUNROOTS:
        runroot = output_root / name
        if not runroot.is_dir():
            continue
        for row in sweep_runroot(runroot):
            rows.append((name, row))
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="c5 seam gate (flags, never refuses)")
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # ALWAYS include the exclusions. An arm that drops out of a sweep without
    # saying so is indistinguishable from one that was never looked at, and
    # that is the same absent-vs-measured confusion in the reader's head.
    applicability = seam_applicability(args.runroot)
    rows = sweep_runroot(args.runroot, include_inapplicable=True)
    if args.json:
        print(
            json.dumps(
                {"applicability": applicability, "rows": rows},
                indent=1,
                sort_keys=True,
            )
        )
    else:
        if not applicability["applicable"]:
            print(f"  NOT APPLICABLE: {applicability['reason']}")
            print("  c5 is a seam conditional; this arm contributes no c5 at all.")
        for row in rows:
            value = row["c5"]
            shown = "  n/a " if value is None else f"{value:.4f}"
            print(f"  c5={shown}  {row['verdict']:52s} {row['label']}")
    # FLAGS ONLY. Exit 0 whatever the shapes are: the eyeball adjudicates, and
    # a gate that exits non-zero here would become a refusal by the back door.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
