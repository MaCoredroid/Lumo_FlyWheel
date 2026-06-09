#!/usr/bin/env python3
"""FR13 tree-verify CORRUPTION GATE.

Catch a tree verifier that drops quality — the "multiple-span corruption ->
quality drop" failure class, which is the multi-spine / packed-token-tree state
contamination signature (see reference_multispine_not_lossless_closed_nonship,
docs/reports/auto_research/fr7-tokentree-gdn-superset-ceiling-20260531.md).

WHAT THAT CORRUPTION LOOKED LIKE (the signature this gate is tuned to catch):
  A single packed token tree shares ONE recurrent Markov state across all spines
  on the Qwen3.6 GDN hybrid. Sibling-spine state contaminates the deeper state
  of the top-1 chain, so path0 is no longer byte-identical to native E5. The
  observable signature (fr7-tokentree-gdn-superset-ceiling):
    - accept/event DROP: 3.150 (E5) -> 1.66 (tree path0). The chain is close to
      E5 at position 1 then COLLAPSES at depth.
    - full-accept rate collapse: 43.6% -> 1.7%.
    - superset_violations > 0 against NATIVE: `winner >= native E5` was false in
      ~42% of events, *even though* the in-tree `winner >= path0` claim showed
      superset_violations == 0. The internal-only superset is a TRAP; the gate
      MUST compare against native, not against the tree's own path0.
    - served-token divergence outside native self-noise: tree tokens differ from
      native at depth in a way native's own b1-vs-b4 run-noise does NOT explain.

  A live, weaker instance of the same class is on disk for a smoke test:
    output/fr13_conv_fix_same8_greedy_token_20260609T074753Z
    native accept/event 3.192 vs tree 1.897, tree TPS 9.16 vs native 19.06.

HOW THE GATE WORKS — three metrics, one comparator, fail-closed:

  Comparator = tree vs native-on-the-served-path, with run-noise masked by a
  native b1-vs-b4 self-noise baseline. We need THREE arms captured on the SAME
  prompts at B=4 CUDA-captured on the real workload:
    --tree-run     : the build under test (tree_mtp).
    --native-run   : native MTP-5 reference at full concurrency (the "b4" arm,
                     i.e. E5 / naive_mtp at MAX_NUM_SEQS=batch_size=4).
    --native-noise-run : a SECOND native MTP-5 arm (b1 or a re-run b4) used ONLY
                     to build the self-noise mask. Positions where the two native
                     arms disagree are run-noise and are EXCLUDED from the loss
                     count. (This is the A/B/C method from
                     FR13_BRANCH_NODE_SELF_NOISE_BIND.md.)

  Metric 1 — per-token argmax-match-rate vs native, self-noise-corrected:
    For each (prompt_id, position) compare tree vs native token. A mismatch is a
    REAL LOSS only if it is OUTSIDE the native self-noise mask (the set of
    positions where native disagrees with itself). The gate FAILS if the
    real-loss rate over self-noise-eligible positions exceeds a small tolerance,
    OR if any single sequence collapses (depth-collapse detector: a real-loss
    run, i.e. >= COLLAPSE_RUN consecutive outside-mask mismatches, which is the
    "close at pos 1, collapses at depth" fingerprint).

  Metric 2 — emitted-token bag-TV <= E5 floor (0.0593):
    Total-variation distance between the tree's and native's emitted-token bags,
    aggregate and per-prompt. Reuses lumo_flywheel_serving total_variation. The
    floor is the E5 self-noise floor; we ALSO compute native-vs-native bag-TV and
    require the tree's bag-TV to not exceed max(floor, native_self_bag_tv).

  Metric 3 — accept/event >= native (superset, against NATIVE not path0):
    From the probe summaries. tree accept/event must be >= native accept/event
    minus a tiny slack. A DROP here is the headline multi-spine fingerprint
    (3.150 -> 1.66). The gate ALSO recomputes per-event superset violations from
    the tree_path_lcp + native spec trace when present (fr10_superset_gate
    evaluate_superset_hard_gate / diagnose_spine_degradation), so a viol>0
    against native is reported and FAILS the gate.

FAIL-CLOSED conditions (any -> FAIL, valid=False):
    - any required arm missing / unreadable / zero records.
    - prompt-token identity not byte-identical across the paired arms
      (reuses fr13_argmax_lcp_localize._compute_prompt_identity; lcp=0 artifact).
    - self-noise mask empty AND there exist mismatches (cannot certify the loss
      is real-vs-noise without a baseline -> refuse to pass).
    - record-key sets differ across arms (cannot pair).
    - accept/event missing from a summary.

USAGE (CPU-only reduction of already-captured B=4 arms):
    python3 scripts/fr13_corruption_gate.py \
        --tree-run   <run>/tree \
        --native-run <run>/native \
        --native-noise-run <run>/native2 \
        --out <run>/fr13_corruption_gate.json

The arm directories are exactly what scripts/fr13_e2e_measure.py produces
(<arm>/{tree,native}_greedy_probe.json + <arm>/*_request_metrics.jsonl, and the
optional logs/ traces). Exit code 0 == PASS, 2 == FAIL/INVALID.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_superset_gate import (  # noqa: E402
    diagnose_spine_degradation,
    evaluate_superset_hard_gate,
    load_spec_trace,
    load_tree_accept_trace,
)
from lumo_flywheel_serving.fr10_equivalence_gate import total_variation  # noqa: E402


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_localizer = _load_script(
    "fr13_argmax_lcp_localize", REPO / "scripts" / "fr13_argmax_lcp_localize.py"
)

# ---- Thresholds (the E5 floor + multi-spine-signature detectors) -----------
# E5 emitted-token bag-TV self-noise floor (FR13_STRATEGY_VERDICT.md).
BAG_TV_FLOOR = 0.0593
# accept/event floor = native MTP-5 (reference_multispine_not_lossless_closed_nonship).
# tree accept/event must be >= native - this slack (a single-event paired jitter).
ACCEPT_PER_EVENT_SLACK = 0.05
# real-loss rate (outside-self-noise mismatches / self-noise-eligible positions)
# must not exceed this. native self-noise on same8 is ~85/368 ~= 0.23, so a tree
# that merely matches native run-noise lands ~0.0 here; the fr7 collapse was ~0.42
# RAW which is overwhelmingly outside the ~0.23 self-noise mask.
MAX_REAL_LOSS_RATE = 0.05
# depth-collapse detector: this many CONSECUTIVE outside-mask mismatches in ANY
# single sequence is the "close at pos1, collapses at depth" multi-spine
# fingerprint -> FAIL regardless of the aggregate rate.
COLLAPSE_RUN = 6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_by_key(probe: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Map (prompt_id, sample_index) -> served token_ids."""
    out: dict[tuple[int, int], list[int]] = {}
    for row in probe.get("records") or []:
        key = (int(row["prompt_id"]), int(row["sample_index"]))
        out[key] = [int(t) for t in row.get("token_ids", [])]
    return out


def _bag(records: dict[tuple[int, int], list[int]]) -> Counter[int]:
    bag: Counter[int] = Counter()
    for tokens in records.values():
        bag.update(tokens)
    return bag


def _accept_per_event(probe: dict[str, Any]) -> float | None:
    summary = probe.get("summary") or {}
    value = summary.get("accepted_per_draft_event")
    return float(value) if value is not None else None


def _warm_tps(probe: dict[str, Any]) -> float | None:
    summary = probe.get("summary") or {}
    value = summary.get("warm_decode_tps")
    return float(value) if value is not None else None


def _self_noise_mask(
    native: dict[tuple[int, int], list[int]],
    native_noise: dict[tuple[int, int], list[int]],
) -> set[tuple[int, int]]:
    """Positions (prompt_id, position) where native disagrees with itself.

    Aggregated across sample_index per prompt_id: a position is masked if ANY
    native/native_noise pair for that prompt disagrees there (run-noise).
    """
    mask: set[tuple[int, int]] = set()
    # group native by prompt_id (use the lowest sample_index as the canonical row)
    def _by_prompt(recs: dict[tuple[int, int], list[int]]) -> dict[int, list[int]]:
        chosen: dict[int, tuple[int, list[int]]] = {}
        for (pid, sidx), toks in recs.items():
            if pid not in chosen or sidx < chosen[pid][0]:
                chosen[pid] = (sidx, toks)
        return {pid: toks for pid, (_, toks) in chosen.items()}

    a = _by_prompt(native)
    b = _by_prompt(native_noise)
    for pid in set(a) & set(b):
        left, right = a[pid], b[pid]
        for pos in range(min(len(left), len(right))):
            if left[pos] != right[pos]:
                mask.add((pid, pos))
        # length disagreement past the common prefix is also run-noise
        for pos in range(min(len(left), len(right)), max(len(left), len(right))):
            mask.add((pid, pos))
    return mask


def _argmax_match(
    tree: dict[tuple[int, int], list[int]],
    native: dict[tuple[int, int], list[int]],
    mask: set[tuple[int, int]],
) -> dict[str, Any]:
    def _by_prompt(recs: dict[tuple[int, int], list[int]]) -> dict[int, list[int]]:
        chosen: dict[int, tuple[int, list[int]]] = {}
        for (pid, sidx), toks in recs.items():
            if pid not in chosen or sidx < chosen[pid][0]:
                chosen[pid] = (sidx, toks)
        return {pid: toks for pid, (_, toks) in chosen.items()}

    t = _by_prompt(tree)
    n = _by_prompt(native)
    eligible = 0
    real_losses = 0
    matches = 0
    compared = 0
    longest_loss_run = 0
    collapse: dict[str, Any] | None = None
    per_prompt: list[dict[str, Any]] = []
    first_real_loss: dict[str, Any] | None = None
    for pid in sorted(set(t) & set(n)):
        tt, nn = t[pid], n[pid]
        run = 0
        p_eligible = p_loss = 0
        for pos in range(min(len(tt), len(nn))):
            compared += 1
            masked = (pid, pos) in mask
            same = tt[pos] == nn[pos]
            if same:
                matches += 1
            if not masked:
                eligible += 1
                p_eligible += 1
                if not same:
                    real_losses += 1
                    p_loss += 1
                    run += 1
                    if first_real_loss is None:
                        first_real_loss = {
                            "prompt_id": pid,
                            "position": pos,
                            "tree_token": tt[pos],
                            "native_token": nn[pos],
                        }
                    if run > longest_loss_run:
                        longest_loss_run = run
                    if run >= COLLAPSE_RUN and collapse is None:
                        collapse = {
                            "prompt_id": pid,
                            "run_end_position": pos,
                            "run_len": run,
                        }
                else:
                    run = 0
            else:
                run = 0
        per_prompt.append(
            {
                "prompt_id": pid,
                "eligible_positions": p_eligible,
                "real_losses": p_loss,
                "real_loss_rate": (p_loss / p_eligible) if p_eligible else None,
            }
        )
    return {
        "compared_positions": compared,
        "raw_argmax_matches": matches,
        "raw_argmax_match_rate": (matches / compared) if compared else None,
        "self_noise_eligible_positions": eligible,
        "real_losses_outside_self_noise": real_losses,
        "real_loss_rate": (real_losses / eligible) if eligible else None,
        "self_noise_corrected_argmax_match_rate": (
            (eligible - real_losses) / eligible if eligible else None
        ),
        "longest_real_loss_run": longest_loss_run,
        "depth_collapse": collapse,
        "first_real_loss": first_real_loss,
        "per_prompt": per_prompt,
    }


def _superset_from_traces(tree_run: Path, native_run: Path, tree_topology: str | None) -> dict[str, Any] | None:
    """Per-event superset vs NATIVE from tree_path_lcp + native spec trace.

    Returns None if the optional traces are absent (the summary-level accept gate
    still runs). When present, a viol>0 against native FAILS — this is the
    fr7 'winner>=native false in ~42% of events' detector, NOT the internal
    'winner>=path0' trap.
    """
    tree_lcp = None
    for name in ("tree_path_lcp.jsonl", "tree_path_lcp_max.jsonl"):
        candidate = tree_run / "logs" / name
        if candidate.exists():
            tree_lcp = candidate
            break
    native_trace = native_run / "logs" / "per_req_spec_trace.jsonl"
    if tree_lcp is None or not native_trace.exists():
        return None
    tree_events = load_tree_accept_trace(
        tree_lcp, speculative_token_tree=tree_topology, source="tree"
    )
    native_events = load_spec_trace(native_trace, source="native")
    if not tree_events or not native_events:
        return None
    hard = evaluate_superset_hard_gate(
        native_events=native_events, tree_events=tree_events
    )
    degrade = diagnose_spine_degradation(
        native_events=native_events, tree_events=tree_events
    )
    return {
        "tree_events": len(tree_events),
        "native_events": len(native_events),
        "superset_hard_passed": hard.passed,
        "superset_violations": len(hard.violations),
        "first_superset_violation": hard.violations[0] if hard.violations else None,
        "superset_metrics": hard.metrics,
        "degradation_passed": degrade.passed,
        "degradation_classification": degrade.metrics.get("classification"),
        "first_degrade_event_index": degrade.metrics.get("first_degrade_event_index"),
    }


def run_gate(
    *,
    tree_run: Path,
    native_run: Path,
    native_noise_run: Path,
    tree_topology: str | None,
) -> dict[str, Any]:
    violations: list[str] = []
    invalid: list[str] = []

    # ---- load arms, fail closed on missing/empty ----
    arms = {
        "tree": tree_run / "tree_greedy_probe.json",
        "native": native_run / "native_greedy_probe.json",
        "native_noise": native_noise_run / "native_greedy_probe.json",
    }
    probes: dict[str, dict[str, Any]] = {}
    for label, path in arms.items():
        if not path.exists():
            invalid.append(f"missing arm probe: {label} -> {path}")
            continue
        probe = _load_json(path)
        if not (probe.get("records") or []):
            invalid.append(f"empty records in arm: {label} -> {path}")
            continue
        probes[label] = probe
    if invalid:
        return {"valid": False, "passed": False, "invalid_reasons": invalid}

    tree_rec = _records_by_key(probes["tree"])
    native_rec = _records_by_key(probes["native"])
    noise_rec = _records_by_key(probes["native_noise"])

    # ---- HARD pairing assert (prompt-token byte-identity), fail closed ----
    pairing: dict[str, Any] = {}
    try:
        pid = _localizer._compute_prompt_identity(tree_run, native_run)
        pairing["tree_vs_native"] = pid.get("byte_identical")
        if not pid.get("byte_identical"):
            invalid.append("prompt-token identity tree-vs-native not byte-identical (lcp=0 artifact risk)")
    except Exception as exc:  # noqa: BLE001
        pairing["tree_vs_native_error"] = str(exc)
        invalid.append(f"prompt identity check failed: {exc}")
    if invalid:
        return {"valid": False, "passed": False, "invalid_reasons": invalid, "prompt_identity": pairing}

    # record-key set must match for tree vs native (cannot pair otherwise)
    if set(tree_rec) != set(native_rec):
        invalid.append("tree/native record-key sets differ; cannot pair served paths")
        return {"valid": False, "passed": False, "invalid_reasons": invalid}

    # ---- self-noise mask ----
    mask = _self_noise_mask(native_rec, noise_rec)
    raw_mismatch_exists = any(
        tree_rec[k][p] != native_rec[k][p]
        for k in (set(tree_rec) & set(native_rec))
        for p in range(min(len(tree_rec[k]), len(native_rec[k])))
    )
    if not mask and raw_mismatch_exists:
        invalid.append(
            "self-noise mask is EMPTY while tree-vs-native mismatches exist; "
            "cannot certify loss-vs-noise without a native b1-vs-b4 baseline -> refuse PASS"
        )
        return {"valid": False, "passed": False, "invalid_reasons": invalid}

    # ---- METRIC 1: per-token argmax-match-rate, self-noise-corrected ----
    argmax = _argmax_match(tree_rec, native_rec, mask)
    real_loss_rate = argmax["real_loss_rate"]
    if real_loss_rate is not None and real_loss_rate > MAX_REAL_LOSS_RATE:
        violations.append(
            f"per-token real-loss rate {real_loss_rate:.4f} > {MAX_REAL_LOSS_RATE} "
            f"(outside-self-noise argmax mismatches; first={argmax['first_real_loss']})"
        )
    if argmax["depth_collapse"] is not None:
        violations.append(
            f"DEPTH COLLAPSE (multi-spine signature): >= {COLLAPSE_RUN} consecutive "
            f"outside-self-noise mismatches at {argmax['depth_collapse']}"
        )

    # ---- METRIC 2: emitted-token bag-TV <= floor ----
    tree_bag = _bag(tree_rec)
    native_bag = _bag(native_rec)
    noise_bag = _bag(noise_rec)
    bag_tv = total_variation(tree_bag, native_bag)
    native_self_bag_tv = total_variation(native_bag, noise_bag)
    bag_tv_budget = max(BAG_TV_FLOOR, native_self_bag_tv)
    if bag_tv > bag_tv_budget:
        violations.append(
            f"emitted-token bag-TV {bag_tv:.4f} > budget {bag_tv_budget:.4f} "
            f"(max(E5 floor {BAG_TV_FLOOR}, native-self {native_self_bag_tv:.4f}))"
        )

    # ---- METRIC 3: accept/event >= native (superset, vs NATIVE) ----
    tree_ape = _accept_per_event(probes["tree"])
    native_ape = _accept_per_event(probes["native"])
    if tree_ape is None or native_ape is None:
        invalid.append("accept/event missing from a probe summary")
        return {"valid": False, "passed": False, "invalid_reasons": invalid}
    accept_delta = tree_ape - native_ape
    if accept_delta < -ACCEPT_PER_EVENT_SLACK:
        violations.append(
            f"accept/event DROP (multi-spine signature): tree {tree_ape:.4f} < "
            f"native {native_ape:.4f} by {-accept_delta:.4f} (slack {ACCEPT_PER_EVENT_SLACK})"
        )

    # ---- per-event superset from traces (the winner>=NATIVE detector) ----
    superset = _superset_from_traces(tree_run, native_run, tree_topology)
    if superset is not None:
        if superset.get("superset_violations", 0) > 0:
            violations.append(
                f"per-event superset violations vs native: "
                f"{superset['superset_violations']} (first={superset['first_superset_violation']})"
            )
        if superset.get("degradation_classification") not in (None, "no_degradation"):
            violations.append(
                f"spine degradation classified: {superset['degradation_classification']} "
                f"at event {superset.get('first_degrade_event_index')}"
            )

    passed = not violations
    return {
        "schema": "fr13.corruption_gate.v1",
        "valid": True,
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "thresholds": {
            "bag_tv_floor": BAG_TV_FLOOR,
            "accept_per_event_slack": ACCEPT_PER_EVENT_SLACK,
            "max_real_loss_rate": MAX_REAL_LOSS_RATE,
            "collapse_run": COLLAPSE_RUN,
        },
        "prompt_identity": pairing,
        "self_noise": {
            "mask_positions": len(mask),
            "native_self_bag_tv": native_self_bag_tv,
        },
        "metric1_argmax_match_self_noise_corrected": argmax,
        "metric2_bag_tv": {
            "emitted_token_bag_tv": bag_tv,
            "budget": bag_tv_budget,
            "native_self_bag_tv": native_self_bag_tv,
            "floor": BAG_TV_FLOOR,
        },
        "metric3_accept_per_event": {
            "tree": tree_ape,
            "native": native_ape,
            "delta": accept_delta,
            "tree_warm_tps": _warm_tps(probes["tree"]),
            "native_warm_tps": _warm_tps(probes["native"]),
        },
        "per_event_superset": superset,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-run", required=True, type=Path)
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument(
        "--native-noise-run",
        required=True,
        type=Path,
        help="second native MTP-5 arm (b1 or re-run b4) for the self-noise mask",
    )
    parser.add_argument(
        "--tree-topology",
        default=(
            "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), "
            "(0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"
        ),
        help="speculative_token_tree string for the per-event superset trace gate",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = run_gate(
        tree_run=args.tree_run,
        native_run=args.native_run,
        native_noise_run=args.native_noise_run,
        tree_topology=args.tree_topology,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (result.get("valid") and result.get("passed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
