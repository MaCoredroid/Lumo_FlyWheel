#!/usr/bin/env python3
"""FR13 big-denominator Phase-3 consolidation: Wilson 95% CI + non-vacuity gate.

Reads the per-arm recurrent-decode-oracle rescore artifacts
(fr13.recurrent_decode_oracle.rescore.v1) + the per-arm reducer src
(fr13.swe_stream_to_oracle_src.v1, for the round-trip-validated denominator and
dropped-turn accounting) and emits one consolidated Phase-3 artifact:

  * clear-margin flip RATE + Wilson 95% CI per arm (CI on the REAL validated
    round-trip position count -- class #12: NOT text length, NOT usage tokens),
  * within-process rep1==rep2 determinism per arm (class #8),
  * NON-VACUITY GATE (class #9, code-enforced asserts):
      - oracle engaged: recurrent_decode_calls > 0 AND RECURRENT_PATH_ENGAGED,
      - GDN layers introspected (>0 or counter is the binding gate),
      - the rescore denominator == the reducer's n_positions_scored (no silent
        re-count; the binomial CI is on the validated round-trip token count),
      - gold-margin NOT streamed logprobs: the oracle records the clean recurrent
        argmax/logprob in-process per step (asserted by schema == the
        recurrent_decode_oracle schema, NOT a streamed-logprob HTTP instrument).

CPU-only (pure stdlib math). Spec-counters-frozen is asserted by the launcher
(FR12_NO_SPECULATIVE_CONFIG=1 -> no speculative_config in the engine -> spec
counters CANNOT advance); this consolidator records that flag-state evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
    return (phat, max(0.0, center - half), min(1.0, center + half))


def fmt_pct_ci(k: int, n: int) -> str:
    phat, lo, hi = wilson_ci(k, n)
    return (
        f"{k}/{n} = {phat*100:.3f}% (Wilson95 [{lo*100:.3f}%, {hi*100:.3f}%], "
        f"halfwidth {(hi-lo)/2*100:.3f}pp)"
    )


def load_arm(rescore_path: str, src_path: str) -> dict[str, Any]:
    rs = json.loads(Path(rescore_path).read_text(encoding="utf-8"))
    src = json.loads(Path(src_path).read_text(encoding="utf-8"))
    # NON-VACUITY ASSERTS (class #9) -----------------------------------------
    assert rs.get("schema") == "fr13.recurrent_decode_oracle.rescore.v1", (
        f"rescore schema mismatch (not the recurrent oracle / streamed-logprobs?): "
        f"{rs.get('schema')}"
    )
    assert src.get("schema") == "fr13.swe_stream_to_oracle_src.v1", (
        f"src schema mismatch: {src.get('schema')}"
    )
    recur_calls = int(rs.get("recurrent_decode_calls", 0))
    engaged = bool(rs.get("RECURRENT_PATH_ENGAGED")) and recur_calls > 0
    assert engaged, (
        f"VACUOUS (#9): oracle not engaged (recurrent_decode_calls={recur_calls}, "
        f"RECURRENT_PATH_ENGAGED={rs.get('RECURRENT_PATH_ENGAGED')})"
    )
    n_rescored = int(rs.get("total_positions", 0))
    n_src = int(src.get("n_positions_scored", 0))
    # denominator integrity (class #12): the rescored positions == the validated
    # round-trip token count the reducer kept (rep1 has exactly n_src positions).
    assert n_rescored == n_src, (
        f"DENOMINATOR MISMATCH (#12): rescored {n_rescored} positions != "
        f"reducer round-trip-validated {n_src} (text-length / silent re-count?)"
    )
    k = int(rs.get("total_clear_margin_flips", 0))
    phat, lo, hi = wilson_ci(k, n_src)
    return {
        "rescore_path": rescore_path,
        "src_path": src_path,
        "RECURRENT_PATH_ENGAGED": engaged,
        "recurrent_decode_calls": recur_calls,
        "n_gdn_layers_introspected": rs.get("n_gdn_layers_introspected"),
        "within_proc_det_all_prompts": bool(rs.get("within_proc_det_all_prompts")),
        "threshold_nat": rs.get("threshold_nat"),
        "top_k": rs.get("top_k"),
        "total_positions_rescored": n_rescored,
        "n_positions_scored_src": n_src,
        "n_turns_total": src.get("n_turns_total"),
        "n_turns_kept": src.get("n_turns_kept"),
        "n_turns_dropped": src.get("n_turns_dropped"),
        "n_positions_dropped": src.get("n_positions_dropped"),
        "total_flips": rs.get("total_flips"),
        "clear_margin_flips": k,
        "per_prompt_clear_margin_flip_counts": rs.get(
            "per_prompt_clear_margin_flip_counts"
        ),
        "clear_margin_rate": phat,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "wilson95_halfwidth": (hi - lo) / 2,
        "rate_ci_str": fmt_pct_ci(k, n_src),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat9-rescore", required=True)
    ap.add_argument("--cat9-src", required=True)
    ap.add_argument("--native-rescore", required=True)
    ap.add_argument("--native-src", required=True)
    ap.add_argument(
        "--spec-frozen-evidence", default="",
        help="path to a JSON/text snapshot proving FR12_NO_SPECULATIVE_CONFIG=1 / no spec config",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat9 = load_arm(args.cat9_rescore, args.cat9_src)
    native = load_arm(args.native_rescore, args.native_src)

    # CI separation (the tight big-denominator answer)
    separated = cat9["wilson95_lo"] > native["wilson95_hi"]
    det_both = cat9["within_proc_det_all_prompts"] and native["within_proc_det_all_prompts"]

    out = {
        "schema": "fr13.bigdenom_rescore_consolidate.v1",
        "compare_target": "cat9 vs native-E5, each vs its own no-spec RECURRENT decode oracle",
        "clear_margin_def": "served != recurrent_argmax AND (served outside oracle top-20 OR deviation_nat > 1.0)",
        "spec_frozen_evidence_path": args.spec_frozen_evidence or None,
        "non_vacuity": {
            "oracle_engaged_both": cat9["RECURRENT_PATH_ENGAGED"] and native["RECURRENT_PATH_ENGAGED"],
            "denominator_is_validated_roundtrip_tokens": (
                cat9["total_positions_rescored"] == cat9["n_positions_scored_src"]
                and native["total_positions_rescored"] == native["n_positions_scored_src"]
            ),
            "gold_margin_in_process_recurrent_not_streamed_logprobs": True,
            "within_proc_determinism_both": det_both,
        },
        "cat9": cat9,
        "native": native,
        "ci_separated_cat9_above_native": separated,
        "within_proc_determinism_both": det_both,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cat9": cat9["rate_ci_str"],
        "native": native["rate_ci_str"],
        "ci_separated": separated,
        "determinism_both": det_both,
        "oracle_engaged_both": out["non_vacuity"]["oracle_engaged_both"],
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
