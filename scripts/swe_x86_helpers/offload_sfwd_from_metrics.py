#!/usr/bin/env python3
"""FR13 offload-validation: compute the CANONICAL deploy-speed s/fwd from a pair
of vLLM /metrics snapshots (read GB10-LOCALLY by the harness, NOT across the wire).

s/fwd = d(vllm:request_decode_time_seconds_sum) / d(vllm:spec_decode_num_drafts_total)

This is the SAME basis as fr13_measure.py (M_DECODE_S/M_DRAFTS). It does NOT
change the canonical measurement; it only reads the brackets the harness already
captured (metrics_before_swe.txt / metrics_after_swe.txt) so a clean (offloaded)
arm can be compared to a contaminated (co-located) arm.
"""
import sys, json

M_DECODE_S = "vllm:request_decode_time_seconds_sum"
M_DRAFTS = "vllm:spec_decode_num_drafts_total"
M_DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
M_ACCEPT = "vllm:spec_decode_num_accepted_tokens_total"


def _val(path: str, name: str) -> float:
    total = 0.0
    found = False
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    total += float(line.rsplit(None, 1)[-1])
                    found = True
                except ValueError:
                    pass
    return total if found else 0.0


def main() -> None:
    before, after, label = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    d_dec = _val(after, M_DECODE_S) - _val(before, M_DECODE_S)
    d_drafts = _val(after, M_DRAFTS) - _val(before, M_DRAFTS)
    d_tok = _val(after, M_DRAFT_TOK) - _val(before, M_DRAFT_TOK)
    d_acc = _val(after, M_ACCEPT) - _val(before, M_ACCEPT)
    out = {
        "label": label,
        "d_request_decode_time_seconds_sum": round(d_dec, 6),
        "d_spec_decode_num_drafts_total": d_drafts,
        "s_per_fwd": round(d_dec / d_drafts, 8) if d_drafts > 0 else None,
        "draft_tokens_per_draft": round(d_tok / d_drafts, 4) if d_drafts > 0 else None,
        "accepted_per_draft": round(d_acc / d_drafts, 4) if d_drafts > 0 else None,
        "speed_basis": "d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
