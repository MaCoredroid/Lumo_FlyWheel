#!/usr/bin/env python3
"""fr13_e4cap_reduce.py — reduce the D-E4CAP prefill-carry trace (campaign sec.51).

INPUT: the JSONL emitted by FR13_APC_PREFILL_CARRY_TRACE (one record per prefill
forward per layer, two sites per pair):
  read : {seq, layer, num_prefills, seq_lens, has_initial_state, state_indices,
          site='read', read_fp}   -- fingerprint is AS-READ (before the :986 zeroing)
  write: {seq, layer, ..., state_indices, site='write', write_indices, written_fp}

The tracer logs a FLAT stream with no request id, so this reducer segments the
stream into cold requests per layer: a new request begins at a COLD chunk-0
(has_initial_state present and all-False). Then, for each layer, it compares
request-1 vs request-2 chunk-by-chunk on read_fp and reports the FIRST divergence
and the block id (state_index) at that row.

VERDICTS (campaign sec.51 D-E4CAP):
  S1_NULL_ROW        first read_fp divergence sits on state_index == 0 (the null
                     block) => request-1's unguarded carry WRITE deposited state
                     into null; every later cold request's carry READ consumes it.
                     fix = mask the :1004 write for index==0 (+ optional zero-null).
  S2_STALE_CARRY     first divergence on a non-zero, non-fresh state_index =>
                     stale carry index escaping the request's own alloc set.
                     fix = correct the carry index.
  RECURRENT_EXONERATED  read_fp identical across requests at every (layer, chunk)
                     AND no null id read => the prefill recurrent carry is clean;
                     hunt non-recurrent carriers (full-attn null / positional /
                     sampler buffers).
  INCONCLUSIVE       could not segment >=2 requests (need >=2 identical cold
                     requests with resets between; raise the trace LIMIT if capped).

Corroborating signals always reported: null-id READ occurrences (state_indices
contains 0) and write_indices==0 events (request writing INTO the null block).
"""
import argparse
import collections
import json
import sys


def load_records(path):
    recs = []
    bad = 0
    with open(path, errors="ignore") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                recs.append(json.loads(ln))
            except Exception:
                bad += 1  # tolerate a truncated final line (container kill)
    return recs, bad


def is_cold_start(rec):
    his = rec.get("has_initial_state")
    return isinstance(his, list) and len(his) > 0 and not any(his)


def segment_layer_reads(reads, num_requests_hint=None):
    """reads: layer's read records in seq order -> list of requests, each a list
    of per-chunk records. New request starts at a cold chunk-0 (all-False his)."""
    requests = []
    cur = None
    for r in reads:
        if is_cold_start(r) or cur is None:
            # cold chunk-0 opens a new request (also the very first record)
            if is_cold_start(r):
                cur = []
                requests.append(cur)
            elif cur is None:
                cur = []
                requests.append(cur)
        cur.append(r)
    # Fallback: no cold markers found at all -> optionally equal-split
    if len(requests) <= 1 and num_requests_hint and num_requests_hint > 1:
        flat = reads
        n = num_requests_hint
        if len(flat) % n == 0 and len(flat) > 0:
            k = len(flat) // n
            requests = [flat[i * k:(i + 1) * k] for i in range(n)]
    return requests


def fp_rows_differ(a, b, tol):
    """Return list of (row, va, vb) where |va-vb| > tol. Length-mismatch => flag
    every row up to min-len plus the extra rows."""
    if a is None or b is None:
        if a is b:
            return []
        return [(-1, a, b)]
    diffs = []
    n = min(len(a), len(b))
    for i in range(n):
        va, vb = a[i], b[i]
        if va is None or vb is None:
            if va is not vb:
                diffs.append((i, va, vb))
            continue
        if abs(float(va) - float(vb)) > tol:
            diffs.append((i, va, vb))
    for i in range(n, max(len(a), len(b))):
        va = a[i] if i < len(a) else None
        vb = b[i] if i < len(b) else None
        diffs.append((i, va, vb))
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="path to fr13_prefill_carry_trace.jsonl")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="abs tolerance on read_fp compare (default 0.0 = exact)")
    ap.add_argument("--num-requests", type=int, default=None,
                    help="fallback equal-split count if cold markers are absent")
    ap.add_argument("--req-a", type=int, default=0, help="first request index")
    ap.add_argument("--req-b", type=int, default=1, help="second request index")
    ap.add_argument("--json", action="store_true", help="emit machine JSON only")
    args = ap.parse_args()

    recs, bad = load_records(args.jsonl)
    reads = [r for r in recs if r.get("site") == "read"]
    writes = [r for r in recs if r.get("site") == "write"]

    # group reads by layer, in seq order
    by_layer = collections.OrderedDict()
    for r in sorted(reads, key=lambda x: x.get("seq", 0)):
        by_layer.setdefault(r.get("layer"), []).append(r)

    # corroborating signals
    null_reads = [
        {"seq": r.get("seq"), "layer": r.get("layer"),
         "state_indices": r.get("state_indices")}
        for r in reads
        if isinstance(r.get("state_indices"), list) and 0 in r["state_indices"]
    ]
    write_null = [
        {"seq": r.get("seq"), "layer": r.get("layer"),
         "write_indices": r.get("write_indices")}
        for r in writes
        if isinstance(r.get("write_indices"), list) and 0 in r["write_indices"]
    ]

    # per-layer request segmentation + first divergence
    layer_reports = []
    n_requests_seen = collections.Counter()
    for layer, rds in by_layer.items():
        requests = segment_layer_reads(rds, args.num_requests)
        n_requests_seen[len(requests)] += 1
        if len(requests) <= max(args.req_a, args.req_b):
            layer_reports.append({
                "layer": layer, "n_requests": len(requests),
                "status": "insufficient_requests",
            })
            continue
        ra = requests[args.req_a]
        rb = requests[args.req_b]
        n_chunks = min(len(ra), len(rb))
        first_div = None
        for c in range(n_chunks):
            diffs = fp_rows_differ(ra[c].get("read_fp"), rb[c].get("read_fp"),
                                   args.tol)
            if diffs:
                # block id at divergence: state_index of the first diverging row
                row = diffs[0][0]
                si_a = ra[c].get("state_indices") or []
                si_b = rb[c].get("state_indices") or []
                block_a = si_a[row] if 0 <= row < len(si_a) else None
                block_b = si_b[row] if 0 <= row < len(si_b) else None
                first_div = {
                    "chunk": c, "row": row,
                    "read_fp_a": diffs[0][1], "read_fp_b": diffs[0][2],
                    "block_id_a": block_a, "block_id_b": block_b,
                    "all_diff_rows": diffs,
                    "his_a": ra[c].get("has_initial_state"),
                    "his_b": rb[c].get("has_initial_state"),
                }
                break
        layer_reports.append({
            "layer": layer, "n_requests": len(requests),
            "n_chunks_compared": n_chunks,
            "first_divergence": first_div,
            "status": "diverges" if first_div else "identical",
        })

    # global verdict
    diverging = [lr for lr in layer_reports if lr.get("status") == "diverges"]
    verdict = "INCONCLUSIVE"
    verdict_layer = None
    if not by_layer:
        verdict = "INCONCLUSIVE"
    elif diverging:
        # earliest divergence by (chunk, then layer order)
        diverging.sort(key=lambda lr: lr["first_divergence"]["chunk"])
        top = diverging[0]
        fd = top["first_divergence"]
        bid = fd["block_id_b"] if fd["block_id_b"] is not None else fd["block_id_a"]
        verdict_layer = top["layer"]
        if bid == 0:
            verdict = "S1_NULL_ROW"
        elif bid is not None:
            verdict = "S2_STALE_CARRY"
        else:
            verdict = "S2_STALE_CARRY"  # divergence but no block id resolvable
    else:
        # no divergence anywhere
        segmented_ok = any(
            lr.get("n_requests", 0) >= 2 for lr in layer_reports
        )
        if segmented_ok:
            verdict = "RECURRENT_EXONERATED" if not null_reads \
                else "IDENTICAL_BUT_NULL_READ_PRESENT"
        else:
            verdict = "INCONCLUSIVE"

    out = {
        "input": args.jsonl,
        "n_records": len(recs),
        "n_reads": len(reads),
        "n_writes": len(writes),
        "malformed_lines": bad,
        "layers": len(by_layer),
        "requests_per_layer_histogram": dict(n_requests_seen),
        "null_id_read_count": len(null_reads),
        "null_id_read_examples": null_reads[:10],
        "write_indices_zero_count": len(write_null),
        "write_indices_zero_examples": write_null[:10],
        "verdict": verdict,
        "verdict_layer": verdict_layer,
        "per_layer": layer_reports,
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print("=== FR13 D-E4CAP prefill-carry reduce ===")
        print(f"input={args.jsonl}")
        print(f"records={len(recs)} reads={len(reads)} writes={len(writes)} "
              f"malformed={bad} layers={len(by_layer)}")
        print(f"requests/layer histogram: {dict(n_requests_seen)}")
        print(f"null-id READ occurrences: {len(null_reads)}"
              + (f"  e.g. {null_reads[0]}" if null_reads else ""))
        print(f"write_indices==0 events: {len(write_null)}"
              + (f"  e.g. {write_null[0]}" if write_null else ""))
        div = [lr for lr in layer_reports if lr.get("status") == "diverges"]
        print(f"diverging layers: {len(div)} / {len(layer_reports)}")
        for lr in div[:12]:
            fd = lr["first_divergence"]
            print(f"  layer={lr['layer']} first-div chunk={fd['chunk']} "
                  f"row={fd['row']} block_id_a={fd['block_id_a']} "
                  f"block_id_b={fd['block_id_b']} "
                  f"fp_a={fd['read_fp_a']} fp_b={fd['read_fp_b']}")
        print(f">>> VERDICT: {verdict}"
              + (f" (layer={verdict_layer})" if verdict_layer else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
