#!/usr/bin/env python3
"""FR13_REPLAY_BOUNDARY reducer (CPU, offline).

Joins tap A (producer: committer replay as-written dst bank rows) to the NEXT
tap B (consumer: scan h0 as-read) per request, byte-diffs as-read vs
as-written, and classifies every joined event:

  - keying_drift : read column/row != the column/row the replay wrote
                   (window slide / lens drift / REQKEY class)
  - never_written_read : the h0 row the scan read was not written by the
                   replay at the prior commit (R6g stale-column class)
  - byte_delta   : same row, different sha (something overwrote the bytes in
                   the producer-write..consumer-read interval; tap C records
                   are the attribution)
  - lens_drift   : consumer lens_now != producer accepted_len (R1/REQKEY)

Also surfaces tap C stale_read copies (native temporal copy sourcing a row
the replay never wrote) and tap D resets (native num_accepted reset-to-1
while the tree lens buffer still holds >1), and emits the legacy-style
nonzero next-read delta counter per event class (flag-ON analogue of the
legacy 2/113 instrument).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(paths):
    records = []
    for p in paths:
        with open(p) as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{p}:{line_no}: bad JSON: {exc}")
                rec["_src"] = f"{p}:{line_no}"
                records.append(rec)
    return records


def spine_path(length):
    return list(range(1, length + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--layer", default="layers.0.linear_attn",
                    help="substring filter for probed layer records")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-anomalies", type=int, default=24)
    args = ap.parse_args()

    records = load_records(args.log)
    headers = [r for r in records if r.get("tap") == "header"]
    init_meta = [r for r in records if r.get("tap") == "init_meta"]
    if not headers:
        raise SystemExit(
            "refusing to reduce: no engagement header record found "
            "(predicted-by-design check)"
        )

    last_a: dict[str, dict] = {}
    c_since_a: dict[str, list] = {}
    joins = []
    b_unjoined = []
    c_stale = []
    d_records = [r for r in records if r.get("tap") == "D"]

    for rec in records:
        tap = rec.get("tap")
        if tap == "A" and args.layer in str(rec.get("layer", "")):
            rid = rec.get("req_id")
            last_a[rid] = rec
            c_since_a[rid] = []
        elif tap == "C":
            rid = rec.get("req_id")
            if rid in c_since_a:
                c_since_a[rid].append(rec)
            if rec.get("stale_read"):
                c_stale.append(rec)
        elif tap == "B" and args.layer in str(rec.get("layer", "")):
            rid = rec.get("req_id")
            a = last_a.get(rid)
            if a is None:
                b_unjoined.append(rec)
                continue
            joins.append((a, rec, list(c_since_a.get(rid, []))))

    summary_rows = []
    anomalies = []
    class_counts = {}
    class_nonzero = {}

    for a, b, cs in joins:
        alen = int(a.get("accepted_len", 0))
        apath = [int(x) for x in (a.get("accepted_path") or [])][:alen]
        window_written = a.get("window_written") or []
        window_now = b.get("window_now") or []
        by_col = a.get("dst_rows") or []
        by_col_map = {int(d["col"]): d for d in by_col}
        sha_by_row = {int(d["row"]): d["sha4096"] for d in by_col}
        first8_by_row = {int(d["row"]): d["first8"] for d in by_col}
        rows_written = sorted({int(d["row"]) for d in by_col})

        if alen == 0:
            klass = "zero_accept"
        elif apath != spine_path(alen):
            klass = "branch_commit"
        else:
            klass = "spine_commit"
        slide = bool(window_written and window_now and
                     window_written != window_now)
        if slide:
            klass += "+slide"

        lens_now = int(b.get("lens_now", -1))
        lens_drift = lens_now != alen
        expected_col = max(alen - 1, 0)
        read_col = int(b.get("read_col", -1))
        read_row = int(b.get("read_row", -1))
        wrote = by_col_map.get(read_col)
        keying_drift = (read_col != expected_col) or (
            wrote is not None and int(wrote["row"]) != read_row
        )
        never_written = read_row not in rows_written
        byte_delta = (
            read_row in sha_by_row
            and b.get("sha4096") != sha_by_row[read_row]
        )
        delta_nonzero = bool(
            lens_drift or keying_drift or never_written or byte_delta
        )
        c_overwrites = [
            c for c in cs
            if args.layer in str(c.get("layer", ""))
            and (int(c.get("dest_row", -1)) == read_row
                 or int(c.get("src_row", -1)) == read_row)
        ]
        row = {
            "event": a.get("event"),
            "b_event": b.get("event"),
            "req_id": b.get("req_id"),
            "class": klass,
            "accepted_len": alen,
            "accepted_path": apath,
            "lens_now": lens_now,
            "lens_drift": lens_drift,
            "read_col": read_col,
            "expected_col": expected_col,
            "read_row": read_row,
            "rows_written": rows_written,
            "keying_drift": keying_drift,
            "never_written_read": never_written,
            "byte_delta": byte_delta,
            "delta_nonzero": delta_nonzero,
            "slide": slide,
            "as_written_first8": first8_by_row.get(read_row),
            "as_read_first8": b.get("first8"),
            "as_written_sha": sha_by_row.get(read_row),
            "as_read_sha": b.get("sha4096"),
            "c_overwrites": [
                {k: c.get(k) for k in (
                    "event", "phase", "copy_func", "src_row", "dest_row",
                    "accept_token_bias", "stale_read", "src_first8",
                    "num_elements", "src_block_idx", "dest_block_idx",
                )}
                for c in c_overwrites
            ],
            "a_src": a.get("_src"),
            "b_src": b.get("_src"),
        }
        summary_rows.append(row)
        class_counts[klass] = class_counts.get(klass, 0) + 1
        if delta_nonzero:
            class_nonzero[klass] = class_nonzero.get(klass, 0) + 1
            anomalies.append(row)

    first_anomaly = anomalies[0] if anomalies else None
    out = {
        "headers": [
            {k: h.get(k) for k in ("ts", "pid", "flags")} for h in headers
        ],
        "init_meta": [
            {k: m.get(k) for k in (
                "mamba_block_size", "num_speculative_blocks")}
            for m in init_meta
        ],
        "counts": {
            "records": len(records),
            "joined_AB": len(joins),
            "b_unjoined_request_first": len(b_unjoined),
            "tap_D": len(d_records),
            "tap_C_stale_read": len(c_stale),
        },
        "next_read_delta_counter": {
            "nonzero": sum(class_nonzero.values()),
            "total": len(joins),
            "per_class_total": class_counts,
            "per_class_nonzero": class_nonzero,
        },
        "first_anomaly": first_anomaly,
        "first_tap_C_stale": c_stale[0] if c_stale else None,
        "first_tap_D": d_records[0] if d_records else None,
        "tap_D_with_tree_lens_gt1": [
            d for d in d_records
            if (d.get("tree_lens_tensor") or 0) > 1
        ][: args.max_anomalies],
        "anomalies_head": anomalies[: args.max_anomalies],
    }
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
