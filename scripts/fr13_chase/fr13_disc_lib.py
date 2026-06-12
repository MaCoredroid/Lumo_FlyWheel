#!/usr/bin/env python3
"""Shared event-walking helpers for the FR13 S1/S2/S3 discriminator run.

All walkers are CPU-only readers of run-window artifacts:
  - probe json (scripts/fr10_quick_decode_tps_probe.py --out)
  - tree_path_lcp.jsonl   (tree committer rows, one per verify event)
  - per_req_spec_trace.jsonl (scheduler per-event acc, native arm)
  - fr10_mtp_draft_trace.jsonl (drafter proposals, native arm)
Flag state for every window is recorded in run_header.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_probe(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probe_records(probe: dict[str, Any]) -> list[dict[str, Any]]:
    recs = probe.get("records")
    if recs is None:
        # multi-mode probe: single mode expected per window
        modes = list(probe["modes"])
        assert len(modes) == 1, modes
        raise RuntimeError("probe json has no records list")
    return sorted(recs, key=lambda r: (r["prompt_id"], r["sample_index"]))


def walk_tree_events(
    lcp_rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assign committer rows to probe records by walking emitted tokens.

    Returns one dict per event:
      prompt_id, event_idx_in_prompt, global_event_idx (file order),
      gen_pos (generated tokens before event), prefix (full token list),
      row (the committer row), clipped (bool).
    Asserts emitted_tokens == served token_ids slice (final event may clip).
    """
    events = []
    it = iter(enumerate(lcp_rows))
    interstitial = []
    for rec in records:
        toks = rec["token_ids"]
        prompt_toks = rec["prompt_token_ids"]
        pos = 1  # token_ids[0] is sampled at prefill, before any spec event
        ev_i = 0
        while pos < len(toks):
            try:
                gidx, row = next(it)
            except StopIteration as exc:
                raise AssertionError(
                    f"ran out of committer rows: prompt {rec['prompt_id']} pos {pos}"
                ) from exc
            emitted = row["emitted_tokens"]
            k = min(len(emitted), len(toks) - pos)
            if emitted[:k] != toks[pos : pos + k]:
                if ev_i == 0:
                    # trailing fully-clipped row from the previous request
                    # (forward ran, all tokens dropped past max_tokens)
                    interstitial.append({"global_row_idx": gidx, "emitted": emitted})
                    if len(interstitial) > 3 * len(records):
                        raise AssertionError("too many interstitial committer rows")
                    continue
                raise AssertionError(
                    f"emitted mismatch prompt {rec['prompt_id']} pos {pos}: "
                    f"emitted={emitted} served={toks[pos:pos + len(emitted)]}"
                )
            events.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "event_idx_in_prompt": ev_i,
                    "global_event_idx": gidx,
                    "gen_pos": pos,
                    "prefix": list(prompt_toks) + toks[:pos],
                    "row": row,
                    "clipped": k < len(emitted),
                    "n_emitted_served": k,
                }
            )
            pos += k
            ev_i += 1
    leftover = sum(1 for _ in it)
    return events, {"leftover_rows": leftover, "interstitial_rows": interstitial}


def walk_native_events(
    spec_rows: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    num_spec: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair scheduler spec rows (acc per event) + drafter rows with the
    served streams.  Greedy invariant: draft[:acc] == served slice and
    (acc==num_spec or draft[acc] != served[pos+acc]).

    Draft rows that fail the invariant for the next expected event are
    skipped (trailing unverified proposals at request end, etc.) and counted.
    Returns events (prompt_id, gen_pos, prefix, acc, draft, global order) and
    diagnostics counts.
    """
    # group spec rows per request id in order of first appearance
    rid_order: list[str] = []
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for row in spec_rows:
        rid = row["rid"]
        if rid not in by_rid:
            by_rid[rid] = []
            rid_order.append(rid)
        by_rid[rid].append(row)
    if len(rid_order) != len(records):
        raise AssertionError(
            f"request count mismatch: {len(rid_order)} rids vs {len(records)} records"
        )
    drafts = [r["draft"][0] if isinstance(r["draft"][0], list) else r["draft"] for r in draft_rows]
    di = 0
    skipped = 0
    events = []
    gidx = 0
    for rec, rid in zip(records, rid_order):
        toks = rec["token_ids"]
        prompt_toks = rec["prompt_token_ids"]
        pos = 1  # token_ids[0] is sampled at prefill, before any spec event
        ev_i = 0
        for row in by_rid[rid]:
            acc = int(row["acc"])
            emitted_n = min(acc + 1, len(toks) - pos)
            expect = toks[pos : pos + emitted_n]
            # find the draft row for this event
            found = None
            scan = di
            while scan < len(drafts):
                d = drafts[scan]
                # clipped final event: only len(toks)-pos accepted tokens were
                # served; compare the draft prefix that actually reached the
                # stream (acc itself can exceed the remaining budget)
                eff = min(acc, len(toks) - pos)
                ok = list(d[:eff]) == toks[pos : pos + eff] and (
                    acc >= len(d)
                    or pos + acc >= len(toks)
                    or d[acc] != toks[pos + acc]
                    or acc + 1 > emitted_n  # clipped final event: bonus never served
                )
                if ok:
                    found = scan
                    break
                scan += 1
                skipped += 1
            if found is None:
                raise AssertionError(
                    f"no draft row matches event: rid={rid} pos={pos} acc={acc} expect={expect}"
                )
            di = found + 1
            events.append(
                {
                    "prompt_id": rec["prompt_id"],
                    "event_idx_in_prompt": ev_i,
                    "global_event_idx": gidx,
                    "gen_pos": pos,
                    "prefix": list(prompt_toks) + toks[:pos],
                    "acc": acc,
                    "draft": list(drafts[found]),
                    "emitted": expect,
                    "clipped": emitted_n < acc + 1,
                }
            )
            pos += emitted_n
            ev_i += 1
            gidx += 1
        if pos != len(toks):
            raise AssertionError(f"stream not fully covered rid={rid}: pos={pos}/{len(toks)}")
    return events, {"draft_rows": len(drafts), "draft_rows_skipped": skipped}


def lockstep_pairs(
    a_events: list[dict[str, Any]], b_events: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Events from two arms with identical (prompt_id, full committed prefix)."""
    index: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    for ev in b_events:
        index[(ev["prompt_id"], tuple(ev["prefix"]))] = ev
    out = []
    for ev in a_events:
        match = index.get((ev["prompt_id"], tuple(ev["prefix"])))
        if match is not None:
            out.append((ev, match))
    return out


CATERPILLAR_SPINE = [0, 1, 3, 5, 7]
CATERPILLAR_ALT_AT_DEPTH = {1: 2, 2: 4, 3: 6, 4: 8}  # depth -> alt node id
CHAIN_SPINE = [0, 1, 2, 3, 4]


def tree_spine_draft(row: dict[str, Any], spine: list[int]) -> list[int]:
    d = row["draft_token_ids"]
    return [int(d[i]) for i in spine]
