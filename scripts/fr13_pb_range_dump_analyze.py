#!/usr/bin/env python3
"""Analyze the FR13_PB_DEBUG_RANGE forensic dump (fr13.pb_range_dump.v1).

Cross-checks the -1-placeholder embedding-OOB signature (dbg7):
  span cols 0..16 = -1, final col valid, packer guard silent.
Prints, for each victim request (any req owning a bad position):
  - flat span layout vs input_batch order (order sanity)
  - scheduler's spec list for the victim (expect all -1)
  - prev_req_id_to_index / prev_positions rows (positional map state)
  - the padded draft tensor row at BOTH the positional index and the
    request-keyed index (pad row vs real drafts — the core question)
  - prev_sampled rows at both indices
  - token_ids_cpu window (what a pure CPU copy would have supplied)
  - identification of the one VALID column's value: matched against
    prev_sampled rows, draft-tensor entries, and the CPU window, to name
    which scatter wrote it (draft_len=0 sample scatter, stock spec scatter,
    FR13 repair, or CPU copy).
Usage: fr13_pb_range_dump_analyze.py <dump.pt>
"""
import sys

import torch


def main() -> int:
    path = sys.argv[1]
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert d.get("schema") == "fr13.pb_range_dump.v1", d.get("schema")

    req_ids = d["req_ids"]
    spans = d["spans_reconstructed"]  # (rid, offset, n) in input_batch order
    bad_pos = set(int(p) for p in d["bad_pos"])
    spec = d["scheduled_spec_decode_tokens"]
    nst = d["num_scheduled_tokens"]
    prev_map = d["prev_req_id_to_index"] or {}
    prev_pos = d["prev_positions"]
    draft = d["draft_token_ids"]
    draft_reqs = d["replay_draft_req_ids"] or []
    sampled_reqs = d["replay_sampled_req_ids"] or []
    prev_sampled = d["prev_sampled_token_ids"]
    nspec = int(d["num_spec_tokens"])

    print(f"vocab={d['vocab']} num_spec_tokens={nspec}")
    print(f"input_batch req order: {req_ids}")
    print(f"num_scheduled_tokens: {nst}")
    print(f"spans (reconstructed): {spans}")
    print(f"prev_req_id_to_index: {prev_map}")
    print(f"prev_positions: {prev_pos}")
    print(f"draft tensor: shape={d['draft_token_ids_shape']} "
          f"row_req_map={draft_reqs}")
    print(f"sampled row_req_map={sampled_reqs}")
    if torch.is_tensor(prev_sampled):
        print(f"prev_sampled rows: {prev_sampled.flatten().tolist()}")

    # order sanity: spans were reconstructed in input_batch order; verify
    # each span length matches num_scheduled for that req.
    for rid, off, n in spans:
        assert int(nst[str(rid)]) == int(n), (rid, nst.get(str(rid)), n)
    print("span-order sanity: OK (input_batch order == flat layout)")

    victims = sorted({rid for (rid, off, n) in spans
                      if any(off <= p < off + n for p in bad_pos)})
    for rid in victims:
        off, n = next((o, s) for (r, o, s) in spans if r == rid)
        cur_index = next(i for i, r in enumerate(req_ids) if str(r) == rid)
        print(f"\n=== VICTIM {rid} (cur_index={cur_index}, span [{off},{off + n})) ===")
        cols_bad = sorted(p - off for p in bad_pos if off <= p < off + n)
        cols_good = [c for c in range(n) if c not in cols_bad]
        print(f"bad cols: {cols_bad}")
        print(f"good cols: {cols_good}")
        sched = spec.get(rid, [])
        print(f"scheduled spec list ({len(sched)}): {sched}")
        prev_index = prev_map.get(rid, None)
        if prev_index is None and prev_pos is not None:
            prev_index = prev_pos[cur_index]
        print(f"positional prev_index: {prev_index}")
        rk_index = draft_reqs.index(rid) if rid in draft_reqs else None
        print(f"request-keyed draft row: {rk_index}")
        if torch.is_tensor(draft):
            for name, idx in (("positional", prev_index), ("req-keyed", rk_index)):
                if idx is not None and 0 <= int(idx) < draft.shape[0]:
                    row = draft[int(idx)].tolist()
                    tag = "PAD(-1)" if all(v < 0 for v in row) else "REAL"
                    print(f"draft row [{name} idx {idx}]: {tag} {row}")
        sk_index = sampled_reqs.index(rid) if rid in sampled_reqs else None
        if torch.is_tensor(prev_sampled):
            for name, idx in (("positional", prev_index), ("req-keyed", sk_index)):
                if idx is not None and 0 <= int(idx) < prev_sampled.shape[0]:
                    print(f"prev_sampled [{name} idx {idx}]: "
                          f"{prev_sampled[int(idx)].flatten().tolist()}")
        win = d["token_ids_cpu_windows"].get(rid)
        if win is not None:
            print(f"token_ids_cpu window (num_computed={win['num_computed']}): "
                  f"{win['window'].tolist()}")
        # name the writer of each good col by matching its value
        around = d["input_ids_around_bad"]
        vals = {}
        for p, t in around.items():
            base = max(0, int(p) - 2)
            for j, v in enumerate(t.tolist()):
                vals[base + j] = v
        for c in cols_good:
            v = vals.get(off + c)
            if v is None:
                continue
            sources = []
            if torch.is_tensor(prev_sampled) and v in prev_sampled.flatten().tolist():
                rows = [i for i in range(prev_sampled.shape[0])
                        if v in prev_sampled[i].flatten().tolist()]
                sources.append(f"prev_sampled rows {rows}")
            if torch.is_tensor(draft):
                rows = [i for i in range(draft.shape[0])
                        if v in draft[i].tolist()]
                if rows:
                    sources.append(f"draft rows {rows}")
            if win is not None and v in win["window"].tolist():
                sources.append("token_ids_cpu window")
            print(f"good col {c} value={v} matches: {sources or 'NOTHING KNOWN'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
