#!/usr/bin/env python3
"""Byte gate: fr13_taw_products_device (S1-full in-capture tensor route) vs
the host-list committer route (fr13_taw_materialize + the python loops in
_lumo_tree_canonical_multidraft_sample). CPU-only, no GPU needed.

Host-route contract being matched (rejection_sampler committer tail):
  output_token_ids.fill_(-1); output_token_ids[i, pos] = out_rows[i][pos]
  accepted_tree_rows <- accepted_rows (== accepted_lens == path_len)
  _gdn_path = [node+1 for node in path[:len]]; _gdn_row = last or 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_dm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fr13_device_multidraft_kernel.py"))
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

g = torch.Generator().manual_seed(42)
fails = 0
for case in range(200):
    nreq = int(torch.randint(1, 6, (1,), generator=g))
    row_cap = int(torch.randint(1, 24, (1,), generator=g))
    cols = int(torch.randint(row_cap, row_cap + 8, (1,), generator=g))
    row_buf = torch.randint(0, 250000, (nreq, row_cap), generator=g)
    path_buf = torch.randint(0, 84, (nreq, row_cap), generator=g)
    row_len = torch.randint(0, row_cap + 1, (nreq,), generator=g)
    # path_len <= row_len is the walk invariant; include 0 (reject-at-root)
    path_len = torch.minimum(
        torch.randint(0, row_cap + 1, (nreq,), generator=g), row_len)

    # ---- host route
    out_rows, accepted_rows, accepted_lens, accepted_node_paths, _ = (
        dm.fr13_taw_materialize(row_buf, row_len, path_buf, path_len))
    ot_host = torch.full((nreq, cols), -1, dtype=torch.long)
    for i, row in enumerate(out_rows):
        for pos, tok in enumerate(row):
            ot_host[i, pos] = int(tok)
    atr_host = torch.tensor(accepted_rows, dtype=torch.long)
    gdn_paths_host = torch.zeros(nreq, min(cols, row_cap), dtype=torch.long)
    gdn_rows_host = torch.zeros(nreq, dtype=torch.long)
    for i, (pth, ln) in enumerate(zip(accepted_node_paths, accepted_lens)):
        gp = [int(n) + 1 for n in pth[: int(ln)]]
        gdn_paths_host[i, : len(gp)] = torch.tensor(gp, dtype=torch.long)
        gdn_rows_host[i] = gp[-1] if gp else 0

    # ---- device route (same tensors, CPU device)
    ot_dev = torch.empty(nreq, cols, dtype=torch.long)
    atr_dev = torch.empty(nreq, dtype=torch.long)
    gdn_paths_dev, gdn_rows_dev = dm.fr13_taw_products_device(
        row_buf, row_len, path_buf, path_len, ot_dev, atr_dev)

    for name, a, b in (("output_token_ids", ot_host, ot_dev),
                       ("accepted_tree_rows", atr_host, atr_dev),
                       ("gdn_paths", gdn_paths_host, gdn_paths_dev),
                       ("gdn_rows", gdn_rows_host, gdn_rows_dev)):
        if not torch.equal(a, b):
            fails += 1
            print(f"case {case} {name} MISMATCH nreq={nreq} row_cap={row_cap} "
                  f"cols={cols}\n host={a}\n dev={b}")
            break

print(f"cases=200 fails={fails}")
assert fails == 0, "BYTE GATE FAIL"
print(">>> PASS — device product route byte-identical to host committer route")
