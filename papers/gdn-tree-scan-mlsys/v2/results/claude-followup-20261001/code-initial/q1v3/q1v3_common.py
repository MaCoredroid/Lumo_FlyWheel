#!/usr/bin/env python3
"""q1v3 shared helpers: topology, tensor/object I/O and paged-state access.

Everything here is pure (no vLLM import) so it is CPU-testable. The in-engine
hooks call exactly these functions on the live device tensors.

Object format: one raw little-endian file per tensor (bf16 stored as its int16
bit pattern, fp32 raw) plus a JSON descriptor {file, dtype, shape, sha256, bytes}.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

try:  # the reducer/tests import this on the host; torch is optional for pure helpers
    import torch
except Exception:  # pragma: no cover
    torch = None

# ----------------------------------------------------------------------------- topology
# hydra27_fixed32 (scripts/fr13_fixed32_topology.py, sha c02703b4...): physical row r = draft node r-1, row 0 = root.
PHYSICAL_PARENT = (-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9, 9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29, 30)
INACTIVE_ROWS = (18, 23, 25, 27)
N_ROWS, N_DRAFTS, MAX_PATH, MAX_DEPTH = 32, 31, 16, 11
SPINE = (0, 1, 4, 9, 14, 19, 24, 26, 28, 29, 30, 31)
TERMINAL_TOKEN = 248046          # <|im_end|>, generation eos
SPECIAL_TOKEN_FLOOR = 248044     # ids >= this are special/control tokens; never used as forced content
SERVED_MODEL = "qwen3.8-27b-nvfp4-radixark"
GDN_SUFFIX, ATTN_SUFFIX = ".linear_attn", ".self_attn.attn"
N_GDN, N_ATTN = 48, 16
LIVE_TAPS = 3                    # causal conv kernel 4 -> 3 history taps (native conv row is [3, D])


def depth(row):
    d = 0
    while row != 0:
        row = PHYSICAL_PARENT[row]; d += 1
    return d


def path_to(row):
    """Root-inclusive physical-row path ending at `row`."""
    if row in INACTIVE_ROWS:
        raise ValueError(f"row {row} is an inactive padding row")
    out = [row]
    while out[-1] != 0:
        out.append(PHYSICAL_PARENT[out[-1]])
    return out[::-1]


def children(row):
    return [r for r in range(1, N_ROWS) if PHYSICAL_PARENT[r] == row and r not in INACTIVE_ROWS]


def validate_path(nodes):
    nodes = [int(n) for n in nodes]
    if not nodes or nodes[0] != 0:
        raise ValueError("path must start at root row 0")
    if any(n in INACTIVE_ROWS or not 0 <= n < N_ROWS for n in nodes):
        raise ValueError(f"path {nodes} uses an inactive/out-of-range row")
    if any(PHYSICAL_PARENT[nodes[i + 1]] != nodes[i] for i in range(len(nodes) - 1)):
        raise ValueError(f"path {nodes} is not a parent chain")
    if len(nodes) - 1 > MAX_DEPTH:
        raise ValueError("path deeper than 11")
    return nodes


# ----------------------------------------------------------------------------- generic io
def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b):
    return hashlib.sha256(bytes(b)).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path, obj, exclusive=False):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str, allow_nan=False)
    if exclusive and os.path.exists(path):
        os.remove(tmp)
        raise FileExistsError(path)
    os.replace(tmp, path)
    return path


def record_digest(doc):
    """Stable digest of a JSON record (excluding its own digest field)."""
    body = {k: v for k, v in doc.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- tensor objects
_DT = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int64": 8, "torch.int32": 4, "torch.float16": 2}


def tensor_to_bytes(t):
    """Exact bytes of a tensor (device -> host copy, bf16 as int16 bit pattern)."""
    c = t.detach().contiguous()
    if c.device.type != "cpu":
        c = c.cpu()
    if c.dtype == torch.bfloat16:
        c = c.view(torch.int16)
    return c.numpy().tobytes()


def bytes_to_tensor(b, dtype, shape):
    dt = {"torch.bfloat16": torch.int16, "torch.float32": torch.float32, "torch.int64": torch.int64,
          "torch.int32": torch.int32, "torch.float16": torch.float16}[dtype]
    t = torch.frombuffer(bytearray(b), dtype=dt).reshape(shape)
    return t.view(torch.bfloat16) if dtype == "torch.bfloat16" else t


def save_tensor(root, rel, t):
    """Write one tensor object; returns its descriptor. `rel` is relative to `root`."""
    b = tensor_to_bytes(t)
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)
    return {"file": rel, "dtype": str(t.dtype), "shape": [int(x) for x in t.shape], "bytes": len(b), "sha256": sha256_bytes(b)}


def load_tensor(root, desc, verify=True):
    path = os.path.join(root, desc["file"])
    with open(path, "rb") as f:
        b = f.read()
    n = 1
    for d in desc["shape"]:
        n *= int(d)
    if len(b) != n * _DT[desc["dtype"]] or len(b) != int(desc["bytes"]):
        raise RuntimeError(f"object {desc['file']}: length {len(b)} != expected {desc['bytes']}")
    if verify and sha256_bytes(b) != desc["sha256"]:
        raise RuntimeError(f"object {desc['file']}: sha256 mismatch")
    return bytes_to_tensor(b, desc["dtype"], desc["shape"])


# ----------------------------------------------------------------------------- paged state access
def kv_index(block_row, page, start, n, device=None):
    """(pages, offsets) int64 tensors addressing logical positions [start, start+n) through a block-table row.
    Page = kernel block size of the attention group (native 64, candidate 1024)."""
    if n <= 0:
        raise ValueError("empty KV range")
    last = start + n - 1
    if last // page >= len(block_row):
        raise RuntimeError(f"block table row ({len(block_row)} blocks of {page}) does not cover position {last}")
    pos = torch.arange(start, start + n, dtype=torch.int64)
    blocks = torch.as_tensor([int(b) for b in block_row], dtype=torch.int64)
    pages = blocks[pos // page]
    if bool((pages <= 0).any()):
        raise RuntimeError("block table maps a requested position to the null block")
    offs = pos % page
    if device is not None:
        pages, offs = pages.to(device), offs.to(device)
    return pages, offs


def check_kv_layout(kvc, page):
    if kvc.dim() != 5 or int(kvc.shape[0]) != 2 or int(kvc.shape[2]) != int(page):
        raise RuntimeError(f"KV cache {tuple(kvc.shape)} is not (2, blocks, {page}, heads, dim)")


def read_kv(kvc, block_row, page, start, n):
    """-> [2, n, H, D] copy (K plane then V plane) of logical positions [start, start+n)."""
    check_kv_layout(kvc, page)
    pages, offs = kv_index(block_row, page, start, n, device=kvc.device)
    if bool((pages >= int(kvc.shape[1])).any()):
        raise RuntimeError("block id out of range of the KV tensor")
    return torch.stack([kvc[0][pages, offs], kvc[1][pages, offs]], dim=0)


def write_kv(kvc, block_row, page, start, src):
    """Write a [2, n, H, D] tensor into logical positions [start, start+n) (in-place index_put; no reallocation)."""
    check_kv_layout(kvc, page)
    n = int(src.shape[1])
    if list(src.shape[2:]) != list(kvc.shape[3:]) or src.dtype != kvc.dtype:
        raise RuntimeError(f"KV source {tuple(src.shape)}/{src.dtype} != cache geometry {tuple(kvc.shape)}/{kvc.dtype}")
    pages, offs = kv_index(block_row, page, start, n, device=kvc.device)
    s = src.to(kvc.device)
    kvc[0].index_put_((pages, offs), s[0])
    kvc[1].index_put_((pages, offs), s[1])


def read_gdn(conv_state, ssm_state, row, taps=LIVE_TAPS):
    """-> (conv[taps, D] copy, ssm[...] copy) of one state row."""
    if not (0 < int(row) < int(conv_state.shape[0])) or int(row) >= int(ssm_state.shape[0]):
        raise RuntimeError(f"GDN state row {row} out of range")
    if conv_state.dim() != 3 or int(conv_state.shape[1]) < taps:
        raise RuntimeError(f"conv state {tuple(conv_state.shape)} is not (rows, taps>={taps}, D)")
    return conv_state[row, :taps].clone(), ssm_state[row].clone()


def write_gdn(conv_state, ssm_state, row, conv, ssm, taps=LIVE_TAPS):
    if list(conv.shape) != [taps, int(conv_state.shape[2])] or conv.dtype != conv_state.dtype:
        raise RuntimeError(f"conv source {tuple(conv.shape)}/{conv.dtype} != destination ({taps},{conv_state.shape[2]})/{conv_state.dtype}")
    if list(ssm.shape) != list(ssm_state.shape[1:]) or ssm.dtype != ssm_state.dtype:
        raise RuntimeError(f"ssm source {tuple(ssm.shape)}/{ssm.dtype} != destination {tuple(ssm_state.shape[1:])}/{ssm_state.dtype}")
    conv_state[row, :taps].copy_(conv.to(conv_state.device))
    ssm_state[row].copy_(ssm.to(ssm_state.device))


def gdn_kv(layer):
    kv = layer.kv_cache
    if isinstance(kv, (list, tuple)) and len(kv) and isinstance(kv[0], (list, tuple)):
        kv = kv[0]
    return kv[0], kv[1]


def attn_kv(layer):
    kv = layer.kv_cache
    return kv[0] if isinstance(kv, (list, tuple)) else kv


def select_layers(ctx):
    """(sorted 48 target GDN names, sorted 16 target attention names). MTP layers ('mtp.' prefix) are excluded."""
    gdn = sorted(k for k in ctx if k.endswith(GDN_SUFFIX) and not k.startswith("mtp."))
    att = sorted(k for k in ctx if k.endswith(ATTN_SUFFIX) and not k.startswith("mtp."))
    if len(gdn) != N_GDN or len(att) != N_ATTN:
        raise RuntimeError(f"target layer registry {len(gdn)}/{len(att)} != {N_GDN}/{N_ATTN}")
    return gdn, att


# ----------------------------------------------------------------------------- metrics (reducer + tests)
def rel_l2(x, ref):
    x = x.double(); ref = ref.double()
    den = float(torch.linalg.vector_norm(ref))
    num = float(torch.linalg.vector_norm(x - ref))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def max_abs(x, ref):
    return float((x.double() - ref.double()).abs().max()) if x.numel() else 0.0


def greedy(logits):
    """Smallest-id argmax of a 1-D logits vector."""
    v = logits.double()
    m = v.max()
    return int(torch.nonzero(v == m).min())


def top_margin(logits):
    v = logits.double()
    top2 = torch.topk(v, 2).values
    return float(top2[0] - top2[1])


def kl_div(ref_logits, x_logits):
    """KL(softmax(ref) || softmax(x)) in nats at temperature 1, full vocabulary, fp64."""
    lp = torch.log_softmax(ref_logits.double(), -1)
    lq = torch.log_softmax(x_logits.double(), -1)
    return float((lp.exp() * (lp - lq)).sum())


def topk_ids(logits, k):
    v = logits.double()
    # stable: sort by (-value, id)
    order = sorted(range(v.numel()), key=lambda i: (-float(v[i]), i)) if v.numel() <= 64 else None
    if order is not None:
        return order[:k]
    vals, idx = torch.topk(v, min(k + 8, v.numel()))
    pairs = sorted(zip((-vals).tolist(), idx.tolist()))
    return [i for _, i in pairs[:k]]
