#!/usr/bin/env python3
"""Build the on-disk tensor ledger for the RadixArk NVFP4 checkpoint by reading
safetensors headers only (no tensor payload is materialized).

Emits: radixark_ledger.json  (per-tensor rows + rolled-up buckets)
"""
import json, os, struct, sys, collections

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/mark/shared/tmp-scratch/nvfp4_port/radixark_ledger.json"

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


rows = []
shards = sorted(p for p in os.listdir(MODEL) if p.endswith(".safetensors"))
for sh in shards:
    hdr = read_header(os.path.join(MODEL, sh))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        shape = meta["shape"]
        nelem = 1
        for d in shape:
            nelem *= d
        nbytes = nelem * DTYPE_BYTES[dt]
        off = meta["data_offsets"]
        assert off[1] - off[0] == nbytes, f"{name}: offset span {off[1]-off[0]} != computed {nbytes}"
        rows.append({"name": name, "shard": sh, "dtype": dt, "shape": shape, "bytes": nbytes})

rows.sort(key=lambda r: r["name"])
total = sum(r["bytes"] for r in rows)


def bucket(name):
    """Classify a tensor into a floor-relevant bucket."""
    if name.startswith("mtp."):
        return "mtp"
    if "visual" in name or "vision" in name:
        return "visual"
    if name.startswith("lm_head"):
        return "lm_head"
    if "embed_tokens" in name:
        return "embed"
    return "target"


buckets = collections.OrderedDict()
for r in rows:
    b = bucket(r["name"])
    d = buckets.setdefault(b, {"bytes": 0, "tensors": 0, "dtypes": collections.Counter()})
    d["bytes"] += r["bytes"]
    d["tensors"] += 1
    d["dtypes"][r["dtype"]] += 1

# target sub-split by dtype so the NVFP4 vs FP8 vs BF16 mix is visible
target_by_dtype = collections.Counter()
target_ct = collections.Counter()
for r in rows:
    if bucket(r["name"]) == "target":
        target_by_dtype[r["dtype"]] += r["bytes"]
        target_ct[r["dtype"]] += 1

out = {
    "model_dir": MODEL,
    "shards": shards,
    "n_tensors": len(rows),
    "total_bytes": total,
    "buckets": {
        k: {"bytes": v["bytes"], "gb": v["bytes"] / 1e9, "tensors": v["tensors"],
            "dtypes": dict(v["dtypes"])}
        for k, v in buckets.items()
    },
    "target_bytes_by_dtype": {k: {"bytes": v, "gb": v / 1e9, "tensors": target_ct[k]}
                              for k, v in sorted(target_by_dtype.items())},
    "lm_head_tensors": [r for r in rows if r["name"].startswith("lm_head")],
    "embed_tensors": [r for r in rows if "embed_tokens" in r["name"]],
    "mtp_tensors": [r for r in rows if r["name"].startswith("mtp.")],
    "rows": rows,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

print(f"tensors={len(rows)}  total_bytes={total}  ({total/1e9:.4f} GB)")
print()
print(f"{'bucket':<10} {'tensors':>8} {'bytes':>15} {'GB':>10}  dtypes")
for k, v in buckets.items():
    print(f"{k:<10} {v['tensors']:>8} {v['bytes']:>15,} {v['bytes']/1e9:>10.4f}  {dict(v['dtypes'])}")
print()
print("target split by dtype:")
for k, v in sorted(target_by_dtype.items()):
    print(f"   {k:<9} {target_ct[k]:>5} tensors {v:>15,} B  {v/1e9:>9.4f} GB")
print()
print("lm_head tensors:")
for r in out["lm_head_tensors"]:
    print(f"   {r['name']:<28} {r['dtype']:<8} {str(r['shape']):<22} {r['bytes']:>13,} B")
print()
print("embed tensors:")
for r in out["embed_tensors"]:
    print(f"   {r['name']:<40} {r['dtype']:<8} {str(r['shape']):<20} {r['bytes']:>13,} B")
print()
print(f"mtp tensors ({len(out['mtp_tensors'])}):")
for r in out["mtp_tensors"]:
    print(f"   {r['name']:<52} {r['dtype']:<8} {str(r['shape']):<18} {r['bytes']:>13,} B")
print(f"   mtp total = {sum(r['bytes'] for r in out['mtp_tensors']):,} B")
