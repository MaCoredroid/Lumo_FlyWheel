#!/usr/bin/env bash
# FR13_APC_FIXED_BUFFER end-to-end lossless gate. Runs the EXACT_SEED state-diff
# harness TWICE, SERIALLY (no parallel testing): FB=0 (shipped Python-list path)
# then FB=1 (new pre-alloc copy_ buffer path). The replay is teacher-forced (a
# fixed token path), so the per-layer GDN ssm_state captures MUST be bit-identical
# iff the buffer port is lossless (the unit test already proved the drained chunk
# is bit-exact; this confirms the real CUDA serving path + env wiring + engagement).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=output/fr13_apc_fixedbuf_statediff/run_$TS; mkdir -p "$OUT"
echo "$OUT" > /home/mark/.claude/jobs/22c39bb9/tmp/fb_statediff_out.txt
echo "=== FB STATE-DIFF GATE  FB=0(list) then FB=1(buffer)  serial  -> $OUT ==="

run_one() {
  local FB=$1
  echo "--- [FB=$FB] EXACT_SEED state-diff boot @ $(date -u +%H:%M:%S) ---"
  FR13_APC_FIXED_BUFFER=$FB bash scripts/fr13_apc_exactseed_statediff.sh > "$OUT/fb${FB}.log" 2>&1
  local rc=$?
  local RD; RD=$(cat /home/mark/.claude/jobs/22c39bb9/tmp/eseed_root.txt 2>/dev/null)
  echo "  [FB=$FB] rc=$rc rundir=$RD @ $(date -u +%H:%M:%S)"
  cp "$RD/logs/on_b1024_gdn.pt" "$OUT/fb${FB}_gdn.pt" 2>/dev/null && echo "  [FB=$FB] capture copied" || echo "  [FB=$FB] NO CAPTURE (boot failed?)"
  cp "$RD/eseed_statediff.jsonl" "$OUT/fb${FB}_statediff.jsonl" 2>/dev/null || true
  # engagement canary: did the chunked chain actually PUBLISH (so the buffer path
  # was exercised)? a trivial all-fallback match would be a false pass.
  local pub; pub=$(grep -ac "ES_CHAIN_PUBLISH" "$RD"/logs/*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  echo "  [FB=$FB] ES_CHAIN_PUBLISH count=$pub (must be >0 for a real test)"
}

run_one 0
run_one 1

echo ""
echo "=== COMPARE per-layer GDN ssm_state captures (FB=0 vs FB=1) ==="
.venv/bin/python - "$OUT/fb0_gdn.pt" "$OUT/fb1_gdn.pt" <<'PY'
import torch, sys, os
pa, pb = sys.argv[1], sys.argv[2]
if not (os.path.exists(pa) and os.path.exists(pb)):
    print("  MISSING CAPTURE(S) -> cannot compare:", pa, os.path.exists(pa), pb, os.path.exists(pb))
    sys.exit(3)
a = torch.load(pa, map_location='cpu'); b = torch.load(pb, map_location='cpu')
def asdict(x):
    if isinstance(x, dict): return x
    if isinstance(x, (list, tuple)): return {i: v for i, v in enumerate(x)}
    return {'<root>': x}
da, db = asdict(a), asdict(b)
print(f"  FB0 entries={len(da)}  FB1 entries={len(db)}")
keys = sorted(set(da) & set(db), key=str)
nok = nbad = 0; gmax = 0.0
for k in keys:
    ta, tb = da[k], db[k]
    # entries may themselves be dicts/lists of tensors (per-prefix capture lists)
    sub_a = asdict(ta) if not torch.is_tensor(ta) else {'_': ta}
    sub_b = asdict(tb) if not torch.is_tensor(tb) else {'_': tb}
    for sk in sorted(set(sub_a) & set(sub_b), key=str):
        x, y = sub_a[sk], sub_b[sk]
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            continue
        if x.shape != y.shape:
            print(f"  SHAPE MISMATCH {k}/{sk}: {tuple(x.shape)} vs {tuple(y.shape)}"); nbad += 1; continue
        if torch.equal(x, y):
            nok += 1
        else:
            d = (x.float() - y.float()).abs().max().item(); gmax = max(gmax, d); nbad += 1
            if nbad <= 10:
                print(f"  DIFF {k}/{sk}: maxabs={d:.3e}")
print(f"=== bit-exact tensors={nok}  differing={nbad}  global_maxdiff={gmax:.3e}")
if nbad == 0 and nok > 0:
    print("VERDICT: ✅ FB LOSSLESS — FB=0 (list) == FB=1 (buffer) bit-for-bit on the real serving path")
else:
    print("VERDICT: ❌ FB MISMATCH — do NOT bake FR13_APC_FIXED_BUFFER; investigate")
PY
echo "=== FB STATE-DIFF GATE DONE @ $(date -u +%H:%M:%S) -> $OUT ==="
