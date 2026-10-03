#!/usr/bin/env bash
# Q1.2b HELD-OUT reduction repair v2.2.1: CPU-ONLY re-reduction of the PRESERVED run q12b-heldout-20260928T013539Z in the pinned image (no GPU,
# OMP/MKL=1, CUDA hidden).  The original run is mounted READ-ONLY; every output goes to a NEW reduction directory.  Refuses unless the SEPARATE
# parent CPU-reduction authorization (AUTH-Q1.2b-HELDOUT-REDUCTION.json or $REDUCTION_AUTH) is approved, names the original run id, the original
# evidence hashes, THIS reducer's bytes and REDUCTION_ID, and the preserved evidence on disk still hashes as bound.  --dry-run prints only.
set -uo pipefail
REPO=/home/mark/lumotree-review-20260927
C=papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927
ORIG_RUN_ID=q12b-heldout-20260928T013539Z
RUN=$REPO/$C/runs/q1.2b-heldout/$ORIG_RUN_ID
AUTH=${REDUCTION_AUTH:-$REPO/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/AUTH-Q1.2b-HELDOUT-REDUCTION.json}
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc
IMAGE_DIGEST=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
REDUCER=$REPO/$C/tools/q1_2b_reduce_v2_2_1.py
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
REDUCTION_ID=${REDUCTION_ID:-UNSET}
OUT=$REPO/$C/runs/q1.2b-heldout-reduction/$REDUCTION_ID
sha() { sha256sum "$1" | cut -d' ' -f1; }
REDUCER_SHA=$(sha "$REDUCER")
CMD="docker run --rm -i --name lumotree-review-q12b-reduce-repair --cidfile $OUT/reduce.cid -e Q12B_CONTAINER_NAME=lumotree-review-q12b-reduce-repair -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -v $REPO:/workspace:ro -v $RUN:/runs/$ORIG_RUN_ID:ro -v $OUT:/reductions/$REDUCTION_ID --entrypoint python3 $IMAGE \
  /workspace/$C/tools/q1_2b_reduce_v2_2_1.py --run /runs/$ORIG_RUN_ID --block evaluation --recompute all --out /reductions/$REDUCTION_ID --authorization /reductions/$REDUCTION_ID/AUTHORIZATION.snapshot.json"
echo "REDUCTION_ID=$REDUCTION_ID"; echo "ORIGINAL_RUN=$RUN (read-only)"; echo "OUT=$OUT"; echo "REDUCER_V2_2_1_SHA=$REDUCER_SHA"; echo "AUTH=$AUTH"; echo "REDUCE: $CMD"
if [[ $DRY == 1 ]]; then echo "(dry-run: nothing launched)"; exit 0; fi
if [[ "$REDUCTION_ID" == "UNSET" ]]; then echo "REDUCTION_ID must be set to the authorization's reduction_id"; exit 3; fi
if [[ -e "$OUT" ]]; then echo "reduction directory already exists: $OUT"; exit 7; fi
python3 - "$AUTH" "$RUN" "$REDUCER" "$REDUCTION_ID" "$ORIG_RUN_ID" <<'PY'
import json, sys, hashlib, os, importlib.util
auth_p, run, reducer, reduction_id, orig_id = sys.argv[1:6]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
spec = importlib.util.spec_from_file_location("rd221", reducer); rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
problems = []
if not os.path.exists(auth_p): problems.append(f"authorization file missing: {auth_p}")
else:
    a = json.load(open(auth_p))
    if a.get("approved") is not True or a.get("authorizes") != "CPU_REDUCTION_ONLY": problems.append("authorization not approved / not CPU_REDUCTION_ONLY")
    if a.get("run_id") != orig_id: problems.append(f"authorization run_id {a.get('run_id')} != {orig_id}")
    if a.get("reduction_id") != reduction_id: problems.append(f"authorization reduction_id {a.get('reduction_id')} != REDUCTION_ID {reduction_id}")
    if a.get("repaired_reducer_sha256") != sha(reducer): problems.append(f"authorization reducer sha {a.get('repaired_reducer_sha256')} != current reducer {sha(reducer)}")
    ev = a.get("original_evidence") or {}
    for key in ("run_receipt_sha256", "original_summary_sha256", "launch_binding_sha256", "gate_snapshot_sha256", "procA_result_sha256", "procB_result_sha256"):
        if ev.get(key) != rd.ORIGINAL_RUN[key]: problems.append(f"authorization original_evidence.{key} != reducer-bound original")
pr, facts = rd.bind_original_run(run, "evaluation", rd.ORIGINAL_RUN, "/nonexistent-preflight-out", auth_p if os.path.exists(auth_p) else None)
problems += [p for p in pr if "reduction output directory" not in p and "reduction_id" not in p]   # output dir is created below; reduction_id checked above
if problems:
    print("REDUCTION REFUSED:"); [print(" -", p) for p in problems]; sys.exit(3)
print("authorization + preserved evidence bound:", json.dumps({k: (v[:12] if isinstance(v, str) and len(v) == 64 else v) for k, v in facts.items() if k != "authorization"}))
PY
rc=$?; [[ $rc -ne 0 ]] && exit $rc
if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > /tmp/q12b_reduce_repair_gpu.txt 2>&1; then echo "nvidia-smi check FAILED to run"; exit 4; fi
if docker ps --format '{{.Names}}' | grep -q '^lumotree-review-q12'; then echo "a q12 container is already running"; exit 4; fi
docker image inspect "$IMAGE" > /tmp/q12b_reduce_repair_image.json || { echo "image inspect failed"; exit 5; }
grep -q "$IMAGE_DIGEST" /tmp/q12b_reduce_repair_image.json || { echo "image digest mismatch"; exit 5; }
mkdir -p "$OUT"
mv /tmp/q12b_reduce_repair_gpu.txt "$OUT/gpu_contention_before.txt"; mv /tmp/q12b_reduce_repair_image.json "$OUT/image_inspect.json"
cp "$AUTH" "$OUT/AUTHORIZATION.snapshot.json"
python3 - "$OUT" "$AUTH" "$RUN" "$REDUCER" "$REDUCTION_ID" "$IMAGE" "$0" <<'PY'
import json, sys, hashlib, os, datetime
out, auth, run, reducer, rid, image, launcher = sys.argv[1:8]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
b = {"schema": "lumo.review-response.q1-2b-reduction-binding.v2.2.1", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "reduction_id": rid, "original_run": run, "original_run_mounted": "read-only",
     "authorization": {"source": auth, "sha256": sha(auth), "snapshot": "AUTHORIZATION.snapshot.json"}, "reducer": {"path": reducer, "sha256": sha(reducer)}, "launcher_sha256": sha(launcher), "image_id": image,
     "original_evidence": {rel: sha(os.path.join(run, rel)) for rel in ("RUN-RECEIPT.json", "summary.v2.json", "LAUNCH-BINDING.json", "GATE-Q1.2b.snapshot.json", "procA/result.json", "procB/result.json", "procA/raw_inventory.jsonl", "procB/raw_inventory.jsonl")},
     "runtime": {"torch": "2.11.0+cu130", "cpu_only": True, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, "claim_boundary": "re-reduction of preserved evidence; the original summary/receipt are not modified; the original reducer did NOT pass"}
json.dump(b, open(os.path.join(out, "REDUCTION-BINDING.json"), "x"), indent=1)
PY
[[ $? -ne 0 ]] && { echo "reduction binding failed"; exit 5; }
write_receipt() {
  python3 - "$OUT" "$1" <<'PY'
import json, os, sys, hashlib, datetime
out, status = sys.argv[1], sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
files = {}
for root, dirs, fns in os.walk(out):
    for fn in fns:
        p = os.path.join(root, fn); rel = os.path.relpath(p, out)
        if rel == "REDUCTION-RECEIPT.json": continue
        files[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
rc = {"schema": "lumo.review-response.q1-2b-reduction-receipt.v2.2.1", "reduction_dir": out, "status": status, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "files": files,
      "reduce_exit": open(os.path.join(out, "reduce_exit.txt")).read().strip() if os.path.exists(os.path.join(out, "reduce_exit.txt")) else None}
json.dump(rc, open(os.path.join(out, "REDUCTION-RECEIPT.json"), "w"), indent=1); print("receipt:", status, "files:", len(files))
PY
}
RECEIPT_WRITTEN=0
trap 'if [[ $RECEIPT_WRITTEN -eq 0 ]]; then write_receipt "ABORTED_OR_FAILED"; fi' EXIT
echo "$CMD" > "$OUT/command_reduce.txt"
date -u +%FT%TZ > "$OUT/reduce_started_utc.txt"
eval "$CMD" > "$OUT/reduce.log" 2>&1
REDUCE_RC=$?
date -u +%FT%TZ > "$OUT/reduce_ended_utc.txt"
echo "exit=$REDUCE_RC" > "$OUT/reduce_exit.txt"; tail -n 3 "$OUT/reduce.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_contention_after.txt" 2>&1
RECEIPT_WRITTEN=1; write_receipt "COMPLETED_reduce_rc=$REDUCE_RC"; exit $REDUCE_RC
