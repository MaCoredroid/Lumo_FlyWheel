#!/usr/bin/env bash
# Q1.2b candidate GDN component verification, launcher v2.1 (= v2 + runner/reducer v2.1 after the boot-lifecycle repair): ONE GPU job = two sequential fresh container processes (A, B)
# sharing one content-addressed tensor store + reducer v2. Refuses unless GATE-Q1.2b.json carries approved:true,
# approved_run_id == RUN_ID, gate == "Q1.2b", and reviewed_hashes/reviewed_scope equal to EVERY executable dependency
# about to run (runner/reducer/launcher v2, fixtures tool, oracle, evaluators v1/v2/v2.1, bf16 ulp, topology, kernel,
# native source copy, fixture manifest, expected observations, policy v2.1, base policy v2, contract v2, verification).
# Writes LAUNCH-BINDING.json (hash-bound identities/scope/run id + snapshot names) which the reducer requires.
# --dry-run prints the exact commands without touching the GPU.
set -uo pipefail
REPO=/home/mark/lumotree-review-20260927
C=papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927
GATE=$REPO/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/GATE-Q1.2b.json
CONTRACT=$REPO/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/Q1-PAIRED-NUMERICAL-CONTRACT-v2.json
IMAGE=sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc
IMAGE_DIGEST=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
# declared pinned runtime (image); the reducer refuses to run anywhere else (host torch 2.4.1/20 threads is NOT this environment)
RT_TORCH="2.11.0+cu130"; RT_TRITON="3.6.0"; RT_GPU="NVIDIA GB10"
REPEATS=2; BLOCK=calibration; NEGATIVES=1
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
RUN_ID=${RUN_ID:-UNSET}
OUT=$REPO/$C/runs/q1.2b/$RUN_ID
TOOLS=$REPO/$C/tools
FM=$REPO/$C/fixtures/q1_2b/manifest.json; OBS=$REPO/$C/fixtures/q1_2b/expected_observations.json
POL=$REPO/$C/policy/q1_component_numerical_policy.v2.1.json; POLBASE=$REPO/$C/policy/q1_component_numerical_policy.v2.json
NATIVE_COPY=$REPO/$C/identity/native_source/fla_ops__fused_sigmoid_gating.py
sha() { sha256sum "$1" | cut -d' ' -f1; }
FM_SHA=$(sha "$FM"); OBS_SHA=$(sha "$OBS"); POL_SHA=$(sha "$POL")
# route flags identical to the Cqc10 receipt subset the runner enforces; CPU threads pinned like the fixture generation
ENVS="-e FR13_FIXED32_MODE=hydra27_fixed32 -e FR13_SUBTREE_PARALLEL=1 -e FR13_SCAN_ALIGN=0 -e FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 -e FR13_RING_EXPORT=1 -e FR13_TREE_RUNROW_INIT=1 -e FR13_FLAGS_INKERNEL=1 -e FR13_FIXED32_COMMIT_DEVICE_FILL=1 -e FR13_FIXED32_KV_REMAP16=1 -e FR13_FIXED32_COMMITTER_LAYER_BATCH=0 -e FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0 -e FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0 -e FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 -e FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1"
cmd_for() {  # $1 = A|B
  echo "docker run --rm -i --gpus all --ipc=host --name lumotree-review-q12b-$1 --cidfile $OUT/proc$1.cid -e Q12B_CONTAINER_NAME=lumotree-review-q12b-$1 $ENVS \
  -v $REPO:/workspace:ro -v $OUT:/out --entrypoint python3 $IMAGE \
  /workspace/$C/tools/q1_component_runner_v2_1.py --fixtures-root /workspace/$C/fixtures/q1_2b --manifest /workspace/$C/fixtures/q1_2b/manifest.json \
  --expected-observations /workspace/$C/fixtures/q1_2b/expected_observations.json --block $BLOCK --out /out/proc$1 --process-tag $1 --image-id $IMAGE \
  --expected-fixture-manifest-sha256 $FM_SHA --expected-observations-sha256 $OBS_SHA --policy /workspace/$C/policy/q1_component_numerical_policy.v2.1.json \
  --expected-policy-sha256 $POL_SHA --gate-snapshot /out/GATE-Q1.2b.snapshot.json --tensor-store /out/tensors --repeats $REPEATS --negatives $NEGATIVES"
}
echo "RUN_ID=$RUN_ID"; echo "OUT=$OUT"; echo "FIXTURE_MANIFEST_SHA=$FM_SHA"; echo "EXPECTED_OBS_SHA=$OBS_SHA"; echo "POLICY_V2_1_SHA=$POL_SHA"; echo "PROCESS A: $(cmd_for A)"; echo "PROCESS B: $(cmd_for B)"
if [[ $DRY == 1 ]]; then echo "(dry-run: nothing launched)"; exit 0; fi
if [[ "$RUN_ID" == "UNSET" ]]; then echo "RUN_ID must be set to the gate's approved_run_id"; exit 3; fi
if [[ -e "$OUT" ]]; then echo "output run directory already exists: $OUT"; exit 7; fi
RT_TORCH="$RT_TORCH" RT_TRITON="$RT_TRITON" RT_GPU="$RT_GPU" python3 - "$GATE" "$IMAGE" "$FM" "$OBS" "$POL" "$POLBASE" "$CONTRACT" "$NATIVE_COPY" "$TOOLS" "$REPEATS" "$BLOCK" "$NEGATIVES" "$0" "$RUN_ID" "$REPO" "$C" <<'PY'
import json, sys, hashlib, os, shutil
gate, image, fm, obs, pol, polbase, contract, native_copy, tools, repeats, block, negatives, launcher, run_id, repo, c = sys.argv[1:17]
g = json.load(open(gate))
def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()
problems = []
if g.get("gate") != "Q1.2b": problems.append("gate file is not GATE-Q1.2b")
if g.get("approved") is not True: problems.append("approved is not true")
if g.get("approved_run_id") != run_id: problems.append(f"approved_run_id {g.get('approved_run_id')} != RUN_ID {run_id}")
rh = g.get("reviewed_hashes") or {}
helpers = {"q1_component_runner_v2_1.py": sha(os.path.join(tools, "q1_component_runner_v2_1.py")), "q1_oracle.py": sha(os.path.join(tools, "q1_oracle.py")),
           "q1_2b_fixtures.py": sha(os.path.join(tools, "q1_2b_fixtures.py")), "q1_policy_evaluator.py": sha(os.path.join(tools, "q1_policy_evaluator.py")),
           "q1_policy_evaluator_v2.py": sha(os.path.join(tools, "q1_policy_evaluator_v2.py")), "q1_policy_evaluator_v2_1.py": sha(os.path.join(tools, "q1_policy_evaluator_v2_1.py")),
           "q1_bf16_ulp.py": sha(os.path.join(tools, "q1_bf16_ulp.py")),
           "scripts/fr13_fixed32_topology.py": sha(os.path.join(repo, "scripts", "fr13_fixed32_topology.py")),
           "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": sha(os.path.join(repo, "src", "lumo_flywheel_serving", "fr10_gdn_tree_kernel.py")),
           "src/lumo_flywheel_serving/fr13_tree_conv_fused.py": sha(os.path.join(repo, "src", "lumo_flywheel_serving", "fr13_tree_conv_fused.py"))}
expect = {"image_id": image, "fixture_manifest": sha(fm), "expected_observations": sha(obs), "policy": sha(pol), "policy_base_v2": sha(polbase), "contract_v2": sha(contract),
          "runner": helpers["q1_component_runner_v2_1.py"], "reducer": sha(os.path.join(tools, "q1_2b_reduce_v2_1.py")), "launcher": sha(launcher),
          "fixtures_tool": helpers["q1_2b_fixtures.py"], "oracle": helpers["q1_oracle.py"], "evaluator_v1": helpers["q1_policy_evaluator.py"], "evaluator_v2": helpers["q1_policy_evaluator_v2.py"],
          "evaluator_v2_1": helpers["q1_policy_evaluator_v2_1.py"], "bf16_ulp": helpers["q1_bf16_ulp.py"], "topology": helpers["scripts/fr13_fixed32_topology.py"],
          "kernel": helpers["src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"], "native_module": sha(native_copy),
          "verification_V2": sha(os.path.join(repo, c, "fixtures", "q1_2b", "verification_V2.json")), "tree_conv_fused": helpers["src/lumo_flywheel_serving/fr13_tree_conv_fused.py"]}
for k, v in expect.items():
    if rh.get(k) != v: problems.append(f"reviewed_hashes.{k} = {rh.get(k)} != current {v}")
sc = g.get("reviewed_scope") or {}
if str(sc.get("repeats")) != str(repeats) or sc.get("block") != block or str(sc.get("negatives")) != str(negatives) or sorted(sc.get("processes", [])) != ["A", "B"]:
    problems.append(f"reviewed_scope {sc} does not match repeats={repeats} block={block} negatives={negatives} processes=[A,B]")
# projected archive vs free disk (worst case, no dedup) -- must fit with margin
fmd = json.load(open(fm)); L = int(fmd["geometry"]["layers_per_fixture"]); nfx = sum(1 for e in fmd["fixtures"] if e["block"] == block)
state = 48 * 128 * 128 * 4; out = 48 * 128 * 2
native = L * 32 * (state + out); cand = L * (32 * out + 28 * state); neg = L * state * 3 + 2 * state
worst = 2 * nfx * (native * int(repeats) + cand * int(repeats) + neg); fixtures_copy = sum(os.path.getsize(os.path.join(repo, c, "fixtures", "q1_2b", e["path"])) for e in fmd["fixtures"] if e["block"] == block)
st = os.statvfs(os.path.dirname(os.path.dirname(os.path.join(repo, c, "runs"))))
avail = st.f_bavail * st.f_frsize
if avail < 2 * worst + fixtures_copy: problems.append(f"insufficient disk: available {avail/1e9:.1f} GB < 2x worst-case archive {2*worst/1e9:.1f} GB + fixtures {fixtures_copy/1e9:.2f} GB")
print(json.dumps({"projected_worst_case_archive_gb": round(worst / 1e9, 2), "projected_dedup_gb": round(nfx * (native + cand + neg) / 1e9, 2), "available_gb": round(avail / 1e9, 1), "fixtures_copy_gb": round(fixtures_copy / 1e9, 2)}))
if problems:
    print("GATE-Q1.2b REFUSES LAUNCH:"); [print(" -", p) for p in problems]; sys.exit(3)
expect["gate"] = sha(gate)
json.dump({"expect": expect, "helpers": helpers, "scope": {"repeats": int(repeats), "block": block, "negatives": int(negatives), "processes": ["A", "B"]}, "run_id": run_id,
           "runtime": {"torch": os.environ["RT_TORCH"], "triton": os.environ["RT_TRITON"], "gpu_name": os.environ["RT_GPU"], "image_id": image}}, open("/tmp/q12b_launch_binding_stage.json", "w"))
print("GATE-Q1.2b approved with matching reviewed hashes, scope and run id")
PY
rc=$?; [[ $rc -ne 0 ]] && exit $rc
mkdir -p "$OUT/tensors" "$OUT/fixtures"
RECEIPT_WRITTEN=0
write_receipt() {
  python3 - "$OUT" "$1" <<'PY'
import json, os, sys, hashlib, datetime
out, status = sys.argv[1], sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()
files = {}; tcount = 0; tbytes = 0
for root, dirs, fns in os.walk(out):
    for fn in fns:
        p = os.path.join(root, fn); rel = os.path.relpath(p, out)
        if rel.startswith("tensors/"):
            tcount += 1; tbytes += os.path.getsize(p); continue
        if rel == "RUN-RECEIPT.json": continue
        files[rel] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
rc = {"schema": "lumo.review-response.q1-2b-run-receipt.v2", "run": out, "status": status, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "files_recursive_excluding_tensor_store": files, "tensor_store": {"objects": tcount, "bytes": tbytes},
      "proc_exit": {t: (open(os.path.join(out, f"proc{t}_exit.txt")).read().strip() if os.path.exists(os.path.join(out, f"proc{t}_exit.txt")) else None) for t in ("A", "B")},
      "reduce_exit": open(os.path.join(out, "reduce_exit.txt")).read().strip() if os.path.exists(os.path.join(out, "reduce_exit.txt")) else None}
json.dump(rc, open(os.path.join(out, "RUN-RECEIPT.json"), "w"), indent=1); print("receipt:", status, "files:", len(files), "tensors:", tcount)
PY
}
trap 'if [[ $RECEIPT_WRITTEN -eq 0 ]]; then write_receipt "ABORTED_OR_FAILED"; fi' EXIT
if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_contention_before.txt" 2> "$OUT/gpu_contention_before.err"; then echo "nvidia-smi contention check FAILED to run"; exit 4; fi
if [[ $(wc -l < "$OUT/gpu_contention_before.txt") -gt 1 ]]; then echo "GPU busy:"; cat "$OUT/gpu_contention_before.txt"; exit 4; fi
if ! docker ps --format '{{.Names}}' > "$OUT/docker_ps_before.txt" 2> "$OUT/docker_ps_before.err"; then echo "docker ps contention check FAILED to run"; exit 4; fi
if grep -q '^lumotree-review-q12' "$OUT/docker_ps_before.txt"; then echo "a q12 container is already running"; exit 4; fi
docker image inspect "$IMAGE" > "$OUT/image_inspect.json" || { echo "image inspect failed"; exit 5; }
grep -q "$IMAGE_DIGEST" "$OUT/image_inspect.json" || { echo "image digest mismatch"; exit 5; }
cp "$GATE" "$OUT/GATE-Q1.2b.snapshot.json"; cp "$FM" "$OUT/fixture_manifest.snapshot.json"; cp "$OBS" "$OUT/expected_observations.snapshot.json"; cp "$POL" "$OUT/policy.snapshot.json"; cp "$POLBASE" "$OUT/policy_base_v2.snapshot.json"; cp "$CONTRACT" "$OUT/contract_v2.snapshot.json"
cp "$REPO/$C/fixtures/q1_2b/verification_V2.json" "$OUT/" 2>/dev/null || true
python3 - "$OUT" "$REPO/$C/fixtures/q1_2b" "$BLOCK" <<'PY'
import json, os, shutil, sys, hashlib, datetime
out, fxroot, block = sys.argv[1:4]
st = json.load(open("/tmp/q12b_launch_binding_stage.json"))
fm = json.load(open(os.path.join(fxroot, "manifest.json")))
for e in fm["fixtures"]:
    if e["block"] != block: continue
    src = os.path.join(fxroot, e["path"]); dst = os.path.join(out, "fixtures", e["path"]); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
    if hashlib.sha256(open(dst, "rb").read()).hexdigest() != e["sha256"]: raise SystemExit(f"fixture copy hash mismatch {e['path']}")
st.update({"schema": "lumo.review-response.q1-2b-launch-binding.v2", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "snapshots": {"gate": "GATE-Q1.2b.snapshot.json", "fixture_manifest": "fixture_manifest.snapshot.json", "expected_observations": "expected_observations.snapshot.json",
                         "policy": "policy.snapshot.json", "policy_base_v2": "policy_base_v2.snapshot.json", "contract_v2": "contract_v2.snapshot.json"}})
json.dump(st, open(os.path.join(out, "LAUNCH-BINDING.json"), "x"), indent=1)
os.remove("/tmp/q12b_launch_binding_stage.json")
PY
[[ $? -ne 0 ]] && { echo "launch binding failed"; exit 5; }
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu,memory.used --format=csv > "$OUT/gpu_state_before.txt"
for P in A B; do
  echo "$(cmd_for $P)" > "$OUT/command_proc$P.txt"
  date -u +%FT%TZ > "$OUT/proc${P}_started_utc.txt"
  eval "$(cmd_for $P)" > "$OUT/proc$P.log" 2>&1
  PROC_RC=$?
  echo "exit=$PROC_RC" > "$OUT/proc${P}_exit.txt"
  date -u +%FT%TZ > "$OUT/proc${P}_ended_utc.txt"
  tail -n 5 "$OUT/proc$P.log"
  if [[ $PROC_RC -ne 0 ]]; then echo "process $P FAILED rc=$PROC_RC; stopping (evidence retained in $OUT)"; RECEIPT_WRITTEN=1; write_receipt "PROCESS_${P}_FAILED_rc=$PROC_RC"; exit $PROC_RC; fi
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/gpu_contention_after.txt" 2>&1
# REDUCTION RUNS INSIDE THE SAME IMMUTABLE IMAGE, CPU-ONLY (no --gpus), single-threaded: the exact C2/metric recomputation must share the
# runtime that produced the fixtures and the GPU-run metrics (host torch 2.4.1 / 20 threads reproduces the Q1.2a host-vs-image mismatch).
REDUCE_CMD="docker run --rm -i --name lumotree-review-q12b-reduce --cidfile $OUT/reduce.cid -e Q12B_CONTAINER_NAME=lumotree-review-q12b-reduce -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -v $REPO:/workspace:ro -v $OUT:/runs/$RUN_ID --entrypoint python3 $IMAGE /workspace/$C/tools/q1_2b_reduce_v2_1.py --run /runs/$RUN_ID --block $BLOCK --recompute all"
echo "$REDUCE_CMD" > "$OUT/command_reduce.txt"
date -u +%FT%TZ > "$OUT/reduce_started_utc.txt"
eval "$REDUCE_CMD" > "$OUT/reduce.log" 2>&1
REDUCE_RC=$?
date -u +%FT%TZ > "$OUT/reduce_ended_utc.txt"
echo "exit=$REDUCE_RC" > "$OUT/reduce_exit.txt"; cat "$OUT/reduce.log"
RECEIPT_WRITTEN=1; write_receipt "COMPLETED_reduce_rc=$REDUCE_RC"; exit $REDUCE_RC
