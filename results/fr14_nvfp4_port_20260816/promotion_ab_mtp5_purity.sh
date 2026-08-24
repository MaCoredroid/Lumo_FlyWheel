#!/usr/bin/env bash
# MTP-5 PURITY ATTESTATION — prove "the plain native kernel, without any of our side
# code" rather than imply it.
#
# Mark's sharpened spec. The arm CHOOSES nativemtp5; this OBSERVES that the choice took
# effect, from inside the running container, and writes the result into the artifact.
# Everything here is read-only.
#
# WHAT IS APPARATUS AND STAYS, disclosed rather than hidden: the host-side offload proxy
# (the agent harness runs through it on alienware) and the 24k output ceiling
# (LUMO_PROXY_MAX_OUTPUT_TOKENS). Both are OUTSIDE the engine and are shared by every arm
# in the comparison, so they do not contaminate "no our-side code IN THE ENGINE" -- but a
# reader deserves to know they are there.
#
# Usage: promotion_ab_mtp5_purity.sh <container> <runroot> <boot-log>
set -uo pipefail
C=${1:?container}; RUNROOT=${2:?runroot}; BOOTLOG=${3:-}
OUT="$RUNROOT/MTP5_PURITY.json"
FA2_SONAMES='_vllm_fa2_C.abi3.so|_vllm_fa2_qrow32|gqa_pair_splitk'

j() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# ---- engine pid: the EngineCore worker, not the API server ------------------
EPID=$(docker exec "$C" bash -lc 'pgrep -f "EngineCore|VLLM::EngineCore" | head -1' 2>/dev/null)
[[ -z "$EPID" ]] && EPID=$(docker exec "$C" bash -lc 'pgrep -f "vllm" | tail -1' 2>/dev/null)

# ---- (1) PATCHER ABSENT -----------------------------------------------------
# The patchers inject NAMED blobs into the installed vllm. If the patcher never ran, the
# installed tree carries zero of our sentinels. This is the load-bearing check: it proves
# absence in the ENGINE'S OWN IMPORT TREE, not merely absence of a command line.
VLLM_DIR=$(docker exec "$C" python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null)
SENTINEL_HITS=$(docker exec "$C" bash -lc "grep -rl '_fr13_\|_fr10_\|FR13_FIXED32\|lumo_flywheel' '$VLLM_DIR' 2>/dev/null | wc -l" 2>/dev/null)
SENTINEL_FILES=$(docker exec "$C" bash -lc "grep -rl '_fr13_\|_fr10_\|FR13_FIXED32\|lumo_flywheel' '$VLLM_DIR' 2>/dev/null | head -8" 2>/dev/null | tr '\n' ';')
PATCHER_IN_LOG=0
if [[ -n "$BOOTLOG" && -f "$BOOTLOG" ]]; then
  PATCHER_IN_LOG=$(grep -cE "fr10_phase4_patch_vllm_tree_gdn|fr13_patch_fa2_tree_bias|in-container patcher" "$BOOTLOG" 2>/dev/null)
fi

# ---- (2) IMPORT CENSUS in the ENGINE process --------------------------------
# Cross-process sys.modules is not readable, so census the engine's OWN open files and
# mappings for our-side paths. A module that was imported leaves its file in the map or
# its __pycache__ in the tree; absence of all three is the strongest available evidence.
ENG_OURSIDE_FDS=$(docker exec "$C" bash -lc "ls -l /proc/$EPID/fd 2>/dev/null | grep -cE 'workspace/(scripts|src)|lumo_flywheel_serving|fr13_|fr10_'" 2>/dev/null)
ENG_OURSIDE_MAPS=$(docker exec "$C" bash -lc "grep -cE 'workspace/(scripts|src)|lumo_flywheel_serving|fr13_|fr10_' /proc/$EPID/maps 2>/dev/null" 2>/dev/null)
WORKSPACE_MOUNTED=$(docker exec "$C" bash -lc '[ -d /workspace ] && echo 1 || echo 0' 2>/dev/null)
WORKSPACE_PYCACHE=$(docker exec "$C" bash -lc 'find /workspace/src /workspace/scripts -name "__pycache__" -newermt "-6 hours" 2>/dev/null | wc -l' 2>/dev/null)

# ---- (3) ATTENTION BACKEND + our .so NOT mapped -----------------------------
FA2_IN_MAPS=$(docker exec "$C" bash -lc "grep -cE '$FA2_SONAMES' /proc/$EPID/maps 2>/dev/null" 2>/dev/null)
BACKEND_LINE=""
if [[ -n "$BOOTLOG" && -f "$BOOTLOG" ]]; then
  BACKEND_LINE=$(grep -oE "Using [A-Za-z0-9_]+ backend|attention backend[^\"]{0,60}|BACKEND=[A-Za-z0-9_]+" "$BOOTLOG" 2>/dev/null | head -3 | tr '\n' ';')
fi
ALL_SO=$(docker exec "$C" bash -lc "grep -oE '/[^ ]*\.so[^ ]*' /proc/$EPID/maps 2>/dev/null | grep -iE 'attn|flash|fa2' | sort -u | head -6" 2>/dev/null | tr '\n' ';')

# ---- (4) vLLM AT REST -------------------------------------------------------
# Honest scope: a byte-diff against pristine upstream needs that upstream present in the
# container, which it is not. What IS checkable in-container is recorded, and what is not
# is named rather than glossed.
VLLM_VER=$(docker exec "$C" python3 -c 'import vllm;print(getattr(vllm,"__version__","unknown"))' 2>/dev/null)
VLLM_DIST=$(docker exec "$C" bash -lc 'ls -d /usr/local/lib/python3*/dist-packages/vllm-*.dist-info 2>/dev/null | head -1' 2>/dev/null)
VLLM_RECORD_OK=$(docker exec "$C" bash -lc "cd \$(dirname '$VLLM_DIR') 2>/dev/null && python3 - <<'PY' 2>/dev/null
import hashlib,base64,csv,os,sys
d='$VLLM_DIST'
rec=os.path.join(d,'RECORD') if d else ''
if not rec or not os.path.isfile(rec): print('NO_RECORD'); sys.exit()
bad=0; checked=0
for row in csv.reader(open(rec)):
    if len(row)<2 or not row[1].startswith('sha256='): continue
    p=row[0]
    if not (p.startswith('vllm/') and ('spec_decode' in p or 'attention' in p or 'model_executor' in p)): continue
    if not os.path.isfile(p): continue
    h=base64.urlsafe_b64encode(hashlib.sha256(open(p,'rb').read()).digest()).rstrip(b'=').decode()
    checked+=1
    if h!=row[1].split('=',1)[1]: bad+=1
print(f'{checked}:{bad}')
PY" 2>/dev/null)

python3 - "$OUT" "$EPID" "$SENTINEL_HITS" "$SENTINEL_FILES" "$PATCHER_IN_LOG" \
  "$ENG_OURSIDE_FDS" "$ENG_OURSIDE_MAPS" "$WORKSPACE_MOUNTED" "$WORKSPACE_PYCACHE" \
  "$FA2_IN_MAPS" "$BACKEND_LINE" "$ALL_SO" "$VLLM_VER" "$VLLM_RECORD_OK" "$VLLM_DIR" <<'PY'
import json,sys
(out,epid,sent,sentf,patlog,fds,maps,wsmnt,wspyc,fa2,backend,allso,ver,rec,vdir)=sys.argv[1:16]
def i(x):
    try: return int(x)
    except Exception: return None
checks={}
checks["1_patcher_absent"]={
 "vllm_dir":vdir,
 "our_sentinels_in_installed_vllm":i(sent),
 "sentinel_files":[f for f in sentf.split(";") if f],
 "patcher_invocations_in_boot_log":i(patlog),
 "PASS": i(sent)==0 and i(patlog)==0,
 "why":"the patchers inject NAMED blobs into the installed vllm; zero sentinels proves absence in the engine's own import tree, not merely absence of a command line"}
checks["2_import_census"]={
 "engine_pid":epid,
 "ourside_open_fds":i(fds),"ourside_mappings":i(maps),
 "workspace_mounted":i(wsmnt),"recent_workspace_pycache":i(wspyc),
 "PASS": i(fds)==0 and i(maps)==0,
 "scope":"cross-process sys.modules is not readable; this censuses the ENGINE's own open files and mappings plus __pycache__ freshness, which is the strongest available evidence"}
checks["3_attention_backend"]={
 "our_fa2_so_mapped_in_engine":i(fa2),
 "attention_so_mapped":[s for s in allso.split(";") if s],
 "backend_lines_from_boot_log":[b for b in backend.split(";") if b],
 "PASS": i(fa2)==0,
 "why":"our forked _vllm_fa2 .so must not appear in /proc/<engine>/maps"}
checked,bad=(rec.split(":")+["",""])[:2] if ":" in rec else ("","")
checks["4_vllm_at_rest"]={
 "vllm_version":ver,"dist_info_RECORD_check":rec,
 "files_checked":i(checked),"files_mismatching":i(bad),
 "PASS": (i(bad)==0) if i(bad) is not None else None,
 "SCOPE_LIMIT":"a byte-diff against pristine UPSTREAM is NOT performed: upstream is not present in the container. What IS checked is the wheel's own dist-info RECORD hashes for the executed paths (spec_decode/attention/model_executor) -- that detects any at-rest modification of those files since install, which is the actionable half. A baked-in diff that predates the wheel build would NOT be caught by this and is named here rather than glossed."}
apparatus={
 "host_side_offload_proxy":"PRESENT and shared by every arm -- the agent harness runs through it; outside the engine",
 "output_ceiling_24k":"LUMO_PROXY_MAX_OUTPUT_TOKENS, shared by every arm; outside the engine",
 "note":"both are apparatus, not engine code; disclosed so 'no our-side code in the engine' is not read as 'no our-side code anywhere'"}
res=[c.get("PASS") for c in checks.values() if c.get("PASS") is not None]
doc={"schema":"fr14.mtp5.purity.v1","checks":checks,"shared_apparatus":apparatus,
     "ALL_PASS":all(res) if res else None,
     "VERDICT":("engine purity OBSERVED: no our-side code in the engine"
                if res and all(res) else
                "PURITY NOT ESTABLISHED -- see failing checks; do not report this arm as 'plain native'")}
open(out,"w").write(json.dumps(doc,indent=1,sort_keys=True)+"\n")
print(json.dumps({k:v.get("PASS") for k,v in checks.items()},indent=1))
print("ALL_PASS:",doc["ALL_PASS"])
PY
echo "[mtp5-purity] -> $OUT"
