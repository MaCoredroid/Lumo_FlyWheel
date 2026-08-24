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
# DISCRIMINATE BY PATH, NOT BASENAME -- corrected after the first live attestation.
# Our forked build carries the SAME basename as vLLM's own stock extension
# (_vllm_fa2_C.abi3.so), so a basename match flags the STOCK WHEEL and reports a false
# 'purity not established'. Stock lives inside the wheel at
# dist-packages/vllm/vllm_flash_attn/; ours is mounted from /workspace or carries a
# name only our builds use (qrow32/gqa_pair_splitk).
OUR_SO_RE='/workspace/[^ ]*[.]so|gqa_pair_splitk|_vllm_fa2_qrow32'
STOCK_SO_DIR='dist-packages/vllm/vllm_flash_attn/'

j() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

# ---- engine pid: the EngineCore worker, not the API server ------------------
EPID=$(docker exec "$C" bash -lc 'pgrep -f "EngineCore|VLLM::EngineCore" | head -1' 2>/dev/null)
[[ -z "$EPID" ]] && EPID=$(docker exec "$C" bash -lc 'pgrep -f "vllm" | tail -1' 2>/dev/null)

# ---- (1) PATCHER ABSENT -----------------------------------------------------
# The patchers inject NAMED blobs into the installed vllm. If the patcher never ran, the
# installed tree carries zero of our sentinels. This is the load-bearing check: it proves
# absence in the ENGINE'S OWN IMPORT TREE, not merely absence of a command line.
VLLM_DIR=$(docker exec "$C" python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null)

# THE ONE DECLARED EXCEPTION (Option A, pass 209). This route applies
# fr14_patch_nvfp4_lmhead.py and NOTHING else, because stock vLLM cannot load this
# port's checkpoint at all: its lm_head is quantized and Qwen3_5ForCausalLM declares
# only lm_head.weight. Purity here therefore means NO SIDE CODE ON THE DECODE PATH,
# not "no side code", and the exception is enumerated rather than waved through.
#
# THE PATTERN IS BROADENED, NOT NARROWED. It previously matched only
# _fr13_|_fr10_|FR13_FIXED32|lumo_flywheel -- which the lm_head shim does not use. Its
# markers are all _fr14_/FR14_, so the OLD check would have passed a tree carrying this
# shim without noticing, and would equally have missed any future FR14 patcher. Adding
# the FR14 family makes the check strictly stronger; the exception below then subtracts
# exactly the blob we have declared, and any other hit still fails.
SENTINEL_RE='_fr13_\|_fr10_\|_fr14_\|FR13_FIXED32\|FR14_\|lumo_flywheel'
# Exactly the tokens fr14_patch_nvfp4_lmhead.py injects, and exactly the four files it
# edits. Enumerated from the shim's own source, not guessed.
SHIM_TOKENS='FR14_LMHEAD_QUANT_ROUTE|FR14_LMHEAD_QUANT_ROUTE_REQUIRED|FR14_LMHEAD_QUANT_ROUTE_PERMISSIVE|FR14_LMHEAD_NVFP4|FR14_SCALAR_SCALE_RESHAPE|FR14_REQUIRE_NVFP4_LMHEAD|_fr14_qm|_fr14_algo'
SHIM_FILES='model_executor/models/qwen3_5.py model_executor/models/qwen3_5_mtp.py model_executor/layers/quantization/modelopt.py model_executor/layers/vocab_parallel_embedding.py'

SENTINEL_ALL=$(docker exec "$C" bash -lc "grep -rl '$SENTINEL_RE' '$VLLM_DIR' 2>/dev/null" 2>/dev/null)
SENTINEL_HITS=$(printf "%s\n" "$SENTINEL_ALL" | awk "NF{c++}END{print c+0}")
SENTINEL_FILES=$(printf '%s\n' "$SENTINEL_ALL" | head -8 | tr '\n' ';')

# A file is an ACCEPTED exception only if it is one of the shim's four targets AND every
# sentinel token in it belongs to the shim. A shim-target file carrying a _fr13_ blob is
# still a violation -- the exception is scoped to the blob, not to the filename.
#
# TOKEN EXTRACTION MUST GRAB WHOLE IDENTIFIERS, NOT THE MATCHED ALTERNATIVE. The first
# version of this loop ran `grep -o "$SENTINEL_RE"`, which returns the alternative that
# matched -- literally "FR14_" and "_fr14_" -- and then tested those truncated strings
# against the full allowlist names. Nothing ever matched, so the shim's own tokens were
# reported as foreign and a correctly-patched tree failed its own declared exception.
# Caught on the first live attestation. Now: extract the complete identifier and compare
# whole-token, exact.
#
# A __pycache__/*.pyc is bytecode compiled FROM one of these sources and carries the same
# strings; it is not independent evidence, so it is mapped back to its source module and
# judged by that source's verdict rather than counted separately.
_IDENT_RE='(_fr1[034]_[A-Za-z0-9_]*|FR1[034]_[A-Za-z0-9_]*|lumo_flywheel[A-Za-z0-9_]*)'
SENTINEL_VIOLATIONS=""
for f in $SENTINEL_ALL; do
  [[ -z "$f" ]] && continue
  _src="$f"
  case "$f" in
    */__pycache__/*.pyc)
      _base=$(basename "$f"); _base="${_base%%.*}"
      _src="$(dirname "$(dirname "$f")")/${_base}.py" ;;
  esac
  _is_target=0
  for t in $SHIM_FILES; do [[ "$_src" == *"$t" ]] && _is_target=1; done
  # BYTECODE INHERITS ITS SOURCE'S VERDICT AND IS NOT TOKEN-SCANNED. A .pyc is compiled
  # from a .py that is already judged on its own, so it carries no independent evidence --
  # and running an identifier regex over BINARY is unsound: `[A-Za-z0-9_]*` runs past the
  # real token into whatever adjacent bytes happen to be alphanumeric, which is exactly
  # what produced the spurious "foreign" tokens _fr14_qmrW, _fr14_qmrv and _fr14_algos on
  # the first corrected run. Judge the source; let the bytecode follow it.
  if [[ "$f" != "$_src" ]]; then
    [[ "$_is_target" == 1 ]] && continue
    SENTINEL_VIOLATIONS+="$f[bytecode-of-untargeted-source:$_src];"
    continue
  fi
  if [[ "$_is_target" == 1 ]]; then
    _foreign=$(docker exec "$C" bash -lc "grep -aoE '$_IDENT_RE' '$f' 2>/dev/null | sort -u | grep -Exv '$SHIM_TOKENS' | head -5" 2>/dev/null | tr '\n' ',')
    [[ -z "${_foreign//,/}" ]] && continue        # only shim tokens -> declared exception
    SENTINEL_VIOLATIONS+="$f[foreign:${_foreign}];"
  else
    SENTINEL_VIOLATIONS+="$f;"
  fi
done
# `grep -c .` PRINTS 0 AND EXITS 1 on no match, so `$(grep -c . || echo 0)` yields the
# two-line string "0\n0", which parses as neither 0 nor an int -- the JSON then recorded
# UNDECLARED_violations: null and the check FAILED with an empty violations list. Counting
# in the shell instead removes the trap entirely.
SENTINEL_VIOLATION_N=0
if [[ -n "${SENTINEL_VIOLATIONS//;/}" ]]; then
  SENTINEL_VIOLATION_N=$(awk -v s="$SENTINEL_VIOLATIONS" 'BEGIN{n=split(s,a,";"); c=0; for(i=1;i<=n;i++) if(length(a[i])>0) c++; print c}')
fi
_SHIM_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/fr14_patch_nvfp4_lmhead.py"
SHIM_SHA=$(sha256sum "$_SHIM_SRC" 2>/dev/null | cut -d' ' -f1)
[[ -n "$SHIM_SHA" ]] || SHIM_SHA="UNREADABLE:$_SHIM_SRC"
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
FA2_IN_MAPS=$(docker exec "$C" bash -lc "grep -cE '$OUR_SO_RE' /proc/$EPID/maps 2>/dev/null" 2>/dev/null)
STOCK_FA=$(docker exec "$C" bash -lc "grep -c '$STOCK_SO_DIR' /proc/$EPID/maps 2>/dev/null" 2>/dev/null)
LAYOUT=$(docker exec "$C" bash -lc '[ -d /workspace/scripts ] && echo forked_layout_workspace_mounted || echo native_layout_no_repo_mount' 2>/dev/null)
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
bad=0; checked=0; badnames=[]
for row in csv.reader(open(rec)):
    if len(row)<2 or not row[1].startswith('sha256='): continue
    p=row[0]
    if not (p.startswith('vllm/') and ('spec_decode' in p or 'attention' in p or 'model_executor' in p)): continue
    if not os.path.isfile(p): continue
    h=base64.urlsafe_b64encode(hashlib.sha256(open(p,'rb').read()).digest()).rstrip(b'=').decode()
    checked+=1
    if h!=row[1].split('=',1)[1]:
        bad+=1; badnames.append(p)
print(f'{checked}:{bad}:' + ','.join(badnames[:12]))
PY" 2>/dev/null)

python3 - "$OUT" "$EPID" "$SENTINEL_HITS" "$SENTINEL_FILES" "$PATCHER_IN_LOG" \
  "$ENG_OURSIDE_FDS" "$ENG_OURSIDE_MAPS" "$WORKSPACE_MOUNTED" "$WORKSPACE_PYCACHE" \
  "$FA2_IN_MAPS" "$BACKEND_LINE" "$ALL_SO" "$VLLM_VER" "$VLLM_RECORD_OK" "$VLLM_DIR" \
  "$STOCK_FA" "$LAYOUT" "$SENTINEL_VIOLATION_N" "$SENTINEL_VIOLATIONS" "$SHIM_SHA" <<'PY'
import json,sys
(out,epid,sent,sentf,patlog,fds,maps,wsmnt,wspyc,fa2,backend,allso,ver,rec,vdir,stockfa,layout,
 violn,violf,shimsha)=sys.argv[1:21]
def i(x):
    try: return int(x)
    except Exception: return None
checks={}
checks["1_no_undeclared_side_code"]={
 "vllm_dir":vdir,
 "DECLARED_EXCEPTION":{
   "what":"NVFP4 lm_head loader shim (fr14_patch_nvfp4_lmhead.py) -- WEIGHT LOADING ONLY",
   "shim_sha256":shimsha,
   "why_permitted":"stock vLLM cannot load this port's checkpoint at all: the checkpoint's lm_head is quantized (lm_head.input_scale/.weight_scale/.weight_scale_2) and Qwen3_5ForCausalLM declares only lm_head.weight. Ruled Option A, pass 209.",
   "why_it_does_not_contaminate":"its four gaps are constructor wiring, quant-method dispatch, key remapping and a numel-preserving reshape -- none touches the decode path, attention, the drafter or speculative decoding",
   "scoped_to_blob_not_filename":"a shim-target file carrying any NON-shim sentinel is still a violation",
   "tokens_excepted":["FR14_LMHEAD_QUANT_ROUTE","FR14_LMHEAD_QUANT_ROUTE_REQUIRED","FR14_LMHEAD_QUANT_ROUTE_PERMISSIVE","FR14_LMHEAD_NVFP4","FR14_SCALAR_SCALE_RESHAPE","FR14_REQUIRE_NVFP4_LMHEAD","_fr14_qm","_fr14_algo"],
   "files_excepted":["model_executor/models/qwen3_5.py","model_executor/models/qwen3_5_mtp.py","model_executor/layers/quantization/modelopt.py","model_executor/layers/vocab_parallel_embedding.py"]},
 "sentinel_files_total":i(sent),
 "sentinel_files":[f for f in sentf.split(";") if f],
 "UNDECLARED_violations":i(violn),
 "violating_files":[f for f in violf.split(";") if f],
 "patcher_invocations_in_boot_log":i(patlog),
 "PASS": i(violn)==0 and i(patlog)==0,
 "why":"the patchers inject NAMED blobs into the installed vllm, so scanning the engine's own import tree proves absence there rather than merely absence of a command line. The sentinel pattern was BROADENED for this route to include the _fr14_/FR14_ family -- the previous pattern (_fr13_/_fr10_/FR13_FIXED32/lumo_flywheel) did not match the lm_head shim's own markers and would have passed a tree carrying it. Broadening then excepting exactly the declared blob is strictly stronger than the old check, not weaker."}
checks["2_import_census"]={
 "engine_pid":epid,
 "ourside_open_fds":i(fds),"ourside_mappings":i(maps),
 "workspace_mounted":i(wsmnt),"recent_workspace_pycache":i(wspyc),
 "PASS": i(fds)==0 and i(maps)==0,
 "scope":"cross-process sys.modules is not readable; this censuses the ENGINE's own open files and mappings plus __pycache__ freshness, which is the strongest available evidence"}
checks["3_attention_backend"]={
 "container_layout_detected":layout,
 "stock_wheel_flash_attn_mapped":i(stockfa),
 "our_fa2_so_mapped_in_engine":i(fa2),
 "attention_so_mapped":[s for s in allso.split(";") if s],
 "backend_lines_from_boot_log":[b for b in backend.split(";") if b],
 "PASS": i(fa2)==0,
 "why":"our forked .so must not appear in /proc/<engine>/maps. Matched BY PATH: stock wheel extensions under dist-packages/vllm/vllm_flash_attn/ share our basename and are EXPECTED on a plain native run -- counting them was a false positive in the first attestation."}
_recparts=(rec.split(":")+["","",""])[:3] if ":" in rec else ("","","")
checked,bad,badnames=_recparts
# THE SHIM EDITS AT-REST FILES BY DESIGN, so this check cannot simply demand zero
# mismatches on this route -- it would fail for the one reason we have declared. Nor
# should it be relaxed to "ignore mismatches": that would blind it to a real one. It is
# therefore made SPECIFIC -- mismatching paths are enumerated and each must be one of the
# shim's four targets. Any other modified file still fails, and the count is kept
# alongside the classification so a reader can see both.
_SHIM_TARGETS=("model_executor/models/qwen3_5.py","model_executor/models/qwen3_5_mtp.py",
               "model_executor/layers/quantization/modelopt.py",
               "model_executor/layers/vocab_parallel_embedding.py")
_bad=[b for b in badnames.split(",") if b]
_undeclared=[b for b in _bad if not any(b.endswith(t) for t in _SHIM_TARGETS)]
checks["4_vllm_at_rest"]={
 "vllm_version":ver,"dist_info_RECORD_check":rec,
 "files_checked":i(checked),"files_mismatching":i(bad),
 "mismatching_paths":_bad,
 "mismatches_explained_by_declared_shim":[b for b in _bad if b not in _undeclared],
 "UNDECLARED_mismatches":_undeclared,
 "PASS": (len(_undeclared)==0) if i(bad) is not None else None,
 "note":"the declared lm_head shim modifies four installed files by design, so a mismatch in exactly those is EXPECTED on this route and is classified rather than ignored. A mismatch anywhere else still fails.",
 "SCOPE_LIMIT":"a byte-diff against pristine UPSTREAM is NOT performed: upstream is not present in the container. What IS checked is the wheel's own dist-info RECORD hashes for the executed paths (spec_decode/attention/model_executor) -- that detects any at-rest modification of those files since install, which is the actionable half. A baked-in diff that predates the wheel build would NOT be caught by this and is named here rather than glossed."}
apparatus={
 "host_side_offload_proxy":"PRESENT and shared by every arm -- the agent harness runs through it; outside the engine",
 "output_ceiling_24k":"LUMO_PROXY_MAX_OUTPUT_TOKENS, shared by every arm; outside the engine",
 "note":"both are apparatus, not engine code; disclosed so 'no our-side code in the engine' is not read as 'no our-side code anywhere'"}
res=[c.get("PASS") for c in checks.values() if c.get("PASS") is not None]
doc={"schema":"fr14.mtp5.purity.v1","checks":checks,"shared_apparatus":apparatus,
     "ALL_PASS":all(res) if res else None,
     "VERDICT":("engine purity: no side code on the decode path; ONE DECLARED EXCEPTION: "
                "NVFP4 lm_head loader shim (weight loading only), sha " + (shimsha or "?")[:16]
                if res and all(res) else
                "PURITY NOT ESTABLISHED -- see failing checks; do not report this arm as 'plain native'")}
open(out,"w").write(json.dumps(doc,indent=1,sort_keys=True)+"\n")
print(json.dumps({k:v.get("PASS") for k,v in checks.items()},indent=1))
print("ALL_PASS:",doc["ALL_PASS"])
PY
echo "[mtp5-purity] -> $OUT"
