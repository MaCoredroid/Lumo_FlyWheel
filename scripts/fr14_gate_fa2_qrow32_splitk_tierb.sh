#!/usr/bin/env bash
# Earn the FR14 split-K Tier-B qualification credential.
#
# Offline kernel work: no serve, no model, no drafter. Two independent probe
# PROCESSES in the pinned image against the pinned .so, reduced into a
# credential, then re-validated through the same door a serve uses.
#
# Two processes is not belt-and-braces, it is the measurement. In-process
# repeats prove the combine does not depend on allocator state within a run;
# only a second process, with a different allocator history and different
# device addresses for the split accumulators, tests the claim that matters --
# that the same inputs give the same bits, full stop. The reducer compares the
# per-case digests key by key rather than trusting two files that both say
# "passed".
#
# GPU discipline: refuses to start if any container exists, names its own
# containers, removes them with --rm, and bounds each run with a timeout.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

case "${FR14_GATE_SPLITK_TIERB:-0}" in
  1) ;;
  0) echo "split-K tier-b gate is disabled; set FR14_GATE_SPLITK_TIERB=1" >&2; exit 2 ;;
  *) echo "FR14_GATE_SPLITK_TIERB must be exactly 0 or 1" >&2; exit 2 ;;
esac

IMAGE='vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776'
SPLITK_SO=${SPLITK_SO:-/home/mark/fr14_splitk_build_20260818/_vllm_fa2_qrow32_gqa_pair_splitk_b1_sm121a.abi3.so}
BOUNDS="$REPO/results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_bounds.json"
PROBE="$REPO/results/fr14_nvfp4_port_20260816/fr14_splitk_fa2_probe.py"
REDUCER="$REPO/scripts/fr14_reduce_splitk_tierb_credential.py"
OUT=${OUT:-/home/mark/shared/tmp-scratch/fr14_splitk_tierb}
CREDENTIAL=${CREDENTIAL:-$REPO/results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json}
PYTHON_BIN=${PYTHON_BIN:-$REPO/.venv/bin/python}

# The probe arguments ARE part of the contract: the pre-registered strength
# floor names four context lengths spanning 20480..40960, five seeds, the
# captured operand scale, eight determinism repeats and an exact-reference
# sweep. Passing anything weaker produces a credential the validator refuses,
# which is the intended relationship between the two.
SEQ_LENS=${SEQ_LENS:-20480,23000,32768,40960}
SEEDS=${SEEDS:-5}
DET_REPS=${DET_REPS:-8}
EXACT_SEQ_LENS=${EXACT_SEQ_LENS:-20480,40960}
EXACT_SEEDS=${EXACT_SEEDS:-3}

# ------------------------------------------------------------- preflight
[[ -x "$PYTHON_BIN" ]] || { echo "python unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$BOUNDS" "$PROBE" "$REDUCER"; do
  [[ -f "$required" && ! -L "$required" ]] \
    || { echo "gate input missing or unsafe: $required" >&2; exit 2; }
done
unset required
[[ -f "$SPLITK_SO" && ! -L "$SPLITK_SO" ]] \
  || { echo "staged split-K .so missing or symlinked: $SPLITK_SO" >&2; exit 2; }
SO_SHA=$(sha256sum "$SPLITK_SO" | cut -d' ' -f1)
PINNED_SHA=$("$PYTHON_BIN" - <<'PY'
import sys
sys.path.insert(0, "scripts")
import fr13_fixed32_contract as contract
print(contract.QROW32_B1_SPLITK_FA2_SHA256)
PY
)
[[ "$SO_SHA" == "$PINNED_SHA" ]] || {
  echo "staged .so is not the pinned split-K binary" >&2
  echo "  staged: $SO_SHA" >&2
  echo "  pinned: $PINNED_SHA" >&2
  exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "pinned image absent: $IMAGE" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the gate run" >&2; exit 2; }

# HEAD binding. A credential names the commit it was earned at; a dirty tree
# means the patcher on disk is not the patcher any commit describes.
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum "$REPO/scripts/fr13_patch_fa2_tree_bias.py" | cut -d' ' -f1)
if [[ -n "$(git status --porcelain=v1 --untracked-files=no -- scripts/fr13_patch_fa2_tree_bias.py scripts/fr13_qrow32_b1_pass_sidecar.py scripts/fr13_fixed32_contract.py results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_bounds.json)" ]]; then
  echo "the files the credential binds to are dirty; commit them first" >&2
  exit 2
fi

mkdir -p "$OUT"
echo "== two independent probe processes =="
for TAG in p0 p1; do
  timeout "${GATE_TIMEOUT:-3000}" docker run --rm \
    --name "fr14_splitk_tierb_$TAG" --gpus all --network none \
    -v "$SPLITK_SO:/so/splitk.so:ro" -v "$PROBE:/probe.py:ro" -v "$OUT:/out" \
    --entrypoint /bin/bash "$IMAGE" -lc \
    "python3 /probe.py --so /so/splitk.so --json /out/probe_$TAG.json \
       --process-tag $TAG --seq-lens $SEQ_LENS --seeds $SEEDS \
       --determinism-reps $DET_REPS --scales captured,legacy0p1 \
       --exact-seq-lens $EXACT_SEQ_LENS --exact-seeds $EXACT_SEEDS" \
    || { echo "probe process $TAG failed" >&2; exit 3; }
done
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "gate left a container behind" >&2; exit 3; }

echo "== reduce into the Tier-B credential, then re-validate it =="
"$PYTHON_BIN" "$REDUCER" \
  --probe "$OUT/probe_p0.json" --probe "$OUT/probe_p1.json" \
  --bounds "$BOUNDS" \
  --source-commit "$SOURCE_COMMIT" \
  --patch-source-sha256 "$PATCH_SOURCE_SHA256" \
  --out "$CREDENTIAL"

echo
echo "credential: $CREDENTIAL"
sha256sum "$CREDENTIAL"
echo
echo "SPLIT-K IS THE PRODUCTION DEFAULT (Mark, FR14 pass 100). Under"
echo "hydra27_fixed32 a plain launch arms it by itself from the launcher"
echo "literals; nothing needs to be set. This credential is what that default"
echo "validates, so staging it here IS the arming step."
echo
echo "To arm it EXPLICITLY (for an A/B, or outside the default's mode gate):"
echo "  FR13_FA2_QROW32_B1_TIER_B_ARM=gqa_pair_splitk"
echo "  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_HOST=<host path to the file above>"
echo "  FR13_FA2_QROW32_B1_TIER_B_CREDENTIAL_SHA256=<sha above>"
echo "The container path is DERIVED by the launcher, never supplied."
echo
echo "It is still a TIER-B credential: the arm serves under it, and the"
echo "byte-exact Tier-A production allowlist still refuses split-K. The exact16"
echo "QC now runs AFTER promotion as verification. The degenerate eyeball on"
echo "served generations was discharged on the round-12 traces; it is not"
echo "discharged by re-running this gate."
