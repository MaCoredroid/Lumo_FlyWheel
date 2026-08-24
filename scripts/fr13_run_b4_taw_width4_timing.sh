#!/usr/bin/env bash
# WIDTH-4 real SWE-Verified B4 timing pair: the exact reference TAW commit vs the
# byte-qualified native-precompute PRODUCTION commit, measured AT THE WIDTH-4
# OPERATING POINT.
#
# WHY THIS RUNNER EXISTS (the whole point -- read this before touching anything)
# ----------------------------------------------------------------------------
# The width-4 Nsight attribution (results/fr13_b4_width4_nsys_20260813/) ranked
# three levers and then listed a fourth it could not rank, section 3 item 4:
#
#     CFWD small-kernel consolidation *(new, unranked)* -- 37.1 ms/step of
#     elementwise + other across 321k tiny instances per step. Not yet a lever
#     because no candidate exists, but it is the third-largest addressable pile
#     and nothing has ever been aimed at it.
#
# The shape-pinned batched TAW committer IS that candidate. Its mechanism is
# collapsing the commit walk from twelve per-level launches to one
# (exact_commit_launches 12 -> 1, walk_levels 12 -> 1 in the pinned tensor-call
# census), which is precisely the "many tiny instances" quantity that was sized.
# It has just earned a widened raw-byte gate -- production_ready, qualified
# batches 2/3/4, zero probability and zero product mismatches -- so what is left
# is to find out whether byte-identical output arrives faster.
#
# WHY THIS IS A SIBLING OF THE GQA-PAIR RUNNER AND NOT A FLAG ON IT
# ------------------------------------------------------------------
# scripts/fr13_run_b4_gqa_width4_timing.sh cannot express this arm. Its
# single-variable delta IS FR13_FA2_QROW32_B4_PRODUCTION_ARM, it hardcodes both
# TAW selectors to 0 as bare env literals, its preflight demands the FA2
# candidate .so identity and a HEAD-bound FA2 dual gate, and its verdict reducer
# validates FA2 credential identities. Bending it into a two-lever runner would
# make BOTH pairs harder to read and would put the FA2 seal at risk of an edit
# made for TAW's sake.
#
# What IS shared, and shared unchanged, is the thing that matters: the sealed
# windowing math in scripts/fr13_b4_width4_window_reduce.py. That reducer
# discovers arms from the refill ledger, the work census and deploy_speed --
# none of which know which lever produced them -- so any lever that serves 4
# slots / pool16 / refill windows reduces with it byte for byte. This runner
# lays its arms out in exactly the shape that reducer globs, and the verdict
# reducer delegates the windowing to it whole.
#
# THE SINGLE VARIABLE, AND THE TRAP IN IT
# ---------------------------------------
# The delta is FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION, 0 on the stock arm
# and 1 (plus the PASS bundle) on the candidate arm. Everything else is pinned
# byte-for-byte across the two arms.
#
# INCLUDING FA2, AND THIS IS THE TRAP. Since 32e240e15 the registry ARMS the
# GQA-pair FA2 kernel by default whenever a credentialed B4 launch does not name
# an arm. So "leave FA2 alone" is no longer a way to hold FA2 fixed -- whether
# the default fires depends on whether the launch presents the dual gate, which
# is a property of the launch and not of this pair. Two arms that differed in
# whether FA2's default fired would differ in TWO levers, and the sealed
# four-pass FA2 gain is 27 ms/step: it would swamp anything TAW does.
#
# So both arms NAME the FA2 production arm explicitly, present the same dual
# gate, and load the same pinned FA2 binary. FA2 is pinned ON, at the promoted
# production default, on BOTH arms. Turning it OFF would have been just as wrong
# in the other direction -- it would measure TAW against a configuration nothing
# ships.
#
# This paired screen is not the formal statistical hardware-floor acceptance
# gate, and the width-4 window class is an INSTRUMENT, not a citable seal.
set -euo pipefail

case "${FR13_RUN_B4_TAW_WIDTH4_TIMING:-0}" in
  1) ;;
  0)
    echo "B4 TAW native-production width-4 timing pair is disabled" >&2
    echo "set FR13_RUN_B4_TAW_WIDTH4_TIMING=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B4_TAW_WIDTH4_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${TAW_PRODUCTION_BUNDLE:?set TAW_PRODUCTION_BUNDLE to the PASS bundle earned at HEAD}"
: "${TAW_PRODUCTION_BUNDLE_SHA256:?set it to that PASS bundle SHA-256}"
: "${TAW_BYTE_GATE_JSON:?set TAW_BYTE_GATE_JSON to the byte-gate verdict that published it}"
: "${TAW_BYTE_GATE_SHA256:?set it to that byte-gate verdict SHA-256}"
# FA2 IS A PINNED CONTROL VARIABLE HERE, NOT THE LEVER -- but it is a
# CREDENTIALED control variable, so its inputs are still required. The launcher
# refuses a named B4 production arm without the pinned binary and its bound dual
# raw-byte gate, and it refuses them at BOOT. Requiring them here means a
# missing FA2 credential costs seconds instead of costing the stock arm's hours.
: "${QROW32_GQA_PAIR_FA2_SO:?set QROW32_GQA_PAIR_FA2_SO to the pinned FA2 production binary}"
: "${QROW32_GQA_PAIR_DUAL_GATE_JSON:?set it to the FA2 dual raw-byte gate PASS produced at HEAD}"
: "${QROW32_GQA_PAIR_DUAL_GATE_SHA256:?set it to that FA2 PASS artifact SHA-256}"

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FIXED32_MODE=${TAW_FIXED32_MODE:-tail6_fixed32}
TAW_SOURCE=scripts/fr13_device_multidraft_kernel.py
TAW_SOURCE_SCHEMA=fr13-fixed32-taw-all-parent-v7
TAW_SOURCE_SHA256=80595b6be9cb9cb8e1449fb3325e1b510e5c00186fa194b05bf16beaaa376687
TAW_CANDIDATE=fixed32_all_parent_commit_v2
TAW_ENGAGEMENT_RELPATH=logs/fr13_fixed32_taw_native_precompute.production_engagement.json
TAW_PRODUCTION_ROUTE=fixed32_native_precompute_production_candidate_return
TAW_PRODUCTION_SIDECAR_RELPATH=logs/fr13_fixed32_taw_native_precompute_production.arm
TAW_DIAGNOSTIC_SIDECAR_RELPATH=logs/fr13_fixed32_taw_native_precompute_diagnostic.arm
TAW_SERVED_BUNDLE_RELPATH=logs/fr13_fixed32_taw_native_precompute.production_pass.json
SIDECAR=scripts/fr13_qrow32_b4_pass_sidecar.py
PATCH_SOURCE=scripts/fr13_patch_fa2_tree_bias.py
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
PAIR_REDUCER=scripts/fr13_b4_taw_width4_pair_reduce.py
WINDOW_REDUCER=scripts/fr13_b4_width4_window_reduce.py
# EVIDENCE_SETS[16] -- byte-pinned identically to fr13_floor_gate.EVIDENCE_SETS.
SUBSET=config/fr13_fixed32/subset_b4_sixteen.json
SUBSET_SHA256=47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c
TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398,astropy__astropy-13453,astropy__astropy-13579,astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
TASK_COUNT=16
SLOTS=4
DRAFT_VOCAB_BLOCKS_HOST=scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
DRAFT_VOCAB_BLOCKS_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
# The FA2 production default, pinned identically on both arms.
FA2_PRODUCTION_ARM=gqa_pair
CANDIDATE_SHA256=af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb
CANDIDATE_BYTES=299813360
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81
B4_KV_CACHE_MEMORY_BYTES=49392123904
DRAFT_VOCAB_ROOT=1
DRAFT_VOCAB_K=65536
MANDATORY_WEIGHT_BYTES=25210209416
MANDATORY_WEIGHT_FLOOR_MS=92.345089436
ONE_SIDED_U95_CAP_MS=106.1968528514
RUN_CLASSIFICATION=real_swe_verified_pool16_b4_taw_native_production_width4_timing
LAUNCH_CLASSIFICATION=real_swe_verified_pool16_b4_taw_native_production_width4_timing_candidate
ENGAGEMENT_SCHEMA=fr13.fixed32.taw_native_precompute.production_engagement.v1
ONLY_ARM_DELTA=TAW_exact_reference_commit_to_native_precompute_production
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
TAW_SOURCE_FILE_SHA256=$(sha256sum "$TAW_SOURCE" | awk '{print $1}')
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
PATCH_SOURCE_SHA256=$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
# The sealed width-4 window reducer discovers arms as <root>/pass_NN/<mode>_*, so
# serving both arms into pass_00 lets that reducer run over this pair UNMODIFIED
# as an independent second read of the same bytes. A multi-pass campaign instead
# passes its own gate root and a per-pass index, so every pass lands in ONE root
# as pass_00..pass_NN and that same sealed reducer reads the whole campaign.
PASS_INDEX=${PASS_INDEX:-0}
PASS_ROOT=${PASS_ROOT:-$RUNROOT_ABS}
[[ "$PASS_INDEX" =~ ^[0-9]+$ ]] \
  || { echo "PASS_INDEX must be a non-negative integer" >&2; exit 2; }
PASS_ROOT=$(realpath -m "$PASS_ROOT")
[[ "$PASS_ROOT" == "$REPO/output/"* ]] \
  || { echo "PASS_ROOT must resolve below $REPO/output" >&2; exit 2; }
PASS_DIR="$PASS_ROOT/pass_$(printf '%02d' "$PASS_INDEX")"

# ARM ORDER -- identical reasoning to the GQA-pair runner. The second arm of a
# pass inherits a warmer page cache and a differently-aged host, and here the two
# arms are STOCK and CANDIDATE, so arm position aliases DIRECTLY into the very
# contrast being screened. A campaign alternates SC/CS on pass parity; a lone
# pair keeps the stock-first default.
ARM_ORDER=${ARM_ORDER:-SC}
case "$ARM_ORDER" in
  SC|CS) ;;
  *) echo "ARM_ORDER must be SC (stock first) or CS (candidate first)" >&2; exit 2 ;;
esac

case "$FIXED32_MODE" in
  tail6_fixed32)
    LOGICAL_TOPOLOGY=Tail23
    ACTIVE_DRAFTS=23
    VALID_MASK=0x7a9ce7ff
    ;;
  hydra27_fixed32)
    LOGICAL_TOPOLOGY=Hydra27
    ACTIVE_DRAFTS=27
    VALID_MASK=0x7abdffff
    ;;
  *)
    echo "TAW_FIXED32_MODE must be tail6_fixed32 or hydra27_fixed32" >&2
    exit 2
    ;;
esac
STOCK_ARM="${FIXED32_MODE}_taw_w4_stock_commit_b4_${TAG}"
CANDIDATE_ARM="${FIXED32_MODE}_taw_w4_native_production_b4_${TAG}"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for input in "$TAW_PRODUCTION_BUNDLE" "$TAW_BYTE_GATE_JSON" \
             "$QROW32_GQA_PAIR_FA2_SO" "$QROW32_GQA_PAIR_DUAL_GATE_JSON"; do
  [[ "$input" == /* && -f "$input" && ! -L "$input" ]] \
    || { echo "timing input must be an absolute regular non-symlink file: $input" >&2; exit 2; }
done
[[ "$TAW_PRODUCTION_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_PRODUCTION_BUNDLE" | awk '{print $1}')" == "$TAW_PRODUCTION_BUNDLE_SHA256" ]] \
  || { echo "TAW production PASS bundle identity mismatch" >&2; exit 2; }
[[ "$TAW_BYTE_GATE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$TAW_BYTE_GATE_JSON" | awk '{print $1}')" == "$TAW_BYTE_GATE_SHA256" ]] \
  || { echo "TAW byte-gate verdict identity mismatch" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_GQA_PAIR_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_GQA_PAIR_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
  || { echo "QROW32_GQA_PAIR_FA2_SO is not the pinned FA2 production binary" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" =~ ^[0-9a-f]{64}$ \
   && "$(sha256sum "$QROW32_GQA_PAIR_DUAL_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" ]] \
  || { echo "FA2 dual raw-byte gate PASS identity mismatch" >&2; exit 2; }
[[ "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" ]] \
  || { echo "canonical 16-task pool subset SHA-256 drift" >&2; exit 2; }
[[ -f "$DRAFT_VOCAB_BLOCKS_HOST" && ! -L "$DRAFT_VOCAB_BLOCKS_HOST" \
   && "$(sha256sum "$DRAFT_VOCAB_BLOCKS_HOST" | awk '{print $1}')" == "$DRAFT_VOCAB_BLOCKS_SHA256" ]] \
  || { echo "pinned root-64K draft-vocabulary block map drifted" >&2; exit 2; }
[[ -f "$PAIR_REDUCER" && ! -L "$PAIR_REDUCER" ]] \
  || { echo "missing the TAW width-4 pair reducer: $PAIR_REDUCER" >&2; exit 2; }
[[ -f "$WINDOW_REDUCER" && ! -L "$WINDOW_REDUCER" ]] \
  || { echo "missing the sealed width-4 window reducer: $WINDOW_REDUCER" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

# EVERY PRECONDITION MUST BE ONE THE RUN CAN SATISFY. Six separate campaign
# fossils were runners bound to an artifact nothing ever wrote, so the reducer,
# the window class AND the engagement emitter are resolved BEFORE any GPU time.
"$PYTHON_BIN" "$PAIR_REDUCER" --self-check \
  || { echo "TAW width-4 pair reducer failed its own self-check" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS" "$PASS_DIR"

# ---------------------------------------------------------------- credential ---
# THE PASS BUNDLE MUST BE RE-EARNED AT THIS HEAD, and the bundle is validated by
# the runtime's OWN entrypoint rather than by a reimplementation here.
# `_fr13_fixed32_taw_native_production_pass` is the exact function the container
# will call before it serves anything, so a bundle that passes here is a bundle
# that will pass there -- and it is what proves the bundle's
# source_contract_sha256 equals the LIVE kernel's, which is the binding that
# makes a byte gate mean anything about the code about to run.
#
# The HEAD binding is the byte-gate verdict's, and it is the same doctrine the
# FA2 dual gate carries: a gate earned before this runner existed is a gate
# earned against different source, so it must be re-earned. Adapting the runner
# moves HEAD; the gate is therefore re-run at the final HEAD before this pair.
"$PYTHON_BIN" - \
  "$TAW_SOURCE" "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_SHA256" "$TAW_CANDIDATE" \
  "$TAW_PRODUCTION_BUNDLE" "$TAW_PRODUCTION_BUNDLE_SHA256" \
  "$TAW_BYTE_GATE_JSON" "$TAW_BYTE_GATE_SHA256" \
  "$SOURCE_COMMIT" "$FIXED32_MODE" "$VALID_MASK" "$ACTIVE_DRAFTS" \
  "$RUNROOT_ABS/taw_bundle_binding.at_launch.json" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
source_schema = sys.argv[2]
source_contract_sha256 = sys.argv[3]
candidate = sys.argv[4]
bundle_path = Path(sys.argv[5])
bundle_sha256 = sys.argv[6]
gate_path = Path(sys.argv[7])
gate_sha256 = sys.argv[8]
source_commit = sys.argv[9]
mode = sys.argv[10]
valid_mask = int(sys.argv[11], 0)
active_drafts = int(sys.argv[12])
out_path = Path(sys.argv[13])

spec = importlib.util.spec_from_file_location("fr13_taw_timing_preflight", source_path)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import the TAW all-parent implementation")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
topology = module._fr13_fixed32_topology()

# The LIVE contract, recomputed from the source about to run -- not read off the
# constant, which is what a drifted source would also carry.
live = module._fr13_fixed32_taw_source_contract(topology, batch_size=4)
if (
    module._FR13_FIXED32_TAW_SOURCE_SCHEMA != source_schema
    or module._FR13_FIXED32_TAW_SOURCE_SHA256 != source_contract_sha256
    or live.get("source_contract_schema") != source_schema
    or live.get("source_contract_sha256") != source_contract_sha256
    or module._FR13_FIXED32_TAW_NATIVE_CANDIDATE != candidate
    or int(topology.VALID_MASK_BY_MODE[mode]) != valid_mask
    or module._fr13_fixed32_expected_active(topology, mode) != active_drafts
    or int(topology.PHYSICAL_DRAFTS) != 31
    or int(topology.PHYSICAL_ROWS) != 32
):
    raise SystemExit("fixed32 TAW source-v7 preflight contract drifted")

# The runtime's own bundle validator. It checks schema, status, candidate,
# source contract schema AND sha against the live constant, mode, valid mask,
# topology binding, the required production batches, and every per-batch PASS
# record inside the bundle.
bundle = module._fr13_fixed32_taw_native_production_pass(
    path=str(bundle_path), expected_mode=mode, expected_batch=4
)
bundle_raw = bundle_path.read_bytes()
if hashlib.sha256(bundle_raw).hexdigest() != bundle_sha256:
    raise SystemExit("TAW PASS bundle changed under the validator")

gate_raw = gate_path.read_bytes()
if hashlib.sha256(gate_raw).hexdigest() != gate_sha256:
    raise SystemExit("TAW byte-gate verdict changed under the validator")
gate = json.loads(gate_raw.decode("ascii"))
if (
    gate.get("status") != "pass"
    or gate.get("candidate") != candidate
    or gate.get("mode") != mode
    or gate.get("source_contract_schema") != source_schema
    or gate.get("source_contract_sha256") != source_contract_sha256
    or gate.get("production_bundle_sha256") != bundle_sha256
    or gate.get("live_bundle_sha256") != bundle_sha256
    or gate.get("qualified_batches") != bundle["qualified_batches"]
    or gate.get("required_production_batches")
    != bundle["required_production_batches"]
    or gate.get("probability_mismatches") != 0
    or gate.get("product_mismatches") != 0
    or gate.get("reference_always_served") is not True
    or gate.get("candidate_returned") is not False
    or gate.get("timing_eligible") is not False
):
    raise SystemExit("TAW byte-gate verdict does not publish this PASS bundle")
if gate.get("source_commit") != source_commit:
    raise SystemExit(
        "TAW byte gate was earned at "
        + repr(gate.get("source_commit"))
        + " but HEAD is "
        + repr(source_commit)
        + "; the gate must be re-earned at the commit that will serve"
    )

binding = {
    "schema": "fr13.fixed32.taw_native_production.timing_binding.v1",
    "status": "bound",
    "candidate": candidate,
    "mode": mode,
    "source_commit": source_commit,
    "source_contract_schema": source_schema,
    "source_contract_sha256": source_contract_sha256,
    "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "production_bundle_sha256": bundle_sha256,
    "byte_gate_sha256": gate_sha256,
    "qualified_batches": bundle["qualified_batches"],
    "required_production_batches": bundle["required_production_batches"],
    "pinned_min_batch": int(module._FR13_FIXED32_TAW_PINNED_MIN_BATCH),
    "valid_mask": valid_mask,
    "active_drafts": active_drafts,
}
out_path.write_text(
    json.dumps(binding, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
print(json.dumps(binding, sort_keys=True))
PY
TAW_BUNDLE_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/taw_bundle_binding.at_launch.json" | awk '{print $1}'
)

# The FA2 credential BOTH arms will serve on. It is a control variable, but a
# credentialed one, and its dual gate carries the same HEAD binding: the sidecar
# refuses a gate whose source_commit is not this commit.
"$PYTHON_BIN" "$SIDECAR" validate \
  --dual-gate "$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
  --expected-dual-gate-sha256 "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm "$FA2_PRODUCTION_ARM" \
  --patch-source "$PATCH_SOURCE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/fa2_dual_gate_binding.at_launch.json"
FA2_DUAL_GATE_BINDING_SHA256=$(
  sha256sum "$RUNROOT_ABS/fa2_dual_gate_binding.at_launch.json" | awk '{print $1}'
)

export BSIZE=4
export CONC=4
export WALL=0
export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"
export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"
export FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER"
export FR13_NEEDS_ALLOW=
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "canonical B4 qualification floor contract drifted" >&2; exit 2; }
# The registry's promoted FA2 default is what both arms are pinned TO, so a
# registry that no longer names it means this runner's pin is no longer "the
# production default" and the contrast has silently changed meaning.
[[ "${FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT:-}" == "$FA2_PRODUCTION_ARM" ]] \
  || { echo "the registry FA2 B4 production default is no longer $FA2_PRODUCTION_ARM" >&2; exit 2; }

"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"

printf 'classification=%s\ntiming_eligible=1\nformal_floor_acceptance_eligible=0\nonly_arm_delta=%s\ncandidate_arm_selector=taw_native_precompute_production\ntaw_candidate=%s\ntaw_engagement_schema=%s\ntaw_production_route=%s\ntopology=%s\nlogical_topology=%s\nactive_drafts=%s\nvalid_mask=%s\nphysical_drafts=31\nphysical_rows_root_inclusive=32\nbatch_size=4\nconcurrency=4\nslots=%s\ntask_pool=%s\ntask_refill=1\nagent_wall=none\nfixed_rows=128\ntask_ids=%s\nsubset=%s\nsubset_sha256=%s\ndraft_vocab_root=%s\ndraft_vocab_k=%s\ndraft_vocab_blocks=%s\ndraft_vocab_blocks_sha256=%s\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nlauncher_pid=%s\nrunroot=%s\npass_dir=%s\npass_index=%s\narm_order=%s\nstock_arm=%s\ncandidate_arm=%s\nsource_commit=%s\ntaw_source_contract_schema=%s\ntaw_source_contract_sha256=%s\ntaw_source_file_sha256=%s\ntaw_production_bundle_sha256=%s\ntaw_byte_gate_sha256=%s\ntaw_bundle_binding_sha256=%s\nfa2_production_arm=%s\nfa2_pinned_identically_on_both_arms=1\nfa2_so_sha256=%s\nfa2_so_size=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\nfa2_dual_gate_sha256=%s\nfa2_dual_gate_binding_sha256=%s\nrunner_sha256=%s\nsidecar_sha256=%s\npatch_source_sha256=%s\nenforce_eager=0\ncudagraph_mode=FULL_AND_PIECEWISE\nkv_cache_memory_bytes=%s\nstarted=%s\n' \
  "$LAUNCH_CLASSIFICATION" "$ONLY_ARM_DELTA" "$TAW_CANDIDATE" \
  "$ENGAGEMENT_SCHEMA" "$TAW_PRODUCTION_ROUTE" \
  "$FIXED32_MODE" "$LOGICAL_TOPOLOGY" "$ACTIVE_DRAFTS" "$VALID_MASK" \
  "$SLOTS" "$TASK_COUNT" \
  "$TASK_IDS" "$SUBSET" "$SUBSET_SHA256" "$DRAFT_VOCAB_ROOT" "$DRAFT_VOCAB_K" \
  "$DRAFT_VOCAB_BLOCKS_CONTAINER" "$DRAFT_VOCAB_BLOCKS_SHA256" \
  "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$$" "$RUNROOT_ABS" "$PASS_DIR" \
  "$PASS_INDEX" "$ARM_ORDER" \
  "$STOCK_ARM" "$CANDIDATE_ARM" \
  "$SOURCE_COMMIT" "$TAW_SOURCE_SCHEMA" "$TAW_SOURCE_SHA256" \
  "$TAW_SOURCE_FILE_SHA256" "$TAW_PRODUCTION_BUNDLE_SHA256" \
  "$TAW_BYTE_GATE_SHA256" "$TAW_BUNDLE_BINDING_SHA256" \
  "$FA2_PRODUCTION_ARM" "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
  "$FA2_DUAL_GATE_BINDING_SHA256" "$RUNNER_SHA256" \
  "$SIDECAR_SHA256" "$PATCH_SOURCE_SHA256" "$B4_KV_CACHE_MEMORY_BYTES" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json" || return $?
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json" || return $?
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" \
     && "$(sha256sum "$TAW_SOURCE" | awk '{print $1}')" == "$TAW_SOURCE_FILE_SHA256" \
     && "$(sha256sum "$SIDECAR" | awk '{print $1}')" == "$SIDECAR_SHA256" \
     && "$(sha256sum "$PATCH_SOURCE" | awk '{print $1}')" == "$PATCH_SOURCE_SHA256" ]] \
    || { echo "B4 TAW width-4 timing source changed during execution" >&2; return 14; }
  [[ "$(sha256sum "$TAW_PRODUCTION_BUNDLE" | awk '{print $1}')" == "$TAW_PRODUCTION_BUNDLE_SHA256" \
     && "$(sha256sum "$TAW_BYTE_GATE_JSON" | awk '{print $1}')" == "$TAW_BYTE_GATE_SHA256" \
     && "$(sha256sum "$QROW32_GQA_PAIR_DUAL_GATE_JSON" | awk '{print $1}')" == "$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
     && "$(sha256sum "$QROW32_GQA_PAIR_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" ]] \
    || { echo "a presented credential changed during timing" >&2; return 14; }
  [[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
     && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
    || { echo "frozen source changed during timing" >&2; return 14; }
  MANIFEST_FINALIZED=1
}

runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    if finalize_manifests; then :; else
      local manifest_rc=$?
      (( rc == 0 )) && rc=$manifest_rc
    fi
  fi
  exit "$rc"
}
trap runner_exit EXIT

# The ONLY difference between the two invocations below is
# FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION and the PASS bundle it requires.
# Everything else -- the FA2 binary AND its production selector, the subset, the
# pool depth, the refill flag, the sampling, the topology, the vocabulary, the
# graph mode -- is byte-for-byte the same.
run_arm() {
  local arm=$1
  local taw_production=$2
  local taw_bundle=""
  if [[ "$taw_production" == "1" ]]; then
    taw_bundle=$TAW_PRODUCTION_BUNDLE
  fi
  echo "===== $arm: pool16 B4 TAW timing taw_production=$taw_production ====="
  # AGENT_WALL_S is passed EMPTY on purpose: the width-4 baseline was measured
  # with no wall, and a wall would truncate tasks and deform the admission
  # ledger that defines the window.
  if env \
      RUNROOT="$PASS_DIR" \
      OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4 AGENT_WALL_S= \
      FR13_B4_TASK_REFILL=1 \
      KV_CACHE_MEMORY_BYTES="$B4_KV_CACHE_MEMORY_BYTES" \
      FR13_FIXED32_B1_DIAGNOSTIC=0 \
      FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT" \
      FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K" \
      FR13_DRAFT_VOCAB_BLOCKS="$DRAFT_VOCAB_BLOCKS_CONTAINER" \
      FR13_NEEDS_ALLOW= \
      FR10_METRICS=0 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
      FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1 \
      FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0 \
      FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
      FR13_SFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}.json" \
      FR13_DFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_dfwd.json" \
      FR13_CFWD_GPU_TIMER_JSON="/workspace/output/fr13_sfwd_sidecar/${arm}_cfwd.json" \
      FR13_DEVICE_MULTIDRAFT=1 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0 \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION="$taw_production" \
      FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$taw_bundle" \
      FR13_FA2_QROW32_B4_TIMING_ARM= \
      FR13_FA2_QROW32_B4_PRODUCTION_ARM="$FA2_PRODUCTION_ARM" \
      FR13_FA2_QROW32_B4_DUAL_GATE_JSON="$QROW32_GQA_PAIR_DUAL_GATE_JSON" \
      FR13_FA2_QROW32_B4_DUAL_GATE_SHA256="$QROW32_GQA_PAIR_DUAL_GATE_SHA256" \
      FR13_FA2_QROW32_B4_EXACT4_TASK_IDS="$TASK_IDS" \
      FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256="$SUBSET_SHA256" \
      FR13_FA2_QROW32_B4_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
      FR13_FA2_QROW32_SO_SHA256="$CANDIDATE_SHA256" \
      FR13_FA2_QROW32_SO_SIZE="$CANDIDATE_BYTES" \
      FR13_FA2_QROW32_FA2_HEAD="$FA2_HEAD" \
      FR13_FA2_QROW32_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
      FR13_FA2_QROW32_SOURCE_COMMIT="$SOURCE_COMMIT" \
      FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
      FR13_FA2_QROW32_LIVE_PAGED_AB_ARM= \
      FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
      FR13_FA2_QROW32_B1_LIVE_AB_ARM= FR13_FA2_QROW32_B1_PRODUCTION_ARM= \
      FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0 FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
      FR13_DFWD_UNIFIED_BM8_INSTANCE_ID= \
      FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
      FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
      FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
      FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
      FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
      FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
      FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
      FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
      FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
      FR13_FIXED32_CUTLASS_WAVE=stock \
      FR13_FIXED32_CUTLASS_WAVE_SO= \
      FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0 \
      FR13_FIXED32_ATTRIBUTION_ONLY=0 \
      FORKED_FA2_SO="$QROW32_GQA_PAIR_FA2_SO" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh \
        "$arm" "$FIXED32_MODE" "$SUBSET" \
        > "$RUNROOT_ABS/$arm.runlog" 2>&1; then
    :
  else
    local serve_rc=$?
    printf 'arm=%s serve_rc=%s ended=%s\n' \
      "$arm" "$serve_rc" "$(date -u +%FT%TZ)" \
      >> "$RUNROOT_ABS/launcher_meta.txt"
    return "$serve_rc"
  fi
  local arm_dir="$PASS_DIR/$arm"
  local container_env="$arm_dir/container_env.txt"
  [[ -f "$container_env" && ! -L "$container_env" ]] \
    || { echo "$arm lacks a regular container environment artifact" >&2; return 4; }
  # THE SINGLE VARIABLE, PROVED FROM THE RECORDED ENVIRONMENT. FA2 is asserted
  # here on BOTH arms with the SAME expected values, which is what makes this a
  # one-lever contrast rather than a two-lever one.
  [[ "$(grep -Fxc "FR13_FIXED32_MODE=$FIXED32_MODE" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_ROOT=$DRAFT_VOCAB_ROOT" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_K=$DRAFT_VOCAB_K" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_DRAFT_VOCAB_BLOCKS=$DRAFT_VOCAB_BLOCKS_CONTAINER" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_B4_TASK_REFILL=1" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$taw_production" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_PRODUCTION_ARM=$FA2_PRODUCTION_ARM" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_TIMING_ARM=" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_B4_DUAL_GATE_SHA256=$QROW32_GQA_PAIR_DUAL_GATE_SHA256" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_SO_SHA256=$CANDIDATE_SHA256" "$container_env")" -eq 1 \
     && "$(grep -Fxc "FR13_FA2_QROW32_LIVE_PAGED_AB=0" "$container_env")" -eq 1 ]] \
    || { echo "$arm did not run the declared single-variable B4 pool16 TAW selector" >&2; return 4; }
  # THE WINDOW IS DEFINED BY THIS LEDGER. Without it there is no width-4 phase
  # to reduce and the whole verdict is vacuous, so it is required here rather
  # than discovered missing hours later at reduce time.
  local ledger="$arm_dir/swe_out/verified/fr13_task_refill_ledger.jsonl"
  local ledger_summary="$arm_dir/swe_out/verified/fr13_task_refill_summary.json"
  [[ -f "$ledger" && ! -L "$ledger" && -f "$ledger_summary" && ! -L "$ledger_summary" ]] \
    || { echo "$arm lacks the admission ledger the width-4 window is DEFINED by" >&2; return 4; }
  # A pool arm's per-task brackets are STAGGERED, and fr13_measure REFUSES a
  # staggered reduction without the engine work census, so it is mandatory.
  local deploy_census="$arm_dir/logs/fr13_fixed32_work_census.jsonl"
  [[ -f "$deploy_census" && ! -L "$deploy_census" ]] \
    || { echo "$arm lacks the work census the B4 bracket reduction is gated on" >&2; return 4; }

  # ------------------------------------------------ TAW engagement attestation
  # Two INDEPENDENT witnesses of what the TAW commit actually served: the
  # engagement artifact the kernel writes at final flush, and the served route
  # published into every work-census event. They are required to agree with each
  # other AND with the recorded environment.
  local engagement="$arm_dir/$TAW_ENGAGEMENT_RELPATH"
  local production_sidecar="$arm_dir/$TAW_PRODUCTION_SIDECAR_RELPATH"
  local diagnostic_sidecar="$arm_dir/$TAW_DIAGNOSTIC_SIDECAR_RELPATH"
  local served_events
  served_events=$(grep -c -- "$TAW_PRODUCTION_ROUTE" "$deploy_census" || true)
  [[ ! -e "$diagnostic_sidecar" && ! -L "$diagnostic_sidecar" ]] \
    || { echo "$arm carries a TAW diagnostic sidecar; this pair serves no byte-A/B arm" >&2; return 4; }
  # THE ATTESTATION CONSISTENCY RULE, STATED AS ITS OWN CHECK.
  # container_env PRODUCTION=0 can NEVER coexist with candidate-served evidence.
  # This is the runner-side half of the authoritative-zero fix in
  # _fr13_fixed32_taw_native_arm_sources: the kernel makes an explicit 0 beat a
  # stray sidecar, and this proves the recorded environment and the served
  # evidence agree afterwards. Either witness alone could be forged by a stale
  # file; the conjunction is what the verdict rests on.
  if [[ "$(grep -Fxc 'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0' "$container_env")" -eq 1 ]]; then
    [[ ! -e "$engagement" && ! -L "$engagement" \
       && ! -e "$production_sidecar" && ! -L "$production_sidecar" \
       && "$served_events" -eq 0 ]] \
      || { echo "$arm records PRODUCTION=0 yet carries TAW candidate-served evidence (engagement=$engagement served_events=$served_events)" >&2; return 4; }
  fi
  if [[ "$taw_production" == "1" ]]; then
    [[ -f "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm lacks the TAW production engagement artifact" >&2; return 4; }
    [[ -f "$production_sidecar" && ! -L "$production_sidecar" ]] \
      || { echo "$arm lacks the TAW production arm sidecar" >&2; return 4; }
    [[ "$served_events" -gt 0 ]] \
      || { echo "$arm never published the TAW production route into its work census" >&2; return 4; }
    # The bundle the arm actually served must be the bundle validated at
    # preflight, byte for byte -- the launcher COPIES it into the container, so
    # a copy of something else would otherwise go unnoticed.
    [[ "$(sha256sum "$arm_dir/$TAW_SERVED_BUNDLE_RELPATH" | awk '{print $1}')" == "$TAW_PRODUCTION_BUNDLE_SHA256" ]] \
      || { echo "$arm served a PASS bundle that is not the validated one" >&2; return 4; }
  else
    # A stock arm that emitted an engagement record would mean the candidate
    # selector leaked across the pair, which would invalidate the comparison.
    [[ ! -e "$engagement" && ! -L "$engagement" ]] \
      || { echo "$arm emitted a TAW production engagement on the stock-commit arm" >&2; return 4; }
    [[ "$served_events" -eq 0 ]] \
      || { echo "$arm published the TAW production route on the stock-commit arm" >&2; return 4; }
  fi

  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$arm" --out-root "$arm_dir/swe_out" \
    --expected-tok-per-draft 31 --batch-size 4 \
    --work-census "$deploy_census" \
    --out "$arm_dir/deploy_speed_fullwall.json"
  printf 'arm=%s serve_rc=0 taw_production=%s taw_served_events=%s container_env_sha256=%s ended=%s\n' \
    "$arm" "$taw_production" "$served_events" \
    "$(sha256sum "$container_env" | awk '{print $1}')" \
    "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/launcher_meta.txt"
}

# Both orders run the SAME two arms with the SAME single-variable delta; only
# which one boots first changes, and launcher_meta.txt records it so a reader
# can never have to infer it from timestamps.
if [[ "$ARM_ORDER" == "SC" ]]; then
  run_arm "$STOCK_ARM" 0
  [[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
    || { echo "Docker state was not clean between the paired arms" >&2; exit 2; }
  run_arm "$CANDIDATE_ARM" 1
else
  run_arm "$CANDIDATE_ARM" 1
  [[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
    || { echo "Docker state was not clean between the paired arms" >&2; exit 2; }
  run_arm "$STOCK_ARM" 0
fi

CANDIDATE_ENGAGEMENT="$PASS_DIR/$CANDIDATE_ARM/$TAW_ENGAGEMENT_RELPATH"
CANDIDATE_ENGAGEMENT_SHA256=$(sha256sum "$CANDIDATE_ENGAGEMENT" | awk '{print $1}')

finalize_manifests

# ------------------------------------------------------------------ verdict ---
# ALL timing math lives in the reducer, over the sealed window class. The runner
# serves and attests; it does not decide.
"$PYTHON_BIN" "$PAIR_REDUCER" \
  --runroot "$RUNROOT_ABS" \
  --stock-arm "$PASS_DIR/$STOCK_ARM" \
  --candidate-arm "$PASS_DIR/$CANDIDATE_ARM" \
  --mode "$FIXED32_MODE" \
  --source-commit "$SOURCE_COMMIT" \
  --subset "$SUBSET" \
  --production-bundle-sha256 "$TAW_PRODUCTION_BUNDLE_SHA256" \
  --byte-gate-sha256 "$TAW_BYTE_GATE_SHA256" \
  --source-contract-sha256 "$TAW_SOURCE_SHA256" \
  --runner-sha256 "$RUNNER_SHA256" \
  --kernel-source-sha256 "$TAW_SOURCE_FILE_SHA256" \
  --out "$RUNROOT_ABS/taw_width4_timing_pair.json"
REDUCE_RC=$?

# An INDEPENDENT second read of the same bytes by the sealed window reducer,
# unmodified. It cannot return a verdict for a 2-arm pair (it wants 4 passes on
# both topologies), and it is not asked to -- it is run for its per-arm windowed
# records, which must agree with the pair reducer's. That it runs at all over a
# TAW pair is the point: the windowing math is lever-agnostic.
"$PYTHON_BIN" "$WINDOW_REDUCER" \
  --gate-root "$RUNROOT_ABS" \
  --source-commit "$SOURCE_COMMIT" \
  --out "$RUNROOT_ABS/width4_window_independent_read.json" || true

printf 'taw_width4_timing_pair=%s candidate_engagement_sha256=%s reduce_rc=%s completed=%s\n' \
  "$RUNROOT_ABS/taw_width4_timing_pair.json" "$CANDIDATE_ENGAGEMENT_SHA256" \
  "$REDUCE_RC" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
exit "$REDUCE_RC"
