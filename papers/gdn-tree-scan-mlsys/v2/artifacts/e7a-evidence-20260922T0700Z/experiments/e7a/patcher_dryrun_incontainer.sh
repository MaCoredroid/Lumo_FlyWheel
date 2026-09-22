#!/usr/bin/env bash
# Runs INSIDE a CPU-only container of the pinned image (root). /snap = script snapshot (ro), /out = run dir (rw),
# /scratch = scratch (rw, holds the pristine .py tree), /workspace = worktree (ro). No GPU, no model, no server.
set -euo pipefail
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
AUDIT=/snap/emitted_globals_audit.py
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a /out/dryrun.log; }
log "python $(python3 --version 2>&1); vllm tree $VLLM"
mkdir -p /scratch/orig_tree
( cd /usr/local/lib/python3.12/dist-packages && find vllm -name '*.py' -type f -print0 | tar --null -cf - -T - ) | tar -xpf - -C /scratch/orig_tree
log "pristine .py tree snapshot: $(find /scratch/orig_tree -name '*.py' | wc -l) files"
python3 -c "import vllm, sys; print('vllm', vllm.__version__)" 2>&1 | tail -1 | tee -a /out/dryrun.log || true

restore(){  # restore every file modified since the stamp from the pristine tree; remove created files
  while IFS= read -r f; do
    rel=${f#/usr/local/lib/python3.12/dist-packages/}
    if [[ -f "/scratch/orig_tree/$rel" ]]; then cp -p "/scratch/orig_tree/$rel" "$f"; else rm -f "$f"; fi
  done < <(find "$VLLM" -type f -newer /out/stamp)
  find "$VLLM" -name '__pycache__' -type d -newer /out/stamp -exec rm -rf {} + 2>/dev/null || true
}
arm(){  # arm <name> <patcher_path> <run_shim:0|1> [extra env assignments...]
  local name=$1 patcher=$2 shim=$3; shift 3
  local d=/out/$name; mkdir -p "$d/files" "$d/logs"
  touch /out/stamp; sleep 1
  log "=== arm $name: patcher=$patcher shim=$shim extra_env=[$*] ==="
  sha256sum "$patcher" > "$d/patcher.sha256"
  local rc=0
  ( export "$@"; python3 "$patcher" ) > "$d/patcher_stdout.txt" 2> "$d/patcher_stderr.txt" || rc=$?
  echo "$rc" > "$d/patcher_rc.txt"; log "patcher rc=$rc"
  if [[ "$shim" == "1" ]]; then
    local src=0
    ( export "$@"; E7A_CAPTURE_SHIM_REPORT="$d/logs/e7a_capture_shim.json" python3 /snap/e7a_capture_shim.py ) > "$d/shim_stdout.txt" 2>&1 || src=$?
    echo "$src" > "$d/shim_rc.txt"; log "shim rc=$src"
    if [[ -n "${EXTRA_SHIM:-}" ]]; then   # e.g. the E7b substitution shim, applied after the E7a shim exactly as a boot would
      local xrc=0
      ( export "$@"; E7B_SHIM_REPORT="$d/logs/e7b_substitute_shim.json" E7B_RUNTIME_DIR="${EXTRA_SHIM_RUNTIME_DIR:-/workspace/papers/gdn-tree-scan-mlsys/v2/experiments/e2}" python3 "$EXTRA_SHIM" ) > "$d/extra_shim_stdout.txt" 2>&1 || xrc=$?
      echo "$xrc" > "$d/extra_shim_rc.txt"; log "extra shim ($EXTRA_SHIM) rc=$xrc"; tail -2 "$d/extra_shim_stdout.txt" | tee -a /out/dryrun.log
    fi
  fi
  : > "$d/modified_files.txt"
  while IFS= read -r f; do
    rel=${f#/usr/local/lib/python3.12/dist-packages/}
    echo "$rel" >> "$d/modified_files.txt"
    mkdir -p "$d/files/$(dirname "$rel")"; cp -p "$f" "$d/files/$rel"
  done < <(find "$VLLM" -type f -name '*.py' -newer /out/stamp | sort)
  log "modified .py files: $(wc -l < "$d/modified_files.txt")"
  ( cd "$d/files" && find . -type f -name '*.py' | sort | xargs sha256sum ) > "$d/files.sha256"
  # audit: delta vs pristine for each modified file
  : > "$d/audit_summary.txt"; mkdir -p "$d/audit"
  while IFS= read -r rel; do
    o="/scratch/orig_tree/$rel"; p="$d/files/$rel"; j="$d/audit/$(echo "$rel" | tr '/' '_').json"
    if [[ -f "$o" ]]; then python3 "$AUDIT" delta "$o" "$p" --json "$j" >> "$d/audit_summary.txt" 2>&1 || true
    else python3 "$AUDIT" audit "$p" --json "$j" >> "$d/audit_summary.txt" 2>&1 || true; fi
  done < "$d/modified_files.txt"
  { grep -c 'NEW undefined globals = \[\]' "$d/audit_summary.txt" || true; } | sed 's/^/files with zero new undefined: /' | tee -a /out/dryrun.log
  grep -v 'NEW undefined globals = \[\]' "$d/audit_summary.txt" | tee -a /out/dryrun.log || true
  restore; log "restored pristine tree; remaining newer files: $(find "$VLLM" -type f -newer /out/stamp | wc -l)"
}
if [[ -n "${EXTRA_SHIM:-}" ]]; then
  # E7b dry-run: worktree (fixed) patcher + E7a shim + the extra shim; then stop
  arm E7B_worktree_patcher_plus_shims /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py 1 E7A_DRYRUN=1
  chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" /out /scratch/orig_tree || true; log "DRYRUN DONE (extra shim arm)"; exit 0
fi
# Arm A: HEAD patcher, attempt-2 (plain route) env  [+ shim, exactly as the capture boot does]
arm A_head_plain /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py 1 E7A_DRYRUN=1
# Arm B: FIXED patcher, same env [+ shim]. The patcher reads sibling files next to itself (e.g. fr13_sg_warmup_capture_inject.py),
# so the fixed copy is placed in a full copy of scripts/ with ONLY the patcher replaced (recorded by sha256 list).
rm -rf /scratch/scripts_fixed; cp -a /workspace/scripts /scratch/scripts_fixed
cp /snap/fr10_phase4_patch_vllm_tree_gdn.FIXED.py /scratch/scripts_fixed/fr10_phase4_patch_vllm_tree_gdn.py
( cd /scratch/scripts_fixed && sha256sum fr10_phase4_patch_vllm_tree_gdn.py fr13_sg_warmup_capture_inject.py ) > /out/scripts_fixed.sha256
( cd /workspace/scripts && sha256sum fr10_phase4_patch_vllm_tree_gdn.py fr13_sg_warmup_capture_inject.py ) > /out/scripts_head.sha256
arm B_fixed_plain /scratch/scripts_fixed/fr10_phase4_patch_vllm_tree_gdn.py 1 E7A_DRYRUN=1
# (A fixed32-mode control arm was tried in the first invocation of this run and refused by the patcher's own
#  _fr13_fixed32_validate_patch_env: the fixed32 route is a separately-gated configuration, not the intended one.)
chown -R "${HOST_UID:-1000}:${HOST_GID:-1000}" /out /scratch/orig_tree || true
log "DRYRUN DONE"
