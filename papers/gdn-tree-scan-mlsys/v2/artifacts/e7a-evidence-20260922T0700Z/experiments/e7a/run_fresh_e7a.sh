#!/usr/bin/env bash
# Feed ONLY provenance-passing fresh payloads to the E7a device harness. A payload qualifies iff its run dir has a
# capture_provenance.json with all_pass=true whose payload_sha256 equals the file's sha256 (the loader then labels it
# VERIFIED-FRESH from the same evidence; pathname plays no role). Usage: run_fresh_e7a.sh <out_dir> <capture_root>...
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921
E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
OUT=$1; shift
PAYS=()
for root in "$@"; do
  for d in "$root"/capture_*_p*/; do
    d=${d%/}
    prov="$d/capture_provenance.json"
    if [[ -f "$prov" ]] && python3 - "$prov" "$d/logs/tree_gdn_capture_payload.pt" <<'PY'
import json, sys, hashlib
v = json.load(open(sys.argv[1])); h = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
sys.exit(0 if (v.get("all_pass") and v.get("payload_sha256") == h) else 1)
PY
    then
      PAYS+=("/work/${d#"$WT"/}/logs/tree_gdn_capture_payload.pt")
    else
      echo "skipping $d (no PROVENANCE PASS)" >&2
    fi
  done
done
[[ ${#PAYS[@]} -gt 0 ]] || { echo "no provenance-passing payloads" >&2; exit 2; }
printf '%s\n' "${PAYS[@]}" > "$OUT.payloads.txt" 2>/dev/null || true
exec bash "$E/run_in_image.sh" "$OUT" "$E/e7a_device.py" --payloads "${PAYS[@]}" --timing --iters 200 --warmup 20
