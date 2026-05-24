#!/bin/bash
# Raw-only incremental committer for conc_probe16d. Validity is keyed on
# elapsed_total (runner_metadata ended_at - started_at) >= 300s, NOT
# codex.elapsed_s (which only reflects the final attempt after a Bundle-B retry
# and wrongly excludes valid long instances). Prints FASTFAIL list (elapsed<300).
set -u
cd /home/mark/shared/lumoFlyWheel
rsync -az alienware:~/swe_conc_probe/output/conc_probe16d/ output/swe_conc_probe16d/ >/dev/null 2>&1
cp /tmp/swe_dgx_steptrace.jsonl output/swe_conc_probe16d/dgx_steptrace.jsonl 2>/dev/null
python3 - <<'PY'
import json, datetime as dt, subprocess, pathlib
def et(meta):
    try:
        d=json.load(open(meta)); s=d.get('started_at'); e=d.get('ended_at')
        if s and e:
            f=lambda t: dt.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()
            return f(e)-f(s)
    except Exception: pass
    return None
base='output/swe_conc_probe16d'; commit=[]; fastfail=[]
for arm in ('c4','c2','c1'):
    pt=pathlib.Path(base)/arm/'per_task'
    if not pt.is_dir(): continue
    for d in sorted(pt.iterdir()):
        meta=d/'runner_metadata.json'
        if not meta.is_file(): continue
        e=et(meta)
        if e is None: continue
        if e < 300: fastfail.append(f"{arm}/{d.name}={e:.0f}s"); continue
        tracked=subprocess.run(['git','ls-files','--error-unmatch',str(meta)],capture_output=True).returncode==0
        if not tracked: commit.append(f"{arm}/per_task/{d.name}")
print("FASTFAIL:", " ".join(fastfail) or "none")
open('/tmp/_commitset','w').write("\n".join(commit))
PY
commitset=$(cat /tmp/_commitset)
if [ -n "$commitset" ]; then
  for x in $commitset; do git add -f "output/swe_conc_probe16d/$x"; done
  git add -f output/swe_conc_probe16d/c4.log output/swe_conc_probe16d/c2.log output/swe_conc_probe16d/c1.log output/swe_conc_probe16d/driver.log output/swe_conc_probe16d/dgx_steptrace.jsonl 2>/dev/null
  n=$(echo "$commitset" | wc -w)
  git commit -q -m "SWE-Bench conc-probe16d raw: +$n valid [$commitset]

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
  git push origin main 2>&1 | tail -1
else echo "nothing new valid to commit"; fi
