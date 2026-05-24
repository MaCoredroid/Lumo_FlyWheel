#!/bin/bash
# Raw-only incremental committer for CNB-55 v4a (config D, B=1) run on alienware.
# Mirrors the no-compute / commit-verbose-raw discipline: rsync the per-task
# bundles back, snapshot steptrace, commit+push everything raw. Grading is a
# SEPARATE off-box step (run_v4a_graders_on_validation.py) after the run.
set -u
cd /home/mark/shared/lumoFlyWheel
rsync -az alienware:~/cnb_v4a/output/run1/ output/cnb_v4a_run1/ >/dev/null 2>&1
cp /tmp/swe_dgx_steptrace.jsonl output/cnb_v4a_run1/dgx_steptrace.jsonl 2>/dev/null
python3 - <<'PY'
import json, pathlib
base=pathlib.Path('output/cnb_v4a_run1'); done=[]; susp=[]
if base.is_dir():
    for d in sorted(base.iterdir()):
        meta=d/'runner_metadata.json'
        if not meta.is_file(): continue
        try: m=json.load(open(meta))
        except Exception: continue
        e=m.get('elapsed_s'); done.append(d.name)
        if e is not None and e<120: susp.append(f"{d.name}={e:.0f}s")
print("DONE_TASKS:", len(done), " ".join(done))
print("SUSPICIOUS(<120s, possible tunnel break):", " ".join(susp) or "none")
open('/tmp/_cnb_done','w').write("\n".join(done))
PY
git add -f output/cnb_v4a_run1/ 2>/dev/null
if git diff --cached --quiet; then
  echo "nothing new to commit"
else
  n=$(wc -l < /tmp/_cnb_done 2>/dev/null || echo 0)
  git commit -q -m "CNB-55 v4a (config D, B=1) raw: ${n}/11 tasks

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
  git pull --rebase --autostash origin main 2>&1 | tail -1
  git push origin main 2>&1 | tail -1
fi
