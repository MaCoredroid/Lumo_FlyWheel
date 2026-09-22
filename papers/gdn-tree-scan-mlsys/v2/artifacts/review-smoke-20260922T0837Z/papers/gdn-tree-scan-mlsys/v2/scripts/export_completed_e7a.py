#!/usr/bin/env python3
"""Create a PRIVATE local evidence snapshot of completed E7a work. Never uploads or runs inference.
Original absolute provenance paths remain unchanged. Build/hash checking is portable; replay launch paths require
an explicit local mapping. JIT caches are excluded but their manifest hashes remain. Not a public release bundle.
"""
from pathlib import Path
import argparse, hashlib, json, tarfile, datetime
ROOTS = [
 "experiments/e7a", "experiments/out-20260921T230815Z-e7a-step2-prefixes", "experiments/out-20260921T231853Z-e7a-step2-native-score",
 "experiments/out-20260922T010711Z-e7a-step2-captures-v4-pilot6to8",
 "experiments/out-20260921T222616Z-e7a-ladder/source_snapshot", "experiments/out-20260921T230026Z-e7a-ladder-v3",
 "experiments/out-20260921T231030Z-e7a-tinygates-v3", "experiments/out-20260922T013313Z-e7a-fresh-pilot-v2",
 "experiments/out-20260922T051611Z-e7a-fresh-confirmation", "experiments/out-20260922T064200Z-e7a-freeze-retrospective-audit",
 "experiments/out-20260922T002242Z-e7a-step2-captures-v4", "experiments/out-20260922T003357Z-e7a-step2-captures-v4-pilot2to8",
 "experiments/out-20260922T013708Z-e7a-step2-captures-v5-confirmation",
 "experiments/out-20260922T024543Z-e7a-step2-captures-v5-confirmation-part2",
 "experiments/out-20260922T050330Z-e7a-step2-captures-v5-confirmation-part3"]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--paper",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists(), "Preserve prior snapshots; choose a new output name"
 files=[]
 for root in ROOTS:
  assert (a.paper/root).is_dir(), root
  files += [f for f in (a.paper/root).rglob("*") if f.is_file() and not any(x in {"triton_cache","__pycache__",".cache"} for x in f.parts)]
 files=sorted(set(files)); entries=[(f,str(f.relative_to(a.paper))) for f in files]
 # Copy exactly the four used historical tensors, preserving an explicit /hist path map.
 hist_manifest=json.loads((a.paper/"experiments/e7a/historical_payload_manifest.json").read_text())
 hist_hashes={r["path"]:r["sha256"] for r in hist_manifest["files"]}
 run=json.loads((a.paper/"experiments/out-20260921T230026Z-e7a-ladder-v3/manifest.json").read_text())
 historical_map=[]
 for original in run["args"]["payloads"]:
  rel=Path(original).relative_to("/hist");f=Path(hist_manifest["source_root"])/rel
  assert hashlib.sha256(f.read_bytes()).hexdigest()==hist_hashes[str(f)],f
  arc="historical-inputs/output/"+str(rel);entries.append((f,arc))
  historical_map.append({"original":original,"host_source":str(f),"archive_path":arc,"sha256":hist_hashes[str(f)]})
 rows=[]
 for f,name in entries: rows.append({"path":name,"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})
 report={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"kind":"private completed E7a evidence snapshot; not a public release", "source_root":str(a.paper),"excluded":["triton_cache","__pycache__",".cache"],"files":rows,"historical_input_map":historical_map}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(a.output,"w:gz") as t:
  for (f,name),r in zip(entries,rows):
   assert hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"], "Source changed during snapshot"
   t.add(f,arcname=r["path"],recursive=False)
 manifest=a.output.with_suffix(a.output.suffix+".manifest.json");manifest.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"files":len(entries),"source_bytes":sum(r["size"] for r in rows),"archive_bytes":a.output.stat().st_size,"archive_sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"manifest":str(manifest)}))
if __name__=="__main__":main()
