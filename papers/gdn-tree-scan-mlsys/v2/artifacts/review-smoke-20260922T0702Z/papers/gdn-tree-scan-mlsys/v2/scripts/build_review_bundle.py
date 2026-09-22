#!/usr/bin/env python3
"""Build a private review checkpoint from explicit manuscript inputs and hash-bound historical evidence.
Not a public release. E7a tensors are in a separately hashed companion archive. No network/inference.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile,io
PAPER=Path(__file__).resolve().parents[1]; REPO=PAPER.parents[2]
STATIC=["main.tex","abstract.tex","ref.bib","IEEEtran.cls","main.bbl","main.pdf","README.md","review-experiments.md","paper.config.yaml",
"p0/P0-REPORT.md","p0/P0-MEASUREMENT-ERRATUM.md","p0/monitor/closure-ledger.md","p0/monitor/paper-redteam-round1.md","p0/monitor/paper-redteam-round2.md","p0/monitor/2026-09-22-paper-redteam2-build.json"]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();assert not a.output.exists(),"Choose a new snapshot filename"
 files={PAPER/n for n in STATIC}
 for directory in ["figures","results","notes","scripts"]:
  files.update(f for f in (PAPER/directory).rglob("*") if f.is_file() and "__pycache__" not in f.parts)
 for r in json.loads((PAPER/"notes/evidence-sources.json").read_text()):
  f=REPO/r["path"];assert hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"],f;files.add(f)
 rows=[{"path":str(f.relative_to(REPO)),"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(files)]
 companion=PAPER/"artifacts/e7a-evidence-20260922T0700Z.tar.gz"
 manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"private review checkpoint; E2/E7b qualification and E1 remain incomplete", "files":rows,
 "companion":{"file":companion.name,"sha256":hashlib.sha256(companion.read_bytes()).hexdigest(),"manifest":companion.name+".manifest.json"}}
 readme="""# Private LumoFlyWheel v2 review checkpoint

This checkpoint contains manuscript sources/PDF, the explicit historical evidence needed to reproduce its accounting, review reports, and a SHA-256 manifest. It is NOT a public release and has not been uploaded. E2/E7b qualification and E1 remain incomplete; read the closure ledger for current limits. The cited historical GitHub revision identifies historical evidence, not the unreleased fresh v2 experiment files.

From the extracted repository root:

    python3 papers/gdn-tree-scan-mlsys/v2/scripts/audit_evidence.py
    cd papers/gdn-tree-scan-mlsys/v2
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first command reproduces archived proxies, not matched throughput. It runs no inference. Build requires LaTeX (IEEEtran, TikZ/PGFPlots and standard packages listed in main.tex); the included class and bibliography support an offline build. PDF byte identity is not expected across TeX versions/timestamps.

The companion `e7a-evidence-20260922T0700Z.tar.gz` and its manifest contain original completed E7a sources, result JSON, frozen records/erratum, provenance records and captured tensors. Extract that archive into a separate evidence directory; archive entries are relative to the original v2 directory. All original absolute provenance paths are retained unchanged and refer to the DGX, so execution requires an explicit local path mapping. This package certifies copied bytes and enables raw-result checks; it does not claim an unmodified full GPU rerun on another host. JIT caches/model weights are excluded; recorded hashes are retained. Keep the companion private pending a separate release/provenance review.
"""
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(a.output,"w:gz") as t:
  for f,r in zip(sorted(files),rows):
   assert hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"]
   t.add(f,arcname=r["path"],recursive=False)
  for name,data in [("REVIEW-CHECKPOINT.md",readme.encode()),("MANIFEST.json",(json.dumps(manifest,indent=2)+"\n").encode())]:
   info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
 print(json.dumps({"file":str(a.output),"sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"files":len(rows),"bytes":a.output.stat().st_size}))
if __name__=="__main__":main()
