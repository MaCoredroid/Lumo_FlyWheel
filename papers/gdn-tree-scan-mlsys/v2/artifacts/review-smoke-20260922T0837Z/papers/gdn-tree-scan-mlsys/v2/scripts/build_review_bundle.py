#!/usr/bin/env python3
"""Build a private review checkpoint from explicit manuscript inputs and hash-bound historical evidence.
Not a public release. Raw tensors are in separately hashed companion archives. No network/inference.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile,io
PAPER=Path(__file__).resolve().parents[1]; REPO=PAPER.parents[2]
STATIC=["main.tex","abstract.tex","ref.bib","IEEEtran.cls","main.bbl","main.pdf","README.md","review-experiments.md","paper.config.yaml",
"p0/P0-REPORT.md","p0/P0-MEASUREMENT-ERRATUM.md","p0/monitor/closure-ledger.md","p0/monitor/paper-redteam-round1.md","p0/monitor/paper-redteam-round2.md","p0/monitor/2026-09-22-paper-redteam2-build.json","p0/monitor/2026-09-22-review-14-frozen-audit.json","p0/historical-run-manifests.json","p0/model-weight-hashes.json","p0/capture-inventory.json","p0/remote-readiness.json","p0/fa2-load-check.json"]
CURRENT_REPORTS=["validation-redteam-round18.md","e1-measurement-redteam.md","e2-minimal-qualification-review.md","e2-closure-fixture-redteam.md","e2-fixture-final-recheck.md","artifact-redteam-round1.md","2026-09-22-paper-review18-build.json","2026-09-22-paper-review18-kv-build.json","2026-09-22-paper-review20-build.json","2026-09-22-e7b-artifact-checkpoint.json"]
DEFAULT_COMPANIONS=["e7a-evidence-20260922T0709Z.tar.gz","e7b-diagnostics-20260922T0801Z.tar.gz"]
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--companion",type=Path,action="append",help="Explicit companion archive; repeat to replace the default list")
 ap.add_argument("--scope",default="private review checkpoint; current-route E2 qualification and E1 remain incomplete")
 a=ap.parse_args();assert not a.output.exists(),"Choose a new snapshot filename"
 files={PAPER/n for n in STATIC}
 files.update(PAPER/"p0/monitor"/n for n in CURRENT_REPORTS)
 files.add(PAPER/"artifacts/README.md")
 for directory in ["figures","results","notes","scripts","p0/stock-image","p0/monitor/snapshots/20260921T2308Z/experiments/e7a"]:
  files.update(f for f in (PAPER/directory).rglob("*") if f.is_file() and "__pycache__" not in f.parts)
 for r in json.loads((PAPER/"notes/evidence-sources.json").read_text()):
  f=REPO/r["path"];assert hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"],f;files.add(f)
 rows=[{"path":str(f.relative_to(REPO)),"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(files)]
 companions=[]
 for companion in a.companion or [PAPER/"artifacts"/n for n in DEFAULT_COMPANIONS]:
  sidecar=Path(str(companion)+".manifest.json")
  assert sidecar.is_file(),sidecar
  companions.append({"file":companion.name,"sha256":digest(companion),"bytes":companion.stat().st_size,"manifest":sidecar.name,"manifest_sha256":digest(sidecar)})
 manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":a.scope, "files":rows,"companions":companions}
 readme="""# Private LumoFlyWheel v2 review checkpoint

This checkpoint contains manuscript sources/PDF, the explicit historical evidence needed to reproduce its accounting, review reports, and a SHA-256 manifest. It is NOT a public release and has not been uploaded. Read MANIFEST.json's scope and the closure ledger for current completion status and limits. The cited historical GitHub revision identifies historical evidence, not the unreleased fresh v2 experiment files.

From the extracted repository root:

    python3 papers/gdn-tree-scan-mlsys/v2/scripts/audit_evidence.py
    cd papers/gdn-tree-scan-mlsys/v2
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first command reproduces archived proxies, not matched throughput. It runs no inference. Build requires LaTeX (IEEEtran, TikZ/PGFPlots and standard packages listed in main.tex); the included class and bibliography support an offline build. PDF byte identity is not expected across TeX versions/timestamps.

The companions listed with archive and manifest hashes in MANIFEST.json contain raw experiment evidence. The E7a companion contains completed E7a sources, result JSON, frozen records/erratum, provenance records and captured tensors. The E7b diagnostic companion contains both stage-isolated three-boot batches and their raw operands, states, hidden activations, logits, exact source snapshots and reviewed continuation/reducer dependencies. These six diagnostic boots had KV remapping disabled; they do not qualify the corrected serving route. Read artifacts/README.md and the closure ledger before interpreting any later companion.

Extract companions into a separate evidence directory; archive entries are relative to the original v2 directory. All original absolute provenance paths are retained unchanged and refer to the DGX, so execution requires an explicit local path mapping. This package certifies copied bytes and enables raw-result checks; it does not claim an unmodified full GPU rerun on another host. The historical tensor mapping is recorded in the E7a companion manifest. Byte-exact ladder-v3/tiny-gate core and device sources are in the paper archive under p0/monitor/snapshots/20260921T2308Z/experiments/e7a; the E7a companion contains the matching production kernel in the earlier source_snapshot. Native prefix-selection pool/scores and all eight pilot captures are included. JIT caches/model weights are excluded; recorded hashes are retained. Keep all companions private pending a separate release/provenance review.
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
