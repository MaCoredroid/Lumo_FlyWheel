#!/usr/bin/env python3
"""Build a private review checkpoint from explicit manuscript inputs and hash-bound audit evidence.
Not a public release. Raw tensors are in separately hashed companion archives. No network/inference.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile,io
PAPER=Path(__file__).resolve().parents[1]; REPO=PAPER.parents[2]
STATIC=["main.tex","abstract.tex","ref.bib","IEEEtran.cls","main.bbl","main.pdf","README.md","review-experiments.md","paper.config.yaml",
"p0/P0-REPORT.md","p0/P0-MEASUREMENT-ERRATUM.md","p0/monitor/closure-ledger.md","p0/monitor/paper-redteam-round1.md","p0/monitor/paper-redteam-round2.md","p0/monitor/2026-09-22-paper-redteam2-build.json","p0/monitor/2026-09-22-review-14-frozen-audit.json","p0/historical-run-manifests.json","p0/model-weight-hashes.json","p0/capture-inventory.json","p0/remote-readiness.json","p0/fa2-load-check.json"]
CURRENT_REPORTS=["validation-redteam-round18.md","e1-measurement-redteam.md","e2-minimal-qualification-review.md","e2-closure-fixture-redteam.md","e2-fixture-final-recheck.md","artifact-redteam-round1.md","2026-09-22-paper-review18-build.json","2026-09-22-paper-review18-kv-build.json","2026-09-22-paper-review20-build.json","2026-09-22-e7b-artifact-checkpoint.json","e2-policyB-b1-redteam.md","e2-b4-final-redteam.md","e2-b4-api-inversion-redteam.md","e2-b4-duplicate-path-review.md","e2-closure-decision.md","paper-selected-route-redteam.md","e1-campaign-launch-redteam.md","2026-09-22-paper-review23-build.json","2026-09-22-selected-route-artifact-checkpoint.json"]
DEFAULT_COMPANIONS=["e7a-evidence-20260922T0709Z.tar.gz","e7b-diagnostics-20260922T0801Z.tar.gz","selected-route-20260922T0930Z.tar.gz","e1-timing-20260922T1225Z.tar.gz"]
CURRENT_REPORTS += ["e1-campaign-final-redteam.md", "e1-campaign-launch-approved.md", "e1-first-native-cell-redteam.md", "paper-whole-draft-redteam-pre-e1.md", "2026-09-22-paper-review28-build.json", "e1-parent-source-checkpoint1001.json", "e1-reply26-parent-stub.json"]
CURRENT_REPORTS += ["e1-native-qualification-redteam.md", "e1-first-tree-timing-redteam.md", "e1-seed-scope-paper-review.md"]
STATIC += ["experiments/e1/E1_ASEXECUTED_DEVIATIONS.md"]
STATIC += ["experiments/STATUS.md", "issues/2026-09-21-v2.csv"]
CURRENT_REPORTS += ["e1-remaining-cells-redteam.md", "e1-remaining-cells-redteam.json", "e1-methods-candidate-redteam.md", "paper-final-e1-redteam.md", "e1-artifact-final-redteam.md", "2026-09-22-e1-artifact-checkpoint.json", "2026-09-22-paper-final-build.json", "2026-09-22-final-source-qa.json", "2026-09-22-memory-cleanup-final.json", "2026-09-22-claude-results-e1-campaign.md"]
STATIC += ["FINAL-REVIEW.md"]
CURRENT_REPORTS += ["historical-numbers-removal-redteam.md", "2026-09-22-historical-removal-edit.json", "2026-09-22-historical-removal-build.json"]
STATIC += ["plan/2026-09-22-system-design-reframe.md", "issues/2026-09-22-system-design-reframe.csv"]
CURRENT_REPORTS += ["design-reframe-redteam.md", "2026-09-22-design-reframe-build.json"]
STATIC += ["issues/2026-09-22-latest-design-e8.csv"]
CURRENT_REPORTS += ["paper-latest-design-e8-redteam.md", "paper-e8-results-redteam.md", "e8-timing-final-redteam.md", "e8-artifact-final-redteam.md", "e8-artifact-final-independent-review.json", "2026-09-22-e8-artifact-checkpoint.json", "2026-09-22-latest-design-e8-build.json", "e8-postcampaign-memory.json"]
DEFAULT_COMPANIONS += ["e8-single-logits-20260922T2255Z.tar.gz"]
CURRENT_REPORTS += ["2026-09-23-agent-abstract-build.json"]
CURRENT_REPORTS += ["2026-09-23-history-correction-build.json"]
STATIC += ["issues/2026-09-23-agent-workload.csv"]
CURRENT_REPORTS += ["paper-agent-workload-redteam.md", "paper-agent-metrics-redteam.md", "2026-09-23-agent-workload-build.json"]
CURRENT_REPORTS += ["2026-09-23-pooled-decode-build.json", "2026-09-23-pooled-speed-claims-audit.json", "2026-09-23-pooled-speed-claims-audit.md", "paper-pooled-speed-redteam.md"]
CURRENT_REPORTS += ["paper-best-shared-task-redteam-2026-09-24.md", "2026-09-24-best-shared-task-build.json", "2026-09-24-best-shared-task-speed-audit.json"]
STATIC += ["issues/2026-09-24-best-shared-task.csv"]
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
 ap.add_argument("--companion",type=Path,action="append",help="Explicit companion archive; repeat to replace the default list")
 ap.add_argument("--scope",default="private best shared-task pooled-decode revision; best Sr12 29.09 vs SGLang 26.89; later runs and ten-task cohort retained; different revisions and caps disclosed; separate numerical qualification; no causal task speedup or public release claim")
 a=ap.parse_args();assert not a.output.exists(),"Choose a new snapshot filename"
 files={PAPER/n for n in STATIC}
 files.update(PAPER/"p0/monitor"/n for n in CURRENT_REPORTS)
 files.add(PAPER/"artifacts/README.md")
 for directory in ["figures","results","notes","scripts","p0/stock-image","p0/monitor/snapshots/20260921T2308Z/experiments/e7a"]:
  files.update(f for f in (PAPER/directory).rglob("*") if f.is_file() and "__pycache__" not in f.parts)
 for r in json.loads((PAPER/"notes/evidence-sources.json").read_text()):
  f=REPO/r["path"];assert hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"],f;files.add(f)
 for audit_name in ("shared-rate-audit.json", "shared-task-rate-audit.json"):
  for path,expected in json.loads((PAPER/"results/agent-workload"/audit_name).read_text())["input_sha256"].items():
   f=(REPO/path).resolve();assert f.is_relative_to(REPO.resolve()),path
   assert digest(f)==expected,f;files.add(f)
 rows=[{"path":str(f.relative_to(REPO)),"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(files)]
 companions=[]
 for companion in a.companion or [PAPER/"artifacts"/n for n in DEFAULT_COMPANIONS]:
  sidecar=Path(str(companion)+".manifest.json")
  assert sidecar.is_file(),sidecar
  companions.append({"file":companion.name,"sha256":digest(companion),"bytes":companion.stat().st_size,"manifest":sidecar.name,"manifest_sha256":digest(sidecar)})
 manifest={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":a.scope, "files":rows,"companions":companions}
 readme="""# Private LumoFlyWheel v2 review checkpoint

This checkpoint contains manuscript sources/PDF, dated task evidence plus separate historical audit records, review reports, and a SHA-256 manifest. It is NOT a public release and has not been uploaded. Read MANIFEST.json's scope and the closure ledger for current completion status and limits. The cited historical GitHub revision identifies historical evidence, not the unreleased fresh v2 experiment files.

From the extracted repository root:

    python3 papers/gdn-tree-scan-mlsys/v2/results/agent-workload/shared_task_reduce.py
    cd papers/gdn-tree-scan-mlsys/v2
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The first command reproduces best and later shared-task pooled decode rates and full-cohort context from 80 original task files plus a raw manifest. It also verifies the three safe audit projections. Its output must match results/agent-workload/shared-task-rate-audit.json under the paper directory. The prior 48-input reducer and its immutable output remain available for the earlier comparison. It runs no inference and uses only Python's standard library. Historical proxy reproduction remains separate in scripts/audit_evidence.py and supplies no current performance claim. Build requires LaTeX (IEEEtran, TikZ/PGFPlots and standard packages listed in main.tex); the included class and bibliography support an offline build. PDF byte identity is not expected across TeX versions/timestamps.

The current Cqc10 task case, safe original task records and source-bound metric reductions are included under results/agent-workload; inspect their manifests and omissions. The five older companions listed with archive and manifest hashes in MANIFEST.json contain raw qualification/audit experiment evidence. E1/E8 local-document timing results are excluded from current manuscript performance. The E7a companion contains completed E7a sources, result JSON, frozen records/erratum, provenance records and captured tensors. The E7b diagnostic companion contains both stage-isolated three-boot batches and their raw operands, states, hidden activations, logits, exact source snapshots and reviewed continuation/reducer dependencies. These six diagnostic boots had KV remapping disabled; they do not qualify the corrected serving route. The selected-route companion contains the failed policy A and bounded corrected B1/B4 qualification. The E1 companion contains all 18 original timing cells, raw events/direct API IDs, preflight/qualification records, loaded sources, exact frozen reducer and independent numerical review. The E8 companion retains the first pre-container failure, both qualification arms, all six timing cells, exact generators and original dependencies, raw joins, counters and frozen aggregate. Its extracted helper verifies eight joins and the complete aggregate; E8 is a B1 component contrast with divergent continuations. Read artifacts/README.md and the closure ledger for the retained failures and limits.

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
