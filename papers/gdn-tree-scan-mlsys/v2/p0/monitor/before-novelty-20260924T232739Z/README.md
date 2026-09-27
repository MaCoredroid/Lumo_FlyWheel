# GDN Tree-Scan v2 — current production method

24 September 2026.

The paper describes the latest deployed NVFP4 Hydra27/fixed32 method located in the production history: forked FA2 GQA-pair split-K4, spine-first KV slots with matching mask columns, full-vocabulary single-logits drafting and fused top-three selection, two-level GDN path scans with transient cut states, fixed32 device acceptance, and one captured committer replay containing 48 native per-layer GDN updates. Prefix caching and serving/drafter/committer graphs are enabled. Query row and position order remain logical. The single-launch GDN and Hydra31 candidates are not the executed workload method.

The September 22 Cat10/E2/E7/E8 jobs revisited a superseded route. Their original data, adverse outcomes and dated analyses remain intact in the archive, but all their numerical results and current-method qualification claims are removed from the paper. A later run date does not make an old implementation current. The current scan and native committer are not claimed to share one update body or to be bit-identical by construction.

Current validation is bound to the deployed binaries: fused-selection parity over 6,840 configurations and 24 graph replays, and the split-K numerical credential with 16 determinism cases across two processes and all nine declared checks passing. The attention probe uses synthetic tensors at scales measured from model operands; it is not a captured-task/full-model equivalence test. Runtime receipts positively identify the engaged attention arm and captured device route.

Among recorded run-pairs producing nonempty patches on both shared SWE-bench Verified Astropy tasks, the best tree run achieves 29.09 pooled tokens/s versus SGLang EAGLE's 26.89 (+8.18%). The retrospective rule applies symmetrically, retains failed-test patches, and excludes the earlier empty-patch SGLang pair from performance while retaining its failed outcome. Other eligible tree runs remain 28.28 and 27.27; the separate latest completed ten-task Cqc10 segment remains 25.63. Best recorded rate and latest completed deployment are explicitly distinct.

Reproduce the workload selection and current-component audit without inference:

```sh
python3 results/agent-workload/patch_producing_rate_reduce.py
python3 results/current-production/reduce.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The outputs must match `results/agent-workload/patch-producing-rate-audit.json` and `results/current-production/audit.json`. The former retains 94 input bindings plus five safe projections. The latter pins 20 source/receipt/gate files and verifies component identity against the workload deployment. Original task files and prior reducers are unchanged. Superseded experiment companions remain private historical audit material.

Current history audits: `notes/latest-production-design-evidence-2026-09-24.md` and `notes/latest-optimization-supersession-2026-09-24.md`. Final source/PDF identity: `artifacts/FINAL-DELIVERY.json`. No inference, publication or push performed.
