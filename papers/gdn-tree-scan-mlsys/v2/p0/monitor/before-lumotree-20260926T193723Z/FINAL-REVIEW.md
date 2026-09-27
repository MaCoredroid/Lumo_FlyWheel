# Current production method revision — 24 September 2026

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

The prior attention-name correction was insufficient: it accurately named an obsolete route but left that route central to the paper. This revision supersedes that framing. Current build and visual verification are recorded in `p0/monitor/2026-09-24-current-production-build.json`; source reviews and the final extracted-package verification are bound in `artifacts/FINAL-DELIVERY.json`.

## Novelty review — 24 September 2026

The current claim set has now undergone primary-source review and two independent mechanism reviews. See `notes/novelty/novelty-review-2026-09-24.md`, `claim-matrix.csv` and `source-register.json`. Direct overlap requires attribution to SpecLA, Trees from Marginals and Snakes and Ladders; attention/drafting/caching precedents were added. The paper is positioned as an implemented systems design with scoped component and workload evidence. A new general verifier, replay mechanism, chain schedule or split-K algorithm is not established. The most specific layout intervention remains a candidate for controlled current-build validation.

This review did not change workload arithmetic or run inference. Earlier implementation-audit PASS statuses are not novelty approvals. The dated review reports describe their exact input snapshots.


## Figure 1 revision — 24 September 2026

The overview now uses a full-width vector diagram with an explicit candidate tree, branch-local target computation, selected path, pending token, and separate recurrent/convolution/KV commitment. The working-revision footnote has been removed. Novelty attribution and all measured results are unchanged. See `notes/novelty/figure-and-novelty-integration-review-2026-09-24.md` and `p0/monitor/2026-09-24-figure-build.json`.
