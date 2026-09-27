# Latest-production manuscript red-team — 2026-09-24

**PASS for the reviewed current-production scope. No unresolved material source, supersession, or component-evidence finding.** No additional experiment is indispensable to the design/component/workload claims as now bounded. This is a source and evidence review, not new inference or PDF visual QA.

Reviewed the complete current `main.tex` and all five reachable inputs: abstract, three TikZ figures, and the agent-workload case. Also read the new current-production input manifest/reducer/audit, original component JSON, actual Cqc10 environment and engagement/boot receipts, and the production history described in the companion audit. The earlier numerical Cat10/E2/E7/E8 sections and claims are absent from the current empirical narrative; the remaining archive language is generic and carries no superseded qualification claim.

## Closed findings

- **Physical layout:** main lines 120 and 208 now correctly describe permutation of KV write slots and ancestry-mask key columns, retaining query-row/position order. This matches the active slot-reorder path, rather than the separate hybrid query-reorder candidate.
- **Probe provenance:** line 255 now explicitly calls the split-K inputs synthetic tensors at scales measured from banked model operands. It does not present them as original workload tensors or a model next-token oracle.
- **Current state lifecycle:** lines 142–180 and the scan figure distinguish transient cut-state exports from persistent request state. They do not claim zero intermediate state traffic, a deployed single GDN launch, explicit scratch burning, or one all-layer replay kernel. The fixed16 native replay graph/48-layer description matches both source and actual boot engagement.
- **Route currency:** the method is the completed Hydra27/full-vocabulary/patched-FA2 production lineage. Later unserved Hydra31 or single-launch/precompute candidates and the September Cat10 diagnostic cannot silently qualify it. FP8 versus NVFP4 does not determine attention backend.

- **Capture boundary:** the final abstract and main line 171 now say captured *recurrent replay*. Convolution/KV publication consume the same path separately. Device acceptance is not assumed captured merely because it uses device tensors; actual `STEP_GRAPH=0` remains explicit in the evidence.

## Evidence recheck

Executed the CPU-only `results/current-production/reduce.py` without file writes. It verified all 20 pinned inputs and reproduced `audit.json` exactly. I also independently inspected the underlying raw keys and source call sites:

- Fused selection: 1,368 input cases, 6,840 block-setting configurations, zero byte mismatches, powered negative controls; 24 captured four-level graph replays, zero mismatch. The original pinned selection binary matches the actual deployment environment. Lines 196 and 253 retain the two-output/tie-convention and component-only scopes.
- Split-K: exact deployed binary and patcher/credential identities match. Sixteen determinism cases, two processes, repeated digest agreement; 93.30657958984375% within two ULP, maximum absolute difference 0.00390625, maximum LSE difference four ULP, all nine checks pass. Lines 255–257 correctly distinguish self-determinism from equality to the incumbent and from downstream quality or sampler-law preservation.
- Actual deployment: the engagement receipt binds all sixteen FA2 layers, split count four, FULL graph and no fallback. Actual boot confirms fp32 SSM cache, subtree/fused-conv route and fixed16 native committer. Cqc10 disables single-launch GDN, layer batching and TAW all-parent precompute; the current manuscript does not present those candidates as deployed.
- Agent observations remain separately scoped: pooled completed-request decode rate, selected shared-task run-pairs, retrospective nonempty-patch eligibility, unchanged adverse records, finite trace-check limitations, separate ten-task cohort and no full-benchmark score. This review does not extend those observations to all tasks or attribute a system rate to one kernel.

The remaining full-model sequential/distribution-preservation and broader quality/performance questions are explicitly outside these claims. Keeping them as limits does not erase the concrete verifier design, deployed mechanisms, component checks, or actual agent workload observations.

## Exact reviewed hashes

```text
916eac468da1efc1ecb87e26b05f5e63ff043903deeaf40dd8888a1b8ebf62d7  main.tex
7f8b9ef1febfd72f2706ef0085b7932570fd1822809b232a05be456fc001b633  abstract.tex
cb2b80e88be09e2fe12924d0b3f276d33924502a7c14a033bca79f06a1201f2b  results/agent-workload/case-study.tex
aaf26c57b55eda1f2842501172043dc6e89f592d369edd4e076fe4297cbc8e84  figures/pipeline.tikz
50d1d7f5d30f4a1ed79ee5634299028d352765a68089bef2c340431d03b46898  figures/state-contract.tikz
19c984883a8b70cc5209c33fa4f62e98aba708ad2b025568d1b2d600a3e4ecd0  figures/scan-replay.tikz
8086b8aa2d682cf566f05daac044bc16280174d05b5fa4536a1830ca2ee43832  results/current-production/inputs.json
59591b49310a4e713f700059c27fc74e81637d478974dd522867563eaceba539  results/current-production/reduce.py
2d0904d5a6c77e0fa0f56a9850dbc0cce3cad4898227a3f78770758edc93c4c2  results/current-production/audit.json
2f5abb1eaf2a50b3365247d0295bc758659ee092491e91ba71dae50eb0adbd63  notes/latest-production-design-evidence-2026-09-24.md
```

Final narrow recheck: removal of the Cat10 name from the introduction/archive-exclusion sentence changes no mechanism, result or scope. PASS remains at the main hash above.

Final whitespace-only hash recheck (2026-09-24): current `main.tex` SHA-256 `4793cefe3803d70e817031cb02b743d66c5c5d0e177ccd2c6356708f5d154894`. Restoring exactly one trailing ASCII space before the newline at line 217 in memory reproduces reviewed SHA-256 `916eac468da1efc1ecb87e26b05f5e63ff043903deeaf40dd8888a1b8ebf62d7`. Abstract remains unchanged. No semantic or visible change; PASS remains.
