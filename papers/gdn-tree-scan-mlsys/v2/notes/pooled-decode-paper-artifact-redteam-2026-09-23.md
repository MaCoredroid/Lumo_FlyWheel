# Pooled-decode manuscript and artifact review — 2026-09-23

## Independent artifact replay

PASS for the reducer dependency set. `results/agent-workload/shared_rate_reduce.py` depends on **48 original inputs total**: 40 Cqc10 inputs under the paper and eight SGLang inputs outside it. Each of the 12 tasks contributes metric pre/post brackets, runner metadata and its evaluation report. Their paths and hashes are enumerated in `shared-rate-audit.json:input_sha256`.

I copied only those 48 verified inputs and the reducer into a fresh temporary repository layout and executed:

```sh
python3 papers/gdn-tree-scan-mlsys/v2/results/agent-workload/shared_rate_reduce.py
```

The default inferred repository root is correct and the output reproduces saved `shared-rate-audit.json` **byte for byte**. No inference, network or additional package is needed. This is an isolated-input replay, not yet a verification of the parent's final archive.

Checked identities:

- Reducer SHA-256: `c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f`.
- Saved audit SHA-256: `91e38415989781a3d2460729f86ed90aa246167c3850d480b03354c84342e9b8`.

## Necessary bundle checks

The initial builder includes the 40 Cqc10 files through the paper's recursive results directory, but its old `notes/evidence-sources.json` does not contain the eight external SGLang files. This was reported to the parent for repair. Minimal sufficient implementation: iterate all `shared-rate-audit.json:input_sha256` entries, verify each current file hash, include the file at the same repository-relative path, and retain the reducer/audit itself. Both SGLang completed-task directories need `vllm_metrics_pre.txt`, `vllm_metrics_post.txt`, `runner_metadata.json`, and `eval/eval_report.json`:

- `results/fr14_nvfp4_port_20260816/sglang16_evidence/run1/swe_out/verified/per_task/astropy__astropy-12907/`
- `results/fr14_nvfp4_port_20260816/sglang16_evidence/r1/swe_out/verified/per_task/astropy__astropy-13033/`

The checkpoint instructions should show the reducer invocation from the extracted repository root. Compare its output with the saved audit after extraction. Existing raw witness/timing archives remain immutable; their historical estimators are not current-paper performance claims.

## Manuscript gate

Pending parent's source-stable signal at this initial record. Required check: all current quantitative speed claims use the single declared pooled formula `(ΣN−R)/(ΣE2E−ΣTTFT)`, with one consistent first-output observation boundary; table/abstract values agree with the saved reduction; Cqc10 ten tasks and SGLang two tasks are identified as different subsets of the same SWE-bench Verified benchmark, supporting a descriptive same-estimator comparison, not a causal speedup. The full source-semantics analysis is in `notes/common-rate-estimator-semantics-2026-09-23.md`.

## Final source-bound closure

**PASS for the reviewed manuscript, common estimator, settings projection and bundle dependency selection.** This section supersedes the initial pending checks above. No remaining material source/metadata inconsistency was found in this bounded review. Parent owns final PDF render and archive extraction/replay; this report does not claim those later operations already happened.

The abstract, case-study equation/table/counters, main-text conclusion and README consistently use `(N−R)/(ΣE2E−ΣTTFT)`. They report tree 25.63 and SGLang 26.89 tokens/s, and the descriptive tree difference −4.69% (unrounded −4.68630779794%). The 10-task/265-request versus 2-task/45-request populations are explicit. The equation states conventional n−1 intervals, including zero arrival gaps within an output batch, and accumulated request time after first output. It excludes between-request tool/agent gaps and makes no GPU-only timing or task-completion-speedup claim. Task-subset and serving-configuration differences are disclosed; both rows are appropriately described as observations on the same named SWE benchmark rather than a paired causal comparison.

The previous inverse-mean TPOT headline (35.46 ms / 28.20 tokens/s) and all-output/E2E rate values are absent from the reviewed current abstract/main/case-study. Audit records may retain them with their original definitions. The abstract environment remains correctly restored.

The builder now iterates every `shared-rate-audit.json:input_sha256` entry, checks repository containment and exact SHA-256, and includes each original at its repository-relative path. All 48 current files match their bindings. I repeated the isolated replay with the final reducer: stdout again matches the saved audit byte for byte. The checkpoint instructions use the correct standard-library-only command. This closes the eight-external-SGLang-input omission.

### SGLang settings claim and repaired artifact projection

`results/agent-workload/sglang-settings-projection.json` is an explicitly labeled, source-hashed, nonsecret projection independently derived from the original files. No raw environment or trace body is copied into it. For both `run1/12907` and `r1/13033`, it verifies and retains:

- Proxy temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, output cap 24000 and auto-continue 0, equal to the Cqc10 proxy receipt.
- Network policy fingerprint `e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4` and its allowlisted network fields, equal to Cqc10 task metadata.
- Actual trace-init `qwen_code_version=0.19.4`, bound to the full original trace SHA without its body.
- Actual model endpoint `max_model_len=262144` and model name, and parsed command settings for FlashInfer, EAGLE 3/1/4 and prefill chunk 8192.
- Source-bound harness launcher settings: Qwen Code driver, concurrency one and agent budget 9000 seconds.

Thus the manuscript's **same requested sampling settings, output cap, Qwen Code version, harness and network policy** statement is supported. This does not assert identical effective sampler implementations or identical serving configurations. The projection is included by the builder's existing results/** recursion, resolving the settings-evidence packaging gap without adding raw environment secrets.

### Final checked SHA-256 identities

- `main.tex`: `2ee3791b9f60bd839ae73a5d81cd106d6725f7192b04de5d85ffc77e70231342`
- `abstract.tex`: `40fbbff240d41c1948a2c669fb1f18d524793dc4f7f97bdd9458ef8086a12798`
- `results/agent-workload/case-study.tex`: `a689968b51b2ff2701c5d9867535f2f054329d1b21e4dd63fbe2d987002b4007`
- `README.md`: `c16b154703aec18c66c8a917fd0c18bfe90272b2ed871b2a6a392a4e8469e8e1`
- `scripts/build_review_bundle.py`: `fdb2db593fa6c6498d2fc06304d55e6d9b48bb122245425db485d54f512fb214`
- `results/agent-workload/shared_rate_reduce.py`: `c201221bff80ff85b104009e9336346c6605f508234f5a68952ad393745cc57f`
- `results/agent-workload/shared-rate-audit.json`: `91e38415989781a3d2460729f86ed90aa246167c3850d480b03354c84342e9b8`
- `results/agent-workload/sglang-settings-projection.json`: `6f9db90fabc5c7310e5e53968255f5c1ebaa9e4fc202ce34fd507b81f860e95d`
