# Worker-observed launcher source review — 2026-09-28

**Disposition: accepted for the bounded prospective source-integration scope after the parent’s corrections below. No remaining blocker in the observation-only launcher delta. This is not boot or workload approval. WP stays closed; measured attempts remain 0/4.**

Reviewed only local source and bounded CPU/shell controls. No generator execution, SSH, Docker, image import, GPU operation, engine boot, request, or gate change was performed by this reviewer. Paths below are relative to `experiments/review-response-20260927/workload-plan/`.

## Final source identity

| File under `launchers/` | SHA-256 |
| --- | --- |
| `make_worker_observed_launchers_v1.py` | `ade33751677f7ee16e95656549c847cbba2265beac82670da818b2d222d44a25` |
| `WORKER-OBSERVED-LAUNCHERS-v1.json` | `97c56f2cd4ecbd680fddd7056b92edda659294b0954a5a79198cf1eb02878718` |
| `ar_agent_serving_v3.sh` | `5424664dbd7861f7478c91f40a8cb0c3369d0b041d77a3880684a2348898d949` |
| `chain_mtp_agent_serving_v3.sh` | `97316911adf682c0c5c178a5fca754eafbba295692a092b4ff44c0d0dd6fad05` |
| `sglang_eagle_owned_v3.sh` | `17f87732bea2827039d64c6258fcb81bcc11118e30bc5cebd5694856785bede0` |
| `lumotree_workload_owned_v2.sh` | `446c6e973e8dfbebf9235831c36492f7b7e54d9cb2fce76f6054db396a1e880f` |

All four predecessor, successor, and diff hashes match the final manifest. Independently recomputing each unified diff reproduces the supplied diff exactly. Removing only the observer preamble/argument guard, observer Docker arguments, added no-bytecode environment, and final-process `PYTHONPATH` activation recovers each pinned predecessor **byte for byte**. Thus the added observation source does not alter model paths, image selection, engine arguments, sampling/seed defaults, tree geometry, patch bodies, numerical settings, or serving capacity settings inherited from those predecessors. This verifies preservation, not independent suitability of every inherited setting for a future run.

## Concrete findings and closures

1. **Superseded Lumo predecessor — closed.** The first generator used `lumotree_workload_owned_v1.sh`, SHA `d9c8fa72eaf67a4f8a59bc90ee250801f7875d1f85fecac893eb0623dae26083`, reintroducing the old host environment instead of retaining the private-sidecar/concurrency prerequisite repair. Final generator line 11 pins `lumotree_workload_owned_v1_1.sh`, SHA `fb50186f8deb9a87ee625714067d1b2efdbd84db3c7a10d7b1c1cefe7a79d681`. Final Lumo line 63 sources host env v1.1, SHA `74b0043ec76d9c5de851207d600e1ffe076e59f3194f1d0516afa636a922d61b`. The inverse-byte check proves those repairs are retained.

2. **Host cache admission — closed.** Initially the preamble accepted a nonempty cache; a temporary-directory control returned rc 0 with a file in `cache/`. The first repair checked only `find` stdout and could accept a failed scan. Final generated lines 13–14 use a checked assignment followed by an emptiness test. Actual final-preamble CPU controls returned rc 0 for an empty cache, rc 3 for a nonempty cache, and rc 3 when an injected shell `find() { return 42; }` returned no output. Nothing in these controls invokes Docker or imports the observer. This checks emptiness before an interpreter could load cached `sitecustomize` or bootstrap bytecode.

3. **Unexpected arguments — closed.** The inherited AR/chain/SGLang parser treated `--dryrun` as ordinary live mode and accepted `--dry-run extra`; Lumo ignored arguments. Final generated line 29 admits zero arguments for Lumo, and zero arguments or exactly one `--dry-run` for the other three. Sixteen actual extracted-preamble/guard controls cover all four launchers with `[]`, `[--dry-run]`, `[--dryrun]`, and `[--dry-run, extra]`; all have the expected result. Unsupported cases refuse with rc 3. Existing launch authorization remains necessary after this guard.

## Mounts, activation, and quoting

The shared preamble uses a quoted Bash array for six mounts: helper code, cache, and policy are read-only; metadata, requests, and installation receipts are writable. It checks the attempt-relative root, rejects comma/colon/newline bind paths, rejects directly symlinked required children/policy, and verifies the policy SHA before building the Docker command. `PYTHONPYCACHEPREFIX` is present at container creation, and `PYTHONDONTWRITEBYTECODE=1` is present for all four recipes.

AR and chain retain `/workspace/src` during their version check and patch/install Python calls, then add `/opt/lumotree-observer/code` immediately before `exec vllm serve` (line 135). SGLang’s recipe directly invokes `python3 -m sglang.launch_server`; the new path is supplied in its container environment, with no new preparation interpreter in that recipe. Lumo retains its original preparation environment; the constant export at line 8354 lies inside the existing double-quoted container command, immediately before its existing final `exec ... vllm serve`. That insertion contains no new expansion, quote, or command-substitution character. Its original surrounding command and serve arguments are preserved exactly.

Independent shell-array rendering of AR, chain, and SGLang recovered all original argument elements after removing only the declared observer additions (59, 59, and 71 original elements respectively). Local Bash 3.2 syntax checks pass those three launchers. Local Bash 3.2 cannot parse Lumo’s **pre-existing** `[[ -v ... ]]` guard; this is not a new syntax failure. The parent reports all four final files passed the remote Bash syntax checks recorded in the manifest. A full real-launcher recording-stub traversal was not performed in this review.

The prospective preparation and bootstrap support the intended cache/identity boundary: `tools/attempt-runtime/probe_boundary_v1.py:68–87` creates fresh directories and hash-bound helper/policy files; `:94–105` verifies actual mounts and startup environment after boot; `tools/runtime-collectors/worker_bootstrap_v2.py:34–48` checks the interpreter’s prefix, no-write state, read-only mount, and continued emptiness. Its activation remains tied to the policy hash. Source hashes inspected: boundary `5d5825a0da31e6ebce17b78f4054949ecc85d1075fac4865f4929ec43aca29f9`; bootstrap `bb4353bde916098c245a9503882b9de33be7aa835048fdea86c38d03024a9cf8`; sitecustomize `2a06b5ca31a9e258652c43009815660a569902b645821b327624b7ce37fb20fb`.

The final manifest appropriately remains `DRAFT_REQUIRES_REVIEW_AND_PARENT_FREEZE`. Final policy/source bindings, complete launcher admission, actual image/mount/worker-installation evidence, capacity checks, and parent authorization remain future runtime obligations. This source review establishes no successful instrumentation boot, numerical qualification, workload outcome, or speed measurement.
