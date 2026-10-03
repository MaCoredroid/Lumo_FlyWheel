# Lumo CPU source-result review — 2026-09-29 UTC

**Disposition: accept the 34 retained source files as bounded evidence from the declared prospective patch profile. Do not call this a complete post-patch source inventory: one reported modified file, `model_executor/layers/batch_invariant.py`, was not exported. No qualified boot, runtime equivalence, or gate approval follows.**

Review used local source/receipt reads, AST extraction, and hashes only. No patcher import/application, launcher, Docker, SSH, network, GPU, model, or new preparation execution was performed. No scientific source or gate was edited.

Run: `experiments/review-response-20260927/workload-plan/inspections/cpu-lumotree-source-20260929T010800Z/`.

## Exact evidence

| Artifact | SHA-256 |
| --- | --- |
| Campaign `tools/cpu_lumotree_source_prepare_v1.py` | `cf54044f3b97938343a23ae62b04536233770798fcb082ea780ec7dcdc1abc79` |
| Run `driver.py` | `8fe1b951d0b20877ad1062f9aa1b0546f5ff299e71f0f78b040ccf0905996b3d` |
| `CREATE-ARGV.json` | `c7fbaf809f188ef8b2e8a6c71bf97a200e40ec1a1c55f73b3f25df0e17c59020` |
| `BEFORE-START.json` | `0220fab12c780ba8b37506bbe7b02ecf9f4419f460f8bfc46051d4a70b118849` |
| `AFTER-EXIT.json` | `2480f00fe9571eafc4b1da9d00b1599685cbfb04f14315de50dc409f495f82d7` |
| `RESULT.json` | `8882d9cf41722e86e1caa277c4ea8303c27be8d3b5ce36a4682f23d40b43f01e` |
| `result/MANIFEST.json` | `05e7c1c0e478813a20a466df2b5849a40604c314e4fe5bb1f9a5af4dca26998a` |
| `result/PATCH-ENVIRONMENT.json` | `b02d8e142cc503d02bb50a9fd3ba54ac23e8a4d27205258a58215bc907facb19` |

All nine preparation input hashes match local files; the profile's three further source pins also match. The saved driver is byte-identical to the current script's embedded template after substituting its `FILES` literal. All **34 declared unique source paths** exist exactly once in the exported source directory; every file's hash and size matches, totaling **3,216,047 bytes**. Within that declared population, 25 hashes changed and nine remained unchanged. No selected path has a null initial hash. Driver and manifest hashes match `RESULT.json`.

## Actual isolation, execution, and preservation

Both inspections identify CID `461ee6e1ab93194d0a08c42cc0d4f6f5b4321788250bcdff9ece1ce60e4b1f92` and immutable image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. They record `runc`, network `none`, no devices/device requests, nonprivileged mode, all capabilities dropped, no-new-privileges, two CPUs, 4 GiB memory and memory+swap limit, and 128 PIDs. The only binds are the repository at `/workspace` and fork at `/tmp/fr13_fork_fa2.so`, both read-only; there is no model mount. CUDA visibility is empty, NVIDIA visibility is `void`, and the driver checks for absent NVIDIA/nvhost device nodes before preparation.

The driver invokes exactly three commands, in order: original tree patcher; original NVFP4 LM-head patcher; derived case-study FA2 patcher with `--skip-source --fixed32-query-tile32-b1-tier-b-serve`. The third script is the declared task-derived source, SHA `5ac584f7a6726427132fdb3cfa0e57c366476669b6632a362d2a1afe07914874`; this audit does not mislabel it as unmodified upstream code. No full launcher or serving command is present. The supplied fork is mounted but this script does not install/load or qualify its binary. `--skip-source` avoids the extension source-build path; the recorded output concerns installed Python edits.

All three patch stderr files and overall stderr are empty. Each command must return zero before the next command/export; the parent process exited zero, with no OOM, PID 0, and exited state. Copy-result return code is zero. The three emitted `/logs` files were preserved: `fr13_dfwd_split.flag` is `1`, `fr13_fixed32_mode.flag` is `hydra27_fixed32\n`, and `fr13_apc_env.flag` contains the three declared APC bridge values, all `1`.

Actual host interval is **01:08:59.732157–01:09:02.787527 UTC**; container execution is **01:08:59.837473557–01:09:02.652933348 UTC**. The `010800Z` identifier is not a measured start time. Cleanup source rechecks exact CID and its owned label and performs checked removal on that CID only. `RESULT.json` records `exact_cid_removed=true`. Removal stdout is not separately retained, so this review accepts the recorded checked-cleanup result and does not claim an independent current-host absence query.

## Profile correspondence and scientific scope

The manifest binds prospective profile `d2f9e0b7aa22d563003b1295c011e6e74efdb3c07de4eae211cb84640ee91f86`. All **506 profile fields** exactly match the **525-field** recorded patch environment. The extra values are the retained image/process environment, including the recorded container hostname, `/root` home, and `C.UTF-8` locale; overlapping base-image values agree with inspect. All three intentionally absent fields remain absent: `FR10_ALLOW_LINEAR_FALLBACK`, `NUM_SPECULATIVE_TOKENS`, and `MAMBA_BLOCK_SIZE`.

The recorded environment keeps `FR14_REQUIRE_NVFP4_LMHEAD=1`, and the exported model source contains the REQUIRED sentinel. It preserves `FR13_DRAFT_VOCAB_K=0`, `PYTHONPATH=/workspace/src`, and the writable private preparation cache prefix. This is consistent with the prior bounded source-profile audits and does not invent a patch-time speculative-token value of 31 where production forwarding omitted it. The supervisor's no-`torch`/no-`vllm` assertion is explicitly supervisor-only: a patch helper may import CPU torch. CPU-only must not be restated as “no torch import.”

The result labels `prospective_profile_only=true` and `source_only=true` correctly. It demonstrates these file transformations under the recorded profile, not equality to every future full-launcher environment, sidecar, credential, later patch/install step, worker import, or runtime allocation. Existing `.lumo.local.env`/launch-time overrides were not observed by this preparation. Actual runtime source and sidecar matching remain necessary before allocation; this artifact cannot replace them.

## F1 — one modified source is outside the retained population

`result/command-2.stdout` reports `batch_invariant.py: True`. The derived FA2 script at lines **10543–10544** maps that transformation to `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/batch_invariant.py`. The preparation script's `FILES` list at line 30 omits it; neither the 34-row manifest nor `result/sources/` contains it.

Therefore “25 changed” is a count within the selected 34, not the number of all modified files. The successful run and retained 34 hashes remain valid. Before claiming complete patch-output coverage or binding a policy that requires this module, separately retain its exact resulting bytes/hash through a parent-owned source-only correction and preserve this original omission. Do not infer its hash from the unchanged image, from another route, or from the patcher's Boolean output. This review requests no model/GPU experiment or full launcher execution.

Recorded model loads, GPU actions, and workload attempts remain zero. No launch readiness, numerical qualification, task quality, or performance claim is supported by this source-preparation run.
