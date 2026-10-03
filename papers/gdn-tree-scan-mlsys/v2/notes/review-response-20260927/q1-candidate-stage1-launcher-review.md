# Candidate stage1 wrapper / diagnostic launcher: independent source review

**HOLD for the wrapper/operational integration.** The diagnostic patch insertion and job/source forwarding are present, but the inherited launch path still performs unauthorized host cleanup/reclaim and the wrapper does not establish the accepted cleanup library's ownership preconditions. Address the bounded items below before a new source freeze/gate. No scientific criteria or kernel credential relaxation is requested.

## Reviewed snapshot

`p0/monitor/review-response-20260927/candidate-stage1-v1-reviewed/PARENT-SNAPSHOT.json`, SHA256 `62520bc0503dfd7d4f822833c1ca38cacb9873bf7559a329cc1d5a475939181c`. Independently checked **49/49** local members against their observed snapshot hashes; zero mismatches. The parent snapshot records 46 freeze matches and the known three mutable members (`launch_env`, `launch_env_renderer`, `tests_hooks`). Those known mutations require the final freeze refresh, not another scientific finding.

| Snapshot-relative source | SHA256 |
|---|---|
| `tools/run_q1_candidate_stage1_v1.sh` | `2f57cf8609be69e6c80e51e1377f99868d54de5449da99c39af945bac06918e2` |
| `tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v1.sh` | `1f776d49111f8c3015fcfd858d19c0bec5128caab562ba168de7d366d6c59ebe` |
| `tools/q1_native_smoke_cleanup_v2_3.sh` | `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524` |
| `tools/q1_make_diag_launcher_v1.py` | `2df7570c3b40940dcf9538193126ceb09a2ffb10d8ac99feae5d1444cee2e338` |
| `identity/launcher-diagnostic.v1.diff` | `cfa48d3cdefd31f78be60d9d05e681f967615b52b2c8be6ab14e90aec1d25735` |
| `tools/q1_patch_candidate_v1.py` | `e1a31b68170f4296e17b6100e09b6fb5772a171c8a6b0c0803bf62773f619dac` |

The parent owns environment semantics and the diagnostic workload label / scientific Tier-B credential decision; the other reviewer owns hooks/hydration semantics. This review covers execution boundaries and inherited operational side effects. It does not approve those other surfaces or a launch.

## Minimal blocking corrections

### L1. The diagnostic branch unconditionally executes legacy global cleanup/reclaim

Generated launcher **6652–6672** invokes:

- `pkill -f sample_dcgm_during_task.py` without proving ownership;
- `recover_host_memory()` imported from the mounted repository;
- `sync` and `sudo -n sysctl vm.drop_caches=3`.

These run on the actual stage1 wrapper path (`bash "$STAGED"`, wrapper **150**) before Docker starts. No operational authorization is consulted. The imported local `src/lumo_flywheel_serving/model_server.py` (**49, 96–121**, SHA256 `12cf68ca8db777db6e8eda1432bc5189231fb50a078de9d2f91200859a8a2632`) defaults recovery ON and invokes a command containing bit-3 cache drops and `swapoff`/`swapon`; it also has no supervisory timeout. Setting its disable environment variable alone does not disable the unconditional outer `sync`/bit-3 call or global `pkill`.

**Repair only the diagnostic branch:** exclude the entire legacy process-sweep / imported recovery / sync / bit-3 block when the exact Q1 diagnostic label is active, recording an explicit skipped disposition. Preserve the original production branch byte-for-byte or equivalently guarded. Any prospective bit-2 reclaim must remain a separately approved one-use operation; do not fold it into this launcher, reuse prior authority, or lower model settings. Keep the subsequent read-only readiness checks. No new experiment is needed to test the exclusion: a shell-command stub must fail if any forbidden mutation is reached.

### L2. Cleanup ownership is not established for a stopped same-name container

Wrapper **75–78** checks running containers only. It installs the accepted cleanup trap at **142–143**. The generated fixed32 branch correctly refuses any already-existing exact name at **6553–6563**, including a stopped container. On that refusal, wrapper **151** calls `finalize`, which invokes the accepted library against `$CONTAINER`; the library can stop/remove that pre-existing object even though this invocation never created it.

The library is unchanged and is **not** being reopened here. Its caller must establish ownership. Restore the native wrapper's fail-closed all-container exact-name absence check **before cleanup is armed**, retain a unique immutable creation CID, and only arm cleanup for this invocation's engine. A refusal before creation must seal evidence without touching an unrelated container. Preserve existing objects on a name collision. Also claim the output/attempt namespace atomically: the current separate `[[ -e "$OUT" ]]` and later `mkdir -p "$OUT"` (**38, 73**) do not enforce the declared `launch_limit=1` against two entrants. An exclusive claim under the same approved run ID is sufficient; no new scheduling system is needed.

### L3. Three actual executable dependencies are absent from the gate's hash set

Wrapper **29–34** verifies 30 dependency keys but omits:

- `scripts/fr13_required_tree_flags.sh`, sourced by the generated launcher at **556**;
- `scripts/gpu_oom_guard.sh`, launched at **8029** on this fixed32 route;
- `tools/q1_reference_hooks_v2.py`, imported by `q1_reference_driver_v2.py:15`, which the candidate driver imports at line 11.

The first two are copied at wrapper **97** and merely recorded in `LAUNCH-BINDING` at **106**. The generated relocation guard's `cmp` proves only that the staged and current repository copies agree, not that either equals reviewed bytes. The third is the old v2 authenticator dependency, distinct from the already-listed v2.1 hook.

**Fix:** add those exact byte hashes to the freeze/gate and validate before execution. This is a small actual dependency closure, not a request to hash hypothetical imports. Observed local helper hashes (not independently checked on the remote host in this review): required flags `74de72c709d5eba3a00146056f2af38c4d8b443b6925c5a8e4feb7fbe8e95e21`; OOM guard `6148206a024c17746bce91c6c3e065275a8d47c24d660c665d7e3d8e29ad265b`; reference hook v2 `78ebde8de667f53e012db41597f6e4ab5970fd8b7c126a0c88f55ac62e6187dd`.

### L4. Prelaunch failures lack terminal receipts; readiness coupling remains incomplete

After output creation at **73**, contention/readiness failures, job-generation failure, staged-copy failure and binding failure occur before `write_receipt`/the EXIT trap are installed at **113–143**. They leave a spent partial run directory without the promised terminal failure seal. The staging command at **97** is not explicitly checked (`set -uo pipefail`, not `-e`); a failed `chmod` can be ignored and a partial copy is only detected indirectly.

Install failure receipt handling as soon as the attempt namespace is claimed, with the no-owned-container state from L2; explicitly check staging. Preserve primary failure and cleanup status through the already accepted finalizer. Initialize finalizer state locally rather than inherit an ambient `FINALIZED` flag.

The current readiness reader (**80–90**) trusts only the latest matching receipt's `boot_hold.next_boot_permitted`; it does not bind that receipt/authorization to the gate, require a successful terminal receipt, or consult `.lock-v2`/unfinished attempt state. This directly intersects the separately reported memory-recovery-v2 authority/seal defects. Couple it to the **settled** recovery hold/receipt contract before enabling a reclaim-dependent boot; do not treat an in-progress/unknown attempt without a completed receipt as “skipped.” A parent-approved no-reclaim boot can record that distinct disposition. No reclaim is authorized now.

## Inherited side-effect census and safe retention

- **Global process kill and cache/swap recovery:** the L1 block is reached and must be excluded for Q1 diagnostics.
- **Unconditional `docker rm -f "$CONTAINER"`:** generated **6566** is in the non-fixed32 `else`. The diagnostic guard requires fixed32 at **5110**, so this deletion branch is not active in the reviewed diagnostic route. Preserve that guard; the distinct wrapper cleanup ownership defect is L2.
- **Owned OOM safety guard:** generated **7997–8017** authenticates a full CID and expected name; **8027–8030** passes both to the guard. The inspected helper's **101–151** rechecks immutable identity before killing/removing only that CID. This is materially different from the global sweep and can remain as a declared owned-engine safety mechanism. Pin its source (L3), put its log in the owned run directory rather than the helper's shared default `output/gpu_oom_guard.log`, and preserve guard identity/disposition. Its fallback kill/removal must never broaden to the name-glob branch. Existing diagnostic fixed32 selection avoids that branch.
- **Run-sidecar deletion/writes:** reviewed `rm` sites at **3333, 4782–6551** target `$LOG_DIR` sidecars, including the fresh diagnostic non-credential file; they do not themselves request unrelated process/host cleanup. Retain them only with `$LOG_DIR` bound to the exclusively owned fresh run directory. The wrapper's **178** deletes only that run's throwaway diagnostic non-credential file.
- **Final engine cleanup:** reuse the accepted v2.3 contract after satisfying L2; it records stop/removal errors and runs only once. No global cleanup should be added to its failure path.

## Confirmed diagnostic/source boundaries and verification limits

The wrapper checks gate type, exact approval/run/scope, image ID, each listed dependency hash and launch limit before attempted boot (**40–71**). The new job SHA, patcher SHA and generated runner/rejection-source hashes are exported at **147–148**. The diagnostic branch verifies/forwards all six Q1 inputs at generated **6791–6815**. Candidate patching runs after the production patch sequence at **7855–7872**, refuses the wrong pre-patch hashes, and precedes provenance/serve. Provenance rechecks job/patcher bytes at **7921–7957**. These are useful existing controls; keep them and the scientific kernel credential checks intact.

Host boot timeout currently starts only **after** the staged launcher returns (**150–156**); it does not supervise that host startup command. Freeze a bounded startup supervisor and route its timeout to the owned-container finalizer when repairing the diagnostic operational branch. Driver HTTP timeout is already bounded (default 3600 s); no repeat/retry is introduced by the wrapper.

Only source reads, hash comparisons and `bash -n` were performed. Local Bash 3.2 parses the wrapper and accepted cleanup library; it refuses the generated script at its unchanged Bash-4 `[[ -v ... ]]` syntax (line 273), so that is a **local tooling limitation**, not a new Linux script defect or a claimed passing generated parse. The supplied Linux test log is author evidence only. The repaired freeze should include the existing Linux parser/outer-inner command-stub test plus the focused L1/L2/L4 negatives; do not run the real launch to validate them.

No SSH, Docker/container, GPU/model/API, process cleanup, sysctl or source mutation was performed. No environment secret values were read into this report or printed. This note is the only file written by this review.
