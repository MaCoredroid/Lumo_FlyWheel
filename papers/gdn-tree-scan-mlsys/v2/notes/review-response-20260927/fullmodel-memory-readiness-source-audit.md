# Full-model memory readiness — bounded source/protocol audit

**Conclusion:** low host `MemFree` does not prevent the frozen v2.7 runner from performing a newly authorized query-only capacity measurement. It **does** prevent the currently frozen model launcher from booting. These are separate decisions. The host floor is an explicitly frozen **operational launch guard**, not a recurrent-state/logit numerical criterion. This note authorizes neither a live query nor a guard change.

Scope: local source and existing saved evidence only. No remote command, process/container/GPU operation, cache action, live probe, source edit or gate change occurred. Earlier accepted query lifecycle/receipt closures are not reopened.

## Exact identities

All seven executable/policy bindings below match `FREEZE-Q1-CODEX-LAUNCH-REPAIR-v1.json` (SHA-256 `e87b98bdd7de6589f878548a7ec973d59e859c0f13d9d85b13cfc204b38e0f77`). Paths below are relative to the paper v2 directory unless otherwise noted.

| Item | SHA-256 |
|---|---|
| `experiments/review-response-20260927/tools/memory_recovery_v2_7.py` | `db30c0db380adedcf32a461f36dc7b8242fa2e08c6cfe24390535387f195adfa` |
| `tools/run_q1_candidate_stage1_v2_4.sh` under the same campaign | `5b91f7183ea227073a768af8efe0c325f4a440c08b26791a819bb1c17955332b` |
| Generated `tools/generated/fr14_leg3_launch_nomiddleware.q1diag.v3_2.sh` | `db89386fb76db8594c00b575bddf369f6f8b05a6183f425734d1679f81de5352` |
| `tools/q1_make_diag_launcher_v3_2.py` | `73387f21c699541ab7113a8b4541e0c72093d41df7b22a40e4dd2bf6c235212b` |
| Repository `scripts/fr14_leg3_launch_nomiddleware.sh` | `11b604b5fc9e652c9a59840511aea9e0952bd56e817d5f27f0ab9bb66bae1313` |
| `p0/monitor/review-response-20260927/Q1-DIAGNOSTIC-EXISTING-SWAP-POLICY-v1.json` | `49c21d7df8e6e47b8ed202fd68c6788d5d890c94818fbae2e5c5133e1cfa48c2` |
| Unchanged `q1_readiness_consumer_v1.py` | `2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9` |

The consumer was read from the accepted immutable `p0/monitor/review-response-20260927/query-only-v27-receipt-repair-reviewed-20260928T202339Z/tools/` copy because the campaign-root consumer is not mirrored locally; its bytes match the freeze. This is not a new remote-deployment verification.

Protocol: query-only proposal v1 SHA `ac4519f4f8e817ede8760101283d173c46efbfb87c00aef8aadbb6a828bde77e`, corrected by v1.1 amendment SHA `7d870c1f5710f91c4479c49f84d3ef99e1b95a6f4eaf64b5385789d5a386c419`. The source freeze explicitly records source acceptance without live launch authorization, requires fresh run-bound query authority, and lists fresh `MemFree>=0.7*MemTotal` among remaining **launch** requirements. Earlier source-only/design approval is not reusable live authority.

## What the current observation establishes

`p0/monitor/review-response-20260927/codex-runtime-20260928T224631Z.json`, SHA `ddfa572638bc7bf7c423b6504f8b11fa932cf8732436ccff70d5747071a13346`, records at 22:46:31Z:

- `MemFree`: 52,083,118,080 B = **48.5062 GiB**; `MemAvailable`: 112,373,149,696 B = **104.6557 GiB**.
- `Cached`: 44.3350 GiB; `SReclaimable`: 3.4720 GiB. These are host accounting, not a measured CUDA-free value or guaranteed reclaim yield.
- Successful empty GPU-compute and container listings. This is their recorded inventory scope, not complete authentication of desktop GPU clients, current /proc target access, or authority locks/holds.
- Existing swap occupancy is approximately 7.8607 GiB. The approved exact diagnostic identity records existing swap instead of applying the production zero-swap refusal; it retains the other memory conditions.

The JSON omits `MemTotal`; therefore the approximately 82.257-GiB host threshold is the supplied/frozen host-budget calculation, not independently recomputed from this particular snapshot. The launcher recomputes it from actual `/proc/meminfo` at launch. The query threshold is separately pinned at exactly **88,326,002,442 B (82.26 GiB)**. Do not substitute the rounded host threshold for that exact byte criterion.

The observation contains **no fresh `torch.cuda.mem_get_info()` result**. Neither `MemAvailable` nor `MemFree` alone establishes what that call would return now. A previous M1 tensor-stage CUDA observation is a different process/time and cannot admit this boot.

## Query-only measurement is not gated on high host MemFree

In `memory_recovery_v2_7.py`:

- Lines 1600–1642 authenticate authority, acquire the campaign lock, check holds/unused marker, verify pinned host/image identity and inventories, perform the read-only target checks, capture counters, recheck authority, reserve the one use, and dispatch `PROFILE_QUERY_ONLY` to `_query_only_readiness`. There is no host-MemFree lower-bound comparison in that dispatch.
- `_targeted_precheck` and `_targeted_recheck` (2401–2454) authenticate held target descriptors/path continuity, record actual `mincore` residency, demand complete privileged /proc coverage and enforce Dirty/Writeback limits at the final recheck. Residency is reported; cold pages are not required. These checks remain necessary and cannot be inferred from the 22:46 snapshot.
- `_query_only_readiness` (2710–2726) calls `_query_stage` once and records postconditions. `_postconditions_targeted` (2630–2663) requires **MemFree to parse**, unchanged drop-cache counters and preserved clients; it does not require a minimum MemFree value. The text “MemFree / clients” in a failure label is not an extra capacity threshold.
- The measurement (242–244) is `torch.cuda.mem_get_info()` plus device/runtime identity. `_query_stage` (2888–3011) creates/starts one exact owned pinned-image container, validates its raw result, removes it with absence proof and compares actual free bytes with the fixed reservation. Insufficient capacity is terminal; there is no second query or reclaim fallback.
- Query-only profile criteria (3373–3447) require zero advice/workers/slab writes, exactly one create/start, fixed image/device/total, valid integral byte values, exact reservation, source-bound raw evidence, owned cleanup, target continuity and the declared Dirty/Writeback bounds. The general verifier additionally requires a terminal SUCCESS bound to the exact run/gate/authority, with no lock, hold or unfinished reservation.

Thus it is admissible **as a prospective operational measurement under fresh authority**, even when host MemFree is below the model-launch floor. It needs no reclaim or model load and no v2.7 source change for that purpose. It is not a zero-side-effect host read: it starts a short CUDA-query container and CUDA context, then must prove cleanup. This reviewer has not performed it. Issuing one is the parent's operational decision.

All earlier consumed authorizations remain consumed. Use the exact `query_only_readiness_v1` profile, source/image/target identity and `--reservation-gib 82.26` in both config emission and execution. Source default 70.51 is not the query-only budget; that prior configuration mismatch is already preserved. Reserve once before create; any post-reservation failure consumes the authority. No retry follows implicitly.

## The model-launch guard is a separate, real blocker

The wrapper invokes the source-bound readiness consumer at lines 154–156. Consumer `decide` (31–53) asks the bound producer to validate the settled run-specific receipt. It does not calculate host MemFree itself. Successful query verification is necessary on this chosen path, but is not the launcher's only guard.

Generated diagnostic launcher lines 6916–6921 compute `required_gib = GPU_UTIL * MemTotal`; lines 6948–6966 refuse either `MemAvailable<80 GiB` or `MemFree<required_gib`. Its exact diagnostic branch permits and records existing swap, while the production fallback retains zero-swap behavior. The host preflight precedes the model `docker run` (7197). A query-only SUCCESS cannot bypass these checks. With the saved host values, the MemAvailable condition passes and the MemFree condition fails.

The production source at lines 6603–6624 describes this as an early unified-memory startup refusal intended to avoid a doomed engine boot. It is operational protection based on an assumed relationship between host free pages and engine reservation. It is **not** a definition of correct state, tokens, logits, tolerance, task quality or experimental outcome. However, the diagnostic swap policy expressly lists this guard as unchanged, and the final launch freeze expressly retains it. Removing/replacing it now would be a **prospective operational-policy and source change**, not “the unchanged launcher” and not something a passing query implicitly authorizes.

The source's error text suggests global cache dropping, stopping containers or lowering utilization. Those strings are not current campaign authority; consumed cache authorities and fixed settings continue to prohibit treating them as instructions. No such remedy is recommended by this audit.

## Smallest next evidence and decision sequence

1. **Separate capacity measurement from model admission explicitly.** The parent can bind a fresh one-use query-only authority/run/gate to the accepted v2.7 bytes and exact 82.26-GiB configuration, with any subsequent model admission conditional. Record that low host MemFree is not a query precondition. All live identity, source, desktop-client, target-access, Dirty/Writeback, lock and hold checks still must pass. No old successful query or consumed cache authority is reused.
2. **Review the one actual query receipt.** Require retained raw create/start/output/ownership/removal/absence evidence; fixed image and GB10 total; finite integer free bytes meeting the unchanged threshold; zero cache actions; target continuity/residency; consumed marker and terminal no-hold state. Preserve an insufficient/failed result as operational evidence and stop the automatic path. The result is instantaneous availability, not a reserved allocation or boot guarantee.
3. **For the unchanged launcher, require its original conditions at boot.** A passing query plus fresh `MemAvailable>=80 GiB` and `MemFree>=0.7*MemTotal`, unchanged deployment settings and all ordinary bindings permits the already scoped root-only R2 attempt under a fresh parent gate. If MemFree still fails, this exact launcher cannot proceed.
4. **If query passes while MemFree fails, resolve the operational discrepancy prospectively.** Before any alternative admission rule, bind/read the actual pinned image's vLLM startup memory check and the implementation/source identity behind its initial memory snapshot; establish which device total/free quantity is used and that `GPU_UTIL=0.7` and its own refusal remain enforced. The currently mirrored source inspected here does not include `gpu_worker.py`; do not replace that evidence with the production comment's blanket allocator claim. A diagnostic-only amended guard would need a separately versioned policy/generator/launcher, exact source/gate/receipt binding, and focused connected controls proving insufficient/invalid/stale-or-wrong-run receipts refuse, the exact diagnostic branch is required, and production behavior is unchanged. This is a review requirement if the parent chooses a repair, not this note's approval of one.
5. **Model execution must still produce its own outcome.** Any authorized attempt must preserve actual resolved memory/cache settings, startup capacity/profile evidence, OOM/timeout/exit and cleanup records, host/swap observations, and complete state/input/logit seals. Query success or healthy boot alone is not numerical qualification. No claim or setting is changed to make the run fit.

## Waiting

Waiting may coincide with unrelated memory being freed, but the saved inventory shows no owned job or model boot currently in progress whose completion supplies a known release event. Neither the protocol nor reviewed runner contains a passive recovery timer or an eventual-MemFree guarantee. Cache/driver accounting can therefore remain a blocker for an unknown duration; repeated idle polling is not a completion plan. The bounded fresh query is the direct way to replace the present CUDA-capacity unknown with evidence, without an invented high-MemFree prerequisite. A failed query still does not authorize reclaim, pressure generation or lowered settings; a passing query still does not override the frozen launch guard.
