# Bounded unfinished candidate v3.2 handoff review

2026-09-28. **Hold for the three integration fixes below; no new scientific experiment or numerical-policy change is needed.** Parent Codex implements and executes; this independent review made no implementation, gate or remote-state change.

## Exact captured source

Snapshot: `p0/monitor/review-response-20260927/candidate-env-v24-unfinished-reviewed-20260928T2050Z/repo/`, 1,130 files / 7,248,420 bytes. It is an authenticated unfinished snapshot, not a freeze. All 1,117 indexed dry-run evidence files match index SHA-256 `85925d2f84b59410e60b8be08c1684549c443fceb4cc11c2cc0626efd507244c`. The log reports 46 passing CPU controls; unchanged suites were not rerun.

| Source | SHA-256 |
| --- | --- |
| Renderer v2.4 | `3f6fd6e7cc7438289042037dd66186d4fbf91bee1843bc3247cf0cc77fe0ff18` |
| Env v2.4 | `0035a9a9cbf41d07509f15f3dceeb36b8b2c0d168c453513fe1cf1ccb095bdd3` |
| Diagnostic generator v3.2 | `73387f21c699541ab7113a8b4541e0c72093d41df7b22a40e4dd2bf6c235212b` |
| Generated launcher v3.2 | `db89386fb76db8594c00b575bddf369f6f8b05a6183f425734d1679f81de5352` |
| Wrapper v2.3 | `f7a180df29c05822cdb46dfb24599d068e9acb9e815ef744b17630ad103fb2cc` |
| Swap observer v1 | `6bb7198878776fb009691b22e490d1f969de26abfb03a148db7561e34cf50880` |
| Connected test source | `ecf6d51b098f63aa9110fd7f9f1e473c84923fb446f7954565ad2861a4f3cad0` |
| Test log | `d60c2a45f1de7166534e56380578f5f11e115dc82fb388f85bb3b38818c066e6` |
| Handoff report | `c8fe202b32e8d9a821cb1ecffde27636e3c3289c73ce6db235e8a9f560472d2f` |

## Minimum fixes before freezing

1. **Render the receipt's TREE.** `q1_candidate_launch_env_v2_4.py:225–235` emits neither TREE nor SPEC_CONFIG; launcher v3.2 defaults TREE at line 3694 and the real topology validator refuses at 4117. Derive TREE from the bound production receipt's `SPEC_CONFIG.speculative_token_tree`, require agreement with sealed serve `speculative_config`, and validate the 31 ordered tuple paths against the pinned Hydra27 topology. Use safe shell quoting. Renderer line 128's restrictive assignment parser currently excludes the tuple-string syntax, so add a narrowly validated TREE serialization case rather than permitting arbitrary shell lines. Verify the launcher-derived complete SPEC_CONFIG equals the receipt. The hypothetical probe already demonstrates this exact tree passes the real validator.

2. **Emit canonical `GPU_UTIL=0.70`.** Renderer line 234 emits float text `0.7`; the fixed32 exact-pair guard at launcher 4667/4727 requires `0.70`. Assert the sealed numeric value is exactly 0.7, then emit `0.70`. Do not weaken the guard or round other values into acceptance. Allocation fraction and the 82.26-GiB reservation remain unchanged.

3. **Bind the active bundle and host validators.** Wrapper v2.3 lines 37–39 still default to env v2.2 / launcher v3.1 / generator v3.1; line 51 still names renderer and launch JSON v2.2. Set successor defaults or an exact final invocation to the repaired bundle. Freeze/gate the selected env, launcher/generator and generator's v3.1 dependency, matching JSON/render provenance, observer, swap policy, and validation runtime. A new host-env hash alone does not describe this bundle.

   The owned `.venv/bin/python` resolves to `/usr/bin/python3.12`, SHA `a7d56a8a764faf7bbf5c164055a48fd072be52287bdeb523a9e07b2042f4e7e1`; topology and contract validators run through it. Separately, host `python3` at launcher 4449–4457 imports `fr13_floor_gate`, requiring NumPy from `/home/mark/.local`. The provisioning record binds interpreter/topology/contract bytes, but NumPy has only origin/version and the actually exercised floor module is not bound as an actual import. Add effective host-python identity, floor source hash and NumPy provider identity (origin/version and package metadata/RECORD identity suffice for this host preparation). Preserve the demonstrated HOME/user-site resolution, or explicitly bind an equivalent arrangement. Do not hide NumPy with a clean HOME, install substitutes, or skip the real validator. Keep `PYTHONDONTWRITEBYTECODE=1`.

## Swap and observation disposition

No further swap-policy change is indicated. ESW1 preserves the production readiness code as its else branch. Its exact diagnostic branch binds label, derived credential/patcher paths and bytes, B1/Hydra27/arm, no-SWE workload and policy hash. It records occupied swap while enforcing the original `MemAvailable>=80GiB` and `MemFree>=float(GPU_UTIL)*MemTotal` arithmetic. The real-host hypothetical control refused insufficient MemFree (81.19 versus 82.26 GiB at that observation); the production control refused used swap. Synthetic sufficient memory proves control flow only, not current resource readiness.

Wrapper line 208 observes healthy state; line 107 observes every sealed terminal state. Together with the launcher's before-boot record, they cover the required boundaries. The observer retains host occupancy, swap I/O, PSI, container PID/StartedAt, exposed process/cgroup facts, and sealed cleanup inspect/OOM/exit state. After removal, live process/cgroup facts may be unavailable and are labelled so; sealed Docker state survives. The healthy/terminal test used a fake CID and a CPU sleep process/cgroup, demonstrating plumbing only. Use the observer-bearing wrapper; the older wrapper lacks these observations. Missing observations or incomplete instrumentation cannot become a qualification pass.

## Remaining recording-stub controls

- Final selected wrapper + final rendered env + actual generated launcher, **without hypothetical TREE/GPU_UTIL appends**: clean and inherited-empty callers reach a refusing Docker-run stub with labelled sufficient synthetic memory. Execute the real topology, contract and floor validators. Assert final argv's tree/SPEC_CONFIG, image, derived credentials, FA2, model, geometry, job/request bindings and unchanged serving values.
- Apply the existing four nonempty private-name controls to that final bundle: each must exit and seal before the launcher, without revealing values.
- Check malformed/missing TREE and non-0.7 utilization refuse at generation/validation; canonical 0.70 passes. Missing required NumPy must refuse, not skip the floor validator.
- Retain the passing exact-identity/production-swap/low-capacity controls; repeat only if ESW1 changes. Retain the healthy/terminal observer fixture, check its final-wrapper binding, and require terminal observation on the final stub refusal.

The full-chain control must distinguish real validators from substituted image-inspect, model-config and memory facts, ending at sentinel rc 97 with nothing created. Actual fresh capacity/readiness and a new one-use parent gate are separate prerequisites. No GPU correctness, timing, workload or performance result follows from these CPU controls. No further polling or live operation was performed.
