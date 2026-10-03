# Lumo workload v2.1 prerequisite review — 2026-09-28

**Disposition: accept this bounded prospective source repair. No concrete defect found in the reviewed delta. This does not approve a launcher run, qualification, workload execution, or a gate change.**

Review used local source reads and extracted assignment/guard CPU controls only. No launcher, generator, SSH, Docker, GPU, model, request, or telemetry writer was executed. All paths below are relative to `experiments/review-response-20260927/` unless stated otherwise.

## Source identity

| File under `workload-plan/launchers/` | SHA-256 |
| --- | --- |
| `make_lumotree_workload_owned_v2_1.py` | `0e7b065def496f5f128e5d80a640cf5c2e97bea9d438b1f8c29bb7a161b925a7` |
| `LUMOTREE-WORKLOAD-OWNED-v2.1.json` | `7f1392976d8de414f2893a669df1b5b376b43510b7110a6de11c5c666884e63a` |
| `lumotree-workload-host-env.v1.2.env` | `c961494bee18b1538494c30aeeccd7b4fd93f58728607d4efe4360c409d125ff` |
| `lumotree_workload_owned_v2_1.sh` | `3d186ac2f44781527f659ac2b4a369d0f8110c7c82f887b8386d846486060943` |
| `lumotree-workload-host-env.v1.1_to_v1.2.diff` | `98c4f1ca3cf97ed3b02bb086d384aecfcb65d9077bbe7d335f011c25c685a38c` |
| `lumotree_workload_owned_v2_to_v2_1.diff` | `59b92a11839ffbc92ec2cb72f9ca8bad62bc74ca7e1e923b96855b1a9c740437` |

All four manifest input hashes and all four output hashes/sizes match disk. Independently recomputed diffs exactly match the supplied diffs. Reversing only the declared changes restores both predecessor files byte for byte: environment v1.1 `74b0043ec76d9c5de851207d600e1ffe076e59f3194f1d0516afa636a922d61b` and observed launcher v2 `446c6e973e8dfbebf9235831c36492f7b7e54d9cb2fce76f6054db396a1e880f`.

## Findings

**The tree is restored from evidence, not inferred from a default.** Generator lines 47–59 compare the complete observed speculative configuration with the archived environment configuration and require the accepted candidate v2.5 environment to contain the identical tree. I additionally checked the original raw files referenced by that candidate receipt: `results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/container_env.SANITIZED.txt`, SHA `15f1e8ea4a9bbc33d8837dfbeddce3e2b9bab62f63411c090b5b8a1da48550d4`, and `docker_full.log`, SHA `97a5431d2890d71d8335979a75b4d9470b4fc22c25c2a511d728459956c31a9d`. The environment's `SPEC_CONFIG` at line 580 and the observed serve arguments at log line 30 agree with both derived bindings. The final assignment contains exactly 31 ordered, unique, nonempty tuple paths, with all proper parents represented and maximum depth 11: 31 physical draft positions plus the root. This is not a claim that every physical position is active or accepted.

Environment line 527 puts that exact literal in `TREE`. It prevents the predecessor's omitted assignment from falling through to the inherited nine-path default. The change restores the intended recorded geometry; it is not byte-equivalent to the incomplete predecessor's fallback behavior. Fixed32 mode, masks, recurrence/attention routes, image/model, and numerical controls are otherwise preserved.

**`0.70` changes representation, not reserved fraction.** Generator lines 60 and 66 retain the receipt's numeric 0.7 and emit the literal required by the existing deeper exact-value guard (`lumotree_workload_owned_v2_1.sh:4742`). The top guard at line 66 is adjusted consistently. Extracting the actual top guard and its required literal assignments gives rc 0 for `0.70`, rc 3 for `0.7`, and rc 3 for `0.60`. No capacity threshold or scientific criterion was relaxed.

**Telemetry relocation is limited to four host values.** The only changed existing assignments are `GPU_UTIL` and the SFWD/CFWD/DFWD/LFWD `*_GPU_TIMER_JSON` destinations; `TREE` is the only added assignment. Unique assignment count is 510 → 511. The existing duplicate `MAX_NUM_SEQS=1` remains identical in both files. All timer enable flags and dump/sample controls are unchanged. Final destinations are distinct constant `/logs/workload_{sfwd,cfwd,dfwd,lfwd}_gpu_timer.json` paths. The retained launcher derives `LOG_DIR=$LUMOTREE_OUT/engine-logs` at line 73 and binds it to `/logs` at line 7173, so forwarded telemetry is scoped to the current attempt. Existing SFWD/DFWD/CFWD forwarding at lines 7242–7247 is unchanged. There is no LFWD forwarding in this delta; the manifest explicitly discloses that the LFWD host-path change does not establish an engine output.

**Shell and generation boundaries remain narrow.** The new tree is limited to digits, spaces, brackets, commas, and parentheses before `ast.literal_eval`, then single-quoted; new path values contain no shell metacharacters. Fifteen extracted actual assignments, including the tree, destinations, and top-guard inputs, round-trip through Bash unchanged. The generator pins all inputs, uses exact single-occurrence replacements, and refuses differing existing outputs. It neither runs the launcher nor accesses the remote host. No new diagnostic label, credential, task identity, timer enablement, or LFWD forwarding is introduced.

## Remaining boundary

The manifest stays `DRAFT_REQUIRES_REVIEW_AND_PARENT_FREEZE` with zero launches and zero workload attempts. Its outstanding requirements remain accurate: qualification, task-bound serving credentials and keyed tables, exact post-patch worker source recipes, production capacity and zero-swap admission, and parent WP freeze. In particular, the workload launcher still requires externally supplied task/ingress identity at lines 70–71; the old diagnostic swap exception is not imported by this repair. The existing observer guards and production launch controls are byte-preserved apart from the reviewed environment binding and numeric spelling. This review does not establish that those remaining admission requirements have been satisfied or that a complete launcher traversal will succeed.
