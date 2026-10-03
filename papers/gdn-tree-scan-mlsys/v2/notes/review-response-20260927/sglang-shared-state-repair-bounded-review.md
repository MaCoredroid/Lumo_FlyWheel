# SGLang shared-target state repair: bounded review

2026-09-28. **Accepted for the reviewed CPU/source scope.** The repaired observer addresses the hybrid-draft applicability issue in `sglang-hybrid-draft-factory-conclusion.md`. No new material defect was found in this repair. This is not installation or workload admission approval; WP remains closed and workload is **0/4**. No remote/container/engine/model/GPU/evaluator operations were performed.

Reviewed `workload-plan/tools/runtime-collectors/sglang_worker_metadata_v1.py` SHA256 **`cfb20c7952eb65ce7a35c13ecaae3cd249d56316f3e082374f874cf0c8903a5c`**. Its dependency `worker_metadata_v1.py` remains at accepted partition-repair SHA `97587d1be791c6621afd834a9014d57a3f2f14167e976fbde195268146492dbb`.

The hybrid draft reads its own `full_kv_pool` K/V metadata and requires mapping `{0: 0}`. It records empty allocated convolution/recurrent groups and empty packed allocations. The target's shared Mamba member is represented only by `shared_target_state_reference`, whose relationship explicitly says it is not a draft allocation. Thus `absent_state` describes the draft's own allocated groups; it should not be rendered as “no shared state object exists.”

The installed wrapper adds a target to its registry only after the target metadata receipt persists successfully. It keys the registry by the actual Mamba object and stores only a weak reference plus plain receipt-reference fields. The draft must retrieve that same object, and the referenced attempt, boot and PID must match. Receipt sequence identifies the previously persisted target record under the same attempt/boot/PID naming convention. No target tensors or strong pool reference are stored by the registry.

The repair retains the preceding source-backed limits: metadata-only dtype/shape, the pinned pool classes, no quantized or post-capture backing, and target/draft role separation. It does not change serving flags, numerical settings, warmup, seeds, task order, workload scope or budget. The existing independent target-layer coverage and connected worker-source/receipt authentication requirements remain integration work; this review does not reopen them or claim they have been implemented.

## Retained CPU evidence

Snapshot: `p0/monitor/review-response-20260927/sglang-shared-state-repair-reviewed-20260928T213500Z/`. Execute its `reviewer_cpu.py` with the configured Python and `-B`. `test-log-attempt1.txt` records **17/17 passing controls**: 11 supplied and six independent. The independent controls cover:

- Exact shared-object reference and no draft recurrent/convolution/packed allocation.
- Wrong boot, attempt or PID refusal.
- Dead or unrelated pool-reference refusal.
- Target persistence failure followed by draft refusal, with no emitted successful receipt.
- Pool and tensor release while the installed class/registry stays alive, with garbage collection disabled.
- Unexpected hybrid-draft attention-map refusal.

The supplied controls additionally exercise target-not-yet-observed refusal, target-first/draft-second persistence, unrelated pools, target allocation failure, dtype mismatch, post-capture and quantized backing, packed shape mismatch, and mapping collision. Only fake objects and local temporary files are used; subprocess/network entry points are blocked. No engine or Torch module is imported.

Reviewed test file SHA256: `2b808d995c37a35a3f55b3da977d047dd66fa329f809795479a5daa2ad7ed930`. The snapshot manifest binds all reviewed source/test bytes, independent controls, their output and this note. Earlier failed-source snapshots and review records remain unchanged.
