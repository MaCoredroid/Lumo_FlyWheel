# Minimal SGLang allocated-state metadata hook

Bounded source guidance only; no implementation or execution. Pinned source root: `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-config-v32-bounded-20260928T1720Z/repo/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/inspections`. Model runner SHA256 `d2924c79228e58bb917926063f81a9b16dfaafc177652f7862e8430cd3e54701`; memory pool `25ff2309585909a2589f1bf7ca094453b13785c054ab30b416a76167fca1069f`; HTTP server `a38c13852efd3a93715c1bee55daeb05d9fad1a7c358c461d05f74107d8c28dc`.

**Hook the end of `ModelRunner.alloc_memory_pool` (`model_runner.py:807–828`), not `initialize` (`:623–665`).** The latter loads the model and resolves dtype; the former assigns the returned live pools to `self.req_to_token_pool` (:818) and `self.token_to_kv_pool` (:819), saves the unified backing (:826), and performs post-pool component wiring (:828). A metadata-only hook after the original method returns can inspect actual tensor objects without reading tensor contents or invoking CUDA.

For the hybrid target, inspect these source-bound paths:

| State | Actual object path | Binding source |
|---|---|---|
| Attention K and V | `self.token_to_kv_pool.full_kv_pool.k_buffer` and `.v_buffer`, each a list of tensors | `memory_pool.py:3555–3603,3632–3644` selects the full-attention pool; `:1950–1951,2056–2092` binds/allocates the buffers |
| Convolution | `self.req_to_token_pool.mamba_pool.mamba_cache.conv`, a tensor list | `:1136–1206` constructs the Mamba pool; `:335–337,771–782` defines/binds actual state |
| Recurrent SSM | `self.req_to_token_pool.mamba_pool.mamba_cache.temporal`, a tensor | same source |

`self.token_to_kv_pool.mamba_pool` is the hybrid KV pool's reference to the same Mamba pool (`:3591`); use it as an identity consistency check rather than double-counting storage. Record object class, each tensor `.dtype`, `.shape`, role (`self.is_draft_worker`, runner :315), rank fields under `self.ps`, PID and boot/attempt identity. Do not call `.cpu()`, `.item()`, `.tolist()`, `data_ptr()`, reductions, CUDA APIs, or synchronization. No actual weight checksum is needed for this narrow cache-dtype receipt.

Separate target and EAGLE draft records. The model runner carries `is_draft_worker`; a role lacking recurrent layers must report that absence against its actual layer/pool type instead of borrowing target state. Likewise fail as unsupported on an unrecognized pool type, rather than guessing tensor paths. The active BF16 route should observe BF16 attention storage; report logical and physical dtype separately if a different quantized pool is unexpectedly encountered.

**Phase boundary:** this is allocation-stage evidence, not proof a task request ran. `alloc_memory_pool` explicitly excludes backends/CUDA graphs (:808); `_init_post_memory_pool_components` must precede decode graph initialization (:833–835). The MHA pool's copy warmup (`memory_pool.py:1851–1854,1901–1910`), CUDA graph/profile forwards, and SGLang's HTTP startup generation (`http_server.py:410–415,2284–2294,2359–2366`) are startup activity, never measured agent traffic or proof of the later named probe's route. Begin named-probe counters only after startup readiness and settled baseline; take measured workload counters after the named probe.

A concrete layout caveat: `post_capture_active` may allocate/rematerialize backing after capture (`memory_pool.py:1918–1951`). If active, do not describe the early hook as final serving backing; re-observe metadata after materialization through the real initialization path, or refuse that unbound case. Retain each allocation receipt with a sequence/phase, never overwrite an earlier allocation and call it final. This requires no extra model request or workload attempt.
