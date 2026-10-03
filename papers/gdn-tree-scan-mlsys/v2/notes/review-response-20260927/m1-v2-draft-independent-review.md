# M1 v2 draft independent review

**Disposition: the original F1–F3 adapter defects are closed at source level; this draft still needs device canonicalization, independent identity enforcement, and an accounting correction before runtime freeze.** This is a bounded review of the `m1-v2-draft-20260928T0428Z` snapshot, not an executor freeze, numerical qualification or launch approval. No SSH, container, GPU, model or author computational-module execution occurred.

## Identity and checks

Snapshot: `p0/monitor/review-response-20260927/m1-v2-draft-20260928T0428Z/`, SHA-256 of `SNAPSHOT.json` **`e81ff9348ae75b597f85392a76e6a94767a0f5fd218abf123598e4e5fdcf0cc4`**. All **14** members independently match their recorded SHA-256 and byte sizes.

| Reviewed file | SHA-256 |
|---|---|
| `tools/m1/m1_adapters_v2.py` | `a5414791a08f1f526b38f5f86c73cd6cee56c739542be820f785d8a7bd3ffcce` |
| `tools/m1/m1_executor_image.py` | `e3f441940407cd8a5e3e9ce9aaf9eb70dab9a402aec7092f83acbc24a85b60ef` |
| `tools/m1/m1_cycle_driver_v2.py` | `8f22778f65b142e7fd1262f8c3b182ba02eaf30ae7efde7b09b1e97bd21e37f3` |
| `tools/m1/m1_accounting_v2.py` | `bb0a9f22bb88c2ec934591c3bdd6e07c696dd97844d4a680e157018bf9654bfa` |
| `M1-STAGE-INIT-VERIFY-PROPOSAL.md` | `5207e29eb578f6615f6da803a607e4baf1b8dbab044305f43de87c97d4a7ac9c` |

I traced the changed adapters/executor/driver against the existing pinned author corpus and Q1.2b runner. Four safe CPU controls extracted the actual `Ref`, `resolve_refs` and `bind_result` definitions via AST, with a minimal tensor stub: exact returned-query identity is preserved; a missing producer raises; local helper values are copied into and bound as the target; and `out=` binds the caller's output storage. All four passed. No Torch/Triton import was used. All snapshot Python files parse. The snapshot's source-bound test log reports **34 passed**, but that suite was not independently rerun (the local interpreters lack Torch/pytest). Unchanged identity-only modules were not exhaustively re-reviewed.

## Required corrections

### R1 — Canonicalize the execution device before comparing tensor devices

`m1_executor_image.py:39,47` defaults to `device="cuda"` and stores `torch.device("cuda")`, with no index. Buffers/operands allocated on this device report an indexed device such as `cuda:0`. `_dispatch` at `:144–145` compares `torch.device(dev) != self.dev`, so a valid indexed tensor is rejected against the unindexed default. In Weaver this is encountered no later than the topology build that passes parent/tree tensors, before verification. Fixing the former CPU-leaf defect therefore does not make the default executor runnable.

**Repair:** resolve the requested CUDA device to its concrete index once in construction, use that canonical object for allocation/staging/checks, and test the default unindexed request against an indexed tensor device with safe injected device objects. Preserve refusal for a genuinely different device. The original persistent device-leaf changes themselves are correct.

### R2 — Compare runtime/source identity to an independent frozen expectation

`m1_executor_image.py:68–69` compares the runner file hash with `A1.lumo_binding(L)['runner']['sha256']`; the latter is computed from the same current runner file at frozen-v1 `m1_adapters.py:314`. Thus both sides change together and a modified runner passes this check. This does not establish the advertised pin `ab27fc9c…dea29` before constructing `Backend`.

Likewise, executor `:43–44` accepts any syntactically valid image digest, `m1_cycle_driver_v2.validate_v2:406` checks only its hex format, and `reduce_runtime` calls that validator. The unchanged base validator (`m1_receipt_schema.py:42–45`) checks source-hash formatting/runtime-field presence, not equality with the proposed production image. The proposal's statement that these validators enforce only `sha256:ffa30d66…1cdc` is not implemented by this snapshot.

**Repair:** the sealed stage caller/executor must load immutable expected source/image identities independently, refuse mismatch before module/Backend execution, and retain observed-versus-expected evidence. A valid-but-wrong image digest and an altered runner must fail. Parent is reviewing that caller/preflight; this finding does not assert a bypass of the still-closed parent gate.

### R3 — Account for the actual gather/remap temporaries or eliminate them

The former missing operations are now executed inside each logical-step boundary, but `m1_adapters_v2.py:356` uses `index_select(...)` **without `out=`**, then copies its newly allocated result into persistent DFS staging. The inverse remap at `:409` likewise allocates an intermediate result and copies it into `out_phys`. `m1_accounting_v2.py:54–64` lists persistent staging/output plus the unchanged wrapper outputs, but no allocations for these intermediates; `:87–95` presents the resulting sum as cumulative volume over every listed allocation/event and unique live storage. The actual path contains additional allocations and copies.

For the proposed `M=3,L=48`, the omitted logical-step temporary allocation volume is **151,879,680 bytes**: `(661,504 bytes for q/k/v/a/b gathers + 393,216 bytes for output remap) × 3 × 48`, excluding warmup. This is a shape-derived correction, not measured allocator traffic or residency.

**Repair:** use the supported `torch.index_select(..., out=destination)` form with correctly shaped persistent destinations, or list the temporary allocations, lifetimes and subsequent copies explicitly. Retain the distinction between operation payload bytes and actual memory traffic. The untimed initialization stage does not need timing results, but the allocation summary must describe the code it executes.

## Bounded closures verified

- **F1:** Weaver query normalization now produces `result_key="q_norm"`, the verifier consumes `Ref("q_norm")`, and the executor resolves and binds the real return (`adapters_v2:268–295`; `cycle_driver_v2:70–100`; `executor_image:143,155,169`). Key `out=` and aligned-local completed stash copies are explicit. The prior unrelated uninitialized query placeholder is gone.
- **F2:** the three ordered author builder entries allocate a real `TreeStructure`, populate its inclusive closure and populate a separate proper-ancestor bitset (`adapters_v2:218–228`). They execute after outside-sequence reset and before consumers (`cycle_driver_v2:276–280`). The image executor compares built contents with CPU references (`:172–184`). The reference formulas match the pinned builders, including inclusive versus proper ancestors and diagonal `-inf`. Build-scoped results survive `begin_step`; static reuse across the fixed topology is explicitly declared. Replay reads the built bitset, so a nonroot leaf includes its ancestors.
- **F3 adapter call sites:** `last_correct_steps` and `leaf_full` are persistent buffers allocated on the owning device, explicitly filled from host metadata and passed to the author kernels (`adapters_v2:197–215,310–317,390–403`). Lumo passes the actual path nodes to the accepted runner's `Backend.publish`, then checks the persistent device metadata against its host plan (`:440–449`; executor `:158–162`). R1 remains an executor-level issue.
- **Publication payloads:** Weaver's full stash/state/index/mask kwargs reach the real replay callable; Lumo receives node IDs, not only an accepted length. Equal-length sibling choices therefore remain distinguishable. No additional recurrence or precision substitution is introduced by this repair.
- **TreeWY complete-cycle semantics:** the driver issues exactly one real fused wrapper call per layer per capture, observes the prior accepted commit only **after** that call, and executes one final fused flush per layer (`cycle_driver_v2:304–328`). With `M=3,L=48`, that is 192 real wrapper calls, 144 commit observations, and 144 gather/remap pairs. The final flush uses the last DFS inputs solely to complete its mandatory fused call, does not gather/remap its discarded output, and includes its publication. This aligns three logical verifications and three durable accepted-state publications with the other methods; it does not establish numerical equality.
- **Method distinctions:** root/inactive geometry, accepted-node mapping, TreeWY's positive-gcum sentinel, author-default normalization/beta seams, scale and refusal of the unimplemented aligned TreeWY variant are unchanged. The raw operand sequence and starting state are shared; subsequent durable states follow each method's own arithmetic. Hash chaining establishes receipt continuity, not equal state values across methods.

The proposed first-step C2 diagnostics, finite-value stop rules, raw output/state preservation, executable stage caller and failure receipts are parent-owned review items. In this snapshot `run_cycle` primarily records digests and call evidence; this note does not certify those proposed checks as implemented. No source edits, new experiment scope, gate change, performance conclusion or numerical acceptance follows from this review.

Final delivery identity addendum: the worker subsequently sealed the same reviewed v2 bytes in `FREEZE-M1-ADAPTERS-v2.json` at `2026-09-28T04:29:06.553586+00:00`, SHA-256 `029995649261206a5cc23ddb3601dbb10a1b52dec72dd4e67aa0024d44f369a3`. All 17 locally present freeze members independently match their hashes/sizes, including the 13 snapshot members named by v2; the unchanged source-manifest generator is the fourteenth snapshot member and is not listed by v2. This changes the delivery identity, not the findings or approval status.
