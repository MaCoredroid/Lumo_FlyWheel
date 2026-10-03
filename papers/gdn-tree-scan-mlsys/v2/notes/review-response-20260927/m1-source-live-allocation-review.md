# M1 source/live-allocation review — 2026-09-28

**Disposition:** the existing accounting is a useful shape ledger, not a worst-live memory bound. The proposed 2.71 GiB / 1.25-factor estimate is unsupported. Retain the parent-approved **12 GiB entry floor**; it is a policy floor, not proof of peak sufficiency. The smallest next step is to record actual allocator peaks and system availability within the already-planned untimed initialization/qualification/warmup, with explicit phase boundaries. This review authorizes no execution, reclaim, floor change, or numerical/timing qualification. Existing F1–F5 closures are unchanged.

This was a local source/JSON review plus integer byte arithmetic. No Torch/CUDA import, GPU call, SSH, container, cleanup, memory operation, or experiment was executed. No implementation or frozen evidence was changed.

## Reviewed sources and actual observation

All paths below are relative to `papers/gdn-tree-scan-mlsys/v2`. `S` denotes `p0/monitor/review-response-20260927/m1-v32-reviewed-20260928T0529Z/repo`; `C` denotes its `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927`; `R` denotes `experiments/review-response-20260927/runs/m1-init-stage/m1-init-20260928T053031Z`. Lines refer to these accepted bytes, except the explicitly named inherited canonical v1 files, whose hashes were checked against the preserved v1 freeze.

Read both `M1-MEASUREMENT-QUALIFICATION-DESIGN-PROPOSAL-v3.md` and `M1-MEASUREMENT-QUALIFICATION-DESIGN-PROPOSAL-v3.1-AMENDMENT.md`, and the parent disposition `p0/monitor/review-response-20260927/m1-v31-design-parent-review.json`. The latter expressly rejects lowering the floor using the proposed multiplier.

`R/out/resource_preflight.json` records **12,935,909,376 free bytes (12.048 GiB)** of 126,175,309,824 total bytes at entry. The four serial methods subsequently completed the initialization stage. There is no recorded CUDA allocation/reservation high-water mark or in-stage minimum free-memory series. Successful completion establishes that this particular initialization ran; it does not establish the successor's peak, additional repeat lifetimes, or numerical qualification.

## Allocation and lifetime inventory

GiB means bytes / 2^30. Counts below describe named tensor payloads, excluding allocator granularity and unknown runtime overhead. They are not a sum-to-peak model.

| Actual object / lifetime | Payload and alias/overlap facts | Source |
|---|---|---|
| Shared original inputs, CPU | 48 copies of q/k `[32,16,128]` bf16, v `[32,48,128]` bf16, a/b `[32,48]` bf16, A/dt `[48]` fp32, and S0 `[48,128,128]` fp32: **182,765,568 B (0.170214 GiB)**. These remain in `run_stage` through all methods. Each method stages a distinct CUDA copy of the same payload; CPU and device copies coexist. | Inherited `tools/m1/m1_cycle_driver.py:38–42`; `tools/q1_2b_fixtures.py:77–98`; `C/tools/m1/m1_stage_collector_v3_2.py:156–170`; `m1_executor_image_v3.py:70–71`. |
| Lumo shared page storage | **16 × 24 × 2,097,152 bf16 elements = 1,610,612,736 B (1.5 GiB)**. Three layers share each underlying page allocation. fp32 recurrent-state views and bf16 convolution views are aliases into these pages, not additional allocations. | `C/tools/q1_component_runner_v2_2.py:172–190,338–339`. |
| Lumo rings, outputs, commit buffers | The receipt lists k/v/a/b rings, spec indices, 48 source stagings, fixed16 graph input buffers and output `[48,32,48,128]` bf16. Output alone is 18,874,368 B. The backend also actually allocates `c1_bank [2,48,128,128]` fp32, **6,291,456 B**, absent from the ledger. Strict/visible masks are int32, **8,192 B**, rather than the ledger's 2,048 B bool description. `prev_lens` is another small omitted allocation. | Runner `:327–362`, mask constructor `:127–133`; kernel `:15074–15135`; raw Lumo `receipt.json/allocations`. |
| Kernel persistent subtree scratch | `subtree_preseed` allocates export `[32,48,128,128]` fp32, **100,663,296 B (0.09375 GiB)**, plus device path/parent/length/mask descriptors. This single shared scratch is stored in module-global `_FR13_SUBTREE_CACHE`, not allocated once per layer. It is absent from the M1 ledger. | `S/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:6184–6351`, particularly `:6300–6346`; runner boot `:142`. |
| Kernel persistent convolution staging | Pregather allocates `[48,1,10240×34]` bf16, **33,423,360 B (0.031128 GiB)**, plus pointer/offset/alias tables, guard flags and a cloned state-source map. This is separate from the runner's 48 source stagings already in the ledger. `_FR13_FIXED32_CONV_PREGATHER` retains banks and source references. | Kernel `:8180–8238,8270–8324`; runner boot `:151–154`. |
| Lumo boot/warm temporary backups | Graph capture saves 48 running-state rows: **150,994,944 B (0.140625 GiB)**. The later postprocess warm phase simultaneously saves that amount of recurrent rows, **33,423,360 B** of conv rows and **33,423,360 B** of conv staging, plus small metadata/qbuf copies: at least **217,841,664 B (0.202881 GiB)** before small temporaries. These two boot phases are sequential, not additive. `index_select(...).clone()` also transiently has the selected result and clone. | Kernel `:15335–15369`; `:15824–15875`. |
| Weaver base buffers | `ssm_states [48,2,48,128,128]` fp32 = **301,989,888 B**; plus k/v/g/beta stashes, ancestor bitset, indices and parent tokens. Actual `bufs` total **327,746,320 B** (author default) / **334,037,776 B** (aligned local; fp32 k stash). Topology build results are additional small allocations. | Inherited `m1_adapters.py:91–98`; `C/tools/m1/m1_adapters_v2.py:197–224`. |
| Weaver disposable warmup and workspace | `clone_bufs` clones **every tensor in the buffer dictionary** while originals stay live: another 327,746,320 / 334,037,776 B, not merely an endpoint-state clone. Query normalization and prefix/U/O allocate during calls. Author `_WORKSPACE` and `_TRI_CACHE` persist in the imported module; matched shape/device workspaces can be reused across the two Weaver policies. Local fp32 normalization/gating has expression intermediates beyond its retained result. | `m1_cycle_driver_v2.py:133–134`; `m1_executor_image.py:118–124`; author `experiments/author-code-20260926/weaver/gdn_tree_triton.py:59–79,438–444`; `m1_adapters.py:108–112`; `m1_adapters_v2.py:263–295`. |
| TreeWY buffers and disposable warmup | State bank 301,989,888 B; vt/kk stashes 75,497,472 B each; gc stash 589,824 B; DFS staging, persistent inverse-remapped outputs and small metadata yield **504,209,936 B** in `bufs`. Disposable warmup clones the whole dictionary again. `kt` refers to `anc_i`; it is not a third mask. Wrapper output is a fresh **393,216 B** per call; cumulative 192 calls is not 192 simultaneously live outputs. | Inherited `m1_adapters.py:224–235`; `m1_adapters_v2.py:321–337`; executor `:118–124`; raw TreeWY allocation ledger. |
| Host captures and diagnostics on UMA | Four durable captures `[48,48,128,128]` fp32 are **603,979,776 B (0.5625 GiB)** retained on **CPU**, plus 18,874,368 B of CPU first outputs. Endpoint diagnostics retain **three float64 reference endpoints per layer = 905,969,664 B (0.84375 GiB)** at once, alongside these captures and shared CPU inputs. This named host overlap is already **1,711,589,376 B (1.594042 GiB)**, before serialization/diagnostic temporaries. | `m1_stage_collector_v3.py:208–235,240–271`; v3.2 `:89–93`. |

The CPU captures are important on shared-memory GB10, but must not be called CUDA clones. Lumo capture additionally creates one device layer clone before copying to CPU; stacking all layer copies briefly overlaps the list with the stacked CPU tensor (`Backend.published_state:441–442`; collector `:271`). Tensor serialization creates a Python byte string of the tensor, and host diagnostics have finite/error and float64 temporaries. The first-output reference routine retains all 32 node states for one layer; assignment can overlap the previous layer's returned state dictionary with construction of the next. These are CPU/UMA costs, not CUDA allocator measurements.

## Why the four ledger totals are not a peak bound

The raw `unique_live_storage_bytes` fields are 1,703,184,468 (Lumo), 329,673,752 (Weaver default), 336,096,280 (aligned), and 504,603,152 (TreeWY). Their declared formula counts each listed transient once and does not model construction, clones, capture, serialization or inter-method retention. The `temporaries: none` wording specifically describes the repaired `index_select(..., out=...)` gather/remap; it cannot be generalized to the full method.

Methods are serial in **one process** (`collector_v3_2.py:164–170`), with no cache teardown between them. Lumo's module-global graph/conv structures keep references to shared pages, rings and commit buffers even after the local executor goes out of scope; Weaver's workspace is also global. Therefore `max(per-method ledger)` is not a justified whole-process bound. This is a lifetime observation, not a recommendation to clear caches: the successor should retain the prespecified warm/cached lifecycle and report it honestly.

CUDA context, driver/module/JIT residency, graph objects/private pools, allocator rounding/reserved cache and library workspace are not quantified by the ledger. `CUDAGraph` creation is explicit at kernel `:15347–15368`; whether both reference and layer-batched graphs persist is flag-dependent. No constant multiplier bounds these bytes. Allocator-reserved bytes already include allocated tensors, and on UMA host RSS/device accounting may overlap; do not add every counter into a fictitious physical total.

Finally, `logical_exported_bytes = 452,984,832` is **three state-publication volumes**, not resident allocation or measured traffic. The content-addressed store contains **205 objects / 2,038,431,744 bytes** (`R/out/m1/tensors/index.json`); this is disk payload after deduplication, not device live storage. Writing it may also consume reclaimable page cache. Neither number establishes a peak.

## Minimal observation points in the already-planned untimed work

Use the existing process and existing U1/U2/U3; do not add another boot, reclaim, or standalone memory trial. Preserve setup versus complete-cycle timing boundaries. Before each untimed phase, synchronize and reset CUDA peak counters; at the end synchronize and record `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `max_memory_reserved`, and `mem_get_info`. Record same-time process RSS/high-water RSS and `/proc/meminfo` MemAvailable/SwapFree as separate host/system evidence. Queries occur outside timed windows.

1. **Process/input baseline, then buffers + staged inputs:** collector immediately before operand generation / driver immediately after `stage_operands` (`v3.2:156`, `cycle_driver_v3:131–144`). This separates CPU fixture construction from CUDA input/bank residency.
2. **Backend construction and complete warm return:** bracket `_ensure_backend` / `Backend.boot_lifecycle`; the peak captures temporary saved rows even after they are released. Endpoint allocation deltas alone miss this peak. For authors, bracket `_warmup_disposable`; it catches original+cloned banks and author workspace initialization.
3. **Complete qualification including evidence:** bracket the existing `run_method`, and emit a point before and after endpoint diagnostics/serialization (`v3.2:89–121`). Host high-water sampling is necessary here because CUDA peaks cannot see retained CPU captures/references. A low-frequency existing-process host sample during this untimed span may locate the host maximum; endpoint-only readings cannot claim it.
4. **After each method and before the next:** preserve cumulative process allocation/reservation plus per-phase peaks. This exposes persistent Lumo/Weaver caches without changing their lifetime. For later repeated U3 warmups, bracket the existing reset/warm sequence again; do not infer repeated-run peak from the initialization stage.

Minimal CUDA observation seam, for future source-reviewed integration only (not executed in this review):

```python
# Enter an already planned untimed phase.
torch.cuda.synchronize(device)
torch.cuda.reset_peak_memory_stats(device)
# existing initialization / qualification / warmup body
torch.cuda.synchronize(device)
free_b, total_b = torch.cuda.mem_get_info(device)
sample = dict(allocated=torch.cuda.memory_allocated(device),
              reserved=torch.cuda.memory_reserved(device),
              peak_allocated=torch.cuda.max_memory_allocated(device),
              peak_reserved=torch.cuda.max_memory_reserved(device),
              free=free_b, total=total_b)
```

Keep the 12 GiB preflight. Abort on failed existing resource checks/OOM; preserve all evidence. A prospective smaller threshold or arbitrary headroom factor is not justified here. Peak samples support an explicitly observed requirement for these phases, not a universal worst-case bound or permission to change the numerical gate.

## Source and evidence identities

| File | SHA256 |
|---|---|
| Accepted snapshot `PARENT-SNAPSHOT.json` | `54054dbeebe39933228da9508a40014c574a866d666328594290c8131fdbe2e5` |
| Accepted `FREEZE-M1-ADAPTERS-v3.2.json` | `cf78f1985807b3b9dfcd1ba5dbf9e6c382283a4b0a838eaf107a6e5228639f4c` |
| Proposal v3 / v3.1 | `a48ce044eb678120e0c2aa2ab5ca56a1f95d8cc0bf5c12b35e0737a7c636bbd9` / `887cca6743daa20950ddd9a3049ee9c728f5d630f2a1c37f9fb7aaa60ed9dc74` |
| Parent proposal disposition | `a66974e7022fb316263a78bd770859bce46baa59f93cddb76a2d58a41322045a` |
| Collector v3.2 / inherited v3 | `e6b4b62b575b18c222210b515dfab2ecf5b420070620d0e1832526d9a4ab43f1` / `512bdab2a0b0e7283d1f09b7b9d85793a147b242a7853179bd3e5b4e937c4940` |
| `m1_executor_image.py` | `e3f441940407cd8a5e3e9ce9aaf9eb70dab9a402aec7092f83acbc24a85b60ef` |
| Backend runner v2.2 | `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29` |
| Production GDN kernel | `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8` |
| Canonical inherited `m1_adapters.py` / `m1_cycle_driver.py` (both matched v1 freeze) | `375a177b1da203a15b8deecc21f971a87a328d037e2f52913c41bd40f1444066` / `a55e2a147c19460c37d0b203ac08b0d65ea23ebfe79653233de7b97f303ff78b` |
| Fixture generator | `eba98d7ccabf8ad602b5274070b92b901ee571af1fba48a82bcf3297cb220039` |
| Author `gdn_tree_triton.py` (matched author manifest) | `b151d4b2ade0451a896e99f83e1656bcddd3d8ad279eab399e994c604e28a1ff` |
| Raw stage summary / resource preflight | `64436e55d1f935e009a89df7e055e02c90ef7f447d87f8b5efd377db80fd2d96` / `75204db65974a9151ee7d4b3a2073cb81adbdd1e4e1f1a97e7024172c2887e52` |
| Raw tensor-store index | `977617743eeb6fc919d37dc2d6c462fccc259f36fbe08a32b06b48ecb0065fdf` |
| Lumo / Weaver default receipts | `e2149dde352b2da0341b6cdd347b8be2e28d02799665b4a3ed5a59ec585b2386` / `cee8423038c6c4b994c4ec6ff28066d99c82816273ebc4b058319f8b482bc5bc` |
| Weaver aligned / TreeWY receipts | `7a89383b863943f4375c71c8ef28b4c97643b0be4164ecfe9f5f1fc03faf75e6` / `34e231ad6f458a626dea3c02b1e22f9b77027286b49c10f85cd1eee5bed1e4db` |

Only small source/receipt/index bytes were read and hashed; tensor payloads were not rehashed or recomputed. This is source inventory and observation planning, not a fresh measurement or replacement of prior raw results.
