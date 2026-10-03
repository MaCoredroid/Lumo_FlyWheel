# Owned campaign evidence residency diagnosis

Read-only measurements: 2026-09-28 23:19:44–23:24:44 UTC. This note seals the three authorized bounded inventory passes. No remote payload content was read or hashed; no cache operation, CUDA query, model boot, or process/container mutation occurred.

## Finding

M1 evidence alone accounts for **28,703,254,437 resident file bytes (26.732 GiB)** at the sampled times. The deduplicated union with the bounded campaign extensions accounts for **29,888,314,142 bytes (27.836 GiB)** across **8,685 unique device/inode pairs**. The earlier measured CUDA capacity shortfall was **36,754,818,826 bytes (34.231 GiB)**. Thus the observed union is **6.395 GiB smaller** than that shortfall, but this is **not a full-campaign insufficiency result**: both extension scans reached their 5,000-file cap inside `q1.2b`, leaving the remainder and subsequent roots unmeasured.

An active native-reference mirror was present in every before/after process inventory. Excluding the entire `m1-native-c1` root while that mirror is reading leaves **19,648,766,144 observed resident bytes (18.299 GiB)**. These values establish attribution to named file pages; they do not establish per-page cleanliness, current exclusive ownership, reclaimability, or an equal increase in CUDA-allocatable memory.

## Measured population, deduplicated

The table retains the latest observation for each `(st_dev, st_ino)`. Earlier pass identities agree on size, inode, mtime and ctime. It is a union across roughly five minutes, not one simultaneous global memory snapshot. Small sidecars from the all-file passes remain included; the final pass considered only `.pt`, `.bin`, `.npy`, `.npz`, and `.safetensors` payloads.

| Root below the owned campaign `runs/` | Unique files | Resident bytes within file | GiB |
| --- | ---: | ---: | ---: |
| `m1-c-baseline` | 23 | 36,573,903 | 0.034 |
| `m1-init-stage` | 268 | 132,931,212 | 0.124 |
| `m1-native-c1` | 311 | 10,239,547,998 | 9.536 |
| `m1-q-c0` | 557 | 4,285,471,205 | 3.991 |
| `m1-q-native` | 56 | 3,959,222,915 | 3.687 |
| `m1-q-reference` | 120 | 10,049,507,204 | 9.359 |
| `q1.2a` | 316 | 142,142,202 | 0.132 |
| `q1.2b` | 7,034 | 1,042,917,503 | 0.971 |

All six M1 roots were enumerated completely. `q1.2a` was reached completely in the extensions. `q1.2b` is partial. The last payload pass stopped after 5,000 payload files at `q1.2b/q12b-calibration-retry-20260928T010045Z/tensors/998e01d87ac4c6162ba4efb97d3ff3a978bc60ff7cfabf0a75102d4547f5da4d.bin` as the next unmeasured candidate. Its elapsed time was 1.212 seconds; the earlier all-file extension took 1.086 seconds. Neither hit the 60-second bound. There were no stat/mincore errors or identity changes. No additional remote scan follows this final authorized pass.

No residency conclusion is available here for the unvisited `q1.2b-heldout`, `q1.2b-heldout-reduction`, `q1.2b-init`, candidate-stage1, or native-smoke roots. The final suffix filter also does not establish coverage of other payload formats. Failed-but-terminal preserved runs are evidence artifacts; “completed run directory” does not mean numerical qualification passed.

## Reader and integrity evidence

Root process inventories were complete before and after each pass: 614/614 PIDs for M1, 622/622 for the first extension, and 621/621 for the final extension; no unknown or vanished entries. The only matching reader was `rsync` PID **1654330**, starttime **481855643**, with a read-only fd under `m1-native-c1/native-reference-pack-20260928T212230Z/`. It was at `20260930.layer04.c2.pt` in the first pass and `20260930.layer29.c2.pt` in the final pass. Its command is the remote sender for the parent’s native mirror (local PTY 73207). No sampled inode had a matching writable fd or mapping in these point-in-time inventories. This does not rule out a later reader or an unscanned inode. The full native mirror root remains excluded from any prospective treatment while this reader is active; merely excluding the one currently open file would be insufficient.

Files were inspected with `lstat`, `O_RDONLY|O_NOFOLLOW`, matching `fstat`, a `PROT_NONE` mapping and `mincore`, then checked again after closing the mapping. No payload pages were touched. Device/inode deduplication removes overlaps across all three receipts. All sampled files use device **66306**; this is not the earlier served-weight target device 2050. Link counts and potential hardlinks outside the allowed roots were not inventoried, and this pass does not prove backing-filesystem advice semantics. Path ownership must not be promoted to global exclusivity.

`ARCHIVED-IDENTITY-BINDING.json` links 1,019 M1 files, totaling 28,533,731,090 resident bytes, to previously verified local review indexes by path and size, retaining their prior content hashes. There were no conflicting prior hashes. The unbound M1 files are principally the older CPU baseline/init-stage artifacts and small sidecars, not a newly verified content population. Removing the active mirror root leaves 18,294,191,296 archive-bound resident M1 bytes. The union additionally reuses the local q1.2a evidence indexes and q1.2b terminal-receipt sidecar hashes; counts are recorded in `UNION-RECONCILIATION.json`. This is historical content provenance plus stable current stat identity, **not fresh content authentication**. Content-addressed tensor filenames alone were not treated as a newly verified hash. The q1.2b retry receipt records 6,216 tensor objects totaling 9,714,843,648 bytes, but does not tell us how many unmeasured objects are currently resident.

## What is established, and what is not

At the first pass, host `Cached` was approximately 44.34 GiB and `Buffers` approximately 9.41 GiB; `Dirty` was 396 KiB and `Writeback` zero. Those aggregate counters do not identify which file pages are clean. `mincore` supplies residency, not dirty status. The inventories cannot attribute the remaining CUDA shortfall to a particular other owner, distinguish all anonymous/kernel/driver allocations, or promise that these file pages are the CUDA allocator's immediate limiting factor. The query and residency samples were also taken at different times.

The prior query-only receipt measured **51,571,183,616 free / 126,175,309,824 total CUDA bytes** against the unchanged **88,326,002,442-byte reservation**. This diagnosis does not supersede that failure or authorize another query, reclaim, guard change, or model launch. The three served model shards were observed nonresident in that query's own receipts; the new observation concerns campaign evidence on a different device.

## Bounded next step

Do not wait for the rsync reader to finish expecting cache eviction: mirror completion can remove a reader exclusion, but it does not itself release file-backed pages. The measured no-reader subset is below the earlier shortfall even under an unjustified one-for-one release assumption. A concrete evidence-file readiness proposal therefore needs either additional residency evidence from the **unmeasured campaign remainder** or independently identified capacity release; this note cannot promise the existing subset will make the boot fit.

If the parent elects to prepare such a proposal, use a separately authorized, cursor-bounded read-only pass starting after the recorded q1.2b position and the unvisited completed roots, then an exact inode whitelist tied to archived evidence. Before any separately approved cache action, establish completed mirrors/archive preservation, action-time no-readers/no-writers, unchanged inode identity, relevant filesystem/advice semantics and explicit operational bounds. Existing one-use authorities remain consumed. This is a proposed prerequisite sequence only; no cache action or follow-on scan is authorized by this review. The fixed reservation and scientific gates remain unchanged, and no experiment/qualification/workload counter advances.

## Evidence hashes

- `RESIDENCY.json`: `e7950a898d2d7b6a217f98b84eabde053ef43bb43c7ae18c163b55a74c4d17ac`; 1,335 unique files in that pass; 2026-09-28T23:19:44.717561+00:00–2026-09-28T23:19:45.598832+00:00.
- `CAMPAIGN-RESIDENCY.json`: `a376c908d07bfbe3b0b9ec6230490a466d24cdcdd63b3e92b406fff5603540e2`; 5,000 unique files in that pass; 2026-09-28T23:23:09.093908+00:00–2026-09-28T23:23:10.138171+00:00.
- `PAYLOAD-RESIDENCY.json`: `a3925ce95fc7845b9c68a668595a784d7ca03a5bff941d0b2b41253cb991f644`; 5,000 unique files in that pass; 2026-09-28T23:24:43.605252+00:00–2026-09-28T23:24:44.761116+00:00.
- `ARCHIVED-IDENTITY-BINDING.json`: `376de1f054f9a8834e90288db5ea36ce237b9a192900678f1da39ba69d3ac728`.
- `UNION-RECONCILIATION.json`: `35a9599fb83af57043f240c63f0ee4529be9c3516abed99fdf80d8ae1db11fda`.
- Earlier query result review: `notes/review-response-20260927/query-only-20260928T231124Z-result-review.md`, SHA `91d2abc615865808e58c123031472d6da4e8f53f0adbe76c703919a1c4c3cf8f`.

All raw/stat/proc receipts and the local reconciliation script are under `p0/monitor/review-response-20260927/owned-m1-residency-diagnosis/`. The companion `REVIEW-SEAL.json` hashes the locally saved reviewer artifacts.
