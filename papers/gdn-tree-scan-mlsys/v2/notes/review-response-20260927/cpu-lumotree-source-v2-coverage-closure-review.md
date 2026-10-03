# LumoTree CPU source export v2: bounded coverage closure

**Disposition: F1 closed for the v2 source-preparation artifact.** The prior review, `cpu-lumotree-source-result-review.md`, remains unchanged. The new export includes every changed Python source identified by the before/after inventories within the installed vLLM package. This is source-preparation evidence only; it does not approve a serving boot, GPU qualification, workload attempt, timing result, or gate change.

## Reviewed identity and delta

Campaign paths are relative to `experiments/review-response-20260927`.

| Artifact | SHA-256 |
| --- | --- |
| `tools/cpu_lumotree_source_prepare_v2.py` | `4d2ea32e4fd0a8c58584f77677bad5a259c0cbe87bcff6762fe410b2ee455bd7` |
| Run `driver.py` | `1c4f8278cbb9de757032b3c45d78353998636788f3877e052b3b7a4ac3e366e1` |
| Run `CREATE-ARGV.json` | `4272a809cf3b1e4e31fb05eacdc8b1e9ab4d83d896c971b14d9c94e8b9cb3416` |
| Run `BEFORE-START.json` | `1afe868bd08eb3163b9fb8b00f0eb09e8c26ac9a8277e49693e200d9b7852a52` |
| Run `AFTER-EXIT.json` | `755fe9b9bab2e5c7f2df21c98b72ab1b4c3ddd20b797f05cf6708a5b587011d1` |
| Run `RESULT.json` | `6c313d28eb77d7053fac4d373c106da702a043a1e2017aa8a62d70cb53d4b8a5` |
| Run `result/MANIFEST.json` | `6325565273b36487c9b700cc4817d85d2aa0c1e4ca750cddf1fbf8430ade9dd7` |
| Run `result/PATCH-ENVIRONMENT.json` | `4b343bc0d33c07a274a951f9975af95084fd89212203bf615f21034564a160d9` |

Run directory: `workload-plan/inspections/cpu-lumotree-source-20260929T012000Z`. Its ID is a label: the host interval is **2026-09-29 01:18:17.765969–01:18:20.823122 UTC**, and the recorded container interval is **01:18:17.849220399–01:18:20.678796587 UTC**.

The v1-to-v2 diff changes the schema/documentation and adds the vLLM Python inventory and union export. Script lines 69–75 inventory Python paths, hashes and sizes before patching; lines 89–96 repeat the inventory, reject deletions, and export the original selection union all changed paths. Lines 103–108 retain both inventories and the explicit coverage definition. The three patch commands, pinned prospective profile, container isolation, execution, and cleanup logic are unchanged. The retained driver was reconstructed byte-for-byte from the script's embedded literal and `FILES` substitution.

## Coverage and source consistency

Local stdlib-only verification, without importing or running the preparation script, established:

- Both inventories contain **1,688 Python paths**, with the same path population. Recomputing the hash/size differences reproduces exactly **26 changed paths**.
- The export contains exactly **35 unique files**, totaling **3,250,855 bytes**: the original 34 selected paths union all changed paths. Every changed path is present. Each raw export's hash and size matches its manifest row and after inventory; before hashes also agree. The export directory has no additional or missing files.
- All prior **34** manifest rows and exported file hashes are identical to v1. The sole addition is `model_executor/layers/batch_invariant.py`, **34,808 bytes**, before SHA `5f3ad0a7109aac430f1145bbbe85935844dbb3abe07fd70f2883698ef1b8b15b`, after SHA `dfd425380e7d0a8311d85669fbe94649285fc6f912716f9a24b543898cba6f4e`.
- All nine pinned source inputs match the local files. Command vectors and profile SHA remain unchanged. All three patch stdout files and `/logs` sidecars match v1 byte-for-byte; patch stderr files are empty. Actual patch environment has the same 525 fields and values except the container-owned `HOSTNAME`.

The scope is **complete changed Python-source coverage within the installed vLLM package under this recorded prospective profile**. The before/after inventories retain hashes and sizes for all 1,688 paths; they do not export all 1,688 raw files. This is not a complete filesystem, non-Python, binary, or dependency inventory.

## Isolation, outcome, and cleanup

The create argv and before/after inspect receipts agree on image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc` and exact CID `583290109e8f86b28a6bbe14b57de5b99796d3b2c32d3e005d786a979d504c0f`. They record `runc`, network `none`, no Devices or DeviceRequests, nonprivileged execution, all capabilities dropped, no-new-privileges, 4 GiB memory and memory+swap, two CPUs, and PID limit 128. The only two bind mounts are read-only repository and FA2 fork inputs; there is no model mount. CUDA visibility is empty and NVIDIA visibility is `void`.

The container moved from created to exited with exit code 0, PID 0, no OOM/error, and no running state. Copy return code is 0. `RESULT.json` binds the manifest and driver hashes and records successful exact-CID removal. Script lines 153–157 verify the CID and ownership label before the checked removal. As in v1, cleanup stdout is not separately retained; this review verifies the recorded result and code path, not live container absence.

**No remaining blocker in the requested export-coverage closure.** The result does not establish full-launcher environment equivalence, loaded-module/binary equivalence, or runtime behavior. Those boundaries remain unchanged.
