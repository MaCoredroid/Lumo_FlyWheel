# Independent review: current model identity

27 September 2026. **Approved for current disk/source identity, with no blocker in the submitted report. This does not approve E3 execution or establish matched runtime behavior across four arms.** Review was local and CPU-only; no capture script, SSH command, model, server, GPU workload or task evaluator was run. Only this note was written.

## Artifact and reduction checks

- Delivered `MANIFEST.json` SHA256 is `c45af0bae100e0744c7d5db4cf401011bbac4210c4c841262ede3ededa3b7a1b`. All **14** listed payload sizes and SHA256 values match. The report SHA256 is `f648bc38484c78d11b79dadb67c3723bf5a003de1a6dee5c6ffba1e8242cfe4c`.
- `HASH-RECEIPT.json` records successful capture with empty stderr and binds both the saved raw JSON (`4ffb7131d158bb55c1295c4c449faa704cfeaf53c5f9076d940d11cd2ef9bece`) and the inspected read-only capture script. No remote rehash was performed for this review.
- Independently recomputed both canonical file-list digests; verified the lock against both raw views and the tree's complete 26-file fixed32 contract. All **11** local source hashes match; all **6** captured remote source hashes match the corresponding local files. The separate prompt receipt binds the remote custom template to local source and the model template to the model-file capture.
- All **six CPU tests passed**. All **four reducer outputs** reproduced byte-for-byte with output intercepted in memory, without overwriting the submitted artifacts. The tests include altered tokenizer, wrong tensor index and out-of-bounds tensor rejection.
- Independently checked all 2,194 tensor spans against shard boundaries, contiguous payload accounting and matching index populations. Both views agree. The three shard records have identical device/inode/size/mtime/ctime and hashes; the second view explicitly reuses the first view's verified inode hashes.

## Supported identity claim

At the saved capture, the inspected tree and SGLang paths contain identical **three weight shards and five tokenizer/template files**. The index and generation configuration also match. This establishes their shared current checkpoint/tokenizer input, not which bytes a historical process loaded or a future process will load. The report maintains that distinction correctly.

The full directory digests differ legitimately: there are six tree-only auxiliary/provenance files and exactly two differing common JSON files. Their recursive differences are the tree's absent KV declarations versus SGLang's `quantization_config.kv_cache_scheme` describing 8-bit float and `quantization.kv_cache_quant_algo = "FP8"`. These metadata differences are bounded and explicit; they do not establish effective engine KV/recurrent-state precision. Stored tensor dtype counts and declared quantization groups are supported, with runtime accumulation/repacking left unclaimed.

The report also correctly separates matching **checkpoint** template files from the **effective launcher** template policy. vLLM specifies the 11,526-byte custom template (`c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`); the model template is 8,952 bytes (`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`), and the inspected SGLang command provides no equivalent override. Equal tokenizer files therefore do not prove equal rendered prompts, parser behavior or input token IDs.

## Byte accounting

| Quantity | Bytes | Supported interpretation |
| --- | ---: | --- |
| Three shard files on disk | 21,921,697,280 | Sum of file sizes |
| Tensor payload | 21,921,428,072 | Sum of all 2,194 tensor spans; 269,208 bytes less than shard storage |
| Mandatory logical weight-read formula | 25,430,574,256 | 16,892,610,688 target decoder/norm + 715,161,608 verifier head + 5 × 849,398,784 MTP + 5 × 715,161,608 drafter head |

The source ledger binds the five-pass formula. The last row counts repeated logical reads and is neither checkpoint storage nor measured resident GPU allocation, traffic or performance. The report does not present it as a new performance measurement.

## Remaining execution gates

AR and chain remain **unbound for E3**: their inspected launchers default to the older FP8 target. The native launcher does expose an opt-in NVFP4 head-loading shim, but its availability is not an E3 selection or qualification receipt; its default remains off. Bind explicit current paths, byte locks, applicable loader/cache behavior and the chosen chain route before launching these controls.

For every arm, freeze image/argv, effective precision/cache settings, tokenizer and parser versions, and rendered-prompt policy, then retain actual startup/load receipts. The current tree/SGLang disk match closes a concrete input-identity gap; it does not close these runtime and protocol gaps. No report correction is required for this bounded approval.
