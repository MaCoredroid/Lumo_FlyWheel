# Fixed32 FA2 qrow16 division-free SO readiness

Status: **host-only shared-object and production-ABI preflight passed; real B1
byte parity remains pending**.

The exact qrow16 kernel from the preceding SM121a admission checkpoint is now
packaged as an AArch64 FA2 shared object. The device payload combines the
1024-row paged-KV coordinate specialization with the tree-bias suffix-tile
early-out. The production qrow16 host launcher object is retained byte-for-byte
outside its CUDA fatbin, and only the candidate SM80 cubin plus compute-80 PTX
are grafted into that object.

The pinned candidate is:

- SHA-256: `106e54d1c82ec7ce7576cbb44bb4aa2342b2985bb58e97aeeca5503275bee3e2`
- size: `299491544` bytes
- external path: `/home/mark/shared/lumoFlyWheel-qrow16-thin/output/fr13_fa2_qrow16_divfree_build_20260802/_vllm_fa2_C.qrow16_divfree_clean_final.abi3.so`

Host GCC 13 linking changed exactly two undefined CUDAStream dynamic-symbol
type tags from `FUNC` to `NOTYPE`. The fail-closed finalizer accepted only
those two known differences and restored the production metadata. Afterward,
all 856 ordered dynamic symbol signatures, 10 `DT_NEEDED` entries, the
RUNPATH, GNU hash, and GNU version sections match the previously real-task-
loaded qrow16 reference. This proves ELF and production-loader ABI readiness;
an actual production-runtime import was not attempted in the host-only track.

The clean SM121a compile remains at 224 registers, zero stack/local/spill,
zero calls, and 5,040 instructions. It preserves 512 BF16 HMMAs, 132 FFMAs,
264 FMULs, 336 `LDGSTS`, 288 `LDSM`, 76 global loads, and 38 global stores.
The candidate PTX has zero signed 64-bit division/remainder instructions,
versus four in the incumbent qrow16 payload.

Commit `bf86081b7b08a3dc5e5dcbe175be280d6e888f96` pins this binary for the
live selector and adds a dedicated default-off real SWE-Verified B1 paged byte
gate. The gate fixes K=65536, root reduction on, the canonical K64 block map,
one physical 32-row request, FULL graph replay, and the canonical real B1 task.
It serves the stock graph result and emits no timing samples.

No GPU, Docker container, synthetic timing probe, or real task was used in
this host-only checkpoint. The candidate remains timing-ineligible and is not
authorized for production until the real K64/root1 B1 byte gate passes.

