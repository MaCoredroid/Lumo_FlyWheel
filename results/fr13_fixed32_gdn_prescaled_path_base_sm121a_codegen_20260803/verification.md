# Verification

Toolchain:

- image: `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`
- Python: 3.12
- PyTorch: 2.10.0+cu130
- Triton: 3.6.0
- backend producer: `ptxas-blackwell` 12.9.86
- host SM121a disassembler: CUDA 13.0.85
- target: `sm_121a`

The image's Triton compiler can produce SM121a cubins, but its bundled
12.9 `nvdisasm` cannot decode SM121a. The commands below mount the host CUDA
13.0.85 disassembler read-only so `cuobjdump` can produce the audited SASS.

From the repository root, run two builds in independent containers and caches:

```bash
ART=results/fr13_fixed32_gdn_prescaled_path_base_sm121a_codegen_20260803
REV=8959f328ce6b5e36c5eb6bbb1cb53c3c6e5f5bbe
IMG=vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
COMMON="-v $PWD:/workspace:ro -v /tmp:/hosttmp \
  -v /usr/local/cuda/bin/nvdisasm:/usr/local/bin/nvdisasm:ro \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/workspace/src \
  -e PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

docker run --rm $COMMON \
  -e TRITON_CACHE_DIR=/tmp/fr13_gdn_prescaled_cache_a \
  --entrypoint /usr/bin/python3 "$IMG" \
  "/workspace/$ART/offline_codegen_audit.py" \
  --repo /workspace --revision "$REV" \
  --output /hosttmp/fr13_gdn_prescaled_codegen_a

docker run --rm $COMMON \
  -e TRITON_CACHE_DIR=/tmp/fr13_gdn_prescaled_cache_b \
  --entrypoint /usr/bin/python3 "$IMG" \
  "/workspace/$ART/offline_codegen_audit.py" \
  --repo /workspace --revision "$REV" \
  --output /hosttmp/fr13_gdn_prescaled_codegen_b

docker run --rm $COMMON --entrypoint /usr/bin/python3 "$IMG" \
  "/workspace/$ART/verify_codegen_outputs.py" \
  --primary /hosttmp/fr13_gdn_prescaled_codegen_a \
  --rebuild /hosttmp/fr13_gdn_prescaled_codegen_b
```

Focused host tests:

```bash
python3 -m pytest -q \
  tests/test_fr13_fixed32_gdn_prescaled_path_base.py \
  tests/test_fr13_fixed32_gdn_single_launch_ordered.py \
  tests/test_fr13_fixed32_gdn_exact_io.py \
  tests/test_fr13_fixed32_gdn_path_bv_live_gate.py \
  tests/test_fr13_fixed32_gdn_static_descriptor_rejection.py \
  tests/test_fr13_fixed32_gdn_prescaled_path_base_codegen_artifact.py
```

The verifier re-disassembles all eight cubins, validates cubin/PTX/SASS
hashes, compares the independent summaries exactly, enforces the incumbent
99-register ceiling with zero stack/local/spill/call use, requires the SASS
reduction in both B1 and B4, and checks that the node and recurrence source
hashes remain unchanged across arms.
