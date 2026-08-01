# Fixed32 GDN level-0 coefficient staging: SM121 build

This directory is an offline compiler artifact for the fixed32 two-launch
coefficient-staging candidate. It is not timing evidence and no GPU device was
exposed during this build.

## Identity

- Kernel source SHA-256: `213b7d9bacbdb570f2544cf38b2d3a8b76bb7c65863be423839a68c4cddda292`
- Container image: `lumo-flywheel-vllm:26.01-py3-v0.19.0-fr9iso`
- Container image ID: `sha256:3e3c5ccc9fa038dab00d5d5192e54cf12c30650e866a643aaef4d450148ead21`
- Triton: `3.6.0`
- Target: `sm_121`, warp size 32, eight warps
- CUDA compiler tools: `13.1.115`

The build command was:

```bash
docker run --rm \
  -e PYTHONPATH=/workspace/src \
  -v /home/mark/lumoFlyWheel-gdn-level0-coeff:/workspace \
  -w /workspace \
  lumo-flywheel-vllm:26.01-py3-v0.19.0-fr9iso \
  python3 scripts/fr13_build_gdn_level0_coeff_sm121.py \
  --output /workspace/results/fr13_fixed32_gdn_level0_coeff_sm121_build_final_20260801
```

Docker reported that the NVIDIA driver was absent. The command did not use
`--gpus` or expose a GPU device.

## Resource result

| Specialization | Registers | Stack | Shared | Local | SASS LDL/STL |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 stock level 0 | 96 | 0 | 1024 | 0 | 0 |
| B1 stock level 1 | 62 | 0 | 1024 | 0 | 0 |
| B1 candidate level 0 | 79 | 0 | 1024 | 0 | 0 |
| B1 candidate level 1 | 48 | 0 | 1024 | 0 | 0 |
| B4 stock level 0 | 78 | 0 | 1024 | 0 | 0 |
| B4 stock level 1 | 76 | 0 | 1024 | 0 | 0 |
| B4 candidate level 0 | 80 | 0 | 1024 | 0 | 0 |
| B4 candidate level 1 | 48 | 0 | 1024 | 0 | 0 |

`build_manifest.json` contains compiler metadata and hashes. Each specialization
also includes CUBIN, PTX, TTGIR, SASS, and `cuobjdump` resource output. The build
script fails if any specialization has nonzero local memory or an `LDL`/`STL`
instruction.
