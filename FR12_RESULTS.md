# FR12 Results

## WY Tree Recurrence Gate

Command:

```bash
python3 scripts/fr12_wy_tree_recurrence_check.py --json-out output/fr12_wy_tree_recurrence_check.json
```

Scope:
- Uses FR10 tree descriptors from `scripts/fr10_gdn_tree_algebra_reference.py`.
- Runs the gated delta recurrence in float64 to avoid the vLLM CPU oracle's fp32 floor.
- Validates parent-inherit plus one-reflector append T against rebuilding WY on each path.
- Validates full per-node state/output against serial per-path GDN semantics.

Result:
- Verdict: `PASS` at threshold `1e-8`.
- Max append-vs-rebuild T/basis error: `0.0`.
- Max append-vs-rebuild operator error: `0.0`.
- Max homogeneous S0 map error: `3.3306690738754696e-16`.
- Max full state vs serial error: `3.0531133177191805e-16`.
- Max output vs serial error: `8.673617379884035e-18`.

Interpretation:

`TREE_ANCESTRY_T_RECURRENCE_CONFIRMED`

The FR12 WY tree recurrence is algebraically exact at float64 floor for the tested FR10 tree shapes. This validates the parent T inheritance plus append rule before building the Triton kernel.

## Existing WY Fused Probe Baseline

Command:

```bash
docker run --rm --gpus all --entrypoint python3 \
  -v /home/mark/shared/lumoFlyWheel:/workspace -w /workspace \
  vllm/vllm-openai:cu130-nightly \
  /workspace/output/gdn_novel_research/wy_tree_fused_probe.py
```

Preflight:
- `docker ps` showed no running containers.
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` showed no active compute processes.

Result:
- Device: `NVIDIA GB10`.
- Tree nodes: `14`, padded nodes: `16`.
- Existing WY fused skeleton: `563.1969451904297 us`.
- Dense FR10-shaped fused kernel: `990.0962829589844 us`.
- Flat FLA chunk baseline: `138.1873607635498 us`.
- WY skeleton vs dense output maxabs: `0.013397216796875`.
- WY skeleton vs dense state maxabs: `0.1730390340089798`.

Interpretation:

`EXISTING_WY_FUSED_PROBE_IS_NOT_CORRECT`

The existing probe is a useful launch-shape and timing skeleton, but its reconstruction shortcut is not an acceptable FR12 implementation. The next kernel step must replace the reconstruction math and re-check against the serial/WY oracle; speed alone is not sufficient.
