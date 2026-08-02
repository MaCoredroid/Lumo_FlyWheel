# B1 wide256 K64/root1 real-gate profile

Status: ready for one real SWE-Verified B1 byte-equality gate. No GPU or
container run was performed while preparing this profile. This diagnostic is
not performance acceptance evidence.

The prior `c668c584...` binary is ineligible because its 256-comparison cap was
exhausted before the fifth required projection shape. The replacement binary
uses a 320-comparison cap and is pinned by path, SHA-256, size, mode, patch
source, generated dispatch, and static kernel-resource evidence in
`manifest.json`.

The runner fixes the workload to K64/root1, physical rows 32, the canonical
64K block map, stock-served byte comparison, the authenticated real task
marker, and all five real projection shapes. Every unrelated kernel candidate
is disabled. `LUMO_SWE_AUTOCOMMIT=0` prevents task output from being committed.

Run from the clean branch HEAD containing this artifact:

```bash
cd /home/mark/lumoFlyWheel-b1-wide256-k64-root-profile
RUNROOT=output/fr13_b1_wide256_recompute_k64_root_gate_20260802T_gateZ \
TAG=wide256_recompute_k64_root_cap320_20260802T_gateZ \
FORKED_FA2_SO=/home/mark/lumoFlyWheel-kernel-integrated/output/fr13_qrow16_production_assets/_vllm_fa2_C.qrow16_num_splits0.abi3.so \
CUTLASS_STREAMK_SO=/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_recompute_stack0_k64_root1_gate_ready_cap320.abi3.so \
FR13_STREAMK_GATE_CANDIDATE=streamk_force_wide256 \
FR13_STREAMK_QUALIFICATION_PROFILE=k64_root \
bash scripts/fr13_run_b1_cutlass_streamk_live_gate.sh
```

Replace `gate` in `RUNROOT` and `TAG` with one shared unique UTC timestamp
before execution. The command must be run only once unless the attempt is
proven not to have reached the real task.
