# CPU reference workload source preparation review — 2026-09-29 UTC

**Disposition: accept the completed CPU-only source preparation as evidence of the exported AR/chain source bytes. Hold any inference that the current observed launchers are ready to boot: their read-only bytecode prefix conflicts with an explicit preparation-time `py_compile` call.** No model boot, numerical qualification, workload result, or performance evidence was produced by this preparation.

Reviewer scope was local source/raw-receipt inspection, hashing, AST extraction, and two isolated standard-library CPU controls. No Docker, SSH, GPU, network operation, engine import, launcher execution, or source/gate modification was performed by the reviewer.

## Bound evidence

Campaign-relative run: `workload-plan/inspections/cpu-vllm-reference-source-20260929T004505Z/`.

| Artifact | SHA-256 |
| --- | --- |
| `tools/cpu_workload_source_prepare_v1.py` | `892fe6e2203e6b08856db194cd2cdc3ff4754f23ea2055ad5f195a4f6cce56bf` |
| Run `driver.py` | `79b9a17231f30275281eb4004c11417abc56d7dc67f348c7b7f1ffb83e392611` |
| `CREATE-ARGV.json` | `ce2232222fb4cca25296327a17bb97985d04e0894967b2eb2aba9d705db0b529` |
| `BEFORE-START.json` | `3e86f0e3b658de8c58c78ebea67a2d39d6c3ad7f739435100a41f53ea1d371d5` |
| `AFTER-EXIT.json` | `58b28f7ee9c56d1d7d7afc8e11e7a256c98794786a850bcbb656d22d9952df33` |
| `RESULT.json` | `406071f40814ad6e67cbe48d7dbe5f1be71e65fdbd596e619ac525b629500e52` |
| `result/MANIFEST.json` | `183597730e406d18c44e786021bbeebb54506ade33d822e835a98108b19bb9e7` |
| `result/fa2-install.json` | `486f35981bd83874fcc145b98c229d06e3dd5273cc58132a4025d818dadc3e23` |

The saved driver is byte-identical to the current preparation script's embedded template after substituting its exact `FILES` list. All three pinned input scripts also match local source. The manifest has precisely the expected 23 unique paths, no missing/extra exported source file, and **1,090,259 bytes**; every exported hash and size matches. Its hash and the driver hash agree with `RESULT.json`.

## Actual preparation and isolation

Both inspections name container `c79a3f6f879d359040c7a79a8fcde4008ee26cd5cb6162f9180164fc88b666d8` and immutable image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`, the AR/chain launchers' required image ID. Actual configuration has `runc`, `NetworkMode=none`, no devices or device requests, `Privileged=false`, all capabilities dropped, no-new-privileges, two CPUs, 2 GiB memory and memory+swap limit, and PID limit 128. Exactly two bind mounts exist: repository → `/workspace` and pinned fork → `/tmp/fr13_fork_fa2.so`; both are read-only. No model mount exists. `CUDA_VISIBLE_DEVICES` is empty and `NVIDIA_VISIBLE_DEVICES=void`; the driver checks for absent NVIDIA/nvhost device nodes before patching.

The writable container root filesystem is intentional: patch outputs and the copied FA2 binary live in its disposable overlay, while the source repository and original fork remain read-only. The driver executes only the NVFP4 LM-head file patcher followed by `q1_reference_fa2_install.py`; the latter imports and calls only the FA2 patcher's `_patch_flash_attn_interface` helper. These operations use file manipulation and compile checks, not model construction. The driver's own no-`torch`/no-`vllm` import assertion is supplemental evidence, not a process-wide tracing claim.

The recorded host interval is **00:44:59.105702–00:45:00.513069 UTC**; container execution is **00:44:59.213939894–00:45:00.02412626 UTC**. The run ID's `004505Z` suffix is an identifier, not its actual start timestamp. Post-inspect reports exited, exit code 0, PID 0, no OOM, and no state error. Both patch stderr files and overall stderr are empty. Copy returned 0. Cleanup source rechecks exact CID and its owned label, then uses checked `docker rm -f` on that CID only; `RESULT.json` records matching removal output as `exact_cid_removed=true`. The raw removal stdout is not separately archived, so this review confirms the recorded checked-cleanup result rather than independently querying present host state.

## Source result and launcher comparison

Five exported files change from their recorded initial hashes: `qwen3_5.py`, `qwen3_5_mtp.py`, `modelopt.py`, `vocab_parallel_embedding.py`, and `flash_attn_interface.py`. The other 18 remain unchanged, including GPU runner, input processor, cache interface, recurrent-state code, and native attention backend. The FA2 receipt reports no problems, fork and installed binary SHA `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`, size 300,123,792 bytes, and interface SHA `dadab8aff63b7f608274834929c954335247366411382f390af92501361044a1`; all source entries cross-check with the exported manifest. The binary was copied and hashed, not loaded or numerically tested.

The two file-producing calls and their order match `ar_agent_serving_v3.sh:135` and `chain_mtp_agent_serving_v3.sh:135`. CPU preparation adds `-B` and changes only the FA2 receipt destination. It intentionally omits the launchers' vLLM version-import check and final serving command. `FR14_REQUIRE_NVFP4_LMHEAD` is absent in the actual preparation environment and is not forwarded by either launcher, so the exported LM-head source uses the same **permissive** compile-time sentinel. This establishes source correspondence, not that a loaded head actually used the intended quantization path.

## Concrete launch-integration blocker

`scripts/fr14_patch_nvfp4_lmhead.py:371–372` explicitly calls `py_compile.compile(..., doraise=True)` on its four outputs. The preparation container has no `PYTHONPYCACHEPREFIX`, so these writes succeed in the overlay. In contrast, the observed AR/chain launchers pass `PYTHONPYCACHEPREFIX=/opt/lumotree-observer/empty-pycache` into the entire container before the patchers, while mounting that directory read-only. `PYTHONDONTWRITEBYTECODE=1` and `-B` suppress automatic bytecode writing; they do **not** suppress an explicit `py_compile.compile` request.

Bounded reproduction: under local UID 501, execute an isolated `python3 -B` with `PYTHONDONTWRITEBYTECODE=1`, set `PYTHONPYCACHEPREFIX` to a temporary directory, assert `sys.dont_write_bytecode`, and explicitly compile a harmless one-line temporary source. With the prefix mode 0555 it returns rc 1 / `PermissionError`; with mode 0755 it returns rc 0 and creates one `.pyc`. A Docker read-only mount imposes the same prohibited write even for container root. The preparation succeeds precisely because its cache environment differs at this stage.

**Minimum correction before launch:** allow a private writable/default overlay cache for file-preparation interpreters, while preserving no automatic bytecode writes, and restore the frozen empty read-only prefix before the final serving interpreter starts. Keep observer activation at that final boundary and retain actual startup-prefix/mount verification. Do not make the serving cache writable or remove compile checks. Validate the repaired preparation/final-exec environment handoff with a connected CPU/stub control; this completed source export need not be repeated solely to establish the already verified file hashes.

The preparation evidence is usable for prospective source-policy assembly. Actual runtime source binding, observer installation, quantization/attention route, model qualification, capacity admission, and workload gates remain unsatisfied by this CPU artifact. Recorded workload attempts and model loads remain zero.
