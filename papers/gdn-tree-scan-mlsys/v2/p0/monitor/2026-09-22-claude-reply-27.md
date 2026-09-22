# Reply 27 — E1 18-cell campaign LAUNCHED (GO received)

- Root: `experiments/out-20260922T100028Z-e1-18cells/` — runner pid 3335600, start 2026-09-22T10:00:28Z, command as recorded in STATUS (`E1_QUAL_MANIFEST=e1/e1_qualification_manifest.v1.json bash e1/e1_run_cells.v2.sh e1/e1_cells.v1.json <root> 1 18`, ambient GPU_UTIL/MAX_MODEL_LEN/SPEC_CONFIG/TREE unset).
- Pre-launch guard (measured 10:00:2xZ): no container, no loop/runner, GPU idle, MemAvailable 102 GiB, swap 0 (the parent's 09:13Z preconditioning holds; each boot re-runs the launchers' guarded recovery outside timing).
- Campaign snapshot: 21 files, SHA256SUMS `d1ab57c53464ef81`; identity guard lines: 8 (patcher, tree kernel, decode modes, device multidraft kernel, fused tree conv, OOM guard, HEAD).
- Cell 1 (block 1, native-5, B1, in-boot untimed preflight) booting: container `e1-cell01-100029` up at 10:00:29Z.
- Protocol in force: serial frozen order; native first boots preflight → live pass/fail → fixed warm-up → timed; final sealed-ledger audit after shutdown; F1 terminal seal; stop on INVALID; failures / insufficient support preserved; no re-run of measured cells. I will report each sealed cell's status/rate in STATUS as it lands and stop only per protocol.
Written 2026-09-22T10:01:19Z.
