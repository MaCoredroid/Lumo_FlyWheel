# Workload configuration v3.3.1 delta: independent bounded review

2026-09-28. **PASS for this narrowly scoped source/configuration delta. No new material delta defect found. The Lumo workload launcher is still nonlaunchable for the reasons explicitly preserved in the package; this is not runtime admission or serving qualification. WP remains closed, 0/4.**

The review covers the new host-env/launcher binding and its preservation of accepted v3.3, not the unchanged scientific design or the whole caller. No live adapters, containers, proxy, tunnels, workloads, GPU operations, host-memory probes or memory operations were executed. SSH was used only to read source/evidence files.

## Preserved source identities

Snapshot: `p0/monitor/review-response-20260927/workload-config-v331-delta-reviewed-20260928T1955Z/`. The remote manifest was stable before/after capture. All 13 delivered files match declared bytes and SHA-256; all 42 indexed dry-run evidence files match their hashes. The snapshot contains 66 files, plus two additional exact caller source files pinned by the submitted tests.

| File (campaign-relative) | SHA-256 |
|---|---|
| `workload-case-study-v1/MANIFEST-configurations-v3.3.1-DELTA.json` | `47dba337b9c7102aa4adab0a2553f8c7dd64f2b023bc0ba69975fd445d65ad18` |
| `workload-case-study-v1/CONFIGURATIONS-v3.3.1-DELTA.json` | `bf72ceaedcd9e6fc0cd4327677bc00c0a70c3d7d2ce0ee7799fd779bdf00db52` |
| `workload-case-study-v1/configurations_v3_3_1_delta.py` | `d6eefd25d66ace423e89c0b74492cc1c788aa4cf2ac4f6cec7e3bfd62e1d91af` |
| `workload-plan/launchers/make_lumotree_workload_owned_v1_1.py` | `070ec3423ed093d76565c7fd0075ad1ddfb7efcbda4e535d28d09ad8e54bee94` |
| `workload-plan/launchers/lumotree-workload-host-env.v1.1.env` | `74b0043ec76d9c5de851207d600e1ffe076e59f3194f1d0516afa636a922d61b` |
| `workload-plan/launchers/lumotree_workload_owned_v1_1.sh` | `fb50186f8deb9a87ee625714067d1b2efdbd84db3c7a10d7b1c1cefe7a79d681` |
| `workload-plan/launchers/LUMOTREE-WORKLOAD-OWNED-v1.1.json` | `e0d6671f59104aff7b2d5b4b91a2c2ec70d11c1e1b2c07ad9bc8fd96362afc40` |
| `tools/test_log.lumotree_workload_launcher_v1_1.attempt1.txt` | `87551be4ecca4e788a6fdba8816084f029fb78acdc6ecf6f36e30e41914e3b81` |
| `runs/lumotree-workload-launcher-dryrun-v1_1/attempt1/INDEX.json` | `f20159c42fc51856d798d450c7b2cf46cae9e4ece7dee1c9df9ee3708b84b4a8` |

Snapshot inventory SHA: `791dc8cc1cd0cac6020f64c2d85609a9b89f1bf1ae9a6bea3eac05d51d9752ab`. Additional dependency inventory SHA: `92fbedeb0397994a12b7f1507ac7c6fe31dcd8534e0aada6814eb2b8add2dc61`.

## Source disposition

- **Preservation verified.** The copied base configuration and manifest are byte-identical to the accepted v3.3 snapshot (`a08a31e3…`, `d6e5e4f4…`). All four workload-v1 artifacts retain the recorded hashes. Applying the delta changes only `records.LUMOTREE.{launcher,argv,host_env,metrics_endpoint}`. The entire common profile and the other three arms remain identical, including the two-host boundary, compaction 20,000, sampling/budgets, parser/cache declarations and four-attempt scope.
- **The approved repair is real and bounded.** Generator lines 124–196 replace exactly four empty private-sidecar assignments with guard/unset pairs and add only `SWE_CONCURRENCY=1` and `PYTHONDONTWRITEBYTECODE=1`. Every other env line is unchanged and remains in the same order, including existing duplicate assignments. Lines 200–236 make only the four recorded launcher edits: banner, env hash, env source path and preamble contract. Independent pure regeneration reproduces the env, launcher, both diffs and binding JSON exactly. The source refuses missing anchors and prohibited env mutations.
- **The refusal is at the actual consumption point.** The new env is sourced in the launcher's main `set -euo pipefail` shell at line 35. Local isolated-shell controls confirm clean and inherited-empty cases unset all four names; each nonempty case exits 2 before reaching the post-source checks and emits only the name, never the injected value. Only the grammar-checked assignment/guard env was executed locally, not the launcher. The retained launcher-private membership guard is still at lines 1884–1890, and the B1 selector still enforces concurrency and source provenance at 2617–2621.
- **The caller refusal description is source-supported.** Exact test-pinned `boot_v3_3.py` (`25396c11918a00b1334e5aa9c343ef83504dd835fb62ae88f163de60e23bf78f`) builds the launch environment at 204–218, preserves runner state at 391–394 and rejects nonzero launcher status at 399–400. `sole_executor_v3_5.py` (`05646194b944c62a2dda726f17190c9a3f3aa229f78934ac738228af10c52117`) handles that failure at 547–555. This verifies the claimed connection; it does not independently qualify every caller failure path or the final root package.
- **Dry-run results are honestly scoped.** The saved B/B2 trace gets past the private-name guard and both B1 clauses, then fails the empty patch-source clause. The retained A failure, C0–C3 early refusals, and hypothetical D/D0 failures are not promoted to successful boots. The 42-file evidence inventory matches, and all nine recorded control results are nonzero, untimed-out, quiescent and without a created CID or Docker call. Independent inspection of B's compressed raw xtrace confirms two passing `1 == 1` clauses followed by the empty patch-source comparison and exit 2. The full connected dry-run and the author's 32-test suite were not rerun by this reviewer.

## Explicit remaining launch boundaries

The delta correctly leaves Lumo overall **BLOCKED** and preserves the five existing blocked fields. Its added source-bound boundary list is not an admission receipt:

1. The real v1.1 path stops at missing `FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256`; the selected credential/patcher pair still needs an explicit bound decision (launcher 2618–2621).
2. The scikit-learn workload identity is still unadmitted (2807).
3. The env still omits `TREE`; the inherited default does not satisfy the fixed32 topology validator (3740, 4112).
4. `GPU_UTIL=0.7` remains numerically the intended setting but conflicts with the inherited exact string requirement `0.70`; the workload preamble also requires `0.7` (37, 4713, 4773). This package intentionally does not silently normalize either side.
5. The production host-memory/zero-swap guard remains intact (6920–6963). Its stored standalone read-only refusal is historical evidence, not a current host-capacity observation or authority for recovery.

These are declared prerequisites for a later separately reviewed launcher/root delta. No guard bypass, synthetic workload identity, hypothetical topology, or memory action should be inferred from this review. No additional broad benchmark or unchanged comparability review is required to accept the present configuration delta itself.

## Independent CPU controls

Command:

```sh
cd /Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-config-v331-delta-reviewed-20260928T1955Z
python3 independent_controls.py
```

**50/50 PASS.** Covers manifest/snapshot identity, base preservation, exact overlay paths/hashes, unchanged common/other-arm settings, pure regeneration, seven generator negatives, four missing-anchor negatives, six isolated env-source controls, retained dry-run status, and the raw B trace. Script SHA `ca3c8f967e267cfe932aadc4c71a1191c6b733d76a2ac96c9e15b871185ea0fe`; result `independent_controls.json` SHA `a288c099567dc5d540af2719f933e7c607686abcbc0dedea1dcc676f237ebd04`.
