# Workload configuration v3.3: bounded independent delta review

Reviewed 2026-09-28. **PASS for the changed CPU/source configuration contract, with one nonblocking documentation correction below. This is not serving qualification, runtime admission, or permission to start anything. WP remains closed, 0/4.** Caller v3.5 and the final root integration were not part of this sealed submission and remain pending.

## Exact evidence

One read-only snapshot of the sealed remote configuration and its cited new evidence was preserved at `p0/monitor/review-response-20260927/workload-config-v33-bounded-20260928T1816Z/`. The configuration and manifest were unchanged across capture. All 42 delivered manifest payloads and all 57 snapshot members match their SHA-256 and byte counts. The snapshot has no missing members. Paths below are relative to `experiments/review-response-20260927/`, except the explicitly identified parent decision and review artifacts.

| File | SHA-256 |
|---|---|
| `workload-case-study-v1/MANIFEST-configurations-v3.3.json` | `d6e5e4f4c7b09a55e9d263baa243841064824513e02c3c3f37efc5ec55a30e27` |
| `workload-case-study-v1/CONFIGURATIONS-v3.3.json` | `a08a31e3c2ea9bac2c101cc3ac6af69b87f048128e6c71e40cc9367a30fe55cd` |
| `workload-case-study-v1/configurations_v3_3.py` | `a0ffb0cfa33e2ee3d6a08a0c997e9fe5f55de3e53c3521d9099125224a79f2e6` |
| `workload-case-study-v1/README-v3.3.md` | `f1ecf1c240718b2e505e76ad71e0488df383d74a15286b5cd13c5a40438fc71d` |
| `workload-case-study-v1/FLATTEN-BINDING-CHECK-v1.json` | `9909758f79acb42b6d73443669263966deb7a5ab927251188a672aa7156442af` |
| `workload-plan/tools/proxy-capture/generated/inference_proxy_capture_v2.py` | `33d8743496858a6d63b2e39874da8f40ca799afe3d2024ed711fe188a69cb571` |
| Parent `p0/monitor/review-response-20260927/parent-workload-two-host-decision-20260928T1801Z.txt` | `030a6d31c1f741d89e8ac96cac98d4f1ca1e0538755210926067323565411feb` |
| Snapshot `SNAPSHOT.json` | `6a73e0044a06e2982ac550572ef46500ac54b79673bed22dd144abda49484b72` |

The comparison baseline is the previously reviewed v3.2 configuration, SHA `d835381a0573edd3801f056920a5502c982582ad80895fd948893e5135d8f46a`, in `workload-config-v32-bounded-20260928T1720Z/`. The older configuration, results, and manifest were not modified.

## Findings and closure

1. **Option A is concretely specified and internally consistent.** Generator lines 398–440 derive Alienware identity, x86 image, network, and the fixed read-only 0.19.4 bundle from receipts. Lines 442–519 distinguish the DGX server client from `ssh://alienware` with `use_ssh_client=True`, and specify DGX proxy `127.0.0.1:8022`, engine `127.0.0.1:9950`, and the reverse forward from Alienware `127.0.0.1:8023`. The agent uses its bridge gateway `172.31.99.1:8023/v1`. Independent receipt checks confirm the declared Docker ID/name/architecture, image ID/digest, network ID/internal flag/gateway/subnet, and bundle identity. This is a source-bound topology, not evidence that the reverse forward presently works.

2. **The four-arm common profile and previous system choices are preserved.** Removing only the new metrics entries and the explicitly changed SGLang flattening description leaves all four `records` equal to v3.2. Likewise, removing the new topology fields and reverting the formerly unresolved compaction cap leaves `common` exactly equal to v3.2. No model, image, parser/cache declaration, engine seed, sampling, concurrency, context/output budget, or arm-specific launcher setting changed in this record. AR and CHAIN retain the accepted shared patched FA2/APC source options; SGLang retains explicit BF16 KV; Lumo's previously declared workload route is not promoted to runtime-qualified by this update.

3. **Compaction budget 20,000 is correctly distinguished from context thresholds.** Generator lines 263–297 bind the 0.19.4 bundle's `COMPACT_MAX_OUTPUT_TOKENS = 2e4` and the side-query request; the proxy cap of 32,768 does not raise a request already capped at 20,000. The changed settings checks cover the seven declared conditions, including compaction input slimming and microcompaction defaults. Independent extracted-function controls reject missing/duplicate/wrong response caps, differing model names, the six forbidden environment overrides, and all six forbidden settings paths. These are pure checks: the actual renderer must still collect every visible settings source and invoke them. The document explicitly leaves fresh-container user/project settings to runtime inspection (generator lines 290–292).

4. **Prompt rendering claims remain bounded.** The proxy producer is now hash-bound, and source inspection supports its insertion before upstream submission. Independent local replay confirms the embedded and pure proxy functions agree with both earlier helper functions on all six original, hash-pinned requests; unsupported non-string text remains flagged, and empty-array divergence remains disclosed. The new 390-body equality report and its source were reviewed, but that corpus reduction was not independently rerun in this bounded pass. Neither the 396-body flattening report nor the six-request token-parity evidence qualifies tokenization on a future served route. The `PENDING_BOOT` and unsupported-content checks remain explicit (generator lines 660–671, 778–781).

5. **The measurement boundary matches the parent decision.** Generator lines 374–379 and 682–695 separate token intervals measured at the DGX proxy from task wall time, which includes the agent-to-DGX hop. They also record that the proxy and SSH client execute on the DGX for every arm. Endpoint/token/time pairing is still a runtime obligation; the record does not establish engine-only latency or machine throughput. One wording correction is needed in the next documentation revision: **README-v3.3.md:174, “The proxy-to-engine timing has no network in it,” should say “has no inter-host network hop.”** Loopback HTTP, local proxy work, and engine scheduling remain. The precise common measurement-boundary field is already correct, so this is not a blocker to the configuration contract. Preserve this attempted/sealed version rather than silently modifying its bytes.

## Runtime boundary still open

Generator lines 522–557 explicitly require the caller v3.5 client split, agent-host bind-path handling, authenticated patch transfer back to the local evaluator interface, owned tunnel lifecycle, and all six admission observations. The observations include DGX Docker identity, Alienware FR14 rules, current reverse-forward permission, a caller interpreter with the required imports, free ports, and renewed host/image/bundle/network checks. Applying network rules or installing dependencies is not an implicit step in this configuration. The historical forward pattern does not prove current Tailscale SSH forwarding permission.

The forthcoming caller must consume these fields and bind the exact configuration SHA; constructor/cleanup routing, actual rendered settings, remote output transport, proxy ownership and terminal measurement reconciliation still need its own source/freeze review. The sealed configuration already labels those boundaries and uses `launch_authorized: false`. No new configuration-only blocker was found that requires another experiment or broader comparability campaign.

## Independent checks

Executed locally, with only standard-library code and AST-extracted pure functions:

```sh
cd /Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-config-v33-bounded-20260928T1816Z
python3 independent_controls.py
```

Result: **43/43 PASS**, 66 hash-bound inputs. Script SHA `01b7cbc43513c4e3ed89fd635ab750710cf0ab5cab3f3fd4f5e77830d57095b9`; result `independent_controls.json` SHA `211053d3f857df5f428ca8407316c16c532c3e53982514434d0cd23ec4fb262b`.

The author's sealed log (`tools/test_log.workload_configurations_v3_3.attempt1.txt`, SHA `675239bf9b5239d2970fb02c587e6746f70e83afd6f85bc938bb573aec6745f5`) reports 46 tests. This review did not rerun that full suite, tokenizers, the full corpus reduction, any remote observation commands beyond reading saved files, or any runtime entry point. No containers, proxy/tunnel processes, network probes, workloads, GPU operations, or memory operations were performed.
