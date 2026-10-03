# Workload fixed32 ingress: bounded source audit

**The exact submitted workload launcher does not install the fixed32 engine middleware.** Its task/secret-file pins and request-ID setting survive from production, but the active path remains the no-middleware derivative. Therefore the bare `X-Request-Id` probe is not demonstrably missing engine authentication on this source. The smallest repair should first bind the actual selected task and the intended existing route; switching to authenticated ingress is a separate integration change, not a necessary consequence of those retained boot variables.

Scope: local source only; exact snapshot hashes are below and in `SOURCE-SNAPSHOT.json`. No secrets/config credential files were read, no engines/proxies imported or executed, no SSH/GPU/process actions, no gates or experiment counters changed. The parent's request-observer randomization repair is outside this review.

Paths beginning `W/` below mean `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/`; other paths are repository-relative.

## 1. Actual route and immediate caller gap

`W/launchers/lumotree_workload_owned_v2_1.sh:5390,5550` assign `FR13_FIXED32_MIDDLEWARE_FLAGS=""`; lines5545–5549 explicitly document the no-middleware branch. The empty value is sent to Docker at7188 and expanded in the vLLM command at8364. There is no nonempty assignment or actual `--middleware` installation in this launcher. Lines5396–5402 separately pin `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`. Those two facts must not be conflated.

The workload preamble at70–72 still requires `FR13_FA2_QROW32_B1_TIERB_WORKLOAD`, `FR13_FIXED32_INGRESS_TASK_IDS`, and `FR13_FIXED32_INGRESS_SECRET_FILE`; lines5403–5453 enforce a real mode0600, non-symlink, schema-valid secret file. `W/tools/attempt-runtime/sole_executor_v3_7.py:615–620` prepares capture/observer environments, but has no fixed32 ingress provisioning or task-bearer helper. `boot_v3_3.py:204–220` inherits host environment and overlays `boot.env`/`extra_env`; it stamps `LUMOTREE_TASK`, which does **not** substitute for the separate ingress/task pins. An operator-inherited value can therefore be present without being an owned selected-task lifecycle. This review did not read live environment or claim those variables are currently absent.

For **`scikit-learn__scikit-learn-9288`**, supplying that single ID alone still fails the current host contract. The launcher's tier-B workload table at1209–1277 admits the named Astropy subsets or synthetic random1024 identity; the selected task is absent. Lines2830–2839 require declared task IDs/subset hash to match that table. Lines5462–5484 admit either a specifically pinned B1 diagnostic task or the evidence-set counts. `scripts/fr13_floor_gate.py:133–179` lists counts4,10,12,15,16, and the diagnostic profiles at181 onward are Astropy. The corresponding in-container validator is `scripts/fr13_patch_fa2_tree_bias.py:6075–6156`; fixing only the host label would leave a second refusal.

**Minimal current-route closure:** add a source-bound, truthful identity for this selected task through the workload generator/host and container validation chain, retain the existing FA2 binary/numerical credential requirements, and have the owned caller supply its exact task/subset binding and private per-attempt secret-file pointer. Do not label this task as an Astropy set, random1024, or the earlier Q1 no-SWE diagnostic. Explicitly record that engine and proxy ingress are disabled if preserving these current route semantics. No auth lifecycle should be claimed from merely mounting a secret file.

## 2. Proxy ownership does not currently enable authenticated ingress

`W/tools/attempt-runtime/proxy_owner_v2.py:48–50,104–110` rejects frozen environment keys matching `SECRET|...`, including **the secret-file pointer name** `LUMO_PROXY_FIXED32_SECRET_FILE`. Its explicit child environment at165–170 does not inherit engine variables. The generated proxy's `Fixed32ProxyIngress.from_env` at1021–1041 returns `None` when all three `LUMO_PROXY_FIXED32_{SECRET_FILE,TASK_IDS,LEDGER_PATH}` values are absent, refuses a partial set, and constructs ingress only when all are present. Thus the existing owner cannot activate this path through its accepted generic environment contract.

`sole_executor_v3_7.py:639–644` starts the ordinary owned proxy;669–677 executes the engine-only probe, records the post-probe baseline, then creates the agent. Its stop path at551–566 and717 stops tunnel/proxy and settles capture. It has no authenticated begin/finalize calls. **That is consistent with the current disabled-ingress route**, rather than proof of an omitted mandatory lifecycle. Enabling the three proxy settings without a corresponding lifecycle would produce401 for absent/invalid task bearers and409 before `begin` (`generated/inference_proxy_capture_v2.py:1043–1059,1078–1131,5550–5603`).

## 3. Capture policy: no conflict with capture v2

`generated/inference_proxy_capture_v2.py:1031–1040` prohibits exactly the legacy full-content dump variables `LUMO_PROXY_PAIR_DUMP_DIR`, `LUMO_PROXY_REQUEST_DUMP_DIR`, and `LUMO_PROXY_SSE_DUMP_DIR` when authenticated ingress is enabled. It does **not** prohibit `LUMO_PROXY_SSE_CAPTURE_DIR` or `LUMO_PROXY_CAPTURE_RUN_ID`, which the owned capture-v2 profile requires at3951–3974 and the owner sets at165–170. The usage/timestamp/body-hash observation path and the old debug dump path are different code paths. Do not disable the reviewed measurement capture or weaken the legacy-dump guard to resolve a nonexistent conflict. Seven bounded AST-only controls against the actual `from_env` method confirmed disabled, partial, capture-v2-compatible and all three forbidden-dump branches, using synthetic strings and a fake constructor that never opens a secret file.

## 4. If authenticated ingress is deliberately chosen later

This requires a complete small lifecycle, not just an Authorization header:

1. Give the owned proxy an explicitly validated private secret-file pointer, canonical task IDs and per-attempt ledger through a typed owner-controlled channel; keep secret bytes out of generic frozen environment/evidence. The agent receives only its derived task bearer, not the engine control bearer/root secret. Existing derivation/reference behavior is in `scripts/run_swe_bench_q36_a.py:374–391,519–538`.
2. Begin the engine and proxy with exact canonical task-set payloads and the engine control bearer; require receipts before admitting task calls. Proxy endpoints are `/admin/fixed32/ingress/{begin,finalize}`; engine endpoints `/fr13/fixed32/ingress/{begin,finalize}`. Generated proxy6346–6400 authenticates these separately; the proxy mints upstream wire IDs and overwrites the engine bearer/task-key/request-ID headers at5841–5889. Do not forward the root bearer to the agent or synthesize task evidence.
3. Resolve the probe boundary explicitly. `runtime_probe_producer_v1_2.py:425–428` requires engine loopback and bypasses the measured proxy. At650 it POSTs `/v1/chat/completions/render`; at651 it sends a chat request with only `X-Request-Id`. If actual engine middleware is installed, `src/lumo_flywheel_serving/inference_proxy.py:2257–2278` rejects the render POST with403 regardless of auth, and2330–2378 requires engine bearer, registered task key and a valid `fr13-chat-...` wire ID for chat. The engine must also be in campaign phase before acceptance (`accept` at1991 onward). Preserve the existing pre-agent probe exclusion from measured counters at`probe_boundary_v1.py:168–198`; do not silently count this non-task probe in measured ingress task totals.
4. Redact before retaining HTTP evidence: `runtime_probe_producer_v1_2.py:625–631` currently stores supplied `request_headers` verbatim on success and error. Adding bearer headers directly would persist credentials. The current X-Request-Id-only call is safe from that specific leak. Agent command/env evidence likewise needs a private credential handoff rather than interpolated bearer text in retained commands.
5. After agent terminal, drain active requests, query task evidence, finalize proxy and engine while both still run, then stop the proxy/server. `Fixed32ProxyIngress.finalize` at1300–1335 refuses active/incomplete work. A process-group stop alone is not an authenticated finalization receipt. Preserve failures and cleanup on every path without retrying the task.

No part of this conditional design is an approval to change the current protocol or run a model. For the presently submitted no-middleware path, the priority is the honest task-identity binding and the separately owned request-ID observation fix, followed by CPU controls proving actual rendered middleware/proxy environment and task-ID agreement.

## Source bindings

- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/launchers/lumotree_workload_owned_v2_1.sh`: `3d186ac2f44781527f659ac2b4a369d0f8110c7c82f887b8386d846486060943`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/sole_executor_v3_7.py`: `88f9519022c51511dd7546f5752f0c272651136f8e63623cd781ff9fcdfa3595`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/probe_boundary_v1.py`: `bf5bc8cb1989522abeb1d61a95d7cb1b1aea4e1467d11effdda417790ab2e7f6`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/proxy_owner_v2.py`: `a1ab225df460d78c5d6dc7dace2b25dd4b321129a1a0a288f05dc4e11c071ad2`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/boot_v3_3.py`: `25396c11918a00b1334e5aa9c343ef83504dd835fb62ae88f163de60e23bf78f`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/agent_v2.py`: `162e19ae11fbb70e22eb726344825b5709b3b356abfc1305792795b72b17588a`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/runtime-collectors/runtime_probe_producer_v1_2.py`: `4ea777422afd973ccb22ed7dcefbf654dd516c15266b56892223b812b6f092bf`.
- `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/proxy-capture/generated/inference_proxy_capture_v2.py`: `33d8743496858a6d63b2e39874da8f40ca799afe3d2024ed711fe188a69cb571`.
- `src/lumo_flywheel_serving/inference_proxy.py`: `1849c8f24908fe1cbcd3f0d65e564a8e3b4561523164aaeb1e14099dad911841`.
- `scripts/run_swe_bench_q36_a.py`: `a924716508292e5a86ae7cde1ee09794dd229665f2a42b958f26cd06fba224c9`.
- `scripts/fr13_floor_gate.py`: `df023a31e1fb2a86c9f1c3a8834e199160b373510b0e1a0da045f9175fbe5025`.
- `scripts/fr13_patch_fa2_tree_bias.py`: `8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2`.

Snapshot and seven-control output are under `p0/monitor/review-response-20260927/workload-fixed32-ingress-source-audit/`; `REVIEW-SEAL.json` binds their hashes. This is a review snapshot, not a launch freeze.
