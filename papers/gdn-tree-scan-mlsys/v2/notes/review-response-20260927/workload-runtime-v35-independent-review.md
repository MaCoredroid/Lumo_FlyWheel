# Workload caller v3.5: bounded independent integration review

**Disposition: changes required before source freeze acceptance; not runtime-ready.** One new tunnel-probe lifecycle defect is reproduced below. The reviewed two-host routing, immutable bundle validation and owned proxy binding show no additional blocker in this bounded pass. Previously accepted caller/proxy v3.3 R1–R4 and F1/F2 remain closed. WP stays closed, **0/4**: exactly one scikit-learn__scikit-learn-9288 attempt each in AR → CHAIN_MTP → SGLANG_EAGLE → LUMOTREE order; no retries, pilots or tuning added.

## Authenticated submission

Snapshot: `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/workload-runtime-v35-reviewed-20260928T192822Z`. `SNAPSHOT-complete.json` authenticates 586 references across 370 fetched files, including configuration delivered-file lists and recursively referenced manifests. Root canonical digest also verifies. Older snapshots and accepted source bytes are preserved; review artifacts are additive.

| Seal | SHA256 |
|---|---|
| Root v3.5 | `085aa0de92dcec9ffe1f16fefab7608895dfab46a6979d85be3288b1ab996b78` |
| Caller v3.5 | `1df5b21472364977490fac4d8edfdbf7569be21c9a6755ef021594602820aa93` |
| Configurations v3.3 | `d6e5e4f4c7b09a55e9d263baa243841064824513e02c3c3f37efc5ec55a30e27` |
| Independent evidence (533 files) | `cb9e6981ac4f1a9103c8dd3ecb0ed6670cb87d1d2f290c6190e7189b3573387c` |

The root review request and later addendum 1 were retained separately. The addendum identifies the Lumo workload env's four empty private exports and required concurrency/venv/swap guards; it is an acknowledged launcher/config successor task, not a newly invented review condition.

## New blocker: probe subprocess can survive cancellation while quiescence passes

`tunnel_owner_v1.py:141–161` creates a separate SSH readiness/probe process. `communicate()` handles TimeoutExpired but not KeyboardInterrupt/SystemExit or other post-Popen exceptional exits. Those probe processes are not stored in the owned tunnel group. During pre-start cancellation, `start():203–205` calls `_fail`, which sees no forwarder (`self.proc is None`) and writes a successful `no_process` quiescence record (`:242–243,277–289`). The spawned probe is neither terminated nor reaped; its incomplete record is not appended. Readiness and post-stop use the same probe routine.

Exact CPU reproduction using the submitted class with injected Popen/killpg: **one probe spawned; one communicate; zero killpg; zero poll; `quiescence.proven=true`**. No subprocess was actually launched. See `REVIEW-probe-cancellation.py/.log` in the snapshot. The source SHA is `6aa8f3f7db221aae2dbe88fad448aa94e264ffa4e0ff68d2e33030a9a2bed2bb`.

Minimal repair: own and retain each probe process group, handle every post-Popen exit with bounded termination/reaping, and require all probe groups plus the forwarder to be proven quiescent before the final tunnel record can pass. Retain interruption/error observations. The timeout branch's final `communicate()` also needs a bound. This is the new probe path; it does not reopen the accepted launcher-group implementation. Parent already relayed this defect for a versioned fix.

## What the changed integration does establish

- **Client routing.** `two_host_v1.py:212–240,386–454` constructs explicit DGX and Alienware clients, checks distinct daemon identities, routes registered server CID/image observations to DGX, and binds agent/evaluator transport to Alienware. Caller boot/release uses the server client; creation, inspect/wait/control via the runtime transport, bundle logs, evaluator operations and exact-CID sweep use the agent client. Independent injected SDK tests confirm no cross-daemon CID lookup and reject swapped clients.
- **Proxy ownership and accounting.** `proxy_owner_v2.py:226–273` verifies the actual reviewed module bytes, starts `python -I -B`, binds the capture run/profile/env and owned-server origin, then proves listener ownership. Caller stops the tunnel and proxy before reduction. `instrumentation_v3_3.py:34–104` verifies ready/stopped receipts, exact run/producer/CID, profile application and unchanged capture bytes, then calls the accepted reducer. No event count is promoted to token count. Completed non-streaming compaction still yields NOT_REDUCED; the producer inventory retains its request.
- **Bundle and cleanup.** Bundle source/mount/image/network checks are real call sites before agent creation; the remote read-only bundle outputs are recomputed using the bound manifest algorithm. Agent-host cleanup and `host-binding.json` are checked before advancing. Unknown/running/unproven actors cannot be silently removed or treated as clean. Four-row and no-retry checks are preserved.

## Smallest remaining integration obligations

These are implementation/readiness gaps already exposed by the configuration/root submission, not a request for more experiments.

1. **Patch transport.** `runtime_v3_2.py:159–162` / `agent_v2.py:19–33` read and stat a local patch, while the actual agent writes on Alienware. The root explicitly has no fetch. Add a source-approved transfer from the exact exited agent CID, retaining raw bytes, archive/member metadata and origin/freshness evidence before local prediction normalization. Do not fabricate an empty patch when absent. The approved local evaluator interface can then consume those authenticated bytes; a shared mount must likewise be explicit and verified.
2. **Actual settings consumption.** Caller static checks enforce response cap, endpoint, bundle and network, but do not invoke `configurations_v3_3.check_compaction_conditions` or inspect all visible settings. An independent pure control confirms the caller accepts `QWEN_MC_KEEP_RECENT=999`, which the accepted S7 contract forbids. Wire the already accepted checker to the actual rendered environment plus observed system/user/project settings before start, retaining absence proofs or settings bytes. This closes the previously pending call-site obligation and requires no new configuration choice.
3. **Runtime probe/admission.** The boot instrumentation producer and actual route/settings evidence are still absent; do not replace those observations with frozen intended values. Supply the existing specified host/network/forward-permission/free-port checks and per-arm route records through the parent-controlled admission process. The Lumo env/config successor noted above must be sealed before inclusion.
4. **Observation label.** Caller `:596–602` checks tunnel liveness only after `runtime.execute()` returns. That method includes official evaluation (`runtime_v3_2.py:176–180`), so `OWNED_TUNNEL_LOST_BEFORE_AGENT_TERMINAL` is too strong when evaluation is enabled: the tunnel may have died after the agent finished. Place the observation at the actual agent-terminal hook or name its actual later boundary; preserve any failure without inferring an earlier time.

## Interpreter question resolved from installed bytes

The worker's Paramiko limitation is **real for the installed docker-py 7.1.0 source**. The actual venv is `/home/mark/shared/lumoFlyWheel/.venv` (worktree `.venv` paths resolve there). `docker/transport/sshconn.py` SHA `4282b59fb44db92ac239b249b101470b7ef001f1cc6c688db626e3fb387bda88` unconditionally imports Paramiko at line 9. `shell_out=True` skips constructing the Paramiko client only at lines 169–172, after module import. `docker/api/client.py` SHA `1542d486504238d251a6fd668bb84eb8a56f3f7f2e3d15fbba2ee403a8bcb550` suppresses the adapter ImportError at 51–54 and raises missing-Paramiko at 179–188 when SSHHTTPAdapter is undefined. The installed metadata confirms version 7.1.0. Read-only `find_spec` under that exact Python finds Docker and returns None for Paramiko. No Docker client or SSH forwarding probe was constructed. Sources and receipt are in `installed-docker-source/`. A caller environment with the required import is needed, but this review installs nothing and does not choose a new transport.

## Validation limits

Eight independent injected CPU controls passed: delivered-placeholder refusal, bundle recompute, explicit two-client routing, wrong/same daemon, and wrong bundle mode/endpoint/network. A separate control reproduces the new probe defect; another exposes the known settings call-site gap. Subprocess, thread-server and socket-connect operations were guarded to raise in this reviewer driver; no live proxy/tunnel/daemon/model/evaluator ran. A tiny pytest decorator/assertion shim invoked selected functions; this is not a full local pytest run. The first reviewer-driver attempt lacked the inert engine fixture's `log` list; its failed log is retained before correcting only that fixture.

Both immutable worker caller logs are retained and report 52 passed (attempt1 SHA `569671471651b7e3e3603a6de9f251e48480c376f88a6fc33f30a42b95c52fdc`; attempt2 SHA `1a07c0ba8b046c0437756eba11f755915e1d411e297aeb88516d2e7b33dd0b7f`). Their loopback subprocess tests were not rerun under this review's no-live-process scope. No accepted source, gate, parameter, task order or budget changed.
