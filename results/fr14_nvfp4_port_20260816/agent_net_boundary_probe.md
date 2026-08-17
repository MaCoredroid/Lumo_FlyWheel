# FR14 AGENT NETWORK BOUNDARY — built, probed, fail-closed

**Closes the STOP finding `3231eeff7` / `agent_network_egress_finding.md`.**
The agent no longer has a route to the internet. It has exactly one reachable
destination — the offload proxy — and everything else fails in under 20 ms.

Measured on alienware (`mark-Alienware-Aurora-ACT1250`, x86_64, Docker 27.5.1,
iptables 1.8.10 nf_tables) on 2026-08-17. Machine-readable companion:
`agent_net_boundary_probe.json`.

---

## 1. The headline, on the exact image of the leaked task

Same instance image, same python one-liner the finding recorded, only the
network changed. `swebench/sweb.eval.x86_64.astropy_1776_astropy-13236:latest`:

| check | `--network=bridge` (no boundary) | `--network=fr14-agent-isolated` |
|---|---|---|
| `http://<gw>:8023/` (offload proxy) | `URLError [Errno 111] Connection refused` 0.0011 s | **`HTTP 200`  0.0024 s** |
| `pull/13236.diff` — **the gold patch** | **`6375 BYTES FETCHED`  0.3524 s** | `URLError [Errno -3] Temporary failure in name resolution`  **0.0155 s** |
| `v5.2/CHANGES.rst` | **`544305 BYTES FETCHED`  0.3502 s** | `URLError [Errno -3] …`  **0.0141 s** |

The 6,375 B is byte-identical to the finding's line 271
(`pull/13236.diff -> SUCCEEDED 6,375 B`). The left column is not a hypothetical:
it is the leak, reproduced today, on the same image, in 352 ms.

Same image, more shapes, boundary ON (`…astropy-14995`):

```
proxy_gateway              200                                                            0.0032s
GOLD_PATCH_13236           URLError: <urlopen error [Errno -3] Temporary failure in name resolution> 0.0165s
pypi                       URLError: <urlopen error [Errno -3] Temporary failure in name resolution> 0.0133s
dns_github                 gaierror: [Errno -3] Temporary failure in name resolution      0.0002s
raw_ip_1.1.1.1:443         OSError: [Errno 101] Network is unreachable                    0.0000s
host_ssh_tailscale         OSError: [Errno 101] Network is unreachable                    0.0000s
--- curl (the shape qwen-code DOES deny) ---
curl_github=000 time=0.000052s   curl exit=6
--- git (a shape nobody denies) ---
fatal: unable to access 'https://github.com/astropy/astropy/': Could not resolve host: github.com
```

`git`, `curl`, `urllib`, a raw IP with no DNS at all — the boundary does not
care which one the model picks, which is the entire point. The finding's
argument was that a denylist of command shapes is theatre because the next route
is `requests`/`socket`/base64. There is no "next route" here.

---

## 2. What was built

`scripts/fr14_agent_network_boundary.sh` — idempotent
`apply | verify | fingerprint | gateway | probe | control | status | teardown`,
run on alienware over ssh with the script piped on **stdin**, so the remote copy
can never drift from the repo copy.

**Topology.** `fr14-agent-isolated`, bridge `fr14agent0`, `172.31.99.0/24`,
gateway `172.31.99.1`, created `--internal`, `enable_icc=false`,
`enable_ip_masquerade=false`, no IPv6.

**Rules** — 9, every one scoped to `-i/-o fr14agent0` and comment-tagged
`fr14-agent-boundary`, so nothing can match `app_default`,
`lumo_auto_alpha_default`, `docker0` or any host interface:

```
-A INPUT -d 127.0.0.1/32 -i fr14agent0 -p tcp -m tcp --dport 8023 -m comment --comment fr14-agent-boundary-allow-proxy -j ACCEPT
-A INPUT -i fr14agent0 -m comment --comment fr14-agent-boundary-deny-input -j REJECT --reject-with icmp-port-unreachable
-A DOCKER-USER -i fr14agent0 -m comment --comment fr14-agent-boundary-deny-forward-src -j REJECT --reject-with icmp-port-unreachable
-A DOCKER-USER -o fr14agent0 -m comment --comment fr14-agent-boundary-deny-forward-dst -j REJECT --reject-with icmp-port-unreachable
-A PREROUTING -d 172.31.99.1/32 -i fr14agent0 -p tcp -m tcp --dport 8023 -m comment --comment fr14-agent-boundary-proxy-dnat -j DNAT --to-destination 127.0.0.1:8023
-A POSTROUTING -s 172.31.99.0/24 -d 127.0.0.1/32 -o lo -p tcp -m tcp --dport 8023 -m comment --comment fr14-agent-boundary-proxy-snat -j SNAT --to-source 127.0.0.1
-A INPUT -i fr14agent0 -m comment --comment fr14-agent-boundary-deny-input6 -j REJECT --reject-with icmp6-port-unreachable      (ip6tables)
-A FORWARD -i fr14agent0 -m comment --comment fr14-agent-boundary-deny-forward6-src -j REJECT --reject-with icmp6-port-unreachable  (ip6tables)
-A FORWARD -o fr14agent0 -m comment --comment fr14-agent-boundary-deny-forward6-dst -j REJECT --reject-with icmp6-port-unreachable  (ip6tables)
```
plus `net.ipv4.conf.fr14agent0.route_localnet=1` and
`net.ipv6.conf.fr14agent0.disable_ipv6=1` — **per-interface, never `all`**.

**Why the proxy is reached by DNAT.** `relaunch_proxy_remote.sh` sets
`LUMO_PROXY_LISTEN_HOST=127.0.0.1`, and that stays. Rebinding it to `0.0.0.0`
would publish the model endpoint to alienware's LAN. Instead the gateway address
is DNAT'ed onto the loopback bind and SNAT'ed back to `127.0.0.1`, so **the proxy
still sees a `127.0.0.1` peer, exactly as it did under `--network=host`** — zero
proxy-side delta.

**REJECT, not DROP**, everywhere: a blocked agent turn must fail in
microseconds, not stall for the connect timeout. Measured: every negative on the
isolated bridge returned in **≤ 0.0165 s**. On the unbounded control,
`169.254.169.254:80` took **6.006 s** — that is the hang the REJECT prevents.

---

## 3. Runner wiring (`scripts/run_swe_bench_q36_a.py`)

* **All three agent docker-run sites** move off `--network=host`. The live
  fixed32 site is `_instance_agent_command` (`SWE_AGENT_ENV=instance_image`);
  `CODEX_TEMPLATE`/`QWEN_CODE_TEMPLATE` carry `--network={agent_network}`
  resolved inside `_agent_template()`, so a caller cannot forget to pass it.
  Every line containing `docker run ` now carries `--network={agent_network}`
  or `--network={_fr14_agent_network()}`; the three surviving `--network=host`
  strings in the file are prose in comments/docstrings, and the regression test
  greps `docker run ` lines specifically so a comment cannot mask a real site.
* **`FR14_AGENT_NET`** selects the network. Unset → the boundary on the offload
  path, `host` for a local GB10 agent (which fixed32 refuses anyway, and where
  the alienware bridge does not exist). `FR14_AGENT_NET=host|0|off|none|no|""`
  is a loud opt-out.
* **FAIL-CLOSED.** `_fr14_require_network_boundary()` is called from
  `_validate_fixed32_agent_runtime_mode()`, which fixed32 already invokes in
  **three** places: `main()`'s preflight (`run_swe_bench_q36_a.py:10311`,
  next to `_inspect_fixed32_agent_placement_remote`, before any task starts),
  `_run_agent_dispatch`, and `_run_agent_instance`. So the boundary is applied
  and verified in the launcher preflight *without* editing the launcher shell
  script, and re-verified before every individual agent launch. It refuses on:
  boundary disabled, local agent, missing script, verify rc≠0, verify with no
  receipt, receipt for a different network, malformed fingerprint, **and a
  fingerprint that changes mid-run**. `apply` runs once per host per process;
  `verify` runs every time.
* **Endpoint rewrite.** `_fr14_rewrite_endpoint()` re-points
  `OPENAI_BASE_URL` `127.0.0.1|localhost|::1` → the gateway **read out of the
  verified receipt**, not a literal. Change the subnet in the shell script and
  the runner follows.
* The verified observation rides into `runner_metadata.json` under
  `agent.fr14_network_boundary` (net/bridge/gateway/proxy_port/rules/fingerprint).

Live gate transcript:

```
[fr14-net-boundary] FR14_NET_BOUNDARY_APPLIED net=fr14-agent-isolated bridge=fr14agent0 gateway=172.31.99.1 proxy_port=8023 rules=9 fingerprint=e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4
{'net': 'fr14-agent-isolated', 'bridge': 'fr14agent0', 'gateway': '172.31.99.1', 'proxy_port': '8023', 'rules': '9', 'fingerprint': 'e3cc…b5e4'}
--- fail-closed: opt-out refused ---
Fixed32BoundaryError: fixed32 requires the FR14 agent network boundary; FR14_AGENT_NET='host' disables it. host networking gives the agent the offload host's full egress (STOP FINDING 3231eeff7)
--- fail-closed: local agent refused ---
Fixed32BoundaryError: fixed32 requires the agent on the offload host; the FR14 network boundary is an alienware-side facility
--- fail-closed: wrong network name refused ---
Fixed32BoundaryError: fr14 boundary verified network 'fr14-agent-isolated' but the agent would join 'fr14-does-not-exist'
```

---

## 4. Fingerprint mechanism

`fingerprint` = `sha256` of material read **back out of the kernel**, never of
what the script intended to install:

* `docker network inspect` → internal / icc / masquerade / ipv6 / subnet /
  gateway / bridge name,
* every `fr14-agent-boundary`-tagged line of `iptables -S`, `iptables -t nat -S`
  and `ip6tables -S`,
* the two per-interface sysctl values.

Current: `e3cc51795829dca6a7ac83a86e7a8e52f4937b469882bbfb2e70c356c6f1b5e4`.
Stable across repeated `apply` (idempotent), and **identical again after a
teardown + rebuild**. Rule *text* alone would not catch a reorder, so `verify`
additionally asserts by line number that the proxy `ACCEPT` precedes the
catch-all `REJECT`.

**Tamper tests — all three detected, `rc=3`:**

| tamper | result |
|---|---|
| delete the proxy `ACCEPT` | `FAIL: missing rules: filter/INPUT[0]` |
| re-insert it *below* the catch-all `REJECT` | `FAIL: proxy ACCEPT does not precede the catch-all REJECT in INPUT` |
| recreate the network without `--internal` | `FAIL: network fr14-agent-isolated is not --internal` |

---

## 5. Full probe transcript, boundary ON

`python:3.12-slim-bookworm` on `fr14-agent-isolated`, with a **labelled
stand-in** on the proxy port (`python3 -m http.server 8023 --bind 127.0.0.1`) —
the real offload proxy was down, and the positive path cannot be proven without
*something* listening (see §6).

```
proxy_gateway_tcp                      connected             0.003s
proxy_gateway_http                     http_status      404  0.014s   <- DNAT reached the loopback bind
gold_patch_fetch                       failed                0.013s
github_https                           failed                0.012s
internet_ip_tcp                        failed  ENETUNREACH   0.000s
internet_ip_tls                        failed  ENETUNREACH   0.000s
dns_external                           failed                0.000s
dns_docker_internal                    failed                0.000s
metadata_endpoint                      failed  ENETUNREACH   0.000s
host_ssh_via_gateway                   failed  ECONNREFUSED  0.000s
host_open_service_via_gateway          failed  ECONNREFUSED  0.000s
host_ssh_via_lan                       failed  ENETUNREACH   0.000s
host_proxy_via_lan                     failed  ENETUNREACH   0.000s
host_open_service_via_lan              failed  ENETUNREACH   0.000s
host_ssh_via_tailscale                 failed  ENETUNREACH   0.000s
tailnet_peer_vllm                      failed  ENETUNREACH   0.000s
other_stack_bridge                     failed  ENETUNREACH   0.000s
docker_default_bridge                  failed  ENETUNREACH   0.000s
docker_socket_tcp                      failed  ECONNREFUSED  0.000s
host_resolved_dns_loopback             failed  ECONNREFUSED  0.000s
raw_socket_egress                      raw_send_accepted     0.000s   (see §7 — contained)
host_loopback_after_route_add          failed  ECONNREFUSED  0.000s
host_loopback_proxy_after_route_add    failed  ECONNREFUSED  0.000s
default_route: no      cap_eff: 00000000a80425fb      uid: 0
container IPv6: only ::1 on lo (no link-local on eth0)
```

### The negative control that makes the above mean anything

Identical probe, `--network=bridge` (docker default, no boundary):

```
proxy_gateway_tcp                      failed  ECONNREFUSED  0.003s
gold_patch_fetch                       fetched          6375 0.384s   <- THE LEAK
github_https                           fetched        574505 0.764s
internet_ip_tcp                        connected             0.028s
internet_ip_tls                        connected             0.035s
dns_external                           resolved              0.034s
metadata_endpoint                      failed                6.006s   <- the 6-second HANG
host_open_service_via_gateway          connected             0.000s   <- host service on 0.0.0.0:8787
host_open_service_via_lan              connected             0.000s
default_route: yes
```

`host_open_service_*` is the control-of-the-control: alienware really is running
a service bound `0.0.0.0:8787`, an unbounded container really does reach it, and
the same target is `ECONNREFUSED` in 0.000 s on the isolated bridge. The
negatives in §5 are the *rules*, not an absent listener.

**Note on alienware's ssh:** there is no kernel listener on `:22` (`ss -lnt`
shows none) — admin access is **Tailscale SSH**, tailscaled intercepting :22 on
`tailscale0`. So `host_ssh_via_tailscale` (`100.83.202.36:22`) is the real admin
path, and it is `ENETUNREACH` from the bridge.

---

## 6. Honest caveats

1. **The positive path was proven against a stand-in, not the live proxy.** The
   offload proxy was not running; `python3 -m http.server 8023 --bind 127.0.0.1`
   stood in and answered `HTTP 200` / `404 /v1/models` through the DNAT. What
   this proves is the *path* (PREROUTING DNAT → loopback bind → SNAT → reply),
   which is proxy-agnostic. What it does not prove is the proxy's own behaviour
   behind it. First real arm should confirm a `/v1/models` 200 from the actual
   proxy.
2. **`ECONNREFUSED` is ambiguous by construction.** With the stand-in stopped,
   `172.31.99.1:8023` returns `ECONNREFUSED` in 0.0141 s and `…:22` returns
   `ECONNREFUSED` in 0.0000 s — a REJECT and a dead-but-DNAT'ed port look the
   same to `connect()`. That is *why* the positive path needs a listener. (The
   ~14 ms vs ~0 ms split is a weak tell: the DNAT'ed port traverses loopback and
   gets a TCP RST; the rejected port gets an ICMP unreachable at INPUT.)
3. **The rules do not survive an alienware reboot.** Deliberate — no persistence
   daemon and no surprise state on a shared box. The runner's gate re-applies
   idempotently before the first agent of every run, and `verify` (not `apply`)
   is the authority.
4. **CHANGES.rst byte count differs from the finding** (544,305 today vs 544,282
   recorded). Same file, same URL; upstream/transfer difference, not a
   measurement error worth chasing. The gold patch matched exactly (6,375).
5. **The eval is out of scope and unchanged.** The parent brief listed
   `:2323` as the eval site; it is not — it is `_instance_agent_command`, the
   live *agent* site. The eval runs through `scripts/swe_eval_x86_worker.py` →
   swebench's own `run_evaluation`, which manages its own containers on docker's
   default network and cannot be re-pointed without patching swebench. It is
   also not a contamination vector: the eval applies an already-frozen patch and
   runs tests with no model in the loop. Left alone, on purpose.

---

## 7. Red-team: pre-answered

**IPv6.** Alienware holds *global* IPv6 on `wlp131s0f0` (`2601:640:8e00:…`), so
this was live, not theoretical. Three layers: the network is created without
`--ipv6`; `net.ipv6.conf.fr14agent0.disable_ipv6=1` removes the host-side
link-local so there is nothing on-link to aim at; and ip6tables REJECTs INPUT
and both FORWARD directions on the bridge. Measured inside the container:
`/proc/net/if_inet6` contains **only `::1` on `lo`** — eth0 has no IPv6 address
at all, not even link-local.

**DNS over the bridge / docker's embedded resolver.** Measured, and this one is
load-bearing. The container gets `nameserver 127.0.0.11` with
`ExtServers: [host(127.0.0.53)]` — i.e. docker is willing to forward from the
*host* namespace, which would sail past every iptables rule. On
`fr14-agent-isolated` it fails: `gaierror [Errno -3]` in **0.0002 s**.
I ran the ablation — an identical bridge with **identical rules but without
`--internal`** — and there **DNS RESOLVES** (`dns_external: resolved, 0.024 s`)
while TCP egress still fails. So `--internal` is doing real work for DNS
containment, and `verify` therefore *asserts* `Internal=true` and refuses
otherwise (tamper test 3 above). Content egress was contained in both arms;
only name resolution leaked.

**`host.docker.internal`.** Not passed (`--add-host host-gateway` is absent), so
it does not resolve: `gaierror` in 0.000 s. Even if it did, it resolves to the
gateway, and the gateway is `ECONNREFUSED` on every port but 8023.

**Metadata endpoint.** `169.254.169.254:80` → `ENETUNREACH` in 0.000 s
(vs a 6.006 s hang on the unbounded control).

**Container-to-container on the same bridge.** `enable_icc=false` at the network
level, plus `DOCKER-USER -i fr14agent0 -j REJECT` and `-o fr14agent0 -j REJECT`,
which cover bridge→bridge traffic since it is forwarded. Two concurrent agents
cannot talk to each other; neither can anything on `app_default`,
`lumo_auto_alpha_default` or `docker0` reach in (`other_stack_bridge`,
`docker_default_bridge`: `ENETUNREACH`).

**CAP_NET_RAW / crafted packets.** Docker grants `CAP_NET_RAW` by default and
the fixed32 agent runs `-u 0:0`; measured `CapEff=00000000a80425fb` — bit 13
(NET_RAW) **set**, bit 12 (NET_ADMIN) **clear**. So the agent *can* craft
packets and `SO_BINDTODEVICE` past the absent default route, but it cannot add
routes or touch netns config. I measured the crafted path directly, zeroing
counters first: on a bridge **with a default route**, 7 hand-built IP+TCP SYN
packets to `1.1.1.1` produced exactly **`7 pkts, 280 bytes` on
`DOCKER-USER … -i <bridge> … REJECT`** — the rules match on *interface*, not on
routing, so raw sockets buy nothing. Source-spoofing buys less: the same 7
packets with `src=10.0.0.113` (the host's own LAN address) never reached FORWARD
at all (`0 pkts`) — martian-source drop, upstream of the rules. On the
production `--internal` bridge the raw send is accepted by the socket but the
packets die at neighbour resolution and never reach the host: `0 pkts` on
DOCKER-USER. `--cap-drop=NET_RAW` is therefore **not needed**; it remains
available as belt if the parent wants it.

**`route_localnet=1` — the deliberate footgun.** It is set on `fr14agent0`
**only**, never on `all`, so it cannot widen any other interface. A container
still cannot address the host's loopback: `127.0.0.53` from inside resolves to
the container's *own* netns loopback (`ECONNREFUSED`, 0.000 s), and adding a
`127.0.0.0/8 dev eth0` route needs NET_ADMIN, which is absent. And if one ever
got it, the INPUT catch-all rejects every loopback destination except
`127.0.0.1:8023` — measured after an explicit `ip route add` attempt
(`route_add_attempt: ip: not found`, then `127.0.0.53:53` → `ECONNREFUSED`).

**The docker socket.** Not mounted into any agent container, and the daemon has
no TCP listener (`172.31.99.1:2375` → `ECONNREFUSED` 0.000 s). A container
escape via docker-in-docker has no handle here.

**Tailnet.** The agent cannot reach alienware's tailscale address, its own ssh
path, or a tailnet peer (`100.103.10.122:9950`, the GB10's vLLM) — all
`ENETUNREACH`. Forwarding out `tailscale0` is caught by DOCKER-USER regardless.

**Scope blast radius.** Every rule matches `fr14agent0` by interface and carries
the `fr14-agent-boundary` comment. After the ablation network was torn down,
`grep -c fr14ctl0` across `iptables -S`, `iptables -t nat -S` and `ip6tables -S`
returned **0** — teardown is complete, and `app_default` /
`lumo_auto_alpha_default` / `docker0` were never matched by anything.

**What I could NOT close.** (a) A future agent docker-run site that hardcodes
`--network=host` — mitigated by a regression test that greps every `docker run`
line in the runner. (b) The proxy itself: anything the agent can make the *proxy*
fetch is still reachable, because the proxy is by definition allowed. That is a
proxy-side question (`/v1/*` surface), not a network-boundary one. (c) Covert
channels through the allowed proxy connection (timing, request content) — out of
scope for an egress boundary.

---

## 8. Regression tests

`tests/test_fr14_agent_network_boundary.py` — **36 passed**, no docker/GPU/net.
Covers: network selection and every opt-out spelling; no `docker run` line
hardcoding host networking; template rendering through both `.format()` call
sites; the instance-image site; endpoint rewrite including the derived-gateway
property; receipt parsing and malformed-fingerprint rejection; and eight
fail-closed refusals including the mid-run fingerprint change and the gate being
wired into `_validate_fixed32_agent_runtime_mode` rather than merely existing.

Whole runner-importing suite: **699 passed, 1 skipped, 3 failed** — the 3
failures are in `tests/test_fr13_fixed32_ingress_wiring.py` and **reproduce with
this branch's changes stashed**, i.e. they belong to the in-flight serve work,
not to this commit.
