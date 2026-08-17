#!/bin/bash
# FR14 agent network boundary — runs ON ALIENWARE (the agent host).
#
# Instance/agent containers move from --network=host to a dedicated bridge
# where the ONLY reachable destination is the offload proxy: the bridge
# gateway's :8023 is DNAT'ed to the proxy's loopback bind (route_localnet),
# every other forwarded packet is REJECTed (fast-fail, no hanging turns),
# and every other INPUT from the bridge is REJECTed (host services shielded
# from the agent). The proxy's own bind/exposure is unchanged.
#
# Why: Qwen3.8 routes around tool-level network denial via shell python
# (see results/fr14_nvfp4_port_20260816/agent_network_egress_finding.md);
# 5 of 14 FR14 traces reached the internet, two fetched their own gold
# patch. A network boundary is the only non-heuristic fix, and exact16 QC
# is gated on it.
#
# Idempotent: safe to run in every launcher preflight. `verify` runs a probe
# container asserting the boundary from inside. Rules do not persist across
# reboot by design — the preflight re-applies.
#
# Subcommands:
#   apply       create/repair the network + rules + sysctls, print fingerprint
#   verify      assert every element is present; exit!=0 otherwise (fail-closed
#               hook used by run_swe_bench_q36_a.py before any fixed32 launch)
#   fingerprint print the rule/topology fingerprint only
#   gateway     print the bridge gateway IP (the agent's OPENAI_BASE_URL host)
#   probe       run a container ON the bridge and assert the boundary from
#               inside it; emits JSON on stdout
#   control     the SAME probe on docker's default bridge (no boundary) — the
#               negative control that proves the probe can see egress at all
#   status      human-readable dump of every element
#   teardown    remove the rules and the network
#
# Scope discipline: every rule matches ONLY the FR14 bridge interface, and
# every rule carries the `fr14-agent-boundary` comment tag. Nothing here can
# match alienware's other docker stacks (app_default, lumo_auto_alpha_default)
# or any host interface.
set -euo pipefail

NET=${FR14_AGENT_NET:-fr14-agent-isolated}
BRIDGE_IF=${FR14_AGENT_NET_BRIDGE:-fr14agent0}
SUBNET=${FR14_AGENT_NET_SUBNET:-172.31.99.0/24}
GW=${FR14_AGENT_NET_GATEWAY:-172.31.99.1}
PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
PROBE_IMAGE=${FR14_NET_PROBE_IMAGE:-python:3.12-slim-bookworm}
TAG=fr14-agent-boundary
SCHEMA=fr14-agent-net-boundary-v1

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo -n"; fi

log() { echo "[fr14-net-boundary] $*" >&2; }
die() { echo "[fr14-net-boundary] FAIL: $*" >&2; exit 3; }

ipt()  { $SUDO iptables "$@"; }
ipt6() { $SUDO ip6tables "$@"; }

# ---------------------------------------------------------------- rule specs
# Order inside each array IS the installed order (index 0 ends up first).
# The proxy ACCEPT must precede the catch-all REJECT.
INPUT_RULES=(
  "-i $BRIDGE_IF -p tcp -d 127.0.0.1 --dport $PROXY_PORT -m comment --comment ${TAG}-allow-proxy -j ACCEPT"
  "-i $BRIDGE_IF -m comment --comment ${TAG}-deny-input -j REJECT --reject-with icmp-port-unreachable"
)
# DOCKER-USER is the documented, docker-stable hook and is consulted FIRST from
# FORWARD, so it wins over every rule docker installs for its own networks.
# The agent needs nothing forwarded at all: no internet, no LAN, no peer
# container. Both directions are rejected.
DOCKER_USER_RULES=(
  "-i $BRIDGE_IF -m comment --comment ${TAG}-deny-forward-src -j REJECT --reject-with icmp-port-unreachable"
  "-o $BRIDGE_IF -m comment --comment ${TAG}-deny-forward-dst -j REJECT --reject-with icmp-port-unreachable"
)
# The proxy binds 127.0.0.1:$PROXY_PORT (relaunch_proxy_remote.sh sets
# LUMO_PROXY_LISTEN_HOST=127.0.0.1) and that bind is deliberately NOT changed:
# widening it to 0.0.0.0 would publish the model endpoint to alienware's LAN.
# Instead the bridge gateway address is DNAT'ed onto the loopback bind, which
# requires route_localnet on the bridge interface ONLY (not `all`), and a SNAT
# back to 127.0.0.1 so the reply is not a martian. Net effect for the proxy:
# the peer address it sees is 127.0.0.1, exactly as under --network=host.
NAT_PRE_RULES=(
  "-i $BRIDGE_IF -d $GW -p tcp --dport $PROXY_PORT -m comment --comment ${TAG}-proxy-dnat -j DNAT --to-destination 127.0.0.1:$PROXY_PORT"
)
NAT_POST_RULES=(
  "-s $SUBNET -d 127.0.0.1 -o lo -p tcp --dport $PROXY_PORT -m comment --comment ${TAG}-proxy-snat -j SNAT --to-source 127.0.0.1"
)
# alienware holds GLOBAL IPv6 addresses on wlp131s0f0, so IPv6 is a live egress
# path, not a theoretical one. The bridge gets disable_ipv6=1 (no host-side
# link-local to aim at) AND explicit ip6tables rejects, because docker's own
# ip6tables integration is off on this host (no DOCKER-USER chain in ip6tables).
IP6_INPUT_RULES=(
  "-i $BRIDGE_IF -m comment --comment ${TAG}-deny-input6 -j REJECT --reject-with icmp6-port-unreachable"
)
IP6_FORWARD_RULES=(
  "-i $BRIDGE_IF -m comment --comment ${TAG}-deny-forward6-src -j REJECT --reject-with icmp6-port-unreachable"
  "-o $BRIDGE_IF -m comment --comment ${TAG}-deny-forward6-dst -j REJECT --reject-with icmp6-port-unreachable"
)

SYSCTL_KEYS=(
  "net.ipv4.conf.${BRIDGE_IF}.route_localnet=1"
  "net.ipv6.conf.${BRIDGE_IF}.disable_ipv6=1"
)

# ---------------------------------------------------------------- primitives
net_exists() { docker network inspect "$NET" >/dev/null 2>&1; }

bridge_exists() { ip link show "$BRIDGE_IF" >/dev/null 2>&1; }

ensure_network() {
  if net_exists; then
    local have_subnet have_gw have_bridge
    have_subnet=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Subnet}}')
    have_gw=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Gateway}}')
    have_bridge=$(docker network inspect "$NET" -f '{{index .Options "com.docker.network.bridge.name"}}')
    [ "$have_subnet" = "$SUBNET" ] || die "network $NET has subnet $have_subnet, expected $SUBNET (teardown first)"
    [ "$have_gw" = "$GW" ] || die "network $NET has gateway $have_gw, expected $GW (teardown first)"
    [ "$have_bridge" = "$BRIDGE_IF" ] || die "network $NET has bridge $have_bridge, expected $BRIDGE_IF (teardown first)"
    return 0
  fi
  # --internal: docker's own DOCKER-ISOLATION DROPs are belt to our DOCKER-USER
  # braces, and it leaves the container with NO default route, so an internet
  # attempt fails with ENETUNREACH immediately instead of waiting on a REJECT.
  # enable_icc=false: agent containers never need to talk to each other.
  # enable_ip_masquerade=false: no SNAT for this subnet, so even a rule gap
  # cannot produce a routable source address.
  docker network create --driver bridge \
    --subnet "$SUBNET" --gateway "$GW" --internal \
    -o com.docker.network.bridge.name="$BRIDGE_IF" \
    -o com.docker.network.bridge.enable_icc=false \
    -o com.docker.network.bridge.enable_ip_masquerade=false \
    "$NET" >/dev/null
  log "created network $NET ($SUBNET gw=$GW bridge=$BRIDGE_IF internal icc=off)"
}

ensure_sysctls() {
  bridge_exists || die "bridge $BRIDGE_IF absent after network create"
  local kv key val have
  for kv in "${SYSCTL_KEYS[@]}"; do
    key=${kv%%=*}; val=${kv#*=}
    have=$($SUDO sysctl -n "$key" 2>/dev/null || echo "")
    [ "$have" = "$val" ] || { $SUDO sysctl -qw "$kv"; log "sysctl $kv"; }
  done
}

# Delete every copy of a rule spec, then re-insert at $pos. Guarantees both
# idempotency and ordering (plain `-C || -A` cannot guarantee ordering).
install_rule() {
  local cmd=$1 table=$2 chain=$3 pos=$4 spec=$5
  local guard=0
  # shellcheck disable=SC2086
  while $cmd -t "$table" -C "$chain" $spec >/dev/null 2>&1; do
    $cmd -t "$table" -D "$chain" $spec
    guard=$((guard + 1)); [ $guard -lt 16 ] || die "rule delete loop: $chain $spec"
  done
  # shellcheck disable=SC2086
  $cmd -t "$table" -I "$chain" "$pos" $spec
}

check_rule() {
  local cmd=$1 table=$2 chain=$3 spec=$4
  # shellcheck disable=SC2086
  $cmd -t "$table" -C "$chain" $spec >/dev/null 2>&1
}

ensure_docker_user_chain() {
  ipt -t filter -n -L DOCKER-USER >/dev/null 2>&1 || ipt -t filter -N DOCKER-USER
}

apply_rules() {
  ensure_docker_user_chain
  local i
  for i in "${!INPUT_RULES[@]}"; do
    install_rule "ipt" filter INPUT "$((i + 1))" "${INPUT_RULES[$i]}"
  done
  for i in "${!DOCKER_USER_RULES[@]}"; do
    install_rule "ipt" filter DOCKER-USER "$((i + 1))" "${DOCKER_USER_RULES[$i]}"
  done
  for i in "${!NAT_PRE_RULES[@]}"; do
    install_rule "ipt" nat PREROUTING "$((i + 1))" "${NAT_PRE_RULES[$i]}"
  done
  for i in "${!NAT_POST_RULES[@]}"; do
    install_rule "ipt" nat POSTROUTING "$((i + 1))" "${NAT_POST_RULES[$i]}"
  done
  for i in "${!IP6_INPUT_RULES[@]}"; do
    install_rule "ipt6" filter INPUT "$((i + 1))" "${IP6_INPUT_RULES[$i]}"
  done
  for i in "${!IP6_FORWARD_RULES[@]}"; do
    install_rule "ipt6" filter FORWARD "$((i + 1))" "${IP6_FORWARD_RULES[$i]}"
  done
}

missing_rules() {
  local i out=""
  for i in "${!INPUT_RULES[@]}"; do
    check_rule "ipt" filter INPUT "${INPUT_RULES[$i]}" || out="$out filter/INPUT[$i]"
  done
  for i in "${!DOCKER_USER_RULES[@]}"; do
    check_rule "ipt" filter DOCKER-USER "${DOCKER_USER_RULES[$i]}" || out="$out filter/DOCKER-USER[$i]"
  done
  for i in "${!NAT_PRE_RULES[@]}"; do
    check_rule "ipt" nat PREROUTING "${NAT_PRE_RULES[$i]}" || out="$out nat/PREROUTING[$i]"
  done
  for i in "${!NAT_POST_RULES[@]}"; do
    check_rule "ipt" nat POSTROUTING "${NAT_POST_RULES[$i]}" || out="$out nat/POSTROUTING[$i]"
  done
  for i in "${!IP6_INPUT_RULES[@]}"; do
    check_rule "ipt6" filter INPUT "${IP6_INPUT_RULES[$i]}" || out="$out filter6/INPUT[$i]"
  done
  for i in "${!IP6_FORWARD_RULES[@]}"; do
    check_rule "ipt6" filter FORWARD "${IP6_FORWARD_RULES[$i]}" || out="$out filter6/FORWARD[$i]"
  done
  printf '%s' "${out# }"
}

# The ACCEPT for the proxy is only a boundary if it sits ABOVE the catch-all
# REJECT, and the REJECT is only a boundary if nothing ACCEPTs before it.
# Fingerprinting the rule TEXT would not catch a reorder, so verify positions.
ordering_ok() {
  local accept_n reject_n
  accept_n=$(ipt -t filter -L INPUT --line-numbers -n 2>/dev/null \
    | grep -F "${TAG}-allow-proxy" | head -1 | awk '{print $1}')
  reject_n=$(ipt -t filter -L INPUT --line-numbers -n 2>/dev/null \
    | grep -F "${TAG}-deny-input" | head -1 | awk '{print $1}')
  [ -n "$accept_n" ] && [ -n "$reject_n" ] && [ "$accept_n" -lt "$reject_n" ]
}

# ---------------------------------------------------------------- fingerprint
# Fingerprints what the KERNEL has, read back from iptables -S, not what this
# script intended to install — so drift, reorder or manual edits change it.
fingerprint_material() {
  echo "schema=$SCHEMA"
  echo "net=$NET bridge=$BRIDGE_IF subnet=$SUBNET gateway=$GW proxy_port=$PROXY_PORT"
  if net_exists; then
    docker network inspect "$NET" -f 'docker internal={{.Internal}} icc={{index .Options "com.docker.network.bridge.enable_icc"}} masq={{index .Options "com.docker.network.bridge.enable_ip_masquerade"}} ipv6={{.EnableIPv6}} subnet={{(index .IPAM.Config 0).Subnet}} gateway={{(index .IPAM.Config 0).Gateway}} bridge={{index .Options "com.docker.network.bridge.name"}}'
  else
    echo "docker network=ABSENT"
  fi
  { ipt  -t filter -S; ipt  -t nat -S; ipt6 -t filter -S; } 2>/dev/null \
    | grep -F -- "$TAG" | sed 's/^/rule /'
  local kv key
  for kv in "${SYSCTL_KEYS[@]}"; do
    key=${kv%%=*}
    echo "sysctl $key=$($SUDO sysctl -n "$key" 2>/dev/null || echo ABSENT)"
  done
}

fingerprint() { fingerprint_material | sha256sum | awk '{print $1}'; }

expected_rule_count() {
  echo $(( ${#INPUT_RULES[@]} + ${#DOCKER_USER_RULES[@]} + ${#NAT_PRE_RULES[@]} \
         + ${#NAT_POST_RULES[@]} + ${#IP6_INPUT_RULES[@]} + ${#IP6_FORWARD_RULES[@]} ))
}

# ---------------------------------------------------------------- subcommands
cmd_apply() {
  ensure_network
  ensure_sysctls
  apply_rules
  ordering_ok || die "INPUT ordering wrong after apply"
  log "applied: $(expected_rule_count) rules, network $NET, gateway $GW"
  echo "FR14_NET_BOUNDARY_APPLIED net=$NET bridge=$BRIDGE_IF gateway=$GW proxy_port=$PROXY_PORT rules=$(expected_rule_count) fingerprint=$(fingerprint)"
}

cmd_verify() {
  net_exists || die "docker network $NET absent"
  bridge_exists || die "bridge interface $BRIDGE_IF absent"
  local have_subnet have_gw have_bridge internal icc ipv6
  have_subnet=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Subnet}}')
  have_gw=$(docker network inspect "$NET" -f '{{(index .IPAM.Config 0).Gateway}}')
  have_bridge=$(docker network inspect "$NET" -f '{{index .Options "com.docker.network.bridge.name"}}')
  internal=$(docker network inspect "$NET" -f '{{.Internal}}')
  icc=$(docker network inspect "$NET" -f '{{index .Options "com.docker.network.bridge.enable_icc"}}')
  ipv6=$(docker network inspect "$NET" -f '{{.EnableIPv6}}')
  [ "$have_subnet" = "$SUBNET" ] || die "subnet drift: $have_subnet != $SUBNET"
  [ "$have_gw" = "$GW" ] || die "gateway drift: $have_gw != $GW"
  [ "$have_bridge" = "$BRIDGE_IF" ] || die "bridge drift: $have_bridge != $BRIDGE_IF"
  [ "$internal" = "true" ] || die "network $NET is not --internal"
  [ "$icc" = "false" ] || die "network $NET has inter-container communication enabled"
  [ "$ipv6" = "false" ] || die "network $NET has IPv6 enabled"
  local miss; miss=$(missing_rules)
  [ -z "$miss" ] || die "missing rules: $miss"
  ordering_ok || die "proxy ACCEPT does not precede the catch-all REJECT in INPUT"
  local kv key have
  for kv in "${SYSCTL_KEYS[@]}"; do
    key=${kv%%=*}; have=$($SUDO sysctl -n "$key" 2>/dev/null || echo "")
    [ "$have" = "${kv#*=}" ] || die "sysctl drift: $key=$have expected ${kv#*=}"
  done
  echo "FR14_NET_BOUNDARY_VERIFIED net=$NET bridge=$BRIDGE_IF gateway=$GW proxy_port=$PROXY_PORT rules=$(expected_rule_count) fingerprint=$(fingerprint)"
}

cmd_status() {
  echo "=== docker network $NET ==="
  docker network inspect "$NET" 2>&1 | sed -n '1,80p' || true
  echo "=== tagged rules ==="
  { ipt -t filter -S; ipt -t nat -S; ipt6 -t filter -S; } 2>/dev/null | grep -F -- "$TAG" || echo "(none)"
  echo "=== sysctls ==="
  local kv; for kv in "${SYSCTL_KEYS[@]}"; do $SUDO sysctl -n "${kv%%=*}" 2>/dev/null \
    | sed "s|^|${kv%%=*} = |"; done
  echo "=== fingerprint ==="
  fingerprint
}

cmd_teardown() {
  local i guard
  for i in "${!INPUT_RULES[@]}"; do
    guard=0; while check_rule ipt filter INPUT "${INPUT_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt -t filter -D INPUT ${INPUT_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  for i in "${!DOCKER_USER_RULES[@]}"; do
    guard=0; while check_rule ipt filter DOCKER-USER "${DOCKER_USER_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt -t filter -D DOCKER-USER ${DOCKER_USER_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  for i in "${!NAT_PRE_RULES[@]}"; do
    guard=0; while check_rule ipt nat PREROUTING "${NAT_PRE_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt -t nat -D PREROUTING ${NAT_PRE_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  for i in "${!NAT_POST_RULES[@]}"; do
    guard=0; while check_rule ipt nat POSTROUTING "${NAT_POST_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt -t nat -D POSTROUTING ${NAT_POST_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  for i in "${!IP6_INPUT_RULES[@]}"; do
    guard=0; while check_rule ipt6 filter INPUT "${IP6_INPUT_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt6 -t filter -D INPUT ${IP6_INPUT_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  for i in "${!IP6_FORWARD_RULES[@]}"; do
    guard=0; while check_rule ipt6 filter FORWARD "${IP6_FORWARD_RULES[$i]}"; do
      # shellcheck disable=SC2086
      ipt6 -t filter -D FORWARD ${IP6_FORWARD_RULES[$i]}; guard=$((guard+1)); [ $guard -lt 16 ] || break; done
  done
  net_exists && docker network rm "$NET" >/dev/null || true
  log "torn down"
}

# ---------------------------------------------------------------- probe
# Runs INSIDE a container attached to the bridge and asserts the boundary from
# the attacker's seat, using the exact technique the finding recorded
# (python urllib, which qwen-code's shell-equivalence denylist does not catch).
PROBE_PY=$(cat <<'PYEOF'
import errno, json, os, socket, ssl, subprocess, sys, time, urllib.request

GW = os.environ["FR14_GW"]
PORT = int(os.environ["FR14_PROXY_PORT"])
LAN = os.environ.get("FR14_HOST_LAN", "")
TS = os.environ.get("FR14_HOST_TS", "")
T = float(os.environ.get("FR14_PROBE_TIMEOUT", "6"))

def errname(e):
    n = getattr(e, "errno", None)
    if n is None and isinstance(e, OSError):
        n = e.args[0] if e.args and isinstance(e.args[0], int) else None
    return errno.errorcode.get(n, None) if n is not None else None

def tcp(name, host, port, expect):
    t0 = time.monotonic()
    r = {"check": name, "target": "%s:%d" % (host, port), "expect": expect}
    try:
        s = socket.create_connection((host, port), timeout=T)
        s.close()
        r.update(outcome="connected", detail="tcp handshake completed")
    except Exception as e:
        r.update(outcome="failed", error=type(e).__name__, errno=errname(e),
                 detail=str(e))
    r["elapsed_s"] = round(time.monotonic() - t0, 3)
    return r

def http(name, url, expect):
    t0 = time.monotonic()
    r = {"check": name, "target": url, "expect": expect}
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        body = urllib.request.urlopen(url, timeout=T, context=ctx).read()
        r.update(outcome="fetched", bytes=len(body),
                 detail=body[:120].decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        r.update(outcome="http_status", status=e.code, detail=str(e))
    except Exception as e:
        r.update(outcome="failed", error=type(e).__name__, errno=errname(e),
                 detail=str(e))
    r["elapsed_s"] = round(time.monotonic() - t0, 3)
    return r

def dns(name, host, expect):
    t0 = time.monotonic()
    r = {"check": name, "target": host, "expect": expect}
    try:
        socket.setdefaulttimeout(T)
        r.update(outcome="resolved", detail=str(socket.getaddrinfo(host, 443)[:1]))
    except Exception as e:
        r.update(outcome="failed", error=type(e).__name__, errno=errname(e),
                 detail=str(e))
    r["elapsed_s"] = round(time.monotonic() - t0, 3)
    return r

checks = []
# --- POSITIVE: the one destination the agent is allowed to reach ------------
checks.append(tcp("proxy_gateway_tcp", GW, PORT, "connected"))
checks.append(http("proxy_gateway_http", "http://%s:%d/v1/models" % (GW, PORT),
                   "fetched or http_status (proxy answers)"))
# --- NEGATIVE: the exact egress recorded in the finding ---------------------
checks.append(http("gold_patch_fetch",
                   "https://patch-diff.githubusercontent.com/raw/astropy/astropy/pull/13236.diff",
                   "failed (this is the 13236 gold patch the agent fetched)"))
checks.append(http("github_https", "https://github.com/", "failed"))
checks.append(tcp("internet_ip_tcp", "1.1.1.1", 80, "failed"))
checks.append(tcp("internet_ip_tls", "1.1.1.1", 443, "failed"))
checks.append(dns("dns_external", "github.com", "failed"))
checks.append(dns("dns_docker_internal", "host.docker.internal", "failed"))
checks.append(tcp("metadata_endpoint", "169.254.169.254", 80, "failed"))
# --- NEGATIVE: alienware's own services. HOST_OPEN is a real service bound
#     0.0.0.0 on this host, so a refusal here is the RULE, not an absent
#     listener -- that is the control that makes the negatives meaningful.
HOST_OPEN = int(os.environ.get("FR14_HOST_OPEN_PORT", "0"))
checks.append(tcp("host_ssh_via_gateway", GW, 22, "failed"))
if HOST_OPEN:
    checks.append(tcp("host_open_service_via_gateway", GW, HOST_OPEN, "failed"))
if LAN:
    checks.append(tcp("host_ssh_via_lan", LAN, 22, "failed"))
    checks.append(tcp("host_proxy_via_lan", LAN, PORT, "failed"))
    if HOST_OPEN:
        checks.append(tcp("host_open_service_via_lan", LAN, HOST_OPEN, "failed"))
if TS:
    # alienware's ssh is TAILSCALE SSH (tailscaled intercepts :22 on tailscale0;
    # there is no kernel listener), so this is the real admin path.
    checks.append(tcp("host_ssh_via_tailscale", TS, 22, "failed"))
PEER = os.environ.get("FR14_TAILNET_PEER", "")
if PEER:
    checks.append(tcp("tailnet_peer_vllm", PEER, 9950, "failed"))
# --- NEGATIVE: docker daemon / other stacks on this host -------------------
checks.append(tcp("other_stack_bridge", "172.18.0.1", 80, "failed"))
checks.append(tcp("docker_default_bridge", "172.17.0.1", 80, "failed"))
checks.append(tcp("docker_socket_tcp", GW, 2375, "failed"))
# --- NEGATIVE: route_localnet is ON for this bridge, so the host loopback is
#     routable FROM the bridge if the container could aim a packet at it. It
#     cannot add a route without NET_ADMIN, and even with one the INPUT
#     catch-all rejects every loopback destination except the proxy port.
checks.append(tcp("host_resolved_dns_loopback", "127.0.0.53", 53, "failed (own netns lo)"))
# --- NEGATIVE: docker grants CAP_NET_RAW by default, so the container can
#     craft packets and SO_BINDTODEVICE past the missing default route. The
#     rules match on the bridge INTERFACE, not on a route, so this must still
#     be contained. If it is not, add --cap-drop=NET_RAW to the agent run.
def raw_egress(name, dst):
    t0 = time.monotonic()
    r = {"check": name, "target": dst, "expect": "failed"}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, b"eth0\0")
        s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        # minimal IPv4 header + TCP SYN toward dst:80, source = our own address
        s.sendto(b"\x45\x00\x00\x28" + b"\x00\x00\x40\x00\x40\x06\x00\x00"
                 + socket.inet_aton("172.31.99.2") + socket.inet_aton(dst)
                 + b"\x30\x39\x00\x50\x00\x00\x00\x01\x00\x00\x00\x00"
                 + b"\x50\x02\x20\x00\x00\x00\x00\x00", (dst, 0))
        r.update(outcome="raw_send_accepted",
                 detail="kernel accepted the raw send; forwarding still gated "
                        "by DOCKER-USER -i <bridge> REJECT")
    except Exception as e:
        r.update(outcome="failed", error=type(e).__name__, errno=errname(e),
                 detail=str(e))
    r["elapsed_s"] = round(time.monotonic() - t0, 3)
    return r

checks.append(raw_egress("raw_socket_egress", "1.1.1.1"))

# --- topology evidence -----------------------------------------------------
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception as e:
        return "ERR %s" % e

topology = {
    "routes_proc_net_route": sh("cat /proc/net/route"),
    "resolv_conf": sh("cat /etc/resolv.conf"),
    "ipv6_addrs": sh("cat /proc/net/if_inet6"),
    "cap_eff": sh("grep CapEff /proc/self/status"),
    "uid": str(os.getuid()),
    "route_add_attempt": sh("ip route add 127.0.0.0/8 dev eth0 2>&1 || true"),
    "default_route_present": "yes" if any(
        len(l.split()) > 2 and l.split()[1] == "00000000"
        for l in sh("cat /proc/net/route").splitlines()[1:]) else "no",
}
# If NET_ADMIN were somehow present the route above would exist; re-test the
# loopback destination after the attempt so the report shows the real answer.
checks.append(tcp("host_loopback_after_route_add", "127.0.0.53", 53,
                  "failed (route add must be denied)"))
checks.append(tcp("host_loopback_proxy_after_route_add", "127.0.0.1", PORT,
                  "failed (own netns lo)"))

print(json.dumps({"schema": "fr14-agent-net-probe-v1", "gateway": GW,
                  "proxy_port": PORT, "checks": checks, "topology": topology},
                 indent=2))
PYEOF
)

cmd_probe() {
  net_exists || die "docker network $NET absent — run apply first"
  local lan ts open_port
  lan=$(ip -4 -brief addr show scope global 2>/dev/null \
        | awk '$1 !~ /^(docker|br-|fr14)/ && $1 != "tailscale0" {print $3}' \
        | head -1 | cut -d/ -f1)
  ts=$(ip -4 -brief addr show tailscale0 2>/dev/null | awk '{print $3}' | head -1 | cut -d/ -f1)
  # CONTROL for the negative checks: a real host service bound 0.0.0.0 (i.e.
  # reachable from the bridge if the rules were absent). Without it a refusal
  # could just mean "nothing was listening".
  open_port=${FR14_HOST_OPEN_PORT:-$(ss -lnt 2>/dev/null \
        | awk '$4 ~ /^0\.0\.0\.0:/ {split($4,a,":"); print a[2]}' | head -1)}
  docker run --rm --network "$NET" \
    -e FR14_GW="$GW" -e FR14_PROXY_PORT="$PROXY_PORT" \
    -e FR14_HOST_LAN="$lan" -e FR14_HOST_TS="$ts" \
    -e FR14_HOST_OPEN_PORT="${open_port:-0}" \
    -e FR14_TAILNET_PEER="${FR14_TAILNET_PEER:-${GB10_TS_IP:-100.103.10.122}}" \
    -e FR14_PROBE_TIMEOUT="${FR14_PROBE_TIMEOUT:-6}" \
    -e PROBE_PY="$PROBE_PY" \
    "$PROBE_IMAGE" python3 -c 'import os;exec(os.environ["PROBE_PY"])'
}

# NEGATIVE CONTROL. The same probe on docker's DEFAULT bridge — no boundary.
# Without it, "every check failed" on the isolated bridge is unfalsifiable
# (it could just mean alienware has no internet). This is the run that shows
# the probe DOES detect egress, by re-fetching the 13236 gold patch.
cmd_control() {
  local lan ts open_port
  lan=$(ip -4 -brief addr show scope global 2>/dev/null \
        | awk '$1 !~ /^(docker|br-|fr14)/ && $1 != "tailscale0" {print $3}' \
        | head -1 | cut -d/ -f1)
  ts=$(ip -4 -brief addr show tailscale0 2>/dev/null | awk '{print $3}' | head -1 | cut -d/ -f1)
  open_port=${FR14_HOST_OPEN_PORT:-$(ss -lnt 2>/dev/null \
        | awk '$4 ~ /^0\.0\.0\.0:/ {split($4,a,":"); print a[2]}' | head -1)}
  docker run --rm --network bridge \
    -e FR14_GW="$(ip -4 -brief addr show docker0 | awk '{print $3}' | cut -d/ -f1)" \
    -e FR14_PROXY_PORT="$PROXY_PORT" \
    -e FR14_HOST_LAN="$lan" -e FR14_HOST_TS="$ts" \
    -e FR14_HOST_OPEN_PORT="${open_port:-0}" \
    -e FR14_TAILNET_PEER="${FR14_TAILNET_PEER:-${GB10_TS_IP:-100.103.10.122}}" \
    -e FR14_PROBE_TIMEOUT="${FR14_PROBE_TIMEOUT:-6}" \
    -e PROBE_PY="$PROBE_PY" \
    "$PROBE_IMAGE" python3 -c 'import os;exec(os.environ["PROBE_PY"])'
}

case "${1:-}" in
  apply)       cmd_apply ;;
  verify)      cmd_verify ;;
  fingerprint) fingerprint ;;
  gateway)     echo "$GW" ;;
  status)      cmd_status ;;
  probe)       cmd_probe ;;
  control)     cmd_control ;;
  teardown)    cmd_teardown ;;
  *) echo "usage: $0 {apply|verify|fingerprint|gateway|status|probe|teardown}" >&2; exit 2 ;;
esac
