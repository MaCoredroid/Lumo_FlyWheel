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
set -euo pipefail

NET=fr14-agent-isolated
SUBNET=172.31.99.0/24
GW=172.31.99.1
PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
PROBE_IMAGE=${FR14_NET_PROBE_IMAGE:-qwen-code-runner:v1}

log() { echo "[fr14-net-boundary] $*"; }

ensure_network() {
  if ! docker network inspect $NET >/dev/null 2>&1; then
    docker network create --driver bridge --subnet $SUBNET $NET >/dev/null
    log "created network $NET ($SUBNET)"