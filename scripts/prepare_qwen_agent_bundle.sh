#!/usr/bin/env bash
# prepare_qwen_agent_bundle.sh — build a relocatable node + qwen-code bundle so
# the SWE agent (SWE_AGENT_ENV=instance_image, §58) can be bind-mounted read-only
# into ANY SWE-bench per-instance eval image at /opt/qwen and run editing
# /testbed with the conda 'testbed' env on PATH.
#
# WHY the OFFICIAL nodejs.org tarball (not the node:22-bookworm binary): the
# official linux-x64 build links against an OLD glibc, so it runs on the
# SWE-bench eval images (ubuntu 20.04/22.04 bases) which may ship an older glibc
# than the runner image. This is the one real portability risk and the tarball
# removes it.
#
# WHY a bundle (not baking deps into qwen-code-runner): the per-instance image
# already carries the correct per-instance project + conda 'testbed' env; we only
# need to inject the node+qwen runtime. One relocatable bundle serves all
# 500+/731 instances with no per-instance image rebuild and stays eval-faithful.
#
# The qwen-code version is PINNED to whatever qwen-code-runner:v1 actually ships
# (cross-arm fairness): resolved from docker/qwen-code-runner.Dockerfile when it
# pins an explicit version, else from the built image itself, else $QWEN_CODE_VERSION.
#
# Idempotent: re-running is a no-op when the bundle already matches the target
# versions (pass FORCE=1 to rebuild). Prints all resolved versions at the end.
#
# Run this ON the codex host (alienware, x86_64). Env overrides:
#   BUNDLE_DIR         (default ~/qwen_agent_bundle)
#   NODE_VERSION       (default 22.23.1 — matches node:22-bookworm in runner:v1)
#   QWEN_CODE_VERSION  (default: resolved from Dockerfile/image; wins if set)
#   RUNNER_IMAGE       (default qwen-code-runner:v1 — version fallback source)
#   FORCE=1            rebuild even if versions already match
#   SKIP_NODE_VERIFY=1 skip the SHASUMS256 checksum verification
set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$HOME/qwen_agent_bundle}"
NODE_VERSION="${NODE_VERSION:-22.23.1}"
RUNNER_IMAGE="${RUNNER_IMAGE:-qwen-code-runner:v1}"
NODE_ARCH="linux-x64"
NODE_DIST="node-v${NODE_VERSION}-${NODE_ARCH}"
NODE_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_DIST}.tar.xz"
SHA_URL="https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/../docker/qwen-code-runner.Dockerfile"

log() { printf '[prepare_qwen_agent_bundle] %s\n' "$*" >&2; }
die() { printf '[prepare_qwen_agent_bundle] FATAL: %s\n' "$*" >&2; exit 1; }

# --- resolve the qwen-code version pin ---------------------------------------
resolve_qwen_version() {
  # 1) explicit env override wins.
  if [ -n "${QWEN_CODE_VERSION:-}" ]; then
    printf '%s' "$QWEN_CODE_VERSION"; return 0
  fi
  # 2) explicit pin in the Dockerfile (e.g. @qwen-code/qwen-code@0.19.4).
  if [ -f "$DOCKERFILE" ]; then
    local spec
    spec="$(grep -oE '@qwen-code/qwen-code@[^ ]+' "$DOCKERFILE" | head -n1 | sed 's/.*@qwen-code\/qwen-code@//')"
    if [ -n "$spec" ] && [ "$spec" != "latest" ]; then
      printf '%s' "$spec"; return 0
    fi
  fi
  # 3) fall back to the version actually installed in the built runner image
  #    (the Dockerfile uses @latest; runner:v1 froze it at build time — this is
  #    the eval-fair pin). Requires the runner image to be present on this host.
  if command -v docker >/dev/null 2>&1 && docker image inspect "$RUNNER_IMAGE" >/dev/null 2>&1; then
    local v
    v="$(docker run --rm --entrypoint qwen "$RUNNER_IMAGE" --version 2>/dev/null \
         | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
    if [ -n "$v" ]; then printf '%s' "$v"; return 0; fi
  fi
  return 1
}

QWEN_CODE_VERSION="$(resolve_qwen_version)" || die \
  "cannot resolve qwen-code version: Dockerfile pins @latest and runner image '${RUNNER_IMAGE}' \
is unavailable to read the frozen version. Set QWEN_CODE_VERSION=<x.y.z> explicitly."

log "target node=${NODE_VERSION} qwen-code=${QWEN_CODE_VERSION} bundle=${BUNDLE_DIR}"

# --- idempotency: skip when the bundle already matches -------------------------
NODE_BIN="${BUNDLE_DIR}/node/bin/node"
QWEN_SHIM="${BUNDLE_DIR}/bin/qwen"
if [ "${FORCE:-0}" != "1" ] && [ -x "$NODE_BIN" ] && [ -x "$QWEN_SHIM" ]; then
  have_node="$("$NODE_BIN" --version 2>/dev/null | sed 's/^v//')"
  have_qwen="$("$NODE_BIN" "${BUNDLE_DIR}/npm/bin/qwen" --version 2>/dev/null \
               | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
  if [ "$have_node" = "$NODE_VERSION" ] && [ "$have_qwen" = "$QWEN_CODE_VERSION" ]; then
    log "bundle already up to date (node=${have_node} qwen-code=${have_qwen}); nothing to do (FORCE=1 to rebuild)."
    log "versions:"
    "$QWEN_SHIM" --version 2>/dev/null | sed 's/^/  qwen: /' >&2 || true
    printf '  node: v%s\n' "$have_node" >&2
    exit 0
  fi
  log "version mismatch (have node=${have_node:-none} qwen=${have_qwen:-none}); rebuilding."
fi

for tool in curl tar sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool '$tool' not found on PATH"
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- download + verify the official node tarball ------------------------------
log "downloading ${NODE_URL}"
curl -fsSL "$NODE_URL" -o "${WORK}/${NODE_DIST}.tar.xz" || die "node tarball download failed"
if [ "${SKIP_NODE_VERIFY:-0}" != "1" ]; then
  log "verifying SHASUMS256"
  curl -fsSL "$SHA_URL" -o "${WORK}/SHASUMS256.txt" || die "SHASUMS256 download failed"
  ( cd "$WORK" && grep " ${NODE_DIST}.tar.xz\$" SHASUMS256.txt | sha256sum -c - ) \
    || die "node tarball checksum verification FAILED"
fi

# --- (re)build the bundle -----------------------------------------------------
rm -rf "${BUNDLE_DIR}/node" "${BUNDLE_DIR}/npm" "${BUNDLE_DIR}/bin"
mkdir -p "${BUNDLE_DIR}/node" "${BUNDLE_DIR}/npm" "${BUNDLE_DIR}/bin"

log "extracting node -> ${BUNDLE_DIR}/node"
tar -xJf "${WORK}/${NODE_DIST}.tar.xz" -C "${BUNDLE_DIR}/node" --strip-components=1

# install qwen-code globally into the bundle using the BUNDLED node/npm so the
# install matches the runtime (no host node dependency). npm creates
# ${BUNDLE_DIR}/npm/bin/qwen (a relative symlink into lib/node_modules), which
# resolves correctly once the whole bundle is mounted at /opt/qwen.
log "installing @qwen-code/qwen-code@${QWEN_CODE_VERSION} into the bundle"
PATH="${BUNDLE_DIR}/node/bin:${PATH}" \
  "${BUNDLE_DIR}/node/bin/npm" install -g \
    --prefix "${BUNDLE_DIR}/npm" \
    "@qwen-code/qwen-code@${QWEN_CODE_VERSION}" \
  || die "npm install of qwen-code failed"

[ -e "${BUNDLE_DIR}/npm/bin/qwen" ] || die \
  "npm did not create npm/bin/qwen — the package bin name may have changed"

# bin/qwen shim: execs the BUNDLED node against the installed qwen entry using
# absolute /opt/qwen paths (the fixed mount point), because inside the eval image
# only /opt/qwen/bin is on PATH and there is no system node.
cat > "$QWEN_SHIM" <<'SHIM'
#!/bin/sh
# qwen-code launcher for the SWE-bench instance-image agent env (§58).
# Uses the bundled (glibc-old-compat) node so it runs on the eval image base.
exec /opt/qwen/node/bin/node /opt/qwen/npm/bin/qwen "$@"
SHIM
chmod +x "$QWEN_SHIM"

# --- report -------------------------------------------------------------------
log "bundle built at ${BUNDLE_DIR}"
log "layout: node/ (official ${NODE_ARCH} runtime), npm/ (qwen-code), bin/qwen (shim)"
log "versions:"
"${BUNDLE_DIR}/node/bin/node" --version 2>/dev/null | sed 's/^/  node: /' >&2 || true
"${BUNDLE_DIR}/node/bin/npm"  --version 2>/dev/null | sed 's/^/  npm:  /' >&2 || true
"${BUNDLE_DIR}/node/bin/node" "${BUNDLE_DIR}/npm/bin/qwen" --version 2>/dev/null \
  | sed 's/^/  qwen: /' >&2 || log "  qwen: (--version probe failed; check the install)"
log "mount into an eval image with:  -v ${BUNDLE_DIR}:/opt/qwen:ro"
log "done."
