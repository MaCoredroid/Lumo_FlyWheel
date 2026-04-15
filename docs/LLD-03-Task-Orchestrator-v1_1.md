# LLD-03 · Task Orchestrator

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S1 → Implement S1  
> Status: DRAFT v1.1 — targeting sign-off

---

## Changelog

| Version | Change |
|---|---|
| v1.1 | **One P0 fix.** P0: Pre-grading mismatch recovery protocol now has a single documented path. All resolutions — including restoring on-disk artifacts to match the prior manifest — require a manifest version bump. This is the state-machine trigger that enables `invalidate_stale_runs()` → `REGRADE_NEEDED` via the signed-off LLD-02 `grading_manifest_ver < new_version` predicate. The "fix artifacts without bumping" branch is eliminated. This aligns with LLD-13 §12.5 which already requires a version bump and changelog entry for any post-freeze artifact change, including restorations. |
| v0.4 | **Three P0 fixes, one P1 fix.** P0-1: SWE-bench patch evaluation no longer depends on an LLD-05 interface. LLD-03 runs the standard `swebench` harness directly — same mechanical step as Codex-Long Phase 2. §14.2 cross-LLD amendment removed. P0-2: Egress blocking added — `iptables` rules restrict bridge-subnet traffic to inference proxy port only. P0-3: Pre-grading hash mismatch handled as dedicated `ManifestMismatchError` path preserving snapshot and trajectory. P1-1: Changelog v0.3 text corrected — `vllm.bind_host` is `127.0.0.1`. |
| v0.3 | **Four P0 fixes, two P1 fixes.** P0-1: SWE-bench outcome flow fixed to stay within signed-off LLD-02 contract. Removed `outcome=None` deferred-callback model and the `update_outcome()` dependency. SWE-bench patch evaluation is now synchronous within `execute_task()`. Terminal outcomes only — no invented values. P0-2: Manifest provenance timing fixed. `manifest_state.reload()` now runs BEFORE `claim_run()` for Codex-Long tasks so `launch_manifest_ver` is accurate. Second reload after agent session before grading. P0-3: `claim_run()` return value now checked. `False` → `DuplicateClaimError`. P0-4: Inference-only reverse proxy introduced. Agent containers reach proxy on bridge gateway; proxy forwards only `/v1/responses` and `/v1/chat/completions`. vLLM stays on `127.0.0.1`. P1-1: `model_context_window` read from model registry, not hardcoded. P1-2: Bind address (`127.0.0.1`) and client address (`127.0.0.1`) separated in config. Containers use proxy via bridge gateway. |
| v0.2 | **Four P0 fixes, two P1 fixes.** P0-1: Codex harness bootstrap added (§5A new section). Task containers do not ship with Codex — the orchestrator bind-mounts the host Codex binary, a per-task `config.toml` generated from the model/provider config, and the `VLLM_API_KEY` env var into each container. LLD-13 env factory images are unchanged (scenario deps only). P0-2: Run-state contract aligned with signed-off LLD-02. `start_run()` replaced with `claim_run()` throughout. Family metadata (`family_id`, `scenario_type`, `launch_manifest_ver`) now passed at claim. `pending_eval` outcome removed — SWE-bench runs are finished with `outcome=None` (LLD-02 schema: outcome is NULL until set); LLD-05 calls `finish_run()` with the final outcome after patch evaluation. Outcome determination for SWE-bench is now explicitly deferred to LLD-05. P0-3: Seed env vars now threaded through to execution. `apply_seed()` output is passed to `invoke_codex()` → `docker_exec_with_timeout()` → `docker exec -e CODEX_SEED=N`. P0-4: Network isolation contract implemented. Agent containers use a dedicated Docker bridge network (`codex-bench-net`) with only the vLLM port forwarded; host networking removed. Resource limits enforced per container: `--memory`, `--cpus`, `--pids-limit`, `--ulimit nofile`. SWE-bench `container:host` string removed. P1-1: Stdout capture rewritten. `process.communicate()` removed — event stream is captured by a single async reader draining `process.stdout` line-by-line to disk, with `process.wait()` used for exit code. No dual-consumer deadlock. P1-2: Exception-path cleanup added. `execute_task()` uses a `finally` block that guarantees container removal, grading workspace cleanup, and extracted-filesystem removal on any failure. Retained snapshots are never removed on crash (LLD-02 §6.5 retention still applies). |
| v0.1 | Initial draft. Dual-track execution loop (SWE-bench patch path + Codex-Long three-phase grading path). Manifest two-phase hash verification per LLD-13 §12.6. Snapshot retention per LLD-02 §6.5 — `docker rmi` is skipped for committed snapshots; grading workspace is still cleaned up. Seed-to-sampling mapping defined. Structured event stream capture. Timeout and health-check contracts. Regrade-only recovery path implemented. |

---

## 1. Purpose & Scope

This document specifies the Task Orchestrator — the component that owns the end-to-end per-task execution loop for both SWE-bench and Codex-Long tasks. It is the central coordinator that connects model serving (LLD-01) to evaluation (LLD-05), latency capture (LLD-04), and trajectory parsing (LLD-06).

**Responsibilities:**

- Launch Docker containers for each task (SWE-bench repo checkouts and Codex-Long env factory images)
- Invoke `codex exec --yolo --json` inside containers with the pinned fairness-contract flags
- Capture the structured JSONL event stream from stdout
- Enforce per-task timeouts
- Flush prefix cache between tasks (`POST /reset_prefix_cache` via LLD-01 §6.2)
- Health-check vLLM readiness before each dispatch (`GET /health` via LLD-01 §9.1)
- Extract patches from SWE-bench runs (feeds LLD-05 SWE-bench evaluator)
- Orchestrate three-phase Codex-Long grading (Phase 1 snapshot, Phase 2 functional checks, Phase 3 integrity verification — contracts defined in LLD-13 §6)
- Enforce LLD-13 §12.6 two-phase manifest hash verification (pre-run and pre-grading)
- Retain committed snapshot images per LLD-02 §6.5 retention policy
- Support regrade-only recovery: re-execute Phase 2+3 from a retained snapshot without re-running the agent session
- Create, update, and finalize run records via LLD-02 APIs (`claim_run()`, `finish_run()`)
- Map integer seeds to Codex sampling parameters
- Coordinate snapshot timing with LLD-04 (latency telemetry)
- Emit raw JSONL into the trajectory directory consumed by LLD-04 and LLD-06

**Out of scope:** Campaign-level orchestration (which tasks to run, in what order — LLD-07), model serving and switching (LLD-01/LLD-07), trajectory parsing and SFT formatting (LLD-06), solve-rate evaluation and result aggregation (LLD-05/LLD-12), SWE-Agent execution (LLD-09 — separate harness with its own execution loop), pool/split management and seal enforcement (LLD-02), and scenario family authoring (LLD-13).

---

## 2. Architecture Overview

```
                            ┌────────────────┐
                            │   LLD-07       │
                            │ Benchmark      │
                            │ Runner         │
                            │                │
                            │ dispatches     │
                            │ TaskSpec       │
                            └───────┬────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                   LLD-03 · TASK ORCHESTRATOR                 │
│                                                              │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────────┐    │
│  │ Health   │  │ Manifest   │  │ Container             │    │
│  │ Check    │  │ Verifier   │  │ Manager               │    │
│  │ (§4)     │  │ (§8)       │  │ (§5, §6, §7)          │    │
│  └──────────┘  └────────────┘  └───────────────────────┘    │
│                                                              │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────────┐    │
│  │ Codex    │  │ Event      │  │ Post-Run              │    │
│  │ Invoker  │  │ Stream     │  │ Grading               │    │
│  │ (§6)     │  │ Capture    │  │ (§7)                   │    │
│  │          │  │ (§9)       │  │ SWE-bench: patch       │    │
│  └──────────┘  └────────────┘  │ Codex-Long: 3-phase   │    │
│                                └───────────────────────┘    │
│                                                              │
│  ┌──────────┐  ┌────────────┐                                │
│  │ Cache    │  │ Seed       │                                │
│  │ Flush    │  │ Mapper     │                                │
│  │ (§4.2)   │  │ (§6.4)     │                                │
│  └──────────┘  └────────────┘                                │
└──────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   LLD-01         LLD-02         LLD-04         LLD-05/06
   /health        run state      /metrics       trajectory
   /reset_cache   claim/finish   snapshots      + grading
```

---

## 3. Task Specification — Input Contract

LLD-07 dispatches individual tasks to the orchestrator as `TaskSpec` records. This is the interface contract between LLD-07 (campaign coordination) and LLD-03 (per-task execution).

### 3.1 TaskSpec Schema

```python
@dataclass(frozen=True)
class TaskSpec:
    # ── Identity ──
    track: str                      # "swe_bench" | "codex_long"
    pool_or_split: str              # e.g. "dev_bench", "train_long", "test_long"
    scenario_id: str                # instance_id (SWE-bench) or "<family_id>/<variant_id>"
    model_id: str                   # e.g. "qwen3.5-27b"
    harness: str                    # "codex" (always for this LLD; "swe_agent" → LLD-09)
    seed: int                       # integer seed for sampling

    # ── Environment ──
    # SWE-bench fields
    repo: Optional[str]             # e.g. "django/django"
    base_commit: Optional[str]      # commit to check out
    instance_id: Optional[str]      # SWE-bench instance ID
    prompt: Optional[str]           # problem statement (from SWE-bench dataset)

    # Codex-Long fields
    family_id: Optional[str]
    variant_id: Optional[str]
    image_digest: Optional[str]     # from LLD-02 CodexLongEnv, sourced from manifest
    scenario_type: Optional[str]    # for metadata propagation to run records

    # ── Dispatch decision ──
    dispatch_decision: str          # "proceed" | "retry" | "regrade_needed" | "rerun_needed"
    attempt: int                    # attempt number (1 for fresh, 2 for retry)

    # ── Regrade-only fields (set when dispatch_decision == "regrade_needed") ──
    regrade_snapshot_ref: Optional[str]  # retained snapshot image ref for regrade-only path

    # ── Execution config ──
    timeout_seconds: int            # per-task wall-clock limit
```

### 3.2 Dispatch Decision Handling

The orchestrator's behavior branches on `dispatch_decision`:

| Decision | Orchestrator Action |
|---|---|
| `proceed` | Full execution: container setup → agent run → post-run processing |
| `retry` | Same as `proceed` but `attempt = 2`; prior crash is logged |
| `rerun_needed` | Same as `proceed` — image changed, full re-execution required |
| `regrade_needed` | Skip agent execution; jump to §7 grading from the retained snapshot image specified in `regrade_snapshot_ref`. Codex-Long only. |

---

## 4. Pre-Task Checks

### 4.1 Health Check

Before any task execution, the orchestrator verifies that the vLLM server is ready and serving the expected model.

```python
async def health_check(
    vllm_host: str,
    vllm_port: int,
    expected_model: str,
    max_retries: int = 5,
    retry_delay_seconds: float = 10.0,
) -> None:
    """
    Verify vLLM readiness before task dispatch.
    
    Checks:
    1. GET /health returns 200
    2. GET /v1/models lists the expected model (or adapter name)
    
    Raises TaskDispatchError if health check fails after max_retries.
    """
    for attempt in range(max_retries):
        try:
            # Check server health
            health_resp = await http_get(f"http://{vllm_host}:{vllm_port}/health")
            if health_resp.status != 200:
                raise HealthCheckError(f"/health returned {health_resp.status}")

            # Check served model
            models_resp = await http_get(
                f"http://{vllm_host}:{vllm_port}/v1/models"
            )
            model_ids = [m["id"] for m in models_resp.json()["data"]]
            if expected_model not in model_ids:
                raise HealthCheckError(
                    f"Expected model '{expected_model}' not in served models: {model_ids}"
                )

            return  # healthy
        except (ConnectionError, HealthCheckError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Health check attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {retry_delay_seconds}s."
                )
                await asyncio.sleep(retry_delay_seconds)
            else:
                raise TaskDispatchError(
                    f"vLLM health check failed after {max_retries} attempts. "
                    f"Last error: {e}. Task: {expected_model}"
                )
```

### 4.2 Prefix Cache Flush

The prefix cache is flushed between tasks to prevent cross-task KV contamination. Within a task, prefix caching is ON (growing conversation prefix benefits from caching across turns).

```python
async def flush_prefix_cache(vllm_host: str, vllm_port: int) -> None:
    """
    Flush vLLM's prefix cache between tasks.
    
    Requires VLLM_SERVER_DEV_MODE=1 (set in LLD-01 §5).
    Does NOT reset /metrics counters — delta-sampling in LLD-04
    handles per-task attribution separately.
    
    Called AFTER the previous task's grading and event capture
    are complete, BEFORE the next task's health check.
    """
    resp = await http_post(
        f"http://{vllm_host}:{vllm_port}/reset_prefix_cache"
    )
    if resp.status == 405:
        raise ConfigError(
            "/reset_prefix_cache returned 405 — VLLM_SERVER_DEV_MODE may not be set. "
            "Set VLLM_SERVER_DEV_MODE=1 in vLLM launch environment."
        )
    if resp.status != 200:
        raise CacheFlushError(f"/reset_prefix_cache returned {resp.status}")
```

### 4.3 LLD-04 Pre-Task Snapshot

Before the task starts, the orchestrator signals LLD-04 to capture a `/metrics` snapshot. This snapshot becomes the `task_before` baseline for delta-sampling (LLD-01 §9.2).

**Timing (v0.6):** `snapshot_before()` is called AFTER `claim_run()` and pre-run verification succeed (§10.1 step 4). This ensures no orphaned snapshots are created for tasks that fail verification or lose the claim race.

```python
# Sequencing (matches LLD-01 §9.2 sampling protocol):
#   1. LLD-03   claim_run + verification (steps 1–3 in §10.1)
#   2. LLD-04 → GET /metrics           (task_before snapshot — step 4)
#   3. LLD-03   starts codex exec      (task begins — step 6)
#   4. LLD-03   stdout EOF / timeout   (task complete)
#   5. LLD-04 → GET /metrics           (task_after snapshot — step 8)
#   6. LLD-04   compute deltas         (per-task attribution)

await latency_capture.snapshot_before(task_id=task_spec.scenario_id)
```

---

## 5. Codex Harness Bootstrap

Task containers (both SWE-bench and Codex-Long) do not ship with the Codex CLI binary or its provider configuration. LLD-13's env factory images provision scenario runtime dependencies only (Node.js, Python, gcc, etc.) — harness tooling is the orchestrator's responsibility. This section specifies how each container is given a working Codex installation that can reach local vLLM.

### 5.1 Host-Side Codex Installation (One-Time, Sprint 1 Setup)

The DGX Spark host has a single pinned Codex CLI installation. The binary, its version, and the git hash are recorded in the benchmark launch log for reproducibility.

```bash
# Sprint 1 setup — run once on the host
npm install -g @openai/codex@<pinned_version>

# Record version for reproducibility
codex --version > /data/codex-bench/codex_version.txt
which codex    >> /data/codex-bench/codex_version.txt
```

The host Codex binary is bind-mounted read-only into every task container at a fixed path. This guarantees all tasks use the identical Codex binary regardless of what the container image ships.

### 5.2 Per-Task Config Generation

Each task run requires a `config.toml` that wires Codex to the current model on local vLLM. The orchestrator generates this file per task because the model name changes across the campaign (LLD-07 switches models) and the adapter name changes in Sprint 3 (LoRA serving per LLD-01 §17.1).

```python
def generate_codex_config(
    task: TaskSpec,
    proxy_host: str,
    proxy_port: int,
    model_registry: dict,
) -> str:
    """
    Generate a Codex config.toml for a specific task execution.
    
    Returns the path to the generated config file.
    
    The config matches the LLD-01 §8.1 canonical pattern:
    - wire_api = "responses" (only valid value since Feb 2026)
    - base_url points to the INFERENCE PROXY (§5.4), not vLLM directly
    - model name matches --served-model-name (base) or --lora-modules key (Sprint 3)
    - stream_idle_timeout_ms = 600000 (10 min for long agentic sessions)
    - model_context_window read from model registry (LLD-01 §3.1 max_model_len)
      — NOT hardcoded, because it varies per model (e.g., 122B starts at 65536)
    """
    config_dir = f"/tmp/codex-bench/configs/{task.scenario_id.replace('/', '_')}"
    os.makedirs(config_dir, exist_ok=True)
    config_path = f"{config_dir}/config.toml"

    # Look up model-specific context window from the registry
    # (LLD-01 §3.1 — max_model_len varies: 131072 for most, 65536 for 122B)
    model_entry = model_registry.get(task.model_id, {})
    context_window = model_entry.get("max_model_len", 131072)
    compact_limit = int(context_window * 0.9)  # ~90% of context window

    # Container reaches the inference proxy on the bridge gateway (§5.4),
    # not vLLM directly. The proxy forwards only /v1/responses and
    # /v1/chat/completions.
    proxy_url = f"http://{proxy_host}:{proxy_port}/v1"

    config_content = f"""
model          = "{task.model_id}"
model_provider = "localvllm"

model_context_window          = {context_window}
model_auto_compact_token_limit = {compact_limit}

[model_providers.localvllm]
name                   = "Local vLLM"
base_url               = "{proxy_url}"
env_key                = "VLLM_API_KEY"
wire_api               = "responses"
stream_idle_timeout_ms = 600000
request_max_retries    = 2
"""
    write_file(config_path, config_content.strip())
    return config_path
```

### 5.3 Container Bind-Mount Contract

Every task container (SWE-bench and Codex-Long) receives the following bind-mounts and environment variables for Codex harness operation. These are in addition to any task-specific mounts.

| Mount / Env | Source (host) | Target (container) | Mode | Purpose |
|---|---|---|---|---|
| Codex binary | Host Codex install dir | `/usr/local/lib/node_modules/@openai/codex` | `ro` | Pinned CLI binary |
| Codex symlink | Host `$(which codex)` | `/usr/local/bin/codex` | `ro` | CLI available on PATH |
| Node.js runtime | Host Node.js install | `/usr/local/bin/node`, `/usr/local/lib/node_modules` | `ro` | Required by Codex (Node.js binary) |
| Per-task config | Generated `config.toml` | `/root/.codex/config.toml` | `ro` | Provider wiring for this model |
| `VLLM_API_KEY` | env var | env var | — | Any non-empty string (vLLM accepts any) |
| `CODEX_SEED` | env var | env var | — | Seed for sampling variation (§6.4) |

```python
def get_codex_harness_mounts(
    config_path: str,
    codex_binary_path: str,
    codex_node_modules: str,
    node_binary_path: str,
) -> dict:
    """
    Return the bind-mount dict for injecting Codex into a task container.
    
    These mounts are combined with any task-specific mounts
    (workspace, grading volumes, etc.) at container creation.
    """
    return {
        codex_binary_path: {"bind": "/usr/local/bin/codex", "mode": "ro"},
        codex_node_modules: {
            "bind": "/usr/local/lib/node_modules/@openai/codex",
            "mode": "ro",
        },
        node_binary_path: {"bind": "/usr/local/bin/node", "mode": "ro"},
        config_path: {"bind": "/root/.codex/config.toml", "mode": "ro"},
    }


def get_codex_harness_env(task: TaskSpec) -> dict:
    """
    Return the environment variables for Codex harness operation.
    """
    return {
        "VLLM_API_KEY": "EMPTY",       # vLLM accepts any non-empty string
        "CODEX_SEED": str(task.seed),   # seed for sampling variation (§6.4)
    }
```

> **Why bind-mount instead of baking Codex into every image?** LLD-13's env factory images are scenario-specific — they contain language runtimes, dependencies, and injected breakage. Baking Codex into every image would create a coupling between the harness version and the scenario build pipeline. If Codex is updated (bug-fix, Sprint 0 patch), every image would need rebuilding and re-hashing, invalidating the entire `benchmark_manifest.lock`. Bind-mounting isolates the concern: scenario images are frozen at Sprint 0b; the Codex binary is pinned separately and recorded in the launch log.

> **Node.js dependency:** Codex CLI is a Node.js application. Some task containers (e.g., Python-only scenarios) do not have Node.js installed. The orchestrator bind-mounts the host Node.js runtime alongside the Codex binary. If a container's own Node.js conflicts with the mounted version (e.g., a Node.js scenario with a different version), the mounted path takes precedence on PATH. Sprint 1 validation must confirm this does not break scenario-specific Node.js tooling.

### 5.4 Inference Proxy and vLLM Reachability

Agent containers must not have direct access to the vLLM server. vLLM exposes management and telemetry endpoints (`/metrics`, `/reset_prefix_cache`, `/v1/models`, `/health`) alongside inference endpoints (`/v1/responses`, `/v1/chat/completions`) on the same port. Giving agent containers access to the full surface would violate the isolation boundary: an agent could flush the prefix cache mid-task, read telemetry to infer grading expectations, or probe model metadata.

The orchestrator runs a lightweight inference-only reverse proxy (`codex-bench-proxy`) on the bridge network gateway. Containers connect to the proxy; the proxy forwards only inference paths to vLLM on loopback. Host-side orchestrator calls (`/health`, `/metrics`, `/reset_prefix_cache`) go directly to vLLM on `127.0.0.1` and are never proxied.

```
┌───────────────────────────────────────────────────┐
│  HOST (DGX Spark)                                 │
│                                                   │
│  vLLM ──────────── 127.0.0.1:8000                 │
│    ▲                    ▲                          │
│    │ (direct)           │ (forward inference only) │
│    │                    │                          │
│  Orchestrator     codex-bench-proxy                │
│  /health            0.0.0.0:8001                   │
│  /metrics             ▲                            │
│  /reset_cache         │ (bridge gateway)           │
│                       │                            │
│         ┌─────────────┼──────────────┐             │
│         │  codex-bench-net bridge    │             │
│         │                            │             │
│         │  ┌─────────┐ ┌──────────┐  │             │
│         │  │ Agent   │ │ Agent    │  │             │
│         │  │ Ctr 1   │ │ Ctr 2   │  │             │
│         │  │→ :8001  │ │→ :8001  │  │             │
│         │  └─────────┘ └──────────┘  │             │
│         └────────────────────────────┘             │
└───────────────────────────────────────────────────┘
```

The proxy is a minimal nginx config:

```nginx
# /etc/codex-bench/proxy.conf
server {
    listen 8001;

    # Inference paths only — everything Codex CLI needs
    location /v1/responses {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 600s;          # 10 min for long agentic turns
        proxy_buffering off;              # streaming required for Codex
    }

    location /v1/chat/completions {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 600s;
        proxy_buffering off;
    }

    # Block everything else — 403 on /metrics, /health, /reset_prefix_cache, /v1/models
    location / {
        return 403 "Blocked by codex-bench-proxy: inference paths only";
    }
}
```

```bash
# Sprint 1 setup — run the proxy on the host
docker run -d \
  --name codex-bench-proxy \
  --network host \
  -v /etc/codex-bench/proxy.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

The proxy runs on the host network (it needs loopback access to vLLM). Containers on the bridge network reach it via the gateway IP at port 8001. The per-task `config.toml` (§5.2) points Codex to `http://<gateway_ip>:8001/v1`, not to vLLM directly.

```python
async def get_bridge_gateway_ip(network_name: str) -> str:
    """Resolve the gateway IP of the Docker bridge network."""
    result = await run_cmd(
        f"docker network inspect {network_name} "
        f"--format '{{{{(index .IPAM.Config 0).Gateway}}}}'"
    )
    return result.stdout.strip()
```

---

## 5A. Network Isolation and Resource Limits

### 5A.1 Docker Network Setup and Egress Blocking (One-Time)

The orchestrator creates a dedicated bridge network at initialization. Agent containers attach to this network. Egress is restricted by `iptables` rules so containers can only reach the inference proxy — not vLLM directly, not any other host service, and not the internet. This satisfies the HLD requirement: "no network except model server."

```bash
#!/bin/bash
# setup_network.sh — One-time setup (Sprint 1)
set -euo pipefail

NETWORK_NAME="codex-bench-net"
SUBNET="172.30.0.0/16"
GATEWAY="172.30.0.1"
PROXY_PORT=8001

# ── 1. Create the bridge network ──
docker network create \
  --driver bridge \
  --subnet "$SUBNET" \
  --gateway "$GATEWAY" \
  "$NETWORK_NAME"

# ── 2. Egress blocking via DOCKER-USER chain ──
# Docker processes the DOCKER-USER chain before its own forwarding rules.
# These rules restrict what bridge-subnet containers can reach.
# Order matters: first match wins.

# Rule 1: Allow return traffic for established connections (required for
# TCP handshake completion and streaming responses from the proxy).
iptables -I DOCKER-USER 1 \
  -s "$SUBNET" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN

# Rule 2: Allow new connections to the inference proxy port on the
# gateway ONLY. This is the sole permitted outbound destination.
iptables -I DOCKER-USER 2 \
  -s "$SUBNET" -d "$GATEWAY" -p tcp --dport "$PROXY_PORT" -j RETURN

# Rule 3: DROP everything else from the bridge subnet.
# This blocks: internet egress, access to vLLM on port 8000,
# access to /metrics, /reset_prefix_cache, /v1/models, SSH,
# and any other host or external service.
iptables -I DOCKER-USER 3 \
  -s "$SUBNET" -j DROP

echo "[NETWORK] codex-bench-net created with egress restricted to proxy port $PROXY_PORT"
```

**Validation (Sprint 1 checklist):**

```bash
# From inside a task container:
# ✓ Should succeed (inference proxy):
curl -s http://172.30.0.1:8001/v1/responses  # → proxy response (or 405 without body)

# ✗ Should fail (vLLM directly):
curl -s --connect-timeout 5 http://172.30.0.1:8000/metrics   # → timeout/connection refused
curl -s --connect-timeout 5 http://172.30.0.1:8000/health    # → timeout/connection refused

# ✗ Should fail (internet):
curl -s --connect-timeout 5 http://example.com  # → timeout/connection refused
```

vLLM remains bound to `127.0.0.1:8000` (unchanged from LLD-01 §5.2 — no amendment needed). The inference proxy binds to `0.0.0.0:8001` on the host and forwards to `127.0.0.1:8000`. Containers on the bridge reach the proxy via the gateway IP (`172.30.0.1:8001`). The iptables rules ensure this is the ONLY reachable destination.

**Two distinct network paths — never conflated:**

| Caller | Target | Address | Endpoints Reachable |
|---|---|---|---|
| Host orchestrator (LLD-03) | vLLM directly | `127.0.0.1:8000` | All: `/health`, `/metrics`, `/reset_prefix_cache`, `/v1/models`, `/v1/responses` |
| Agent container (Codex CLI) | Inference proxy | `172.30.0.1:8001` | Inference only: `/v1/responses`, `/v1/chat/completions` (enforced by proxy + iptables) |

### 5A.2 Resource Limits

Every agent container is launched with resource limits that enforce the HLD fairness boundary ("resource limits enforced") and prevent runaway containers from destabilizing the host or other runs.

```python
CONTAINER_RESOURCE_LIMITS = {
    "mem_limit": "32g",       # 32 GB RAM — generous for agentic sessions
    "cpus": 4.0,              # 4 CPU cores
    "pids_limit": 1024,       # prevent fork bombs
    "ulimits": [
        {"name": "nofile", "soft": 65536, "hard": 65536},  # file descriptors
    ],
    "storage_opt": {
        "size": "50G",        # writable layer limit (overlay2 only)
    },
}
```

| Limit | Value | Rationale |
|---|---|---|
| `--memory` | 32 GB | Codex + scenario tooling (npm, gcc, test suites). 32 GB is generous; most scenarios use < 8 GB. |
| `--cpus` | 4.0 | Agent sessions are I/O-bound (waiting for model responses). 4 cores is sufficient for tool execution (compilation, test runs). |
| `--pids-limit` | 1024 | Prevents fork bombs from runaway test suites or agent-triggered scripts. |
| `--ulimit nofile` | 65536 | Matches typical Linux server defaults. Prevents file-descriptor exhaustion. |
| `--storage-opt size` | 50 GB | Limits the writable layer size. Only enforced on overlay2 with `dm.basesize` configured. Prevents disk exhaustion from unbounded file creation. |

These limits are identical across SWE-bench and Codex-Long containers (fairness contract).

### 5A.3 Combined Container Launch Helper

```python
async def launch_task_container(
    image: str,
    container_name: str,
    task: TaskSpec,
    task_volumes: dict,
    config: OrchestratorConfig,
) -> str:
    """
    Launch a task container with Codex harness, network isolation,
    and resource limits.
    
    Combines:
    - Task-specific volumes (workspace, etc.)
    - Codex harness mounts (§5.3)
    - Bridge network (§5A.1)
    - Resource limits (§5A.2)
    
    The container's Codex config.toml points to the inference proxy
    on the bridge gateway (§5.4), not to vLLM directly. The proxy
    exposes only /v1/responses and /v1/chat/completions.
    
    Returns the container ID.
    """
    # Generate per-task Codex config pointing to the PROXY, not vLLM
    gateway_ip = await get_bridge_gateway_ip(config.network.name)
    config_path = generate_codex_config(
        task,
        proxy_host=gateway_ip,
        proxy_port=config.network.proxy_port,
        model_registry=config.model_registry,
    )

    # Combine all mounts
    all_volumes = {**task_volumes}
    all_volumes.update(get_codex_harness_mounts(
        config_path=config_path,
        codex_binary_path=config.codex.binary_path,
        codex_node_modules=config.codex.node_modules_path,
        node_binary_path=config.codex.node_binary_path,
    ))

    # Combine env vars
    env_vars = get_codex_harness_env(task)

    container_id = await docker_run(
        image=image,
        name=container_name,
        volumes=all_volumes,
        network=config.network.name,     # bridge network, not host
        workdir="/workspace",
        environment=env_vars,
        detach=True,
        **CONTAINER_RESOURCE_LIMITS,
    )

    return container_id
```

---

## 5B. Container Lifecycle — SWE-bench Path

### 5B.1 SWE-bench Container Setup

SWE-bench tasks use the standard SWE-bench Docker evaluation images. The orchestrator checks out the target repo at the specified commit and launches the container with Codex harness injection.

```python
async def setup_swe_bench_container(
    task: TaskSpec,
    config: OrchestratorConfig,
) -> ContainerContext:
    """
    Prepare a Docker container for a SWE-bench task.
    
    Steps:
    1. Pull/verify the SWE-bench evaluation Docker image for the task's repo
    2. Create a working directory with the repo checked out at base_commit
    3. Write the AGENTS.md / problem statement into the workspace
    4. Launch the container with Codex harness, bridge network, and resource limits
    
    Returns a ContainerContext with the container ID and workspace path.
    """
    container_name = f"swe-bench-{task.instance_id}-{task.model_id}-seed{task.seed}-a{task.attempt}"

    # Workspace preparation
    workspace = f"{config.paths.output_dir}/workspaces/{container_name}"
    os.makedirs(workspace, exist_ok=True)

    # Clone and checkout
    await run_cmd(
        f"git clone --depth=1 https://github.com/{task.repo}.git {workspace}/repo"
    )
    await run_cmd(f"git -C {workspace}/repo checkout {task.base_commit}")

    # Write problem statement as AGENTS.md
    write_file(f"{workspace}/repo/AGENTS.md", task.prompt)

    # Launch with harness bootstrap + isolation
    container_id = await launch_task_container(
        image=get_swe_bench_image(task.repo),
        container_name=container_name,
        task=task,
        task_volumes={f"{workspace}/repo": {"bind": "/workspace", "mode": "rw"}},
        config=config,
    )

    return ContainerContext(
        container_id=container_id,
        container_name=container_name,
        workspace_path=f"{workspace}/repo",
        track="swe_bench",
    )
```

### 5B.2 SWE-bench Post-Run: Patch Extraction

After the Codex session completes, the orchestrator extracts the diff between the agent's final state and the base commit. This patch feeds LLD-05 (SWE-bench evaluator).

```python
async def extract_swe_bench_patch(
    container: ContainerContext,
    output_dir: str,
    task: TaskSpec,
) -> Optional[str]:
    """
    Extract the git diff from a completed SWE-bench container.
    
    Returns the path to the patch file, or None if the agent
    made no changes (outcome = no_patch).
    """
    patch_path = f"{output_dir}/patches/{task.instance_id}_{task.model_id}_seed{task.seed}.patch"

    # Generate diff inside the container
    result = await docker_exec(
        container.container_id,
        "git diff HEAD",
        workdir="/workspace",
    )

    if result.stdout.strip():
        write_file(patch_path, result.stdout)
        return patch_path
    else:
        return None  # no_patch
```

### 5B.3 SWE-bench Container Teardown

```python
async def teardown_swe_bench_container(container: ContainerContext) -> None:
    """Remove the SWE-bench container and clean up workspace."""
    await docker_rm(container.container_id, force=True)
    # Workspace retained for debugging if configured; otherwise cleaned up
```

### 5B.4 SWE-bench Evaluation — Bilateral Contract (LLD-03 ↔ LLD-05)

LLD-03 "drives LLD-05 (Patch Converter for SWE-bench) on session close" (per the LLD index). LLD-05 owns the patch-to-predictions conversion, the `swebench` harness invocation, and the result report. LLD-03 invokes it and reads the verdict.

This is a same-sprint design contract: LLD-05 is "Design S1 → Implement S1" and is not yet signed off. The interface below is a required contract that LLD-05's design must implement. Both sides are defined here so the contract is reviewable in one place. LLD-05's full design document will reference this section.

**Invocation contract (LLD-03 calls):**

```bash
codex-bench-eval-swe \
  --instance-id <instance_id> \
  --patch-path <path_to_patch_file> \
  --output-dir <path_for_eval_artifacts> \
  --dataset-name princeton-nlp/SWE-bench_Verified
```

| Argument | Type | Description |
|---|---|---|
| `--instance-id` | string | SWE-bench instance ID (e.g., `django__django-11099`) |
| `--patch-path` | path | Path to the `.patch` file extracted by LLD-03 (§5B.2) |
| `--output-dir` | path | Directory where LLD-05 writes evaluation artifacts |
| `--dataset-name` | string | SWE-bench dataset identifier (always `princeton-nlp/SWE-bench_Verified`) |

**Exit code contract (LLD-03 reads, LLD-05 sets):**

| Exit Code | Meaning | LLD-03 Maps To |
|---|---|---|
| 0 | All relevant tests pass — patch resolves the issue | `outcome = "resolved"` |
| 1 | Tests fail — patch does not resolve the issue | `outcome = "failed"` |
| 2 | Evaluation infrastructure error (harness crash, Docker failure) | `outcome = "crash"` |

LLD-03 reads ONLY the exit code for outcome determination. It does not parse predictions.jsonl, eval reports, or log files — those are LLD-05/LLD-12 artifacts.

**Artifact contract (LLD-05 writes, LLD-03 does not consume):**

| Artifact | Format | Consumer | Purpose |
|---|---|---|---|
| `{output_dir}/{instance_id}/predictions.jsonl` | Standard SWE-bench predictions format | LLD-12, external reproducibility | Matches the format expected by the upstream `swebench` package |
| `{output_dir}/{instance_id}/eval_report.json` | Structured JSON | LLD-12 | Per-instance evaluation metadata for result aggregation |
| `{output_dir}/{instance_id}/eval.log` | Plain text | Debugging | Raw evaluation log from the `swebench` harness |

**Implementation responsibilities (LLD-05 owns):**

The `codex-bench-eval-swe` entry point must:

1. Convert the patch file to the standard SWE-bench predictions JSONL format
2. Invoke `python -m swebench.harness.run_evaluation` with the correct arguments for the current upstream harness version
3. Parse the harness output to determine pass/fail
4. Write the artifacts listed above
5. Return the appropriate exit code

The implementation details (exact `swebench` CLI arguments, report parsing logic, Docker image management for the SWE-bench test containers) are determined during LLD-05's full design and are not specified here. ARM64 (DGX Spark) execution is a project decision; LLD-05 must validate that `swebench.harness.run_evaluation` works correctly on ARM64 during Sprint 1 setup.

**LLD-03 wrapper (consumer side):**

```python
async def drive_swe_bench_eval(
    instance_id: str,
    patch_path: str,
    output_dir: str,
) -> str:
    """
    Invoke LLD-05's SWE-bench evaluation pipeline and return the outcome.
    
    LLD-03 "drives" LLD-05 per the index — it invokes the evaluation
    CLI and reads the exit code. LLD-05 owns the implementation:
    predictions.jsonl conversion, swebench harness invocation, and
    report generation.
    
    Returns: "resolved", "failed", or "crash"
    """
    eval_output = f"{output_dir}/swe_eval/{instance_id}"
    os.makedirs(eval_output, exist_ok=True)

    result = await run_cmd(
        f"codex-bench-eval-swe "
        f"--instance-id {instance_id} "
        f"--patch-path {patch_path} "
        f"--output-dir {eval_output} "
        f"--dataset-name princeton-nlp/SWE-bench_Verified",
        timeout_seconds=900,
    )

    if result.returncode == 0:
        return "resolved"
    elif result.returncode == 1:
        return "failed"
    else:
        logger.warning(
            f"SWE-bench eval infrastructure error for {instance_id}: "
            f"exit code {result.returncode}"
        )
        return "crash"
```

**Codex-Long evaluation interface (no CLI needed):**

For Codex-Long tasks, LLD-03 produces `verify_result.json` via the three-phase grading pipeline (§7, implementing LLD-13 §6 contracts). LLD-05 reads this file from the path stored in LLD-02 run records for solve rate aggregation and milestone partial-credit reporting. LLD-05 does not execute verifiers — `verify.sh` is the sole execution authority (LLD-13 §9.2). No CLI entry point is required for this path.

---

## 6. Container Lifecycle — Codex-Long Path

### 6.1 Codex-Long Container Setup

Codex-Long tasks use the Docker environment factory images built by LLD-13 §5. Containers are launched by image digest — not by tag — for artifact-level reproducibility.

```python
async def setup_codex_long_container(
    task: TaskSpec,
    manifest: dict,
    config: OrchestratorConfig,
) -> ContainerContext:
    """
    Launch a Codex-Long agent container from the env factory image.
    
    Pre-condition: verify_pre_run_hashes() has ALREADY been called
    before claim_run() in execute_task() (§10.1). This function does
    not repeat the verification — it trusts that the caller has
    confirmed the manifest hashes before claiming the run slot.
    
    The container receives the Codex harness (§5) via bind-mount,
    connects to vLLM via the bridge network (§5A), and runs with
    resource limits (§5A.2).
    
    The container does NOT have access to:
    - Verifier scripts (verifiers/<family_id>/)
    - Milestone check scripts
    - Oracle solutions
    - Test reference data (verifier_data/<family_id>/)
    - The trusted grading image
    """
    container_name = (
        f"codex-long-{task.family_id}-{task.variant_id}"
        f"-{task.model_id}-seed{task.seed}-a{task.attempt}"
    )

    # Launch container by digest with harness bootstrap + isolation
    # No task-specific volumes — the env factory image has /workspace baked in
    container_id = await launch_task_container(
        image=task.image_digest,       # sha256:<hex> — directly runnable
        container_name=container_name,
        task=task,
        task_volumes={},               # workspace is baked into the image
        config=config,
    )

    # Verify no verifier artifacts leaked into the container
    # (defense-in-depth — should be enforced at build time by LLD-13 §5.4)
    leaked = await docker_exec(
        container_id,
        "test -d /verifier || test -d /verifier_data || test -d /oracle",
    )
    if leaked.returncode == 0:
        await docker_rm(container_id, force=True)
        raise IntegrityError(
            f"Verifier artifacts found inside agent container for "
            f"{task.scenario_id}. Image may be corrupt or incorrectly built."
        )

    return ContainerContext(
        container_id=container_id,
        container_name=container_name,
        workspace_path="/workspace",
        track="codex_long",
        family_id=task.family_id,
        variant_id=task.variant_id,
    )
```

### 6.2 Pre-Run Hash Verification (LLD-13 §12.6 Phase 1)

```python
def verify_pre_run_hashes(task: TaskSpec, manifest: dict) -> None:
    """
    Validate that the agent container's locked artifacts match the manifest.
    
    Checks (from LLD-13 §12.6 enforcement contract, Phase 1):
    1. Image digest matches manifest's image_digest
    2. AGENTS.md hash matches manifest's agents_md_hash
    3. Family spec hash matches manifest's family_spec_hash
    
    Aborts the run on any mismatch. Does not silently proceed
    with drifted artifacts.
    """
    entry = find_manifest_entry(manifest, task.family_id, task.variant_id)

    # Step 2: Image digest
    actual_image_digest = get_local_image_digest(task.image_digest)
    if actual_image_digest != entry["image_digest"]:
        raise ManifestMismatchError(
            f"Image digest mismatch for {task.scenario_id}: "
            f"expected {entry['image_digest']}, got {actual_image_digest}"
        )

    # Step 3: AGENTS.md hash
    agents_md_path = get_agents_md_from_image(task.image_digest)
    actual_agents_md_hash = sha256_file(agents_md_path)
    if actual_agents_md_hash != entry["agents_md_hash"]:
        raise ManifestMismatchError(
            f"AGENTS.md hash mismatch for {task.scenario_id}: "
            f"expected {entry['agents_md_hash']}, got {actual_agents_md_hash}"
        )

    # Step 4: Family spec hash
    family_spec_path = f"scenario_families/{task.family_id}/family.yaml"
    actual_spec_hash = sha256_file(family_spec_path)
    if actual_spec_hash != entry["family_spec_hash"]:
        raise ManifestMismatchError(
            f"Family spec hash mismatch for {task.scenario_id}: "
            f"expected {entry['family_spec_hash']}, got {actual_spec_hash}"
        )

    logger.info(
        f"Pre-run hash verification passed for {task.scenario_id} "
        f"(manifest v{manifest['manifest_version']})"
    )
```

### 6.3 Codex Invocation

The `codex exec` invocation is identical for both SWE-bench and Codex-Long tasks. The pinned flags match the HLD §5 fairness contract exactly.

```python
async def invoke_codex(
    container: ContainerContext,
    task: TaskSpec,
    output_dir: str,
) -> CodexResult:
    """
    Run codex exec --yolo --json inside the task container.
    
    The event stream (JSONL) is captured to a file.
    
    Pre-conditions (guaranteed by launch_task_container §5A.3):
    - Codex binary is mounted and on PATH
    - config.toml wires Codex to the correct model on vLLM
    - VLLM_API_KEY is set
    - CODEX_SEED is set (seed-to-sampling mapping, §6.4)
    
    Persistent sessions are used (no --ephemeral) per HLD §5.
    """
    trajectory_path = (
        f"{output_dir}/trajectories/"
        f"trajectory_{task.model_id}_{task.scenario_id.replace('/', '_')}"
        f"_seed{task.seed}_a{task.attempt}.jsonl"
    )

    # Build the codex exec command per HLD §5
    codex_cmd = _build_codex_command(task)

    # Execute inside the container with timeout.
    # No additional env vars needed here — CODEX_SEED and VLLM_API_KEY
    # are already set as container-level env vars at launch time (§5.3).
    start_time = time.monotonic()
    result = await docker_exec_with_timeout(
        container_id=container.container_id,
        command=codex_cmd,
        timeout_seconds=task.timeout_seconds,
        stdout_sink=trajectory_path,
    )
    wall_time = time.monotonic() - start_time

    return CodexResult(
        trajectory_path=trajectory_path,
        exit_code=result.returncode,
        wall_time_seconds=wall_time,
        timed_out=result.timed_out,
        stderr=result.stderr,
    )


def _build_codex_command(task: TaskSpec) -> str:
    """
    Build the pinned codex exec command per HLD §5 fairness contract.
    
    Flags pinned across all models:
    - exec                          Non-interactive batch execution
    - --yolo                        Bypass all approvals and sandbox
    - --json                        JSONL event stream to stdout
    - -c 'web_search="disabled"'    No web access
    - -c 'model_reasoning_effort="high"'  High reasoning budget
    - -c 'personality="pragmatic"'  Fixed personality
    
    No --ephemeral during collection — persistent sessions are the safety net.
    
    Note: -m is not needed here — the model is specified in the
    per-task config.toml (§5.2) which is bind-mounted at container
    launch time. The config.toml's `model` field is the source of truth.
    """
    prompt = _get_prompt(task)

    # Model identity comes from config.toml (§5.2), not -m flag.
    # CODEX_SEED is already set as a container-level env var (§5.3).
    return (
        f"codex exec "
        f"--yolo "
        f"--json "
        f"-c 'web_search=\"disabled\"' "
        f"-c 'model_reasoning_effort=\"high\"' "
        f"-c 'personality=\"pragmatic\"' "
        f"-C /workspace "
        f'"{prompt}"'
    )


def _get_prompt(task: TaskSpec) -> str:
    """
    Get the agent prompt for a task.
    
    SWE-bench: The problem statement from the SWE-bench dataset,
    directing the agent to read AGENTS.md and fix the issue.
    
    Codex-Long: A generic prompt directing the agent to read AGENTS.md,
    which contains the full task description visible at /workspace/AGENTS.md.
    """
    if task.track == "swe_bench":
        return (
            "Read AGENTS.md for the problem description. Fix the reported issue "
            "in this repository. Make the minimal changes necessary to resolve "
            "the failing test case described in the issue."
        )
    else:
        return (
            "Read AGENTS.md for the task description. Complete the task described "
            "there. The repository is at /workspace."
        )
```

### 6.4 Seed-to-Sampling Mapping

Seeds control sampling non-determinism in Codex. LLD-02 assigns integer seeds; this LLD maps them to the actual mechanism that introduces variation. The seed is injected as the `CODEX_SEED` environment variable at container launch time (§5.3, `get_codex_harness_env()`), not at `codex exec` invocation time. This guarantees the seed is active for the entire container lifecycle.

```python
# Seed injection is handled by get_codex_harness_env() (§5.3):
#
#   def get_codex_harness_env(task: TaskSpec) -> dict:
#       return {
#           "VLLM_API_KEY": "EMPTY",
#           "CODEX_SEED": str(task.seed),
#       }
#
# This env dict is passed to launch_task_container() (§5A.3)
# which sets it as container-level env vars via `docker run -e`.
```

> **Open question — seed mechanism:** The exact Codex CLI mechanism for seed-controlled variation needs Sprint 0 validation. If `CODEX_SEED` is not respected, the fallback is to append `\n[Seed: {seed}]` to the prompt, which produces variation through model sampling. The important contract is: same task + same model + different seeds → statistically independent trajectories.

---

## 7. Post-Run Processing

### 7.1 Outcome Determination

After `codex exec` completes (or times out), the orchestrator determines the task outcome.

```python
def determine_outcome(
    codex_result: CodexResult,
    track: str,
    patch_path: Optional[str] = None,
    verify_result: Optional[dict] = None,
) -> str:
    """
    Map execution results to the HLD failure contract outcomes.
    
    Outcomes (from LLD-02 §5.1):
      resolved  — task solved
      failed    — task attempted but not solved
      no_patch  — SWE-bench only: agent produced no diff
      timeout   — execution exceeded time limit
      crash     — infrastructure failure
    
    For SWE-bench: outcome determination is handled inline in
    execute_task() (§10.1) because patch evaluation is synchronous.
    This function is only called for Codex-Long tasks.
    """
    # Infrastructure crash detection
    if codex_result.exit_code not in (0, 1) and not codex_result.timed_out:
        if _is_infrastructure_error(codex_result.stderr):
            return "crash"

    if codex_result.timed_out:
        return "timeout"

    if track == "codex_long":
        if verify_result is None:
            return "crash"  # grading failed to produce results
        return "resolved" if verify_result.get("pass") else "failed"

    raise ValueError(f"Unexpected track in determine_outcome: {track}")
```

### 7.2 SWE-bench Post-Run Pipeline

```
codex exec completes
        │
        ▼
   Extract git diff (§5B.2)
        │
        ├── diff is empty → outcome = no_patch
        │
        └── diff exists → write patch file
                │
                ▼
         Container teardown (§5B.3)
                │
                ▼
         LLD-03 drives LLD-05 — invokes codex-bench-eval-swe (§5B.4)
         LLD-05 owns predictions.jsonl conversion + swebench harness
         LLD-03 reads exit code: 0=resolved, 1=failed, 2=crash
                │
                ├── tests pass → outcome = resolved
                └── tests fail → outcome = failed
                │
                ▼
         Signal LLD-04 for post-task /metrics snapshot
                │
                ▼
         Record run via LLD-02 finish_run(outcome=<terminal>)
         (outcome is always a terminal value from the HLD contract)
                │
                ▼
         Prefix cache flush (§4.2)
```

### 7.3 Codex-Long Post-Run Pipeline — Three-Phase Grading

This pipeline implements the grading contracts defined in LLD-13 §6 and §7. The orchestrator owns the execution; LLD-13 defines the contracts.

```
codex exec completes (or times out — timeout still graded for milestones)
        │
        ▼
   Phase 1: Snapshot (§7.4)
   docker commit → codex-long-snapshot/<run_id>
   Agent container removed after commit
        │
        ▼
   Pre-grading hash verification (§7.5) — LLD-13 §12.6 Phase 2
   Verify: grader image digest, verifier hash,
           milestone hashes, verifier_data hash
        │
        ├── hash mismatch → abort grading, preserve trajectory
        │
        ▼
   Phase 2: Functional checks (§7.6)
   Container FROM committed snapshot, --network none
   Run test commands from family spec functional_checks
   Capture exit codes + logs to /functional/
        │
        ▼
   Phase 3: Integrity verification (§7.7)
   Container FROM trusted codex-long-grader image, --network none
   Agent filesystem mounted read-only at /agent/
   Phase 2 results at /functional/ (read-only)
   Verifier scripts at /verifier/ (read-only)
   Verifier data at /verifier_data/ (read-only)
   Run /verifier/verify.sh → verify_result.json
        │
        ▼
   Parse verify_result.json → outcome + milestones
        │
        ▼
   Signal LLD-04 for post-task /metrics snapshot
        │
        ▼
   Record run via LLD-02 finish_run()
   (includes snapshot_image_ref for retention)
        │
        ▼
   Cleanup: remove Phase 2/3 containers and /grading workspace
   DO NOT remove committed snapshot image (LLD-02 §6.5 retention)
        │
        ▼
   Prefix cache flush (§4.2)
```

### 7.4 Phase 1 — Snapshot

```python
async def phase1_snapshot(
    container: ContainerContext,
    run_id: str,
) -> str:
    """
    Commit the agent container to a snapshot image.
    
    docker commit preserves ENV, WORKDIR, USER, PATH in the committed
    image config — required for Phase 2 functional checks to run
    correctly inside the agent's own runtime.
    
    Returns the snapshot image reference.
    """
    snapshot_ref = _snapshot_image_name(run_id)

    await run_cmd(f"docker commit {container.container_id} {snapshot_ref}")
    await docker_rm(container.container_id, force=True)

    logger.info(f"Phase 1 snapshot committed: {snapshot_ref}")
    return snapshot_ref


def _snapshot_image_name(run_id: str) -> str:
    """
    Deterministic snapshot name from run metadata.
    
    Convention from LLD-02 §6.5:
    codex-long-snapshot/<scenario_id>/<model_id>/<harness>/seed<N>/attempt<M>
    """
    return f"codex-long-snapshot/{run_id}"
```

### 7.5 Pre-Grading Hash Verification (LLD-13 §12.6 Phase 2)

```python
def verify_pre_grading_hashes(
    task: TaskSpec,
    manifest: dict,
    grader_image_ref: str,
) -> None:
    """
    Validate grading artifacts against the manifest before launching
    Phase 2 or Phase 3 containers.
    
    Checks (from LLD-13 §12.6 enforcement contract, Phase 2):
    7. Grader image digest matches manifest's grader_image_digest
    8. Verifier script hash matches manifest's verifier_hash
    9. Each milestone script hash matches manifest's milestone_hashes
    10. Verifier_data tree hash matches manifest's verifier_data_hash
    
    On mismatch: raises ManifestMismatchError with `affected_artifact`
    set to the LLD-02 _ARTIFACT_RECOVERY key (§6.3) for operator
    diagnostics and manifest-bump scoping. The handler in execute_task()
    records the crash and re-raises to LLD-07 as a non-retryable
    blocked condition (§10.1).
    """
    entry = find_manifest_entry(manifest, task.family_id, task.variant_id)

    # Step 7: Grader image digest
    actual_grader_digest = get_local_image_digest(grader_image_ref)
    expected_grader_digest = manifest["grader_image_digest"]
    if actual_grader_digest != expected_grader_digest:
        raise ManifestMismatchError(
            f"Grader image digest mismatch: "
            f"expected {expected_grader_digest}, got {actual_grader_digest}.",
            affected_artifact="grader_image",
        )

    # Step 8: Verifier script hash
    verifier_path = f"verifiers/{task.family_id}/verify.sh"
    actual_verifier_hash = sha256_file(verifier_path)
    if actual_verifier_hash != entry["verifier_hash"]:
        raise ManifestMismatchError(
            f"Verifier hash mismatch for {task.scenario_id}: "
            f"expected {entry['verifier_hash']}, got {actual_verifier_hash}",
            affected_artifact="verifier",
        )

    # Step 9: Milestone script hashes
    for milestone_id, expected_hash in entry.get("milestone_hashes", {}).items():
        milestone_path = f"verifiers/{task.family_id}/milestones/{milestone_id}.sh"
        actual_hash = sha256_file(milestone_path)
        if actual_hash != expected_hash:
            raise ManifestMismatchError(
                f"Milestone hash mismatch for {task.scenario_id}/{milestone_id}: "
                f"expected {expected_hash}, got {actual_hash}",
                affected_artifact="milestone",
            )

    # Step 10: Verifier data tree hash
    verifier_data_dir = f"verifier_data/{task.family_id}/"
    actual_data_hash = sha256_tree(verifier_data_dir)
    if actual_data_hash != entry["verifier_data_hash"]:
        raise ManifestMismatchError(
            f"Verifier data hash mismatch for {task.scenario_id}: "
            f"expected {entry['verifier_data_hash']}, got {actual_data_hash}",
            affected_artifact="verifier_data",
        )

    logger.info(
        f"Pre-grading hash verification passed for {task.scenario_id} "
        f"(manifest v{manifest['manifest_version']})"
    )
```

### 7.6 Phase 2 — Functional Checks

```python
async def phase2_functional_checks(
    snapshot_ref: str,
    task: TaskSpec,
    family_spec: dict,
    grading_dir: str,
) -> None:
    """
    Run functional checks inside the agent's own runtime (sandboxed).
    
    Launches a container FROM the committed snapshot image.
    The container has:
    - The agent's full runtime (npm, gcc, PATH, installed packages)
    - --network none (no external access)
    - A shared volume for writing results to /functional/
    
    It does NOT have:
    - Verifier scripts, milestone checks, or oracle solutions
    - Access to the trusted grading image
    
    Functional check commands come from the family spec's
    grading_invariant.functional_checks list (LLD-13 §3.1).
    """
    functional_dir = f"{grading_dir}/functional"
    os.makedirs(functional_dir, exist_ok=True)

    checks = family_spec["grading_invariant"]["functional_checks"]

    for check in checks:
        check_id = check["id"]
        command = check["command"]
        timeout = check.get("timeout_seconds", 120)

        container_name = f"phase2-{task.scenario_id.replace('/', '-')}-{check_id}"

        # Build the command that captures exit code and output
        wrapped_cmd = (
            f"sh -c '{command} > /functional/{check_id}_output.log 2>&1; "
            f"echo $? > /functional/{check_id}_exit_code'"
        )

        await docker_run_and_wait(
            image=snapshot_ref,
            name=container_name,
            network="none",
            volumes={functional_dir: "/functional/"},
            command=wrapped_cmd,
            timeout_seconds=timeout,
            remove_after=True,
        )

        logger.info(
            f"Phase 2 check '{check_id}' complete for {task.scenario_id}"
        )
```

### 7.7 Phase 3 — Integrity Verification

```python
async def phase3_integrity_verification(
    snapshot_ref: str,
    task: TaskSpec,
    grading_dir: str,
    grader_image_ref: str,
) -> dict:
    """
    Run integrity verification in the trusted grading container.
    
    Launches a container FROM the trusted codex-long-grader image.
    All tools (bash, jq, sha256sum, grep) come from this trusted
    image — they cannot be subverted by the agent.
    
    Bind-mounts (all read-only except /results/):
    - /agent/         ← agent filesystem extracted from snapshot
    - /functional/    ← Phase 2 results
    - /verifier/      ← verifier scripts + milestones
    - /verifier_data/ ← reference checksums, golden files
    - /results/       ← output (writable)
    
    Runs /verifier/verify.sh which produces verify_result.json
    containing both the binary pass/fail verdict AND milestone results.
    verify.sh is the single source of truth (LLD-13 §9.2).
    """
    results_dir = f"{grading_dir}/results"
    agent_root_dir = f"{grading_dir}/agent_root"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(agent_root_dir, exist_ok=True)

    # Extract agent filesystem from snapshot (driver-agnostic, per LLD-13 §6.4)
    extract_container = f"agent-fs-extract-{task.scenario_id.replace('/', '-')}"
    await run_cmd(
        f"docker create --name {extract_container} {snapshot_ref} true"
    )
    await run_cmd(
        f"docker cp {extract_container}:/. {agent_root_dir}/"
    )
    await run_cmd(f"docker rm {extract_container}")

    # Launch Phase 3 in trusted grading container
    container_name = f"phase3-{task.scenario_id.replace('/', '-')}"

    await docker_run_and_wait(
        image=grader_image_ref,
        name=container_name,
        network="none",
        volumes={
            agent_root_dir: {"bind": "/agent/", "mode": "ro"},
            f"{grading_dir}/functional": {"bind": "/functional/", "mode": "ro"},
            f"verifiers/{task.family_id}": {"bind": "/verifier/", "mode": "ro"},
            f"verifier_data/{task.family_id}": {"bind": "/verifier_data/", "mode": "ro"},
            results_dir: {"bind": "/results/", "mode": "rw"},
        },
        command="/verifier/verify.sh",
        timeout_seconds=300,  # integrity checks should be fast; 5 min generous limit
        remove_after=True,
    )

    # Parse verify_result.json
    result_path = f"{results_dir}/verify_result.json"
    if not os.path.exists(result_path):
        logger.error(
            f"Phase 3 did not produce verify_result.json for {task.scenario_id}"
        )
        return {"pass": False, "milestones": {}, "errors": ["No verify_result.json"]}

    with open(result_path) as f:
        verify_result = json.load(f)

    logger.info(
        f"Phase 3 complete for {task.scenario_id}: "
        f"pass={verify_result.get('pass')}, "
        f"milestones={verify_result.get('milestones', {})}"
    )

    return verify_result
```

### 7.8 Post-Grading Cleanup

```python
async def cleanup_grading(
    grading_dir: str,
    snapshot_ref: str,
    retain_snapshot: bool = True,
) -> None:
    """
    Clean up grading workspace after Phase 2+3 complete.
    
    CRITICAL: The committed snapshot image is NOT removed.
    Per LLD-02 §6.5, snapshot images are retained for all finished
    Codex-Long runs to support regrade-only recovery after manifest
    bumps. The snapshot is the handle that makes REGRADE_NEEDED
    implementable without re-running the agent session.
    
    Only the grading workspace (extracted filesystem, Phase 2/3
    result files) is cleaned up.
    """
    # Clean up extracted agent filesystem and grading results
    if os.path.exists(grading_dir):
        shutil.rmtree(grading_dir)

    # DO NOT remove the committed snapshot image
    if not retain_snapshot:
        # Only allowed during explicit operator-initiated purge
        # or after campaign finalization (LLD-02 §6.5 rules)
        await run_cmd(f"docker rmi {snapshot_ref}")
```

### 7.9 Regrade-Only Recovery Path

When LLD-07 dispatches a task with `dispatch_decision = "regrade_needed"`, the orchestrator skips the agent session and jumps directly to Phase 2+3 grading from the retained snapshot.

```python
async def execute_regrade(
    task: TaskSpec,
    manifest: dict,
    grader_image_ref: str,
    output_dir: str,
) -> dict:
    """
    Re-execute grading (Phase 2 + Phase 3) from a retained snapshot
    without re-running the agent session.
    
    Used when a grading artifact (verifier, milestone, verifier_data,
    family_spec, grader_image) has been updated via a manifest bump
    but the agent session output is still valid.
    
    The snapshot_ref comes from the invalidated run's snapshot_image_ref
    field in LLD-02 (passed via TaskSpec.regrade_snapshot_ref).
    
    Cost: seconds (vs 40–110 min for a full rerun).
    """
    snapshot_ref = task.regrade_snapshot_ref
    if not snapshot_ref:
        raise TaskDispatchError(
            f"Regrade requested for {task.scenario_id} but no "
            f"snapshot_image_ref is available. Cannot regrade."
        )

    # Verify the snapshot image exists
    if not await docker_image_exists(snapshot_ref):
        raise TaskDispatchError(
            f"Retained snapshot image '{snapshot_ref}' not found. "
            f"Snapshot may have been pruned. Full rerun required."
        )

    # Pre-grading hash verification — manifest may have changed
    verify_pre_grading_hashes(task, manifest, grader_image_ref)

    # Load family spec for functional check commands
    family_spec = load_family_spec(task.family_id)

    # Execute Phase 2 + Phase 3
    grading_dir = f"{output_dir}/grading/{task.scenario_id.replace('/', '_')}"
    await phase2_functional_checks(snapshot_ref, task, family_spec, grading_dir)
    verify_result = await phase3_integrity_verification(
        snapshot_ref, task, grading_dir, grader_image_ref
    )

    # Cleanup grading workspace (retain snapshot)
    await cleanup_grading(grading_dir, snapshot_ref, retain_snapshot=True)

    return verify_result
```

---

## 8. Manifest Loading and Configuration

### 8.1 Manifest State

The orchestrator loads the benchmark manifest once at initialization and refreshes it before each Codex-Long task to catch mid-campaign manifest bumps.

```python
class ManifestState:
    """
    In-memory cache of benchmark_manifest.lock with refresh support.
    
    The manifest is the root of trust for all hash verification.
    It is loaded from disk at startup and can be refreshed before
    each Codex-Long task to pick up post-freeze bumps.
    """

    def __init__(self, manifest_path: str, grader_image_tag: str):
        self.manifest_path = manifest_path
        self.grader_image_tag = grader_image_tag
        self.manifest: dict = {}
        self.manifest_version: int = 0
        self.grader_image_ref: str = ""
        self.reload()

    def reload(self) -> None:
        with open(self.manifest_path) as f:
            self.manifest = yaml.safe_load(f)
        self.manifest_version = self.manifest["manifest_version"]
        self.grader_image_ref = (
            f"{self.grader_image_tag}@{self.manifest['grader_image_digest']}"
        )
        logger.info(f"Manifest loaded: version {self.manifest_version}")
```

### 8.2 Orchestrator Configuration

```yaml
# orchestrator_config.yaml

vllm:
  bind_host: "127.0.0.1"   # vLLM server bind address — unchanged from LLD-01 §5.2
  client_host: "127.0.0.1" # host-side client address for /health, /metrics, /reset_prefix_cache
  port: 8000

network:
  name: "codex-bench-net"   # Docker bridge network (§5A.1)
  subnet: "172.30.0.0/16"
  gateway: "172.30.0.1"
  proxy_port: 8001          # inference-only proxy port (§5.4)

model_registry_path: "model_registry.yaml"  # LLD-01 §3.1 — loaded at init into config.model_registry (dict)
# OrchestratorConfig.__init__() loads this YAML file into a dict keyed by model ID.
# Downstream functions receive config.model_registry as dict, not path string.
# Same load-once-at-init pattern as ManifestState for benchmark_manifest.lock.

paths:
  output_dir: "/data/codex-bench/output"
  trajectory_dir: "/data/codex-bench/output/trajectories"
  patch_dir: "/data/codex-bench/output/patches"
  grading_dir: "/data/codex-bench/grading"
  manifest_path: "benchmark_manifest.lock"
  scenario_families_dir: "scenario_families"
  verifiers_dir: "verifiers"
  verifier_data_dir: "verifier_data"

grading:
  grader_image_tag: "codex-long-grader"
  phase2_default_timeout: 120
  phase3_timeout: 300

execution:
  swe_bench_timeout: 7200      # 2 hours for SWE-bench
  codex_long_timeout: 9000     # 2.5 hours for Codex-Long (longer sessions)
  health_check_retries: 5
  health_check_delay: 10.0

codex:
  binary_path: "/usr/local/bin/codex"                          # host path to codex binary
  node_modules_path: "/usr/local/lib/node_modules/@openai/codex"  # host path to codex package
  node_binary_path: "/usr/local/bin/node"                      # host Node.js runtime
```

---

## 9. Structured Event Stream Capture

### 9.1 JSONL Event Stream

`codex exec --json` produces a JSONL event stream on stdout. Each line is a self-contained JSON object representing one event in the Codex session (model turns, tool calls, tool results, system messages).

```python
async def capture_event_stream(
    process: asyncio.subprocess.Process,
    trajectory_path: str,
    task_id: str,
) -> None:
    """
    Capture the JSONL event stream from codex exec stdout.
    
    Events are written to the trajectory file line-by-line as they
    arrive (streaming capture, not buffered). This ensures partial
    trajectories are preserved on crash or timeout.
    
    The trajectory file is the primary data artifact consumed by:
    - LLD-04 (latency telemetry — turn-level timestamps)
    - LLD-06 (trajectory parser — SFT training format)
    
    No --ephemeral: persistent session files are also maintained
    by Codex as a backup (HLD §5).
    """
    with open(trajectory_path, "w") as f:
        # Write header with run metadata
        header = {
            "type": "run_metadata",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "codex_flags": "--yolo --json",
        }
        f.write(json.dumps(header) + "\n")

        async for line in process.stdout:
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                f.write(decoded + "\n")
                f.flush()  # ensure durability on crash
```

### 9.2 Trajectory Directory Structure

```
output/
  trajectories/
    trajectory_qwen3.5-27b_django__django-11099_seed1_a1.jsonl     # SWE-bench
    trajectory_qwen3.5-27b_dep-migration-npm_lodash-3-to-4_seed1_a1.jsonl  # Codex-Long
  patches/
    django__django-11099_qwen3.5-27b_seed1.patch                   # SWE-bench only
  grading/
    dep-migration-npm_lodash-3-to-4/                               # Codex-Long only
      functional/
        npm_test_exit_code
        npm_test_output.log
      results/
        verify_result.json
```

---

## 10. Main Execution Loop

### 10.1 Single-Task Execution

```python
async def execute_task(
    task: TaskSpec,
    pool_manager: PoolManager,         # LLD-02
    latency_capture: LatencyCapture,   # LLD-04
    manifest_state: ManifestState,
    config: OrchestratorConfig,
) -> RunResult:
    """
    End-to-end execution of a single task.
    
    This is the primary entry point called by LLD-07 for each dispatched task.
    Handles both SWE-bench and Codex-Long tracks, including the regrade-only path.
    
    Run-state contract (LLD-02 signed-off §5/§7):
    - claim_run() atomically claims the slot (exec_state='running').
      Returns False if another claimant already won → DuplicateClaimError.
    - finish_run() records completion with a TERMINAL outcome from
      the HLD failure contract: resolved / failed / no_patch / timeout / crash.
      No invented outcome values. No deferred callbacks.
    
    SWE-bench outcome flow (§5B.4):
    - Patch evaluation is SYNCHRONOUS within this function. After patch
      extraction, LLD-03 invokes the `codex-bench-eval-swe` CLI (§5B.4)
      which LLD-05 implements. LLD-03 reads the exit code for the
      terminal outcome (resolved/failed/crash) before finish_run().
      No update_outcome() API needed on signed-off LLD-02.
    
    Manifest timing contract (P0-2 v0.3 fix):
    - For Codex-Long: manifest_state.reload() runs BEFORE claim_run()
      so launch_manifest_ver is accurate.
    - A second manifest_state.reload() runs AFTER the agent session
      and BEFORE pre-grading hash checks so grading_manifest_ver is accurate.
    - Mid-run bumps are detected at the correct phase boundary.
    
    Cleanup contract:
    - A finally block guarantees container/workspace cleanup on any failure.
    - Retained snapshot images are never removed on crash (LLD-02 §6.5).
    """
    run_id = make_run_id(task)
    container: Optional[ContainerContext] = None
    snapshot_ref: Optional[str] = None
    grading_dir: Optional[str] = None
    codex_result: Optional[CodexResult] = None
    telemetry_started: bool = False

    # ── Regrade-only shortcut (Codex-Long only) ──
    if task.dispatch_decision == "regrade_needed":
        return await _execute_regrade_path(
            task, pool_manager, manifest_state, config
        )

    # ── 1. Pre-task checks ──
    await flush_prefix_cache(config.vllm.client_host, config.vllm.port)
    await health_check(
        config.vllm.client_host, config.vllm.port, task.model_id,
        max_retries=config.execution.health_check_retries,
    )

    # ── 2. Manifest reload and pre-run verification (Codex-Long) ──
    # BEFORE claim_run — do not consume an attempt on artifact drift.
    if task.track == "codex_long":
        manifest_state.reload()
        verify_pre_run_hashes(task, manifest_state.manifest)

    # ── 3. Atomically claim the run slot (LLD-02 §7) ──
    claimed = pool_manager.claim_run(
        track=task.track,
        pool_or_split=task.pool_or_split,
        scenario_id=task.scenario_id,
        model_id=task.model_id,
        harness=task.harness,
        seed=task.seed,
        attempt=task.attempt,
        family_id=task.family_id if task.track == "codex_long" else None,
        scenario_type=task.scenario_type if task.track == "codex_long" else None,
        launch_manifest_ver=(
            manifest_state.manifest_version
            if task.track == "codex_long" else None
        ),
    )
    if not claimed:
        raise DuplicateClaimError(
            f"Run slot already claimed for {task.scenario_id} "
            f"model={task.model_id} seed={task.seed} attempt={task.attempt}."
        )

    launch_manifest_ver = manifest_state.manifest_version if task.track == "codex_long" else None

    try:
        # ── 4. LLD-04 pre-task metrics snapshot ──
        # Inside the try block so that if snapshot_before() throws,
        # the except handler calls finish_run(crash) and the claimed
        # run slot is not stranded in 'running' state.
        await latency_capture.snapshot_before(task_id=task.scenario_id)
        telemetry_started = True
        # ── 5. Container setup ──
        if task.track == "swe_bench":
            container = await setup_swe_bench_container(task, config)
        else:
            container = await setup_codex_long_container(
                task, manifest_state.manifest, config
            )

        # ── 6. Codex invocation ──
        codex_result = await invoke_codex(
            container, task, config.paths.output_dir,
        )

        # ── 7. Post-run processing ──
        verify_result = None

        if task.track == "swe_bench":
            patch_path = await extract_swe_bench_patch(
                container, config.paths.output_dir, task
            )
            await teardown_swe_bench_container(container)
            container = None

            # Determine outcome — LLD-03 "drives" LLD-05 (per index).
            # LLD-05 owns predictions.jsonl conversion and swebench harness.
            # LLD-03 reads the exit code for the terminal outcome.
            if codex_result.timed_out:
                outcome = "timeout"
            elif _is_infrastructure_error(codex_result.stderr):
                outcome = "crash"
            elif patch_path is None:
                outcome = "no_patch"
            else:
                outcome = await drive_swe_bench_eval(
                    instance_id=task.instance_id,
                    patch_path=patch_path,
                    output_dir=config.paths.output_dir,
                )

        else:
            # Codex-Long three-phase grading
            snapshot_ref = await phase1_snapshot(container, run_id)
            container = None  # agent container removed by phase1_snapshot

            # Reload manifest AFTER agent session for accurate grading_manifest_ver
            manifest_state.reload()

            verify_pre_grading_hashes(
                task, manifest_state.manifest, manifest_state.grader_image_ref
            )

            family_spec = load_family_spec(task.family_id)
            grading_dir = f"{config.paths.grading_dir}/{run_id}"

            await phase2_functional_checks(
                snapshot_ref, task, family_spec, grading_dir
            )
            verify_result = await phase3_integrity_verification(
                snapshot_ref, task, grading_dir, manifest_state.grader_image_ref
            )
            outcome = determine_outcome(
                codex_result, "codex_long", verify_result=verify_result
            )
            await cleanup_grading(grading_dir, snapshot_ref, retain_snapshot=True)
            grading_dir = None  # mark as cleaned up

        # ── 8. Record run completion with TERMINAL outcome ──
        # outcome is always one of: resolved / failed / no_patch / timeout / crash
        # — matches the HLD failure contract and LLD-02 §5.1 exactly.
        pool_manager.finish_run(
            track=task.track,
            pool_or_split=task.pool_or_split,
            scenario_id=task.scenario_id,
            model_id=task.model_id,
            harness=task.harness,
            seed=task.seed,
            attempt=task.attempt,
            outcome=outcome,
            wall_time_seconds=codex_result.wall_time_seconds,
            trajectory_path=codex_result.trajectory_path,
            grading_manifest_ver=(
                manifest_state.manifest_version
                if task.track == "codex_long" else None
            ),
            snapshot_image_ref=snapshot_ref,
            codex_long_pass=(
                verify_result.get("pass") if verify_result else None
            ),
            milestone_results=(
                verify_result.get("milestones") if verify_result else None
            ),
        )

        return RunResult(
            task=task,
            outcome=outcome,
            trajectory_path=codex_result.trajectory_path,
            wall_time_seconds=codex_result.wall_time_seconds,
            verify_result=verify_result,
        )

    except ManifestMismatchError as e:
        # ── Grading blocked — artifact hash mismatch (v1.1) ──
        #
        # Pre-grading hash verification failed. The agent session completed
        # and the snapshot exists, but on-disk grading artifacts do not
        # match the manifest. This fires ONLY from verify_pre_grading_hashes().
        #
        # LLD-03's responsibility:
        #   1. Record the run as finished/crash with snapshot preserved
        #   2. Re-raise to LLD-07 as a non-retryable blocked condition
        #
        # RECOVERY PROTOCOL (operator-driven, single path):
        #   All resolutions require a manifest version bump — even if the
        #   fix is restoring on-disk artifacts to match the prior manifest.
        #   This is required because the signed-off LLD-02 invalidation
        #   API transitions runs to REGRADE_NEEDED only when
        #   grading_manifest_ver < new_manifest_version. Without a bump,
        #   the run stays a current crash with no path to regrade.
        #
        #   1. Operator fixes the artifact mismatch (either update on-disk
        #      artifacts OR fix the manifest to match).
        #   2. Operator bumps manifest_version in benchmark_manifest.lock
        #      with a changelog entry per LLD-13 §12.5 — even for a
        #      restore-to-prior-state fix. The bump is the state-machine
        #      trigger, not just a bookkeeping step.
        #   3. Operator runs invalidate_stale_runs() for the affected
        #      family/families with the new manifest version.
        #      → grading_manifest_ver (= launch_manifest_ver) < new version
        #      → is_current=0, recovery_action="regrade_only"
        #   4. Next dispatch cycle: check_dispatch_eligible() returns
        #      REGRADE_NEEDED. LLD-03 re-executes Phase 2+3 from the
        #      retained snapshot.
        #
        #   There is no "fix artifacts without bumping" path. This is
        #   intentional: LLD-13 §12.5 already requires a version bump
        #   and changelog entry for any post-freeze artifact change.
        #   A restore-to-prior-state is still a change that must be
        #   auditable in the manifest changelog.
        #
        # REQUIRED LLD-07 CONTRACT (same-sprint):
        #   LLD-07 must catch ManifestMismatchError and treat it as a
        #   NON-RETRYABLE blocked condition. The task must not be
        #   re-dispatched until the operator completes the recovery
        #   protocol above. If LLD-07 retries, the retry hits the same
        #   mismatch — an expensive no-op loop.

        logger.error(
            f"GRADING BLOCKED for {task.scenario_id}: {e}. "
            f"artifact={e.affected_artifact}, "
            f"launch_ver={launch_manifest_ver}, "
            f"current_ver={manifest_state.manifest_version}. "
            f"Run recorded as crash with snapshot preserved. "
            f"Recovery: operator must fix artifacts, bump manifest, "
            f"then run invalidate_stale_runs(). See §10.1 recovery protocol."
        )

        pool_manager.finish_run(
            track=task.track,
            pool_or_split=task.pool_or_split,
            scenario_id=task.scenario_id,
            model_id=task.model_id,
            harness=task.harness,
            seed=task.seed,
            attempt=task.attempt,
            outcome="crash",
            wall_time_seconds=codex_result.wall_time_seconds if codex_result else 0,
            trajectory_path=codex_result.trajectory_path if codex_result else None,
            grading_manifest_ver=launch_manifest_ver,
            snapshot_image_ref=snapshot_ref,  # preserved for future regrade
        )
        raise  # LLD-07 catches ManifestMismatchError specifically

    except Exception as e:
        logger.error(f"Task execution failed for {task.scenario_id}: {e}")
        pool_manager.finish_run(
            track=task.track,
            pool_or_split=task.pool_or_split,
            scenario_id=task.scenario_id,
            model_id=task.model_id,
            harness=task.harness,
            seed=task.seed,
            attempt=task.attempt,
            outcome="crash",
            wall_time_seconds=0,
            trajectory_path=None,
        )
        raise

    finally:
        # ── Guaranteed cleanup on any exit path ──
        # Retained snapshot images are NOT removed here (LLD-02 §6.5).

        # Telemetry pairing: if snapshot_before was called, always call
        # snapshot_after so LLD-04 never has an orphaned before-snapshot.
        # On error paths, the after-snapshot captures whatever vLLM state
        # exists at failure time — LLD-04 can detect the anomaly from
        # the missing/short task duration.
        if telemetry_started:
            try:
                await latency_capture.snapshot_after(task_id=task.scenario_id)
            except Exception:
                pass  # telemetry failure must not mask the real error

        if container is not None:
            try:
                await docker_rm(container.container_id, force=True)
            except Exception as cleanup_err:
                logger.warning(
                    f"Failed to clean up agent container "
                    f"{container.container_id}: {cleanup_err}"
                )

        if grading_dir is not None and os.path.exists(grading_dir):
            try:
                shutil.rmtree(grading_dir)
            except Exception as cleanup_err:
                logger.warning(
                    f"Failed to clean up grading workspace "
                    f"{grading_dir}: {cleanup_err}"
                )

        config_dir = f"/tmp/codex-bench/configs/{task.scenario_id.replace('/', '_')}"
        if os.path.exists(config_dir):
            try:
                shutil.rmtree(config_dir)
            except Exception:
                pass  # non-critical temp file
```

---

## 11. Timeout Enforcement

### 11.1 Per-Task Timeout

```python
async def docker_exec_with_timeout(
    container_id: str,
    command: str,
    timeout_seconds: int,
    stdout_sink: str,
) -> ExecResult:
    """
    Execute a command inside a container with a wall-clock timeout.
    
    Stdout capture design (P1-1 fix):
    - A single async reader drains process.stdout line-by-line to disk.
    - A separate reader drains process.stderr to an in-memory buffer.
    - process.wait() is used for exit code — NOT process.communicate(),
      which would compete with the stdout reader for the same pipe
      and cause deadlocks or dropped events.
    - Both readers and process.wait() are gathered under asyncio.wait_for()
      for unified timeout handling.
    
    On timeout:
    - The codex exec process is killed (SIGTERM → SIGKILL after 10s)
    - The container is NOT immediately destroyed (Phase 1 snapshot
      may still be taken for Codex-Long tasks)
    - Partial trajectory data is preserved (streaming capture to disk)
    - The ExecResult.timed_out flag is set
    
    Timeout values (from config):
    - SWE-bench: 7200s (2 hours)
    - Codex-Long: 9000s (2.5 hours — longer multi-turn sessions)
    
    These are conservative defaults. Gate 4 pilot measures actual
    wall-clock per scenario type; timeouts may be adjusted after
    Gate 4 completes.
    
    Note: env vars (CODEX_SEED, VLLM_API_KEY) are set at container
    launch time (§5.3), not at docker exec time. docker exec inherits
    the container's environment.
    """
    process = await asyncio.create_subprocess_exec(
        "docker", "exec",
        container_id, "bash", "-c", command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _drain_stderr() -> str:
        """Read stderr to memory. Stderr is small (error messages only)."""
        data = await process.stderr.read()
        return data.decode("utf-8", errors="replace")

    try:
        # Three concurrent tasks under a single timeout:
        # 1. capture_event_stream: drains stdout line-by-line to disk
        # 2. _drain_stderr: reads stderr to memory
        # 3. process.wait(): waits for exit code
        #
        # capture_event_stream is the sole consumer of process.stdout.
        # _drain_stderr is the sole consumer of process.stderr.
        # No dual-consumer conflict.
        stdout_task = asyncio.create_task(
            capture_event_stream(process, stdout_sink, container_id)
        )
        stderr_task = asyncio.create_task(_drain_stderr())
        wait_task = asyncio.create_task(process.wait())

        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, wait_task),
            timeout=timeout_seconds,
        )

        stderr_text = stderr_task.result()

        return ExecResult(
            returncode=process.returncode,
            stderr=stderr_text,
            timed_out=False,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Task timed out after {timeout_seconds}s in container {container_id}"
        )
        # SIGTERM first, then SIGKILL after grace period
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        return ExecResult(
            returncode=-1,
            stderr="TIMEOUT",
            timed_out=True,
        )
```

### 11.2 Grading Timeouts

Phase 2 and Phase 3 have their own timeouts separate from the agent session timeout.

| Phase | Default Timeout | Rationale |
|---|---|---|
| Phase 2 (per functional check) | From family spec `timeout_seconds` (default 120s) | Test suite runtime varies by scenario |
| Phase 3 (integrity verification) | 300s (5 min) | Integrity checks are fast; 5 min is generous |

If Phase 2 times out, the functional check is marked as failed (exit code captured as -1). Phase 3 can still proceed — a timed-out Phase 2 does not prevent integrity verification.

---

## 12. Error Handling and Recovery

### 12.1 Infrastructure Crash Detection

```python
def _is_infrastructure_error(stderr: str) -> bool:
    """
    Distinguish infrastructure failures from task-level failures.
    
    Infrastructure errors (→ outcome = crash, eligible for retry):
    - vLLM server crash or OOM
    - Docker daemon failure
    - Codex CLI internal error (not a model/task failure)
    - Disk full
    
    Task-level failures (→ outcome = failed):
    - Agent could not solve the task
    - Agent produced incorrect output
    - Model produced malformed tool calls
    """
    crash_patterns = [
        "CUDA out of memory",
        "vLLM server",
        "docker: Error",
        "No space left on device",
        "codex: internal error",
        "ConnectionRefusedError",
    ]
    return any(pattern.lower() in stderr.lower() for pattern in crash_patterns)
```

### 12.2 Retry Behavior

Per LLD-02 §5.5: if a task crashes due to infrastructure failure, the orchestrator may retry once (max 2 total attempts). LLD-07 handles this by re-dispatching the task with `dispatch_decision = "retry"` and `attempt = 2`.

The orchestrator does not retry autonomously — retry decisions are made by LLD-07 based on the dispatch eligibility check from LLD-02.

### 12.3 Partial Trajectory Preservation

On crash or timeout, the trajectory file is partially written (streaming capture). This partial data may still be useful for:

- Diagnostic analysis (where did the agent get stuck?)
- Turn-level latency metrics for completed turns (LLD-04)
- SFT data for completed-but-not-solved runs (LLD-06 filters on outcome)

Partial trajectories are never used for training without explicit filtering by LLD-06/LLD-10.

---

## 13. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-01** vLLM Serving Layer | Calls `GET /health` and `GET /v1/models` for readiness gating (host-side, `127.0.0.1:8000`). Calls `POST /reset_prefix_cache` between tasks (host-side, never proxied). Codex CLI targets `/v1/responses` for inference via the inference proxy on `<gateway>:8001` (§5.4). Coordinates with LLD-04 for `/metrics` snapshot timing (host-side). No LLD-01 amendment required — vLLM stays bound to `127.0.0.1` (§5A.1). |
| **LLD-02** Data Pool Manager | Calls `claim_run()` to atomically claim a run slot (returns bool — `False` on duplicate claim). Calls `finish_run()` to record completion with a terminal outcome from the HLD contract (`resolved` / `failed` / `no_patch` / `timeout` / `crash`). Passes `family_id`, `scenario_type`, and `launch_manifest_ver` at claim for Codex-Long runs. Records `snapshot_image_ref` for Codex-Long runs. No invented outcome values; no deferred callbacks. |
| **LLD-04** Latency Telemetry Capture | Coordinates pre-task and post-task `/metrics` snapshot timing. Emits turn-level timestamps in the trajectory JSONL that LLD-04 uses for per-turn metric attribution. |
| **LLD-05** Evaluator (Dual-Track) | *SWE-bench path:* LLD-03 "drives" LLD-05 by invoking `codex-bench-eval-swe` after patch extraction (§5B.4). LLD-05 owns predictions.jsonl conversion, swebench harness invocation, and report generation. LLD-03 reads the exit code for the terminal outcome. LLD-05 also aggregates outcomes from LLD-02 run records into solve rates, bootstrap CIs, and result tables. *Codex-Long path:* LLD-03 produces `verify_result.json` via the three-phase grading pipeline. LLD-05 reads this for solve rates and milestone aggregation. LLD-05 does not execute verifiers — `verify.sh` is the sole execution authority (LLD-13 §9.2). |
| **LLD-06** Trajectory Parser | Emits raw JSONL trajectory files that LLD-06 parses into SFT training records. Trajectory file paths are recorded in LLD-02 run records so LLD-06 can locate them. |
| **LLD-07** Benchmark Runner | Receives dispatched `TaskSpec` records from LLD-07. Reports `RunResult` back. LLD-07 manages campaign-level coordination (which tasks, which order, model switching) — this LLD handles per-task execution only. **Required contract (same-sprint):** LLD-07 must catch `ManifestMismatchError` from `execute_task()` and treat it as a non-retryable blocked condition. The task must not be re-dispatched until an operator resolves the artifact drift and bumps the manifest. See §10.1 handler comments. |
| **LLD-09** mini-SWE-Agent | No direct interaction. LLD-09 is a separate harness with its own execution loop. Both LLD-03 and LLD-09 operate on the same Codex-Long Docker envs and invoke the same verifiers, but through independent code paths. |
| **LLD-13** Codex-Long Scenario Framework | Primary consumer. Launches agent containers from the Docker env factory images (§6). Implements the three-phase grading protocol (§7). Enforces the two-phase manifest hash verification (§12.6). Retains snapshot images per LLD-02 §6.5 (cross-LLD contract change — see §14). |

---

## 14. Cross-LLD Contract Amendments

### 14.1 LLD-13 Snapshot Retention Amendment

The signed-off LLD-13 (v0.6, §6.4) specifies that LLD-03 removes the committed snapshot image (`docker rmi codex-long-snapshot/<run_id>`) and the extracted agent filesystem (`/grading/<run_id>/agent_root`) after grading completes.

This LLD amends that behavior per LLD-02 §6.5:

- **Snapshot image (`codex-long-snapshot/<run_id>`):** RETAINED after grading. Not removed. This snapshot is the handle that makes LLD-02's `REGRADE_NEEDED` dispatch path implementable without re-executing the agent session.
- **Extracted agent filesystem (`/grading/<run_id>/agent_root`):** STILL CLEANED UP. This is a temporary extraction created for Phase 3 inspection; it is not the retained artifact.
- **Phase 2 and Phase 3 containers:** STILL CLEANED UP (`--rm` on docker run).

**Snapshot removal** is allowed only when:
1. The run has been superseded by a new current attempt (LLD-02 §6.5 rule 1)
2. The campaign is fully finalized in LLD-12 (LLD-02 §6.5 rule 2)
3. An explicit operator-initiated purge (LLD-02 §6.5 rule 3)

**LLD-13 errata required:** Add a note to LLD-13 §6.4 stating that snapshot image retention is governed by LLD-02 §6.5 and LLD-03 §14.1. The `docker rmi codex-long-snapshot/<run_id>` line in the Phase 3 example code should be annotated as "post-campaign cleanup only — not immediate post-grading."

### 14.2 LLD-05 SWE-bench Evaluation Contract (Same-Sprint, Bilateral)

LLD-05 is "Design S1 → Implement S1" — the same sprint as LLD-03. It is not signed off. The full bilateral contract is defined in §5B.4, specifying both sides of the interface in one reviewable section:

- **LLD-03 side (consumer):** invocation CLI, exit code reading, `drive_swe_bench_eval()` wrapper
- **LLD-05 side (implementer):** argument table, exit code semantics, artifact format, implementation responsibilities
- **Codex-Long path:** no CLI needed — LLD-05 reads `verify_result.json` from run records

**Sign-off condition:** LLD-03's SWE-bench execution path is not independently complete until LLD-05 explicitly adopts §5B.4 as its authoritative interface contract. The index and HLD place SWE-bench grading ownership on LLD-05; §5B.4 defines what LLD-03 needs from that ownership. LLD-05's design must reference §5B.4 and implement the `codex-bench-eval-swe` entry point. Until that happens, LLD-03's SWE-bench path is specified but has an open dependency on a same-sprint deliverable. This is tracked as a sign-off condition, not a design flaw — LLD-03 does not attempt to own or redefine LLD-05's scope.

**Signed-off LLD impact — none:**
- **LLD-02:** `claim_run()` / `finish_run()` / `invalidate_stale_runs()` used exactly as signed off. `invalidate_stale_runs()` is not called by LLD-03 directly — it is an operator/LLD-07 action after manifest bumps.
- **LLD-01:** vLLM stays on `127.0.0.1`. Inference proxy handles container reachability.
- **LLD-13:** Only the snapshot retention amendment (§14.1) requires an errata note.

---

## 15. Sprint 1 Validation Checklist

### Pre-Implementation

- [ ] Confirm `codex exec --yolo --json` produces complete JSONL event stream on DGX Spark
- [ ] Confirm seed mechanism: does `CODEX_SEED` env var produce distinct trajectories? If not, implement prompt-suffix fallback (§6.4)
- [ ] Set up `codex-bench-net` Docker bridge network (§5A.1)
- [ ] Set up inference proxy (`codex-bench-proxy`, nginx) on host port 8001 (§5.4)
- [ ] Confirm bridge-networked containers can reach inference proxy via gateway IP — and confirm `/metrics`, `/reset_prefix_cache`, `/v1/models` return 403 through the proxy
- [ ] Confirm egress blocking: from inside a container, `curl http://example.com` and `curl http://172.30.0.1:8000/metrics` both time out / connection refused (§5A.1 validation commands)
- [ ] Confirm bind-mounted Codex binary + Node.js runtime works inside env factory images (§5.3) — particularly for Codex-Long images that may have their own Node.js version
- [ ] Confirm `docker commit` preserves ENV/WORKDIR/PATH for Phase 2 functional checks
- [ ] Confirm `docker cp` extraction is driver-agnostic on DGX Spark (rootful Docker, overlay2)
- [ ] Confirm resource limits (`--memory 32g --cpus 4 --pids-limit 1024`) do not impede normal task execution

### End-to-End Validation (HLD Sprint 1 checklist item)

- [ ] 10 SWE-bench tasks through the full orchestrator pipeline:
  - Container setup (with Codex harness bootstrap) → `codex exec` → trajectory capture → patch extraction → `codex-bench-eval-swe` (LLD-05 CLI, §5B.4) → finish_run(outcome=terminal)
  - Verify: trajectory JSONL is complete, patch applies, outcome is resolved/failed/no_patch, LLD-02 run state is correct
- [ ] 10 Codex-Long scenarios through the full orchestrator pipeline:
  - Container setup (with Codex harness bootstrap) → `codex exec` → Phase 1 snapshot → Phase 2 functional checks → Phase 3 integrity verification → LLD-05 consume verify_result.json
  - Verify: three-phase grading produces correct verify_result.json, manifest hash checks pass, snapshot is retained, LLD-02 run state is correct (claim_run → finish_run with outcome + snapshot_image_ref)
- [ ] Prefix cache flush between tasks confirmed (monitor `/metrics` cache counters)
- [ ] Health check behavior on vLLM restart confirmed (retries → success after server comes up)
- [ ] Timeout behavior confirmed (partial trajectory preserved, outcome = timeout)
- [ ] Crash detection and retry confirmed (outcome = crash → LLD-07 re-dispatches with attempt = 2)
- [ ] Cleanup on exception path confirmed (no leaked containers or grading workspaces after crash)
- [ ] Duplicate claim detection confirmed (two concurrent dispatches for same task → one succeeds, one raises DuplicateClaimError)
- [ ] Manifest mismatch recovery confirmed: artificially bump a verifier hash, run a Codex-Long task → ManifestMismatchError caught, run recorded with snapshot_image_ref preserved, trajectory_path preserved

---

## 16. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **Seed mechanism unvalidated** | MEDIUM | Sprint 0 validation item. If `CODEX_SEED` is not honored, prompt-suffix fallback produces sufficient variation for statistical analysis but is less clean. Seed is injected at container launch via `get_codex_harness_env()` (§5.3). |
| **Codex binary + Node.js mount conflicts** | MEDIUM | Codex requires Node.js. Some env factory images (e.g., Node.js scenarios) ship their own Node.js. The bind-mounted host Node.js takes PATH precedence. Sprint 1 must validate that scenario-specific Node.js tooling (npm, npx) still works when the host Node.js is mounted at `/usr/local/bin/node`. If conflicts arise, mount Codex to an isolated path (e.g., `/opt/codex/`) and invoke via full path. |
| **SWE-bench Docker image availability** | LOW | SWE-bench provides official Docker evaluation images. If image pull fails, pre-build and cache locally during Sprint 1 setup. |
| **Synchronous SWE-bench eval adds per-task latency** | MEDIUM | `codex-bench-eval-swe` (LLD-05) runs synchronously after the agent session. At ~1–5 min per eval, this adds to per-task wall-clock but is small relative to the 40–110 min Codex session. |
| **Phase 2 functional check timeout varies by scenario** | MEDIUM | Family spec `timeout_seconds` field controls per-check timeout. During Gate 4 pilot, log actual Phase 2 wall-clock and flag scenarios with unexpectedly slow test suites (>120s). |
| **Docker disk usage from retained snapshots** | LOW | At ~50–200 MB per snapshot × ~400 Codex-Long runs = ~20–80 GB. Well within DGX Spark disk. Monitor with periodic `docker system df`. |
| **`codex exec` on ARM64 may have undocumented behavior** | MEDIUM | Sprint 0 Gates 1/1b validate Codex CLI behavior on the DGX Spark. Any ARM64-specific issues surface there. |
| **Manifest bump during active task** | LOW | Two-reload protocol: `manifest_state.reload()` before `claim_run()` (accurate launch_manifest_ver) and after agent session (accurate grading_manifest_ver). A bump during the agent session is detected at the second reload before pre-grading hash checks. |
| **Pre-grading manifest mismatch is non-retryable** | MEDIUM | `ManifestMismatchError` is re-raised to LLD-07 which must treat it as blocked, not retryable. All resolutions require a manifest version bump (§10.1 recovery protocol). If LLD-07 retries without operator intervention, the retry hits the same mismatch. |
| **Inference proxy is a single point of failure** | LOW | The proxy is a trivial nginx config. If the proxy crashes, Codex CLI gets connection errors → task fails → outcome = crash → retry. Sprint 1: add a health check for the proxy alongside the vLLM health check. |
| **iptables rules require root and persist across reboots** | LOW | The egress blocking rules (§5A.1) are in the DOCKER-USER chain. They must be re-applied after a reboot. The setup script should be run as part of Sprint 1 host initialization. Use `iptables-save`/`iptables-restore` or a systemd unit for persistence. |
| **Codex multi-turn 400 errors (vLLM Responses API gap)** | HIGH | LLD-01 §8.3 documents this risk. Sprint 0 Gate 1 validates end-to-end. If the issue is not resolved, Codex-Bench may require a vLLM upgrade or Codex CLI patch before any collection begins. |

---

## 17. Open Questions — Status

| Question | Status |
|---|---|
| Seed-to-behavior mapping for Codex CLI | **OPEN — Sprint 0 validation. See §6.4. Seed is injected via `CODEX_SEED` env var at container launch (§5.3). Fallback: prompt-suffix variation.** |
| SWE-bench Docker image source and caching strategy | **OPEN — Determine whether to use SWE-bench evaluation images directly or custom images. Decision during Sprint 1 setup.** |
| SWE-bench prompt format (AGENTS.md vs inline problem statement) | **OPEN — The exact prompt format for SWE-bench tasks needs alignment with the SWE-bench dataset format. §6.3 provides a placeholder; finalize during implementation.** |
| Parallel grading across completed Codex-Long runs | **OPEN — Not implemented in v0.4. Potential optimization if grading becomes a bottleneck. LLD-13 §16 notes this as possible.** |
| Codex binary mount path conflicts with scenario Node.js | **OPEN — Sprint 1 validation. If host Node.js at `/usr/local/bin/node` conflicts with scenario-specific Node.js tooling, move Codex to an isolated path (`/opt/codex/`). See §16 known issues.** |
| iptables rule persistence across reboots | **OPEN — The egress-blocking rules (§5A.1) must be re-applied after reboot. Use `iptables-save`/`iptables-restore` or a systemd unit. Decision during Sprint 1 setup.** |
| SWE-bench outcome flow | **Resolved (§5B.4 v0.7) — bilateral contract defined in one section. LLD-03 side: invocation + exit code reading. LLD-05 side: argument table, artifact format, implementation responsibilities. Same-sprint design, no signed-off doc amendments (§14.2).** |
| Codex config.toml management across models | **Resolved (§5.2) — config.toml is generated per-task by `generate_codex_config()` and bind-mounted into each container. No cross-task config state.** |
| Network isolation model for agent containers | **Resolved (§5.4, §5A v0.4) — bridge network + inference proxy + iptables egress blocking. Containers can only reach the proxy. vLLM stays on loopback. No LLD-01 amendment.** |
| Manifest provenance timing | **Resolved (§10.1 v0.3) — two-reload protocol: before claim_run and before grading.** |
| Atomic duplicate claim detection | **Resolved (§10.1 v0.3) — `claim_run()` return value checked.** |
| Bind address vs client address conflation | **Resolved (§8.2 v0.3) — config separates `vllm.bind_host` from `vllm.client_host`. Containers use proxy.** |
| Pre-grading mismatch recovery | **Resolved (§10.1 v1.1) — single recovery path: all resolutions require a manifest version bump (even restoring artifacts to match). Bump is the state-machine trigger for `invalidate_stale_runs()` → `REGRADE_NEEDED`. No "fix without bumping" branch. Aligns with LLD-13 §12.5 post-freeze rules. LLD-07 must treat `ManifestMismatchError` as non-retryable.** |

---

*LLD-03 · Task Orchestrator · Draft v1.1 · April 2026*
