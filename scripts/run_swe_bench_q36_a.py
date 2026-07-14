#!/usr/bin/env python3
"""SWE-Bench per-instance orchestrator for Q36-A + Codex CLI 0.128.0.

Spec ref: docs/reports/auto_research/swe-bench-bounded-time-spec-20260520.md
          §9 (artifact layout), §11 (per-task protocol), §7 (concurrency).

For each instance from a pre-registered subset (built by
scripts/build_swe_bench_subset.py):
  1. Hydrate the workspace at the SWE-Bench base_commit, drop AGENTS.md
     with the problem_statement.
  2. Launch codex-runner:v1 Docker against the codex-bench proxy at
     :8022 with a wall budget.
  3. Diff workspace vs base_commit -> patch.diff (per-attempt artifact).
  4. Invoke codex-bench-eval-swe on the patch.
  5. Write per-task artifacts under
     output/swe_bench_q36_a_temp06/<dataset>/per_task/<instance_id>/.
  6. Aggregate predictions.jsonl + campaign_summary.json.

Defaults follow the spec:
  - Concurrency: 1 (LLD-05 §4.6 default; bump after Sprint-1 validation).
  - Codex wall budget: NO harness cap by default (0 = unlimited; codex self-limits on
    stream_idle_timeout_ms + turn limit). Eval buffer: 30 min. Override: SWE_AGENT_WALL_S / --agent-wall-s.
  - Proxy: http://127.0.0.1:8022/v1
  - Reasoning effort: high (carried over from launch_qwen36_ablation_point.py).
  - Temperature is governed by the vLLM relaunch bundle (Q36-A: temp=0.6).
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "swe_bench_q36_a_temp06"
DEFAULT_REPO_CACHE = REPO_ROOT / ".cache" / "swe_bench_repos"
DEFAULT_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_ENDPOINT = "http://127.0.0.1:8022/v1"
DEFAULT_METRICS_URL = "http://127.0.0.1:9950/metrics"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_AGENT_WALL_S = 0  # per-attempt codex wall. 0 = NO upper limit (user 2026-07-01):
# hard tasks that genuinely need a long time were TIMED OUT mid-fix -> partial/wrong patch ->
# tests_failed (e.g. chain5 astropy-13453: timed_out=True at 27min under the old 25-min cap). With
# nudge-only (SWE_EMPTY_PATCH_RETRIES=0) a task gets ONE attempt, so the old fresh-session retry no
# longer silently grants extra wall-time -> we remove the harness cap entirely and let codex run to
# its own natural completion. This does NOT mean "hang forever": codex self-terminates on its own
# stream_idle_timeout_ms=600000 (10-min no-output idle) and its internal turn limit, and per-turn
# generation is bounded by MAX_OUTPUT (32768). Set SWE_AGENT_WALL_S>0 (or --agent-wall-s N) to
# reinstate a hard per-attempt wall of N seconds. NOTE: serial (concurrency=1) so a single stuck
# task can stall the whole sweep; the codex idle timeout is the real backstop.
DEFAULT_EVAL_TIMEOUT_S = 30 * 60
DEFAULT_MODEL_NAME_TAG = "qwen3.6-27b-fp8::qwen-code-0.19.4::q36-a"
# Same capture path used by launch_qwen36_ablation_point.py / Track B benches.
DEFAULT_PROXY_CAPTURE = Path("/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl")
DEFAULT_DCGM_SAMPLER = REPO_ROOT / "scripts" / "sample_dcgm_during_task.py"
DEFAULT_DCGM_INTERVAL_S = 0.1  # 10 Hz — 1/10th of Track B's 100 Hz default to keep
# per-task dcgm_samples.jsonl under GitHub's 100 MB per-file hard limit. At 10 Hz a
# 30-min Codex task lands ~6 MB; 100 Hz lands ~60 MB and risks rejection at the
# campaign scale (500 + 731 instances).

CODEX_TEMPLATE = (
    "docker run --rm --name {container_name} --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} -e HOME=/tmp "
    "-w /workspace codex-runner:v1 "
    "codex exec --json --skip-git-repo-check "
    "--dangerously-bypass-approvals-and-sandbox -C /workspace "
    "-c 'model_provider=\"local-proxy\"' "
    "-c 'model_providers.local-proxy={{name=\"local-proxy\","
    "base_url=\"{endpoint}\",env_key=\"OPENAI_API_KEY\","
    "wire_api=\"responses\",stream_idle_timeout_ms=600000}}' "
    "-c 'model_reasoning_effort=\"high\"' "
    "-c 'model_supports_reasoning_summaries=true' "
    "-c 'model_reasoning_summary=\"auto\"' "
    "--model {model}"
)

# qwen-code agent harness (SWE_AGENT=qwen_code). Qwen's own CLI: native XML tools +
# search/replace edits over /v1/chat/completions — aligned with Qwen's training, unlike
# codex's V4A apply_patch. ENTRYPOINT of qwen-code-runner:v1 is `qwen`, so we append its
# flags; the prompt is appended as the LAST argv element = the value of the trailing `-p`
# (works for both the local argv path and the remote shlex.quote path). --yolo auto-approves
# all edits/shell; --max-session-turns bounds a task (no harness wall) so a stuck task can't
# hang the serial queue. Sampling (temp 0.6) is forced proxy-side on /v1/chat/completions.
QWEN_CODE_TEMPLATE = (
    "docker run --rm --name {container_name} --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} "
    "-e OPENAI_MODEL={model} -e QWEN_MODEL={model} -e HOME=/tmp "
    # R1 context-budget fix (FR13_CONTEXT_COMPRESSION_DESIGN, verified in-image v0.19.4):
    # qwen-code reserves max(ESCALATED_MAX_TOKENS=64000, outputTokenLimit=65536) output tokens
    # unless overridden -> contextLimit = 131072-65536 = 65536 -> hard limit 48875.2. The proxy
    # caps real output at LUMO_PROXY_MAX_OUTPUT_TOKENS=32768, so reserve exactly that:
    # contextLimit=98304 -> hard limit 75304 (+54%). Uniform across all arms (eval-fair).
    "-e QWEN_CODE_MAX_OUTPUT_TOKENS=32768 "
    # FR13 §59 belt: raise qwen-code's stream-idle abort (built-in default 120000ms)
    # so a transient mid-stream upstream flake (GB10 emit/transport wedge) is not
    # converted into a fatal patch-less give-up. qwen-code 0.19.4 reads
    # QWEN_STREAM_IDLE_TIMEOUT_MS (env precedence: config field > env > default).
    # Modest raise (240000ms) paired with the runner stall-watchdog; on the live
    # ssh/offload path ${VAR:-240000} shell-expands remotely. Overridable via env.
    # NOTE: double braces — this string goes through str.format(); single braces
    # parse as a replacement field and crash dispatch (KeyError, tcfix_i5).
    "-e QWEN_STREAM_IDLE_TIMEOUT_MS=${{QWEN_STREAM_IDLE_TIMEOUT_MS:-240000}} "
    "-w /workspace qwen-code-runner:v1 "
    # user 2026-07-07: NO turn limit (100000 = effectively unlimited); tasks run to natural
    # submit/give-up. Backstop = 600s stall-watchdog + QWEN_STREAM_IDLE_TIMEOUT_MS, NOT a turn cap.
    "--yolo --output-format stream-json --max-session-turns 100000 -p"
)


def _agent_template() -> str:
    """Pick the agent command template from SWE_AGENT (default 'qwen_code' — user 2026-07-07).
    Codex has a built-in nudge/auto-continue that suppresses give-ups and CONFOUNDS the give-up
    gate; qwen-code is the honest agent (native XML tools, aligned with Qwen training)."""
    agent = os.environ.get("SWE_AGENT", "qwen_code").strip().lower()
    if agent in ("qwen_code", "qwen-code", "qwen"):
        return QWEN_CODE_TEMPLATE
    return CODEX_TEMPLATE


# --- FR13 §58: SWE_AGENT_ENV=instance_image (default unset='legacy') -----------
# When unset/'legacy' EVERY code path below is byte-identical to the current
# behavior. When ='instance_image', the qwen-code agent runs INSIDE the official
# SWE-bench per-instance eval image editing /testbed (a real git checkout at
# base_commit with the conda 'testbed' env), so it can `import astropy` + run a
# repro pytest before finishing — the env the grader already uses. The node+qwen
# runtime is injected read-only from a relocatable host bundle (built by
# scripts/prepare_qwen_agent_bundle.sh) so NO per-instance image is rebuilt.
def _swe_agent_env() -> str:
    """Agent environment. DEFAULT = 'instance_image' (user 2026-07-06): the agent
    runs INSIDE the official SWE-bench per-instance eval image with the conda
    'testbed' env — the benchmark-faithful setup (smoke-proven §67: import astropy
    editable + qwen 0.19.4 in-image). Requires the per-instance image on the codex
    host + the ~/qwen_agent_bundle (both provisioned; run prepare_qwen_agent_bundle.sh
    once). Set SWE_AGENT_ENV=legacy (or worktree) to fall back to the old
    qwen-code-runner:v1-over-bare-worktree behavior (which cannot self-verify, §58).
    A missing image FAILS LOUD per-instance (never silently falls back)."""
    val = os.environ.get("SWE_AGENT_ENV", "instance_image").strip().lower()
    if val in ("legacy", "worktree", "0", "off", "none"):
        return "legacy"
    return "instance_image"


def _instance_image_name(instance_id: str, *, arch: str = "x86_64",
                         namespace: str = "swebench", tag: str = "latest") -> str:
    """Per-instance SWE-bench eval image name, derived via swebench's OWN naming
    util (TestSpec.instance_image_key) rather than hand-rolled — so it is
    byte-identical to what the eval targets. The eval (swe_eval_x86_worker.py)
    calls make_test_spec(namespace='swebench', instance_image_tag='latest') whose
    arch defaults to 'x86_64' (the codex host, alienware, is x86_64).
    instance_image_key reads ONLY instance_id/arch/namespace/tag, so a minimal
    TestSpec reproduces the exact name (incl. the '__'->'_1776_' remap + .lower()).
    Fails loud (ImportError) if swebench is not installed on the codex host."""
    from swebench.harness.test_spec.test_spec import TestSpec
    spec = TestSpec(
        instance_id=instance_id, repo="", version="", repo_script_list=[],
        eval_script_list=[], env_script_list=[], arch=arch,
        FAIL_TO_PASS=[], PASS_TO_PASS=[], language="py", docker_specs={},
        namespace=namespace, instance_image_tag=tag,
    )
    return spec.instance_image_key


def _host_arch(remote_host: str | None) -> str:
    """uname -m of the AGENT host (local when remote_host is None). Returns the
    raw machine string ('x86_64' | 'aarch64' | ...)."""
    if remote_host:
        r = _net_retry(["ssh", *_EVAL_SSH_OPTS, remote_host, "uname -m"],
                       what=f"host_arch:{remote_host}", timeout=30, max_attempts=2)
        return (r.stdout or "").strip() or "unknown"
    return os.uname().machine


# uname -m -> the arch token swebench uses in image names.
_SWEB_ARCH = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "arm64", "arm64": "arm64"}


def _record_image_arch_tag(workspace: Path, instance_id: str, entry: dict) -> None:
    """USER 2026-07-05: some official SWE-bench per-instance images are x86_64-only
    (fine for the alienware offload, NOT runnable on GB10/aarch64) — persist a
    per-instance arch tag so FUTURE runs know placement constraints without
    rediscovering them. Sidecar: <workspace>/../images_arch.json (merge-update);
    the same fields also ride the returned codex_meta into runner_metadata.json."""
    try:
        sidecar = Path(workspace).parent / "images_arch.json"
        data: dict = {}
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[instance_id] = entry
        sidecar.write_text(json.dumps(data, indent=1, sort_keys=True),
                           encoding="utf-8")
    except Exception:
        pass  # tag is observability; never fail the run for it


# Fully-explicit container PATH (NO host '$PATH' dependency — a literal '$PATH'
# would survive shlex.split on the local argv path and land in the container
# verbatim, and would expand to the WRONG host on the ssh path). Puts the qwen
# shim + the conda 'testbed' env (import astropy / pytest) ahead of conda base +
# the standard ubuntu PATH the eval image ships.
_INSTANCE_CONTAINER_PATH = (
    "/opt/qwen/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
# In-container wrapper. Contains NO single quotes so it survives `bash -c
# '<wrapper>'` under BOTH shlex.split (local argv) AND the remote ssh login
# shell. It: (1) materializes AGENTS.md into /testbed from a base64 env
# (brace/quote-safe for any problem statement), (2) runs qwen whose
# --output-format stream-json emits NDJSON events live (one JSON object per line,
# flushed as each event occurs) as the ONLY stdout -> the trace. This replaces the
# old buffered `json` single-array framing, which did one giant process.stdout.write
# at process.exit() that drained only ONE 64KiB pipe buffer -> every >64KB trace was
# clamped to exactly 65536 bytes, cut mid-string, losing the terminal result record.
# NDJSON also matches the .jsonl filename and lets the byte-growth watchdog see real
# streaming growth, (3) extracts the patch with the SAME git flags as legacy
# _extract_patch, INSIDE /testbed (a real git checkout at base_commit — the
# secondary broken-.git finding does not apply here), writing it to the /out
# bind-mount (a fresh dir, NOT /testbed, so it never pollutes the diff; AGENTS.md
# is untracked so `git diff <base_commit>` excludes it exactly as legacy does),
# (4) preserves qwen's exit code.
_INSTANCE_WRAPPER = (
    "printf %s \"$SWE_AGENTS_B64\" | base64 -d > /testbed/AGENTS.md; "
    "PROMPT=$(printf %s \"$SWE_PROMPT_B64\" | base64 -d); "
    "/opt/qwen/bin/qwen --yolo --output-format stream-json --max-session-turns 100000 -p \"$PROMPT\"; "
    "rc=$?; "
    # Diff against HEAD, NOT base_commit. In the SWE-bench per-instance image /testbed
    # HEAD is `<base_commit> + the committed env-setup commit` (efa06c664 "SWE-bench";
    # e.g. it pins setuptools==68.0.0 in pyproject.toml). `git diff <base_commit>` would
    # capture that env-setup delta (~305B) as part of the patch (polluting/masking the
    # agent's real edits, breaking eval) — the §73 matrix-abort bug. The env-setup is
    # COMMITTED at HEAD, so `git diff HEAD` yields ONLY the agent's uncommitted source
    # edits (matches the gold-patch convention; eval applies on top of the same
    # env-setup state). AGENTS.md is untracked so still excluded.
    "git -C /testbed diff --no-color --binary HEAD > /out/patch.diff 2>/dev/null; "
    "exit $rc"
)


def _instance_agent_command(*, container_name: str, image: str, endpoint: str,
                            model: str, host_out_dir: str, bundle_src: str,
                            agents_md_b64: str, prompt_b64: str,
                            base_commit: str) -> str:
    """Render the instance_image docker command. Same qwen flags/env as the
    legacy qwen-code template, but the image is the per-instance eval image, the
    node+qwen runtime is bind-mounted read-only from the host bundle at
    /opt/qwen, PATH is prepended, workdir is /testbed, and AGENTS.md/prompt are
    passed base64 (brace/quote-safe). Pure string builder — self-testable with no
    docker/GPU. Runs as -u 0:0 because /testbed + the conda env are root-owned in
    the eval image."""
    return (
        f"docker run --rm --name {container_name} --network=host -u 0:0 "
        f"-e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} "
        f"-e OPENAI_MODEL={model} -e QWEN_MODEL={model} -e HOME=/tmp "
        f"-e QWEN_CODE_MAX_OUTPUT_TOKENS=32768 "
        # §79: instance-image agents were MISSING the idle-timeout raise (the legacy
        # QWEN_CODE_TEMPLATE had it, this default path did not) => qwen used its 120s
        # default => at B>=4 a request that QUEUES for capacity waits >120s for its
        # first chunk and the client aborts "after 0 chunks" => 0B. 600s tolerates the
        # B=4 queue (heartbeat can't cover pre-first-byte queue wait). Overridable.
        f"-e QWEN_STREAM_IDLE_TIMEOUT_MS=${{QWEN_STREAM_IDLE_TIMEOUT_MS:-600000}} "
        f"-e PATH={_INSTANCE_CONTAINER_PATH} "
        f"-e SWE_AGENTS_B64='{agents_md_b64}' -e SWE_PROMPT_B64='{prompt_b64}' "
        f"-e SWE_BASE_COMMIT='{base_commit}' "
        f"-v {host_out_dir}:/out -v {bundle_src}:/opt/qwen:ro "
        f"-w /testbed {image} "
        f"bash -c '{_INSTANCE_WRAPPER}'"
    )

# Default operator prompt (first attempt).
DEFAULT_AGENT_PROMPT = (
    "Read the task prompt at /workspace/AGENTS.md and complete it in this workspace. "
    "Edit the source files directly to implement the fix. Do not write a diff file -- "
    "modify the files in place so that running pytest passes the tests described in the prompt."
)
# Bundle B #2/#8: retry prompt when the first attempt left no patch (agent
# returned without editing source).
RETRY_PROMPT_EMPTY = (
    "Your previous attempt finished WITHOUT leaving any code change in the working tree. "
    "Re-read /workspace/AGENTS.md, inspect the relevant source files, and EDIT them now to "
    "implement the fix. Do not stop until you have made a concrete source edit. Do not waste "
    "time on environment setup or pip/conda installs -- the grader uses its own environment."
)
# Bundle B #3/#8: retry prompt when the first attempt looped on the same failing
# command (e.g. repeated pip/build errors) and left no patch.
RETRY_PROMPT_SETUP_LOOP = (
    "Your previous attempt repeatedly hit the same failing command (likely an environment/"
    "install/build step) and never edited the source. STOP trying that approach entirely. "
    "The grader builds its own environment, so you do NOT need the project to install or import. "
    "Read /workspace/AGENTS.md and the relevant source files, and directly EDIT the source to "
    "implement the fix."
)


def _prior_attempt_brief(trace_path: Path) -> str:
    """Summarize the prior codex attempt (files it inspected + its last message) so an
    agent_gave_up retry can be re-driven IN-CONTEXT rather than re-exploring from scratch."""
    files: list[str] = []
    last_msg = ""
    try:
        for line in trace_path.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line).get("item", {})
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "command_execution":
                cmd = str(item.get("command", ""))
                for tok in cmd.replace("'", " ").replace('"', " ").split():
                    t = tok.strip(",;:()[]")
                    if t.endswith(".py") and "/" in t and t not in files:
                        files.append(t)
            elif item.get("type") == "agent_message":
                txt = item.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    last_msg = txt.strip()
    except Exception:  # noqa: BLE001
        pass
    parts: list[str] = []
    if files:
        parts.append("You already inspected: " + ", ".join(files[:8]) + ".")
    if last_msg:
        parts.append('Your last message was: "' + last_msg[:400] + '".')
    return " ".join(parts)


def _retry_prompt_continue(brief: str) -> str:
    """State-conditional continuation for agent_gave_up: the agent EXPLORED then emitted a
    tool-call-free terminal reply (general temp-0.6 codex flake; reasoning is inert on Qwen3.6
    in this stack). Re-drive it with its own accumulated context + a hard must-act directive."""
    pre = (brief + " ") if brief else ""
    return (
        "Your previous attempt EXPLORED the code but STOPPED with NO edit in the working tree -- "
        "that is a FAILED attempt, not a completed one. " + pre +
        "Do NOT read or grep files again; you have enough context. Your VERY NEXT action MUST be "
        "an apply_patch that edits the source to implement the fix described in /workspace/AGENTS.md. "
        "Every response you produce must call a tool -- never end your turn with an analysis-only or "
        "summary message. Do not stop until the source files are edited."
    )


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_GIT_LOCK = threading.Lock()  # serialize per-task git ops when --concurrency>1


def _autocommit_task_artifacts(task_dir: Path, instance_id: str) -> None:
    """Best-effort auto-commit+push of ONE task's curated trace artifacts to the
    shared branch. NEVER raises (whole-body try/except = the `|| true` contract).
    output/ is .gitignored so traces are force-added (git add -f), matching the
    440k+ artifacts already tracked under output/. Commit uses an EXPLICIT pathspec
    (git commit -- <paths>, NOT add-all) so it never sweeps another session's staged
    work on the shared branch, and carries the repo Co-Authored-By trailer. _GIT_LOCK
    serializes git so --concurrency>1 tasks don't race on index.lock.
    LUMO_SWE_AUTOCOMMIT=0 disables it (e.g. a big campaign that doesn't want a push
    per task)."""
    try:
        if os.environ.get("LUMO_SWE_AUTOCOMMIT", "1").strip().lower() in ("0", "off", "false", "no"):
            return
        rels = ("runner_metadata.json", "patch.diff", "qwen_trace.jsonl",
                "eval/predictions.jsonl", "eval/eval_report.json",
                "orchestrator_crash.json")
        paths = [str(task_dir / r) for r in rels if (task_dir / r).is_file()]
        if not paths:
            return
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        def _git(argv, timeout=120):
            return subprocess.run(["git", *argv], cwd=str(REPO_ROOT), env=env,
                                  check=False, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=timeout).returncode
        msg = (f"FR13 SWE auto-commit: {instance_id} trace artifacts\n\n"
               "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
        with _GIT_LOCK:
            _git(["add", "-f", "--", *paths])
            # -m/msg MUST precede the `--` pathspec separator; nothing-to-commit=>rc1
            # (ok), non-fast-forward push=>rc!=0 (ok).
            if _git(["commit", "-m", msg, "--", *paths]) == 0:
                _git(["push", "origin", "HEAD"], timeout=300)
    except Exception:
        pass  # auto-commit is observability; a git error must never fail the run


def _metrics_text(metrics_url: str) -> str:
    """Fetch a Prometheus metrics snapshot from the vLLM container."""
    import urllib.request
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"# metrics fetch failed: {type(exc).__name__}: {exc}\n"


def _sidecar_fwd_gpu_totals() -> dict | None:
    """FR13_SFWD_GPU_TIMER: cumulative pure-decode-forward GPU stats from the worker
    timer's JSON sidecar(s): {seconds, steps, drafts}. ALL THREE are restricted to
    the SAME pure-decode steps the timer measured (prefill/mixed/idle EXCLUDED), so
    seconds/steps (per-forward) and seconds/drafts (per-spec-event) are prefill-
    INDEPENDENT. CRITICAL: do NOT divide seconds by the GLOBAL
    spec_decode_num_drafts_total -- that counts drafts on mixed prefill+decode steps
    the timer excludes, reintroducing a prefill-load-dependent confound (measured:
    ~49% of global drafts land on non-timed steps at B=4 deployment). The worker
    writes the sidecar per-pid to FR13_SFWD_GPU_TIMER_JSON; in single-API-server mode
    the worker Counter is NOT aggregated into the API-server /metrics, so this sidecar
    is the robust channel. Returns None when the timer is off / no sidecar. Sums
    across per-pid files (one per worker)."""
    base = os.environ.get("FR13_SFWD_GPU_TIMER_JSON")
    if not base:
        return None
    # the boot env sets a /workspace path (docker -v "$REPO:/workspace"); on the
    # host that is REPO_ROOT. Translate so this host-side reader finds the file.
    if base.startswith("/workspace/"):
        base = str(REPO_ROOT / base[len("/workspace/"):])
    import glob as _glob
    pat = base.replace("{pid}", "*") if "{pid}" in base else base + ".*"
    files = set(_glob.glob(pat))
    if os.path.exists(base):
        files.add(base)
    secs = steps = drafts = 0.0
    found = False
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            secs += float(d.get("decode_forward_gpu_seconds", 0.0))
            steps += float(d.get("n_pure_decode_steps_timed", 0.0))
            drafts += float(d.get("n_drafts_in_timed_steps", 0.0))
            found = True
        except Exception:  # noqa: BLE001
            continue
    return {"seconds": secs, "steps": steps, "drafts": drafts} if found else None


def _sidecar_span_totals(env_name: str) -> dict | None:
    """FR13_DFWD/CFWD_GPU_TIMER: cumulative component GPU-span stats from a
    _Fr13SpanTimer JSON sidecar (schema fr13.span_gpu_timer.v1): {seconds,
    spans}. drafter = propose_draft_token_ids (all D spine forwards);
    committer = the spec-decode rejection-sampler dispatch in _sample. Same
    per-pid file layout + /workspace->host translation as the sfwd sidecar
    (_sidecar_fwd_gpu_totals); sums across per-pid files. Returns None when
    the timer is off / no sidecar."""
    base = os.environ.get(env_name)
    if not base:
        return None
    if base.startswith("/workspace/"):
        base = str(REPO_ROOT / base[len("/workspace/"):])
    import glob as _glob
    pat = base.replace("{pid}", "*") if "{pid}" in base else base + ".*"
    files = set(_glob.glob(pat))
    if os.path.exists(base):
        files.add(base)
    secs = spans = 0.0
    found = False
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            secs += float(d.get("gpu_seconds", 0.0))
            spans += float(d.get("n_spans", 0.0))
            found = True
        except Exception:  # noqa: BLE001
            continue
    return {"seconds": secs, "spans": spans} if found else None


def _metrics_snapshot(metrics_url: str) -> str:
    """/metrics text, plus (FR13_SFWD_GPU_TIMER) synthetic counter lines carrying the
    worker timer's cumulative pure-decode-forward GPU seconds AND its MATCHED
    denominators (pure-decode forward count + drafts-on-those-steps). The matched
    denominators are essential for prefill-independence: dividing the pure-decode-only
    GPU seconds by the GLOBAL spec_decode_num_drafts_total (which counts mixed-step
    drafts) reintroduces the prefill confound; dividing by the matched count (both
    pure-decode-only) does not. In single-API-server mode /metrics does not expose
    these worker counters, so we synthesize them here for the reducer's per-task
    bracket path.

    Double-count guard: if /metrics ALREADY exposes the seconds counter (multiprocess
    aggregation active), use that and do NOT append. No-op / byte-identical when the
    timer is off / no sidecar."""
    text = _metrics_text(metrics_url)
    if "fr13_decode_forward_gpu_seconds_total" not in text:
        t = _sidecar_fwd_gpu_totals()
        if t is not None:
            if not text.endswith("\n"):
                text += "\n"
            text += f"vllm:fr13_decode_forward_gpu_seconds_total {t['seconds']:.9f}\n"
            text += f"vllm:fr13_decode_forward_gpu_steps_total {t['steps']:.1f}\n"
            text += f"vllm:fr13_decode_forward_gpu_drafts_total {t['drafts']:.1f}\n"
    # FR13_DFWD/CFWD_GPU_TIMER: the SAME synthetic-line route for the drafter /
    # committer span timers (their prometheus Counters are also worker-process-
    # local in single-API-server mode; the sidecar is the robust channel). The
    # spans counter exists ONLY as a synthetic line (the worker Counter carries
    # seconds alone), so a multiprocess-aggregated /metrics run reports seconds
    # without spans -> the reducer's ms-per-step is null there. No-op /
    # byte-identical when a timer is off / no sidecar.
    for _mbase, _env in (
        ("fr13_drafter_gpu", "FR13_DFWD_GPU_TIMER_JSON"),
        ("fr13_committer_gpu", "FR13_CFWD_GPU_TIMER_JSON"),
    ):
        if f"{_mbase}_seconds_total" in text:
            continue  # multiprocess /metrics already exposes the counter
        st = _sidecar_span_totals(_env)
        if st is None:
            continue
        if not text.endswith("\n"):
            text += "\n"
        text += f"vllm:{_mbase}_seconds_total {st['seconds']:.9f}\n"
        text += f"vllm:{_mbase}_spans_total {st['spans']:.1f}\n"
    return text


def _spawn_dcgm_sampler(out_path: Path, interval_s: float = DEFAULT_DCGM_INTERVAL_S
                       ) -> subprocess.Popen[bytes] | None:
    """Spawn the same DCGM/NVML sampler Track B uses, writing JSONL to out_path."""
    if not DEFAULT_DCGM_SAMPLER.is_file():
        return None
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    interp = str(venv_python) if venv_python.is_file() else sys.executable
    cmd = [
        interp, str(DEFAULT_DCGM_SAMPLER),
        "--out", str(out_path),
        "--interval-s", str(interval_s),
        "--allow-unstamped-smoke",
    ]
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001
        return None


def _stop_dcgm_sampler(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    try:
        proc.send_signal(2)  # SIGINT — sampler writes a clean tail on signal
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _load_subset(subset_json: Path) -> tuple[str, list[str]]:
    payload = json.loads(subset_json.read_text())
    dataset_name = payload["dataset_name"]
    instance_ids = list(payload["instance_ids"])
    return dataset_name, instance_ids


def _load_dataset(dataset_name: str) -> dict[str, dict]:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    out: dict[str, dict] = {}
    for ex in ds:
        out[ex["instance_id"]] = dict(ex)
    return out


def _repo_clone_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _ensure_repo_cache(repo: str, cache_root: Path) -> Path:
    safe = repo.replace("/", "__")
    cache_path = cache_root / safe
    if not cache_path.is_dir():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _repo_clone_url(repo), str(cache_path)],
            check=True,
        )
    return cache_path


def _fetch_commit(cache_path: Path, base_commit: str) -> None:
    rc = subprocess.run(
        ["git", "-C", str(cache_path), "cat-file", "-e", base_commit],
    ).returncode
    if rc != 0:
        subprocess.run(
            ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
            check=False,
        )


def _hydrate_workspace(
    *,
    cache_path: Path,
    base_commit: str,
    workspace_path: Path,
) -> None:
    if _swe_agent_env() == "instance_image":
        # No source worktree: /testbed lives inside the per-instance eval image.
        # The workspace dir is only a host-side staging area (AGENTS.md + the
        # /out mount that receives the extracted patch).
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        workspace_path.mkdir(parents=True, exist_ok=True)
        return
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    # Absolutize: `git -C <cache> worktree add <relpath>` resolves the
    # destination against <cache>, not against the script CWD.
    abs_workspace = workspace_path.resolve()
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "add", "--detach",
         str(abs_workspace), base_commit],
        check=True,
    )


def _remove_workspace(cache_path: Path, workspace_path: Path) -> None:
    if _swe_agent_env() == "instance_image":
        # instance_image mode never registered a git worktree; just drop the
        # host staging dir (cache_path is None in this mode).
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)
        return
    abs_workspace = workspace_path.resolve() if workspace_path.exists() else workspace_path
    if not abs_workspace.exists():
        return
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "remove", "--force",
         str(abs_workspace)],
        check=False,
    )
    if abs_workspace.exists():
        shutil.rmtree(abs_workspace, ignore_errors=True)


def _write_agents_md(workspace: Path, instance: dict) -> None:
    body = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    # Bundle B #7: model_reasoning_effort="high" is inert on Qwen 3.6 in this
    # stack (Harmony-path only), so steer reasoning depth via the operator
    # prompt instead. Also pre-empts the common failure modes: don't burn the
    # budget on env setup (the grader builds its own env), and always emit a
    # real source edit before finishing.
    body.append("## How to work (important)")
    body.append("")
    if _swe_agent_env() == "instance_image":
        # instance_image mode: the agent runs INSIDE the prepared testbed env
        # (conda 'testbed' with the project editable-installed, on PATH), so the
        # "do NOT install/build" band-aid is replaced with a "reproduce + verify"
        # instruction — the whole point of §58 is that self-verification is now
        # possible.
        body.append(
            "- Reason carefully and thoroughly before each tool call. First inspect "
            "the relevant source files to confirm your understanding of the bug, "
            "then make the minimal correct edit.\n"
            "- You have a WORKING prepared environment: the project is already "
            "installed and importable (run `python -c \"import <project>\"` to "
            "confirm). REPRODUCE the bug with a short `python`/`pytest` command, "
            "make your fix, then RE-RUN it to verify the fix before finishing. You "
            "do NOT need to (re)install or build the project.\n"
            "- You MUST finish by leaving an actual code change in the working tree. "
            "Do not stop until you have edited the source files to implement the fix."
        )
    else:
        body.append(
            "- Reason carefully and thoroughly before each tool call. First inspect "
            "the relevant source files to confirm your understanding of the bug, "
            "then make the minimal correct edit.\n"
            "- Do NOT spend your time trying to `pip install` or build/conda the "
            "project — the grader runs in its own prepared environment. If an "
            "install/build command fails, do not retry it; just edit the source.\n"
            "- You MUST finish by leaving an actual code change in the working tree. "
            "Do not stop until you have edited the source files to implement the fix."
        )
    body.append("")
    (workspace / "AGENTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _classify_empty_patch_cause(trace_path: Path) -> str:
    """Inspect a codex trace to decide why an empty-patch run produced no edit.

    Returns 'setup_loop' if the agent repeatedly ran the same failing command
    (>=3 identical failing command_execution items — the pip/conda/build loop),
    else 'agent_gave_up'. Drives the Bundle B #2/#3/#8 state-conditional retry.
    """
    try:
        from collections import Counter
        failing: Counter = Counter()
        for line in trace_path.read_text(errors="replace").splitlines():
            if '"type":"command_execution"' not in line:
                continue
            try:
                item = json.loads(line).get("item", {})
            except Exception:  # noqa: BLE001
                continue
            if item.get("type") != "command_execution":
                continue
            ec = item.get("exit_code")
            cmd = item.get("command")
            if isinstance(cmd, str) and ec not in (0, None):
                failing[cmd.strip()] += 1
        if failing and max(failing.values()) >= 3:
            return "setup_loop"
    except Exception:  # noqa: BLE001
        pass
    return "agent_gave_up"


def _extract_patch(cache_path: Path, workspace: Path, base_commit: str) -> str:
    if _swe_agent_env() == "instance_image":
        # The patch was produced by `git -C /testbed diff --no-color --binary
        # <base_commit>` INSIDE the eval image (same flags as below) and copied to
        # <workspace>/patch.diff by the agent run. Read it back here.
        pf = Path(workspace) / "patch.diff"
        return pf.read_text(encoding="utf-8", errors="replace") if pf.is_file() else ""
    # Stage tracked-file diffs and untracked file additions against base_commit.
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color", "--binary", base_commit],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


def _run_agent_local(
    *,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    base_commit: str | None = None,
    prompt: str = DEFAULT_AGENT_PROMPT,
) -> dict[str, Any]:
    """Run codex-runner:v1 and capture trace/stdout/stderr to separate files.

    Track B layout (matching launch_qwen36_ablation_point.py):
      - trace_path : the stream-json NDJSON event stream from the qwen-code agent
      - stdout_path: qwen_stdout.log (kept for parity; usually empty when
        stream-json is the agent output mode)
      - stderr_path: qwen_stderr.log (qwen-code CLI stderr noise)
    """
    if _swe_agent_env() == "instance_image":
        return _run_agent_instance(
            remote_host=None, workspace=workspace, endpoint=endpoint, model=model,
            timeout_s=timeout_s, instance_id=instance_id, base_commit=base_commit,
            stdout_path=stdout_path, stderr_path=stderr_path, trace_path=trace_path,
            prompt=prompt,
        )
    container_name = f"swe-agent-{instance_id.replace('/', '_')[:48]}-{int(time.time())}"
    cmd = _agent_template().format(
        container_name=container_name,
        workspace=str(workspace),
        endpoint=endpoint,
        model=model,
    )
    # Pass the prompt as a separate argv element instead of shell-embedding it.
    # The dynamic agent_gave_up retry prompt is built from the prior-attempt trace
    # (arbitrary code/quotes/braces); inlining it as "\"{prompt}\"" + shlex.split
    # raised ValueError("No closing quotation") on an unbalanced quote and crashed
    # the orchestrator (verdict=orchestrator_crash -> NORESULT). A list argv passes
    # the prompt verbatim with no shell parsing, robust to any content.
    cmd_argv = shlex.split(cmd)
    cmd_argv.append(prompt)
    started = time.monotonic()
    rc: int | None = None
    timed_out = False
    # Codex emits its --json event stream to stdout and any human-readable
    # messages / errors to stderr. Track B keeps them in separate files.
    with trace_path.open("w", encoding="utf-8") as trace_f, \
         stderr_path.open("w", encoding="utf-8") as stderr_f:
        try:
            completed = subprocess.run(
                cmd_argv,
                stdout=trace_f,
                stderr=stderr_f,
                # timeout_s<=0 => NO harness wall (codex runs to its own idle/turn limit).
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)),
                check=False,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(
                ["docker", "kill", container_name], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                subprocess.run(
                    ["docker", "wait", container_name], check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                pass
            rc = -1
    elapsed = time.monotonic() - started
    # stdout_path kept for layout parity with Track B; the agent doesn't
    # write to it under `codex exec --json`.
    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    return {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
    }


# Eval-offload config. When EVAL_HOST is set (via --eval-host), the eval
# step runs on a native x86_64 box over SSH instead of locally on aarch64
# (recovers old-python instances that can't build the conda env on arm64,
# and keeps the whole dataset single-arch). See scripts/swe_eval_offload.py.
EVAL_HOST: str | None = None
_EVAL_SSH_OPTS = [
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
    "-o", "StrictHostKeyChecking=accept-new",
]

# --- FR13 CODEX-OFFLOAD config (the unified-memory contamination fix) ---------
# When AGENT_HOST is set (via --codex-host), the codex-runner Docker runs on a
# native x86 box (alienware) instead of locally on the GB10. The workspace is
# rsynced there before the run and the (modified) workspace is rsynced back
# after, so patch extraction + eval stay on the GB10 unchanged. This leaves the
# GB10 running ONLY vLLM during the codex agent loop, so the unified-memory
# bandwidth (273 GB/s, shared Grace CPU + Blackwell GPU) is uncontended and the
# timing-sensitive deploy-speed numbers (s/fwd) are clean. The lossless numbers
# are timing-independent so unaffected; only WHERE the codex compute runs moves.
# The codex docker on alienware hits the alienware-LOCAL proxy (AGENT_ENDPOINT,
# default 127.0.0.1:8023 — 8022 is occupied on alienware by an unrelated host
# service). Reuses the eval-offload SSH/rsync plumbing (_net_retry below).
AGENT_HOST: str | None = None
# alienware-local proxy the offloaded codex docker hits (8022 is taken on
# alienware by an unrelated host service; the offload proxy listens on 8023).
AGENT_ENDPOINT: str | None = None
_AGENT_REMOTE_ROOT = "~/lumo_proxy_offload/codex_work"
_AGENT_NET_LOG = DEFAULT_OUT_ROOT / "swe_codex_offload_network.log"
_REMOTE_BASE = "~/swe_eval_offload"
_REMOTE_WORKER = "~/swe_eval_offload/swe_eval_x86_worker.py"
_REMOTE_VENV_PY = "~/swe_eval_offload/venv/bin/python"
_REMOTE_HF_HOME = "~/.cache/huggingface"
_EVAL_NET_LOG = DEFAULT_OUT_ROOT / "swe_eval_offload_network.log"


def _eval_net_log(msg: str) -> None:
    try:
        _EVAL_NET_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _EVAL_NET_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _net_retry(argv: list[str], *, what: str, timeout: int,
               max_attempts: int = 5, ok_rcs: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    backoff = 5
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            if cp.returncode in ok_rcs:
                if attempt > 1:
                    _eval_net_log(f"RECOVERED {what} on attempt {attempt}")
                return cp
            last = cp
            if cp.returncode == 255:
                _eval_net_log(f"FAIL {what} attempt={attempt}/{max_attempts} rc=255(transport)")
            else:
                _eval_net_log(f"NONRETRY {what} rc={cp.returncode} (remote ran)")
                return cp
        except subprocess.TimeoutExpired:
            _eval_net_log(f"TIMEOUT {what} attempt={attempt}/{max_attempts} after {timeout}s")
            last = subprocess.CompletedProcess(argv, 124, "", "timeout")
        if attempt < max_attempts:
            time.sleep(backoff)
            backoff = min(backoff * 3, 300)
    _eval_net_log(f"GIVEUP {what} after {max_attempts} attempts")
    return last if last is not None else subprocess.CompletedProcess(argv, 1, "", "unknown")


def _run_eval_remote(
    *, host: str, instance_id: str, patch_path: Path, output_dir: Path,
    dataset_name: str, model_name: str, timeout_s: int, eval_log_path: Path,
) -> dict[str, Any]:
    """Offload one eval to a native x86_64 host over SSH (network-tolerant)."""
    started = time.monotonic()
    remote_dir = f"{_REMOTE_BASE}/work/{instance_id}"
    mk = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"mkdir -p {remote_dir} && echo ok"],
                    what=f"mkdir:{instance_id}", timeout=30)
    if mk.returncode != 0:
        eval_log_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "offloaded": True}
    up = _net_retry(["scp", *_EVAL_SSH_OPTS, str(patch_path), f"{host}:{remote_dir}/patch.diff"],
                    what=f"scp_up:{instance_id}", timeout=120)
    if up.returncode != 0:
        eval_log_path.write_text(f"scp patch up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        return {"exit_code": -1, "elapsed_s": round(time.monotonic() - started, 3), "offloaded": True}
    remote_cmd = (
        f"cd {_REMOTE_BASE} && HF_HOME={_REMOTE_HF_HOME} {_REMOTE_VENV_PY} {_REMOTE_WORKER} "
        f"--instance-id {instance_id} --patch-path {remote_dir}/patch.diff "
        f"--output-dir {remote_dir}/out --dataset-name '{dataset_name}' "
        f"--model-name '{model_name}' --timeout-s {timeout_s} --cache-level env"
    )
    ev = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, remote_cmd],
                    what=f"eval:{instance_id}", timeout=timeout_s + 900,
                    max_attempts=3, ok_rcs=(0, 1, 2))
    output_dir.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("w", encoding="utf-8") as f:
        f.write(f"[offload host={host} worker_rc={ev.returncode}]\n")
        f.write(ev.stdout or "")
        f.write("\n-- stderr --\n")
        f.write(ev.stderr or "")
    # fetch artifacts
    for fname, attempts in (("eval_report.json", 4), ("normalized_eval.json", 3),
                            ("eval.log", 3), ("predictions.jsonl", 2)):
        _net_retry(["scp", *_EVAL_SSH_OPTS, f"{host}:{remote_dir}/out/{fname}", str(output_dir / fname)],
                   what=f"scp_{fname}:{instance_id}", timeout=120, max_attempts=attempts)
    _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"rm -rf {remote_dir}"],
               what=f"cleanup:{instance_id}", timeout=30, max_attempts=2)
    return {"exit_code": ev.returncode, "elapsed_s": round(time.monotonic() - started, 3),
            "offloaded": True, "eval_host": host}


# ---- FR13 CODEX-OFFLOAD: run the codex-runner Docker on alienware -----------
# rsync robustness (req #3): a dropped workspace sync must NOT lose the run.
# --partial keeps a half-transferred file for the next attempt to resume;
# --append-verify resumes + re-checksums; the _net_retry loop reconnects.
_RSYNC_RESILIENT = [
    "rsync", "-az", "--partial", "--append-verify", "--timeout=120",
    "-e", "ssh " + " ".join(_EVAL_SSH_OPTS),
]


def _agent_net_log(msg: str) -> None:
    try:
        _AGENT_NET_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AGENT_NET_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{_iso_now()}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


# --- FR13 §59 runner stall-watchdog (default OFF) ----------------------------
# Diagnosis: a per-request GB10 emit/transport wedge froze the offloaded qwen-code
# agent mid-stream. With SWE_AGENT_WALL_S=0 the harness imposes NO wall, so a
# pre-first-byte wedge (Mode B, the client idle guard never arms) hung one banked
# run for 37 min until an external watchdog. This bounds BOTH modes by KILLING the
# ssh (and best-effort the remote container) when the agent's trace stops growing,
# keying on TRACE GROWTH (not wallclock) so a legitimately slow-but-progressing
# task survives. It KILLS + CLASSIFIES (cause=infra_stall_suspect); it NEVER
# retries or discards — that would risk the firm nudge ban.
def _stall_kill_s() -> float:
    """LUMO_SWE_STALL_KILL_S seconds; 0/unset/invalid => 0.0 (watchdog OFF)."""
    raw = os.environ.get("LUMO_SWE_STALL_KILL_S", "").strip()
    if not raw:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return val if val > 0 else 0.0


def _kill_proc(proc: Any) -> None:
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 - already-dead proc, etc.
        pass


def _monitor_proc_with_stall_watchdog(
    proc: Any,
    *,
    trace_path: Path,
    wall_timeout_s: float | None,
    stall_kill_s: float,
    poll_s: float = 5.0,
    on_stall_kill: Any = None,
) -> dict[str, Any]:
    """Block until ``proc`` exits, enforcing an optional wall timeout AND (when
    stall_kill_s>0) a trace-growth stall-watchdog.

    Returns metadata: {returncode, timed_out, stall_killed, last_trace_growth_ts}.
    On a stall (no ``trace_path`` byte-growth for stall_kill_s) the local proc is
    killed AFTER invoking on_stall_kill() (best-effort remote cleanup). On a wall
    timeout the local proc is killed (caller handles remote cleanup, mirroring the
    legacy path). NEVER retries. stall_kill_s<=0 disables the stall-watchdog, so
    the OFF behavior matches the legacy blocking subprocess.run(timeout=wall)."""
    start = time.monotonic()
    last_size = -1
    last_growth_mono = start
    last_growth_ts = time.time()
    timed_out = False
    stall_killed = False
    rc: int | None = None
    while True:
        try:
            rc = proc.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        try:
            size = trace_path.stat().st_size
        except OSError:
            size = last_size
        if size > last_size:
            last_size = size
            last_growth_mono = now
            last_growth_ts = time.time()
        if wall_timeout_s and wall_timeout_s > 0 and (now - start) >= wall_timeout_s:
            timed_out = True
            _kill_proc(proc)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = -1
            break
        if stall_kill_s and stall_kill_s > 0 and (now - last_growth_mono) >= stall_kill_s:
            stall_killed = True
            if on_stall_kill is not None:
                try:
                    on_stall_kill()
                except Exception:  # noqa: BLE001 - best-effort remote cleanup
                    pass
            _kill_proc(proc)
            try:
                rc = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rc = -1
            break
    return {
        "returncode": rc if rc is not None else -1,
        "timed_out": timed_out,
        "stall_killed": stall_killed,
        "last_trace_growth_ts": last_growth_ts,
    }


def _run_agent_remote(
    *,
    host: str,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    base_commit: str | None = None,
    prompt: str = DEFAULT_AGENT_PROMPT,
) -> dict[str, Any]:
    """Run codex-runner:v1 ON alienware (x86) over SSH, keeping the GB10 vLLM-only.

    Workspace lifecycle: rsync the GB10 worktree -> alienware, run the codex
    docker there (against the alienware-LOCAL proxy at `endpoint`), rsync the
    modified workspace back. Patch extraction + eval stay on the GB10 unchanged.
    Network-robust: resilient rsync (req #3) + a generous SSH timeout for the
    codex run; an SSH transport failure (rc 255) is CLASSIFIED network-drop and
    flagged (req #4) so the harness never reads a wire-stalled trajectory as a
    real run.
    """
    if _swe_agent_env() == "instance_image":
        return _run_agent_instance(
            remote_host=host, workspace=workspace, endpoint=endpoint, model=model,
            timeout_s=timeout_s, instance_id=instance_id, base_commit=base_commit,
            stdout_path=stdout_path, stderr_path=stderr_path, trace_path=trace_path,
            prompt=prompt,
        )
    started = time.monotonic()
    safe_id = instance_id.replace("/", "_")[:48]
    remote_ws = f"{_AGENT_REMOTE_ROOT}/{safe_id}/workspace"
    container_name = f"swe-agent-{safe_id}-{int(time.time())}"
    net_drop = False

    mk = _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"mkdir -p {remote_ws} && echo ok"],
                    what=f"codex_mkdir:{instance_id}", timeout=30)
    if mk.returncode != 0:
        net_drop = mk.returncode == 255
        stderr_path.write_text(f"remote mkdir failed rc={mk.returncode}\n{mk.stderr}", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return {"elapsed_s": round(time.monotonic() - started, 3), "exit_code": -1,
                "timed_out": False, "container_name": container_name,
                "offloaded": True, "codex_host": host, "network_drop": net_drop}

    # push the hydrated worktree (trailing slash = contents into remote_ws)
    up = _net_retry([*_RSYNC_RESILIENT, f"{str(workspace).rstrip('/')}/", f"{host}:{remote_ws}/"],
                    what=f"codex_ws_up:{instance_id}", timeout=600, max_attempts=5)
    if up.returncode != 0:
        net_drop = True
        _agent_net_log(f"WS_UP_FAIL {instance_id} rc={up.returncode} (network-drop)")
        stderr_path.write_text(f"workspace rsync up failed rc={up.returncode}\n{up.stderr}", encoding="utf-8")
        trace_path.write_text("", encoding="utf-8")
        return {"elapsed_s": round(time.monotonic() - started, 3), "exit_code": -1,
                "timed_out": False, "container_name": container_name,
                "offloaded": True, "codex_host": host, "network_drop": net_drop}

    # the codex docker on alienware mounts the remote workspace + hits the
    # alienware-local proxy endpoint. ~ expands on the remote login shell.
    remote_cmd = _agent_template().format(
        container_name=container_name,
        workspace=remote_ws,
        endpoint=endpoint,
        model=model,
    )
    # BUGFIX: CODEX_TEMPLATE no longer carries a {prompt} placeholder — the local path
    # appends the prompt as a separate argv element (see L459-466). The remote path used
    # to pass prompt=prompt into .format(), which silently DROPPED it (str.format ignores
    # unused kwargs) -> the remote codex launched with NO prompt -> "No prompt provided via
    # stdin" -> empty patch / agent_gave_up in ~3s (every offload SWE episode). Append the
    # prompt as a shell-quoted positional arg so it survives the ssh remote login shell and
    # reaches `codex exec` as its positional prompt, mirroring the local argv append.
    remote_cmd = remote_cmd + " " + shlex.quote(prompt)
    timed_out = False
    rc: int | None = None
    # run codex over SSH; trace (codex --json) -> trace_path, ssh stderr -> stderr_path.
    # SSH timeout = codex wall + a teardown buffer; a true codex timeout is
    # handled by killing the remote container (mirrors the local path).
    ssh_codex = ["ssh", *_EVAL_SSH_OPTS, "-o", f"ConnectTimeout=20", host, remote_cmd]
    # FR13 §59: Popen + trace-growth stall-watchdog (env LUMO_SWE_STALL_KILL_S,
    # default 0=OFF). Wall timeout is enforced inside the monitor so OFF behavior
    # matches the legacy subprocess.run(timeout=wall). timeout_s<=0 => no wall.
    stall_kill_s = _stall_kill_s()
    wall_timeout_s = None if timeout_s <= 0 else max(timeout_s, 30) + 120
    stall_killed = False
    last_trace_growth_ts: float | None = None

    def _remote_stall_kill() -> None:
        # best-effort kill the remote container so a stalled run doesn't linger
        _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"docker kill {container_name} 2>/dev/null; "
                    f"docker wait {container_name} 2>/dev/null; echo killed"],
                   what=f"codex_stall_kill:{instance_id}", timeout=60, max_attempts=2)

    with trace_path.open("w", encoding="utf-8") as tf, \
         stderr_path.open("w", encoding="utf-8") as ef:
        proc = subprocess.Popen(ssh_codex, stdout=tf, stderr=ef)
        mon = _monitor_proc_with_stall_watchdog(
            proc, trace_path=trace_path, wall_timeout_s=wall_timeout_s,
            stall_kill_s=stall_kill_s, poll_s=5.0, on_stall_kill=_remote_stall_kill,
        )
    rc = mon["returncode"]
    timed_out = mon["timed_out"]
    stall_killed = mon["stall_killed"]
    last_trace_growth_ts = mon["last_trace_growth_ts"]
    if stall_killed:
        # infra_stall_suspect: the agent trace stopped growing (a suspected GB10
        # emit/transport wedge). Killed + CLASSIFIED; NEVER retried (nudge ban).
        _agent_net_log(f"CODEX_STALL_KILL {instance_id} no trace growth for {stall_kill_s}s "
                       "(infra_stall_suspect; NOT agent_gave_up, NOT retried)")
        rc = -1
    elif timed_out:
        # best-effort kill the remote container so it doesn't linger on alienware
        _net_retry(["ssh", *_EVAL_SSH_OPTS, host, f"docker kill {container_name} 2>/dev/null; "
                    f"docker wait {container_name} 2>/dev/null; echo killed"],
                   what=f"codex_kill:{instance_id}", timeout=60, max_attempts=2)
        rc = -1
    elif rc == 255:
        # SSH transport died — CLASSIFY network-drop (req #4), not a model run.
        net_drop = True
        _agent_net_log(f"CODEX_SSH_TRANSPORT_DROP {instance_id} rc=255 (network-drop, not a fork)")
    elapsed = time.monotonic() - started

    # rsync the (modified) workspace back to the GB10 for patch extraction.
    # delete-after keeps the GB10 worktree authoritative for git diff.
    down = _net_retry([*_RSYNC_RESILIENT, f"{host}:{remote_ws}/", f"{str(workspace).rstrip('/')}/"],
                      what=f"codex_ws_down:{instance_id}", timeout=600, max_attempts=5)
    if down.returncode != 0:
        net_drop = True
        _agent_net_log(f"WS_DOWN_FAIL {instance_id} rc={down.returncode} (network-drop) — "
                       "patch may be incomplete; FLAGGED")
    # cleanup the remote workspace (non-fatal)
    _net_retry(["ssh", *_EVAL_SSH_OPTS, host,
                f"docker rm -f {container_name} 2>/dev/null; rm -rf {_AGENT_REMOTE_ROOT}/{safe_id}; echo ok"],
               what=f"codex_cleanup:{instance_id}", timeout=60, max_attempts=2)

    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    result: dict[str, Any] = {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
        "offloaded": True,
        "codex_host": host,
        "network_drop": net_drop,
        "ws_down_rc": down.returncode,
        "stall_killed": stall_killed,
    }
    if stall_killed:
        result["cause"] = "infra_stall_suspect"
        result["last_trace_growth_ts"] = last_trace_growth_ts
    return result


def _run_agent_instance(
    *,
    remote_host: str | None,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    base_commit: str | None,
    stdout_path: Path,
    stderr_path: Path,
    trace_path: Path,
    prompt: str = DEFAULT_AGENT_PROMPT,
) -> dict[str, Any]:
    """SWE_AGENT_ENV=instance_image (§58): run qwen-code INSIDE the SWE-bench
    per-instance eval image editing /testbed, with the node+qwen runtime injected
    read-only from the host bundle. remote_host=None -> local docker (GB10);
    else -> docker on the codex host (alienware) over SSH, mirroring the legacy
    offload plumbing (fail-loud image precondition, network-drop classification,
    generous SSH timeout). The wrapper extracts the patch to a /out bind-mount;
    we copy it back to <workspace>/patch.diff where _extract_patch reads it."""
    import base64 as _b64
    started = time.monotonic()
    safe_id = instance_id.replace("/", "_")[:48]
    container_name = f"swe-qwen-{safe_id}-{int(time.time())}"
    # ARCH PLACEMENT (user 2026-07-05): derive the image variant from the AGENT
    # host's architecture. Some instances only publish x86_64 images — those must
    # run on the x86 offload host; on an aarch64 host with no arm64 variant we
    # fail LOUD (never silently fall back to the bare-worktree env, class 9).
    host_machine = _host_arch(remote_host)
    sweb_arch = _SWEB_ARCH.get(host_machine)
    if sweb_arch is None:
        raise RuntimeError(
            f"SWE_AGENT_ENV=instance_image: unsupported agent-host arch "
            f"{host_machine!r} (host={remote_host or 'local'})")
    image = _instance_image_name(instance_id, arch=sweb_arch)
    _record_image_arch_tag(workspace, instance_id, {
        "image": image, "image_arch": sweb_arch, "agent_host": remote_host or "local",
        "host_machine": host_machine})
    if not base_commit:
        raise RuntimeError(
            "SWE_AGENT_ENV=instance_image requires base_commit (the agent's "
            "patch is `git diff <base_commit>` inside /testbed)")
    agents_md_path = Path(workspace) / "AGENTS.md"
    agents_md_text = (agents_md_path.read_text(encoding="utf-8")
                      if agents_md_path.is_file() else "")
    agents_b64 = _b64.b64encode(agents_md_text.encode("utf-8")).decode("ascii")
    # The shared operator/retry prompts reference `/workspace/AGENTS.md` (the
    # legacy qwen-code-runner mount). In instance_image mode AGENTS.md is written
    # to /testbed/AGENTS.md, the workdir is /testbed, and there is NO /workspace
    # mount — so remap the path here (confined to this mode; legacy never calls
    # this function) or the agent is told to read a file that does not exist.
    prompt = prompt.replace("/workspace/", "/testbed/")
    prompt_b64 = _b64.b64encode(prompt.encode("utf-8")).decode("ascii")
    net_drop = False
    timed_out = False
    rc: int | None = None
    patch_local = Path(workspace) / "patch.diff"

    if remote_host:
        # Fail loud (memory: fail-loud on missing infra): the per-instance eval
        # image MUST already be on the codex host — no silent fallback.
        insp = _net_retry(
            ["ssh", *_EVAL_SSH_OPTS, remote_host,
             f"docker image inspect {shlex.quote(image)} >/dev/null 2>&1 "
             f"&& echo present || echo absent"],
            what=f"img_inspect:{instance_id}", timeout=60, max_attempts=3)
        if "present" not in (insp.stdout or ""):
            raise RuntimeError(
                f"SWE_AGENT_ENV=instance_image: image {image} ABSENT on codex host "
                f"{remote_host} (host arch={host_machine}). Pull it / run the eval "
                f"first; if this instance publishes no {sweb_arch} variant it is "
                f"arch-restricted — run the agent on a host matching an available "
                f"arch (x86_64 => alienware offload). Refusing to fall back.")
        remote_out = f"{_AGENT_REMOTE_ROOT}/{safe_id}/out"
        mk = _net_retry(
            ["ssh", *_EVAL_SSH_OPTS, remote_host,
             f"mkdir -p {remote_out} && rm -f {remote_out}/patch.diff && echo ok"],
            what=f"qwen_out_mkdir:{instance_id}", timeout=30)
        if mk.returncode != 0:
            net_drop = mk.returncode == 255
            stderr_path.write_text(f"remote out mkdir failed rc={mk.returncode}\n{mk.stderr}",
                                   encoding="utf-8")
            trace_path.write_text("", encoding="utf-8")
            return {"elapsed_s": round(time.monotonic() - started, 3), "exit_code": -1,
                    "timed_out": False, "container_name": container_name, "offloaded": True,
                    "codex_host": remote_host, "network_drop": net_drop,
                    "agent_env": "instance_image", "instance_image": image,
                    "host_arch": host_machine, "image_arch": sweb_arch}
        # ~ expands on the remote login shell (the whole cmd is one ssh arg).
        cmd = _instance_agent_command(
            container_name=container_name, image=image, endpoint=endpoint, model=model,
            host_out_dir=remote_out, bundle_src="~/qwen_agent_bundle",
            agents_md_b64=agents_b64, prompt_b64=prompt_b64, base_commit=base_commit)
        ssh_cmd = ["ssh", *_EVAL_SSH_OPTS, "-o", "ConnectTimeout=20", remote_host, cmd]
        try:
            with trace_path.open("w", encoding="utf-8") as tf, \
                 stderr_path.open("w", encoding="utf-8") as ef:
                completed = subprocess.run(
                    ssh_cmd, stdout=tf, stderr=ef,
                    timeout=(None if timeout_s <= 0 else max(timeout_s, 30) + 120),
                    check=False)
            rc = completed.returncode
            if rc == 255:
                net_drop = True
                _agent_net_log(f"QWEN_SSH_TRANSPORT_DROP {instance_id} rc=255 (network-drop, not a fork)")
        except subprocess.TimeoutExpired:
            timed_out = True
            _net_retry(["ssh", *_EVAL_SSH_OPTS, remote_host,
                        f"docker kill {container_name} 2>/dev/null; "
                        f"docker wait {container_name} 2>/dev/null; echo killed"],
                       what=f"qwen_kill:{instance_id}", timeout=60, max_attempts=2)
            rc = -1
        elapsed = time.monotonic() - started
        # copy the extracted patch back (ok_rcs includes 1 = scp "no such file"
        # when the agent left no edit -> empty patch, handled downstream).
        pd = _net_retry(
            ["scp", *_EVAL_SSH_OPTS, f"{remote_host}:{remote_out}/patch.diff", str(patch_local)],
            what=f"qwen_patch_down:{instance_id}", timeout=120, max_attempts=4, ok_rcs=(0, 1))
        if pd.returncode != 0 and not patch_local.is_file():
            patch_local.write_text("", encoding="utf-8")
        _net_retry(["ssh", *_EVAL_SSH_OPTS, remote_host,
                    f"docker rm -f {container_name} 2>/dev/null; "
                    f"rm -rf {_AGENT_REMOTE_ROOT}/{safe_id}; echo ok"],
                   what=f"qwen_cleanup:{instance_id}", timeout=60, max_attempts=2)
        if not stdout_path.is_file():
            stdout_path.write_text("", encoding="utf-8")
        return {"elapsed_s": round(elapsed, 3), "exit_code": rc if rc is not None else -1,
                "timed_out": timed_out, "container_name": container_name, "offloaded": True,
                "codex_host": remote_host, "network_drop": net_drop,
                "agent_env": "instance_image", "instance_image": image,
                    "host_arch": host_machine, "image_arch": sweb_arch}

    # ---- local docker path (secondary; production uses --codex-host) ----
    insp = subprocess.run(["docker", "image", "inspect", image],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if insp.returncode != 0:
        raise RuntimeError(
            f"SWE_AGENT_ENV=instance_image: image {image} ABSENT locally "
            f"(host arch={host_machine}; x86_64-only instances must run offloaded "
            f"on alienware via --codex-host) "
            f"(pull it first; refusing to fall back)")
    cmd = _instance_agent_command(
        container_name=container_name, image=image, endpoint=endpoint, model=model,
        host_out_dir=str(Path(workspace).resolve()),
        bundle_src=os.path.expanduser("~/qwen_agent_bundle"),
        agents_md_b64=agents_b64, prompt_b64=prompt_b64, base_commit=base_commit)
    cmd_argv = shlex.split(cmd)
    try:
        with trace_path.open("w", encoding="utf-8") as tf, \
             stderr_path.open("w", encoding="utf-8") as ef:
            completed = subprocess.run(
                cmd_argv, stdout=tf, stderr=ef,
                timeout=(None if timeout_s <= 0 else max(timeout_s, 30)), check=False)
        rc = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["docker", "kill", container_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            subprocess.run(["docker", "wait", container_name], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except subprocess.TimeoutExpired:
            pass
        rc = -1
    elapsed = time.monotonic() - started
    if not patch_local.is_file():
        patch_local.write_text("", encoding="utf-8")
    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    return {"elapsed_s": round(elapsed, 3), "exit_code": rc if rc is not None else -1,
            "timed_out": timed_out, "container_name": container_name,
            "agent_env": "instance_image", "instance_image": image,
                    "host_arch": host_machine, "image_arch": sweb_arch}


def _run_agent_dispatch(**kwargs: Any) -> dict[str, Any]:
    """Route the codex run to alienware (AGENT_HOST set) or local (GB10).

    On the offload path the codex docker on alienware must hit the alienware-
    LOCAL proxy, so the GB10-side `--endpoint` is overridden with AGENT_ENDPOINT.
    """
    if AGENT_HOST:
        kwargs = dict(kwargs)
        if AGENT_ENDPOINT:
            kwargs["endpoint"] = AGENT_ENDPOINT
        return _run_agent_remote(host=AGENT_HOST, **kwargs)
    return _run_agent_local(**kwargs)


def _run_eval(
    *,
    instance_id: str,
    patch_path: Path,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    timeout_s: int,
    eval_log_path: Path,
) -> dict[str, Any]:
    if EVAL_HOST:
        return _run_eval_remote(
            host=EVAL_HOST, instance_id=instance_id, patch_path=patch_path,
            output_dir=output_dir, dataset_name=dataset_name, model_name=model_name,
            timeout_s=timeout_s, eval_log_path=eval_log_path,
        )
    cbe_exe = shutil.which("codex-bench-eval-swe")
    if cbe_exe is None:
        cbe_exe = str(REPO_ROOT / ".venv" / "bin" / "codex-bench-eval-swe")
    cmd = [
        cbe_exe,
        "--instance-id", instance_id,
        "--patch-path", str(patch_path),
        "--output-dir", str(output_dir),
        "--dataset-name", dataset_name,
        "--model-name", model_name,
        "--timeout-s", str(timeout_s),
        "--cache-level", "env",
    ]
    started = time.monotonic()
    with eval_log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    return {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 3),
    }


def _process_one(
    *,
    instance_id: str,
    instance: dict,
    dataset_name: str,
    per_task_root: Path,
    repo_cache_root: Path,
    endpoint: str,
    model: str,
    model_name: str,
    agent_wall_s: int,
    eval_timeout_s: int,
    skip_existing: bool,
) -> dict[str, Any]:
    # Use absolute paths everywhere so docker volume mounts and git
    # worktree add (which resolves relative to -C cache) both work.
    task_dir = (per_task_root / instance_id).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    qwen_stdout = task_dir / "qwen_stdout.log"
    qwen_stderr = task_dir / "qwen_stderr.log"
    qwen_trace = task_dir / "qwen_trace.jsonl"
    prompt_md = task_dir / "prompt.md"
    metrics_pre = task_dir / "vllm_metrics_pre.txt"
    metrics_post = task_dir / "vllm_metrics_post.txt"
    per_turn_json = task_dir / "vllm_per_turn.json"
    dcgm_samples = task_dir / "dcgm_samples.jsonl"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    started_iso = _iso_now()
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset_name": dataset_name,
        "started_at": started_iso,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
    }

    cache_path = None
    try:
        if _swe_agent_env() == "instance_image":
            # instance_image mode: /testbed lives inside the per-instance eval
            # image, so skip the repo clone/fetch/worktree entirely; the
            # workspace is only a host staging dir (AGENTS.md + /out patch sink).
            _hydrate_workspace(
                cache_path=None,
                base_commit=instance["base_commit"],
                workspace_path=workspace_path,
            )
        else:
            cache_path = _ensure_repo_cache(instance["repo"], repo_cache_root)
            _fetch_commit(cache_path, instance["base_commit"])
            _hydrate_workspace(
                cache_path=cache_path,
                base_commit=instance["base_commit"],
                workspace_path=workspace_path,
            )
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        if cache_path is not None:
            _remove_workspace(cache_path, workspace_path)
        return summary

    # Write prompt.md mirroring the Track B layout. The codex agent receives
    # both the docker-CLI prompt (the "Read the task prompt..." string) and
    # the AGENTS.md content inside /workspace.
    prompt_md.write_text(
        "## Codex CLI invocation prompt\n\n"
        '"Read the task prompt at /workspace/AGENTS.md and complete it in this '
        "workspace. Edit the source files directly to implement the fix. Do not "
        "write a diff file -- modify the files in place so that running pytest "
        'passes the tests described in the prompt."\n\n'
        f"## AGENTS.md (workspace/{instance_id})\n\n"
        + (workspace_path / "AGENTS.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Snapshot proxy capture byte offset and Prometheus metrics before Codex.
    proxy_capture = DEFAULT_PROXY_CAPTURE
    proxy_offset_before = (
        proxy_capture.stat().st_size if proxy_capture.is_file() else 0
    )
    metrics_pre.write_text(_metrics_snapshot(DEFAULT_METRICS_URL), encoding="utf-8")

    # Start the DCGM/NVML sampler in parallel with the Codex agent.
    dcgm_proc = _spawn_dcgm_sampler(dcgm_samples)

    codex_meta = _run_agent_dispatch(
        workspace=workspace_path,
        endpoint=endpoint,
        model=model,
        timeout_s=agent_wall_s,
        instance_id=instance_id,
        base_commit=instance["base_commit"],
        stdout_path=qwen_stdout,
        stderr_path=qwen_stderr,
        trace_path=qwen_trace,
    )
    summary["agent"] = codex_meta
    # Backward-compat: keep the legacy "codex" key so existing reducers
    # (meta.get("codex") in fr13_bigdenom_swe_serve.sh, fr13_standard_metrics.py)
    # still resolve. New reducers read meta.get("agent") or meta.get("codex").
    summary["codex"] = codex_meta

    _stop_dcgm_sampler(dcgm_proc)
    metrics_post.write_text(_metrics_snapshot(DEFAULT_METRICS_URL), encoding="utf-8")

    patch_text = ""
    try:
        patch_text = _extract_patch(cache_path, workspace_path, instance["base_commit"])
    except Exception as exc:  # noqa: BLE001
        summary["patch_extract_error"] = f"{type(exc).__name__}: {exc}"

    # Bundle B #2/#3/#8: if the first attempt left no patch, classify why and
    # re-launch codex ONCE with a state-conditional directive prompt. Bounded
    # to a single retry to cap the wall-time premium.
    if not patch_text.strip():
        cause = _classify_empty_patch_cause(qwen_trace)
        # Bundle B #2/#3/#8 -> in-context continuation: an agent_gave_up attempt EXPLORED
        # then emitted a tool-call-free terminal reply (general temp-0.6 codex flake). Re-drive
        # it up to SWE_EMPTY_PATCH_RETRIES times with its OWN accumulated context + a hard
        # must-act directive (a plain fresh re-roll re-confirms the stop ~88% of the time).
        # DEFAULT 0 = NUDGE-ONLY (project decision, user 2026-07-01): the new-context retry (a
        # FRESH codex session with only a brief of the prior attempt) is REMOVED by default. It
        # predates and is SUPERSEDED by the in-session nudge (LUMO_PROXY_AUTO_CONTINUE_MESSAGE),
        # which recovers empty-patch give-ups IN-SESSION with FULL context preserved — strictly
        # better, because the clean-context restart "empirically breaks the explain-instead-of-act
        # stall" (relaunch_proxy_remote.sh). Running both was redundant + the retry is ~3x slower on
        # hard tasks + loses context. Set SWE_EMPTY_PATCH_RETRIES>=1 ONLY to re-enable the legacy
        # fresh-session retry (small tail-resolve gain at a big wall-time cost).
        max_retries = max(0, int(os.environ.get("SWE_EMPTY_PATCH_RETRIES", "0")))
        summary["empty_patch_retry"] = {"cause": cause, "attempted": True,
                                        "max_retries": max_retries, "recovered_patch_bytes": 0}
        prev_trace = qwen_trace
        for ridx in range(1, max_retries + 1):
            if cause == "setup_loop":
                retry_prompt = RETRY_PROMPT_SETUP_LOOP
            else:
                retry_prompt = _retry_prompt_continue(_prior_attempt_brief(prev_trace))
            suffix = "" if ridx == 1 else str(ridx)
            print(f"[{_iso_now()}]    {instance_id}: empty patch ({cause}); retry {ridx}/{max_retries}",
                  flush=True)
            retry_trace = task_dir / f"qwen_trace_retry{suffix}.jsonl"
            retry_stderr = task_dir / f"qwen_stderr_retry{suffix}.log"
            retry_stdout = task_dir / f"qwen_stdout_retry{suffix}.log"
            retry_dcgm = _spawn_dcgm_sampler(task_dir / f"dcgm_samples_retry{suffix}.jsonl")
            try:
                codex_meta_retry = _run_agent_dispatch(
                    workspace=workspace_path, endpoint=endpoint, model=model,
                    timeout_s=agent_wall_s, instance_id=instance_id,
                    base_commit=instance["base_commit"],
                    stdout_path=retry_stdout, stderr_path=retry_stderr, trace_path=retry_trace,
                    prompt=retry_prompt,
                )
            except Exception as exc:  # noqa: BLE001 — a retry dispatch failure must never crash the task
                _stop_dcgm_sampler(retry_dcgm)
                summary[f"codex_retry{suffix}_dispatch_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{_iso_now()}]    {instance_id}: retry {ridx} dispatch failed "
                      f"({type(exc).__name__}: {exc}); keeping empty patch as failed", flush=True)
                break
            _stop_dcgm_sampler(retry_dcgm)
            summary[f"codex_retry{suffix}"] = codex_meta_retry
            try:
                retry_patch = _extract_patch(cache_path, workspace_path, instance["base_commit"])
            except Exception as exc:  # noqa: BLE001
                retry_patch = ""
                summary[f"patch_extract_error_retry{suffix}"] = f"{type(exc).__name__}: {exc}"
            prev_trace = retry_trace
            if retry_patch.strip():
                patch_text = retry_patch
                summary["empty_patch_retry"]["recovered_patch_bytes"] = len(retry_patch)
                summary["empty_patch_retry"]["attempts_used"] = ridx
                print(f"[{_iso_now()}]    {instance_id}: retry {ridx} produced {len(retry_patch)}B patch",
                      flush=True)
                break
        else:
            summary["empty_patch_retry"]["attempts_used"] = max_retries

    metrics_post.write_text(_metrics_snapshot(DEFAULT_METRICS_URL), encoding="utf-8")

    # Slice the new proxy rows into a per-task file (matches Track B layout).
    # This must happen after the optional empty-patch retry so retry traffic is
    # included in the task's request-metrics evidence.
    per_task_metrics = task_dir / "vllm_request_metrics.jsonl"
    proxy_row_count = 0
    try:
        if proxy_capture.is_file():
            with proxy_capture.open("rb") as src:
                src.seek(proxy_offset_before)
                payload = src.read()
            per_task_metrics.write_bytes(payload)
            summary["vllm_request_metrics_bytes"] = len(payload)
            proxy_row_count = payload.count(b"\n")
        else:
            per_task_metrics.write_bytes(b"")
            summary["vllm_request_metrics_bytes"] = 0
            summary["vllm_request_metrics_warning"] = (
                "proxy capture file not present; verbose request metrics unavailable"
            )
    except Exception as exc:  # noqa: BLE001
        summary["vllm_request_metrics_error"] = f"{type(exc).__name__}: {exc}"

    # Write a simple vllm_per_turn.json (delta of pre/post snapshot sizes
    # plus the per-task proxy row count). This is a smaller artifact than
    # Track B's full Prometheus per-request normalization but preserves the
    # filename + role in the artifact set.
    try:
        per_turn_json.write_text(
            json.dumps(
                {
                    "schema": "lumo.swe_bench_q36_a.vllm_per_turn.v1",
                    "instance_id": instance_id,
                    "metrics_pre_bytes": metrics_pre.stat().st_size,
                    "metrics_post_bytes": metrics_post.stat().st_size,
                    "proxy_request_rows": proxy_row_count,
                    "model_id": model,
                    "endpoint": endpoint,
                    "metrics_url": DEFAULT_METRICS_URL,
                    "deferred_full_normalization": True,
                    "deferred_reason": (
                        "SWE-Bench campaign captures raw pre/post snapshots and the "
                        "proxy capture slice; full per-request Prometheus normalization "
                        "deferred to LLD-12 / closeout aggregation."
                    ),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        summary["vllm_per_turn_error"] = f"{type(exc).__name__}: {exc}"

    patch_path.write_text(patch_text, encoding="utf-8")
    summary["patch_bytes"] = len(patch_text)

    # Always run the evaluator -- the CLI handles empty-patch -> exit 1.
    eval_meta = _run_eval(
        instance_id=instance_id,
        patch_path=patch_path,
        output_dir=eval_output,
        dataset_name=dataset_name,
        model_name=model_name,
        timeout_s=eval_timeout_s,
        eval_log_path=eval_log,
    )
    summary["eval"] = eval_meta

    eval_report_path = eval_output / "eval_report.json"
    if eval_report_path.is_file():
        try:
            summary["eval_report"] = json.loads(eval_report_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    # Tear down the worktree to free disk; preserve patch and artifacts.
    _remove_workspace(cache_path, workspace_path)

    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(per_task_root: Path, summary_path: Path, predictions_path: Path,
               started_at: str, ended_at: str, model_name: str) -> dict[str, Any]:
    instance_summaries: list[dict[str, Any]] = []
    verdict_counter: Counter = Counter()
    failure_mode_counter: Counter = Counter()
    repo_counter: Counter = Counter()
    repo_pass_counter: Counter = Counter()
    eval_wall: list[float] = []
    codex_wall: list[float] = []
    predictions_lines: list[str] = []
    for task_dir in sorted(p for p in per_task_root.iterdir() if p.is_dir()):
        meta_path = task_dir / "runner_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        instance_summaries.append(meta)
        verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
        failure_mode = (meta.get("eval_report") or {}).get("failure_mode", "missing")
        verdict_counter[verdict] += 1
        failure_mode_counter[failure_mode] += 1
        repo = meta.get("repo") or "unknown"
        repo_counter[repo] += 1
        if verdict == "resolved":
            repo_pass_counter[repo] += 1
        if (meta.get("eval_report") or {}).get("eval_wall_clock_seconds") is not None:
            eval_wall.append(float(meta["eval_report"]["eval_wall_clock_seconds"]))
        _agent_block = meta.get("agent") or meta.get("codex") or {}
        if _agent_block.get("elapsed_s") is not None:
            codex_wall.append(float(_agent_block["elapsed_s"]))
        pred_file = task_dir / "eval" / "predictions.jsonl"
        if pred_file.is_file():
            predictions_lines.extend(
                line for line in pred_file.read_text().splitlines() if line.strip()
            )

    def _percentiles(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        xs = sorted(xs)
        def _pct(p: float) -> float:
            i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[i], 3)
        return {"p50": _pct(0.5), "p90": _pct(0.9), "p99": _pct(0.99),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    summary = {
        "model_name_or_path": model_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "instances_total": len(instance_summaries),
        "verdict_counts": dict(verdict_counter),
        "failure_mode_counts": dict(failure_mode_counter),
        "per_repo_total": dict(repo_counter),
        "per_repo_resolved": dict(repo_pass_counter),
        "eval_wall_seconds": _percentiles(eval_wall),
        "codex_wall_seconds": _percentiles(codex_wall),
        "resolved_rate": (
            round(verdict_counter["resolved"] / len(instance_summaries), 4)
            if instance_summaries else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    predictions_path.write_text("\n".join(predictions_lines) + ("\n" if predictions_lines else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True,
                        help="JSON subset emitted by build_swe_bench_subset.py")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dataset-tag", default=None,
                        help="Override the per-dataset subdirectory name "
                             "(default: 'verified' or 'pro' inferred from subset).")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME_TAG)
    parser.add_argument("--agent-wall-s", type=int,
                        default=int(os.environ.get("SWE_AGENT_WALL_S", str(DEFAULT_AGENT_WALL_S))))
    parser.add_argument("--eval-timeout-s", type=int, default=DEFAULT_EVAL_TIMEOUT_S)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Agent concurrency. LLD-05 §4.6 default is 1; raise after Sprint-1 validation.")
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N instances (for smoke runs).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances whose runner_metadata.json is already on disk.")
    parser.add_argument("--eval-host", default=None,
                        help="SSH host to offload the eval step to (native x86_64). When set, "
                             "the agent runs locally on arm64 but eval runs on the x86 box — "
                             "recovers old-python instances + keeps the dataset single-arch.")
    parser.add_argument("--agent-host", "--codex-host", dest="agent_host", default=None,
                        help="SSH host to offload the SWE agent Docker to (native x86_64, "
                             "e.g. alienware). When set, the agent loop runs on the x86 box "
                             "(workspace rsynced there + back) so the GB10 runs ONLY vLLM and the "
                             "unified-memory bandwidth is uncontended = clean deploy-speed numbers. "
                             "--codex-host is a deprecated alias kept so queued serve scripts parse.")
    parser.add_argument("--agent-endpoint", "--codex-endpoint", dest="agent_endpoint", default=None,
                        help="The agent docker on --agent-host hits this (alienware-local) proxy "
                             "endpoint (default http://127.0.0.1:8023/v1). Distinct from --endpoint "
                             "(the on-GB10 legacy proxy) because 8022 is taken on alienware. "
                             "--codex-endpoint is a deprecated alias.")
    args = parser.parse_args(argv)

    global EVAL_HOST, AGENT_HOST, AGENT_ENDPOINT
    EVAL_HOST = args.eval_host
    if EVAL_HOST:
        print(f"[eval-offload] eval step -> {EVAL_HOST} (native x86_64)", flush=True)
    AGENT_HOST = args.agent_host
    AGENT_ENDPOINT = args.agent_endpoint or (
        "http://127.0.0.1:8023/v1" if AGENT_HOST else None)
    if AGENT_HOST:
        print(f"[codex-offload] codex docker -> {AGENT_HOST} (x86); endpoint={AGENT_ENDPOINT} "
              f"(GB10 stays vLLM-only — uncontended deploy-speed)", flush=True)

    dataset_name, instance_ids = _load_subset(args.subset)
    if args.limit is not None:
        instance_ids = instance_ids[: args.limit]
    dataset_tag = args.dataset_tag or ("pro" if "Pro" in dataset_name else "verified")
    dataset_out = args.out_root / dataset_tag
    per_task_root = dataset_out / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    print(f"=== [{_iso_now()}] dataset={dataset_name} tag={dataset_tag} n={len(instance_ids)} "
          f"concurrency={args.concurrency} ===", flush=True)
    dataset_records = _load_dataset(dataset_name)
    missing = [i for i in instance_ids if i not in dataset_records]
    if missing:
        print(f"WARNING: {len(missing)} subset instances missing from dataset: {missing[:5]}",
              flush=True)
        instance_ids = [i for i in instance_ids if i in dataset_records]

    args.repo_cache.mkdir(parents=True, exist_ok=True)

    # Startup-time worktree GC: prune any stale worktrees registered in
    # cached repos whose checkout directory either no longer exists OR
    # whose parent per_task/<id>/ directory lacks a runner_metadata.json
    # (signature of an aborted run). Without this, orchestrator restarts
    # accumulate stale worktrees that hold 50-300 MB of checkout files
    # each and never get cleaned, eating into the tight ~7-8 GiB host
    # MemAvailable budget on this unified-memory DGX Spark host.
    if args.repo_cache.is_dir():
        for repo_cache in sorted(args.repo_cache.iterdir()):
            if not repo_cache.is_dir():
                continue
            try:
                out = subprocess.run(
                    ["git", "-C", str(repo_cache), "worktree", "list", "--porcelain"],
                    capture_output=True, text=True, check=False, timeout=10,
                )
                if out.returncode != 0:
                    continue
            except Exception:  # noqa: BLE001
                continue
            for line in out.stdout.splitlines():
                if not line.startswith("worktree "):
                    continue
                wt_path = Path(line[len("worktree "):])
                if wt_path == repo_cache:
                    continue  # main checkout
                parent = wt_path.parent
                runner_meta = parent / "runner_metadata.json"
                if not wt_path.exists() or not runner_meta.is_file():
                    print(f"[startup-gc] removing stale worktree: {wt_path}", flush=True)
                    subprocess.run(
                        ["git", "-C", str(repo_cache), "worktree", "remove",
                         "--force", str(wt_path)],
                        capture_output=True, check=False, timeout=15,
                    )
                    if wt_path.exists():
                        shutil.rmtree(wt_path, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(repo_cache), "worktree", "prune"],
                capture_output=True, check=False, timeout=10,
            )

    started_at = _iso_now()
    summaries: list[dict[str, Any]] = []

    def _job(iid: str) -> dict[str, Any]:
        t0 = time.time()
        print(f"[{_iso_now()}] -> {iid}", flush=True)
        try:
            res = _process_one(
                instance_id=iid,
                instance=dataset_records[iid],
                dataset_name=dataset_name,
                per_task_root=per_task_root,
                repo_cache_root=args.repo_cache,
                endpoint=args.endpoint,
                model=args.model,
                model_name=args.model_name,
                agent_wall_s=args.agent_wall_s,
                eval_timeout_s=args.eval_timeout_s,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"instance_id": iid, "status": "orchestrator_crash",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
            # PERSIST the crash (2026-07-06): previously the traceback lived only
            # in this in-memory dict — an orchestrator_crash left NOTHING on disk
            # to diagnose (cost a wasted GPU boot to rediscover via reproduction).
            # Print to the run log AND write a per-task crash file.
            print(f"[{_iso_now()}] !! {iid} orchestrator_crash: {res['error']}\n"
                  f"{res['traceback']}", flush=True)
            try:
                crash_dir = per_task_root / iid  # same layout as _process_one task_dir
                crash_dir.mkdir(parents=True, exist_ok=True)
                (crash_dir / "orchestrator_crash.json").write_text(
                    json.dumps(res, indent=1), encoding="utf-8")
            except Exception:
                pass
        verdict = (res.get("eval_report") or {}).get("verdict", res.get("status", "?"))
        print(f"[{_iso_now()}] <- {iid} verdict={verdict} elapsed_total={time.time()-t0:.1f}s",
              flush=True)
        _autocommit_task_artifacts(per_task_root / iid, iid)
        return res

    if args.concurrency <= 1:
        for iid in instance_ids:
            summaries.append(_job(iid))
    else:
        with _cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for res in ex.map(_job, instance_ids):
                summaries.append(res)

    ended_at = _iso_now()
    summary = _aggregate(
        per_task_root=per_task_root,
        summary_path=dataset_out / "campaign_summary.json",
        predictions_path=dataset_out / "predictions.jsonl",
        started_at=started_at,
        ended_at=ended_at,
        model_name=args.model_name,
    )
    print(f"=== [{ended_at}] DONE n={summary['instances_total']} "
          f"resolved_rate={summary.get('resolved_rate')} "
          f"verdicts={summary['verdict_counts']} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
