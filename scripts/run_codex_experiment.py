#!/usr/bin/env python3
"""Drive a codex-on-x86 experiment (config x model x task-set x concurrency) and,
as EACH task finishes, automatically rsync -> join (Nsight GPU metrics) -> commit
-> push its artifacts incrementally.

Topology (see memory project-swe-bench-concurrency-probe): vLLM + proxy + Nsight
run on the DGX; codex + the suite orchestrator run on alienware over the reverse
tunnel; results are rsync'd back to this repo. This script runs on the DGX.

It does NOT switch vLLM config itself (that is a sudo/ModelServer op): set the
config first (e.g. /tmp/relaunch_qwen36_A.py) and pass --config to label the run.
Concurrency defaults to 1 (feedback-no-parallel-testing). Nsight is OFF by default
to keep decode-tps clean; --nsight first-task captures ONE representative window
(spec §13.7: occasional, not per-completion) with --discard-environment so the
.nsys-rep carries no secrets.

Example:
  python scripts/run_codex_experiment.py --exp-tag q36a_swe5 --suite swe \
    --config A --subset docs/.../subset.json --limit 5 --nsight first-task
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

REPO = Path("/home/mark/shared/lumoFlyWheel")
ALIEN = "alienware"
PROXY = os.environ.get("LUMO_CODEX_PROXY_MODELS_URL",
                       "http://127.0.0.1:8022/v1/models")
VLLM_METRICS = os.environ.get("LUMO_VLLM_METRICS_URL",
                              "http://127.0.0.1:9950/metrics")
NSYS = "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
STEPTRACE = os.environ.get("LUMO_SWE_DGX_STEPTRACE",
                           "/tmp/swe_dgx_steptrace.jsonl")
REQUEST_METRICS = os.environ.get(
    "LUMO_TRACK_B_REQUEST_METRICS_OUT",
    "/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl")
PER_REQ_SPEC_TRACE = os.environ.get(
    "LUMO_PER_REQ_SPEC_TRACE",
    "/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl")
TREE_ACCEPT_PATH_TRACE = os.environ.get(
    "LUMO_TREE_ACCEPT_PATH_LOG",
    "/tmp/lumo-l0c-fp8-cutlass-run30-logs/tree_accept_path.jsonl")
TREE_PATH_LCP_TRACE = os.environ.get(
    "LUMO_TREE_PATH_LCP_LOG",
    "/tmp/lumo-l0c-fp8-cutlass-run30-logs/tree_path_lcp_max.jsonl")
INDEPENDENT_WINNER_TRACE = os.environ.get(
    "LUMO_IR_WINNER_TRACE_FILE",
    "/tmp/lumo-l0c-fp8-cutlass-run30-logs/independent_winner_trace.jsonl")
FORBIDDEN_CODEX_PROTOCOL_MARKERS = ("<think>", "</think>", "<|host|>")
VLLM_RELAUNCH_TIMEOUT_S = int(os.environ.get("LUMO_VLLM_RELAUNCH_TIMEOUT_S", "3600"))


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return sh(["ssh", "-o", "ConnectTimeout=8", ALIEN, remote_cmd], timeout=timeout)


def log(msg: str) -> None:
    print(f"[exp {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _local_file_size(path: str) -> int:
    p = Path(path)
    return p.stat().st_size if p.is_file() else 0


def _remote_file_size(path: str) -> int:
    quoted = json.dumps(path)
    r = ssh(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"p = Path({quoted})\n"
        "print(p.stat().st_size if p.is_file() else 0)\n"
        "PY",
        timeout=30,
    )
    try:
        return int((r.stdout or "0").strip())
    except ValueError:
        return 0


def validate_spines(spines: object) -> int:
    if not (1 <= int(spines) <= 10):
        raise ValueError(f"--spines must be in [1, 10], got {spines}")
    return int(spines)


def parse_spines(spines: str) -> int:
    try:
        return validate_spines(spines)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def set_temperature(temp: str) -> None:
    """Restart the proxy forcing a sampling temperature (1.0 or 0.6). Cheap,
    no GPU/sudo - just re-execs the proxy with LUMO_PROXY_FORCE_TEMPERATURE set."""
    env = dict(os.environ, LUMO_PROXY_FORCE_TEMPERATURE=str(temp))
    proxy_url = urlparse(PROXY)
    proxy_port = proxy_url.port
    if proxy_port and proxy_port != 8022:
        env.update({
            "LUMO_PROXY_LISTEN_HOST": "0.0.0.0",
            "LUMO_PROXY_LISTEN_PORT": str(proxy_port),
            "LUMO_PROXY_UPSTREAM_BASE_URL": "http://127.0.0.1:9950",
            "LUMO_PROXY_PID_FILE": "/tmp/lumo-l0c-fp8-cutlass-run30-logs/codex_inference_proxy.pid",
            "LUMO_PROXY_LOG_PATH": "/tmp/lumo-l0c-fp8-cutlass-run30-logs/codex_inference_proxy.log",
            "LUMO_PROXY_STATE_ROOT": "/tmp/lumo-l0c-fp8-cutlass-run30-state",
            "LUMO_PROXY_NOHUP_PATH": "/tmp/lumo-l0c-fp8-cutlass-run30-logs/codex_inference_proxy.nohup",
        })
    log(f"setting proxy temperature={temp} (restarting proxy)")
    sh(["bash", str(REPO / "scripts/swe_x86_helpers/relaunch_proxy.sh")], env=env)
    for _ in range(20):
        time.sleep(1)
        p = sh(["curl", "-s", "-m4", "-o", "/dev/null", "-w", "%{http_code}", PROXY])
        if p.stdout.strip() in {"200", "403"}:
            return
    sys.exit("proxy did not come back after temperature change")


def apply_config(
    config: str,
    mtp: int = 1,
    kv_cache_dtype: str | None = None,
    row_mode: str = "tree",
    spines: int = 2,
) -> None:
    """Relaunch vLLM into the requested config and wait for READY. D/E use the
    parameterized round relaunch (/tmp/relaunch_qwen36_round.py, which also
    applies the per-agent spec-step trace patch); A/off fall back to the older
    /tmp scripts. Needs LUMO_SUDO_PASSWORD (host-memory recovery)."""
    if not os.environ.get("LUMO_SUDO_PASSWORD"):
        sys.exit("LUMO_SUDO_PASSWORD required to relaunch vLLM (source .lumo.local.env)")
    round_script = "/tmp/relaunch_qwen36_round.py"
    if config in ("D", "E", "F", "Fb"):
        round_src = REPO / "scripts/swe_x86_helpers/relaunch_qwen36_round.py"
        if not round_src.exists():
            sys.exit(f"round relaunch source missing: {round_src}")
        round_dst = Path(round_script)
        if (not round_dst.exists()
                or round_dst.read_text() != round_src.read_text()):
            round_dst.write_text(round_src.read_text())
        cmd = [str(REPO / ".venv/bin/python"), round_script, "--config", config]
        if config in ("E", "F", "Fb"):
            cmd += ["--mtp", str(mtp)]
        if config == "Fb":
            spine_count = validate_spines(spines)
            cmd += ["--row-mode", row_mode, "--spines", str(spine_count)]
        if kv_cache_dtype:
            cmd += ["--kv-cache-dtype", kv_cache_dtype]
    else:
        script = {"A": "/tmp/relaunch_qwen36_A.py", "off": "/tmp/relaunch_qwen36_off.py"}.get(config)
        if not script or not Path(script).exists():
            sys.exit(f"relaunch script for config {config} not found: {script}")
        cmd = [str(REPO / ".venv/bin/python"), script]
    # Reset the per-agent spec trace BEFORE relaunch: the fresh container opens a
    # new file handle, so each round's trace starts clean (and we never delete it
    # out from under a live handle -- the cause of the round-1 unlinked-inode loss).
    sh(["rm", "-f", PER_REQ_SPEC_TRACE, TREE_ACCEPT_PATH_TRACE,
        TREE_PATH_LCP_TRACE, INDEPENDENT_WINNER_TRACE])
    log(f"relaunching vLLM config={config} mtp={mtp if config in ('E','F','Fb') else '-'} "
        f"row_mode={row_mode if config == 'Fb' else '-'} "
        f"spines={spines if config == 'Fb' else '-'} "
        f"policy={os.environ.get('LUMO_IR_PUBLIC_COMMIT_POLICY', 'lossless') if config == 'Fb' and row_mode == 'independent' else '-'} "
        f"kv={kv_cache_dtype or 'bundle-default'} (model load ~ several min)")
    r = sh(cmd, timeout=VLLM_RELAUNCH_TIMEOUT_S)
    if "READY" not in (r.stdout + r.stderr):
        sys.exit(f"config {config} relaunch did not reach READY:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
    for _ in range(60):
        m = sh(["curl", "-s", "-m4", VLLM_METRICS])
        if "vllm:" in m.stdout:
            log(f"vLLM config {config} serving"); return
        time.sleep(5)
    sys.exit(f"vLLM not serving after config {config} relaunch")


def preflight() -> None:
    # proxy reachable (403 guard counts as alive); vLLM serving; tmux infra up
    p = sh(["curl", "-s", "-m5", "-o", "/dev/null", "-w", "%{http_code}", PROXY])
    if p.stdout.strip() not in {"200", "403"}:
        sys.exit(f"proxy not reachable at {PROXY} (http={p.stdout!r})")
    m = sh(["curl", "-s", "-m5", VLLM_METRICS])
    if "vllm:" not in m.stdout:
        sys.exit(f"vLLM metrics not serving at {VLLM_METRICS} - is the engine ready?")
    t = sh(["tmux", "has-session", "-t", "swe_infra"])
    if t.returncode != 0:
        log("swe_infra tmux missing; rebuilding via setup_tmux_infra.sh")
        sh(["bash", str(REPO / "scripts/swe_x86_helpers/setup_tmux_infra.sh")])


def require_request_metrics_live() -> None:
    """Prove the proxy emits request metrics and the x86 mirror receives them.

    The SWE task runner slices the mirrored capture by byte offset. If this
    check fails, launching a campaign would recreate the zero-byte per-task
    contamination this closeout is trying to avoid.
    """
    local_before = _local_file_size(REQUEST_METRICS)
    remote_before = _remote_file_size(REQUEST_METRICS)
    payload = {
        "model": "qwen3.6-27b",
        "input": "Reply with exactly: OK",
        "max_output_tokens": 4,
        "stream": False,
    }
    log(
        "request-metrics smoke: "
        f"local_before={local_before} remote_before={remote_before}"
    )
    remote_payload = json.dumps(json.dumps(payload))
    smoke_cmd = (
        "curl -sS -m 180 -H 'Content-Type: application/json' "
        "-H 'Authorization: Bearer EMPTY' -f -X POST "
        "http://127.0.0.1:8022/v1/responses "
        f"--data-binary {remote_payload}"
    )
    r = subprocess.CompletedProcess([], returncode=1, stdout="", stderr="")
    deadline = time.time() + 90
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = ssh(smoke_cmd, timeout=210)
        if r.returncode == 0:
            break
        if "Couldn't connect to server" not in (r.stderr or ""):
            break
        log(f"request-metrics smoke waiting for x86 proxy tunnel (attempt={attempt})")
        time.sleep(3)
    if r.returncode != 0:
        sys.exit(
            "request-metrics x86-tunnel smoke request failed:\n"
            f"{r.stdout[-500:]}\n{r.stderr[-500:]}"
        )
    local_after = _local_file_size(REQUEST_METRICS)
    deadline = time.time() + 30
    while local_after <= local_before and time.time() < deadline:
        time.sleep(1)
        local_after = _local_file_size(REQUEST_METRICS)
    if local_after <= local_before:
        sys.exit(
            "request-metrics smoke did not append locally: "
            f"before={local_before} after={local_after} path={REQUEST_METRICS}"
        )
    deadline = time.time() + 30
    remote_after = _remote_file_size(REQUEST_METRICS)
    while remote_after <= remote_before and time.time() < deadline:
        time.sleep(2)
        remote_after = _remote_file_size(REQUEST_METRICS)
    if remote_after <= remote_before:
        sys.exit(
            "request-metrics smoke did not append on alienware mirror: "
            f"before={remote_before} after={remote_after} path={REQUEST_METRICS}"
        )
    log(
        "request-metrics smoke ok: "
        f"local_after={local_after} remote_after={remote_after}"
    )


def launch_suite(args) -> None:
    out_root = f"output/{args.exp_tag}"
    # NOTE: keep nohup as its OWN statement (';'-separated) and background ONLY
    # it with a trailing '&'. If joined with '&&', bash backgrounds the whole
    # AND-list as one subshell that runs the long python in the FOREGROUND,
    # holding the ssh channel open until the run ends (ssh then never returns).
    if args.suite == "swe":
        extra = f"--limit {args.limit}" if args.limit else ""
        skip_existing = " --skip-existing" if args.skip_existing else ""
        cmd = (
            f'cd ~/swe_conc_probe ; export HF_HOME=$HOME/.cache/huggingface ; '
            f'mkdir -p {out_root} ; '
            f'nohup $HOME/swe_eval_offload/venv/bin/python '
            f'scripts/run_swe_bench_q36_a.py --subset {args.subset} --out-root {out_root} '
            f'--dataset-tag {args.exp_tag} --agent-wall-s {args.agent_wall_s} '
            f'--eval-timeout-s {args.eval_timeout_s} --concurrency {args.concurrency} '
            f'{extra}{skip_existing} --repo-cache $HOME/swe_conc_probe/repo_cache '
            f'> {out_root}/driver.log 2>&1 </dev/null & echo launched pid=$!'
        )
    else:  # cnb
        cmd = (
            f'cd ~/cnb_v4a ; mkdir -p output/{args.exp_tag} ; '
            f'nohup python3 /tmp/run_cnb_v4a_x86.py --out output/{args.exp_tag} '
            f'> output/{args.exp_tag}/driver.log 2>&1 </dev/null & echo launched pid=$!'
        )
    r = sh(["ssh", "-n", "-o", "ConnectTimeout=8", ALIEN, cmd], timeout=60)
    log(f"suite launch: {r.stdout.strip() or r.stderr.strip()}")


def remote_exp_dir(args) -> str:
    return (f"~/swe_conc_probe/output/{args.exp_tag}" if args.suite == "swe"
            else f"~/cnb_v4a/output/{args.exp_tag}")


def suite_running(args) -> bool:
    pat = "run_swe_bench_q36_a.py" if args.suite == "swe" else "run_cnb_v4a_x86.py"
    r = ssh(f'ps -eo cmd | grep -c "[{pat[0]}]{pat[1:]}"')
    try:
        return int(r.stdout.strip() or "0") > 0
    except ValueError:
        return False


def capture_nsight(args, seconds: int) -> Path | None:
    """One representative GPU-metrics window; secret-free (--discard-environment)."""
    rep = REPO / f"output/{args.exp_tag}/nsight_{args.exp_tag}"
    rep.parent.mkdir(parents=True, exist_ok=True)
    # wait until decode active
    for _ in range(120):
        m = sh(["curl", "-s", "-m4", VLLM_METRICS])
        for line in m.stdout.splitlines():
            if line.startswith("vllm:num_requests_running") and float(line.split()[-1]) > 0:
                break
        else:
            time.sleep(3); continue
        break
    log(f"nsight: capturing {seconds}s gb20y window -> {rep}.nsys-rep")
    import os
    pw = os.environ.get("LUMO_SUDO_PASSWORD", "")
    cmd = (f'echo "{pw}" | sudo -S -p "" {NSYS} profile --gpu-metrics-devices=0 '
           f'--gpu-metrics-frequency=10 --gpu-metrics-set=gb20y --sample=none '
           f'--cpuctxsw=none --trace=none --discard-environment=true --delay=2 '
           f'-o {rep} --force-overwrite=true sleep {seconds}')
    sh(["bash", "-lc", cmd], timeout=seconds + 120)
    sh(["bash", "-lc", f'echo "{pw}" | sudo -S -p "" chown mark:mark {rep}.nsys-rep'])
    sqlite = Path(f"{rep}.sqlite")
    sh([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
        "--output", str(sqlite), f"{rep}.nsys-rep"])
    return sqlite if sqlite.exists() else None


def join_metrics(args, nsight_sqlite: Path | None) -> Path | None:
    if not nsight_sqlite or not nsight_sqlite.exists():
        return None
    out = REPO / f"output/{args.exp_tag}/joined_decode_gpu.csv"
    cmd = [".venv/bin/python", "scripts/join_nsight_decode_metrics.py",
           "--nsight-sqlite", str(nsight_sqlite),
           "--request-metrics", REQUEST_METRICS, "--out", str(out)]
    if Path(STEPTRACE).exists():
        cmd += ["--steptrace", STEPTRACE]
    r = sh(cmd, cwd=str(REPO))
    log("join: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:200]))
    return out if out.exists() else None


def rsync_back(args) -> Path:
    local = REPO / f"output/{args.exp_tag}"
    local.mkdir(parents=True, exist_ok=True)
    src = remote_exp_dir(args).replace("~", f"/home/mark") + "/"
    sh(["rsync", "-az", f"{ALIEN}:{src}", str(local) + "/"])
    sh(["cp", STEPTRACE, str(local / "dgx_steptrace.jsonl")])
    # per-agent spec-step trace (bind-mounted from the vLLM container) -- clean
    # per-request decode steps/acceptance regardless of batch size
    sh(["cp", PER_REQ_SPEC_TRACE, str(local / "per_req_spec_trace.jsonl")])
    if Path(TREE_ACCEPT_PATH_TRACE).exists():
        sh(["cp", TREE_ACCEPT_PATH_TRACE, str(local / "tree_accept_path.jsonl")])
    if Path(TREE_PATH_LCP_TRACE).exists():
        sh(["cp", TREE_PATH_LCP_TRACE, str(local / "tree_path_lcp_max.jsonl")])
    if Path(INDEPENDENT_WINNER_TRACE).exists():
        sh(["cp", INDEPENDENT_WINNER_TRACE, str(local / "independent_winner_trace.jsonl")])
    return local


def finalize_tree_superset(local: Path) -> None:
    trace = local / "tree_path_lcp_max.jsonl"
    if not trace.exists():
        return
    summary = local / "tree_path_lcp_superset_summary.json"
    cmd = [
        sys.executable,
        str(REPO / "scripts/verify_tree_path_lcp_superset.py"),
        str(trace),
        "--json",
    ]
    result = sh(cmd, cwd=str(REPO))
    if result.returncode != 0:
        sys.exit(
            "tree path-LCP superset verification failed:\n"
            f"{result.stdout}{result.stderr}"
        )
    summary.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    data = json.loads(result.stdout)
    log(
        "tree superset verified: "
        f"rows={data['rows']} avg_path0={data['avg_path0_lcp']:.3f} "
        f"avg_winner={data['avg_accepted_len']:.3f} "
        f"recovered_tokens={data['recovered_token_total']}"
    )


def finalize_independent_winner(args, local: Path) -> None:
    if args.config != "Fb" or args.row_mode != "independent" or args.spines <= 1:
        return
    trace = local / "independent_winner_trace.jsonl"
    summary = local / "independent_winner_summary.json"
    cmd = [
        sys.executable,
        str(REPO / "scripts/verify_independent_winner_trace.py"),
        str(trace),
        "--json",
        "--require-recovery",
    ]
    result = sh(cmd, cwd=str(REPO))
    if result.returncode != 0:
        sys.exit(
            "independent winner trace verification failed:\n"
            f"{result.stdout}{result.stderr}"
        )
    summary.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    data = json.loads(result.stdout)
    log(
        "independent winner verified: "
        f"rows={data['rows']} viol={data['superset_violations']} "
        f"copy_missing_sum={data['copy_missing_sum']} "
        f"suppressed={data.get('hidden_winner_suppressed_events', 0)} "
        f"policy={','.join((data.get('policies') or {}).keys()) or '-'} "
        f"recovered_tokens={data['recovered_token_total']}"
    )


def finalize_agentic_summary(args, local: Path) -> Path:
    summary = local / "agentic_summary.json"
    cmd = [
        sys.executable,
        str(REPO / "scripts/summarize_round_f_agentic_arm.py"),
        "--exp-dir",
        str(local),
        "--label",
        args.exp_tag,
        "--nodes",
        str(args.spines if args.config == "Fb" else args.mtp),
        "--out",
        str(summary),
    ]
    result = sh(cmd, cwd=str(REPO))
    if result.returncode != 0:
        sys.exit(f"agentic summary build failed:\n{result.stdout}{result.stderr}")
    if args.config == "Fb" and args.row_mode == "independent" and args.spines > 1:
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["independent_rows_public_commit"] = {
                "policy": os.environ.get("LUMO_IR_PUBLIC_COMMIT_POLICY", "lossless"),
                "selector_enabled": False,
                "lossless_public_stream": True,
                "hidden_recovery_publication": "suppressed_no_lossless_selector",
            }
            winner_summary = local / "independent_winner_summary.json"
            if winner_summary.exists():
                payload["independent_winner_summary"] = json.loads(
                    winner_summary.read_text(encoding="utf-8")
                )
            summary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            sys.exit(f"agentic summary policy annotation failed: {exc}")
    log(f"agentic summary written: {summary.relative_to(REPO)}")
    return summary


def finalize_speed_comparison(args, local: Path, summary_path: Path) -> bool:
    if not args.speed_baseline_agentic_summary:
        return False
    baseline_path = Path(args.speed_baseline_agentic_summary)
    if not baseline_path.is_absolute():
        baseline_path = REPO / baseline_path
    current = json.loads(summary_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    current_tps = (current.get("steptrace") or {}).get("decode_tps")
    baseline_tps = (baseline.get("steptrace") or {}).get("decode_tps")
    payload = {
        "current_summary": str(summary_path.relative_to(REPO)),
        "baseline_summary": str(baseline_path.relative_to(REPO)),
        "current_decode_tps": current_tps,
        "baseline_decode_tps": baseline_tps,
        "speedup": (
            float(current_tps) / float(baseline_tps)
            if current_tps is not None and baseline_tps else None
        ),
        "require_speed_win": bool(args.require_speed_win),
    }
    if payload["speedup"] is not None:
        payload["speed_win"] = payload["speedup"] > 1.0
    out = local / "speed_comparison.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        "speed comparison: "
        f"current_tps={current_tps} baseline_tps={baseline_tps} "
        f"speedup={payload['speedup']}"
    )
    if args.require_speed_win and not payload.get("speed_win"):
        return True
    return False


def task_verdict(meta: Path) -> tuple[str, float | None]:
    try:
        d = json.loads(meta.read_text())
    except Exception:
        return "?", None
    if not d.get("ended_at"):
        return "running", None
    er = d.get("eval_report") or {}
    verdict = er.get("verdict") or ("done" if d.get("codex", {}).get("exit_code") is not None else "?")
    s, e = d.get("started_at"), d.get("ended_at")
    elapsed = None
    if s and e:
        f = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        elapsed = f(e) - f(s)
    return verdict, elapsed


def require_swe_task_request_metrics(args, meta: Path) -> None:
    if args.suite != "swe":
        return
    metrics = meta.parent / "vllm_request_metrics.jsonl"
    if not metrics.is_file() or metrics.stat().st_size == 0:
        ssh(
            "pkill -TERM -f "
            + json.dumps(f"run_swe_bench_q36_a.py.*{args.exp_tag}")
            + " 2>/dev/null || true",
            timeout=30,
        )
        sys.exit(
            "SWE task finished without nonzero vLLM request metrics; "
            "aborting before commit to avoid contaminated evidence: "
            f"tag={args.exp_tag} task={meta.parent.name} path={metrics}"
        )


def _codex_protocol_marker_hits(task_dir: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for trace in sorted(task_dir.glob("codex_trace*.jsonl")):
        try:
            lines = trace.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") if isinstance(event, dict) else None
            if not isinstance(item, dict):
                continue
            values: list[str] = []
            text = item.get("text")
            command = item.get("command")
            if isinstance(text, str):
                values.append(text)
            if isinstance(command, str):
                values.append(command)
            for value in values:
                if any(marker in value for marker in FORBIDDEN_CODEX_PROTOCOL_MARKERS):
                    hits.append((trace.name, lineno, value[:300].replace("\n", "\\n")))
                    break
    return hits


def require_swe_task_protocol_clean(args, meta: Path) -> None:
    if args.suite != "swe":
        return
    hits = _codex_protocol_marker_hits(meta.parent)
    if not hits:
        return
    ssh(
        "pkill -TERM -f "
        + json.dumps(f"run_swe_bench_q36_a.py.*{args.exp_tag}")
        + " 2>/dev/null || true",
        timeout=30,
    )
    details = "\n".join(
        f"{trace}:{lineno}: {snippet}" for trace, lineno, snippet in hits[:8]
    )
    sys.exit(
        "SWE task Codex trace contains raw model protocol markers; "
        "aborting before commit to avoid contaminated evidence: "
        f"tag={args.exp_tag} task={meta.parent.name}\n{details}"
    )


def commit_task(args, task_id: str, verdict: str, joined: Path | None) -> None:
    rel = f"output/{args.exp_tag}"
    paths = [f"{rel}/*/per_task/{task_id}", f"{rel}/per_task/{task_id}",
             f"{rel}/driver.log", f"{rel}/dgx_steptrace.jsonl",
             f"{rel}/per_req_spec_trace.jsonl"]
    if (REPO / rel / "tree_accept_path.jsonl").exists():
        paths.append(f"{rel}/tree_accept_path.jsonl")
    if (REPO / rel / "tree_path_lcp_max.jsonl").exists():
        paths.append(f"{rel}/tree_path_lcp_max.jsonl")
    if (REPO / rel / "tree_path_lcp_superset_summary.json").exists():
        paths.append(f"{rel}/tree_path_lcp_superset_summary.json")
    if (REPO / rel / "independent_winner_trace.jsonl").exists():
        paths.append(f"{rel}/independent_winner_trace.jsonl")
    if (REPO / rel / "independent_winner_summary.json").exists():
        paths.append(f"{rel}/independent_winner_summary.json")
    if (REPO / rel / "agentic_summary.json").exists():
        paths.append(f"{rel}/agentic_summary.json")
    if (REPO / rel / "speed_comparison.json").exists():
        paths.append(f"{rel}/speed_comparison.json")
    if joined:
        paths.append(str(joined.relative_to(REPO)))
    nrep = REPO / f"{rel}/nsight_{args.exp_tag}.nsys-rep"
    if nrep.exists():
        paths.append(f"{rel}/nsight_{args.exp_tag}.nsys-rep")
    for p in paths:
        sh(["bash", "-lc", f"git add -f {p} 2>/dev/null"], cwd=str(REPO))
    fb_desc = f", row_mode {args.row_mode}, spines {args.spines}" if args.config == "Fb" else ""
    msg = (f"{args.suite} {args.exp_tag} (config {args.config}{fb_desc}, temp {args.temp or 'as-set'}, "
           f"c={args.concurrency}) +task {task_id} verdict={verdict}\n\n"
           f"Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>")
    c = sh(["git", "commit", "-q", "-m", msg], cwd=str(REPO))
    if c.returncode == 0:
        push = sh(["git", "push", "origin", "main"], cwd=str(REPO))
        log(f"committed+pushed {task_id} (verdict={verdict}); push: {push.stdout.strip().splitlines()[-1] if push.stdout.strip() else push.stderr.strip().splitlines()[-1] if push.stderr.strip() else 'ok'}")
    else:
        log(f"commit {task_id}: nothing to commit (already tracked?)")


def commit_final_artifacts(args, joined: Path | None) -> None:
    rel = f"output/{args.exp_tag}"
    paths = [
        f"{rel}/campaign_summary.json",
        f"{rel}/{args.exp_tag}/campaign_summary.json",
        f"{rel}/{args.exp_tag}/predictions.jsonl",
        f"{rel}/agentic_summary.json",
        f"{rel}/dgx_steptrace.jsonl",
        f"{rel}/per_req_spec_trace.jsonl",
        f"{rel}/driver.log",
    ]
    for name in [
        "tree_accept_path.jsonl",
        "tree_path_lcp_max.jsonl",
        "tree_path_lcp_superset_summary.json",
        "independent_winner_trace.jsonl",
        "independent_winner_summary.json",
        "speed_comparison.json",
    ]:
        if (REPO / rel / name).exists():
            paths.append(f"{rel}/{name}")
    if joined:
        paths.append(str(joined.relative_to(REPO)))
    for p in paths:
        sh(["bash", "-lc", f"git add -f {p} 2>/dev/null"], cwd=str(REPO))
    msg = (
        f"Finalize {args.suite} {args.exp_tag} artifacts\n\n"
        f"Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
    )
    c = sh(["git", "commit", "-q", "-m", msg], cwd=str(REPO))
    if c.returncode == 0:
        push = sh(["git", "push", "origin", "main"], cwd=str(REPO))
        tail = (
            push.stdout.strip().splitlines()[-1]
            if push.stdout.strip()
            else push.stderr.strip().splitlines()[-1]
            if push.stderr.strip()
            else "ok"
        )
        log(f"final artifacts committed+pushed; push: {tail}")
    else:
        log("final artifacts: nothing to commit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-tag", required=True)
    ap.add_argument("--suite", choices=["swe", "cnb"], default="swe")
    ap.add_argument("--config", choices=["A", "D", "off", "E", "F", "Fb", "G"], required=True,
                    help="ablation/spec config; with --apply-config the runner relaunches vLLM into it. "
                         "E=Qwen3.6 native MTP head; F=tree MTP; Fb=batched-path MTP; G=combB (per-position mix)")
    ap.add_argument("--apply-config", action="store_true",
                    help="relaunch vLLM into --config (A/off/E have relaunch scripts; needs LUMO_SUDO_PASSWORD)")
    ap.add_argument("--temp", choices=["1.0", "0.6"], default=None,
                    help="force sampling temperature by restarting the proxy")
    ap.add_argument("--mtp", type=int, default=1,
                    help="config E num_speculative_tokens (MTP depth) when --apply-config")
    ap.add_argument("--row-mode", choices=["tree", "independent"], default="tree",
                    help="config Fb row layout when --apply-config")
    ap.add_argument("--spines", type=parse_spines, default=None,
                    help="config Fb spine/independent-row count when --apply-config; must be in [1, 10]")
    ap.add_argument("--kv-cache-dtype", default=None,
                    choices=["auto", "fp8_e5m2", "fp8_e4m3"],
                    help="override realized KV cache dtype on relaunch (fp8_e4m3 = realized FP8 "
                         "KV for the fp8 checkpoint; default uses the bundle value)")
    ap.add_argument("--subset", help="subset json path ON ALIENWARE (swe)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--agent-wall-s", type=int, default=1800)
    ap.add_argument("--eval-timeout-s", type=int, default=1800)
    ap.add_argument("--nsight", default="off",
                    help="off | first-task | <seconds> (one representative window)")
    ap.add_argument("--no-commit", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume an existing SWE tag by skipping finished tasks; forbidden for clean benchmark arms")
    ap.add_argument("--speed-baseline-agentic-summary", default=None,
                    help="optional baseline agentic_summary.json for final decode_tps comparison")
    ap.add_argument("--require-speed-win", action="store_true",
                    help="fail finalization unless current decode_tps is greater than the baseline")
    ap.add_argument("--poll-s", type=int, default=30)
    args = ap.parse_args()
    if args.spines is None:
        try:
            args.spines = validate_spines(os.environ.get("LUMO_TREE_SPINES", "2"))
        except ValueError as exc:
            if args.config == "Fb":
                ap.error(str(exc))
            args.spines = 2
    if args.suite == "swe" and not args.subset:
        sys.exit("--subset required for swe")
    if args.concurrency != 1:
        log(f"WARNING: concurrency={args.concurrency} (feedback-no-parallel-testing: default is 1)")

    if args.apply_config:
        apply_config(
            args.config,
            args.mtp,
            kv_cache_dtype=args.kv_cache_dtype,
            row_mode=args.row_mode,
            spines=args.spines,
        )
    if args.temp:
        set_temperature(args.temp)
    preflight()
    if args.suite == "swe":
        require_request_metrics_live()
    log(f"experiment {args.exp_tag}: config={args.config} temp={args.temp or 'as-set'} "
        f"row_mode={args.row_mode if args.config == 'Fb' else '-'} "
        f"spines={args.spines if args.config == 'Fb' else '-'} "
        f"concurrency={args.concurrency} suite={args.suite} nsight={args.nsight}")
    launch_suite(args)

    nsight_sqlite: Path | None = None
    if args.nsight != "off":
        secs = 150 if args.nsight == "first-task" else int(args.nsight)
        nsight_sqlite = capture_nsight(args, secs)

    committed: set[str] = set()
    while True:
        local = rsync_back(args)
        # find finished per_task dirs
        for meta in list(local.glob("**/per_task/*/runner_metadata.json")) + \
                    list(local.glob("**/per_task/*/result.json")):
            tid = meta.parent.name
            if tid in committed:
                continue
            verdict, elapsed = task_verdict(meta)
            if verdict in {"running", "?"}:
                continue
            require_swe_task_request_metrics(args, meta)
            require_swe_task_protocol_clean(args, meta)
            joined = join_metrics(args, nsight_sqlite)
            if not args.no_commit:
                commit_task(args, tid, verdict, joined)
            committed.add(tid)
            log(f"task {tid}: verdict={verdict} elapsed={elapsed:.0f}s" if elapsed else f"task {tid}: verdict={verdict}")
        if not suite_running(args):
            # final sweep to catch the last task
            local = rsync_back(args)
            remaining = [m for m in local.glob("**/per_task/*/runner_metadata.json")
                         if m.parent.name not in committed and task_verdict(m)[0] not in {"running", "?"}]
            for meta in remaining:
                tid = meta.parent.name
                verdict, elapsed = task_verdict(meta)
                require_swe_task_request_metrics(args, meta)
                require_swe_task_protocol_clean(args, meta)
                joined = join_metrics(args, nsight_sqlite)
                if not args.no_commit:
                    commit_task(args, tid, verdict, joined)
                committed.add(tid)
            finalize_tree_superset(local)
            finalize_independent_winner(args, local)
            agentic_summary = finalize_agentic_summary(args, local)
            speed_failure = finalize_speed_comparison(args, local, agentic_summary)
            if not args.no_commit:
                commit_final_artifacts(args, nsight_sqlite)
            if speed_failure:
                sys.exit(
                    "required speed win not met; final evidence was written"
                    " and committed before failing the run"
                )
            log(f"suite finished; {len(committed)} task(s) committed")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
