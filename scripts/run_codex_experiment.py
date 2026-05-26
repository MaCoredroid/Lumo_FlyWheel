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
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel")
ALIEN = "alienware"
PROXY = "http://127.0.0.1:8022/v1/models"
VLLM_METRICS = "http://127.0.0.1:9950/metrics"
NSYS = "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
STEPTRACE = "/tmp/swe_dgx_steptrace.jsonl"
REQUEST_METRICS = "/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"
PER_REQ_SPEC_TRACE = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl"


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ssh(remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return sh(["ssh", "-o", "ConnectTimeout=8", ALIEN, remote_cmd], timeout=timeout)


def log(msg: str) -> None:
    print(f"[exp {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def set_temperature(temp: str) -> None:
    """Restart the proxy forcing a sampling temperature (1.0 or 0.6). Cheap,
    no GPU/sudo - just re-execs the proxy with LUMO_PROXY_FORCE_TEMPERATURE set."""
    import os
    env = dict(os.environ, LUMO_PROXY_FORCE_TEMPERATURE=str(temp))
    log(f"setting proxy temperature={temp} (restarting proxy)")
    sh(["bash", str(REPO / "scripts/swe_x86_helpers/relaunch_proxy.sh")], env=env)
    for _ in range(20):
        time.sleep(1)
        p = sh(["curl", "-s", "-m4", "-o", "/dev/null", "-w", "%{http_code}", PROXY])
        if p.stdout.strip() in {"200", "403"}:
            return
    sys.exit("proxy did not come back after temperature change")


def apply_config(config: str, mtp: int = 1, kv_cache_dtype: str | None = None) -> None:
    """Relaunch vLLM into the requested config and wait for READY. D/E use the
    parameterized round relaunch (/tmp/relaunch_qwen36_round.py, which also
    applies the per-agent spec-step trace patch); A/off fall back to the older
    /tmp scripts. Needs LUMO_SUDO_PASSWORD (host-memory recovery)."""
    import os
    if not os.environ.get("LUMO_SUDO_PASSWORD"):
        sys.exit("LUMO_SUDO_PASSWORD required to relaunch vLLM (source .lumo.local.env)")
    round_script = "/tmp/relaunch_qwen36_round.py"
    if config in ("D", "E", "F") and Path(round_script).exists():
        cmd = [str(REPO / ".venv/bin/python"), round_script, "--config", config]
        if config in ("E", "F"):
            cmd += ["--mtp", str(mtp)]
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
    sh(["rm", "-f", PER_REQ_SPEC_TRACE])
    log(f"relaunching vLLM config={config} mtp={mtp if config in ('E','F') else '-'} "
        f"kv={kv_cache_dtype or 'bundle-default'} (model load ~ several min)")
    r = sh(cmd, timeout=1200)
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
        sys.exit(f"proxy not reachable on 8022 (http={p.stdout!r})")
    m = sh(["curl", "-s", "-m5", VLLM_METRICS])
    if "vllm:" not in m.stdout:
        sys.exit("vLLM /metrics on 9950 not serving - is the engine ready?")
    t = sh(["tmux", "has-session", "-t", "swe_infra"])
    if t.returncode != 0:
        log("swe_infra tmux missing; rebuilding via setup_tmux_infra.sh")
        sh(["bash", str(REPO / "scripts/swe_x86_helpers/setup_tmux_infra.sh")])


def launch_suite(args) -> None:
    out_root = f"output/{args.exp_tag}"
    # NOTE: keep nohup as its OWN statement (';'-separated) and background ONLY
    # it with a trailing '&'. If joined with '&&', bash backgrounds the whole
    # AND-list as one subshell that runs the long python in the FOREGROUND,
    # holding the ssh channel open until the run ends (ssh then never returns).
    if args.suite == "swe":
        extra = f"--limit {args.limit}" if args.limit else ""
        cmd = (
            f'cd ~/swe_conc_probe ; export HF_HOME=$HOME/.cache/huggingface ; '
            f'mkdir -p {out_root} ; '
            f'nohup $HOME/swe_eval_offload/venv/bin/python '
            f'scripts/run_swe_bench_q36_a.py --subset {args.subset} --out-root {out_root} '
            f'--dataset-tag {args.exp_tag} --agent-wall-s {args.agent_wall_s} '
            f'--eval-timeout-s {args.eval_timeout_s} --concurrency {args.concurrency} '
            f'{extra} --skip-existing --repo-cache $HOME/swe_conc_probe/repo_cache '
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
    return local


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


def commit_task(args, task_id: str, verdict: str, joined: Path | None) -> None:
    rel = f"output/{args.exp_tag}"
    paths = [f"{rel}/*/per_task/{task_id}", f"{rel}/per_task/{task_id}",
             f"{rel}/driver.log", f"{rel}/dgx_steptrace.jsonl",
             f"{rel}/per_req_spec_trace.jsonl"]
    if joined:
        paths.append(str(joined.relative_to(REPO)))
    nrep = REPO / f"{rel}/nsight_{args.exp_tag}.nsys-rep"
    if nrep.exists():
        paths.append(f"{rel}/nsight_{args.exp_tag}.nsys-rep")
    for p in paths:
        sh(["bash", "-lc", f"git add -f {p} 2>/dev/null"], cwd=str(REPO))
    msg = (f"{args.suite} {args.exp_tag} (config {args.config}, temp {args.temp or 'as-set'}, "
           f"c={args.concurrency}) +task {task_id} verdict={verdict}\n\n"
           f"Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>")
    c = sh(["git", "commit", "-q", "-m", msg], cwd=str(REPO))
    if c.returncode == 0:
        push = sh(["git", "push", "origin", "main"], cwd=str(REPO))
        log(f"committed+pushed {task_id} (verdict={verdict}); push: {push.stdout.strip().splitlines()[-1] if push.stdout.strip() else push.stderr.strip().splitlines()[-1] if push.stderr.strip() else 'ok'}")
    else:
        log(f"commit {task_id}: nothing to commit (already tracked?)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-tag", required=True)
    ap.add_argument("--suite", choices=["swe", "cnb"], default="swe")
    ap.add_argument("--config", choices=["A", "D", "off", "E", "F", "G"], required=True,
                    help="ablation/spec config; with --apply-config the runner relaunches vLLM into it. "
                         "E=Qwen3.6 native MTP head; F=combA (MTP+SD tau-threshold); G=combB (per-position mix)")
    ap.add_argument("--apply-config", action="store_true",
                    help="relaunch vLLM into --config (A/off/E have relaunch scripts; needs LUMO_SUDO_PASSWORD)")
    ap.add_argument("--temp", choices=["1.0", "0.6"], default=None,
                    help="force sampling temperature by restarting the proxy")
    ap.add_argument("--mtp", type=int, default=1,
                    help="config E num_speculative_tokens (MTP depth) when --apply-config")
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
    ap.add_argument("--poll-s", type=int, default=30)
    args = ap.parse_args()
    if args.suite == "swe" and not args.subset:
        sys.exit("--subset required for swe")
    if args.concurrency != 1:
        log(f"WARNING: concurrency={args.concurrency} (feedback-no-parallel-testing: default is 1)")

    if args.apply_config:
        apply_config(args.config, args.mtp, kv_cache_dtype=args.kv_cache_dtype)
    if args.temp:
        set_temperature(args.temp)
    preflight()
    log(f"experiment {args.exp_tag}: config={args.config} temp={args.temp or 'as-set'} "
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
                joined = join_metrics(args, nsight_sqlite)
                if not args.no_commit:
                    commit_task(args, tid, verdict, joined)
                committed.add(tid)
            log(f"suite finished; {len(committed)} task(s) committed")
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
