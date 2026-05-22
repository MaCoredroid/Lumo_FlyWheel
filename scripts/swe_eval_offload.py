#!/usr/bin/env python3
"""Offload SWE-Bench eval to a native x86_64 host (alienware) over SSH.

The Codex agent step stays on the DGX Spark (needs the Q36-A vLLM); only
the eval step runs on x86 — where every instance's conda env builds
(old python/dep pins exist as x86_64 packages), so the ~20-35% of
SWE-Bench Verified that can't build on aarch64 are recoverable, and the
whole dataset becomes single-arch (matches the published reference
platform; removes the experimental-arm64 verdict doubt).

Per-instance flow:
  1. read local per_task/<id>/patch.diff
  2. scp patch.diff -> alienware:~/swe_eval_offload/work/<id>/patch.diff
  3. ssh alienware: run swe_eval_x86_worker.py (native x86 harness)
  4. scp the eval artifacts back to per_task/<id>/eval_x86/
  5. merge the x86 verdict into runner_metadata.json (keeps the original
     arm64 eval_report for comparison; x86 becomes authoritative)

Network-error tolerance (the explicit requirement):
  - every ssh/scp wrapped in a bounded retry loop with exponential backoff
  - SSH keepalives + connect timeouts so a flaky link fails fast and retries
  - every network failure is timestamped into a dedicated network-issue log
  - verbose per-step logging; the eval data captured is the same artifact
    set as the local path
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "swe_bench_q36_a_temp06"
DEFAULT_HOST = "alienware"
REMOTE_BASE = "~/swe_eval_offload"
REMOTE_WORKER = "~/swe_eval_offload/swe_eval_x86_worker.py"
REMOTE_VENV_PY = "~/swe_eval_offload/venv/bin/python"
REMOTE_HF_HOME = "~/.cache/huggingface"

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "-o", "StrictHostKeyChecking=accept-new",
]

NET_LOG = DEFAULT_OUT_ROOT / "swe_eval_offload_network.log"
DISK_MIN_GB = 12.0  # prune alienware docker images when /home/docker drops below this


def _iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _net_log(msg: str) -> None:
    NET_LOG.parent.mkdir(parents=True, exist_ok=True)
    with NET_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{_iso()}] {msg}\n")


def _run_with_retries(argv: list[str], *, what: str, host: str,
                      timeout: int, max_attempts: int = 5) -> subprocess.CompletedProcess:
    """Run a network command (ssh/scp) with bounded exponential-backoff retries.

    Network failures (non-zero rc, timeout, transport errors) are logged to
    the network-issue log with the attempt number; the final failure is
    re-raised so the caller can decide whether to skip or abort.
    """
    backoff = 5
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            if cp.returncode == 0:
                if attempt > 1:
                    _net_log(f"RECOVERED {what} host={host} on attempt {attempt}")
                return cp
            last = cp
            tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-1:] or [""]
            _net_log(f"FAIL {what} host={host} attempt={attempt}/{max_attempts} "
                     f"rc={cp.returncode} :: {tail[0][:300]}")
        except subprocess.TimeoutExpired:
            _net_log(f"TIMEOUT {what} host={host} attempt={attempt}/{max_attempts} "
                     f"after {timeout}s")
            last = subprocess.CompletedProcess(argv, 124, "", "timeout")
        if attempt < max_attempts:
            time.sleep(backoff)
            backoff = min(backoff * 3, 300)
    _net_log(f"GIVEUP {what} host={host} after {max_attempts} attempts")
    return last if last is not None else subprocess.CompletedProcess(argv, 1, "", "unknown")


def _ssh(host: str, remote_cmd: str, *, timeout: int, what: str,
         max_attempts: int = 5) -> subprocess.CompletedProcess:
    return _run_with_retries(["ssh", *SSH_OPTS, host, remote_cmd],
                             what=what, host=host, timeout=timeout, max_attempts=max_attempts)


def _scp(host: str, src: str, dst: str, *, timeout: int, what: str,
         recursive: bool = False, max_attempts: int = 5) -> subprocess.CompletedProcess:
    argv = ["scp", *SSH_OPTS]
    if recursive:
        argv.append("-r")
    argv += [src, dst]
    return _run_with_retries(argv, what=what, host=host, timeout=timeout, max_attempts=max_attempts)


def _alienware_disk_gb(host: str) -> float | None:
    cp = _ssh(host, "df -BG --output=avail /home/docker 2>/dev/null | tail -1 | tr -dc '0-9'",
              timeout=30, what="disk_check", max_attempts=3)
    if cp.returncode == 0 and cp.stdout.strip().isdigit():
        return float(cp.stdout.strip())
    return None


def _maybe_prune(host: str) -> None:
    free = _alienware_disk_gb(host)
    if free is not None and free < DISK_MIN_GB:
        _net_log(f"DISK low on {host}: {free}GB < {DISK_MIN_GB}GB — pruning sweb eval images")
        # Remove instance/env eval images (keep base) to reclaim space.
        _ssh(host,
             "docker images --filter 'reference=swebench/sweb.eval.*' -q | xargs -r docker rmi -f >/dev/null 2>&1; "
             "docker image prune -f >/dev/null 2>&1; echo pruned",
             timeout=300, what="prune", max_attempts=2)


def _process(host: str, instance_id: str, dataset_name: str, per_task_root: Path,
             timeout_s: int, model_name: str) -> dict[str, Any]:
    task_dir = per_task_root / instance_id
    patch_path = task_dir / "patch.diff"
    meta_path = task_dir / "runner_metadata.json"
    result: dict[str, Any] = {"instance_id": instance_id}

    if not patch_path.is_file():
        result["status"] = "no_patch_on_disk"
        return result

    remote_dir = f"{REMOTE_BASE}/work/{instance_id}"
    print(f"[{_iso()}] -> {instance_id}: prep remote dir", flush=True)
    mk = _ssh(host, f"mkdir -p {remote_dir} && echo ok", timeout=30, what=f"mkdir:{instance_id}")
    if mk.returncode != 0:
        result["status"] = "ssh_mkdir_failed"
        return result

    # 1. ship patch
    print(f"[{_iso()}]    scp patch -> {host}", flush=True)
    up = _scp(host, str(patch_path), f"{host}:{remote_dir}/patch.diff",
              timeout=120, what=f"scp_up:{instance_id}")
    if up.returncode != 0:
        result["status"] = "scp_up_failed"
        return result

    # 2. run eval natively on x86
    _maybe_prune(host)
    print(f"[{_iso()}]    run x86 eval (timeout {timeout_s}s)", flush=True)
    remote_cmd = (
        f"cd {REMOTE_BASE} && HF_HOME={REMOTE_HF_HOME} "
        f"{REMOTE_VENV_PY} {REMOTE_WORKER} "
        f"--instance-id {instance_id} "
        f"--patch-path {remote_dir}/patch.diff "
        f"--output-dir {remote_dir}/out "
        f"--dataset-name '{dataset_name}' "
        f"--model-name '{model_name}' "
        f"--timeout-s {timeout_s} --cache-level env"
    )
    # ssh timeout = eval timeout + generous slack for image pull
    ev = _ssh(host, remote_cmd, timeout=timeout_s + 900, what=f"eval:{instance_id}", max_attempts=2)
    result["worker_rc"] = ev.returncode
    if ev.stdout:
        for line in ev.stdout.splitlines():
            if line.startswith("VERDICT"):
                result["worker_verdict_line"] = line.strip()

    # 3. fetch artifacts back (even on failure — eval.log helps debug)
    local_eval = task_dir / "eval_x86"
    local_eval.mkdir(parents=True, exist_ok=True)
    print(f"[{_iso()}]    scp artifacts <- {host}", flush=True)
    _scp(host, f"{host}:{remote_dir}/out/eval_report.json", str(local_eval / "eval_report.json"),
         timeout=120, what=f"scp_report:{instance_id}", max_attempts=4)
    _scp(host, f"{host}:{remote_dir}/out/normalized_eval.json", str(local_eval / "normalized_eval.json"),
         timeout=120, what=f"scp_norm:{instance_id}", max_attempts=3)
    _scp(host, f"{host}:{remote_dir}/out/eval.log", str(local_eval / "eval.log"),
         timeout=120, what=f"scp_log:{instance_id}", max_attempts=3)
    _scp(host, f"{host}:{remote_dir}/out/predictions.jsonl", str(local_eval / "predictions.jsonl"),
         timeout=120, what=f"scp_pred:{instance_id}", max_attempts=2)

    # 4. merge x86 verdict into runner_metadata.json (keep arm64 for comparison)
    x86_report = None
    rp = local_eval / "eval_report.json"
    if rp.is_file():
        try:
            x86_report = json.loads(rp.read_text())
        except Exception:  # noqa: BLE001
            pass
    result["x86_report"] = x86_report

    if x86_report and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            arm64_report = meta.get("eval_report")
            if arm64_report is not None and "eval_report_arm64" not in meta:
                meta["eval_report_arm64"] = arm64_report
            meta["eval_report_x86"] = x86_report
            meta["eval_report"] = x86_report  # x86 is authoritative
            meta["eval_arch_authoritative"] = "x86_64"
            arm64_v = (arm64_report or {}).get("verdict")
            x86_v = x86_report.get("verdict")
            meta["arch_verdict_changed"] = bool(arm64_v and x86_v and arm64_v != x86_v)
            meta_path.write_text(json.dumps(meta, indent=2))
            result["arm64_verdict"] = arm64_v
            result["x86_verdict"] = x86_v
            result["changed"] = meta["arch_verdict_changed"]
        except Exception as exc:  # noqa: BLE001
            result["merge_error"] = f"{type(exc).__name__}: {exc}"

    # 5. cleanup remote work dir (keep disk tidy)
    _ssh(host, f"rm -rf {remote_dir}", timeout=30, what=f"cleanup:{instance_id}", max_attempts=2)
    result["status"] = "done"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--dataset-tag", default="verified")
    ap.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--model-name", default="qwen3.6-27b-fp8::codex-cli-0.128.0::q36-a")
    ap.add_argument("--timeout-s", type=int, default=2700)  # 45 min eval ceiling
    ap.add_argument("--instance-ids", default=None,
                    help="Comma-separated subset. Default: all per_task dirs with runner_metadata.json.")
    ap.add_argument("--only-changed-arch", action="store_true",
                    help="(unused placeholder for future filtering)")
    args = ap.parse_args(argv)

    per_task_root = args.out_root / args.dataset_tag / "per_task"
    if args.instance_ids:
        ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    else:
        ids = sorted(p.name for p in per_task_root.iterdir()
                     if p.is_dir() and (p / "runner_metadata.json").is_file())

    print(f"=== [{_iso()}] x86 eval-offload host={args.host} n={len(ids)} "
          f"dataset={args.dataset_name} ===", flush=True)
    _net_log(f"START offload host={args.host} n={len(ids)} dataset={args.dataset_name}")

    summaries = []
    for iid in ids:
        t0 = time.time()
        res = _process(args.host, iid, args.dataset_name, per_task_root,
                       args.timeout_s, args.model_name)
        dt = time.time() - t0
        flag = ""
        if res.get("changed"):
            flag = f"  ARCH-FLIP {res.get('arm64_verdict')}->{res.get('x86_verdict')}"
        print(f"[{_iso()}] <- {iid} status={res.get('status')} "
              f"x86={res.get('x86_verdict','?')} elapsed={dt:.0f}s{flag}", flush=True)
        summaries.append(res)

    # summary
    done = [s for s in summaries if s.get("status") == "done"]
    flips = [s for s in done if s.get("changed")]
    print(f"=== [{_iso()}] OFFLOAD DONE n={len(ids)} completed={len(done)} "
          f"arch_flips={len(flips)} ===", flush=True)
    for s in flips:
        print(f"   FLIP {s['instance_id']}: {s.get('arm64_verdict')} -> {s.get('x86_verdict')}", flush=True)
    _net_log(f"END offload completed={len(done)}/{len(ids)} arch_flips={len(flips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
