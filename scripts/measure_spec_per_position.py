#!/usr/bin/env python3
"""Canonical direct greedy spec-decode per-position measurement.

This intentionally bypasses the SWE/agentic driver. At temperature 0 the agent
can give up before decoding, producing rows=0 and hiding the model behavior.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = Path("tests/fixtures/spec_probe_prompts.json")
DEFAULT_TREE_LCP = Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs/tree_path_lcp_max.jsonl")
DEFAULT_VLLM_LOG = Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.6-27b.log")
SPINE_A_PATH = [0, 2, 4, 6, 8]


class GuardError(RuntimeError):
    """Raised when a measurement confound guard fails."""


def validate_sampling(temperature: float, top_p: float) -> None:
    # Guard for the greedy-argmax confound: spec acceptance is exact argmax
    # equality, so non-greedy sampling is not this measurement.
    if float(temperature) != 0.0 or float(top_p) != 1.0:
        raise GuardError(
            "direct spec probe is locked to greedy sampling: "
            f"temperature={temperature!r}, top_p={top_p!r}; require 0 and 1"
        )


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    headers = {"Authorization": "Bearer EMPTY"}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_spec_metrics(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"drafts": 0.0, "accepted_total": 0.0, "pos": [0.0] * 16}
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            parsed["drafts"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            parsed["accepted_total"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            match = re.search(r'position="(\d+)"', line)
            if match:
                pos = int(match.group(1))
                if pos >= len(parsed["pos"]):
                    parsed["pos"].extend([0.0] * (pos + 1 - len(parsed["pos"])))
                parsed["pos"][pos] = float(line.rsplit(" ", 1)[1])
    return parsed


def _latest_container_id(log_path: Path = DEFAULT_VLLM_LOG) -> str | None:
    if not log_path.exists():
        return None
    found = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "[VLLM-CONTAINER] id=" in line:
            found = line.rsplit("id=", 1)[1].strip()
    return found


def _docker_env(container_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", container_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _live_vllm_cmdline() -> str:
    result = subprocess.run(
        ["ps", "-eo", "args"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        if "vllm serve" in line and "--served-model-name" in line:
            return line
    return ""


def detect_batch_invariant(endpoint: str, log_path: Path = DEFAULT_VLLM_LOG) -> dict[str, Any]:
    cmdline = _live_vllm_cmdline()
    has_flash = "--attention-backend FLASH_ATTN" in cmdline or "--attention-backend=FLASH_ATTN" in cmdline
    container_id = _latest_container_id(log_path)
    env = _docker_env(container_id) if container_id else []
    has_batch_invariant = "VLLM_BATCH_INVARIANT=1" in env
    return {
        "endpoint": endpoint,
        "container_id": container_id,
        "cmdline_has_flash_attn": has_flash,
        "env_has_vllm_batch_invariant": has_batch_invariant,
        "cmdline": cmdline,
    }


def validate_live_server(endpoint: str, guard: dict[str, Any]) -> None:
    # Guard for the missing-batch-invariance confound: greedy acceptance is
    # exact argmax equality, so GEMM drift between proposal/verify inflates
    # acc0 and collapses acceptance.
    if not guard.get("env_has_vllm_batch_invariant"):
        raise GuardError(
            "batch-invariance guard failed: live container env does not expose "
            "VLLM_BATCH_INVARIANT=1; refusing greedy acceptance measurement"
        )
    if not guard.get("cmdline_has_flash_attn"):
        raise GuardError(
            "batch-invariance guard failed: live vLLM command does not show "
            "--attention-backend FLASH_ATTN"
        )
    try:
        _http_json(endpoint.rstrip("/") + "/v1/models", timeout=5)
    except Exception as exc:  # pragma: no cover - exercised only against live server
        raise GuardError(f"endpoint is not a live direct vLLM server: {endpoint}: {exc}") from exc


def lcp_from_tree_record(record: dict[str, Any], path: list[int] | None = None) -> int:
    path = path or SPINE_A_PATH
    for score in record.get("path_scores") or []:
        if [int(x) for x in score.get("path", [])] == path:
            return int(score.get("lcp", 0))
    accepted = [int(x) for x in record.get("accepted_node_ids", [])]
    lcp = 0
    for got, want in zip(accepted, path):
        if got != want:
            break
        lcp += 1
    return lcp


def summarize_lcps(lcps: list[int]) -> dict[str, Any]:
    n = len(lcps)
    return {
        "n_events": n,
        "avg": sum(lcps) / n if n else None,
        "acc0": sum(v == 0 for v in lcps) / n if n else None,
        "full5": sum(v >= 5 for v in lcps) / n if n else None,
        "per_position": [sum(v >= k for v in lcps) / n if n else None for k in range(1, 6)],
        "counts": dict(sorted(Counter(lcps).items())),
    }


def summarize_e5_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    drafts = after["drafts"] - before["drafts"]
    pos = [after["pos"][i] - before["pos"][i] for i in range(5)]
    return {
        "n_events": drafts,
        "avg": sum(pos) / drafts if drafts else None,
        "acc0": 1.0 - (pos[0] / drafts) if drafts else None,
        "full5": pos[4] / drafts if drafts else None,
        "per_position": [value / drafts if drafts else None for value in pos],
        "metric_delta": {
            "drafts": drafts,
            "accepted_total": after["accepted_total"] - before["accepted_total"],
            "pos": pos,
        },
    }


def _completion(endpoint: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    start = time.time()
    try:
        data = _http_json(endpoint.rstrip("/") + "/v1/completions", payload, timeout=900)
        choice = data["choices"][0]
        return {
            "ok": True,
            "elapsed_s": round(time.time() - start, 3),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "text_prefix": (choice.get("text") or "")[:240],
        }
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return {"ok": False, "elapsed_s": round(time.time() - start, 3), "error": repr(exc)}


def _tree_lcps_between(trace_path: Path, start_ts: float, end_ts: float) -> list[int]:
    lcps: list[int] = []
    if not trace_path.exists():
        return lcps
    for raw in trace_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if start_ts <= float(row.get("ts", 0.0)) <= end_ts:
            lcps.append(lcp_from_tree_record(row))
    return lcps


def print_table(label: str, summary: dict[str, Any]) -> None:
    per = summary["per_position"]
    print(f"{label}: n={summary['n_events']} avg={summary['avg']:.3f} acc0={100 * summary['acc0']:.1f}% full5={100 * summary['full5']:.1f}%")
    print("  pos1  pos2  pos3  pos4  pos5")
    print("  " + "  ".join(f"{100 * p:5.1f}" for p in per))


def run_measure(args: argparse.Namespace) -> dict[str, Any]:
    validate_sampling(args.temperature, args.top_p)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise SystemExit(f"prompt file must be a non-empty JSON list of strings: {args.prompts}")

    guard = detect_batch_invariant(args.endpoint)
    if not args.no_live_guards:
        validate_live_server(args.endpoint, guard)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = parse_spec_metrics(_http_text(args.endpoint.rstrip("/") + "/metrics", timeout=5))
    start_ts = time.time()
    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_completion, args.endpoint, args.model, prompt, args.max_tokens, args.temperature, args.top_p): index
            for index, prompt in enumerate(prompts)
        }
        for future in cf.as_completed(futures):
            item = future.result()
            item["prompt_index"] = futures[future]
            results.append(item)
            print(
                f"done {item['prompt_index']} ok={item['ok']} "
                f"tokens={item.get('completion_tokens')} elapsed={item['elapsed_s']}",
                flush=True,
            )
    end_ts = time.time() + 2.0
    after = parse_spec_metrics(_http_text(args.endpoint.rstrip("/") + "/metrics", timeout=5))

    if args.mode == "e5":
        spec = summarize_e5_delta(before, after)
        source = "vllm Prometheus spec_decode accepted_tokens_per_pos delta"
    else:
        lcps = _tree_lcps_between(Path(args.tree_lcp_log), start_ts, end_ts)
        spec = summarize_lcps(lcps)
        source = "tree_path_lcp_max.jsonl path0_lcp/spine-A path_scores [0,2,4,6,8]"
        if spec["n_events"] == 0:
            raise SystemExit(
                "tree mode produced zero path0_lcp events; require verifier-independent "
                "tree_path_lcp_max.jsonl rows, not accepted_len"
            )

    if not all(r.get("ok") for r in results):
        raise SystemExit("one or more direct completion requests failed; refusing partial measurement")
    if not spec["n_events"]:
        raise SystemExit("measurement produced zero spec events; refusing rows=0 result")

    measurement = {
        "schema": "spec_per_position.v1",
        "mode": args.mode,
        "source": source,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompts": str(args.prompts),
        "prompt_sha256": __import__("hashlib").sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "request_count": len(prompts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "concurrency": args.concurrency,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "guard": guard,
        "metrics_before": before,
        "metrics_after": after,
        "spec": spec,
        "results": sorted(results, key=lambda r: r["prompt_index"]),
    }
    (out / "measurement.json").write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")
    print_table(args.mode, spec)
    return measurement


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["e5", "tree"], required=True)
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tree-lcp-log", default=str(DEFAULT_TREE_LCP))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--no-live-guards", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        run_measure(args)
        return 0
    except GuardError as exc:
        print(f"guard failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
