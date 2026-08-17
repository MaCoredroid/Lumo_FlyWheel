#!/usr/bin/env python3
"""Scan qwen-code traces for agent network egress that reached the outside world.

WHY THIS EXISTS. The FR14 no-net settings change removed `web_fetch` from the
agent's tool registry, and the model immediately routed around it with
``python -c "import urllib.request ..."`` inside ``run_shell_command`` --
fetching, among other things, the gold patch for the task it was solving
(results/fr14_nvfp4_port_20260816/agent_network_egress_finding.md).

qwen-code enforces ``WebFetch`` deny rules against *equivalent* shell commands,
which is why ``curl`` came back "denied by permission rules" -- but that detector
is a heuristic and does not recognise a python one-liner. Until the agent
container is given a real network boundary, the only honest position is: measure
it on the evidence, every time, and void the quality verdict of any task that
reached the network.

A task that fetched its own gold patch can manufacture a resolve. This gate is
what stops that from being discovered months later.

DELIBERATELY EVIDENCE-ONLY: it reads banked traces and never touches a run. It
is cheap enough to run over every runroot on disk.

    python3 scripts/fr14_agent_egress_scan.py <path-or-glob> [...] [--json out.json]

Exit 0 = no task reached the network. Exit 2 = at least one did.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from pathlib import Path

# Command shapes that can move bytes off-box. Deliberately broad: a false
# positive costs one manual look, a false negative costs a silent contaminated
# resolve.
NETWORK_COMMAND = re.compile(
    r"(urllib|requests\.|httpx|aiohttp|socket\.|curl\s|wget\s|nc\s|ssh\s"
    r"|git\s+clone|pip\s+install|https?://)",
    re.IGNORECASE,
)
# qwen-code's own refusal string for a call its permission rules blocked.
DENIED = "denied by permission rules"


def _content_items(event: dict) -> list[dict]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return [c for c in content if isinstance(c, dict)] if isinstance(content, list) else []


def scan_trace(path: Path) -> dict:
    """Pair every network-shaped shell call with its result and judge it."""
    pending: dict[str, str] = {}
    attempts: list[dict] = []
    denials: list[dict] = []
    excluded_tool_calls: list[str] = []

    for lineno, line in enumerate(path.open(errors="replace")):
        try:
            event = json.loads(line)
        except Exception:
            continue

        if event.get("type") == "result":
            for denial in event.get("permission_denials") or []:
                if isinstance(denial, dict):
                    denials.append(
                        {"tool_name": denial.get("tool_name"),
                         "tool_use_id": denial.get("tool_use_id")}
                    )

        for item in _content_items(event):
            if item.get("type") == "tool_use":
                name = item.get("name")
                if name in ("web_fetch", "web_search"):
                    excluded_tool_calls.append(f"{name}@{lineno}")
                if name == "run_shell_command":
                    command = (item.get("input") or {}).get("command", "")
                    if NETWORK_COMMAND.search(command):
                        pending[item.get("id")] = command
            elif item.get("type") == "tool_result":
                use_id = item.get("tool_use_id")
                if use_id not in pending:
                    continue
                rendered = json.dumps(item.get("content"))
                attempts.append({
                    "line": lineno,
                    "command": " ".join(pending.pop(use_id).split())[:200],
                    "reached_network": DENIED not in rendered,
                    "result_bytes": len(rendered),
                    "result_head": " ".join(rendered.split())[:160],
                })

    reached = [a for a in attempts if a["reached_network"]]
    return {
        "trace": str(path),
        "network_shell_attempts": len(attempts),
        "reached_network": len(reached),
        "permission_denials": len(denials),
        "excluded_tool_calls": excluded_tool_calls,
        "clean": not reached and not excluded_tool_calls,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="trace files, dirs, or globs")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary line"
    )
    args = parser.parse_args()

    traces: list[Path] = []
    for raw in args.paths:
        for hit in sorted(globmod.glob(raw, recursive=True)):
            p = Path(hit)
            if p.is_dir():
                traces.extend(sorted(p.rglob("qwen_trace.jsonl")))
            elif p.name == "qwen_trace.jsonl":
                traces.append(p)
    if not traces:
        print("no qwen_trace.jsonl found", file=sys.stderr)
        return 2

    reports = [scan_trace(t) for t in traces]
    dirty = [r for r in reports if not r["clean"]]

    for report in reports:
        if args.quiet and report["clean"]:
            continue
        mark = "CLEAN" if report["clean"] else "REACHED-NET"
        print(
            f"{mark:<12} attempts={report['network_shell_attempts']:<3} "
            f"reached={report['reached_network']:<3} "
            f"denials={report['permission_denials']:<3} "
            f"web_tool_calls={len(report['excluded_tool_calls']):<3} "
            f"{report['trace']}"
        )
        for attempt in report["attempts"]:
            if attempt["reached_network"]:
                print(f"                 line {attempt['line']}: {attempt['command'][:150]}")

    payload = {
        "schema": "fr14.agent_egress_scan.v1",
        "traces_scanned": len(reports),
        "traces_that_reached_the_network": len(dirty),
        "verdict": "CLEAN" if not dirty else "REACHED-NET",
        "reports": reports,
    }
    if args.json:
        args.json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(
        f"\n{len(reports)} traces scanned, {len(dirty)} reached the network "
        f"-> {payload['verdict']}"
    )
    return 0 if not dirty else 2


if __name__ == "__main__":
    raise SystemExit(main())
