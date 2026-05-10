#!/usr/bin/env python3
"""Round 2 activation checker — verify prelaunch patches landed.

Runs against a live vLLM container (default
``lumo-vllm-track-b-suffix``) and asserts every sentinel from the
T1 + T3 prelaunch chain is present in the patched site-packages
files. Prints a pass/fail line per sentinel + a structured JSON
summary, exits non-zero if any check fails.

Useful as the very first thing an operator runs after a vLLM
relaunch through ``ModelServer``: confirms the prelaunch hook
actually fired (it's silent on success today, so a missing patch
is otherwise invisible until the next sweep produces unexpected
numbers).

Sentinels checked:

- vLLM forced tool_choice parser fix (Track B 2026-05-08)
- T1 session-scoping wrapper (vllm/v1/spec_decode/suffix_decoding.py)
- T3 phase 2 oracle middleware install hook (api_server.py +
  presence of vllm/v1/spec_decode/lumo_oracle_registry.py)
- T3 phase 3 composite drafting + schema-aware drafter module
  (suffix_decoding.py sentinel + presence of
  vllm/v1/spec_decode/lumo_schema_aware_drafter.py)

The presence-of-file checks are the strongest signal -- the
sentinel-in-source checks catch in-place patches that may have
fired but failed to drop the supporting modules.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import PurePosixPath


CHECKS: list[dict] = [
    {
        "name": "forced_tool_choice_parser_patch",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py",
        "sentinel": "Local patch (Lumo Track B 2026-05-08)",
    },
    {
        "name": "t1_session_scoping_wrapper",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py",
        "sentinel": "T1_SESSION_SCOPING_APPLIED",
    },
    {
        "name": "t3_phase2_oracle_middleware_install_hook",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py",
        "sentinel": "T3_ORACLE_MIDDLEWARE_APPLIED",
    },
    {
        "name": "t3_phase2_oracle_registry_module_present",
        "kind": "exists",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/lumo_oracle_registry.py",
    },
    {
        "name": "t3_phase3_composite_drafting_patch",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py",
        "sentinel": "T3_COMPOSITE_DRAFTING_APPLIED",
    },
    {
        "name": "t3_phase3_schema_aware_drafter_module_present",
        "kind": "exists",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/lumo_schema_aware_drafter.py",
    },
    {
        "name": "t2_t4_composite_drafting_patch",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py",
        "sentinel": "T2_T4_COMPOSITE_DRAFTING_APPLIED",
    },
    {
        "name": "t4_plan_structure_drafter_module_present",
        "kind": "exists",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/lumo_plan_structure_drafter.py",
    },
    {
        "name": "runtime_speculative_config_suffix",
        "kind": "runtime_log",
        # vLLM's EngineCore initializer logs the SpeculativeConfig at
        # startup. If the launcher came up with a TunedConfigBundle that
        # had an empty spec_decode dict, this line will be absent (and
        # 'speculative_config=None' will be present instead). Catches
        # the entire-stack-silently-broken regression where every patch
        # sentinel still PASSes but the engine never speculates.
        "marker": "method='suffix'",
        "absent_marker": "speculative_config=None",
    },
    {
        "name": "runtime_track_b_disabled_helper",
        "kind": "sentinel",
        "path": "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/suffix_decoding.py",
        "sentinel": "def _lumo_track_b_disabled",
    },
]


def _docker_exec(container: str, command: list[str], timeout: int = 30) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["docker", "exec", container] + command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _check_sentinel(container: str, path: str, sentinel: str) -> tuple[bool, str]:
    rc, stdout, stderr = _docker_exec(
        container,
        ["bash", "-lc", f"grep -c {shell_escape(sentinel)} {shell_escape(path)} || true"],
    )
    if rc != 0 and not stdout:
        return False, f"docker exec failed: rc={rc} stderr={stderr.strip()[:200]}"
    count_line = stdout.strip().splitlines()[-1] if stdout.strip() else "0"
    try:
        count = int(count_line)
    except ValueError:
        return False, f"unparseable grep output: {count_line!r}"
    if count <= 0:
        return False, f"sentinel {sentinel!r} not found in {path}"
    return True, f"sentinel found ({count} occurrence{'s' if count > 1 else ''})"


def _check_exists(container: str, path: str) -> tuple[bool, str]:
    rc, stdout, _ = _docker_exec(
        container,
        ["bash", "-lc", f"test -f {shell_escape(path)} && echo OK || echo MISSING"],
    )
    last = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if last == "OK":
        return True, "file present"
    return False, f"file missing at {path}"


def _check_runtime_log(container: str, marker: str, absent_marker: str | None) -> tuple[bool, str]:
    """Check that the engine startup log contains ``marker`` (and, if
    given, does NOT contain ``absent_marker``). Used to gate runtime
    config that source-file sentinels can't observe — e.g. the
    presence of ``method='suffix'`` in the EngineCore init line."""

    proc = subprocess.run(
        ["docker", "logs", container],
        capture_output=True,
        text=True,
        timeout=30,
    )
    log_text = (proc.stdout or "") + (proc.stderr or "")
    has_marker = marker in log_text
    has_absent = bool(absent_marker) and absent_marker in log_text
    if has_marker and not has_absent:
        return True, f"runtime marker {marker!r} present"
    if has_absent:
        return False, f"runtime regression: {absent_marker!r} present in startup log"
    return False, f"runtime marker {marker!r} not found in startup log"


def shell_escape(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def run_checks(container: str) -> tuple[bool, list[dict]]:
    results: list[dict] = []
    all_pass = True
    for check in CHECKS:
        if check["kind"] == "sentinel":
            ok, detail = _check_sentinel(container, check["path"], check["sentinel"])
        elif check["kind"] == "exists":
            ok, detail = _check_exists(container, check["path"])
        elif check["kind"] == "runtime_log":
            ok, detail = _check_runtime_log(
                container, check["marker"], check.get("absent_marker")
            )
        else:
            ok, detail = False, f"unknown check kind: {check['kind']}"
        results.append(
            {
                "name": check["name"],
                "kind": check["kind"],
                "path": check.get("path"),
                "sentinel": check.get("sentinel"),
                "marker": check.get("marker"),
                "absent_marker": check.get("absent_marker"),
                "passed": ok,
                "detail": detail,
            }
        )
        if not ok:
            all_pass = False
    return all_pass, results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Round 2 prelaunch patches landed in a live "
            "vLLM container. Run after a ModelServer relaunch."
        )
    )
    parser.add_argument(
        "--container",
        default="lumo-vllm-track-b-suffix",
        help="docker container name (default: lumo-vllm-track-b-suffix)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON output path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("docker") is None:
        print("docker not found on PATH", file=sys.stderr)
        return 3
    try:
        all_pass, results = run_checks(args.container)
    except subprocess.TimeoutExpired:
        print(f"timed out checking container {args.container!r}", file=sys.stderr)
        return 4
    print(f"# Round 2 activation check for {args.container}")
    for r in results:
        marker = "PASS" if r["passed"] else "FAIL"
        print(f"  [{marker}] {r['name']}: {r['detail']}")
    payload = {
        "schema": "lumo.track_b.round2_activation.v1",
        "container": args.container,
        "all_passed": all_pass,
        "checks": results,
    }
    if args.output:
        out_path = PurePosixPath(args.output)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if not all_pass:
        print("\nOne or more activation checks failed. Triggers:", file=sys.stderr)
        print("  - vLLM was not relaunched after the patches landed in main.", file=sys.stderr)
        print("  - The prelaunch hook crashed before the patches applied.", file=sys.stderr)
        print("  - The container in --container is not the live vLLM.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
