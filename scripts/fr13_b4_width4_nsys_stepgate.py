#!/usr/bin/env python3
"""Step-gated Nsight capture controller for the B4 width-4 operating point.

DIAGNOSTIC. NOT CITABLE. This tool drives a profiler, not a measurement: the
capture it opens perturbs the arm it runs inside (CUPTI), so nothing it emits
may be used as acceptance evidence for any run class.

WHY A CONTROLLER EXISTS AT ALL
------------------------------
`results/fr13_b4_refill_citable_20260812/width4_window.md` §6 pins the capture
contract for this profile:

    Gate the profiler on the step counter, not on wall time -- wall-time gating
    would drift with the hydration lag.

The shipped nsys path (`scripts/fr13_fixed32_b1_nsys_profile.sh`) is wall-gated:
it hard-pins `--delay 1200 --duration 300`. Between an arm's admit and the
engine's first forward step for that task sits repo hydration -- 118 forward
steps / ~43 s on tail23 pass 0 -- and that lag is a property of the task draw,
not of the clock. A fixed delay therefore cannot name the step range it lands
on, which is exactly the failure §6 forbids.

So the launcher is put into deferred collection (`--start-later=true`, which is
DEFAULT-OFF and leaves the B1 path byte-identical) and this controller opens and
closes the session against the ABSOLUTE forward-step counter

    vllm:fr13_decode_forward_gpu_steps_total

which is the same counter the width-4 reduction uses to index the work census
(`width4_window.md` §1: "The bracket's own fr13_decode_forward_gpu_steps_total
then *indexes* the census range"). Gate and reduction share one coordinate.

WHY NOT TAIL THE WORK CENSUS INSTEAD
------------------------------------
The census is the other carrier of `forward_step_index`, but it is buffered
behind an explicit flush request/ack protocol
(`/logs/fr13_fixed32_flush_request.json` -> `..._ack.json`), so a host tail sees
an arbitrarily stale step index. The Prometheus counter is live. The census is
still used -- offline, after the arm ends -- to prove the capture's step range
carried the batch widths this class claims.

THE BRACKET IS AN INTERVAL, NOT A POINT
---------------------------------------
`nsys start` is not instantaneous and the engine keeps stepping across it, so
the exact step at which collection began is not observable. This tool refuses to
invent one. It scrapes the counter immediately BEFORE and immediately AFTER each
control call and publishes both, giving

    inner = [open_hi, close_lo]   steps the capture certainly contains
    outer = [open_lo, close_hi]   steps the capture is certainly contained by

The reduction prices kernels against `inner` and reports the ambiguity
`outer - inner` as the attribution's own error bar. A capture whose ambiguity is
a material fraction of its length is refused rather than reported.

HYDRATION IS EXCLUDED BY A STEP-BASED TEST, NOT BY A GUESS
----------------------------------------------------------
§6: "what to avoid -- starting the capture at arm start. The first steps of an
arm run at width 1-2 while the initial four tasks hydrate." Depth-4 admission
happens almost immediately (all four slots are submitted at once), so the
window's `census_first_forward_step` is 0 on all eight banked arms and a
depth-based gate would fire during hydration. The engine-side condition is
therefore tested directly: trailing events/step, computed from the counter pair
(drafts, steps) over a trailing step interval, must hold at or above
--min-events-per-step before the session is armed. On the banked arms the window
means are 3.567-3.678, and width 1-2 accounts for only ~141 of 5545 steps.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "fr13.b4_width4_nsys_capture.v1"

STEPS_COUNTER = "vllm:fr13_decode_forward_gpu_steps_total"
DRAFTS_COUNTER = "vllm:fr13_decode_forward_gpu_drafts_total"

# Every counter the reduction's phase split is built from. Scraped whole at each
# bracket edge so the split is recomputable offline from these files alone.
REQUIRED_COUNTERS = (
    STEPS_COUNTER,
    DRAFTS_COUNTER,
    "vllm:fr13_decode_forward_gpu_seconds_total",
    "vllm:fr13_drafter_gpu_seconds_total",
    "vllm:fr13_committer_gpu_seconds_total",
    "vllm:fr13_decode_step_wall_seconds_total",
    "vllm:fr13_decode_step_wall_steps_total",
    "vllm:fr13_decode_step_wall_drafts_total",
)


class GateError(RuntimeError):
    """The step-gated capture cannot be driven to a valid bracket."""


def _scrape(url: str, timeout_s: float) -> str:
    # A booting vLLM refuses, resets AND half-opens this port in turn, so every
    # transport failure must land in one bucket. urllib.error.URLError is an
    # OSError subclass, as are ConnectionResetError/ConnectionRefusedError and
    # socket.timeout; http.client raises several of its own. Catching OSError +
    # HTTPException covers the lot -- an earlier version caught only URLError
    # and died on a reset while the model was still loading.
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as handle:
            if handle.status != 200:
                raise GateError(f"metrics endpoint returned HTTP {handle.status}")
            return handle.read().decode("utf-8")
    except (OSError, http.client.HTTPException) as error:
        raise GateError(
            f"metrics endpoint unreachable ({type(error).__name__}): {error}"
        ) from error


def _scrape_retry(url: str, timeout_s: float, *, attempts: int = 12,
                  backoff_s: float = 5.0) -> str:
    """Scrape, tolerating transient transport failures.

    Used for BOTH the polling loops and the bracket edges. Retrying an edge is
    safe -- and strictly better than dying mid-capture -- because a slower edge
    only widens the measured open/close ambiguity, which is published and
    bounded, rather than silently biasing the bracket.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _scrape(url, timeout_s)
        except GateError as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(backoff_s)
    raise GateError(f"metrics scrape failed after {attempts} attempts: {last}")


def _counter(text: str, name: str) -> float:
    """Read one unlabelled Prometheus counter, refusing duplicates."""
    found: list[float] = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # These fr13 counters are emitted without labels.
        if line.startswith(name + " "):
            parts = line.split()
            if len(parts) != 2:
                raise GateError(f"malformed counter line for {name}: {line!r}")
            found.append(float(parts[1]))
    if len(found) != 1:
        raise GateError(f"expected exactly one {name} sample, found {len(found)}")
    return found[0]


def _counters(text: str) -> dict[str, float]:
    return {name: _counter(text, name) for name in REQUIRED_COUNTERS}


def _steps(text: str) -> int:
    value = _counter(text, STEPS_COUNTER)
    if value != int(value) or value < 0:
        raise GateError(f"{STEPS_COUNTER} is not a non-negative integer: {value}")
    return int(value)


def _save(directory: Path, name: str, text: str) -> str:
    path = directory / name
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def _nsys(
    *,
    nsys_bin: str,
    container_id: str,
    action: str,
    session: str,
    timeout_s: int,
) -> str:
    """Run `nsys start|stop` INSIDE the container that owns the session.

    The session is created by the nsys frontend running as the container's PID 1,
    so it lives in that container's namespace and is not addressable from the
    host.
    """
    command = [
        "docker",
        "exec",
        container_id,
        nsys_bin,
        action,
        f"--session={session}",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise GateError(
            f"nsys {action} failed rc={completed.returncode}: {output.strip()[:2000]}"
        )
    return output


def _session_state(
    *, nsys_bin: str, container_id: str, session: str, timeout_s: int
) -> str | None:
    completed = subprocess.run(
        ["docker", "exec", container_id, nsys_bin, "sessions", "list",
         "--output-format=json"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(rows, list):
        return None
    matches = [r for r in rows if isinstance(r, dict) and r.get("name") == session]
    if len(matches) != 1:
        return None
    state = matches[0].get("state")
    return state if isinstance(state, str) else None


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{stamp}] {message}", flush=True)


def _wait_for_endpoint(url: str, deadline: float, poll_s: float) -> None:
    _log(f"waiting for metrics endpoint {url}")
    while time.time() < deadline:
        try:
            text = _scrape(url, timeout_s=10.0)
        except GateError:
            time.sleep(poll_s)
            continue
        try:
            _steps(text)
        except GateError:
            # Endpoint is up but the fr13 counters are not registered yet.
            time.sleep(poll_s)
            continue
        _log("metrics endpoint is live and carrying the fr13 step counter")
        return
    raise GateError("timed out waiting for a live fr13 metrics endpoint")


def _wait_for_arm_condition(
    *,
    url: str,
    deadline: float,
    poll_s: float,
    start_step: int,
    min_events_per_step: float,
    trailing_steps: int,
) -> dict[str, Any]:
    """Block until the absolute step counter clears `start_step` AND the engine
    is demonstrably at width, measured over a trailing step interval."""
    _log(
        f"arming: need steps >= {start_step} and trailing events/step >= "
        f"{min_events_per_step} over {trailing_steps} steps"
    )
    history: list[tuple[int, float]] = []
    last_report = 0.0
    while time.time() < deadline:
        text = _scrape_retry(url, timeout_s=15.0)
        steps = _steps(text)
        drafts = _counter(text, DRAFTS_COUNTER)
        history.append((steps, drafts))
        # Keep only what is needed to span the trailing interval.
        while len(history) > 2 and history[-1][0] - history[1][0] >= trailing_steps:
            history.pop(0)

        trailing = None
        if history[-1][0] - history[0][0] >= trailing_steps:
            d_steps = history[-1][0] - history[0][0]
            d_drafts = history[-1][1] - history[0][1]
            if d_steps > 0:
                trailing = d_drafts / d_steps

        now = time.time()
        if now - last_report >= 60.0:
            last_report = now
            _log(
                f"  steps={steps} trailing_events_per_step="
                f"{'n/a' if trailing is None else f'{trailing:.3f}'}"
            )

        if steps >= start_step and trailing is not None:
            if trailing >= min_events_per_step:
                _log(
                    f"ARMED at steps={steps} trailing_events_per_step={trailing:.3f}"
                )
                return {
                    "armed_at_steps": steps,
                    "trailing_events_per_step": trailing,
                    "trailing_step_span": history[-1][0] - history[0][0],
                }
        time.sleep(poll_s)
    raise GateError("timed out waiting for the width-4 arm condition")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--nsys-bin", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--capture-steps", type=int, required=True)
    parser.add_argument("--min-events-per-step", type=float, default=3.4)
    parser.add_argument("--trailing-steps", type=int, default=200)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--arm-timeout-s", type=int, default=7200)
    parser.add_argument("--capture-timeout-s", type=int, default=3600)
    parser.add_argument("--exec-timeout-s", type=int, default=180)
    # The capture bracket is only meaningful if the control-call ambiguity is
    # small against it. Refuse rather than report a smeared bracket.
    parser.add_argument("--max-edge-ambiguity-steps", type=int, default=25)
    args = parser.parse_args()

    if args.capture_steps <= 0:
        raise GateError("--capture-steps must be positive")
    if args.start_step < 0:
        raise GateError("--start-step must be non-negative")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "citable": False,
        "acceptance_valid": False,
        "diagnostic_only": True,
        "session": args.session,
        "container_id": args.container_id,
        "requested": {
            "start_step": args.start_step,
            "capture_steps": args.capture_steps,
            "min_events_per_step": args.min_events_per_step,
            "trailing_steps": args.trailing_steps,
            "max_edge_ambiguity_steps": args.max_edge_ambiguity_steps,
        },
    }

    now = time.time()
    _wait_for_endpoint(args.metrics_url, now + args.arm_timeout_s, args.poll_s)

    state = _session_state(
        nsys_bin=args.nsys_bin,
        container_id=args.container_id,
        session=args.session,
        timeout_s=args.exec_timeout_s,
    )
    _log(f"pre-arm nsys session state: {state}")
    if state != "DelayedCollection":
        raise GateError(
            "session is not in DelayedCollection before arming; refusing to "
            f"drive a session in state {state!r} (is LUMO_NSYS_START_LATER=1?)"
        )
    record["session_state_before_start"] = state

    armed = _wait_for_arm_condition(
        url=args.metrics_url,
        deadline=time.time() + args.arm_timeout_s,
        poll_s=args.poll_s,
        start_step=args.start_step,
        min_events_per_step=args.min_events_per_step,
        trailing_steps=args.trailing_steps,
    )
    record["arm_condition"] = armed

    # ---- OPEN EDGE -------------------------------------------------------
    open_lo_text = _scrape_retry(args.metrics_url, timeout_s=15.0)
    open_lo = _steps(open_lo_text)
    record["metrics_open_lo_path"] = _save(
        out_dir, "metrics_capture_open_lo.txt", open_lo_text
    )
    record["counters_open_lo"] = _counters(open_lo_text)
    record["open_lo_steps"] = open_lo

    _log(f"nsys start (counter before call: {open_lo})")
    record["nsys_start_output"] = _nsys(
        nsys_bin=args.nsys_bin,
        container_id=args.container_id,
        action="start",
        session=args.session,
        timeout_s=args.exec_timeout_s,
    )

    open_hi_text = _scrape_retry(args.metrics_url, timeout_s=15.0)
    open_hi = _steps(open_hi_text)
    record["metrics_open_hi_path"] = _save(
        out_dir, "metrics_capture_open_hi.txt", open_hi_text
    )
    record["counters_open_hi"] = _counters(open_hi_text)
    record["open_hi_steps"] = open_hi
    _log(f"collection open; counter after call: {open_hi} (ambiguity {open_hi - open_lo})")

    state = _session_state(
        nsys_bin=args.nsys_bin,
        container_id=args.container_id,
        session=args.session,
        timeout_s=args.exec_timeout_s,
    )
    record["session_state_during_capture"] = state
    _log(f"session state during capture: {state}")
    if state != "Collection":
        raise GateError(f"session did not enter Collection; state={state!r}")

    # ---- HOLD ------------------------------------------------------------
    target = open_hi + args.capture_steps
    _log(f"holding capture until steps >= {target}")
    deadline = time.time() + args.capture_timeout_s
    last_report = 0.0
    while True:
        text = _scrape_retry(args.metrics_url, timeout_s=15.0)
        steps = _steps(text)
        if steps >= target:
            break
        if time.time() >= deadline:
            raise GateError(
                f"timed out holding capture: steps={steps} target={target}"
            )
        now = time.time()
        if now - last_report >= 60.0:
            last_report = now
            _log(f"  capture in progress: steps={steps}/{target}")
        time.sleep(args.poll_s)

    # ---- CLOSE EDGE ------------------------------------------------------
    close_lo_text = _scrape_retry(args.metrics_url, timeout_s=15.0)
    close_lo = _steps(close_lo_text)
    record["metrics_close_lo_path"] = _save(
        out_dir, "metrics_capture_close_lo.txt", close_lo_text
    )
    record["counters_close_lo"] = _counters(close_lo_text)
    record["close_lo_steps"] = close_lo

    _log(f"nsys stop (counter before call: {close_lo})")
    record["nsys_stop_output"] = _nsys(
        nsys_bin=args.nsys_bin,
        container_id=args.container_id,
        action="stop",
        session=args.session,
        timeout_s=max(args.exec_timeout_s, 600),
    )

    close_hi_text = _scrape_retry(args.metrics_url, timeout_s=15.0)
    close_hi = _steps(close_hi_text)
    record["metrics_close_hi_path"] = _save(
        out_dir, "metrics_capture_close_hi.txt", close_hi_text
    )
    record["counters_close_hi"] = _counters(close_hi_text)
    record["close_hi_steps"] = close_hi
    _log(
        f"collection closed; counter after call: {close_hi} "
        f"(ambiguity {close_hi - close_lo})"
    )

    record["session_state_after_stop"] = _session_state(
        nsys_bin=args.nsys_bin,
        container_id=args.container_id,
        session=args.session,
        timeout_s=args.exec_timeout_s,
    )

    # ---- BRACKET -------------------------------------------------------
    open_ambiguity = open_hi - open_lo
    close_ambiguity = close_hi - close_lo
    inner_steps = close_lo - open_hi
    outer_steps = close_hi - open_lo
    record["bracket"] = {
        "definition": (
            "inner = [open_hi, close_lo] steps the capture certainly contains; "
            "outer = [open_lo, close_hi] steps the capture is certainly "
            "contained by. The true collection boundaries are unobservable "
            "because the engine steps across each nsys control call."
        ),
        "inner_first_step": open_hi,
        "inner_last_step_exclusive": close_lo,
        "inner_steps": inner_steps,
        "outer_first_step": open_lo,
        "outer_last_step_exclusive": close_hi,
        "outer_steps": outer_steps,
        "open_edge_ambiguity_steps": open_ambiguity,
        "close_edge_ambiguity_steps": close_ambiguity,
        "total_ambiguity_steps": outer_steps - inner_steps,
        "ambiguity_fraction_of_inner": (
            (outer_steps - inner_steps) / inner_steps if inner_steps > 0 else None
        ),
    }

    if inner_steps <= 0:
        raise GateError(f"capture inner bracket is empty: {inner_steps} steps")
    for label, value in (
        ("open", open_ambiguity),
        ("close", close_ambiguity),
    ):
        if value < 0:
            raise GateError(f"{label} edge counter went backwards: {value}")
        if value > args.max_edge_ambiguity_steps:
            raise GateError(
                f"{label} edge ambiguity {value} steps exceeds the "
                f"{args.max_edge_ambiguity_steps}-step limit; the bracket is "
                "too smeared to attribute against"
            )

    record["ok"] = True
    manifest = out_dir / "capture_manifest.json"
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, manifest)
    _log(f"capture manifest written: {manifest}")
    _log(
        f"INNER bracket steps [{open_hi}, {close_lo}) = {inner_steps} steps; "
        f"ambiguity {outer_steps - inner_steps} steps"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
