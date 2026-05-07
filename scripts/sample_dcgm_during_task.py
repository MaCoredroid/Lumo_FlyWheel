#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NvmlSampler:
    def __init__(self, gpu_index: int) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("pynvml is required for live GPU sampling") from exc
        self._nvml = pynvml
        self._nvml.nvmlInit()
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._gpu_index = gpu_index

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

    def sample(self) -> dict[str, Any]:
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        try:
            power_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            power_w = None
        return {
            "ts": _now_iso(),
            "gpu": self._gpu_index,
            "telemetry_source": "nvml",
            "profile_fields_available": False,
            "dram_active_pct": None,
            "sm_active_pct": None,
            "sm_occupancy_pct": None,
            "pipe_tensor_active_pct": None,
            "pipe_fp16_active_pct": None,
            "gpu_util_pct": float(util.gpu),
            "mem_copy_util_pct": float(util.memory),
            "power_w": power_w,
        }


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    sampler = NvmlSampler(args.gpu)
    started = time.monotonic()
    samples = 0
    try:
        with out_path.open("a", encoding="utf-8") as handle:
            while not stop:
                handle.write(json.dumps(sampler.sample(), separators=(",", ":")) + "\n")
                samples += 1
                if samples % args.flush_every == 0:
                    handle.flush()
                if args.duration_s is not None and time.monotonic() - started >= args.duration_s:
                    break
                time.sleep(args.interval_s)
            handle.flush()
    finally:
        sampler.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample GPU telemetry while one Track B E2E task runs.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to sample.")
    parser.add_argument("--interval-s", type=float, default=0.01, help="Sampling interval; 0.01 is 100 Hz.")
    parser.add_argument("--duration-s", type=float, default=None, help="Optional maximum duration.")
    parser.add_argument("--flush-every", type=int, default=100, help="Flush every N samples.")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"sample_dcgm_during_task.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
