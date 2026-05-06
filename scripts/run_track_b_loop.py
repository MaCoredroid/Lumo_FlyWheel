#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
_PYTHONPATH_PARTS = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
if str(SRC_ROOT) not in _PYTHONPATH_PARTS:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC_ROOT), *_PYTHONPATH_PARTS])

from lumo_flywheel_serving.kernel_activation import resolve_kernel_runtime_activation  # noqa: E402
from lumo_flywheel_serving.model_server import ModelServer  # noqa: E402
from lumo_flywheel_serving.registry import ModelConfig, load_registry  # noqa: E402
from lumo_flywheel_serving.tuned_config import (  # noqa: E402
    StructuredValidationError,
    compute_workload_distribution_id,
    default_weight_version_id,
    make_tuned_config_bundle,
)


_SUPPORTED_VLLM_CONFIG_FIELDS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "max_model_len",
    "kv_cache_dtype",
}
_SUPPORTED_SPEC_DECODE_METHODS = {"ngram"}
_SUPPORTED_SPEC_DECODE_FIELDS = {
    "method",
    "num_speculative_tokens",
    "prompt_lookup_min",
    "prompt_lookup_max",
}
_SUPPORTED_KERNEL_SELECTION_FIELDS = {
    "attention_backend",
    "deltanet_kernel",
    "fp8_gemm_kernel",
    "torch_compile_mode",
    "cuda_graph_capture",
}
_SURFACE_HISTORY_LIMIT = 12


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_tsv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _candidate_ids(round_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in (round_dir / "candidates").glob("[0-9][0-9][0-9]"):
        if path.is_dir():
            ids.append(int(path.name))
    return sorted(ids)


def _next_candidate_id(round_dir: Path) -> str:
    ids = _candidate_ids(round_dir)
    return f"{(max(ids) + 1) if ids else 0:03d}"


def _render_agent_prompt(round_dir: Path, candidate_dir: Path, candidate_id: str) -> str:
    spec = _load_yaml(round_dir / "round_spec.yaml")
    strategy = (round_dir / "strategy_brief.md").read_text(encoding="utf-8")
    prior = (round_dir / "prior_cutlass_memory.md").read_text(encoding="utf-8")
    audit = _load_json(round_dir / "completion_audit.json") or {}
    quality_history = ""
    quality_history_path = round_dir / "quality_gate_history.tsv"
    if quality_history_path.is_file():
        quality_history = "\n".join(quality_history_path.read_text(encoding="utf-8").splitlines()[-12:])
    branch_log = _load_json(round_dir / "branch_log.json")
    surface_history = _candidate_surface_history(round_dir)
    exhausted_surface_brief = _render_exhausted_surface_brief(surface_history)
    baseline_tps = float(spec.get("baseline_decode_tps") or 7.5)
    previous_best_tps = _previous_best_decode_tps(round_dir, baseline_tps=baseline_tps)
    incremental_multiplier = float(
        spec.get("success_criteria", {}).get("candidate_acceptance_incremental_speedup_at_least", 1.2)
    )
    candidate_accept_tps = previous_best_tps * incremental_multiplier
    return "\n".join(
        [
            "# Track B Auto-Research Candidate Authoring",
            "",
            "You are a fresh implementation worker inside a Karpathy-style auto-research loop.",
            "The controller owns measurement, gates, keep/discard, and ledgers. Your job is to author exactly one candidate artifact.",
            "",
            "## Hard Rules",
            "",
            f"- Candidate id: `{candidate_id}`",
            f"- Candidate directory: `{candidate_dir}`",
            "- Write only inside that candidate directory.",
            "- Do not edit source files, tests, quality fixtures, prior memory, or round ledgers.",
            "- Do not run expensive live benchmarks; the controller runs gates after you exit.",
            "- Preserve target model weights and sampling behavior.",
            "- Do not propose measurement/accounting/workload-shape changes. The controller must keep the fixed CUTLASS-style first-five real-workload gate.",
            "- Build on prior CUTLASS negative memory; do not propose another tile/schedule/stage mutation unless your config changes the available serving surface.",
            "- Do not repeat an exact serving surface already measured in this round; the controller may reject duplicate surfaces before benchmarking.",
            "",
            "## Required Files",
            "",
            "Write these files before exiting:",
            "",
            "1. `candidate_analysis.md` with these bullets:",
            "   - speed_thesis",
            "   - expected_affected_counter",
            "   - quality_risk",
            "   - why_not_prior_failure",
            "",
            "2. `serve_config.yaml` with one of these supported controller surfaces:",
            "   - `request_shaping.target_concurrency: <1-8>` for batching/concurrency experiments",
            "   - `prefix_cache` settings for prefix-cache experiments",
            "   - `vllm_config` runtime overrides for max_num_seqs (1-64), max_num_batched_tokens (1-16384), enable_chunked_prefill (bool), enable_prefix_caching (bool), gpu_memory_utilization (0.0-0.95), max_model_len (1-131072), or kv_cache_dtype (`fp8_e5m2` or `auto` only)",
            "   - `spec_decode` settings for vLLM ngram speculative decoding: method `ngram`, num_speculative_tokens 1-8, prompt_lookup_min 1-16, prompt_lookup_max 1-64",
            "   - `kernel_selection` settings for repo-owned vLLM launch choices: attention_backend (`flashinfer`, `triton`, `flash-attn-3`, `flash-attn-4`, or `vllm-default`), fp8_gemm_kernel (`cublas` or `cutlass`), torch_compile_mode (`default`, `reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs`), cuda_graph_capture (`on` or `off`), deltanet_kernel (`triton-chunked-delta-v2`)",
            "",
            "3. Optional `notes.md` with any blocker or measurement caveat.",
            "",
            "## Current Objective",
            "",
            f"- Baseline decode: `{spec.get('baseline_decode_tps')}` tok/s",
            f"- Final target decode: `{spec.get('target_decode_tps')}` tok/s",
            f"- Candidate acceptance gate this iteration: `{candidate_accept_tps:.3f}` tok/s (`{incremental_multiplier:.2f}x` over previous best `{previous_best_tps:.3f}` tok/s)",
            "- Speed gate: real vLLM workload window; 5 completions per task, first cold completion discarded, next 4 warm completions counted.",
            "- The official speed metric is decode-time warm TPS from `throughput.json`; wall-clock aggregate throughput from concurrent requests is diagnostic only.",
            f"- Best audit so far: `{audit.get('best_decode_tps')}` tok/s",
            "",
            "## Recent Controller Outcomes",
            "",
            "Exhausted serving surfaces:",
            "",
            "```text",
            exhausted_surface_brief,
            "```",
            "",
            "Quality gate history tail:",
            "",
            "```tsv",
            quality_history,
            "```",
            "",
            "Branch log summary:",
            "",
            "```json",
            json.dumps(branch_log, indent=2, sort_keys=True)[:6000] if branch_log is not None else "[]",
            "```",
            "",
            "## Strategy Brief",
            "",
            strategy,
            "",
            "## Prior CUTLASS Memory",
            "",
            prior,
        ]
    )


def _canonical_surface(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_surface(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_surface(item) for item in value]
    return value


def _surface_payload(config: dict[str, Any]) -> dict[str, Any]:
    surface: dict[str, Any] = {}
    if "request_shaping" in config:
        surface["request_shaping"] = config["request_shaping"]
    if "prefix_cache" in config:
        surface["prefix_cache"] = config["prefix_cache"]
    if "vllm_config" in config:
        vllm_config = config["vllm_config"]
        if isinstance(vllm_config, dict):
            effective_vllm_config = dict(vllm_config)
            effective_vllm_config.setdefault("enable_prefix_caching", True)
            effective_vllm_config.setdefault("enable_chunked_prefill", True)
            surface["vllm_config"] = effective_vllm_config
        else:
            surface["vllm_config"] = vllm_config
    if "spec_decode" in config:
        spec_decode = config["spec_decode"]
        if isinstance(spec_decode, dict):
            effective_spec_decode = dict(spec_decode)
            effective_spec_decode.setdefault("method", "ngram")
            effective_spec_decode.setdefault("num_speculative_tokens", 4)
            effective_spec_decode.setdefault("prompt_lookup_min", 2)
            effective_spec_decode.setdefault("prompt_lookup_max", 6)
            surface["spec_decode"] = effective_spec_decode
        else:
            surface["spec_decode"] = spec_decode
    if "kernel_selection" in config:
        surface["kernel_selection"] = config["kernel_selection"]
    return surface


def _surface_signature(config: dict[str, Any]) -> str:
    surface = _canonical_surface(_surface_payload(config))
    if not surface:
        return "unsupported:{}"
    return json.dumps(surface, sort_keys=True, separators=(",", ":"))


def _candidate_surface_history(round_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_path in sorted((round_dir / "candidates").glob("[0-9][0-9][0-9]")):
        if not candidate_path.is_dir():
            continue
        config, error = _load_serve_config(candidate_path)
        if error is not None or config is None:
            continue
        result = _load_json(candidate_path / "controller_result.json") or {}
        throughput = _load_json(candidate_path / "throughput.json") or {}
        decode_tps = result.get("decode_tps") or throughput.get("warm_decode_tps") or throughput.get("decode_tps")
        rows.append(
            {
                "candidate_id": candidate_path.name,
                "signature": _surface_signature(config),
                "status": result.get("status"),
                "reason": result.get("reason"),
                "decode_tps": decode_tps,
            }
        )
    return rows


def _render_exhausted_surface_brief(surface_history: list[dict[str, Any]]) -> str:
    if not surface_history:
        return "No prior serving surfaces measured in this round."
    lines: list[str] = []
    for row in surface_history[-_SURFACE_HISTORY_LIMIT:]:
        decode_tps = row.get("decode_tps")
        rendered_tps = "n/a" if decode_tps is None else f"{float(decode_tps):.6f} tok/s"
        lines.append(
            f"{row['candidate_id']}: {row['status'] or 'unknown'} "
            f"{rendered_tps} {row.get('reason') or ''} surface={row['signature']}"
        )
    if _request_shaping_only_exhausted(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: request_shaping-only candidates are exhausted; "
            "prefer a vllm_config runtime candidate that changes actual vLLM launch capacity."
        )
    if _runtime_capacity_family_flat(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: tested runtime-capacity variants are flat at baseline-level "
            "throughput; avoid another candidate that only changes max_num_seqs, "
            "max_num_batched_tokens, or gpu_memory_utilization."
        )
    if _has_spec_decode_b1_failure_after_speed(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: a spec_decode candidate cleared speed preflight but failed "
            "B-1 equivalence with empty or truncated concurrent outputs; the next candidate "
            "must explicitly reduce that quality risk while preserving the speculative-decode "
            "speed gain."
        )
    if _has_failed_spec_decode_measurement(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: ngram spec_decode launched but failed real-workload "
            "measurement on a broader shape; avoid retrying aggressive ngram settings "
            "without a narrower lookup window or captured server-stack evidence."
        )
    if _has_valid_rejected_spec_decode(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: a stable ngram spec_decode candidate produced the best "
            "valid measurement but still missed preflight; continue only with a distinct "
            "ngram spec_decode shape, not another flat launch-capacity-only variant."
        )
    if _spec_decode_three_min_two_family_exhausted(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: the local ngram family with num_speculative_tokens=3 "
            "and prompt_lookup_min=2 is exhausted: max=8 was fast but failed B-1, "
            "max=6 was stable but far below preflight, and max=4 crashed the warm "
            "measurement. Do not spend the next candidate on max-only interpolation "
            "inside this family; move to a different speculative-depth/minimum pair "
            "or a different supported serving surface."
        )
    if _spec_decode_two_token_family_plateaued(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: the 2-token ngram family has plateaued below the "
            "post-best acceptance gate: lookup 2-16 and 2-8 were fast-but-insufficient, "
            "while lookup 3-16 fell back to baseline. Do not spend the next candidate "
            "on 2-token ngram lookup-window interpolation; move to a different "
            "serving surface such as kernel_selection or an evidence-backed nonlocal "
            "spec_decode shape."
        )
    if _has_unstable_high_depth_min_two_spec_decode(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: high-depth ngram shapes with prompt_lookup_min=2 are "
            "unstable or flat in this workload; the next candidate should not spend "
            "another attempt on num_speculative_tokens>=4 with min=2. Prefer the "
            "unmeasured kernel_selection surface before more speculative-depth search."
        )
    if _has_duplicate_deltanet_default_kernel_selection(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: kernel_selection.deltanet_kernel=triton-chunked-delta-v2 "
            "is baseline-equivalent for this model and has already been rejected as a "
            "duplicate serving surface. Do not retry that axis; choose a measured-distinct "
            "kernel_selection axis or return to a genuinely new speculative-decode shape."
        )
    if _has_duplicate_default_runtime_config_surface(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: runtime configs made only of default prefix/chunked "
            "flags and kv_cache_dtype=fp8_e5m2 have already been rejected as duplicate "
            "serving surfaces. Do not retry default-runtime bookkeeping knobs; use a "
            "measured-distinct launch setting or a new nonlocal speculative-decode shape."
        )
    if _has_any_spec_decode(surface_history) and not _has_any_kernel_selection(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: kernel_selection is now supported by the Track B "
            "controller and is unmeasured in this round. Prefer a kernel_selection "
            "candidate over another ngram-only candidate unless there is direct new "
            "evidence for the ngram shape."
        )
    elif not _has_any_spec_decode(surface_history):
        lines.append("")
        lines.append(
            "Controller guidance: vLLM ngram spec_decode is supported and unmeasured in this round; "
            "prefer a spec_decode candidate before more launch-shape-only variants."
        )
    return "\n".join(lines)


def _has_any_spec_decode(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if "spec_decode" in signature:
            return True
    return False


def _has_any_kernel_selection(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if "kernel_selection" in signature:
            return True
    return False


def _has_duplicate_deltanet_default_kernel_selection(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        kernel_selection = signature.get("kernel_selection")
        if not isinstance(kernel_selection, dict):
            continue
        if kernel_selection.get("deltanet_kernel") != "triton-chunked-delta-v2":
            continue
        if row.get("reason") == "duplicate_serving_surface":
            return True
    return False


def _has_duplicate_default_runtime_config_surface(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        vllm_config = signature.get("vllm_config")
        if not isinstance(vllm_config, dict):
            continue
        if row.get("reason") != "duplicate_serving_surface":
            continue
        if vllm_config.get("enable_prefix_caching") is True and vllm_config.get("enable_chunked_prefill") is True:
            return True
        if vllm_config.get("kv_cache_dtype") == "fp8_e5m2":
            return True
    return False


def _has_failed_spec_decode_measurement(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if "spec_decode" not in signature:
            continue
        if row.get("status") == "rejected" and row.get("reason") == "throughput_measure_failed":
            return True
    return False


def _has_valid_rejected_spec_decode(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if "spec_decode" not in signature:
            continue
        if row.get("status") == "rejected" and row.get("decode_tps") is not None:
            return True
    return False


def _has_spec_decode_b1_failure_after_speed(surface_history: list[dict[str, Any]]) -> bool:
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if "spec_decode" not in signature:
            continue
        if row.get("status") == "rejected" and row.get("reason") == "b1_equivalence_failed":
            if row.get("decode_tps") is not None:
                return True
    return False


def _spec_decode_three_min_two_family_exhausted(surface_history: list[dict[str, Any]]) -> bool:
    outcomes_by_max: dict[int, set[str]] = {}
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        spec_decode = signature.get("spec_decode")
        if not isinstance(spec_decode, dict):
            continue
        if spec_decode.get("method") != "ngram":
            continue
        if spec_decode.get("num_speculative_tokens") != 3 or spec_decode.get("prompt_lookup_min") != 2:
            continue
        try:
            prompt_lookup_max = int(spec_decode.get("prompt_lookup_max"))
        except (TypeError, ValueError):
            continue
        reason = str(row.get("reason") or "")
        if row.get("decode_tps") is not None:
            reason = f"{reason}:measured"
        outcomes_by_max.setdefault(prompt_lookup_max, set()).add(reason)
    return (
        any(reason.startswith("b1_equivalence_failed") for reason in outcomes_by_max.get(8, set()))
        and any(reason.startswith("speed_below_candidate_acceptance") for reason in outcomes_by_max.get(6, set()))
        and any(reason.startswith("throughput_measure_failed") for reason in outcomes_by_max.get(4, set()))
    )


def _spec_decode_two_token_family_plateaued(surface_history: list[dict[str, Any]]) -> bool:
    measured_rejections: set[tuple[int, int]] = set()
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        spec_decode = signature.get("spec_decode")
        if not isinstance(spec_decode, dict):
            continue
        if spec_decode.get("method") != "ngram" or spec_decode.get("num_speculative_tokens") != 2:
            continue
        if row.get("status") != "rejected" or row.get("reason") != "speed_below_candidate_acceptance":
            continue
        if row.get("decode_tps") is None:
            continue
        try:
            prompt_lookup_min = int(spec_decode.get("prompt_lookup_min"))
            prompt_lookup_max = int(spec_decode.get("prompt_lookup_max"))
        except (TypeError, ValueError):
            continue
        measured_rejections.add((prompt_lookup_min, prompt_lookup_max))
    return {(2, 16), (3, 16), (2, 8)}.issubset(measured_rejections)


def _has_unstable_high_depth_min_two_spec_decode(surface_history: list[dict[str, Any]]) -> bool:
    high_depth_outcomes = 0
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        spec_decode = signature.get("spec_decode")
        if not isinstance(spec_decode, dict):
            continue
        if spec_decode.get("method") != "ngram":
            continue
        try:
            num_speculative_tokens = int(spec_decode.get("num_speculative_tokens"))
            prompt_lookup_min = int(spec_decode.get("prompt_lookup_min"))
        except (TypeError, ValueError):
            continue
        if num_speculative_tokens < 4 or prompt_lookup_min != 2:
            continue
        reason = str(row.get("reason") or "")
        if reason == "throughput_measure_failed":
            high_depth_outcomes += 1
        elif row.get("decode_tps") is not None and float(row["decode_tps"]) < 8.0:
            high_depth_outcomes += 1
    return high_depth_outcomes >= 2


def _runtime_capacity_family_flat(surface_history: list[dict[str, Any]]) -> bool:
    flat_rejections = 0
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        vllm_config = signature.get("vllm_config")
        if not isinstance(vllm_config, dict) or row.get("status") != "rejected":
            continue
        changed_capacity = any(
            key in vllm_config
            for key in ("max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization")
        )
        decode_tps = row.get("decode_tps")
        if not changed_capacity or decode_tps is None:
            continue
        if float(decode_tps) < 8.0:
            flat_rejections += 1
    return flat_rejections >= 3


def _request_shaping_only_exhausted(surface_history: list[dict[str, Any]]) -> bool:
    request_only_rejections = 0
    seen_concurrency: set[int] = set()
    for row in surface_history:
        try:
            signature = json.loads(row["signature"])
        except (TypeError, json.JSONDecodeError):
            continue
        if set(signature) != {"request_shaping"}:
            continue
        target = signature["request_shaping"].get("target_concurrency")
        if not isinstance(target, int):
            continue
        seen_concurrency.add(target)
        if row.get("status") == "rejected":
            request_only_rejections += 1
    return request_only_rejections >= 4 and len(seen_concurrency) >= 3


def _has_prior_surface_signature(round_dir: Path, signature: str, *, current_candidate_id: str) -> bool:
    for row in _candidate_surface_history(round_dir):
        if row["candidate_id"] == current_candidate_id:
            continue
        if row["signature"] == signature:
            return True
    return False


def _spawn_codex(round_dir: Path, candidate_dir: Path, prompt: str, timeout_s: int) -> dict[str, Any]:
    last_message = candidate_dir / "agent_last_message.txt"
    transcript = candidate_dir / "agent_session.jsonl"
    prompt_path = candidate_dir / "iteration_brief.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    argv = [
        "codex",
        "-c",
        'model="gpt-5.5"',
        "-c",
        'model_reasoning_effort="high"',
        "exec",
        "--cd",
        str(round_dir),
        "--json",
        "--output-last-message",
        str(last_message),
        "--skip-git-repo-check",
        "-",
    ]
    started = time.monotonic()
    with tempfile_file() as stdout_file, tempfile_file() as stderr_file:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"agent_binary_missing: {exc}"}
        assert proc.stdin is not None
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
        while proc.poll() is None:
            if timeout_s > 0 and time.monotonic() - started >= timeout_s:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
            time.sleep(1.0)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_bytes = stdout_file.read()
        stderr_bytes = stderr_file.read()
    transcript.write_bytes(stdout_bytes)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"agent_exit_{proc.returncode}: {stderr_bytes.decode('utf-8', errors='replace')[:4000]}",
            "transcript": str(transcript),
        }
    return {"ok": True, "transcript": str(transcript), "last_message": str(last_message)}


def _restore_external_agent_edits(candidate_dir: Path) -> dict[str, Any]:
    candidate_rel = candidate_dir.resolve().relative_to(REPO_ROOT).as_posix()
    names = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if names.returncode != 0:
        return {"ok": False, "reason": "git_diff_name_only_failed", "detail": names.stderr[-2000:]}
    paths = [
        line.strip()
        for line in names.stdout.splitlines()
        if line.strip() and not line.strip().startswith(f"{candidate_rel}/")
    ]
    if not paths:
        return {"ok": True, "external_paths": []}
    patch = subprocess.run(
        ["git", "diff", "--binary", "--", *paths],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if patch.returncode != 0:
        return {"ok": False, "reason": "git_diff_patch_failed", "paths": paths, "detail": patch.stderr[-2000:]}
    patch_path = candidate_dir / "agent_external_edits.patch"
    patch_path.write_text(patch.stdout, encoding="utf-8")
    restored = subprocess.run(
        ["git", "apply", "-R"],
        cwd=REPO_ROOT,
        input=patch.stdout,
        text=True,
        capture_output=True,
    )
    if restored.returncode != 0:
        return {
            "ok": False,
            "reason": "git_apply_reverse_failed",
            "paths": paths,
            "patch_ref": str(patch_path.relative_to(candidate_dir.parent.parent)),
            "detail": restored.stderr[-2000:],
        }
    return {
        "ok": True,
        "external_paths": paths,
        "patch_ref": str(patch_path.relative_to(candidate_dir.parent.parent)),
        "restored": True,
    }


class tempfile_file:
    def __enter__(self):
        import tempfile

        self._handle = tempfile.TemporaryFile()
        return self._handle

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()


def _load_serve_config(candidate_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    config_path = candidate_dir / "serve_config.yaml"
    if not config_path.is_file():
        return None, "serve_config_missing"
    try:
        return _load_yaml(config_path), None
    except Exception as exc:
        return None, f"serve_config_invalid: {exc}"


def _parse_target_concurrency_from_config(
    config: dict[str, Any],
    *,
    default_from_runtime_config: int | None = None,
) -> int | None:
    request_shaping = config.get("request_shaping")
    if isinstance(request_shaping, dict):
        value = request_shaping.get("target_concurrency") or request_shaping.get("concurrent_requests")
        if value is not None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            if 1 <= parsed <= 8:
                return parsed
    prefix_cache = config.get("prefix_cache")
    if isinstance(prefix_cache, dict) and bool(prefix_cache.get("enabled", True)):
        return 4
    if default_from_runtime_config is not None:
        return max(1, min(int(default_from_runtime_config), 8))
    return None


def _parse_vllm_config_overrides(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = config.get("vllm_config")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "vllm_config_must_be_mapping"
    unknown = sorted(set(raw) - _SUPPORTED_VLLM_CONFIG_FIELDS)
    if unknown:
        return None, f"unsupported_vllm_config_fields:{','.join(unknown)}"
    issue = _validate_vllm_config_override_values(raw)
    if issue is not None:
        return None, issue
    return dict(raw), None


def _parse_spec_decode_config(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = config.get("spec_decode")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "spec_decode_must_be_mapping"
    unknown = sorted(set(raw) - _SUPPORTED_SPEC_DECODE_FIELDS)
    if unknown:
        return None, f"unsupported_spec_decode_fields:{','.join(unknown)}"
    method = raw.get("method", "ngram")
    if method not in _SUPPORTED_SPEC_DECODE_METHODS:
        return None, "invalid_spec_decode_method:must_be_ngram"
    parsed: dict[str, Any] = {"method": method}
    int_ranges = {
        "num_speculative_tokens": (1, 8),
        "prompt_lookup_min": (1, 16),
        "prompt_lookup_max": (1, 64),
    }
    for key, (minimum, maximum) in int_ranges.items():
        value = raw.get(key)
        if value is None:
            if key == "num_speculative_tokens":
                value = 4
            elif key == "prompt_lookup_min":
                value = 2
            else:
                value = 6
        if not isinstance(value, int) or isinstance(value, bool):
            return None, f"invalid_spec_decode_{key}:must_be_int"
        if value < minimum or value > maximum:
            return None, f"invalid_spec_decode_{key}:must_be_{minimum}_to_{maximum}"
        parsed[key] = value
    if parsed["prompt_lookup_min"] > parsed["prompt_lookup_max"]:
        return None, "invalid_spec_decode_prompt_lookup_range:min_gt_max"
    return parsed, None


def _parse_kernel_selection_config(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = config.get("kernel_selection")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "kernel_selection_must_be_mapping"
    unknown = sorted(set(raw) - _SUPPORTED_KERNEL_SELECTION_FIELDS)
    if unknown:
        return None, f"unsupported_kernel_selection_fields:{','.join(unknown)}"
    parsed = _normalize_kernel_selection_values(raw)
    plan = resolve_kernel_runtime_activation(parsed)
    if not plan.supported:
        unsupported = ",".join(f"{knob.axis}={knob.value}" for knob in plan.unsupported_knobs)
        return None, f"unsupported_kernel_selection:{unsupported}"
    return parsed, None


def _normalize_kernel_selection_values(raw: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(raw)
    cuda_graph_capture = parsed.get("cuda_graph_capture")
    if isinstance(cuda_graph_capture, bool):
        parsed["cuda_graph_capture"] = "on" if cuda_graph_capture else "off"
    elif isinstance(cuda_graph_capture, str):
        normalized = cuda_graph_capture.lower()
        if normalized == "true":
            parsed["cuda_graph_capture"] = "on"
        elif normalized == "false":
            parsed["cuda_graph_capture"] = "off"
    return parsed


def _validate_vllm_config_override_values(raw: dict[str, Any]) -> str | None:
    int_ranges = {
        "max_num_seqs": (1, 64),
        "max_num_batched_tokens": (1, 16384),
        "max_model_len": (1, 131072),
    }
    for key, (minimum, maximum) in int_ranges.items():
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool):
            return f"invalid_vllm_config_{key}:must_be_int"
        if value < minimum or value > maximum:
            return f"invalid_vllm_config_{key}:must_be_{minimum}_to_{maximum}"
    for key in ("enable_chunked_prefill", "enable_prefix_caching"):
        if key in raw and not isinstance(raw[key], bool):
            return f"invalid_vllm_config_{key}:must_be_bool"
    if "gpu_memory_utilization" in raw:
        value = raw["gpu_memory_utilization"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "invalid_vllm_config_gpu_memory_utilization:must_be_number"
        if float(value) < 0.0 or float(value) > 0.95:
            return "invalid_vllm_config_gpu_memory_utilization:must_be_0.0_to_0.95"
    if "kv_cache_dtype" in raw:
        value = raw["kv_cache_dtype"]
        if value not in {"fp8_e5m2", "auto"}:
            return "invalid_vllm_config_kv_cache_dtype:must_be_fp8_e5m2_or_auto"
    return None


def _previous_best_decode_tps(round_dir: Path, *, baseline_tps: float) -> float:
    best = float(baseline_tps)
    for path in sorted(round_dir.glob("candidates/*/throughput.json")):
        payload = _load_json(path) or {}
        raw = payload.get("warm_decode_tps") or payload.get("decode_tps")
        if raw is None:
            continue
        try:
            best = max(best, float(raw))
        except (TypeError, ValueError):
            continue
    return best


def _resolve_existing_path(raw: str | Path, *, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in (base_dir / path, REPO_ROOT / path, path):
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve()


def _base_vllm_config(config: ModelConfig) -> dict[str, Any]:
    return {
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "enable_prefix_caching": config.enable_prefix_caching,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "kv_cache_dtype": config.kv_cache_dtype,
    }


def _merged_vllm_config(config: ModelConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    merged = _base_vllm_config(config)
    merged.update(overrides)
    return merged


def _runtime_server(args: argparse.Namespace) -> ModelServer:
    return ModelServer(
        registry_path=_resolve_existing_path(args.registry_path),
        port=args.port,
        container_name=args.runtime_container_name,
        logs_root=args.runtime_logs_root,
        triton_cache_root=args.runtime_triton_cache_root,
        state_root=args.state_root,
        proxy_port=args.runtime_proxy_port,
        ready_timeout_s=args.runtime_ready_timeout_s,
    )


def _active_tuned_config_bundle_safe(server: ModelServer, model: str) -> tuple[str | None, Any | None, str | None]:
    try:
        bundle_path, bundle = server.active_tuned_config_bundle(model)
    except StructuredValidationError as exc:
        return None, None, exc.message
    except Exception as exc:
        return None, None, str(exc)
    return bundle_path, bundle, None


def _write_runtime_tuned_config_bundle(
    args: argparse.Namespace,
    *,
    round_dir: Path,
    candidate_dir: Path,
    candidate_id: str,
    candidate_config: dict[str, Any],
    vllm_config_overrides: dict[str, Any] | None,
    spec_decode_config: dict[str, Any] | None,
    kernel_selection_config: dict[str, Any] | None,
    workload_file: Path,
    target_tps: float,
    candidate_accept_tps: float,
) -> Path:
    registry_path = _resolve_existing_path(args.registry_path)
    registry = load_registry(registry_path)
    if args.model not in registry:
        raise RuntimeError(f"model_not_in_registry:{args.model}")
    model_config = registry[args.model]
    previous_bundle_path, previous_bundle, previous_bundle_warning = _active_tuned_config_bundle_safe(
        _runtime_server(args),
        args.model,
    )
    bundle = make_tuned_config_bundle(
        model_id=args.model,
        family_id=model_config.served_model_name,
        weight_version_id=default_weight_version_id(model_config),
        workload_distribution_id=compute_workload_distribution_id(workload_file),
        vllm_config=_merged_vllm_config(model_config, dict(vllm_config_overrides or {})),
        request_shaping=dict(candidate_config.get("request_shaping") or {}),
        kernel_selection=dict(kernel_selection_config or {}),
        spec_decode=dict(spec_decode_config or {}),
        objective={
            "decode_speed_at_least_tps": target_tps,
            "candidate_accept_decode_tps": candidate_accept_tps,
            "metric": "warm_decode_tps",
        },
        measurement_trace_ref=str((candidate_dir / "throughput.json").relative_to(round_dir)),
        search_trace_ref=str(candidate_dir.relative_to(round_dir)),
        baseline_bundle_id=previous_bundle.bundle_id if previous_bundle is not None else None,
        regression_guard={
            "b1_required": True,
            "b2_required_before_final": True,
            "b3_required_before_final": True,
        },
        safety_rails={
            "preserve_model_weights": True,
            "preserve_sampling_behavior": True,
            "runtime_config_fields_only": sorted(_SUPPORTED_VLLM_CONFIG_FIELDS),
            "spec_decode_fields_only": sorted(_SUPPORTED_SPEC_DECODE_FIELDS),
            "kernel_selection_fields_only": sorted(_SUPPORTED_KERNEL_SELECTION_FIELDS),
        },
        round_provenance={
            "round_type": "track_b_auto_research_runtime_config",
            "candidate_id": candidate_id,
            "confidence": "experimental",
            "latency_above_slo": False,
            "workload_descriptor_path": str(workload_file),
            "previous_tuned_config_path": previous_bundle_path,
            "previous_tuned_config_warning": previous_bundle_warning,
        },
    )
    bundle_path = candidate_dir / "tuned_config_bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle.as_dict(), sort_keys=False), encoding="utf-8")
    return bundle_path


def _apply_runtime_config_candidate(
    args: argparse.Namespace,
    *,
    round_dir: Path,
    candidate_dir: Path,
    candidate_id: str,
    candidate_config: dict[str, Any],
    vllm_config_overrides: dict[str, Any] | None,
    spec_decode_config: dict[str, Any] | None,
    kernel_selection_config: dict[str, Any] | None,
    workload_file: Path,
    target_tps: float,
    candidate_accept_tps: float,
) -> dict[str, Any]:
    server = _runtime_server(args)
    previous_bundle_path, previous_bundle, previous_bundle_warning = _active_tuned_config_bundle_safe(server, args.model)
    bundle_path = _write_runtime_tuned_config_bundle(
        args,
        round_dir=round_dir,
        candidate_dir=candidate_dir,
        candidate_id=candidate_id,
        candidate_config=candidate_config,
        vllm_config_overrides=vllm_config_overrides,
        spec_decode_config=spec_decode_config,
        kernel_selection_config=kernel_selection_config,
        workload_file=workload_file,
        target_tps=target_tps,
        candidate_accept_tps=candidate_accept_tps,
    )
    loaded_bundle = server.load_tuned_config(bundle_path, bundle_confidence_policy="warn")
    server.start(args.model)
    return {
        "bundle_path": str(bundle_path),
        "bundle_id": loaded_bundle.bundle_id,
        "previous_bundle_path": previous_bundle_path,
        "previous_bundle_id": previous_bundle.bundle_id if previous_bundle is not None else None,
        "previous_bundle_warning": previous_bundle_warning,
    }


def _restore_runtime_config(args: argparse.Namespace, previous_bundle_path: str | None) -> dict[str, Any]:
    server = _runtime_server(args)
    if previous_bundle_path:
        bundle = server.load_tuned_config(previous_bundle_path, bundle_confidence_policy="warn")
        server.start(args.model)
        return {"ok": True, "restored_bundle_path": previous_bundle_path, "restored_bundle_id": bundle.bundle_id}
    server.state_store.clear_active_bundle()
    server.start(args.model)
    return {"ok": True, "restored_bundle_path": None, "restored_bundle_id": None}


def _run_cmd(argv: list[str], *, cwd: Path, output_path: Path | None = None) -> tuple[int, str]:
    completed = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    text = completed.stdout + completed.stderr
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
    return completed.returncode, text


def _tail_file(path: Path, *, max_lines: int = 400) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=max_lines))
    except OSError as exc:
        return f"<failed to read {path}: {exc}>\n"


def _snapshot_runtime_logs(args: argparse.Namespace, candidate_dir: Path, *, reason: str) -> str | None:
    logs_root = Path(args.runtime_logs_root)
    if not logs_root.is_dir():
        return None
    log_paths = sorted(logs_root.glob("*.log"))
    if not log_paths:
        return None
    snapshot_path = candidate_dir / "runtime_logs_on_failure.log"
    parts = [f"# Runtime log snapshot: {reason}\n"]
    for log_path in log_paths:
        parts.append(f"\n==> {log_path} <==\n")
        parts.append(_tail_file(log_path))
    snapshot_path.write_text("".join(parts), encoding="utf-8")
    return snapshot_path.name


def _evaluate_candidate_core(
    args: argparse.Namespace,
    round_dir: Path,
    candidate_dir: Path,
    *,
    concurrency: int,
    workload_file: Path,
    target_tps: float,
    candidate_accept_tps: float,
    previous_best_tps: float,
) -> dict[str, Any]:
    throughput_path = candidate_dir / "throughput.json"
    spec = _load_yaml(round_dir / "round_spec.yaml")
    cmd = [
        str(REPO_ROOT / "scripts" / "measure_track_b_real_workload.py"),
        "--workload-file",
        str(workload_file),
        "--endpoint",
        args.endpoint,
        "--health-url",
        args.health_url,
        "--metrics-url",
        args.metrics_url,
        "--reset-prefix-cache-url",
        args.reset_prefix_cache_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--task-count",
        str(args.task_count),
        "--completions-per-task",
        str(args.completions_per_task),
        "--cold-completions",
        str(args.cold_completions),
        "--warm-concurrency",
        str(concurrency),
        "--prompt-token-cap",
        str(args.prompt_token_cap),
        "--max-output-token-cap",
        str(args.max_output_token_cap),
        "--baseline-decode-tps",
        str(spec.get("baseline_decode_tps", 7.5)),
        "--target-multiplier",
        str(spec.get("target_multiplier", 5.0)),
        "--reset-prefix-cache",
        "--output",
        str(throughput_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "throughput_command.log")
    if code not in {0, 2} or not throughput_path.is_file():
        return {"status": "rejected", "reason": "throughput_measure_failed", "detail": text[-2000:]}
    throughput = _load_json(throughput_path) or {}
    decode_tps = float(throughput.get("warm_decode_tps") or throughput.get("decode_tps") or 0.0)
    if decode_tps < candidate_accept_tps:
        return {
            "status": "rejected",
            "reason": "speed_below_candidate_acceptance",
            "decode_tps": decode_tps,
            "candidate_accept_decode_tps": candidate_accept_tps,
            "previous_best_decode_tps": previous_best_tps,
            "target_decode_tps": target_tps,
        }
    b1_path = candidate_dir / "b1_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_batch_equivalence.py"),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--prompt-count",
        str(max(4, concurrency)),
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        str(min(args.prefix_words, 1024)),
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b1_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b1_command.log")
    b1 = _load_json(b1_path) or {}
    if code != 0 or not b1.get("pass"):
        return {
            "status": "rejected",
            "reason": "b1_equivalence_failed",
            "decode_tps": decode_tps,
            "candidate_accept_decode_tps": candidate_accept_tps,
            "previous_best_decode_tps": previous_best_tps,
            "target_decode_tps": target_tps,
            "b1": b1,
        }
    trace_file = _load_yaml(round_dir / "round_spec.yaml").get("workload_trace")
    if not trace_file:
        return {
            "status": "accepted_for_speed_not_promoted",
            "reason": "workload_trace_missing_for_b2_b3",
            "decode_tps": decode_tps,
            "candidate_accept_decode_tps": candidate_accept_tps,
            "previous_best_decode_tps": previous_best_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
        }
    trace_path = Path(str(trace_file))
    b2_path = candidate_dir / "b2_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_workload_equivalence.py"),
        "--suite",
        "b2",
        "--trace-file",
        str(trace_path),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--probe-count",
        "4",
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        "512",
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b2_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b2_command.log")
    b2 = _load_json(b2_path) or {}
    if code != 0 or not b2.get("pass"):
        return {
            "status": "rejected",
            "reason": "b2_workload_equivalence_failed",
            "decode_tps": decode_tps,
            "candidate_accept_decode_tps": candidate_accept_tps,
            "previous_best_decode_tps": previous_best_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
            "b2_ref": str(b2_path.relative_to(round_dir)),
            "b2": b2,
        }
    b3_path = candidate_dir / "b3_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_workload_equivalence.py"),
        "--suite",
        "b3",
        "--trace-file",
        str(trace_path),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--probe-count",
        "8",
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        "1024",
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b3_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b3_command.log")
    b3 = _load_json(b3_path) or {}
    if code != 0 or not b3.get("pass"):
        return {
            "status": "rejected",
            "reason": "b3_workload_equivalence_failed",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
            "b2_ref": str(b2_path.relative_to(round_dir)),
            "b3_ref": str(b3_path.relative_to(round_dir)),
            "b3": b3,
        }
    final_status = "accepted_final" if decode_tps >= target_tps else "accepted_candidate"
    return {
        "status": final_status,
        "decode_tps": decode_tps,
        "candidate_accept_decode_tps": candidate_accept_tps,
        "previous_best_decode_tps": previous_best_tps,
        "target_decode_tps": target_tps,
        "concurrency": concurrency,
        "throughput_ref": str(throughput_path.relative_to(round_dir)),
        "b1_ref": str(b1_path.relative_to(round_dir)),
        "b2_ref": str(b2_path.relative_to(round_dir)),
        "b3_ref": str(b3_path.relative_to(round_dir)),
    }


def _evaluate_candidate(args: argparse.Namespace, round_dir: Path, candidate_dir: Path, candidate_id: str) -> dict[str, Any]:
    analysis_path = candidate_dir / "candidate_analysis.md"
    if not analysis_path.is_file():
        return {"status": "rejected", "reason": "candidate_analysis_missing"}
    candidate_config, config_error = _load_serve_config(candidate_dir)
    if config_error is not None or candidate_config is None:
        return {"status": "rejected", "reason": config_error or "serve_config_missing"}
    signature = _surface_signature(candidate_config)
    if args.reject_duplicate_surfaces and _has_prior_surface_signature(
        round_dir,
        signature,
        current_candidate_id=candidate_id,
    ):
        return {
            "status": "rejected",
            "reason": "duplicate_serving_surface",
            "surface_signature": signature,
        }
    vllm_config_overrides, vllm_config_error = _parse_vllm_config_overrides(candidate_config)
    if vllm_config_error is not None:
        return {"status": "rejected", "reason": vllm_config_error}
    spec_decode_config, spec_decode_error = _parse_spec_decode_config(candidate_config)
    if spec_decode_error is not None:
        return {"status": "rejected", "reason": spec_decode_error}
    kernel_selection_config, kernel_selection_error = _parse_kernel_selection_config(candidate_config)
    if kernel_selection_error is not None:
        return {"status": "rejected", "reason": kernel_selection_error}
    runtime_candidate = (
        vllm_config_overrides is not None
        or spec_decode_config is not None
        or kernel_selection_config is not None
    )
    default_concurrency = 4 if runtime_candidate else None
    if vllm_config_overrides is not None and vllm_config_overrides.get("max_num_seqs") is not None:
        default_concurrency = int(vllm_config_overrides["max_num_seqs"])
    concurrency = _parse_target_concurrency_from_config(
        candidate_config,
        default_from_runtime_config=default_concurrency,
    )
    if concurrency is None:
        return {"status": "rejected", "reason": "unsupported_or_missing_serve_config"}

    spec = _load_yaml(round_dir / "round_spec.yaml")
    workload_file_raw = args.workload_file or spec.get("workload_file")
    if not workload_file_raw:
        return {"status": "rejected", "reason": "real_workload_file_missing"}
    workload_file = _resolve_existing_path(workload_file_raw)
    target_tps = float(spec["success_criteria"]["decode_speed_at_least_tps"])
    baseline_tps = float(spec.get("baseline_decode_tps", 7.5))
    incremental_multiplier = float(
        spec.get("success_criteria", {}).get("candidate_acceptance_incremental_speedup_at_least", 1.2)
    )
    previous_best_tps = _previous_best_decode_tps(round_dir, baseline_tps=baseline_tps)
    candidate_accept_tps = previous_best_tps * incremental_multiplier

    runtime_activation: dict[str, Any] | None = None
    if runtime_candidate:
        if not args.apply_runtime_config:
            return {
                "status": "rejected",
                "reason": "runtime_config_requires_apply_flag",
                "vllm_config": vllm_config_overrides,
                "spec_decode": spec_decode_config,
                "kernel_selection": kernel_selection_config,
            }
        try:
            runtime_activation = _apply_runtime_config_candidate(
                args,
                round_dir=round_dir,
                candidate_dir=candidate_dir,
                candidate_id=candidate_id,
                candidate_config=candidate_config,
                vllm_config_overrides=vllm_config_overrides,
                spec_decode_config=spec_decode_config,
                kernel_selection_config=kernel_selection_config,
                workload_file=workload_file,
                target_tps=target_tps,
                candidate_accept_tps=candidate_accept_tps,
            )
        except Exception as exc:
            restore: dict[str, Any] | None = None
            if args.restore_runtime_after_candidate:
                try:
                    restore = _restore_runtime_config(args, None)
                except Exception as restore_exc:
                    restore = {"ok": False, "error": str(restore_exc)[-2000:]}
            return {
                "status": "rejected",
                "reason": "runtime_config_apply_failed",
                "detail": str(exc)[-2000:],
                "vllm_config": vllm_config_overrides,
                "spec_decode": spec_decode_config,
                "kernel_selection": kernel_selection_config,
                "runtime_restore_after_apply_failure": restore,
            }

    result: dict[str, Any] | None = None
    try:
        result = _evaluate_candidate_core(
            args,
            round_dir,
            candidate_dir,
            concurrency=concurrency,
            workload_file=workload_file,
            target_tps=target_tps,
            candidate_accept_tps=candidate_accept_tps,
            previous_best_tps=previous_best_tps,
        )
        if runtime_activation is not None:
            result["runtime_config_ref"] = str(Path(runtime_activation["bundle_path"]).relative_to(round_dir))
            result["runtime_config_id"] = runtime_activation["bundle_id"]
            result["vllm_config"] = vllm_config_overrides
            result["spec_decode"] = spec_decode_config
            result["kernel_selection"] = kernel_selection_config
        return result
    finally:
        if (
            runtime_activation is not None
            and result is not None
            and result.get("reason") == "throughput_measure_failed"
        ):
            snapshot_ref = _snapshot_runtime_logs(args, candidate_dir, reason=str(result.get("reason")))
            if snapshot_ref is not None:
                result["runtime_log_snapshot_ref"] = str((candidate_dir / snapshot_ref).relative_to(round_dir))
        if runtime_activation is not None and args.restore_runtime_after_candidate:
            restore_path = candidate_dir / "runtime_restore_result.json"
            try:
                restore = _restore_runtime_config(args, runtime_activation["previous_bundle_path"])
            except Exception as exc:
                restore = {"ok": False, "error": str(exc)[-2000:]}
            _write_json(restore_path, restore)
            if result is not None:
                result["runtime_restore_ref"] = str(restore_path.relative_to(round_dir))


def _update_ledgers(round_dir: Path, candidate_id: str, result: dict[str, Any]) -> None:
    branch_log_path = round_dir / "branch_log.json"
    branch_log = json.loads(branch_log_path.read_text(encoding="utf-8")) if branch_log_path.is_file() else []
    if not isinstance(branch_log, list):
        branch_log = []
    branch_log.append({"candidate_id": candidate_id, **result, "recorded_at": _now()})
    _write_json(branch_log_path, branch_log)
    if result["status"].startswith("accepted"):
        _append_tsv(
            round_dir / "quality_gate_history.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "speed",
                "status": "pass",
                "score_json": {"decode_tps": result.get("decode_tps"), "target_decode_tps": result.get("target_decode_tps")},
                "artifact_ref": result.get("throughput_ref", ""),
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
        )
        _append_tsv(
            round_dir / "quality_gate_history.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "b1_strong_equivalence",
                "status": "pass",
                "score_json": {"concurrency": result.get("concurrency")},
                "artifact_ref": result.get("b1_ref", ""),
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
        )
        if result.get("b2_ref"):
            _append_tsv(
                round_dir / "quality_gate_history.tsv",
                {
                    "candidate_id": candidate_id,
                    "tier": "b2_workload_equivalence",
                    "status": "pass",
                    "score_json": {"concurrency": result.get("concurrency")},
                    "artifact_ref": result.get("b2_ref", ""),
                    "recorded_at": _now(),
                },
                ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
            )
        if result.get("b3_ref"):
            _append_tsv(
                round_dir / "quality_gate_history.tsv",
                {
                    "candidate_id": candidate_id,
                    "tier": "b3_workload_equivalence",
                    "status": "pass",
                    "score_json": {"concurrency": result.get("concurrency")},
                    "artifact_ref": result.get("b3_ref", ""),
                    "recorded_at": _now(),
                },
                ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
            )
    else:
        failing_metric_by_reason = {
            "throughput_measure_failed": "real_workload_measurement",
            "speed_below_candidate_acceptance": "warm_decode_tps",
            "speed_below_target": "warm_decode_tps",
            "runtime_config_apply_failed": "runtime_config",
            "unsupported_or_missing_serve_config": "serve_config",
        }
        _append_tsv(
            round_dir / "mutations_rejected.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "controller",
                "cost_bucket": result.get("reason", "rejected"),
                "reason": result.get("reason", "rejected"),
                "first_failing_metric": failing_metric_by_reason.get(result.get("reason", ""), "controller"),
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "cost_bucket", "reason", "first_failing_metric", "recorded_at"],
        )


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    round_dir = args.round_dir.resolve()
    if not (round_dir / "round_spec.yaml").is_file():
        raise RuntimeError(f"round_spec.yaml missing: {round_dir}")
    if args.state_root is None:
        args.state_root = round_dir / "runtime_state"
    monitor_rows: list[dict[str, Any]] = []
    for _ in range(args.max_attempts):
        candidate_id = _next_candidate_id(round_dir)
        candidate_dir = round_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True)
        prompt = _render_agent_prompt(round_dir, candidate_dir, candidate_id)
        spawn = _spawn_codex(round_dir, candidate_dir, prompt, args.agent_timeout_s)
        external_edit_guard = _restore_external_agent_edits(candidate_dir)
        spawn["external_edit_guard"] = external_edit_guard
        (candidate_dir / "spawn_result.json").write_text(
            json.dumps(spawn, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not spawn.get("ok"):
            result = {"status": "rejected", "reason": "agent_spawn_failed", "spawn": spawn}
        elif not external_edit_guard.get("ok"):
            result = {
                "status": "rejected",
                "reason": "agent_external_edit_restore_failed",
                "external_edit_guard": external_edit_guard,
            }
        else:
            result = _evaluate_candidate(args, round_dir, candidate_dir, candidate_id)
        (candidate_dir / "controller_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _update_ledgers(round_dir, candidate_id, result)
        monitor_rows.append({"candidate_id": candidate_id, **result})
        if result.get("status") in {"accepted_candidate", "accepted_final"} and not args.keep_searching_after_accept:
            break
    summary = {
        "schema": "lumo.track_b.loop_run.v1",
        "round_dir": str(round_dir),
        "attempts": monitor_rows,
        "completed_at": _now(),
    }
    _write_json(round_dir / "loop_monitor_latest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Track B Karpathy-style auto-research controller loop.")
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--agent-timeout-s", type=int, default=900)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workload-file")
    parser.add_argument("--port", type=int, default=9950, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--completions-per-task", type=int, default=5)
    parser.add_argument("--cold-completions", type=int, default=1)
    parser.add_argument("--prompt-token-cap", type=int, default=0)
    parser.add_argument("--max-output-token-cap", type=int, default=0)
    parser.add_argument("--prefix-words", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--apply-runtime-config", action="store_true")
    parser.add_argument("--reject-duplicate-surfaces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-runtime-after-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--registry-path", type=Path, default=REPO_ROOT / "model_registry.yaml")
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--runtime-container-name", default="lumo-vllm-l0c-fp8-cutlass-run30")
    parser.add_argument("--runtime-logs-root", type=Path, default=Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs"))
    parser.add_argument("--runtime-triton-cache-root", type=Path, default=Path("/tmp/lumo-l0c-fp8-cutlass-run30-triton"))
    parser.add_argument("--runtime-proxy-port", type=int, default=8011)
    parser.add_argument("--runtime-ready-timeout-s", type=int, default=900)
    parser.add_argument("--keep-searching-after-accept", action="store_true")
    args = parser.parse_args()
    result = run_loop(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
