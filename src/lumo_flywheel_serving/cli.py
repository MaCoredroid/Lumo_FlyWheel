from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import subprocess
import time
from pathlib import Path

import requests

from .auto_research import AutoResearchRoundManager, OfflineAutoResearchRunner, SyntheticWorkloadDistribution, load_baseline_bundle
from .metrics import LatencyCapture, aggregate_by_model, load_telemetry
from .model_server import DEFAULT_VLLM_DOCKERFILE, DEFAULT_VLLM_IMAGE, ModelServer, REPO_ROOT
from .registry import load_registry
from .round_driver import RoundContext, run_round


def _prefix_cache_probe_messages(prior_reply: str | None = None) -> list[dict[str, str]]:
    # Keep the shared prefix long enough to populate cache blocks between turns.
    shared_prefix = " ".join(["cacheprobe"] * 2048)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are concise. Reply with the single token OK.",
        },
        {
            "role": "user",
            "content": (
                "Reference context follows. Do not quote it back.\n"
                f"{shared_prefix}\n\n"
                "Reply with the single token OK."
            ),
        },
    ]
    if prior_reply is not None:
        messages.extend(
            [
                {"role": "assistant", "content": prior_reply},
                {"role": "user", "content": "Using the same reference context, reply with the single token OK again."},
            ]
        )
    return messages


def _responses_probe_request(model: str, prompt: str, previous_response_id: str | None = None) -> dict[str, object]:
    request: dict[str, object] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 8,
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    return request


def _response_id(payload: dict[str, object]) -> str:
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise RuntimeError("Responses API smoke probe did not return a response id")
    return response_id


def _response_error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    body = response.text.strip()
    return body or f"HTTP {response.status_code}"


def _raise_for_smoke_response(response: requests.Response, *, phase: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        error_text = _response_error_text(response)
        if response.status_code == 404 and "Response with id" in error_text:
            raise RuntimeError(
                f"Responses API {phase} failed: backend did not persist response ids for follow-up chaining. "
                f"vLLM returned 404: {error_text}"
            ) from exc
        raise RuntimeError(
            f"Responses API {phase} failed with HTTP {response.status_code}: {error_text}"
        ) from exc


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key, value)


def _prepare_hf_runtime() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        hf_home.mkdir(parents=True, exist_ok=True)
        probe = hf_home / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        fallback = Path.home() / ".cache" / "huggingface"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(fallback)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ModelServer._api_key()}"}


def _metric_schema_variant(schema: dict[str, str]) -> str:
    variant_keys = (
        "prompt_tokens",
        "generation_tokens",
        "cache_queries",
        "cache_hits",
    )
    values = [schema[key] for key in variant_keys if key in schema]
    if not values:
        legacy_keys = (
            "prefix_cache_queries",
            "prefix_cache_hits",
        )
        values = [schema[key] for key in legacy_keys if key in schema]
    if values and all(key.endswith("_total") for key in values):
        return "openmetrics_total"
    return "legacy_no_total"


def _smoke_request_model_name(server: object, model_id: str) -> str:
    registry = getattr(server, "registry", None)
    if registry is None:
        return model_id
    config = registry[model_id]
    request_model_name = getattr(server, "_request_model_name", None)
    if callable(request_model_name):
        return request_model_name(config)
    return getattr(config, "served_model_name", model_id)


def _proxy_responses_url(server: object, port: int) -> str:
    proxy_base_url = getattr(server, "proxy_base_url", None)
    if callable(proxy_base_url):
        return f"{proxy_base_url()}/v1/responses"
    return f"http://127.0.0.1:{port + 1}/v1/responses"


def _smoke_pool_or_split(args: argparse.Namespace) -> str:
    return getattr(args, "pool_or_split", "public_dev")


def _smoke_output_dir(args: argparse.Namespace) -> str:
    return getattr(args, "output_dir", "output")


def _smoke_task_id(args: argparse.Namespace) -> str:
    task_id = getattr(args, "smoke_task_id", None)
    if isinstance(task_id, str) and task_id.strip():
        return task_id.strip()
    return f"smoke-test/{args.model_id}/{int(time.time())}"


def _smoke_seed(args: argparse.Namespace) -> int:
    return int(getattr(args, "smoke_seed", 0))


def _smoke_attempt(args: argparse.Namespace) -> int:
    attempt = int(getattr(args, "smoke_attempt", 1))
    if attempt < 1:
        raise RuntimeError("Smoke telemetry attempt must be >= 1")
    return attempt


def _tool_call_probe_request(model: str) -> dict[str, object]:
    return {
        "model": model,
        "input": "Call the codex_tool_probe function with no arguments. Do not answer in prose.",
        "max_output_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "codex_tool_probe",
                    "description": "Smoke-test tool used to validate Codex-compatible Responses function calls.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }


def _has_structured_tool_call(payload: dict[str, object]) -> bool:
    output = payload.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and isinstance(item.get("name"), str):
            return True
    return False


def _chat_message_text(payload: dict[str, object], default: str = "OK") -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return default
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return default
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return default
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return default


def cmd_bootstrap_runtime(args: argparse.Namespace) -> int:
    for raw_path in (args.models_root, args.logs_root, args.triton_cache_root):
        Path(raw_path).mkdir(parents=True, exist_ok=True)
    dockerfile = Path(args.dockerfile)
    subprocess.run(
        ["docker", "build", "-t", args.image, "-f", str(dockerfile), str(REPO_ROOT)],
        check=True,
    )
    return 0


def cmd_build_image(args: argparse.Namespace) -> int:
    dockerfile = Path(args.dockerfile)
    subprocess.run(
        ["docker", "build", "-t", args.image, "-f", str(dockerfile), str(REPO_ROOT)],
        check=True,
    )
    return 0


def cmd_download_model(args: argparse.Namespace) -> int:
    _load_env_file(args.env_file)
    _prepare_hf_runtime()
    from huggingface_hub import snapshot_download

    registry = load_registry(args.registry)
    config = registry[args.model_id]
    if not config.hf_repo:
        raise RuntimeError(f"Model {args.model_id} does not have an hf_repo entry yet")
    if not config.hf_revision:
        raise RuntimeError(
            f"Model {args.model_id} is missing hf_revision. Refuse to download an unpinned checkpoint; "
            "confirm the upstream commit and update model_registry.yaml first."
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to download models")
    config.local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.hf_repo,
        revision=config.hf_revision,
        local_dir=str(config.local_path),
        token=token,
    )
    return 0


def _server(args: argparse.Namespace) -> ModelServer:
    return ModelServer(
        registry_path=args.registry,
        port=args.port,
        image=args.image,
        container_name=args.container_name,
        logs_root=args.logs_root,
        triton_cache_root=args.triton_cache_root,
        use_sleep_mode=args.use_sleep_mode,
        proxy_port=args.proxy_port,
        state_root=args.state_root,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    server = _server(args)
    server.start(model_id=args.model_id, enable_request_logging=args.enable_request_logging)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    server = _server(args)
    server.stop(missing_ok=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    server = _server(args)
    payload = {
        "health": server.health().status_code,
        "models": server.models().json(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    server = _server(args)
    server.start(model_id=args.model_id, enable_request_logging=args.enable_request_logging)
    try:
        request_model_name = _smoke_request_model_name(server, args.model_id)
        pool_or_split = _smoke_pool_or_split(args)
        output_dir = _smoke_output_dir(args)
        task_id = _smoke_task_id(args)
        seed = _smoke_seed(args)
        attempt = _smoke_attempt(args)
        telemetry_capture = LatencyCapture(
            "127.0.0.1",
            args.port,
            output_dir,
            args.model_id,
            pool_or_split,
        )
        asyncio.run(telemetry_capture.resolve_schema())
        schema = telemetry_capture.resolved_schema
        asyncio.run(telemetry_capture.snapshot_before(task_id=task_id, seed=seed, attempt=attempt))
        health = server.health()
        models = server.models().json()
        first_chat_request = {
            "model": request_model_name,
            "messages": _prefix_cache_probe_messages(),
            "max_tokens": 8,
            "temperature": 0,
        }
        first_chat = requests.post(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            headers=_auth_headers(),
            json=first_chat_request,
            timeout=180,
        )
        first_chat.raise_for_status()
        first_chat_payload = first_chat.json()
        first_reply = _chat_message_text(first_chat_payload)
        second_chat_request = {
            "model": request_model_name,
            "messages": _prefix_cache_probe_messages(prior_reply=first_reply),
            "max_tokens": 8,
            "temperature": 0,
        }
        second_chat = requests.post(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            headers=_auth_headers(),
            json=second_chat_request,
            timeout=180,
        )
        second_chat.raise_for_status()
        first_responses_call = requests.post(
            f"http://127.0.0.1:{args.port}/v1/responses",
            headers=_auth_headers(),
            json=_responses_probe_request(request_model_name, "Reply with the single token OK."),
            timeout=180,
        )
        _raise_for_smoke_response(first_responses_call, phase="initial turn")
        first_responses_payload = first_responses_call.json()
        first_response_id = _response_id(first_responses_payload)
        second_responses_call = requests.post(
            f"http://127.0.0.1:{args.port}/v1/responses",
            headers=_auth_headers(),
            json=_responses_probe_request(
                request_model_name,
                "Reply with the single token OK again.",
                previous_response_id=first_response_id,
            ),
            timeout=180,
        )
        _raise_for_smoke_response(second_responses_call, phase="follow-up turn")
        second_response_id = _response_id(second_responses_call.json())
        tool_probe_response = requests.post(
            _proxy_responses_url(server, args.port),
            headers=_auth_headers(),
            json=_tool_call_probe_request(request_model_name),
            timeout=180,
        )
        _raise_for_smoke_response(tool_probe_response, phase="Codex tool-call probe")
        tool_probe_payload = tool_probe_response.json()
        if not _has_structured_tool_call(tool_probe_payload):
            raise RuntimeError(
                "Codex tool-call probe failed: proxy/vLLM path did not return a structured function_call item."
            )
        telemetry_metrics = asyncio.run(telemetry_capture.snapshot_after(task_id))
        if telemetry_metrics.anomalies:
            raise RuntimeError(f"Telemetry smoke produced anomalous record: {telemetry_metrics.anomalies}")
        cache_hit_delta = telemetry_metrics.cache_hits
        if cache_hit_delta <= 0:
            raise RuntimeError(
                "Expected prefix cache hits after repeated-prefix chat turns, but /metrics did not increase."
            )
        if telemetry_metrics.kv_computed_tokens > telemetry_metrics.prompt_tokens:
            raise RuntimeError(
                "Telemetry smoke violated kv_computed_tokens <= prompt_tokens."
            )
        if telemetry_metrics.cache_hits > telemetry_metrics.cache_queries:
            raise RuntimeError(
                "Telemetry smoke violated cache_hits <= cache_queries."
            )
        expected_request_count = 5
        if telemetry_metrics.ttft_count != expected_request_count:
            raise RuntimeError(
                f"Telemetry smoke expected ttft_count={expected_request_count} for the five smoke requests, "
                f"but observed {telemetry_metrics.ttft_count}."
            )
        gpu_compute_s = telemetry_metrics.prefill_sum_s + telemetry_metrics.decode_sum_s
        if gpu_compute_s >= telemetry_metrics.wall_clock_s:
            raise RuntimeError(
                "Telemetry smoke violated prefill_sum_s + decode_sum_s < wall_clock_s."
            )
        telemetry_records = load_telemetry(str(Path(output_dir) / "telemetry"))
        telemetry_summary = aggregate_by_model(
            telemetry_records,
            {(task_id, args.model_id, seed, attempt)},
        )
        if len(telemetry_summary) != 1 or telemetry_summary[0].n_tasks != 1:
            raise RuntimeError(
                "Telemetry smoke could not reload and aggregate exactly one smoke telemetry record."
            )
        server.record_launch_metadata(
            args.model_id,
            direct_api_smoke_status="pass",
            metric_schema_variant=_metric_schema_variant(schema),
            prefix_cache_hits_delta=round(cache_hit_delta, 3),
            codex_tool_probe_status="pass",
        )
        server.flush_prefix_cache()
        print(
            json.dumps(
                {
                    "direct_api_smoke_status": "pass",
                    "health": health.status_code,
                    "models": models,
                    "schema": schema,
                    "chat_completion_ids": [first_chat_payload.get("id"), second_chat.json().get("id")],
                    "responses_ids": [first_response_id, second_response_id],
                    "tool_call_probe_status": "pass",
                    "reset_prefix_cache_status": 200,
                    "prefix_cache_hits_delta": cache_hit_delta,
                    "telemetry_task_id": task_id,
                    "telemetry_path": telemetry_capture.writer_path,
                    "telemetry_record": {
                        "seed": seed,
                        "attempt": attempt,
                        "ttft_ms": telemetry_metrics.ttft_ms,
                        "prefill_throughput_tps": telemetry_metrics.prefill_throughput_tps,
                        "decode_throughput_tps": telemetry_metrics.decode_throughput_tps,
                        "cache_hit_rate_pct": telemetry_metrics.cache_hit_rate_pct,
                        "prompt_tokens": telemetry_metrics.prompt_tokens,
                        "kv_computed_tokens": telemetry_metrics.kv_computed_tokens,
                        "gen_tokens": telemetry_metrics.gen_tokens,
                        "prefill_sum_s": telemetry_metrics.prefill_sum_s,
                        "decode_sum_s": telemetry_metrics.decode_sum_s,
                        "ttft_count": telemetry_metrics.ttft_count,
                        "cache_queries": telemetry_metrics.cache_queries,
                        "cache_hits": telemetry_metrics.cache_hits,
                        "wall_clock_s": telemetry_metrics.wall_clock_s,
                        "anomalies": telemetry_metrics.anomalies,
                    },
                    "telemetry_summary": {
                        "model_id": telemetry_summary[0].model_id,
                        "pool_or_split": telemetry_summary[0].pool_or_split,
                        "n_tasks": telemetry_summary[0].n_tasks,
                    },
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        server.record_launch_metadata(
            args.model_id,
            direct_api_smoke_status="escalated",
            direct_api_smoke_error=str(exc),
        )
        raise
    finally:
        if not args.keep_running:
            server.stop(missing_ok=True)


def cmd_switch_model(args: argparse.Namespace) -> int:
    server = _server(args)
    server.switch_model(model_id=args.model_id, enable_request_logging=args.enable_request_logging)
    return 0


def cmd_annotate_log(args: argparse.Namespace) -> int:
    server = _server(args)
    metadata: dict[str, str] = {}
    for raw_entry in args.entries:
        if "=" not in raw_entry:
            raise RuntimeError(f"Invalid metadata entry '{raw_entry}'. Expected key=value.")
        key, value = raw_entry.split("=", 1)
        if not key:
            raise RuntimeError(f"Invalid metadata entry '{raw_entry}'. Metadata keys must be non-empty.")
        metadata[key] = value
    server.record_launch_metadata(args.model_id, **metadata)
    return 0


def cmd_load_tuned_config(args: argparse.Namespace) -> int:
    server = _server(args)
    bundle = server.load_tuned_config(args.bundle_path)
    print(
        json.dumps(
            {
                "status": "loaded",
                "bundle_id": bundle.bundle_id,
                "model_id": bundle.model_id,
                "weight_version_id": bundle.weight_version_id,
                "bundle_path": str(Path(args.bundle_path)),
            },
            indent=2,
        )
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    server = _server(args)
    payload = server.resume_last_known_good(
        from_baseline=args.from_baseline,
        enable_request_logging=args.enable_request_logging,
    )
    print(json.dumps({"status": "resumed", **payload}, indent=2))
    return 0


def _auto_research_manager(args: argparse.Namespace) -> AutoResearchRoundManager:
    return AutoResearchRoundManager(
        registry_path=args.registry,
        repo_root=REPO_ROOT,
        tuned_config_root=args.tuned_config_root,
        port=args.port,
        proxy_port=args.proxy_port,
    )


def _auto_research_help_only(args: argparse.Namespace) -> int:
    if getattr(args, "help_only", False):
        print(json.dumps({"subcommand": args.auto_research_command, "status": "registered"}))
        return 0
    return -1


def _require_auto_research_args(args: argparse.Namespace, *names: str) -> None:
    missing: list[str] = []
    for name in names:
        value = getattr(args, name)
        if isinstance(value, str):
            if not value.strip():
                missing.append(name)
        elif value is None:
            missing.append(name)
    if missing:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise RuntimeError(f"Missing required arguments: {rendered}")


def cmd_auto_research_bootstrap_round(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "model_id", "family_id", "sprint", "workload_file")
    manager = _auto_research_manager(args)
    payload = manager.bootstrap_round(
        model_id=args.model_id,
        family_id=args.family_id,
        sprint=args.sprint,
        workload_file=args.workload_file,
        weight_version_id=args.weight_version_id,
        round_root=args.round_root,
        active_layer=args.active_layer,
        baseline_bundle=args.baseline_bundle,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_measure(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id", "candidate")
    manager = _auto_research_manager(args)
    payload = manager.measure(round_id=args.round_id, candidate_path=args.candidate, harness=args.harness)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_commit_candidate(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id", "iteration", "status", "notes")
    manager = _auto_research_manager(args)
    payload = manager.commit_candidate(
        round_id=args.round_id,
        iteration=args.iteration,
        status=args.status,
        notes=args.notes,
        harness=args.harness,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_rescreen(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id")
    manager = _auto_research_manager(args)
    payload = manager.rescreen(round_id=args.round_id, top_k=args.top_k, profile=args.profile, harness=args.harness)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_validate_holdout(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id", "candidate_uuid")
    manager = _auto_research_manager(args)
    payload = manager.validate_holdout(round_id=args.round_id, candidate_uuid=args.candidate_uuid, harness=args.harness)
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("pass") else 1


def cmd_auto_research_finalize_round(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id")
    manager = _auto_research_manager(args)
    payload = manager.finalize_round(round_id=args.round_id, dry_run=args.dry_run)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_status(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "round_id")
    manager = _auto_research_manager(args)
    try:
        payload = manager.status(round_id=args.round_id)
    except FileNotFoundError:
        print(json.dumps({"round_id": args.round_id, "phase": "missing"}))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_run(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "model_id", "family_id", "workload_file")
    manager = _auto_research_manager(args)
    payload = manager.run_non_agent(
        model_id=args.model_id,
        family_id=args.family_id,
        workload_file=args.workload_file,
        baseline_bundle=args.baseline_bundle,
        weight_version_id=args.weight_version_id,
        round_root=args.round_root,
        iteration_cap=args.iteration_cap,
        harness_type=args.harness,
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_research_run_round(args: argparse.Namespace) -> int:
    if (code := _auto_research_help_only(args)) >= 0:
        return code
    _require_auto_research_args(args, "model_id", "family_id", "workload_file")
    manager = _auto_research_manager(args)
    bootstrap = manager.bootstrap_round(
        model_id=args.model_id,
        family_id=args.family_id,
        sprint=args.sprint,
        workload_file=args.workload_file,
        weight_version_id=args.weight_version_id,
        round_root=args.round_root,
        harness_type=args.harness,
        skip_preflight=args.harness == "synthetic",
        active_layer=args.active_layer,
        baseline_bundle=args.baseline_bundle,
    )
    ctx = RoundContext.from_bootstrap_json(
        bootstrap,
        harness_mode=args.harness,
        registry_path=Path(args.registry),
        tuned_config_root=Path(args.tuned_config_root),
        port=args.port,
        proxy_port=args.proxy_port,
        iteration_cap=args.iteration_cap,
    )
    result = run_round(ctx)
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.outcome in {"ROUND_BUNDLE_READY", "ROUND_BASELINE_RETAINED"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lumo FlyWheel vLLM serving tooling")
    parser.set_defaults(func=None)
    parser.add_argument("--registry", default="model_registry.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image", default=DEFAULT_VLLM_IMAGE)
    parser.add_argument("--dockerfile", default=str(DEFAULT_VLLM_DOCKERFILE))
    parser.add_argument("--container-name", default="lumo-vllm")
    parser.add_argument("--proxy-port", type=int, default=8001)
    parser.add_argument("--logs-root", default="/logs")
    parser.add_argument("--triton-cache-root", default="/tmp/triton_cache")
    parser.add_argument("--use-sleep-mode", action="store_true")
    parser.add_argument("--state-root", default=str(REPO_ROOT / "output" / "serving_state"))
    parser.add_argument("--tuned-config-root", default=str(REPO_ROOT / "output" / "tuned_configs"))

    subparsers = parser.add_subparsers(dest="command")

    bootstrap = subparsers.add_parser("bootstrap-runtime")
    bootstrap.add_argument("--models-root", default="/models")
    bootstrap.add_argument("--logs-root", default="/logs")
    bootstrap.add_argument("--triton-cache-root", default="/tmp/triton_cache")
    bootstrap.set_defaults(func=cmd_bootstrap_runtime)

    build_image = subparsers.add_parser("build-image")
    build_image.set_defaults(func=cmd_build_image)

    download = subparsers.add_parser("download-model")
    download.add_argument("model_id")
    download.add_argument("--env-file")
    download.set_defaults(func=cmd_download_model)

    serve = subparsers.add_parser("serve")
    serve.add_argument("model_id")
    serve.add_argument("--enable-request-logging", action="store_true")
    serve.set_defaults(func=cmd_serve)

    switch = subparsers.add_parser("switch-model")
    switch.add_argument("model_id")
    switch.add_argument("--enable-request-logging", action="store_true")
    switch.set_defaults(func=cmd_switch_model)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(func=cmd_stop)

    status = subparsers.add_parser("status")
    status.set_defaults(func=cmd_status)

    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("model_id")
    smoke.add_argument("--enable-request-logging", action="store_true")
    smoke.add_argument("--keep-running", action="store_true")
    smoke.add_argument("--output-dir", default="output")
    smoke.add_argument("--pool-or-split", default="public_dev")
    smoke.add_argument("--smoke-task-id")
    smoke.add_argument("--smoke-seed", type=int, default=0)
    smoke.add_argument("--smoke-attempt", type=int, default=1)
    smoke.set_defaults(func=cmd_smoke_test)

    annotate = subparsers.add_parser("annotate-log")
    annotate.add_argument("model_id")
    annotate.add_argument("entries", nargs="+")
    annotate.set_defaults(func=cmd_annotate_log)

    load_tuned = subparsers.add_parser("load-tuned-config")
    load_tuned.add_argument("bundle_path")
    load_tuned.set_defaults(func=cmd_load_tuned_config)

    auto_research = subparsers.add_parser("auto-research")
    auto_research_subparsers = auto_research.add_subparsers(dest="auto_research_command")
    auto_bootstrap = auto_research_subparsers.add_parser("bootstrap-round")
    auto_bootstrap.add_argument("--help-only", action="store_true")
    auto_bootstrap.add_argument("--model-id")
    auto_bootstrap.add_argument("--family-id")
    auto_bootstrap.add_argument("--sprint")
    auto_bootstrap.add_argument("--workload-file")
    auto_bootstrap.add_argument("--weight-version-id")
    auto_bootstrap.add_argument("--active-layer", choices=["L1", "L2"], default="L1")
    auto_bootstrap.add_argument("--baseline-bundle")
    auto_bootstrap.add_argument("--round-root", default=str(REPO_ROOT / "output" / "auto_research"))
    auto_bootstrap.set_defaults(func=cmd_auto_research_bootstrap_round)

    auto_measure = auto_research_subparsers.add_parser("measure")
    auto_measure.add_argument("--help-only", action="store_true")
    auto_measure.add_argument("--round-id")
    auto_measure.add_argument("--candidate")
    auto_measure.add_argument("--harness", choices=["real", "synthetic"], default=None)
    auto_measure.set_defaults(func=cmd_auto_research_measure)

    auto_commit = auto_research_subparsers.add_parser("commit-candidate")
    auto_commit.add_argument("--help-only", action="store_true")
    auto_commit.add_argument("--round-id")
    auto_commit.add_argument("--iteration")
    auto_commit.add_argument("--status", choices=["baseline", "keep", "discard", "crash", "harness_fault"])
    auto_commit.add_argument("--notes")
    auto_commit.add_argument("--harness", choices=["real", "synthetic"], default=None)
    auto_commit.set_defaults(func=cmd_auto_research_commit_candidate)

    auto_rescreen = auto_research_subparsers.add_parser("rescreen")
    auto_rescreen.add_argument("--help-only", action="store_true")
    auto_rescreen.add_argument("--round-id")
    auto_rescreen.add_argument("--top-k", type=int, default=3)
    auto_rescreen.add_argument("--profile", default="full")
    auto_rescreen.add_argument("--harness", choices=["real", "synthetic"], default=None)
    auto_rescreen.set_defaults(func=cmd_auto_research_rescreen)

    auto_holdout = auto_research_subparsers.add_parser("validate-holdout")
    auto_holdout.add_argument("--help-only", action="store_true")
    auto_holdout.add_argument("--round-id")
    auto_holdout.add_argument("--candidate-uuid")
    auto_holdout.add_argument("--harness", choices=["real", "synthetic"], default=None)
    auto_holdout.set_defaults(func=cmd_auto_research_validate_holdout)

    auto_finalize = auto_research_subparsers.add_parser("finalize-round")
    auto_finalize.add_argument("--help-only", action="store_true")
    auto_finalize.add_argument("--round-id")
    auto_finalize.add_argument("--dry-run", action="store_true")
    auto_finalize.set_defaults(func=cmd_auto_research_finalize_round)

    auto_status = auto_research_subparsers.add_parser("status")
    auto_status.add_argument("--help-only", action="store_true")
    auto_status.add_argument("--round-id")
    auto_status.add_argument("--json", action="store_true")
    auto_status.set_defaults(func=cmd_auto_research_status)

    auto_run_round = auto_research_subparsers.add_parser("run-round")
    auto_run_round.add_argument("--help-only", action="store_true")
    auto_run_round.add_argument("--model-id")
    auto_run_round.add_argument("--family-id")
    auto_run_round.add_argument("--sprint", default="sprint-0")
    auto_run_round.add_argument("--workload-file")
    auto_run_round.add_argument("--weight-version-id")
    auto_run_round.add_argument("--active-layer", choices=["L1", "L2"], default="L1")
    auto_run_round.add_argument("--baseline-bundle")
    auto_run_round.add_argument("--harness", choices=["real", "synthetic"], default="real")
    auto_run_round.add_argument("--iteration-cap", type=int, default=12)
    auto_run_round.add_argument("--round-root", default=str(REPO_ROOT / "output" / "auto_research"))
    auto_run_round.set_defaults(func=cmd_auto_research_run_round)

    auto_research_run = auto_research_subparsers.add_parser("run")
    auto_research_run.add_argument("--help-only", action="store_true")
    auto_research_run.add_argument("model_id", nargs="?")
    auto_research_run.add_argument("--family-id")
    auto_research_run.add_argument("--workload-file")
    auto_research_run.add_argument("--baseline-bundle")
    auto_research_run.add_argument("--weight-version-id")
    auto_research_run.add_argument("--harness", choices=["real", "synthetic"], default="real")
    auto_research_run.add_argument("--iteration-cap", type=int, default=12)
    auto_research_run.add_argument("--round-root", default=str(REPO_ROOT / "output" / "auto_research"))
    auto_research_run.set_defaults(func=cmd_auto_research_run)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--enable-request-logging", action="store_true")
    resume.add_argument("--from-baseline", action="store_true")
    resume.set_defaults(func=cmd_resume)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
