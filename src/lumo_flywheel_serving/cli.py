from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import requests

from .metrics import parse_prometheus_text, resolve_metric_schema
from .model_server import DEFAULT_VLLM_DOCKERFILE, DEFAULT_VLLM_IMAGE, ModelServer, REPO_ROOT
from .registry import load_registry


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
        "prefix_cache_queries",
        "prefix_cache_hits",
    )
    values = [schema[key] for key in variant_keys if key in schema]
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
        health = server.health()
        models = server.models().json()
        metrics_before = parse_prometheus_text(server.metrics().text)
        schema = resolve_metric_schema(metrics_before)
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
        first_reply = first_chat_payload["choices"][0]["message"]["content"].strip() or "OK"
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
        metrics_after = parse_prometheus_text(server.metrics().text)
        cache_hit_metric = schema["prefix_cache_hits"]
        cache_hit_delta = metrics_after.get(cache_hit_metric, 0.0) - metrics_before.get(cache_hit_metric, 0.0)
        if cache_hit_delta <= 0:
            raise RuntimeError(
                "Expected prefix cache hits after repeated-prefix chat turns, but /metrics did not increase."
            )
        server.record_launch_metadata(
            args.model_id,
            direct_api_smoke_status="pass",
            metric_schema_variant=_metric_schema_variant(schema),
            prefix_cache_hits_delta=round(cache_hit_delta, 3),
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
                    "reset_prefix_cache_status": 200,
                    "prefix_cache_hits_delta": cache_hit_delta,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lumo FlyWheel vLLM serving tooling")
    parser.set_defaults(func=None)
    parser.add_argument("--registry", default="model_registry.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image", default=DEFAULT_VLLM_IMAGE)
    parser.add_argument("--dockerfile", default=str(DEFAULT_VLLM_DOCKERFILE))
    parser.add_argument("--container-name", default="lumo-vllm")
    parser.add_argument("--logs-root", default="/logs")
    parser.add_argument("--triton-cache-root", default="/tmp/triton_cache")
    parser.add_argument("--use-sleep-mode", action="store_true")

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
    smoke.set_defaults(func=cmd_smoke_test)

    annotate = subparsers.add_parser("annotate-log")
    annotate.add_argument("model_id")
    annotate.add_argument("entries", nargs="+")
    annotate.set_defaults(func=cmd_annotate_log)
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
