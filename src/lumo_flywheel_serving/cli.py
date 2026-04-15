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
        health = server.health()
        models = server.models().json()
        metrics_before = parse_prometheus_text(server.metrics().text)
        schema = resolve_metric_schema(metrics_before)
        chat_request = {
            "model": args.model_id,
            "messages": [
                {"role": "system", "content": "You are concise. Reply with the single token OK."},
                {"role": "user", "content": "Reply with the single token OK."},
            ],
            "max_tokens": 8,
            "temperature": 0,
        }
        first_chat = requests.post(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json=chat_request,
            timeout=180,
        )
        first_chat.raise_for_status()
        second_chat = requests.post(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json=chat_request,
            timeout=180,
        )
        second_chat.raise_for_status()
        responses_call = requests.post(
            f"http://127.0.0.1:{args.port}/v1/responses",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": args.model_id,
                "input": "Reply with the single token OK.",
                "max_output_tokens": 8,
            },
            timeout=180,
        )
        responses_call.raise_for_status()
        metrics_after = parse_prometheus_text(server.metrics().text)
        cache_hit_metric = schema["prefix_cache_hits"]
        cache_hit_delta = metrics_after.get(cache_hit_metric, 0.0) - metrics_before.get(cache_hit_metric, 0.0)
        server.flush_prefix_cache()
        print(
            json.dumps(
                {
                    "health": health.status_code,
                    "models": models,
                    "schema": schema,
                    "chat_completion_ids": [first_chat.json().get("id"), second_chat.json().get("id")],
                    "responses_id": responses_call.json().get("id"),
                    "reset_prefix_cache_status": 200,
                    "prefix_cache_hits_delta": cache_hit_delta,
                },
                indent=2,
            )
        )
        return 0
    finally:
        if not args.keep_running:
            server.stop(missing_ok=True)


def cmd_switch_model(args: argparse.Namespace) -> int:
    server = _server(args)
    server.switch_model(model_id=args.model_id, enable_request_logging=args.enable_request_logging)
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
