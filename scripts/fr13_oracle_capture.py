#!/usr/bin/env python3
"""FR13 Boot-1 NON-MTP ORACLE: per-served-position clean teacher-forced argmax.

This is the shared lossless reference for Boot3's gold-margin probe + the node-7
sub-op ladder. The server MUST be a genuine no-spec boot
(fr10_launch_speed_server.sh + FR12_NO_SPECULATIVE_CONFIG=1, FLASH_ATTN, eager,
speculative_config=None) so regular greedy decode is the ground truth.

Two phases:
  1) STREAM: run the 4 pinned probes (RAW completions, greedy, max_tokens=128,
     seed=1313, top_p=1.0). Run twice (rep1/rep2) -> within-boot determinism.
     The served stream is the byte-identical reference prefix.
  2) TEACHER-FORCE: for EACH served position i in the rep1 stream, feed
     prompt_ids + served_ids[:i] and decode max_tokens=1 with top-k logprobs.
     The single emitted token IS the clean teacher-forced argmax at position i.
     Each position is repeated twice for per-position within-boot determinism.

Mirrors the canonical rider battery: /v1/completions, RAW text prompt (no chat
template), return_token_ids=True, temperature=0.0, top_p=1.0, seed=1313,
vllm_xargs={"fr10_decode_mode":"tree_mtp"} (server default; on the no-spec boot
the mode arg is inert since there is no speculator).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "http://127.0.0.1:9950"
DEFAULT_MODEL = "qwen3.6-27b"


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _get_text(endpoint: str, path: str, timeout: float) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _wait_health(endpoint: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            _get_text(endpoint, "/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not healthy in {timeout_s}s: {last}")


def _read_prompts(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"json prompts file must be a list of strings: {path}")
    prompts = [p for p in data if p]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def _tokenize(endpoint: str, model: str, prompt: str, timeout: float) -> list[int]:
    data = _post_json(endpoint, "/tokenize", {"model": model, "prompt": prompt}, timeout=timeout)
    toks = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(toks, list):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return [int(t) for t in toks]


def _stream_prompt(endpoint, model, prompt, max_tokens, top_k, mode, seed, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": top_k,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    return {
        "served_token_ids": choice.get("token_ids") or [],
        "served_tokens": lp.get("tokens") or [],
        "served_token_logprobs": lp.get("token_logprobs") or [],
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text"),
    }


def _tf_one(endpoint, model, context_ids, top_k, mode, seed, timeout):
    payload = {
        "model": model,
        "prompt": context_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": top_k,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    tops = lp.get("top_logprobs") or [{}]
    return {
        "argmax_token_id": (choice.get("token_ids") or [None])[0],
        "argmax_token": (lp.get("tokens") or [None])[0],
        "argmax_token_logprob": (lp.get("token_logprobs") or [None])[0],
        "top_logprobs": tops[0] if tops else {},
    }


def _metrics_excerpt(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if "spec_decode" in line and not line.startswith("#"):
            try:
                out[line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
            except Exception:  # noqa: BLE001
                pass
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--mode", default="tree_mtp")
    p.add_argument("--seed", type=int, default=1313)
    p.add_argument("--request-timeout", type=float, default=300)
    p.add_argument("--wait-health", type=float, default=120)
    args = p.parse_args()

    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    try:
        _post_json(args.endpoint, "/reset_prefix_cache", {}, timeout=30)
    except Exception:  # noqa: BLE001
        pass

    prompts = _read_prompts(args.prompts_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- engagement: ZERO spec_decode counters BEFORE any work ----
    metrics_pre = ""
    try:
        metrics_pre = _get_text(args.endpoint, "/metrics", timeout=10)
    except Exception:  # noqa: BLE001
        pass
    spec_pre = _metrics_excerpt(metrics_pre)

    # ---- PHASE 1: served stream, rep1/rep2 within-boot det ----
    stream_reps: list[list[dict[str, Any]]] = []
    for rep in range(2):
        recs = []
        for pid, prompt in enumerate(prompts):
            r = _stream_prompt(
                args.endpoint, args.model, prompt, args.max_tokens,
                args.top_k, args.mode, args.seed, args.request_timeout,
            )
            r["prompt_id"] = pid
            recs.append(r)
        stream_reps.append(recs)

    stream_within_boot_det = [
        stream_reps[0][pid]["served_token_ids"] == stream_reps[1][pid]["served_token_ids"]
        for pid in range(len(prompts))
    ]

    # ---- engagement: ZERO spec_decode counters AFTER the stream ----
    metrics_post = ""
    try:
        metrics_post = _get_text(args.endpoint, "/metrics", timeout=10)
    except Exception:  # noqa: BLE001
        pass
    spec_post = _metrics_excerpt(metrics_post)

    stream_artifact = {
        "phase": "stream",
        "arm": "non_mtp_oracle_nospec",
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "ts": time.time(),
        "prompts": prompts,
        "records": stream_reps[0],
        "records_rep2": stream_reps[1],
        "within_boot_det_rep1_eq_rep2": stream_within_boot_det,
        "spec_decode_counters_pre": spec_pre,
        "spec_decode_counters_post": spec_post,
    }
    (out_dir / "oracle_stream.json").write_text(
        json.dumps(stream_artifact, indent=2) + "\n", encoding="utf-8"
    )

    # ---- PHASE 2: per-served-position teacher-forced argmax ----
    # Resumable: each prompt's positions are checkpointed to a per-prompt file
    # (oracle_tf_p{pid}.json). Re-running skips already-complete prompts.
    tf_records: list[dict[str, Any]] = []
    total_positions = 0
    tf_det_positions = 0
    for pid, prompt in enumerate(prompts):
        served = stream_reps[0][pid]["served_token_ids"]
        served_toks = stream_reps[0][pid]["served_tokens"]
        ckpt = out_dir / f"oracle_tf_p{pid}.json"
        if ckpt.exists():
            done = json.loads(ckpt.read_text(encoding="utf-8"))
            if done.get("served_len") == len(served) and len(done.get("positions", [])) == len(served):
                tf_records.append(done)
                total_positions += len(done["positions"])
                tf_det_positions += sum(1 for q in done["positions"] if q["within_boot_det"])
                continue
        prompt_ids = _tokenize(args.endpoint, args.model, prompt, args.request_timeout)
        positions = []
        for i in range(len(served)):
            context_ids = prompt_ids + served[:i]
            rep1 = _tf_one(
                args.endpoint, args.model, context_ids, args.top_k,
                args.mode, args.seed, args.request_timeout,
            )
            rep2 = _tf_one(
                args.endpoint, args.model, context_ids, args.top_k,
                args.mode, args.seed, args.request_timeout,
            )
            det = rep1["argmax_token_id"] == rep2["argmax_token_id"]
            matches_served = rep1["argmax_token_id"] == served[i]
            positions.append(
                {
                    "served_position": i,
                    "context_len": len(context_ids),
                    "served_token_id": served[i],
                    "served_token": served_toks[i] if i < len(served_toks) else None,
                    "oracle_argmax_token_id": rep1["argmax_token_id"],
                    "oracle_argmax_token": rep1["argmax_token"],
                    "oracle_argmax_logprob": rep1["argmax_token_logprob"],
                    "oracle_top_logprobs": rep1["top_logprobs"],
                    "within_boot_det": det,
                    "oracle_argmax_matches_served": matches_served,
                }
            )
            total_positions += 1
            if det:
                tf_det_positions += 1
        rec = {
            "prompt_id": pid,
            "prompt_len_tokens": len(prompt_ids),
            "served_len": len(served),
            "positions": positions,
        }
        ckpt.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        tf_records.append(rec)

    tf_artifact = {
        "phase": "teacher_force_per_position",
        "arm": "non_mtp_oracle_nospec",
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "top_k": args.top_k,
        "ts": time.time(),
        "prompts": prompts,
        "records": tf_records,
        "total_positions": total_positions,
        "tf_within_boot_det_positions": tf_det_positions,
        "tf_all_positions_det": tf_det_positions == total_positions,
        "spec_decode_counters_post_tf": _metrics_excerpt(
            _get_text(args.endpoint, "/metrics", timeout=10)
        ),
    }
    (out_dir / "oracle_teacher_forced.json").write_text(
        json.dumps(tf_artifact, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "stream_out": str(out_dir / "oracle_stream.json"),
        "tf_out": str(out_dir / "oracle_teacher_forced.json"),
        "n_prompts": len(prompts),
        "served_lens": [len(r["served_token_ids"]) for r in stream_reps[0]],
        "stream_within_boot_det_rep1_eq_rep2": stream_within_boot_det,
        "total_positions": total_positions,
        "tf_within_boot_det_positions": tf_det_positions,
        "tf_all_positions_det": tf_det_positions == total_positions,
        "n_spec_decode_counters_pre": len(spec_pre),
        "n_spec_decode_counters_post": len(spec_post),
        "spec_decode_counters_post": spec_post,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
