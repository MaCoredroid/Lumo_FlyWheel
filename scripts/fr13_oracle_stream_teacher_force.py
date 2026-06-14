#!/usr/bin/env python3
"""FR13 BOOT-2 oracle: teacher-force the NON-MTP no-spec oracle at EACH served
position of a SAME-BUILD tree stream, then compute the CHANNEL-2 flips.

CHANNEL-2 flip = the tree-served token at position i differs from the clean
no-spec-oracle argmax conditioned on the IDENTICAL prefix (prompt_ids +
served_token_ids[:i]). A flip is "clear margin" when the oracle prefers its own
argmax over the tree-served token by deviation > THRESHOLD nat (default 1.0).

The oracle prefix is built from the EXACT same-build tree stream's served ids
(NOT a banked oracle, NOT a re-tokenized text) so the comparison is on this
boot's own stream per feedback_no_cross_boot_byte_gate.

Runs the teacher-force twice (rep1/rep2) for within-boot determinism of the
clean per-position argmax.

Subcommands:
  run     -- boot must already be up; teacher-force every served position of
             the tree stream, dump per-position oracle argmax + top-k + the
             channel-2 flip classification.
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


def _tokenize(endpoint: str, model: str, prompt: str, timeout: float) -> list[int]:
    data = _post_json(endpoint, "/tokenize", {"model": model, "prompt": prompt}, timeout=timeout)
    toks = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(toks, list):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return [int(t) for t in toks]


def _force_one(
    endpoint: str,
    model: str,
    context_ids: list[int],
    top_k: int,
    mode: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
        "argmax_logprob": (lp.get("token_logprobs") or [None])[0],
        "top_logprobs": tops[0] if tops else {},
    }


def _metrics_excerpt(text: str) -> dict[str, float]:
    keep = [
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    ]
    out: dict[str, float] = {}
    for line in text.splitlines():
        for name in keep:
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except Exception:  # noqa: BLE001
                    pass
    return out


def _lp_of(top_map: dict[str, Any], token: str | None) -> float | None:
    if token is None:
        return None
    if token in top_map:
        return float(top_map[token])
    return None


def run(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)

    tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))
    prompts = tree["prompts"]
    records = tree["records"]

    # zero-spec engagement: snapshot the spec counters BEFORE any oracle query.
    metrics_before = {}
    try:
        metrics_before = _metrics_excerpt(_get_text(args.endpoint, "/metrics", timeout=10))
    except Exception:  # noqa: BLE001
        pass

    try:
        _post_json(args.endpoint, "/reset_prefix_cache", {}, timeout=30)
    except Exception:  # noqa: BLE001
        pass

    thr = args.threshold
    per_prompt: list[dict[str, Any]] = []
    total_positions = 0
    total_flips = 0
    total_clear_flips = 0

    for pid in range(len(records)):
        prompt = prompts[pid]
        served_ids = records[pid]["served_token_ids"]
        served_toks = records[pid].get("served_tokens") or [None] * len(served_ids)
        served_tops = records[pid].get("top_logprobs") or [{}] * len(served_ids)
        prompt_ids = _tokenize(args.endpoint, args.model, prompt, args.request_timeout)

        positions: list[dict[str, Any]] = []
        flips: list[dict[str, Any]] = []
        for i in range(len(served_ids)):
            context_ids = prompt_ids + served_ids[:i]
            r1 = _force_one(
                args.endpoint, args.model, context_ids, args.top_k, args.mode,
                args.seed, args.request_timeout,
            )
            r2 = _force_one(
                args.endpoint, args.model, context_ids, args.top_k, args.mode,
                args.seed, args.request_timeout,
            )
            within = r1["argmax_token_id"] == r2["argmax_token_id"]
            served_id = served_ids[i]
            oracle_id = r1["argmax_token_id"]
            flip = served_id != oracle_id
            # margin (nat): oracle's own argmax logprob minus the served token's
            # logprob in the SAME clean oracle distribution.
            oracle_top = r1["top_logprobs"] or {}
            served_str = served_toks[i] if i < len(served_toks) else None
            lp_oracle_argmax = r1["argmax_logprob"]
            lp_served_in_oracle = _lp_of(oracle_top, served_str)
            if lp_oracle_argmax is not None and lp_served_in_oracle is not None:
                deviation = lp_oracle_argmax - lp_served_in_oracle
            else:
                deviation = None  # served token outside oracle top-k => >top-k span
            clear = flip and (
                deviation is None or deviation > thr
            )
            pos_rec = {
                "pos": i,
                "served_token_id": served_id,
                "served_token": served_str,
                "oracle_argmax_id": oracle_id,
                "oracle_argmax_token": r1["argmax_token"],
                "oracle_argmax_logprob": lp_oracle_argmax,
                "served_lp_in_oracle": lp_served_in_oracle,
                "deviation_nat": deviation,
                "flip": flip,
                "clear_margin": clear,
                "within_boot_det": within,
            }
            if flip:
                # full top-5 of oracle for the flip, for the ladder/report
                pos_rec["oracle_top5"] = sorted(
                    ((k, float(v)) for k, v in oracle_top.items()),
                    key=lambda kv: kv[1], reverse=True,
                )[:5]
                pos_rec["tree_served_top5"] = sorted(
                    ((k, float(v)) for k, v in (served_tops[i] or {}).items()),
                    key=lambda kv: kv[1], reverse=True,
                )[:5]
                flips.append(pos_rec)
                total_flips += 1
                if clear:
                    total_clear_flips += 1
            positions.append(pos_rec)
            total_positions += 1

        within_all = all(p["within_boot_det"] for p in positions)
        per_prompt.append(
            {
                "prompt_id": pid,
                "served_len": len(served_ids),
                "prompt_token_len": len(prompt_ids),
                "n_flips": len(flips),
                "n_clear_flips": sum(1 for f in flips if f["clear_margin"]),
                "within_boot_det_all": within_all,
                "flips": flips,
                "positions": positions,
            }
        )

    metrics_after = {}
    try:
        metrics_after = _metrics_excerpt(_get_text(args.endpoint, "/metrics", timeout=10))
    except Exception:  # noqa: BLE001
        pass

    # zero-spec engagement: the spec counters must NOT advance during oracle work
    spec_delta = {}
    for k in set(list(metrics_before) + list(metrics_after)):
        spec_delta[k] = metrics_after.get(k, 0.0) - metrics_before.get(k, 0.0)

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "top_k": args.top_k,
        "threshold_nat": thr,
        "ts": time.time(),
        "source_tree": args.tree,
        "spec_metrics_before": metrics_before,
        "spec_metrics_after": metrics_after,
        "spec_metrics_delta_during_oracle": spec_delta,
        "total_positions": total_positions,
        "total_flips": total_flips,
        "total_clear_margin_flips": total_clear_flips,
        "within_boot_det_all_prompts": all(
            p["within_boot_det_all"] for p in per_prompt
        ),
        "per_prompt": per_prompt,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "total_positions": total_positions,
                "total_flips": total_flips,
                "total_clear_margin_flips": total_clear_flips,
                "per_prompt_flips": [(p["prompt_id"], p["n_flips"], p["n_clear_flips"]) for p in per_prompt],
                "within_boot_det_all_prompts": artifact["within_boot_det_all_prompts"],
                "spec_metrics_delta_during_oracle": spec_delta,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tree", required=True, help="SAME-BUILD tree served_stream.json")
    r.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--mode", default="tree_mtp")
    r.add_argument("--seed", type=int, default=1313)
    r.add_argument("--threshold", type=float, default=1.0, help="clear-margin nat threshold")
    r.add_argument("--request-timeout", type=float, default=300)
    r.add_argument("--wait-health", type=float, default=120)
    r.set_defaults(func=run)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
