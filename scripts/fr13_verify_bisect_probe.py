#!/usr/bin/env python3
"""FR13 deep-row verify-gap BISECT classifier (per-served-position clean-argmax).

For ONE tree server (whatever speed-fix flag combo it booted with):
  1. capture the served stream for the 4 pinned probes (rep1+rep2, within-boot det)
     via the canonical rider battery (RAW text, max_tokens=128, greedy seed 1313,
     fr10_decode_mode=tree_mtp).
  2. for EACH served position p (1..len), teacher-force a CLEAN max_tokens=1 forward
     on the byte-identical served prefix (prompt_ids + served[:p]) on the SAME tree
     server, capture top-20 logprobs. The tree server's own max_tokens=1 forward is
     the validated clean reference (close workflow wf_3c6f5c0a: native E5 teacher-
     forced==served 4/4; the tree max_tokens=1 forward gives the clean argmax).
  3. flag CLEAR-MARGIN flip iff served_token_id != clean_argmax_id AND
     (clean_argmax_lp - served_token_lp) > THRESHOLD (default 1.0 nat). If the served
     token is absent from the clean top-20, the deviation exceeds the top-20 span =>
     CLEAR-MARGIN (lower-bounded).
  4. count clear-margin flips per prompt + total, and tag whether the two KNOWN deep-row
     structural-boundary flips (p2 pos21 ' code', p3 the code-fence/'Let' flip) are
     present in THIS combo's stream (positions may move when flags change the stream,
     so we tag by (served_token, clean_argmax) identity + structural context, AND we
     report every clear-margin flip with its context for manual deep-row attribution).

Usage:
  capture  --arm <combo> --prompts-file ... --out <capture.json>
  classify --capture <capture.json> --out <classify.json> [--threshold 1.0] [--prefix-only]
           (classify re-uses the SAME running server to teacher-force; run while the
            combo server is still up.)
  bisect   --capture <capture.json> --out <bisect.json> [--threshold 1.0]
           (capture + classify in one pass against the running server.)
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


def _post(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _get(endpoint: str, path: str, timeout: float) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _wait_health(endpoint: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            _get(endpoint, "/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not healthy in {timeout_s}s: {last}")


def _read_prompts(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [p for p in data if p]


def _tokenize(endpoint: str, model: str, prompt: str, timeout: float) -> list[int]:
    d = _post(endpoint, "/tokenize", {"model": model, "prompt": prompt}, timeout=timeout)
    return [int(t) for t in d["tokens"]]


def _serve_one(endpoint, model, prompt, max_tokens, mode, seed, timeout):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "seed": seed, "logprobs": 20,
        "return_token_ids": True, "vllm_xargs": {"fr10_decode_mode": mode},
    }
    d = _post(endpoint, "/v1/completions", payload, timeout=timeout)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "served_token_ids": ch.get("token_ids") or [],
        "served_tokens": (lp.get("tokens") or []),
        "text": ch.get("text"),
    }


def _metrics_excerpt(text: str) -> dict[str, float]:
    keep = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    out: dict[str, float] = {}
    for line in text.splitlines():
        for name in keep:
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except Exception:  # noqa: BLE001
                    pass
    return out


def capture(args):
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    try:
        _post(args.endpoint, "/reset_prefix_cache", {}, 30)
    except Exception:  # noqa: BLE001
        pass
    prompts = _read_prompts(args.prompts_file)
    reps = []
    for _ in range(2):
        recs = []
        for pid, prompt in enumerate(prompts):
            r = _serve_one(args.endpoint, args.model, prompt, args.max_tokens,
                           args.mode, args.seed, args.request_timeout)
            r["prompt_id"] = pid
            recs.append(r)
        reps.append(recs)
    within = [reps[0][p]["served_token_ids"] == reps[1][p]["served_token_ids"]
              for p in range(len(prompts))]
    metrics = ""
    try:
        metrics = _get(args.endpoint, "/metrics", 10)
    except Exception:  # noqa: BLE001
        pass
    art = {
        "arm": args.arm, "endpoint": args.endpoint, "model": args.model,
        "mode": args.mode, "seed": args.seed, "max_tokens": args.max_tokens,
        "ts": time.time(), "prompts": prompts, "records": reps[0],
        "records_rep2": reps[1], "within_boot_det_rep1_eq_rep2": within,
        "metrics_excerpt": _metrics_excerpt(metrics),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "out": str(out),
                      "within_boot_det": within,
                      "served_lens": [len(r["served_token_ids"]) for r in reps[0]]}))
    return 0


def _topmap(d):
    return {k: float(v) for k, v in (d or {}).items()}


def classify(args):
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    cap = json.loads(Path(args.capture).read_text(encoding="utf-8"))
    prompts = cap["prompts"]
    thr = args.threshold
    try:
        _post(args.endpoint, "/reset_prefix_cache", {}, 30)
    except Exception:  # noqa: BLE001
        pass

    all_flips = []
    per_prompt = []
    for pid in range(len(prompts)):
        served_ids = cap["records"][pid]["served_token_ids"]
        served_toks = cap["records"][pid]["served_tokens"]
        prompt_ids = _tokenize(args.endpoint, args.model, prompts[pid], args.request_timeout)
        npos = len(served_ids)
        if args.max_positions and args.max_positions > 0:
            npos = min(npos, args.max_positions)
        flips_this = []
        # served[0] is the prefill bonus (first generated token); the context to predict
        # served position p is prompt + served[:p]. p=0 is predicted by the prompt alone.
        for p in range(npos):
            ctx = prompt_ids + served_ids[:p]
            payload = {
                "model": args.model, "prompt": ctx, "max_tokens": 1,
                "temperature": 0.0, "top_p": 1.0, "seed": args.seed,
                "logprobs": 20, "return_token_ids": True,
                "vllm_xargs": {"fr10_decode_mode": args.mode},
            }
            d = _post(args.endpoint, "/v1/completions", payload, args.request_timeout)
            ch = d["choices"][0]
            lp = ch.get("logprobs") or {}
            clean_id = (ch.get("token_ids") or [None])[0]
            clean_tok = (lp.get("tokens") or [None])[0]
            clean_lp = (lp.get("token_logprobs") or [None])[0]
            top = _topmap((lp.get("top_logprobs") or [{}])[0])
            served_id = served_ids[p]
            served_tok = served_toks[p] if p < len(served_toks) else None
            # find served token logprob in the clean dist (by string key; the top_logprobs
            # dict is keyed by token string in OpenAI completions). fall back: absent.
            served_clean_lp = top.get(served_tok) if served_tok is not None else None
            is_match = (served_id == clean_id)
            if served_clean_lp is not None:
                dev = clean_lp - served_clean_lp
                absent = False
            else:
                # served token not in clean top-20 => deviation exceeds the top-20 span.
                # lower-bound dev by (clean_lp - min(top-20 lp)); flag CLEAR-MARGIN.
                if top:
                    dev = clean_lp - min(top.values())
                else:
                    dev = float("inf")
                absent = True
            clear_margin = (not is_match) and (dev > thr)
            if clear_margin:
                ctx_str = "".join(served_toks[max(0, p - 6):p]) if served_toks else ""
                flips_this.append({
                    "pos": p,
                    "served_tok": served_tok, "served_id": served_id,
                    "clean_argmax_tok": clean_tok, "clean_argmax_id": clean_id,
                    "clean_argmax_lp": clean_lp,
                    "served_clean_lp": served_clean_lp,
                    "served_absent_from_top20": absent,
                    "deviation_nat": round(dev, 4),
                    "context_excerpt": ctx_str,
                })
        per_prompt.append({"prompt_id": pid, "n_positions": npos,
                           "clear_margin_flips": len(flips_this), "flips": flips_this})
        for f in flips_this:
            f2 = dict(f); f2["prompt_id"] = pid
            all_flips.append(f2)

    # tag the two known deep-row structural-boundary flips by identity:
    #  p2: served ' code'(1970) but clean argmax ' files'(3425)
    #  p3: served 'Let'(9764) but clean argmax a code-fence token ('```' 71093 / 44780)
    known = {
        "p2_code_vs_files": any(
            f["prompt_id"] == 2 and f["served_id"] == 1970 and f["clean_argmax_id"] == 3425
            for f in all_flips),
        "p3_Let_vs_codefence": any(
            f["prompt_id"] == 3 and f["served_id"] == 9764 and f["clean_argmax_id"] in (71093, 44780)
            for f in all_flips),
    }
    total = sum(pp["clear_margin_flips"] for pp in per_prompt)
    result = {
        "arm": cap.get("arm"),
        "threshold_nat": thr,
        "within_boot_det_rep1_eq_rep2": cap.get("within_boot_det_rep1_eq_rep2"),
        "metrics_excerpt": cap.get("metrics_excerpt"),
        "per_prompt_clear_margin_flip_counts": [pp["clear_margin_flips"] for pp in per_prompt],
        "total_clear_margin_flips": total,
        "known_deep_row_flips_present": known,
        "per_prompt": per_prompt,
        "all_clear_margin_flips": all_flips,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "arm": cap.get("arm"),
        "total_clear_margin_flips": total,
        "per_prompt": result["per_prompt_clear_margin_flip_counts"],
        "known_deep_row_flips_present": known,
        "out": str(out),
    }, indent=2))
    return 0


def bisect(args):
    capture(args)
    args.capture = args.out
    # classify writes to args.classify_out
    cls_args = argparse.Namespace(**vars(args))
    cls_args.out = args.classify_out
    cls_args.wait_health = 0
    classify(cls_args)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--arm", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--prompts-file", required=True)
    c.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--mode", default="tree_mtp")
    c.add_argument("--seed", type=int, default=1313)
    c.add_argument("--request-timeout", type=float, default=300)
    c.add_argument("--wait-health", type=float, default=180)
    c.set_defaults(func=capture)

    cl = sub.add_parser("classify")
    cl.add_argument("--capture", required=True)
    cl.add_argument("--out", required=True)
    cl.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    cl.add_argument("--model", default=DEFAULT_MODEL)
    cl.add_argument("--mode", default="tree_mtp")
    cl.add_argument("--seed", type=int, default=1313)
    cl.add_argument("--threshold", type=float, default=1.0)
    cl.add_argument("--max-positions", type=int, default=0)
    cl.add_argument("--request-timeout", type=float, default=300)
    cl.add_argument("--wait-health", type=float, default=0)
    cl.set_defaults(func=classify)

    b = sub.add_parser("bisect")
    b.add_argument("--arm", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--classify-out", required=True)
    b.add_argument("--prompts-file", required=True)
    b.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--mode", default="tree_mtp")
    b.add_argument("--seed", type=int, default=1313)
    b.add_argument("--threshold", type=float, default=1.0)
    b.add_argument("--max-positions", type=int, default=0)
    b.add_argument("--request-timeout", type=float, default=300)
    b.add_argument("--wait-health", type=float, default=180)
    b.set_defaults(func=bisect)
    return p


def main():
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
