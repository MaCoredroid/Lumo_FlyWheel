#!/usr/bin/env python3
"""FR13 gold-gate per-token MARGIN capture at greedy probe forks.

Mirrors the canonical rider battery EXACTLY (so the served-token fork indices
reproduce the banked [17,15,21,61]):
  /v1/completions, RAW text prompt (no chat template), return_token_ids=True,
  max_tokens=128, temperature=0.0, top_p=1.0, seed=1313,
  vllm_xargs={"fr10_decode_mode": "tree_mtp"}  (server default on BOTH launchers)

ADDS top-20 logprobs per generated position so a fork can be classified
NEAR-TIE (lossless) vs CLEAR-MARGIN (real loss).

Subcommands:
  capture  -- run the 4 pinned prompts, dump served ids + per-position top-20.
             Runs twice (rep1/rep2) for the within-boot determinism check.
  reduce   -- given tree.json + native.json, find first served-token fork per
             prompt and compute both-arm margins + classify.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _one_prompt(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    top_k: int,
    mode: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    # OpenAI completions logprobs object:
    #   tokens: [str]               (the sampled/served token strings)
    #   token_logprobs: [float]     (logprob of the served token)
    #   top_logprobs: [ {str: float}, ... ]  one dict per generated position
    return {
        "served_token_ids": choice.get("token_ids") or [],
        "served_tokens": lp.get("tokens") or [],
        "served_token_logprobs": lp.get("token_logprobs") or [],
        "top_logprobs": lp.get("top_logprobs") or [],
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text"),
    }


def capture(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    try:
        _post_json(args.endpoint, "/reset_prefix_cache", {}, timeout=30)
    except Exception:  # noqa: BLE001
        pass

    prompts = _read_prompts(args.prompts_file)

    # draft-shape engagement assertion via /metrics delta is recorded by the
    # caller; here we record the served stream + top-k for both reps.
    reps: list[list[dict[str, Any]]] = []
    for rep in range(2):
        rep_records: list[dict[str, Any]] = []
        for pid, prompt in enumerate(prompts):
            rec = _one_prompt(
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                args.top_k,
                args.mode,
                args.seed,
                args.request_timeout,
            )
            rec["prompt_id"] = pid
            rep_records.append(rec)
        reps.append(rep_records)

    # within-boot determinism: rep1 served ids == rep2 served ids per prompt
    within_boot_det = []
    for pid in range(len(prompts)):
        within_boot_det.append(
            reps[0][pid]["served_token_ids"] == reps[1][pid]["served_token_ids"]
        )

    metrics_text = ""
    try:
        metrics_text = _get_text(args.endpoint, "/metrics", timeout=10)
    except Exception:  # noqa: BLE001
        pass

    artifact = {
        "arm": args.arm,
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
        "records": reps[0],  # rep1 is canonical
        "records_rep2": reps[1],
        "within_boot_det_rep1_eq_rep2": within_boot_det,
        "metrics_excerpt": _metrics_excerpt(metrics_text),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "n_prompts": len(prompts),
                "within_boot_det_rep1_eq_rep2": within_boot_det,
                "served_lens": [len(r["served_token_ids"]) for r in reps[0]],
                "top_lens": [len(r["top_logprobs"]) for r in reps[0]],
            }
        )
    )
    return 0


def _metrics_excerpt(text: str) -> dict[str, float]:
    keep = {
        "vllm:spec_decode_num_draft_tokens_total": None,
        "vllm:spec_decode_num_drafts_total": None,
        "vllm:spec_decode_num_accepted_tokens_total": None,
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        for name in keep:
            if line.startswith(name + " ") or line.startswith(name + "{"):
                try:
                    out[name] = float(line.rsplit(" ", 1)[1])
                except Exception:  # noqa: BLE001
                    pass
    return out


def _top_map(top_logprobs_pos: dict[str, Any]) -> dict[str, float]:
    """Normalize a single-position top_logprobs dict to {token_str: logprob}."""
    out: dict[str, float] = {}
    for k, v in (top_logprobs_pos or {}).items():
        out[k] = float(v)
    return out


def _rank_and_lp(top_map: dict[str, float], token: str) -> tuple[int | None, float | None]:
    """Return (rank, logprob) of `token` within the top-k map. rank 1 = argmax.
    None if token not present in the captured top-k."""
    items = sorted(top_map.items(), key=lambda kv: kv[1], reverse=True)
    for i, (tok, lp) in enumerate(items):
        if tok == token:
            return i + 1, lp
    return None, None


def reduce(args: argparse.Namespace) -> int:
    tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))

    thr = args.threshold
    forks: list[dict[str, Any]] = []
    for pid in range(len(tree["records"])):
        tr = tree["records"][pid]
        na = native["records"][pid]
        t_ids = tr["served_token_ids"]
        n_ids = na["served_token_ids"]
        t_toks = tr["served_tokens"]
        n_toks = na["served_tokens"]
        t_tops = tr["top_logprobs"]
        n_tops = na["top_logprobs"]

        # first served-token fork
        fork_idx = None
        for i in range(min(len(t_ids), len(n_ids))):
            if t_ids[i] != n_ids[i]:
                fork_idx = i
                break
        if fork_idx is None:
            forks.append(
                {
                    "prompt_id": pid,
                    "fork_idx": -1,
                    "note": "no served-token fork within captured horizon",
                    "len_tree": len(t_ids),
                    "len_native": len(n_ids),
                }
            )
            continue

        i = fork_idx
        T_t = t_toks[i] if i < len(t_toks) else None
        T_n = n_toks[i] if i < len(n_toks) else None
        T_t_id = t_ids[i]
        T_n_id = n_ids[i]

        tree_map = _top_map(t_tops[i]) if i < len(t_tops) else {}
        native_map = _top_map(n_tops[i]) if i < len(n_tops) else {}

        # tree distribution: rank/lp of its own served token (argmax) and of native's token
        tr_rank_Tt, tr_lp_Tt = _rank_and_lp(tree_map, T_t)
        tr_rank_Tn, tr_lp_Tn = _rank_and_lp(tree_map, T_n)
        # native distribution: rank/lp of its own served token (argmax) and of tree's token
        na_rank_Tn, na_lp_Tn = _rank_and_lp(native_map, T_n)
        na_rank_Tt, na_lp_Tt = _rank_and_lp(native_map, T_t)

        # margins: positive = how much the arm's own choice beats the OTHER arm's token
        margin_tree = (
            (tr_lp_Tt - tr_lp_Tn)
            if (tr_lp_Tt is not None and tr_lp_Tn is not None)
            else None
        )
        margin_native = (
            (na_lp_Tn - na_lp_Tt)
            if (na_lp_Tn is not None and na_lp_Tt is not None)
            else None
        )

        # classification
        both_present = (margin_tree is not None) and (margin_native is not None)
        if both_present:
            near_tie = (margin_tree <= thr) and (margin_native <= thr)
            verdict = "NEAR-TIE" if near_tie else "CLEAR-MARGIN"
        else:
            # other arm's token absent from this arm's top-k => margin exceeds the
            # top-k logprob span. Treat as CLEAR-MARGIN-CANDIDATE and surface the
            # available bound so the reader can judge.
            verdict = "OTHER-ABSENT-FROM-TOPK"
            near_tie = None

        forks.append(
            {
                "prompt_id": pid,
                "fork_idx": fork_idx,
                "T_t_str": T_t,
                "T_t_id": T_t_id,
                "T_n_str": T_n,
                "T_n_id": T_n_id,
                "tree_rank_Tt": tr_rank_Tt,
                "tree_lp_Tt": tr_lp_Tt,
                "tree_rank_Tn": tr_rank_Tn,
                "tree_lp_Tn": tr_lp_Tn,
                "native_rank_Tn": na_rank_Tn,
                "native_lp_Tn": na_lp_Tn,
                "native_rank_Tt": na_rank_Tt,
                "native_lp_Tt": na_lp_Tt,
                "margin_tree": margin_tree,
                "margin_native": margin_native,
                "verdict": verdict,
                "tree_top5": sorted(tree_map.items(), key=lambda kv: kv[1], reverse=True)[:5],
                "native_top5": sorted(native_map.items(), key=lambda kv: kv[1], reverse=True)[:5],
                "served_context_excerpt_tree": (tr.get("text") or "")[: _ctx_char(tr, i)],
            }
        )

    any_clear = any(f.get("verdict") in ("CLEAR-MARGIN", "OTHER-ABSENT-FROM-TOPK") for f in forks)
    all_near = all(
        f.get("verdict") == "NEAR-TIE" for f in forks if f.get("fork_idx", -1) >= 0
    )
    overall = (
        "ALL_NEAR_TIE_LOSSLESS"
        if (all_near and not any_clear)
        else "CLEAR_MARGIN_REAL_LOSS"
        if any_clear
        else "INCONCLUSIVE"
    )

    result = {
        "threshold_logprob": thr,
        "threshold_note": args.threshold_note,
        "tree_arm": tree.get("arm"),
        "native_arm": native.get("arm"),
        "tree_within_boot_det": tree.get("within_boot_det_rep1_eq_rep2"),
        "native_within_boot_det": native.get("within_boot_det_rep1_eq_rep2"),
        "tree_metrics": tree.get("metrics_excerpt"),
        "native_metrics": native.get("metrics_excerpt"),
        "banked_fork_cross_check": args.banked,
        "forks": forks,
        "overall_verdict": overall,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _tokenize(endpoint: str, model: str, prompt: str, timeout: float) -> list[int]:
    data = _post_json(endpoint, "/tokenize", {"model": model, "prompt": prompt}, timeout=timeout)
    toks = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(toks, list):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return [int(t) for t in toks]


def teacher_force(args: argparse.Namespace) -> int:
    """Clean per-fork next-token distribution: feed prompt_ids + shared
    continuation token ids, decode max_tokens=1 with top-k logprobs.

    The tree spec-decode path attaches unreliable top_logprobs to accepted
    DRAFT tokens (server-side). A max_tokens=1 teacher-forced query emits a
    single verified token whose logprob IS reliable. The fork positions +
    shared continuation ids are read from a prior pair of capture artifacts so
    BOTH arms are forced on byte-identical token context."""
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    try:
        _post_json(args.endpoint, "/reset_prefix_cache", {}, timeout=30)
    except Exception:  # noqa: BLE001
        pass

    tree = json.loads(Path(args.tree).read_text(encoding="utf-8"))
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))
    prompts = tree["prompts"]

    forks: list[dict[str, Any]] = []
    for pid in range(len(prompts)):
        t_ids = tree["records"][pid]["served_token_ids"]
        n_ids = native["records"][pid]["served_token_ids"]
        fork_idx = None
        for i in range(min(len(t_ids), len(n_ids))):
            if t_ids[i] != n_ids[i]:
                fork_idx = i
                break
        if fork_idx is None:
            forks.append({"prompt_id": pid, "fork_idx": -1})
            continue
        assert t_ids[:fork_idx] == n_ids[:fork_idx], f"prefix mismatch p{pid}"
        forks.append(
            {
                "prompt_id": pid,
                "fork_idx": fork_idx,
                "shared_continuation_ids": n_ids[:fork_idx],
            }
        )

    out_records: list[dict[str, Any]] = []
    for f in forks:
        pid = f["prompt_id"]
        if f["fork_idx"] < 0:
            out_records.append({"prompt_id": pid, "fork_idx": -1})
            continue
        prompt_ids = _tokenize(args.endpoint, args.model, prompts[pid], args.request_timeout)
        context_ids = prompt_ids + f["shared_continuation_ids"]
        # repeat twice for within-boot determinism of the clean dist
        recs = []
        for rep in range(2):
            payload: dict[str, Any] = {
                "model": args.model,
                "prompt": context_ids,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": args.seed,
                "logprobs": args.top_k,
                "return_token_ids": True,
                "vllm_xargs": {"fr10_decode_mode": args.mode},
            }
            data = _post_json(args.endpoint, "/v1/completions", payload, timeout=args.request_timeout)
            choice = data["choices"][0]
            lp = choice.get("logprobs") or {}
            tops = lp.get("top_logprobs") or [{}]
            recs.append(
                {
                    "emitted_token_id": (choice.get("token_ids") or [None])[0],
                    "emitted_token": (lp.get("tokens") or [None])[0],
                    "emitted_token_logprob": (lp.get("token_logprobs") or [None])[0],
                    "top_logprobs": tops[0] if tops else {},
                }
            )
        out_records.append(
            {
                "prompt_id": pid,
                "fork_idx": f["fork_idx"],
                "context_len": len(context_ids),
                "rep1": recs[0],
                "rep2": recs[1],
                "within_boot_det": recs[0]["emitted_token_id"] == recs[1]["emitted_token_id"],
            }
        )

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "top_k": args.top_k,
        "ts": time.time(),
        "source_tree": args.tree,
        "source_native": args.native,
        "records": out_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "forks": [
                    {
                        "prompt_id": r["prompt_id"],
                        "fork_idx": r.get("fork_idx"),
                        "emitted": r.get("rep1", {}).get("emitted_token"),
                        "within_boot_det": r.get("within_boot_det"),
                    }
                    for r in out_records
                ],
            }
        )
    )
    return 0


def _ctx_char(rec: dict[str, Any], token_idx: int) -> int:
    toks = rec.get("served_tokens") or []
    if token_idx <= 0:
        return 0
    return sum(len(t) for t in toks[:token_idx])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--arm", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--prompts-file", required=True)
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--top-k", type=int, default=20)
    c.add_argument("--mode", default="tree_mtp")
    c.add_argument("--seed", type=int, default=1313)
    c.add_argument("--request-timeout", type=float, default=300)
    c.add_argument("--wait-health", type=float, default=120)
    c.set_defaults(func=capture)

    r = sub.add_parser("reduce")
    r.add_argument("--tree", required=True)
    r.add_argument("--native", required=True)
    r.add_argument("--out", required=True)
    r.add_argument(
        "--threshold",
        type=float,
        default=2.3026,
        help="logprob-margin threshold; <= => NEAR-TIE per arm (default ln(10)=2.3026, "
        "i.e. served token at most 10x more probable than the other arm's token).",
    )
    r.add_argument("--threshold-note", default="")
    r.add_argument("--banked", default="")
    r.set_defaults(func=reduce)

    tf = sub.add_parser("teacher-force")
    tf.add_argument("--arm", required=True)
    tf.add_argument("--out", required=True)
    tf.add_argument("--tree", required=True, help="tree capture json (for fork idx + shared prefix ids)")
    tf.add_argument("--native", required=True, help="native capture json (for fork idx + shared prefix ids)")
    tf.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    tf.add_argument("--model", default=DEFAULT_MODEL)
    tf.add_argument("--top-k", type=int, default=20)
    tf.add_argument("--mode", default="tree_mtp")
    tf.add_argument("--seed", type=int, default=1313)
    tf.add_argument("--request-timeout", type=float, default=300)
    tf.add_argument("--wait-health", type=float, default=120)
    tf.set_defaults(func=teacher_force)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
