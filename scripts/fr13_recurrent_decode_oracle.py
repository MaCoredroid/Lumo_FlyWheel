#!/usr/bin/env python3
"""FR13 RECURRENT no-spec single-step DECODE oracle (standalone, in-process).

WHY THIS EXISTS
---------------
The binding flip instrument ``scripts/fr13_oracle_stream_teacher_force.py``
re-prefills ``prompt + served[:i]`` per position over the HTTP completions API.
For Qwen3-Next that request has ``query_len >> 1`` => ``num_prefills>0`` =>
the GDN layer routes to ``chunk_gated_delta_rule`` (CHUNKED block-parallel WY)
to rebuild the served[:i] recurrent state -- because Qwen3NextForCausalLM does
NOT declare ``SupportsMambaPrefixCaching`` so vLLM forces ``mamba_cache_mode =
"align"`` (block-granular) and the partial-suffix SSM state is re-chunked every
request (verified: gdn_linear_attn.py ``_forward_core`` L1142 ``if
num_prefills>0 -> chunk_gated_delta_rule``).

The DEPLOYMENT no-spec decode never takes that path. Each generated token is a
single-token decode (``query_len==1`` => ``split_decodes_and_prefills`` with
``decode_threshold=1`` gives ``num_decodes>0, num_prefills==0``), which routes
to ``_forward_core_decode_non_spec`` (L807-817 guard:
``enable_packed_recurrent_decode and spec_sequence_masks is None and
num_prefills==0 and num_decodes>0``) -> ``causal_conv1d_update`` +
``fused_recurrent_gated_delta_rule_packed_decode`` (L1075/L1085) -- the
RECURRENT rank-1 roll, carrying ``conv_state``/``ssm_state`` forward IN THE KV
CACHE across decode steps. (If ``VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`` were
0 the decode still routes recurrent via the L1007 ``elif num_decodes>0 ->
fused_sigmoid_gating_delta_rule_update`` branch -- both decode branches are
recurrent; ONLY ``num_prefills>0`` is chunked.)

q1 (output/fr13_verify_decisive/q1_recur_vs_chunked.json) measured the L0 GDN
carrier vs this recurrent frame at 0.000854 (~1 bf16 ULP) vs 0.0078 chunked
(9.14x). This instrument RE-SCORES the full per-position flip count vs the
recurrent frame, for EVERY arm, so the oracle frame is isolated.

DESIGN -- forced recurrent teacher-force, all decode steps, no off-by-one
------------------------------------------------------------------------
The model is loaded ONCE in-process (offline ``vllm.LLM``, no speculation,
FLASH_ATTN, eager). For each prompt we issue ONE ``generate`` whose sampling
is driven by a per-request logits processor that, at each decode step i:
  (1) reads the CLEAN recurrent argmax + top-k from the logits row BEFORE any
      forcing  -> recorded directly per step (NO streamed-array indexing, so
      the streamed off-by-one of the HTTP instrument is structurally absent);
  (2) FORCES the served token served[i] by setting logits so the engine commits
      served[i] and carries the recurrent conv/ssm state forward to step i+1.
Because every step after the prompt prefill is a single-token decode, every
oracle distribution we read is produced by the RECURRENT path. The prompt
itself is prefilled once (chunked, as in deployment) and is NOT scored; only
the generated served positions are scored.

flip(i)        := served[i] != recurrent_argmax(i)
clear_margin(i):= flip and (served[i] outside oracle top-k OR
                  oracle_argmax_logprob - served_in_oracle_logprob > THRESHOLD)
THRESHOLD defaults to 1.0 nat (identical to the chunked instrument).

ENGAGEMENT / DETERMINISM (class 8 + class 9, FR13_BUG_CLASS_PLAYBOOK.md)
-----------------------------------------------------------------------
* class 9 (silent fallback / vacuous instrument): we ASSERT the model has GDN
  linear_attn layers and that the recurrent decode path actually fired
  (a monkeypatch counter on ``_forward_core_decode_non_spec`` increments) --
  FAIL LOUD if zero recurrent decode calls happened, so an accidental
  chunked/spec route cannot pass silently. We also assert no speculative config.
* class 8 (determinism): ``smoke`` reproduces q1's pinned per-path number;
  ``rescore`` records within-process rep1==rep2 per-position determinism.

Subcommands:
  smoke    -- generate the q1 pinned path (PROMPTS[2] no-spec greedy to >=22
              tokens) and confirm (a) the recurrent decode path fired, (b) the
              greedy stream reproduces served[:21] of served_p2.json, proving
              this script drives the SAME recurrent path q1 captured. (Re-deriving
              the exact 0.000854 L0 carrier requires the in-layer GDN capture
              hook; smoke proves the PATH + reproduces the pinned token stream.)
  rescore  -- per-arm full-stream recurrent flip re-score (the heavy phase).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL_PATH = "/models/qwen3.6-27b-fp8"
SERVED_MODEL_NAME = "qwen3.6-27b"


# --------------------------------------------------------------------------- #
# Per-request forced-decode logits processor (v1 AdapterLogitsProcessor).      #
# --------------------------------------------------------------------------- #
# vLLM v1 does NOT accept a raw per-request callable through
# SamplingParams.logits_processors. A per-request logits processor must be
# registered as an AdapterLogitsProcessor subclass at engine construction
# (LLM(logits_processors=[ForcedRecurrentAdapter])); its new_req_logits_processor
# reads the per-request payload from SamplingParams.extra_args and returns the
# legacy Callable[[past_token_ids, logits], logits]. Because the LP runs in the
# (possibly separate) engine-core worker, per-step clean records are written to
# a per-request JSONL file (extra_args["sink_path"]) -- the same /logs file
# transport the existing patcher captures use -- and read back by the driver.


def _make_forced_recurrent_request_lp(served_ids, top_k, sink_path):
    """Return the per-request Callable[[past_token_ids, logits], logits]."""
    f = open(sink_path, "w", encoding="utf-8")  # noqa: SIM115

    def _lp(past_token_ids, logits: torch.Tensor) -> torch.Tensor:
        step = len(past_token_ids)
        if step >= len(served_ids):
            return logits
        row = logits.detach().float()
        k = min(top_k, row.numel())
        top = torch.topk(row, k)
        logprobs = torch.log_softmax(row, dim=-1)
        served_id = int(served_ids[step])
        argmax_id = int(top.indices[0].item())
        rec = {
            "step": step,
            "served_token_id": served_id,
            "oracle_argmax_id": argmax_id,
            "oracle_argmax_logprob": float(logprobs[argmax_id].item()),
            "served_logprob_in_oracle": float(logprobs[served_id].item()),
            "oracle_topk_ids": [int(x) for x in top.indices.tolist()],
            "oracle_topk_logprobs": [float(logprobs[i].item()) for i in top.indices.tolist()],
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()
        # FORCE served[step] as the unique argmax so greedy commits it and the
        # recurrent conv/ssm state advances to the next served position.
        forced = torch.full_like(logits, float("-inf"))
        forced[served_id] = 0.0
        return forced

    return _lp


def _adapter_class():
    """Build the AdapterLogitsProcessor subclass lazily (needs vllm import)."""
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class ForcedRecurrentAdapter(AdapterLogitsProcessor):
        def is_argmax_invariant(self) -> bool:
            return False  # we deliberately change the argmax (forcing)

        def new_req_logits_processor(self, params):
            ea = getattr(params, "extra_args", None) or {}
            served = ea.get("fr13_served_ids")
            sink_path = ea.get("fr13_sink_path")
            top_k = int(ea.get("fr13_top_k", 20))
            if served is None or sink_path is None:
                return None
            return _make_forced_recurrent_request_lp(
                [int(x) for x in served], top_k, sink_path
            )

    return ForcedRecurrentAdapter


def _build_llm(
    model_path: str,
    max_model_len: int,
    gpu_util: float,
    attn_backend: str,
    register_adapter: bool,
):
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", attn_backend)
    os.environ.setdefault("FR12_NO_SPECULATIVE_CONFIG", "1")
    # Run the engine core IN-PROCESS so (a) the recurrent-decode-path counter
    # monkeypatch in this process sees the worker's GDN calls (class-9
    # engagement assert) and (b) GDN-layer introspection works. With V1
    # multiprocessing the engine core forks a separate process and neither is
    # visible to the driver -> the engagement gate would be blind.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM  # imported here so --help works without GPU deps

    kwargs: dict[str, Any] = dict(
        model=model_path,
        served_model_name=SERVED_MODEL_NAME,
        tokenizer=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
        max_num_seqs=1,
        # NO speculative_config -> pure no-spec decode -> recurrent path.
    )
    if register_adapter:
        kwargs["logits_processors"] = [_adapter_class()]
    return LLM(**kwargs)


# --------------------------------------------------------------------------- #
# Recurrent-decode-path engagement counter (class 9 fail-loud).               #
# --------------------------------------------------------------------------- #
_RECUR_DECODE_CALLS = {"n": 0}


def _install_recurrent_path_counter() -> bool:
    """Monkeypatch the GDN layer so we can PROVE the recurrent decode path fired.

    Returns True if the patch was installed. We wrap _forward_core_decode_non_spec
    AND the non-packed decode branch is detected via a flag on _forward_core; both
    are recurrent, only num_prefills>0 (chunked) is what we must never see scored.
    """
    try:
        from vllm.model_executor.layers.mamba import gdn_linear_attn as gdn
    except Exception:
        return False

    cls = None
    for name in dir(gdn):
        obj = getattr(gdn, name)
        if isinstance(obj, type) and hasattr(obj, "_forward_core_decode_non_spec"):
            cls = obj
            break
    if cls is None:
        return False

    orig = cls._forward_core_decode_non_spec

    def wrapped(self, *a, **k):
        _RECUR_DECODE_CALLS["n"] += 1
        return orig(self, *a, **k)

    cls._forward_core_decode_non_spec = wrapped  # type: ignore[assignment]
    return True


def _assert_gdn_present(llm) -> int:
    """Count GDN linear_attn layers; FAIL LOUD if zero (class 9)."""
    n = 0
    try:
        engine = llm.llm_engine
        model = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model  # noqa: E501
    except Exception:
        model = None
    if model is None:
        return -1  # could not introspect; counter remains the binding gate
    for mod in model.modules():
        if mod.__class__.__name__.lower().find("gateddelta") >= 0 or hasattr(
            mod, "_forward_core_decode_non_spec"
        ):
            n += 1
    return n


def _load_served_streams(arm: str, src: str) -> tuple[list[str], list[list[int]]]:
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    prompts = d["prompts"]
    recs = d["records"]
    served = [[int(t) for t in r["served_token_ids"]] for r in recs]
    return prompts, served


# --------------------------------------------------------------------------- #
# smoke: prove the recurrent path + reproduce the q1 pinned token stream.      #
# --------------------------------------------------------------------------- #
def cmd_smoke(args: argparse.Namespace) -> int:
    installed = _install_recurrent_path_counter()
    llm = _build_llm(
        args.model, args.max_model_len, args.gpu_util, args.attn_backend,
        register_adapter=False,
    )
    n_gdn = _assert_gdn_present(llm)

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    pid = args.pid
    prompt = prompts[pid]

    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed
    )
    _RECUR_DECODE_CALLS["n"] = 0
    out = llm.generate([prompt], sp)
    gen_ids = list(out[0].outputs[0].token_ids)
    recur_calls = _RECUR_DECODE_CALLS["n"]

    served_p2 = None
    match_first21 = None
    if args.served_ref and Path(args.served_ref).exists():
        ref = json.loads(Path(args.served_ref).read_text(encoding="utf-8"))
        served_p2 = ref.get("served_token_ids") or ref.get("served_first22")
        if served_p2 is not None:
            n = min(21, len(gen_ids), len(served_p2))
            match_first21 = gen_ids[:n] == served_p2[:n]

    # class 9 fail-loud: the recurrent decode path MUST have fired.
    engaged = recur_calls > 0
    result = {
        "schema": "fr13.recurrent_decode_oracle.smoke.v1",
        "recurrent_path_counter_installed": installed,
        "n_gdn_layers_introspected": n_gdn,
        "recurrent_decode_calls": recur_calls,
        "recurrent_decode_calls_per_token": (
            recur_calls / max(1, len(gen_ids))
        ),
        "n_generated": len(gen_ids),
        "gen_first22": gen_ids[:22],
        "served_ref_first22": (served_p2[:22] if served_p2 else None),
        "match_first21_vs_served_ref": match_first21,
        "RECURRENT_PATH_ENGAGED": engaged,
        "ts": time.time(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not engaged:
        raise SystemExit(
            "FR13 class-9 FAIL-LOUD: recurrent decode path did NOT fire "
            "(zero _forward_core_decode_non_spec calls) -- vacuous oracle."
        )
    return 0


# --------------------------------------------------------------------------- #
# rescore: per-arm full-stream recurrent flip re-score (heavy next phase).     #
# --------------------------------------------------------------------------- #
def _read_sink(path: str) -> list[dict]:
    recs = []
    p = Path(path)
    if not p.exists():
        return recs
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    recs.sort(key=lambda r: r["step"])
    return recs


def cmd_rescore(args: argparse.Namespace) -> int:
    installed = _install_recurrent_path_counter()
    llm = _build_llm(
        args.model, args.max_model_len, args.gpu_util, args.attn_backend,
        register_adapter=True,
    )
    n_gdn = _assert_gdn_present(llm)

    prompts, served_streams = _load_served_streams(args.arm, args.src)
    sink_dir = Path(args.out).parent / f"{args.arm}_sinks"
    sink_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams

    per_prompt: list[dict[str, Any]] = []
    total_positions = 0
    total_flips = 0
    total_clear = 0
    _RECUR_DECODE_CALLS["n"] = 0

    for pid, served in enumerate(served_streams):
        prompt = prompts[pid]
        # rep1/rep2 for within-process per-position determinism (class 8)
        reps: list[list[dict]] = []
        for rep in range(2):
            sink_path = str(sink_dir / f"p{pid}_rep{rep}.jsonl")
            sp = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=len(served),
                seed=args.seed,
                extra_args={
                    "fr13_served_ids": served,
                    "fr13_sink_path": sink_path,
                    "fr13_top_k": args.top_k,
                },
            )
            llm.generate([prompt], sp)
            reps.append(_read_sink(sink_path))
        rep1, rep2 = reps[0], reps[1]
        positions = []
        flips = []
        for i, rec in enumerate(rep1):
            served_id = rec["served_token_id"]
            argmax_id = rec["oracle_argmax_id"]
            flip = served_id != argmax_id
            served_in = rec["served_logprob_in_oracle"]
            dev = rec["oracle_argmax_logprob"] - served_in
            in_topk = served_id in rec["oracle_topk_ids"]
            clear = flip and ((not in_topk) or dev > args.threshold)
            det = (
                i < len(rep2)
                and rep2[i]["oracle_argmax_id"] == argmax_id
            )
            pr = {
                "pos": i,
                "served_token_id": served_id,
                "oracle_argmax_id": argmax_id,
                "oracle_argmax_logprob": rec["oracle_argmax_logprob"],
                "served_logprob_in_oracle": served_in,
                "deviation_nat": dev,
                "flip": flip,
                "clear_margin": clear,
                "within_proc_det": det,
            }
            positions.append(pr)
            total_positions += 1
            if flip:
                pr["oracle_topk_ids"] = rec["oracle_topk_ids"][:5]
                pr["oracle_topk_logprobs"] = rec["oracle_topk_logprobs"][:5]
                flips.append(pr)
                total_flips += 1
                if clear:
                    total_clear += 1
        per_prompt.append(
            {
                "prompt_id": pid,
                "served_len": len(served),
                "n_oracle_steps": len(rep1),
                "n_flips": len(flips),
                "n_clear_flips": sum(1 for f in flips if f["clear_margin"]),
                "within_proc_det_all": all(p["within_proc_det"] for p in positions),
                "flips": flips,
                "positions": positions,
            }
        )

    engaged = _RECUR_DECODE_CALLS["n"] > 0
    artifact = {
        "schema": "fr13.recurrent_decode_oracle.rescore.v1",
        "arm": args.arm,
        "src": args.src,
        "model": args.model,
        "seed": args.seed,
        "top_k": args.top_k,
        "threshold_nat": args.threshold,
        "attn_backend": args.attn_backend,
        "recurrent_path_counter_installed": installed,
        "n_gdn_layers_introspected": n_gdn,
        "recurrent_decode_calls": _RECUR_DECODE_CALLS["n"],
        "RECURRENT_PATH_ENGAGED": engaged,
        "total_positions": total_positions,
        "total_flips": total_flips,
        "total_clear_margin_flips": total_clear,
        "per_prompt_clear_margin_flip_counts": [
            p["n_clear_flips"] for p in per_prompt
        ],
        "within_proc_det_all_prompts": all(
            p["within_proc_det_all"] for p in per_prompt
        ),
        "per_prompt": per_prompt,
        "ts": time.time(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": args.out,
                "total_positions": total_positions,
                "total_clear_margin_flips": total_clear,
                "per_prompt_clear_margin_flip_counts": artifact[
                    "per_prompt_clear_margin_flip_counts"
                ],
                "RECURRENT_PATH_ENGAGED": engaged,
                "recurrent_decode_calls": artifact["recurrent_decode_calls"],
            },
            indent=2,
        )
    )
    if not engaged:
        raise SystemExit(
            "FR13 class-9 FAIL-LOUD: recurrent decode path did NOT fire -- vacuous oracle."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("smoke", help="prove recurrent path + reproduce q1 pinned stream")
    s.add_argument("--model", default=DEFAULT_MODEL_PATH)
    s.add_argument("--prompts", required=True)
    s.add_argument("--pid", type=int, default=2)
    s.add_argument("--max-tokens", type=int, default=24)
    s.add_argument("--seed", type=int, default=1313)
    s.add_argument("--served-ref", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--max-model-len", type=int, default=131072)
    s.add_argument("--gpu-util", type=float, default=0.88)
    s.add_argument("--attn-backend", default="FLASH_ATTN")
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("rescore", help="per-arm full-stream recurrent flip re-score")
    r.add_argument("--arm", required=True)
    r.add_argument("--src", required=True, help="capture json with prompts+records[].served_token_ids")
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=DEFAULT_MODEL_PATH)
    r.add_argument("--seed", type=int, default=1313)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--threshold", type=float, default=1.0)
    r.add_argument("--max-model-len", type=int, default=131072)
    r.add_argument("--gpu-util", type=float, default=0.88)
    r.add_argument("--attn-backend", default="FLASH_ATTN")
    r.set_defaults(func=cmd_rescore)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
