#!/usr/bin/env python3
r"""FR13 APC TEACHER-FORCED per-token LOGIT-ARGMAX gate (temperature-independent).

=============================================================================
DESIGN / PLAN  (read this whole docstring before running)
=============================================================================

## Why this exists
The free-running APC shadow gate (``scripts/fr13_apc_shadow_gate.py``) gave a
CONFOUNDED result: at temp=0 Qwen3-Next is in the degenerate greedy regime and
metastable, so the autotune kernel choice ALONE forks the *generated* trajectory
(two cache-OFF warmups diverged: tool call at 617 chars vs degenerate runaway at
2627). A free-running stream cannot separate a real cache defect from a
metastable cascade, and on GB10 (BATCH_INVARIANT=0) the cascade is always live.

This gate removes the cascade entirely. We TEACHER-FORCE both arms through ONE
fixed token sequence (the cache-OFF generated continuation) and compare the
per-position next-token *distribution* (argmax + top1/top2 logit margin). There
is no free-running feedback, so temperature is irrelevant: at fixed inputs the
two arms either produce the same argmax or they do not. The reference arm is
cache-OFF on THIS boot (the real incumbent), never a proxy or a backend name.

## Reuse (cited file:line — read these first)
- ``scripts/fr13_gold_margin_probe.py`` ``teacher_force()`` L363-481 and
  ``_tokenize()`` L355-360: the closest existing primitive. It (a) tokenizes a
  RAW string via ``/tokenize`` (L356), (b) sends a LIST-OF-INT-IDS prompt to
  ``/v1/completions`` with ``max_tokens=1`` + top-k ``logprobs`` (L411-426), and
  (c) resets the prefix cache via ``/reset_prefix_cache`` (L375). This gate
  copies that exact request shape; the only addition is the cache-ON-vs-OFF
  pairing and the per-position margin classification.
- ``scripts/fr13_oracle_capture.py`` ``_tf_one()`` L108-129: the canonical
  "force prompt_ids + served[:i], decode max_tokens=1, read top-k" call. Proves
  a LIST-OF-INT prompt to ``/v1/completions`` is the established teacher-force
  vehicle on this stack (used for the non-spec oracle).
- ``scripts/fr13_apc_shadow_gate.py`` L23-29 (``reset_cache``), L94-101 (deep
  ``/v1/responses`` send + ``cached_tokens`` read), L143 (the VACUOUS guard:
  cache-ON must report ``cached_tokens>0``, cache-OFF ``==0``). We reuse the
  cache-engagement guard idea but apply it to the ``/v1/completions`` arms.
- ``src/lumo_flywheel_serving/inference_proxy.py`` ``normalize_responses_request_payload``
  L163-254: how a codex ``/v1/responses`` request is normalized (dev/system
  roles -> ``instructions`` L257-278, function tools flattened L236-253) BEFORE
  going upstream. The caller normalizes the captured turn with this first; this
  gate consumes the already-normalized request.
- ``scripts/fr13_swe_stream_to_oracle_src.py`` ``_prompt_from_request()`` L66-87:
  the PROVEN render of a ``/v1/responses`` request -> chat-template string via
  ``tok.apply_chat_template(msgs, add_generation_prompt=True)`` (instructions ->
  system, each ``input`` ``message`` item -> its role). This is exactly how we
  obtain the rendered prompt for the chat turn.

## THE HARD PROBLEM: getting the EXACT prompt_token_ids for the chat turn
The captured turn is a ``/v1/responses`` CHAT request (tools + instructions +
input items), but the teacher-force primitive needs vLLM's exact
``prompt_token_ids``. Options evaluated (evidence in parentheses):

  (1) Does ``/v1/responses`` / ``/v1/chat/completions`` RETURN prompt/output
      token ids? -> NO, not reliably. The Responses schema *has* the keys
      (``prompt``, ``input_messages``, ``output_messages`` in a real captured
      response) but they come back **null** in practice, and the SSE path emits
      no per-token ids. ``FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md`` L96-102 states the
      ``/v1/responses`` path "does NOT reliably carry per-token ids" and ranks
      ``return_token_ids`` passthrough as best-effort route (A), to be abandoned
      if it does not round-trip. ``inference_proxy.py`` has NO token-id
      passthrough. REJECTED as the primary source.

  (2) Can ``/v1/completions`` take ``prompt`` as a LIST OF TOKEN IDS and return
      ``prompt_logprobs`` over the whole echoed sequence (one warm + one cold
      call, compare per-position argmax over the continuation)? The list-prompt
      half is YES and proven (``_tf_one`` L108-119). BUT ``prompt_logprobs`` is
      used NOWHERE in this repo (grep: 0 hits), is UNPROVEN on this fork, and —
      critically — re-introduces the exact spec-decode confound the whole FR13
      toolkit avoids: under tree/MTP spec-decode the per-position logprobs over a
      forced sequence are computed in the VERIFY path and can be attached to
      DRAFT positions, so they are not the clean single-token-decode argmax
      (``fr13_gold_margin_probe.teacher_force`` L367-371 documents this and is
      why it uses ``max_tokens=1``). REJECTED: the prompt_logprobs route does NOT
      have the max_tokens=1 protection, so it could be confounded even when the
      gate logic is correct. (Kept as an optional ``--mode prompt_logprobs``
      cross-check, OFF by default, with a loud caveat in its output.)

  (3) Can ``/tokenize`` apply the chat template (messages -> ids) to reproduce
      vLLM's EXACT prompt tokenization? Partially: vLLM's ``/tokenize`` accepts a
      raw ``prompt`` string (proven, ``_tokenize`` L356) and applies BOS/no chat
      template. To get vLLM's exact rendered chat prompt we (a) render the chat
      turn to a STRING with the host tokenizer's ``apply_chat_template``
      (``_prompt_from_request`` L66-87, proven), then (b) send THAT STRING to the
      server's ``/tokenize`` so the ids are vLLM's own (not a host/version skew).
      This is the chosen route. CAVEAT it shares with route (B) of the test plan
      (L103-112): multi-turn harmony history items (reasoning / function_call /
      function_call_output) are NOT byte-reproducible via ``apply_chat_template``.
      We DEFEND this with an explicit round-trip + content guard (below); a turn
      that cannot be reproduced byte-exact is REFUSED, never silently mis-scored
      (class #12). For the APC gate the cleanest input is an INITIAL-kind turn
      (``kind == "initial"``, only ``message`` input items, no history), which
      renders byte-exact.

  (4) Any repo capture hook that already dumps ``prompt_token_ids``? The proxy
      pair-dump (``LUMO_PROXY_PAIR_DUMP_DIR``) captures TEXT only, not ids
      (test plan L96-97). The ``return_token_ids`` field is a COMPLETIONS-arm
      feature (raw prompt), not a responses-arm one. No existing hook gives the
      chat-turn ``prompt_token_ids`` for free.

## RECOMMENDED METHOD (and why)
Route (3) + the established ``max_tokens=1`` teacher-force (route 2's vehicle
WITHOUT its prompt_logprobs hazard):

  STEP A. Render the normalized ``/v1/responses`` turn to a prompt STRING with
          ``apply_chat_template`` (``_prompt_from_request``-style). Send that
          string to the SERVER's ``/tokenize`` -> ``prompt_token_ids`` are vLLM's
          own ids for that exact string.  [matches vLLM's tokenizer/version]
  STEP B. Generate the cache-OFF continuation ONCE (reset_prefix_cache, then a
          single ``/v1/completions`` over ``prompt_token_ids`` with
          ``max_tokens=N``, ``return_token_ids=True``) to obtain the fixed
          continuation ``cont_ids`` we will teacher-force. The cache-OFF stream
          is the reference; we never free-run the comparison.
  STEP C. For each continuation position ``i``, build context = prompt_ids +
          cont_ids[:i] and do the SAME ``max_tokens=1`` decode TWICE per arm:
            - cache-OFF arm: ``/reset_prefix_cache`` then decode (fresh prefill)
            - cache-ON  arm: decode again immediately (warm prefix hit)
          Read top-k logprobs; the emitted token is a single VERIFIED token whose
          logprob is reliable (the ``max_tokens=1`` protection, L367-371).
  STEP D. Compare argmax(OFF) vs argmax(ON) at each position, plus the top1-top2
          logit margin in each arm. First position where argmax DIFFERS is the
          fork; classify by margin (below).

Why this over prompt_logprobs: it inherits the toolkit's proven spec-decode
protection (single verified token per position), reuses the exact request shape
already in three FR13 scripts, and keeps the reference = cache-OFF on this boot.
Cost is O(N) requests vs O(1) for prompt_logprobs; acceptable for a ~128-256
token continuation, and it is the only route that is *known* clean on this fork.

## FALLBACK if route (3) round-trip fails
If ``detokenize(prompt_ids) != rendered_string`` (chat-template / harmony skew),
the gate REFUSES that turn (prints the mismatch, exits 3). Recovery options the
operator picks from: (a) use an ``initial``-kind pair-dump turn (no history -> no
harmony items -> renders byte-exact); (b) if a future server build adds reliable
``return_token_ids`` on the responses arm, capture ``prompt_token_ids`` directly
and pass via ``--prompt-ids-json`` (skips STEP A) — supported here as an input.

## CACHE-ENGAGEMENT GUARD (else the gate is VACUOUS — shadow gate L143)
We cannot read ``cached_tokens`` off ``/v1/completions`` the way the responses
arm does. Instead we assert engagement structurally per position:
  - cache-OFF arm is ALWAYS preceded by ``/reset_prefix_cache`` (fresh prefill).
  - cache-ON arm is the immediate re-decode of the SAME context (prefix is warm).
  - Plus a within-boot DETERMINISM guard: each arm is decoded TWICE; if an arm's
    two reps disagree at a position, that position is autotune-noise
    (BATCH_INVARIANT=0) and any ON-vs-OFF diff there is NOT a cache verdict
    (marked ``noise_nondet``, excluded from the confident-flip count).

## CLASSIFICATION (temperature-independent, logit-space)
At each position with argmax(OFF) != argmax(ON):
  margin_off = lp(OFF top1) - lp(OFF token chosen by ON)   [how hard OFF rejects ON's pick]
  margin_on  = lp(ON  top1) - lp(ON  token chosen by OFF)
  - CONFIDENT flip  : both margins > --margin-floor  AND both arms within-boot
                      deterministic  => a REAL cache defect (the cache changed a
                      decision the model held confidently).
  - NEAR-TIE flip   : max(margins) <= --margin-floor  => noise-level, the two
                      tokens are within the floor of each other (not a defect).
  - NOISE-NONDET    : an arm's two reps disagree => autotune, not adjudicable.
Output: first argmax-diff position + its margin + class; AND the total count of
CONFIDENT flips over the continuation. VERDICT:
  - LOSSLESS              : zero CONFIDENT flips (all diffs are NEAR-TIE/NOISE).
  - NOT-LOSSLESS          : >=1 CONFIDENT flip (report the first + the count).
  - INCONCLUSIVE          : the round-trip / engagement guards failed.

=============================================================================
LIVE-SERVER REQUIRED. This script posts to a vLLM port the CALLER runs (it does
NOT boot anything). Running it without ``--i-have-a-live-server`` exits with a
guard so it cannot be mistaken for a self-contained run. See ``--help``.
=============================================================================

## EXACT CLI to run once a server is up (caller provides the port)
  # 1) normalize the captured turn with the proxy normalizer (one-liner, CPU):
  #    python -c "import json,sys; sys.path.insert(0,'src'); \
  #      from lumo_flywheel_serving.inference_proxy import normalize_responses_request_payload as N; \
  #      r=json.load(open('PAIR.json'))['request']; \
  #      json.dump(N(r), open('/tmp/norm_req.json','w'))"
  #
  # 2) run the gate (cache-ON vs cache-OFF, both on this boot):
  .venv/bin/python scripts/fr13_apc_teacher_forced_logit_gate.py \
      --i-have-a-live-server \
      --endpoint http://127.0.0.1:9953 \
      --model qwen3.6-27b \
      --request-json /tmp/norm_req.json \
      --model-path /models/qwen3.6-27b-fp8 \
      --max-cont-tokens 192 \
      --margin-floor 2.3026 \
      --out output/fr13_apc_tf_gate/verdict.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "http://127.0.0.1:9953"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_MODEL_PATH = "/models/qwen3.6-27b-fp8"
# ln(10) == served token at most 10x more probable than the other arm's token.
# Same default + meaning as fr13_gold_margin_probe.reduce --threshold (L516).
DEFAULT_MARGIN_FLOOR = 2.3026


# --------------------------------------------------------------------------- #
# HTTP helpers (same shape as fr13_gold_margin_probe / fr13_oracle_capture)
# --------------------------------------------------------------------------- #
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
    last: Exception | None = None
    while time.time() < deadline:
        try:
            _get_text(endpoint, "/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not healthy in {timeout_s}s: {last}")


def _reset_cache(endpoint: str) -> None:
    try:
        _post_json(endpoint, "/reset_prefix_cache", {}, timeout=60)
    except Exception:  # noqa: BLE001
        pass


def _tokenize(endpoint: str, model: str, prompt: str, timeout: float) -> list[int]:
    """vLLM /tokenize on a RAW string => the server's own ids (matches its
    tokenizer/version). Same call as fr13_gold_margin_probe._tokenize L355-360."""
    data = _post_json(endpoint, "/tokenize", {"model": model, "prompt": prompt}, timeout=timeout)
    toks = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(toks, list):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return [int(t) for t in toks]


# --------------------------------------------------------------------------- #
# STEP A: render the /v1/responses chat turn -> prompt string (host template)
# Mirrors fr13_swe_stream_to_oracle_src._prompt_from_request L66-87.
# --------------------------------------------------------------------------- #
def _render_prompt_string(request: dict[str, Any], model_path: str) -> tuple[str, int, Any]:
    """instructions -> system; each input `message` item -> its role; render with
    the model chat template + add_generation_prompt. Host tokenizer only (CPU).

    NOTE harmony history items (reasoning/function_call/function_call_output) are
    NOT byte-reproducible here; we guard with a /tokenize round-trip below and
    REFUSE a turn that does not reproduce (class #12). Prefer an `initial`-kind
    turn (message-only input)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    msgs: list[dict[str, str]] = []
    instr = request.get("instructions")
    if isinstance(instr, str) and instr:
        msgs.append({"role": "system", "content": instr})
    non_message_items = 0
    for item in request.get("input") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            non_message_items += 1
            continue
        content = item.get("content")
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        else:
            text = content if isinstance(content, str) else ""
        msgs.append({"role": item.get("role", "user"), "content": text})
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return rendered, non_message_items, tok


# --------------------------------------------------------------------------- #
# Teacher-force ONE position: max_tokens=1, top-k logprobs, list-of-ids prompt.
# Same request shape as fr13_oracle_capture._tf_one L108-129. The max_tokens=1
# decode emits a single VERIFIED token whose logprob is reliable even under
# spec-decode (fr13_gold_margin_probe L367-371).
# --------------------------------------------------------------------------- #
def _tf_one(
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


def _generate_cont(
    endpoint: str,
    model: str,
    prompt_ids: list[int],
    n: int,
    mode: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    """STEP B: one cache-OFF /v1/completions over prompt_ids for the fixed
    continuation we teacher-force. cache-OFF == the reference."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": n,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
    choice = data["choices"][0]
    return {
        "cont_ids": choice.get("token_ids") or [],
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text"),
    }


# --------------------------------------------------------------------------- #
# Margin helper: rank/logprob of a token id inside a top-k map (keyed by string).
# The top_logprobs map is token-STRING -> logprob (OpenAI completions schema),
# so we resolve by the served token STRING for each id.
# --------------------------------------------------------------------------- #
def _lp_of(top_map: dict[str, float], token_str: str | None) -> float | None:
    if token_str is None:
        return None
    v = top_map.get(token_str)
    return float(v) if v is not None else None


def _margin(top_map: dict[str, float], own_tok: str | None, other_tok: str | None) -> float | None:
    own = _lp_of(top_map, own_tok)
    other = _lp_of(top_map, other_tok)
    if own is None or other is None:
        return None  # other arm's token absent from this arm's top-k => > top-k span
    return own - other


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    ep, model = args.endpoint, args.model
    if args.wait_health:
        _wait_health(ep, args.wait_health)

    # ---- STEP A: prompt_token_ids -------------------------------------------------
    if args.prompt_ids_json:
        # FALLBACK input: a future build that reliably emits prompt_token_ids.
        prompt_ids = json.loads(Path(args.prompt_ids_json).read_text(encoding="utf-8"))
        prompt_ids = [int(x) for x in prompt_ids]
        rendered, non_msg, roundtrip_ok = None, None, True
    else:
        request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        rendered, non_msg, _tok = _render_prompt_string(request, args.model_path)
        # Bind to the SERVER's tokenizer (vLLM's own ids), not the host ids.
        prompt_ids = _tokenize(ep, model, rendered, args.request_timeout)
        # CLASS-#12 round-trip guard: the rendered string must reproduce byte-exact
        # from the server's ids (else harmony/template skew => REFUSE).
        rt = _post_json(ep, "/detokenize", {"model": model, "tokens": prompt_ids}, args.request_timeout)
        detok = rt.get("prompt") if isinstance(rt, dict) else None
        roundtrip_ok = (detok == rendered)
        if non_msg:
            print(f"[warn] {non_msg} non-`message` input item(s) (harmony history) — "
                  f"prefer an `initial`-kind turn; round-trip guard will adjudicate.")
        if not roundtrip_ok:
            verdict = {
                "VERDICT": "INCONCLUSIVE",
                "reason": "prompt round-trip failed (chat-template / harmony skew); refusing turn (class #12)",
                "rendered_head": (rendered or "")[:200],
                "detok_head": (detok or "")[:200] if isinstance(detok, str) else detok,
            }
            _write(args.out, {"verdict": verdict})
            print(json.dumps(verdict, indent=2))
            return 3

    # ---- STEP B: cache-OFF continuation (the fixed teacher-forcing target) --------
    _reset_cache(ep)
    gen = _generate_cont(ep, model, prompt_ids, args.max_cont_tokens, args.mode, args.seed, args.request_timeout)
    cont_ids = gen["cont_ids"]
    if not cont_ids:
        verdict = {"VERDICT": "INCONCLUSIVE", "reason": "empty cache-OFF continuation"}
        _write(args.out, {"verdict": verdict})
        print(json.dumps(verdict, indent=2))
        return 3

    # ---- STEP C+D: per-position cache-OFF (post-reset) vs cache-ON (warm) ----------
    positions: list[dict[str, Any]] = []
    first_argmax_diff: dict[str, Any] | None = None
    confident_flips = 0
    near_tie_flips = 0
    noise_nondet = 0

    for i in range(len(cont_ids)):
        context = prompt_ids + cont_ids[:i]

        # cache-OFF: fresh prefill, decoded twice for within-boot determinism.
        _reset_cache(ep)
        off1 = _tf_one(ep, model, context, args.top_k, args.mode, args.seed, args.request_timeout)
        off2 = _tf_one(ep, model, context, args.top_k, args.mode, args.seed, args.request_timeout)
        # cache-ON: immediate re-decode of the SAME context (prefix warm); twice.
        on1 = _tf_one(ep, model, context, args.top_k, args.mode, args.seed, args.request_timeout)
        on2 = _tf_one(ep, model, context, args.top_k, args.mode, args.seed, args.request_timeout)

        off_det = off1["argmax_token_id"] == off2["argmax_token_id"]
        on_det = on1["argmax_token_id"] == on2["argmax_token_id"]
        argmax_diff = off1["argmax_token_id"] != on1["argmax_token_id"]

        cls = None
        margin_off = margin_on = None
        if argmax_diff:
            # OFF's view: how hard does OFF reject ON's pick? (positive = strongly)
            margin_off = _margin(off1["top_logprobs"], off1["argmax_token"], on1["argmax_token"])
            margin_on = _margin(on1["top_logprobs"], on1["argmax_token"], off1["argmax_token"])
            if not (off_det and on_det):
                cls = "NOISE-NONDET"
                noise_nondet += 1
            else:
                present = (margin_off is not None) and (margin_on is not None)
                if not present:
                    # other arm's token outside top-k => margin exceeds the captured
                    # top-k span => treat as CONFIDENT (surface the bound for the reader).
                    cls = "CONFIDENT"
                    confident_flips += 1
                elif (margin_off > args.margin_floor) and (margin_on > args.margin_floor):
                    cls = "CONFIDENT"
                    confident_flips += 1
                else:
                    cls = "NEAR-TIE"
                    near_tie_flips += 1
            if first_argmax_diff is None:
                first_argmax_diff = {
                    "position": i,
                    "off_argmax_id": off1["argmax_token_id"],
                    "off_argmax_str": off1["argmax_token"],
                    "on_argmax_id": on1["argmax_token_id"],
                    "on_argmax_str": on1["argmax_token"],
                    "margin_off": margin_off,
                    "margin_on": margin_on,
                    "off_within_boot_det": off_det,
                    "on_within_boot_det": on_det,
                    "class": cls,
                }

        positions.append({
            "position": i,
            "forced_token_id": cont_ids[i],
            "off_argmax_id": off1["argmax_token_id"],
            "on_argmax_id": on1["argmax_token_id"],
            "argmax_diff": argmax_diff,
            "off_within_boot_det": off_det,
            "on_within_boot_det": on_det,
            "margin_off": margin_off,
            "margin_on": margin_on,
            "class": cls,
        })

    # ---- VERDICT ------------------------------------------------------------------
    if confident_flips > 0:
        v = "NOT-LOSSLESS"
    else:
        v = "LOSSLESS"
    verdict = {
        "VERDICT": v,
        "margin_floor_logprob": args.margin_floor,
        "n_continuation_positions": len(cont_ids),
        "n_argmax_diffs": sum(1 for p in positions if p["argmax_diff"]),
        "confident_flips": confident_flips,
        "near_tie_flips": near_tie_flips,
        "noise_nondet_positions": noise_nondet,
        "first_argmax_diff": first_argmax_diff,
        "reference_arm": "cache-OFF (this boot)",
        "prompt_len_tokens": len(prompt_ids),
        "roundtrip_ok": roundtrip_ok,
        "note": ("CONFIDENT flip = real cache defect; NEAR-TIE = within margin floor (noise-level); "
                 "NOISE-NONDET = autotune (BATCH_INVARIANT=0), not adjudicable."),
    }
    _write(args.out, {"verdict": verdict, "positions": positions,
                      "cont_text_head": (gen.get("text") or "")[:400]})
    print(json.dumps(verdict, indent=2))
    return 0 if v == "LOSSLESS" else 1


def _write(out_path: str, obj: dict[str, Any]) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FR13 APC teacher-forced per-token logit-argmax gate (temp-independent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="LIVE SERVER REQUIRED — see module docstring for the exact CLI.",
    )
    p.add_argument("--i-have-a-live-server", action="store_true",
                   help="REQUIRED ack: this gate posts to a running vLLM port (it boots nothing).")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name for the API")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="host tokenizer dir for apply_chat_template render (STEP A)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--request-json", help="a NORMALIZED /v1/responses request payload (one turn)")
    src.add_argument("--prompt-ids-json", help="FALLBACK: a JSON list of prompt_token_ids (skips render)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-cont-tokens", type=int, default=192,
                   help="length of the cache-OFF continuation to teacher-force")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--margin-floor", type=float, default=DEFAULT_MARGIN_FLOOR,
                   help="logprob margin <= floor => NEAR-TIE (noise); default ln(10)=2.3026")
    p.add_argument("--mode", default="tree_mtp",
                   help="vllm_xargs.fr10_decode_mode (match the served arm under test)")
    p.add_argument("--seed", type=int, default=1313)
    p.add_argument("--request-timeout", type=float, default=300)
    p.add_argument("--wait-health", type=float, default=120)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.i_have_a_live_server:
        raise SystemExit(
            "REFUSING: this gate needs a LIVE vLLM server (it boots nothing). "
            "Start/point at a server, then pass --i-have-a-live-server. See --help / the module docstring."
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
