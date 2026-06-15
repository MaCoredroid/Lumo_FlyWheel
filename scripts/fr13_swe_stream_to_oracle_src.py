#!/usr/bin/env python3
"""FR13 reducer: proxy pair-dump -> recurrent-decode-oracle ``--src`` schema.

The ONE new load-bearing piece of harness for the big-denominator diffuse test
(FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md sec 1.3, route B). The proxy pair-dump
(``inference_proxy.py::_pair_dump_upstream``, env ``LUMO_PROXY_PAIR_DUMP_DIR``,
default OFF) writes one JSON per UPSTREAM /v1/responses turn: the exact request
payload (= the oracle PROMPT context) paired with the parsed response (= the
served output TEXT + usage). The /v1/responses SSE path emits NO per-token ids,
so we re-tokenize the served text (route B) and turn each turn into one
``(prompt, served_token_ids)`` pair for ``fr13_recurrent_decode_oracle.py
rescore --src``.

CLASS-#12 GUARD (the core of this reducer): re-derived token ids are a fiction
unless they round-trip. For every turn we ``assert tok.decode(ids) ==
served_text`` byte-for-byte; a turn that fails the round-trip is DROPPED from the
denominator and COUNTED (``dropped_turns`` / ``n_positions_dropped`` in the
artifact, ``roundtrip_ok`` per kept turn) -- never silently mis-scored. The
denominator the binomial CI is computed on is this validated round-trip token
count, NOT text length and NOT usage.output_tokens (those include the harmony
channel/structural tokens the served-content stream excludes).

The emitted schema is exactly what ``_load_served_streams`` consumes:
  {prompts:[str...], records:[{served_token_ids:[int...]}...]}  (index-aligned).
Extra per-turn provenance keys (roundtrip_ok, kind, seq, served_text_len, ...)
are additive and ignored by the oracle.

CPU-only (host tokenizer, no GPU). NON-VACUITY: refuses to emit an empty src
(zero kept turns) unless --allow-empty; that would make the rescore vacuous (#9).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = "/models/qwen3.6-27b-fp8"


def _served_text_from_response(response: Any) -> str:
    """Concatenate one turn's served output items into the served TEXT stream.

    Order = the response.output order the model emitted: reasoning content text,
    then message content text, then function_call arguments (the served
    generated content; harmony channel/structural markers are NOT part of the
    scored content stream)."""
    parts: list[str] = []
    output = (response or {}).get("output") or []
    for item in output:
        t = item.get("type")
        if t in ("reasoning", "message"):
            for c in item.get("content") or []:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    parts.append(c["text"])
        elif t == "function_call":
            args = item.get("arguments")
            if isinstance(args, str):
                parts.append(args)
    return "".join(parts)


def _prompt_from_request(request: Any, tok) -> str:
    """Render the turn's request (/v1/responses payload) to the prompt-context
    string via the model chat template (instructions -> system, each input
    ``message`` item -> its role). Multi-turn harmony history items
    (reasoning/function_call/function_call_output) are not byte-reproducible via
    apply_chat_template; the binding guard is the served-text round-trip, the
    prompt is context (the oracle reads the clean recurrent argmax conditioned on
    it before forcing each served id)."""
    request = request or {}
    msgs: list[dict[str, str]] = []
    if isinstance(request.get("instructions"), str) and request["instructions"]:
        msgs.append({"role": "system", "content": request["instructions"]})
    for item in request.get("input") or []:
        if item.get("type") != "message":
            continue
        c = item.get("content")
        if isinstance(c, list):
            text = "".join(cc.get("text", "") for cc in c if isinstance(cc, dict))
        else:
            text = c if isinstance(c, str) else ""
        msgs.append({"role": item.get("role", "user"), "content": text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_src(dump_dir: str, model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    files = sorted(glob.glob(os.path.join(dump_dir, "pair_*.json")))
    prompts: list[str] = []
    records: list[dict[str, Any]] = []
    turn_meta: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    n_positions_kept = 0
    n_positions_dropped = 0
    for fp in files:
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        served_text = _served_text_from_response(d.get("response"))
        if not served_text:  # empty served stream for this turn -> nothing to score
            continue
        ids = tok.encode(served_text, add_special_tokens=False)
        roundtrip_ok = tok.decode(ids) == served_text  # CLASS-#12 byte-exact gate
        meta = {
            "file": os.path.basename(fp),
            "seq": d.get("seq"),
            "kind": d.get("kind"),
            "served_text_len": len(served_text),
            "n_tokens": len(ids),
            "roundtrip_ok": roundtrip_ok,
        }
        if not roundtrip_ok:
            n_positions_dropped += len(ids)
            dropped.append(meta)
            continue
        prompts.append(_prompt_from_request(d.get("request"), tok))
        records.append({"served_token_ids": [int(x) for x in ids]})
        turn_meta.append(meta)
        n_positions_kept += len(ids)
    return {
        "schema": "fr13.swe_stream_to_oracle_src.v1",
        "model": model_path,
        "dump_dir": dump_dir,
        "prompts": prompts,
        "records": records,
        "turns": turn_meta,
        "n_turns_total": len(files),
        "n_turns_kept": len(records),
        "n_turns_dropped": len(dropped),
        "dropped_turns": dropped,
        "n_positions_scored": n_positions_kept,
        "n_positions_dropped": n_positions_dropped,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, help="LUMO_PROXY_PAIR_DUMP_DIR for one arm")
    ap.add_argument("--out", required=True, help="oracle --src JSON to write")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--allow-empty", action="store_true", help="permit zero kept turns (#9 vacuity escape)")
    args = ap.parse_args()
    src = build_src(args.dump_dir, args.model)
    if src["n_turns_kept"] == 0 and not args.allow_empty:
        sys.stderr.write(
            f"VACUOUS (#9): 0 kept turns from {src['n_turns_total']} pair-dumps in {args.dump_dir} "
            f"(dropped {src['n_turns_dropped']}); refusing to write. Use --allow-empty to override.\n"
        )
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(src), encoding="utf-8")
    sys.stderr.write(
        f"[fr13_swe_stream_to_oracle_src] kept={src['n_turns_kept']}/{src['n_turns_total']} turns, "
        f"dropped={src['n_turns_dropped']}, positions_scored={src['n_positions_scored']}, "
        f"positions_dropped={src['n_positions_dropped']} -> {args.out}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
