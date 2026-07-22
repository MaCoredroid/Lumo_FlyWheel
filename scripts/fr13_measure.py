#!/usr/bin/env python3
"""FR13 CANONICAL speed / lossless MEASUREMENT infra — DEPLOYMENT regime.

ONE validated entry point that measures, for ANY arm (native MTP-N / any
caterpillar TREE shape / OPT flags), on the CANONICAL regime = the real
SWE-Verified + codex DEPLOYMENT trajectory (the codex agent loop on real
SWE-bench-Verified tasks, chat-templated via /v1/responses, multi-turn, real
tool calls). The four deployment numbers, on the deployment trajectory:
  {s/fwd, accept/event, committed/event, derived-TPS}  (SPEED, instrument OFF)
    -> cmd_deploy_speed: raw-counter delta of the per-task /metrics brackets
       run_swe_bench_q36_a.py already captures during the codex loop.
  {clear-margin flip rate + Wilson CI vs each arm's OWN no-spec RECURRENT
   decode oracle; native-E5 = the within-floor BAR}    (LOSSLESS, instrument ON)
    -> cmd_deploy_lossless: consumes the big-denom rescore consolidation.

THE DEPLOYMENT REGIME IS CANONICAL; THE RAW-/v1/completions PATH IS DEPRECATED
------------------------------------------------------------------------------
The handrolled regime (cmd_speed / cmd_capture_q / cmd_temp06_drift on
prompts_swe4.json sent as a RAW string to /v1/completions with NO chat template)
is OFF-DISTRIBUTION for this chat/thinking-trained model. native E5's served
stream on prompt 0 REPEATS [271,248068,271,248069,271,40] = "\\n<think>\\n</think>
\\nI" (a degenerate empty-<think></think> loop, verified in
output/fr13_measure/native_e5_q_temp06_on.json) — that off-distribution
degeneration (NOT a kernel bug) tanked native accept to ~1.589 and forked the
stream cross-boot (the GB10 near-tie). The no-spec oracle ranks the COHERENT
continuation correct by ~11 nats, so the real model decode is coherent; only the
raw-prompt spec boots degenerate. THE FIX (user): measure on the DEPLOYMENT
regime, which the big-denom ALREADY proved faithful + representative (codex on
astropy-12907: native ~= cat9, 13.99% vs 13.55% clear-margin flips, NO degenerate
loop, spec-vs-non-spec CONFIRMED). The raw-/v1/completions subcommands below are
KEPT only as a documented OFF-DISTRIBUTION cautionary note + (flagged) for the
regime-robust s/fwd cross-check; they are NOT the deployment number.

DEPLOYMENT-regime numbers come from cmd_deploy_speed / cmd_deploy_lossless ONLY.

(historical) the original raw-regime root-cause — root-caused 2026-06-15
-----------------------------------------------------------------
Two hand-rolled probes gave two different "native accept" on the SAME
prompts_swe4 / max_tokens 128 / temp 0 / seed 1313:
  * fr13_speed_phase0.sh drove fr10_quick_decode_tps_probe.py with
    samples_per_prompt=4 + a /v1/chat/completions warmup -> native accept 1.70
  * the gold-gate (fr13_gold_margin_probe capture, samples_per_prompt=1) ->
    the banked 3.161.
DIAGNOSIS (code+artifact verified, see FR13_SPEED_MEASURE_INFRA.md):
BOTH probes send a RAW STRING to /v1/completions and the server tokenizes the
IDENTICAL 681-token context (offline-confirmed: phase0 prompt_token_ids ==
tok(prompt[0]) byte-for-byte; first served tokens [271,248068,...,40] match the
gold stream then FORK at served index 6). So "raw-vs-tokenized" is NOT the
discriminator — both prefill the same bytes. The served greedy streams FORK
across the two runs (phase0 -> a degenerate "<think></think>...None" loop that
accepts at 1.70; gold -> a coherent "I'll start by exploring" stream that
accepts at 3.16), each stream is WITHIN-boot deterministic (all 4 phase0 samples
identical; gold rep1==rep2). This is the GB10 same-prefill cross-run greedy
trajectory fork (the autotune/realization floor, feedback_no_cross_boot_byte_gate)
amplified by the chat-template warmup + samples_per_prompt. accept/event is
TRAJECTORY-DEPENDENT (bug-class #12 "non-like-for-like trajectories"); it is NOT
apple-to-apple across runs unless the served stream is PINNED.

THE FIX baked into this infra: (1) the canonical regime is the gold-gate one
(samples_per_prompt=1, B=1 sequential, NO chat-template warmup — a single raw
self-warm on prompt[0]); (2) accept/event is recorded WITH its served-stream
fingerprint and is explicitly labelled trajectory-bound, never compared across
two free-running boots; (3) the lossless verdict is each arm vs its OWN no-spec
recurrent oracle on the arm's OWN served stream (capture-once), so the
cross-run fork is removed as a variable.

THE CANONICAL REGIME (baked, no re-roll)
----------------------------------------
  prompts = output/fr13_acceptance_ladder/prompts_swe4.json (4 SWE prompts)
  seed = 1313, max_tokens = 128
  /v1/completions, RAW text prompt, return_token_ids=True
  temp 0 / top_p 1.0  (greedy)  OR  temp 0.6 / top_p 0.95  (deployment sampling)
  FR10_METRICS=0, VLLM_BATCH_INVARIANT=0  (pinned in the launcher, asserted here)
  vllm_xargs={"fr10_decode_mode": <native:naive_mtp | tree:tree_mtp>}
  warmup = ONE raw self-warm request (NOT chat template), then reset_prefix_cache
  B=1 = SEQUENTIAL (samples run one at a time); B=4 = client batch of 4

TRUTHFUL SPEED ACCOUNTING (the ONLY allowed basis)
--------------------------------------------------
  s/fwd        = d(vllm:request_decode_time_seconds_sum) / d(vllm:spec_decode_num_drafts_total)
                 decode-only, per spec-event, RAW /metrics counter delta.
                 ~B-invariant for the bandwidth-bound GB10 B=1 decode.
  accept/event = d(vllm:spec_decode_num_accepted_tokens_total) / d(spec_drafts)
                 B-DEPENDENT (co-residency degrades it) -> B=1 and B=4 are
                 DIFFERENT NUMBERS and are labelled with batch_size + a served-
                 stream fingerprint (trajectory-bound).
  committed/ev = accept/event + 1 (the bonus token).
  TPS          = committed_per_event / s_fwd  -> reported as DERIVED, not measured.
BANNED as a speed basis (blocked + asserted): TPS, accept, wall-clock,
per-request HTTP req_elapsed. The module raises if asked to report any of these
as s/fwd.

INSTRUMENT ON/OFF MODE SEPARATION (user)
----------------------------------------
The instrumentation AFFECTS speed, so SPEED and LOSSLESS are SEPARATE boots/numbers.
  mode=OFF (CLEAN deployment): FR10_METRICS=0, NO logprobs/q-capture, NO flip/
    oracle, NO FR12/13 diagnostics. This is the bytes the user ships. SPEED
    (s/fwd, accept) is measured ONLY here. An OFF run carries instrument="OFF".
  mode=ON  (lossless/drift instrumentation): the full-stream q-capture (top-K
    logprobs) + the flip/temp-0.6 reduce. Adds real decode tax (extra top-K
    logprob + DtoH). LOSSLESS/drift is measured ONLY here. That boot's s/fwd is
    used ONLY to QUANTIFY the instrument tax (diag-residue = OFF s/fwd vs ON
    s/fwd), NEVER as the deployment speed.
Every emitted number records {"instrument": "OFF"|"ON"}. assert_no_mode_mix()
raises if a speed verdict cites an ON number or a lossless verdict cites an OFF
capture. diag_residue() computes the OFF-vs-ON s/fwd tax per arm (expect <=2.5%,
46e89f22, but MEASURED not assumed).

SUBCOMMANDS
-----------
  CANONICAL (DEPLOYMENT regime = real SWE-Verified + codex):
  deploy-speed   OFF-mode CANONICAL: s/fwd + accept/event + committed + derived
                 TPS on the REAL codex trajectory, from the per-task /metrics
                 brackets run_swe_bench_q36_a.py captures. Same truthful basis as
                 `speed`, no degenerate fork. THE deployment SPEED number.
  deploy-lossless ON-mode CANONICAL: the within-floor lossless verdict from the
                 big-denom rescore consolidation (clear-margin flip rate + Wilson
                 CI vs each arm's OWN no-spec recurrent oracle; native-E5 = BAR).
                 THE deployment LOSSLESS number.

  DEPRECATED (raw-/v1/completions, OFF-DISTRIBUTION — cautionary / regime-robust
  s/fwd cross-check only, never the deployment number):
  speed        OFF-mode: raw-counter s/fwd + accept/event + committed + derived
               TPS for one arm at a given batch_size + temp. Capture-once served
               streams + fingerprint. No logprobs (clean deployment path).
  capture-q    ON-mode: the gold-gate-style per-served-position top-K
               log_softmax over the WHOLE served stream (this is the spec verify
               forward dist q) + per-position truncated tail mass. THE NEW piece.
  flip         lossless temp-0 argmax: reduce a served stream vs its OWN no-spec
               recurrent oracle (delegates to fr13_recurrent_decode_oracle.py).
  temp06-drift the binding temp-0.6 gate: combine capture-q (q) + recurrent
               oracle top-K (p) -> per-position TV(softmax(q/0.6),softmax(p/0.6))
               + KL + over-floor vector. + the realized multi-seed bag-TV with
               the p95 native floor. CPU reduce over ON-mode captures.
  diag-residue OFF-vs-ON s/fwd tax per arm.
  reconcile    compare measured {s/fwd, accept} vs the banked historic numbers.

Each subcommand FAILS LOUD on disengagement (class-9): tok/draft must == N
(native) / == len(TREE) (tree); has_tree_parent_indices / tree_sample_accept for
tree; RECURRENT_PATH_ENGAGED for the oracle. Records NOTHING on disengagement.
Within-boot determinism (class-8) rep1==rep2 is checked on captures.
Cross-boot byte gate is BANNED (feedback_no_cross_boot_byte_gate) and not used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ENDPOINT = "http://127.0.0.1:9950"
DEFAULT_MODEL = "qwen3.6-27b"
CANONICAL_PROMPTS = str(REPO / "output" / "fr13_acceptance_ladder" / "prompts_swe4.json")
CANONICAL_SEED = 1313
CANONICAL_MAX_TOKENS = 128
# the tokenizer behind the served model (id<->decoded-string re-key for GAP-1).
CANONICAL_TOKENIZER = "/models/qwen3.6-27b-fp8"

# raw /metrics counters (the ONLY allowed speed basis)
M_DECODE_S = "vllm:request_decode_time_seconds_sum"
M_DRAFTS = "vllm:spec_decode_num_drafts_total"
M_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
M_DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
M_GEN_TOK = "vllm:generation_tokens_total"
# per-request-NORMALIZED decode rate (NOT the concurrency-summed decode_seconds_sum):
# request_time_per_output_token = per-request mean time-per-output-token (decode
# phase). count/sum = 1/avg(TPOT) = the per-STREAM decode token rate. FR10 flagged
# gen_tok/decode_s_sum as concurrency-deflated at B>1; this is the non-deflated basis.
M_TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
M_TPOT_COUNT = "vllm:request_time_per_output_token_seconds_count"
# prefill time (decode-span EXCLUDES it, but it eats wall -> drags gen/wall aggregate;
# prefill_frac exposes the workload confound, user 2026-06-16).
M_PREFILL_S = "vllm:request_prefill_time_seconds_sum"
# FR13_SFWD_GPU_TIMER counter: GPU-active time in the PURE-DECODE model-forward
# (the tree TREE_ATTN / MTP verify forward), summed over decode steps via async
# cuda events in gpu_model_runner.execute_model. PREFILL-INDEPENDENT (excludes
# interleaved chunked-prefill + idle that the wall-span M_DECODE_S absorbs at
# B>1). Present only when the server booted with FR13_SFWD_GPU_TIMER=1 (the timer
# is DEFAULT-OFF + byte-identical); absent => 0.0 here and s_per_fwd_gpu = None.
M_DECODE_FWD_GPU_S = "vllm:fr13_decode_forward_gpu_seconds_total"
# MATCHED denominators for s_per_fwd_gpu (both restricted to the SAME pure-decode
# steps the timer measured). s_per_fwd_gpu MUST divide M_DECODE_FWD_GPU_S by one of
# these, NOT by the GLOBAL M_DRAFTS (spec_decode_num_drafts_total): the global counter
# includes drafts on mixed prefill+decode steps the timer excludes (~49% at B=4
# deployment), so M_DECODE_FWD_GPU_S/M_DRAFTS reintroduces a prefill-load-dependent
# confound that is arm-dependent -> defeats prefill-independence. M_DECODE_FWD_GPU_STEPS
# = pure-decode forwards (per-FORWARD basis, matches the metric name + the banked B=1
# 0.218 per-forward); M_DECODE_FWD_GPU_DRAFTS = drafts on those steps (per-spec-event).
M_DECODE_FWD_GPU_STEPS = "vllm:fr13_decode_forward_gpu_steps_total"
M_DECODE_FWD_GPU_DRAFTS = "vllm:fr13_decode_forward_gpu_drafts_total"
# FR13_DFWD/CFWD_GPU_TIMER component span timers (_Fr13SpanTimer): drafter =
# propose_draft_token_ids (all D spine forwards); committer = the spec-decode
# rejection-sampler dispatch in gpu_model_runner._sample (accept/LCP/bonus +
# commit). Sidecar-synthesized into the per-task brackets by
# run_swe_bench_q36_a._metrics_snapshot (the same route as the sfwd counters).
# DEFAULT-OFF timers; absent lines => 0.0 delta here => the deploy_speed
# drafter_/committer_ fields are null (never a crash).
M_DRAFTER_GPU_S = "vllm:fr13_drafter_gpu_seconds_total"
M_DRAFTER_GPU_SPANS = "vllm:fr13_drafter_gpu_spans_total"
M_COMMITTER_GPU_S = "vllm:fr13_committer_gpu_seconds_total"
M_COMMITTER_GPU_SPANS = "vllm:fr13_committer_gpu_spans_total"
# FR13_STEP_WALL: MEASURED full-step wall basis (start-to-start deltas between
# consecutive pure-decode steps, idle-capped, chain broken on mixed/prefill).
# Gate (user-mandated): every speed verdict must report the derived fullstep
# TPS AGAINST this measured wall TPS; the residual (wall - fwd - drafter -
# committer) is the "other overhead" bucket the derived basis cannot see.
M_STEP_WALL_S = "vllm:fr13_decode_step_wall_seconds_total"
M_STEP_WALL_DRAFTS = "vllm:fr13_decode_step_wall_drafts_total"
M_STEP_WALL_STEPS = "vllm:fr13_decode_step_wall_steps_total"
COUNTERS = [M_DECODE_S, M_DRAFTS, M_ACCEPTED, M_DRAFT_TOK, M_GEN_TOK,
            M_TPOT_SUM, M_TPOT_COUNT, M_PREFILL_S, M_DECODE_FWD_GPU_S,
            M_DECODE_FWD_GPU_STEPS, M_DECODE_FWD_GPU_DRAFTS,
            M_DRAFTER_GPU_S, M_DRAFTER_GPU_SPANS,
            M_COMMITTER_GPU_S, M_COMMITTER_GPU_SPANS,
            M_STEP_WALL_S, M_STEP_WALL_DRAFTS, M_STEP_WALL_STEPS]

# Forms that must NEVER be reported as s/fwd (blocked + asserted, class-12).
BANNED_SPEED_BASES = {"tps", "accept", "wall", "wall_clock", "req_elapsed", "http_elapsed"}


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                 #
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
    last = None
    while time.time() < deadline:
        try:
            _get_text(endpoint, "/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    raise RuntimeError(f"server not healthy in {timeout_s}s: {last}")


def _scrape(endpoint: str) -> dict[str, float]:
    text = _get_text(endpoint, "/metrics", timeout=10)
    out = {c: 0.0 for c in COUNTERS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            try:
                out[name] += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return out


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: float(after.get(k, 0.0) - before.get(k, 0.0)) for k in after}


def read_sfwd_gpu_sidecar() -> dict[str, float] | None:
    """FR13_SFWD_GPU_TIMER sidecar (cumulative): the prefill-independent
    decode-forward GPU-time channel. vLLM only calls setup_multiprocess_
    prometheus() when api_server_count>1, so in the single-API-server deployment
    the worker's Counter is NOT aggregated into the API-server /metrics (separate
    process, no shared PROMETHEUS_MULTIPROC_DIR). The worker therefore ALSO dumps
    a throttled JSON sidecar (FR13_SFWD_GPU_TIMER_JSON), which carries:
      decode_forward_gpu_seconds  (cumulative pure-decode model-forward GPU s)
      n_drafts_in_timed_steps     (drafts processed IN those timed steps =
                                    the MATCHING s/fwd_gpu denominator; at B>1,
                                    mixed prefill+decode steps are excluded from
                                    the GPU numerator, so this -- NOT the engine's
                                    total drafts -- is the correct denominator).
    Returns None if no sidecar is found (server booted timer-OFF / api_server>1).
    """
    import glob as _glob
    pattern = os.environ.get(
        "FR13_SFWD_GPU_TIMER_JSON_GLOB",
        str(REPO / "output" / "fr13_sfwd_gpu_verify" / "logs"
            / "fr13_sfwd_gpu_timer.json.*"),
    )
    files = _glob.glob(pattern)
    if not files:
        return None
    f = max(files, key=os.path.getmtime)
    try:
        d = json.loads(Path(f).read_text())
    except Exception:  # noqa: BLE001
        return None
    return {
        "decode_forward_gpu_seconds": float(d.get("decode_forward_gpu_seconds", 0.0)),
        "n_drafts_in_timed_steps": float(d.get("n_drafts_in_timed_steps", 0.0)),
        "n_pure_decode_steps_timed": float(d.get("n_pure_decode_steps_timed", 0.0)),
    }


def _read_prompts(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"prompts file must be a JSON list of strings: {path}")
    prompts = [p for p in data if p]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


# --------------------------------------------------------------------------- #
# GAP-1 RE-KEY: q top_logprobs is keyed by the DECODED token STRING (vLLM's    #
# completions logprobs lose the token id in serialization); the recurrent      #
# oracle p is keyed by token ID. To compare them by ID we build a reverse map  #
# decoded_string -> token_id from the SERVED MODEL's own tokenizer, once.      #
# MEASURED (this model's 248k-vocab): the decoded->id map is COLLISION-FREE    #
# over the captured top-K support (0 collisions / 1506 distinct keys), so the  #
# re-key is unambiguous; any string that does not map to a single id is kept   #
# in an "unmapped::<string>" bucket so its probability mass is NEVER dropped    #
# (the TV stays an upper-faithful reduce, never vacuous).                       #
# --------------------------------------------------------------------------- #
_DEC2ID_CACHE: dict[str, dict[str, int]] = {}


def _build_dec2id(tokenizer_path: str) -> dict[str, int]:
    """decoded_string -> token_id reverse map from the served model's tokenizer.

    Built ONCE per tokenizer_path. Collisions (a decoded string produced by >1
    id) are DROPPED from the unambiguous map (the caller then routes that string
    to the unmapped bucket) so a many-to-one decode never silently mis-assigns
    mass to the wrong id. CPU-only (transformers in the host venv)."""
    if tokenizer_path in _DEC2ID_CACHE:
        return _DEC2ID_CACHE[tokenizer_path]
    from transformers import AutoTokenizer  # host venv, CPU
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tok.get_vocab())
    decoded = tok.batch_decode([[i] for i in range(vocab_size)])
    seen: dict[str, int] = {}
    collisions: set[str] = set()
    for tid, dec in enumerate(decoded):
        if dec in seen and seen[dec] != tid:
            collisions.add(dec)
        else:
            seen[dec] = tid
    for c in collisions:
        seen.pop(c, None)  # ambiguous -> route to unmapped bucket
    _DEC2ID_CACHE[tokenizer_path] = seen
    return seen


def rekey_q_to_ids(
    top_logprobs_str: dict[str, float],
    dec2id: dict[str, int],
    served_id: int | None,
    served_str: str | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Re-key one position's string-keyed q top_logprobs to token-ID keys.

    Returns ({str(token_id)|"unmapped::<s>": logprob}, stats). The served-token
    ANCHOR is honored: if the position's served string is present it is forced
    to the EXACT served id (the served id is known from return_token_ids, so the
    served candidate is never mis-mapped even in the unlikely collision case)."""
    out: dict[str, float] = {}
    n_mapped = 0
    n_unmapped = 0
    for s, lp in (top_logprobs_str or {}).items():
        if served_str is not None and served_id is not None and s == served_str:
            out[str(int(served_id))] = float(lp)  # served anchor: exact id
            n_mapped += 1
            continue
        tid = dec2id.get(s)
        if tid is None:
            out[f"unmapped::{s}"] = float(lp)
            n_unmapped += 1
        else:
            out[str(tid)] = float(lp)
            n_mapped += 1
    return out, {"n_mapped": n_mapped, "n_unmapped": n_unmapped}


# --------------------------------------------------------------------------- #
# Arm spec — the SHAPE descriptor for native MTP-N / TREE caterpillar          #
# --------------------------------------------------------------------------- #
def parse_arm(arm: str, tree: str | None) -> dict[str, Any]:
    """Resolve an arm name to its served decode-mode + expected tok/draft.

    arm = 'native_eN' (N in 3..8) -> naive_mtp, expected tok/draft = N
    arm = 'tree' (with --tree TREE) or 'cat9'/'cat10'/... -> tree_mtp,
          expected tok/draft = len(TREE). The TREE must be BUILT on the server
          (the launcher derives num_spec + tree from len(TREE)); if a caller
          asks for a shape the booted server did not build, the engagement
          assert FAILS LOUD (tok/draft != len(TREE)).
    """
    if arm.startswith("native_e"):
        try:
            n = int(arm[len("native_e"):])
        except ValueError as exc:
            raise ValueError(f"native arm must be native_eN, got {arm!r}") from exc
        if not 1 <= n <= 8:
            raise ValueError(f"native MTP-N out of range: {n}")
        return {"arm": arm, "mode": "naive_mtp", "expected_tok_per_draft": n,
                "is_tree": False, "tree": None}
    # tree arm
    if tree is None:
        raise ValueError(
            f"tree arm {arm!r} requires --tree '[(0,), (0,0), ...]' (fail-loud on "
            "unbuilt shape: the canonical cat9 is "
            "[(0,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0), (0,1), (0,0,1), "
            "(0,0,0,1), (0,0,0,0,1)])"
        )
    import ast
    nodes = ast.literal_eval(tree)
    if not isinstance(nodes, (list, tuple)) or not nodes:
        raise ValueError(f"--tree must be a non-empty list of tuples: {tree!r}")
    return {"arm": arm, "mode": "tree_mtp", "expected_tok_per_draft": len(nodes),
            "is_tree": True, "tree": tree, "tree_len": len(nodes)}


# --------------------------------------------------------------------------- #
# Engagement asserts (class-9 fail-loud) — BEFORE any number is recorded       #
# --------------------------------------------------------------------------- #
def assert_engaged(spec: dict[str, Any], metric_delta: dict[str, float],
                   *, context: str) -> dict[str, Any]:
    """tok/draft over the measured window MUST equal the arm's expected shape.

    This is the cheap, mode-agnostic engagement gate from raw /metrics counters
    (no logprobs needed): num_draft_tokens / num_drafts == N (native) ==
    len(TREE) (tree). If the server silently fell back (e.g. linear) or built a
    different tree, tok/draft != expected and we record NOTHING.
    """
    drafts = metric_delta.get(M_DRAFTS, 0.0)
    draft_tok = metric_delta.get(M_DRAFT_TOK, 0.0)
    if drafts <= 0:
        raise RuntimeError(
            f"class-9 FAIL-LOUD [{context}]: zero spec_drafts in the window "
            "(speculation did not engage) -- recording nothing."
        )
    tok_per_draft = draft_tok / drafts
    expected = float(spec["expected_tok_per_draft"])
    if abs(tok_per_draft - expected) > 1e-6:
        raise RuntimeError(
            f"class-9 FAIL-LOUD [{context}]: tok/draft={tok_per_draft} != "
            f"expected {expected} for arm {spec['arm']} -- the served shape is "
            "NOT what was requested (silent fallback / wrong tree). Nothing recorded."
        )
    return {"tok_per_draft": tok_per_draft, "expected_tok_per_draft": expected,
            "engaged": True}


def assert_no_mode_mix(records: list[dict[str, Any]]) -> None:
    """No speed number may cite an ON capture; no lossless number an OFF capture.

    Each record carries {"kind": "speed"|"lossless"|"drift", "instrument":
    "OFF"|"ON"}. SPEED must be OFF; lossless/drift must be ON. Raise on mix.
    """
    for r in records:
        kind = r.get("kind")
        inst = r.get("instrument")
        if kind == "speed" and inst != "OFF":
            raise RuntimeError(
                f"MODE MIX: speed number {r.get('label')!r} came from "
                f"instrument={inst} (must be OFF=clean deployment). Refusing to report."
            )
        if kind in ("lossless", "drift") and inst != "ON":
            raise RuntimeError(
                f"MODE MIX: {kind} number {r.get('label')!r} came from "
                f"instrument={inst} (must be ON). Refusing to report."
            )


def assert_speed_basis(basis: str) -> None:
    if basis.lower() in BANNED_SPEED_BASES:
        raise RuntimeError(
            f"BANNED speed basis {basis!r}: s/fwd MUST be "
            "d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total). "
            "TPS/accept/wall/req_elapsed are NEVER the basis (class-12)."
        )


# --------------------------------------------------------------------------- #
# Stream fingerprint — bind accept/event to its served trajectory             #
# --------------------------------------------------------------------------- #
def stream_fingerprint(served_streams: list[list[int]]) -> str:
    import hashlib
    h = hashlib.sha256()
    for s in served_streams:
        h.update(bytes(str(s), "utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Request drivers                                                              #
# --------------------------------------------------------------------------- #
def _one_request(endpoint, model, prompt, *, n, max_tokens, temperature, top_p,
                 seed, mode, logprobs, timeout):
    payload: dict[str, Any] = {
        "model": model,
        "prompt": [prompt] * n if n > 1 else prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
    return data["choices"]


def _self_warm(endpoint, model, prompt, mode, seed, timeout):
    """ONE raw self-warm request (NOT chat template). The chat-template warmup
    was the phase0 confound; the canonical warmup is a raw /v1/completions on
    the real prompt, then reset_prefix_cache so the load starts cold+warm-kernel."""
    _one_request(endpoint, model, prompt, n=1, max_tokens=16, temperature=0.0,
                 top_p=1.0, seed=seed, mode=mode, logprobs=None, timeout=timeout)
    try:
        _post_json(endpoint, "/reset_prefix_cache", {}, timeout=30)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# SPEED (OFF mode) — the deployment number                                     #
# --------------------------------------------------------------------------- #
def cmd_speed(args: argparse.Namespace) -> int:
    assert_speed_basis(args.basis)  # blocks banned bases
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    spec = parse_arm(args.arm, args.tree)
    prompts = _read_prompts(args.prompts_file)

    # canonical warmup: ONE raw self-warm, then reset_prefix_cache.
    _self_warm(args.endpoint, args.model, prompts[0], spec["mode"], args.seed,
               args.request_timeout)

    before = _scrape(args.endpoint)
    sc_before = read_sfwd_gpu_sidecar()
    served_streams: list[list[int]] = []
    if args.batch_size == 1:
        # B=1 SEQUENTIAL: one prompt, one sample, at a time.
        for prompt in prompts:
            choices = _one_request(
                args.endpoint, args.model, prompt, n=1,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, seed=args.seed, mode=spec["mode"],
                logprobs=None, timeout=args.request_timeout,
            )
            for ch in choices:
                served_streams.append([int(x) for x in (ch.get("token_ids") or [])])
    else:
        # B=batch: one client batch of `batch_size` identical samples per prompt
        # (co-residency present; accept/event degrades, s/fwd ~invariant).
        for prompt in prompts:
            choices = _one_request(
                args.endpoint, args.model, prompt, n=args.batch_size,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, seed=args.seed, mode=spec["mode"],
                logprobs=None, timeout=args.request_timeout,
            )
            for ch in choices:
                served_streams.append([int(x) for x in (ch.get("token_ids") or [])])
    after = _scrape(args.endpoint)
    sc_after = read_sfwd_gpu_sidecar()
    md = _delta(after, before)

    eng = assert_engaged(spec, md, context=f"speed {spec['arm']} B={args.batch_size}")

    drafts = md[M_DRAFTS]
    s_fwd = md[M_DECODE_S] / drafts if drafts > 0 else None
    # FR13_SFWD_GPU_TIMER (prefill-independent): the pure-decode model-forward GPU
    # seconds per spec event. PREFERRED channel = the worker JSON sidecar (the
    # /metrics counter is NOT aggregated in single-API-server mode). CRITICAL
    # denominator: the drafts processed IN the GPU-TIMED pure-decode steps
    # (sidecar n_drafts_in_timed_steps), NOT the engine's total drafts -- at B>1
    # the mixed prefill+decode steps are excluded from the GPU numerator, so
    # dividing by total drafts would bias s/fwd_gpu low. At B=1 SEQUENTIAL there
    # is no co-resident interleave so timed-drafts == total drafts and s_fwd_gpu
    # ~= the wall-span s_fwd (the pure-decode reference).
    s_fwd_gpu = None
    fwd_gpu_d = 0.0
    timed_drafts_d = 0.0
    if sc_before is not None and sc_after is not None:
        fwd_gpu_d = sc_after["decode_forward_gpu_seconds"] - sc_before["decode_forward_gpu_seconds"]
        timed_drafts_d = sc_after["n_drafts_in_timed_steps"] - sc_before["n_drafts_in_timed_steps"]
        if fwd_gpu_d > 0 and timed_drafts_d > 0:
            s_fwd_gpu = fwd_gpu_d / timed_drafts_d
    if s_fwd_gpu is None:
        # fallback: /metrics counter (multi-API-server mode), total-drafts denom
        fwd_gpu = md.get(M_DECODE_FWD_GPU_S, 0.0)
        if drafts > 0 and fwd_gpu > 0:
            s_fwd_gpu = fwd_gpu / drafts
            fwd_gpu_d = fwd_gpu
            timed_drafts_d = drafts
    accept_per_event = md[M_ACCEPTED] / drafts if drafts > 0 else None
    committed_per_event = (accept_per_event + 1.0) if accept_per_event is not None else None
    derived_tps = (committed_per_event / s_fwd) if (s_fwd and committed_per_event) else None

    rec = {
        "schema": "fr13.measure.speed.v1",
        "kind": "speed",
        "instrument": "OFF",
        "label": f"speed_{spec['arm']}_b{args.batch_size}_t{args.temperature}",
        "arm": spec["arm"],
        "mode": spec["mode"],
        "tree": spec.get("tree"),
        "batch_size": args.batch_size,
        "b1_sequential": args.batch_size == 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompts_file": args.prompts_file,
        "n_served_streams": len(served_streams),
        "served_stream_fingerprint": stream_fingerprint(served_streams),
        "served_lens": [len(s) for s in served_streams],
        "engagement": eng,
        "speed_basis": "d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)",
        "s_per_fwd": s_fwd,
        "s_per_fwd_note": (
            "WALL-SPAN basis; at B=1 SEQUENTIAL there is no co-resident prefill "
            "interleave so it ~= s_per_fwd_gpu (the pure-decode reference)."
        ),
        # FR13_SFWD_GPU_TIMER: prefill-independent decode-forward GPU time per
        # spec event (None unless the server booted FR13_SFWD_GPU_TIMER=1).
        "s_per_fwd_gpu": s_fwd_gpu,
        "s_per_fwd_gpu_basis": "d(decode_forward_gpu_seconds)/d(n_drafts_in_timed_steps) [sidecar]",
        "s_per_fwd_gpu_decode_forward_gpu_seconds_delta": fwd_gpu_d,
        "s_per_fwd_gpu_timed_drafts_delta": timed_drafts_d,
        "accept_per_event": accept_per_event,
        "accept_per_event_note": (
            "B-DEPENDENT + TRAJECTORY-BOUND: valid only for this served_stream_"
            "fingerprint at this batch_size; NOT apple-to-apple across boots "
            "(same-prefill greedy fork, bug-class #12)."
        ),
        "committed_per_event": committed_per_event,
        "derived_tps": derived_tps,
        "derived_tps_note": "DERIVED = committed_per_event / s_fwd; NOT measured.",
        "raw_counter_delta": md,
        "ts": time.time(),
    }
    if args.dump_streams:
        rec["served_streams"] = served_streams
    assert_no_mode_mix([rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in (
        "arm", "batch_size", "temperature", "s_per_fwd", "s_per_fwd_gpu",
        "accept_per_event", "committed_per_event", "derived_tps",
        "served_stream_fingerprint", "instrument")}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CAPTURE-Q (ON mode) — the NEW full-stream spec verify dist q                 #
# --------------------------------------------------------------------------- #
def cmd_capture_q(args: argparse.Namespace) -> int:
    """Per-served-position top-K log_softmax over the WHOLE served stream.

    This extends the gold_margin top_logprobs capture (which records the spec
    verify forward dist) from the 4 fork positions to the FULL stream, and
    records the per-position truncated tail mass (1 - sum exp(topk_logprobs)) as
    the truncation error bar. The result is q at T=1; the temp-0.6 reduce applies
    /0.6 (the per-position additive constant cancels in the softmax). NOTHING
    records this today -- this was the gap.
    """
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    spec = parse_arm(args.arm, args.tree)
    prompts = _read_prompts(args.prompts_file)
    # GAP-1: build the decoded-string -> token-id reverse map ONCE so q can be
    # emitted ID-KEYED at capture time (the recurrent oracle p is id-keyed).
    dec2id = None if args.no_rekey else _build_dec2id(args.tokenizer)
    _self_warm(args.endpoint, args.model, prompts[0], spec["mode"], args.seed,
               args.request_timeout)
    before = _scrape(args.endpoint)

    rekey_unmapped_total = 0
    rekey_mapped_total = 0
    reps: list[list[dict[str, Any]]] = []
    for _rep in range(2):  # within-boot determinism (class-8)
        rep_records: list[dict[str, Any]] = []
        for pid, prompt in enumerate(prompts):
            ch = _one_request(
                args.endpoint, args.model, prompt, n=1,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, seed=args.seed, mode=spec["mode"],
                logprobs=args.top_k, timeout=args.request_timeout,
            )[0]
            lp = ch.get("logprobs") or {}
            top = lp.get("top_logprobs") or []
            served_ids = [int(x) for x in (ch.get("token_ids") or [])]
            served_toks = lp.get("tokens") or []
            tail_mass = []
            top_ids: list[dict[str, float] | None] = []  # GAP-1: id-keyed q
            for i, pos in enumerate(top):
                if pos:
                    s = sum(math.exp(v) for v in pos.values())
                    tail_mass.append(max(0.0, 1.0 - s))
                    if dec2id is not None:
                        sid = served_ids[i] if i < len(served_ids) else None
                        sstr = served_toks[i] if i < len(served_toks) else None
                        q_ids, st = rekey_q_to_ids(pos, dec2id, sid, sstr)
                        top_ids.append(q_ids)
                        rekey_mapped_total += st["n_mapped"]
                        rekey_unmapped_total += st["n_unmapped"]
                    else:
                        top_ids.append(None)
                else:
                    tail_mass.append(None)
                    top_ids.append(None)
            rep_records.append({
                "prompt_id": pid,
                "served_token_ids": served_ids,
                "served_tokens": served_toks,
                "served_token_logprobs": lp.get("token_logprobs") or [],
                "top_logprobs": top,                # q at T=1, STRING-keyed (raw)
                "top_logprobs_ids": top_ids,        # GAP-1: SAME q, TOKEN-ID keyed
                "per_position_tail_mass": tail_mass,  # truncation error bar
                "finish_reason": ch.get("finish_reason"),
            })
        reps.append(rep_records)
    after = _scrape(args.endpoint)
    md = _delta(after, before)
    eng = assert_engaged(spec, md, context=f"capture-q {spec['arm']}")

    within_boot_det = [
        reps[0][pid]["served_token_ids"] == reps[1][pid]["served_token_ids"]
        for pid in range(len(prompts))
    ]
    artifact = {
        "schema": "fr13.measure.capture_q.v1",
        "kind": "drift",
        "instrument": "ON",
        "label": f"q_{spec['arm']}_t{args.temperature}",
        "arm": spec["arm"],
        "mode": spec["mode"],
        "tree": spec.get("tree"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "prompts": prompts,
        "records": reps[0],
        "records_rep2": reps[1],
        "within_boot_det_rep1_eq_rep2": within_boot_det,
        "served_stream_fingerprint": stream_fingerprint(
            [r["served_token_ids"] for r in reps[0]]),
        "engagement": eng,
        "raw_counter_delta": md,
        # GAP-1 re-key provenance: q is now emitted ID-KEYED (top_logprobs_ids)
        # via the served model's tokenizer so temp06-drift can align with the
        # id-keyed recurrent oracle p (no more id_string_mismatch vacuum).
        "q_id_keyed": (not args.no_rekey),
        "rekey_tokenizer": (None if args.no_rekey else args.tokenizer),
        "rekey_mapped_total": rekey_mapped_total,
        "rekey_unmapped_total": rekey_unmapped_total,
        "rekey_unmapped_frac": (
            rekey_unmapped_total / max(1, rekey_mapped_total + rekey_unmapped_total)),
        "ts": time.time(),
    }
    assert_no_mode_mix([artifact])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "arm": spec["arm"], "instrument": "ON",
        "within_boot_det": within_boot_det,
        "served_lens": [len(r["served_token_ids"]) for r in reps[0]],
        "n_positions_with_q": [len(r["top_logprobs"]) for r in reps[0]],
        "median_tail_mass": [
            (statistics.median([t for t in r["per_position_tail_mass"] if t is not None])
             if any(t is not None for t in r["per_position_tail_mass"]) else None)
            for r in reps[0]
        ],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# temp-0.6 DRIFT reduce (CPU) — TV(softmax(q/0.6), softmax(p/0.6))             #
# --------------------------------------------------------------------------- #
def _logprob_map(top_logprobs_pos: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in (top_logprobs_pos or {}).items()}


def _softmax_over_support_at_temp(lp_map: dict[str, float], temp: float) -> dict[str, float]:
    """softmax(logprob/temp) over the captured top-K support. Since logprob =
    log_softmax(logit) = logit - C, logprob/temp = logit/temp - C/temp; the
    additive C/temp cancels in the re-softmax over the SAME support. So this is
    exactly softmax(logit/temp) restricted to the captured support."""
    if not lp_map:
        return {}
    scaled = {k: v / temp for k, v in lp_map.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    z = sum(exps.values())
    return {k: e / z for k, e in exps.items()}


def _tv(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    # KL(p || q) in the direction the sampler "feels" (target p re-weighted by q)
    out = 0.0
    for k, pv in p.items():
        if pv > 0:
            out += pv * math.log(pv / max(q.get(k, 0.0), eps))
    return out


def _p_topk_by_pos(p_art: dict[str, Any], pid: int, p_path: str) -> dict[int, dict[str, float]]:
    """Build {pos -> {str(token_id): logprob}} for one prompt from the recurrent
    oracle artifact. GAP-1: the reduced positions[] only retain oracle_topk on
    FLIP positions, so to score the FULL stream we read the per-prompt SINK
    JSONL (which the LP writes for EVERY step with the full top-K) when present;
    otherwise we use whatever positions[] carry (full when the oracle was run
    with --full-topk-all-positions, else flips only)."""
    pp = {p["prompt_id"]: p for p in p_art["per_prompt"]}.get(pid)
    out: dict[int, dict[str, float]] = {}
    if pp is None:
        return out
    # 1) preferred: the per-prompt sink JSONL (full top-K at EVERY step).
    sink_dir = p_art.get("sink_dir")
    candidates = []
    if sink_dir:
        candidates.append(Path(sink_dir) / f"p{pid}_rep0.jsonl")
    # default sink layout from fr13_recurrent_decode_oracle.cmd_rescore:
    base = Path(p_path).parent / f"{p_art.get('arm','')}_sinks"
    candidates.append(base / f"p{pid}_rep0.jsonl")
    for sink in candidates:
        if sink.exists():
            for line in sink.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                step = rec["step"]
                ids = rec.get("oracle_topk_ids") or []
                lps = rec.get("oracle_topk_logprobs") or []
                if ids and lps:
                    out[step] = {str(int(t)): float(v) for t, v in zip(ids, lps)}
            if out:
                return out
    # 2) fallback: positions[] (full if --full-topk-all-positions was used).
    for pr in pp.get("positions", []):
        ids = pr.get("oracle_topk_ids")
        lps = pr.get("oracle_topk_logprobs")
        if ids and lps:
            out[pr["pos"]] = {str(int(t)): float(v) for t, v in zip(ids, lps)}
    return out


def cmd_temp06_drift(args: argparse.Namespace) -> int:
    """Reduce the per-position TV/KL(q,p) at temp-0.6 for ONE arm vs its OWN
    recurrent oracle, ALIGNED BY TOKEN ID (GAP-1). q = capture-q artifact (spec
    verify top-K, now ID-KEYED via top_logprobs_ids); p = the recurrent-decode
    oracle top-K (token-id keyed) over the FULL served stream. Each arm vs its
    OWN oracle on the arm's OWN served stream (apples-to-apples)."""
    q_art = json.loads(Path(args.q).read_text(encoding="utf-8"))
    p_art = json.loads(Path(args.p).read_text(encoding="utf-8"))
    if q_art.get("instrument") != "ON":
        raise RuntimeError("temp06-drift q artifact must be instrument=ON (capture-q).")
    # p artifact = recurrent oracle rescore. Assert it is the recurrent frame.
    if not p_art.get("RECURRENT_PATH_ENGAGED", False):
        raise RuntimeError(
            "class-9 FAIL-LOUD: p oracle artifact does not assert "
            "RECURRENT_PATH_ENGAGED -- refusing (wrong/chunked oracle frame)."
        )
    temp = args.temp
    q_recs = q_art["records"]

    # GAP-1: q must be ID-KEYED. If the capture-q artifact predates the re-key,
    # build the reverse map here and re-key on the fly (so old artifacts still
    # reduce non-vacuously) rather than silently falling back to string keys.
    dec2id = None
    if not q_art.get("q_id_keyed", False):
        if args.no_rekey:
            raise RuntimeError(
                "temp06-drift: q artifact is NOT id-keyed (q_id_keyed != True) and "
                "--no-rekey was set -> the TV would be VACUOUS (id/string mismatch). "
                "Re-run capture-q with the re-key, or drop --no-rekey here."
            )
        dec2id = _build_dec2id(args.tokenizer)

    forks: list[dict[str, Any]] = []
    all_tv: list[float] = []
    all_kl: list[float] = []
    over_floor_total = 0
    n_string_mismatch = 0
    served_in_p_mismatch = 0
    for qrec in q_recs:
        pid = qrec["prompt_id"]
        p_by_pos = _p_topk_by_pos(p_art, pid, args.p)
        if not p_by_pos:
            forks.append({"prompt_id": pid, "note": "no oracle top-K for prompt"})
            continue
        q_tops_str = qrec["top_logprobs"]
        q_tops_ids = qrec.get("top_logprobs_ids") or [None] * len(q_tops_str)
        q_tail = qrec.get("per_position_tail_mass") or []
        served_ids = qrec.get("served_token_ids") or []
        served_toks = qrec.get("served_tokens") or []
        per_pos: list[dict[str, Any]] = []
        for i in range(len(q_tops_str)):
            pmap = p_by_pos.get(i)
            if pmap is None:
                continue
            # build the ID-KEYED q for this position.
            q_id_pos = q_tops_ids[i] if i < len(q_tops_ids) else None
            if q_id_pos is None:
                if dec2id is None:
                    n_string_mismatch += 1
                    continue
                sid = served_ids[i] if i < len(served_ids) else None
                sstr = served_toks[i] if i < len(served_toks) else None
                q_id_pos, _ = rekey_q_to_ids(q_tops_str[i], dec2id, sid, sstr)
            qmap = {str(k): float(v) for k, v in q_id_pos.items()}
            q_at = _softmax_over_support_at_temp(qmap, temp)
            p_at = _softmax_over_support_at_temp(pmap, temp)
            # full cross-key TV over the UNION of id supports (mass on the
            # symmetric difference is counted, not dropped) -- this is the real
            # truncated-support TV(softmax(q/T), softmax(p/T)).
            overlap = set(q_at) & set(p_at)
            tv = _tv(q_at, p_at)
            kl = _kl(p_at, q_at)
            served_id = str(int(served_ids[i])) if i < len(served_ids) else None
            q_served = q_at.get(served_id) if served_id else None
            p_served = p_at.get(served_id) if served_id else None
            if served_id is not None and p_served is None:
                served_in_p_mismatch += 1
            entry = {
                "pos": i,
                "tv_q_p_at_temp": tv,
                "kl_p_q_at_temp": kl,
                "n_support_overlap": len(overlap),
                "q_support": len(q_at),
                "p_support": len(p_at),
                "q_served_prob_at_temp": q_served,
                "p_served_prob_at_temp": p_served,
                "q_tail_mass_T1": (q_tail[i] if i < len(q_tail) else None),
                "align_status": "ok",
            }
            per_pos.append(entry)
            all_tv.append(tv)
            all_kl.append(kl)
            if args.per_position_floor is not None and tv > args.per_position_floor:
                over_floor_total += 1
        forks.append({
            "prompt_id": pid,
            "n_positions": len(per_pos),
            "mean_tv": (statistics.fmean([e["tv_q_p_at_temp"] for e in per_pos])
                        if per_pos else None),
            "max_tv": (max(e["tv_q_p_at_temp"] for e in per_pos) if per_pos else None),
            "positions": per_pos if args.dump_positions else per_pos[:5],
        })

    result = {
        "schema": "fr13.measure.temp06_drift.v2",
        "kind": "drift",
        "instrument": "ON",
        "label": f"temp06_drift_{q_art.get('arm')}",
        "arm": q_art.get("arm"),
        "temp": temp,
        "q_artifact": args.q,
        "p_artifact": args.p,
        "aligned_by": "token_id",
        "q_id_keyed_source": ("capture-q top_logprobs_ids" if q_art.get("q_id_keyed")
                              else "on-the-fly host-tokenizer re-key"),
        "per_position_floor": args.per_position_floor,
        "mean_tv_q_p_at_temp": (statistics.fmean(all_tv) if all_tv else None),
        "p95_tv_q_p_at_temp": (
            sorted(all_tv)[int(0.95 * (len(all_tv) - 1))] if all_tv else None),
        "max_tv_q_p_at_temp": (max(all_tv) if all_tv else None),
        "mean_kl_p_q_at_temp": (statistics.fmean(all_kl) if all_kl else None),
        "n_positions_scored": len(all_tv),
        "n_string_mismatch_skipped": n_string_mismatch,
        "served_token_absent_from_p_count": served_in_p_mismatch,
        "over_floor_count": over_floor_total,
        "error_bar_note": (
            "TV is over the TRUNCATED top-K supports; per-position q_tail_mass_T1 "
            "(1 - sum exp(top-K logprob)) is the truncation error bar -- the TV is "
            "exact within that tail mass. served_token_absent_from_p_count flags "
            "positions where the sampled served token left the oracle top-K "
            "(expected for a temp-0.6 SAMPLED q vs a GREEDY-argmax-ranked p)."
        ),
        "per_position_note": (
            "PAIR the scalar with the per-position vector "
            "(reference_scalar_metric_per_token_blindspot): "
            "over_floor_count names WHERE the drift crosses the native floor."
        ),
        "interpretation_note": (
            "This TV is q (the SPEC-VERIFY forward dist at temp 0.6 over the SAMPLED "
            "served stream) vs p (the no-spec RECURRENT oracle teacher-forced onto "
            "that SAME served stream). Both arms compare to their OWN oracle on their "
            "OWN stream. Use the LOSSLESS gate = per-position over-floor count vs the "
            "native temp-0.6 self-floor + the multi-seed bag-TV (cmd_bag_tv); a high "
            "raw mean TV alone is NOT a lossless miss (sampling spreads q)."
        ),
        "forks": forks,
        "ts": time.time(),
    }
    assert_no_mode_mix([result])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "arm", "temp", "aligned_by", "mean_tv_q_p_at_temp", "p95_tv_q_p_at_temp",
        "max_tv_q_p_at_temp", "n_positions_scored", "over_floor_count",
        "served_token_absent_from_p_count", "instrument")}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# BAG-TV (deployment cross-check) — realized multi-seed served streams         #
# --------------------------------------------------------------------------- #
def _bag_tv(stream_a: list[int], stream_b: list[int]) -> float:
    from collections import Counter
    ca, cb = Counter(stream_a), Counter(stream_b)
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in keys)


def cmd_bag_tv(args: argparse.Namespace) -> int:
    """Multi-seed realized BAG-TV with the p95 native-self floor (gate (b)).

    --native is a list of native temp-0.6 served-stream captures (N seeds); the
    floor = p95 of C(N,2) native-vs-native bag-TV draws. --cat is the cat9
    captures (same seeds); cat-vs-native bag-TV PASSES iff <= floor_p95.
    Upgrades the single-draw 0.1133 to a robust threshold (class-12)."""
    def _load_streams(paths: list[str]) -> list[list[int]]:
        streams: list[list[int]] = []
        for path in paths:
            art = json.loads(Path(path).read_text(encoding="utf-8"))
            for r in art["records"]:
                streams.append([int(x) for x in r["served_token_ids"]])
        return streams

    native = _load_streams(args.native)
    if len(native) < 2:
        raise RuntimeError("bag-tv native floor needs >=2 native streams (seeds).")
    native_draws = [_bag_tv(a, b) for a, b in combinations(native, 2)]
    native_draws.sort()
    floor_p95 = native_draws[int(0.95 * (len(native_draws) - 1))]

    cat_vs_native = None
    verdict = None
    if args.cat:
        cat = _load_streams(args.cat)
        cross = [_bag_tv(c, n) for c in cat for n in native]
        cat_vs_native = statistics.fmean(cross)
        verdict = "PASS_within_floor" if cat_vs_native <= floor_p95 else "ABOVE_floor"

    result = {
        "schema": "fr13.measure.bag_tv.v1",
        "kind": "drift",
        "instrument": "ON",
        "label": "bag_tv_temp06",
        "n_native_streams": len(native),
        "n_native_draws": len(native_draws),
        "native_floor_p95": floor_p95,
        "native_floor_mean": statistics.fmean(native_draws),
        "cat_vs_native_bag_tv": cat_vs_native,
        "verdict": verdict,
        "ts": time.time(),
    }
    assert_no_mode_mix([result])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# GAP-2 PAIRED TEACHER-FORCED ACCEPT — fork-immune verification-efficiency edge #
# --------------------------------------------------------------------------- #
# WHY: free-running accept/event FORKS cross-boot at the GB10 token-6 near-tie
# (autotune floor, feedback_no_cross_boot_byte_gate). So a free-running cat9-vs-
# native accept is NOT apple-to-apple (bug-class #12 non-like-for-like
# trajectories). The fix: pin ONE reference trajectory (the no-spec RECURRENT
# oracle GREEDY stream = deployment-correct ground truth) and have BOTH arms
# verify that SAME fixed token sequence; accept/event per arm on identical
# content = the structural edge (cat9 superset vs native linear).
#
# TWO modes, ONE distinction documented on every record:
#   * mode=structural (CPU, validatable NOW): along the fixed reference, the
#     GREEDY verify accepts the reference token at position i iff it equals the
#     arm's VERIFIER argmax (the captured q argmax). The per-event accepted run
#     is the count of consecutive reference tokens the arm's verifier would
#     argmax-confirm before the first miss; we segment the reference into spec
#     events of the arm's draft depth D and sum accepts. This uses ONLY the
#     captured per-position verifier dist q (no GPU), is fork-immune (BOTH arms
#     anchored to the SAME reference), and is apples-to-apple.
#   * mode=force (GPU, deferred / orchestrate hook): boot the arm, force the
#     reference stream as the served sequence (the spec-verify must commit the
#     reference tokens), and read d(num_accepted)/d(num_drafts) live. This is the
#     ground-truth paired-accept; structural is its cheap fork-immune proxy.
#
# DISTINCTION (printed on every record): paired-accept (this) = apples-to-apple
# STRUCTURAL edge for the break-even; deployment-accept (cmd_speed) = free-
# running, trajectory-variable (the floor). NEVER cross-compare the two.
def _load_reference_streams(path: str) -> tuple[list[list[int]], str, str]:
    """Load the reference trajectory (deployment-correct ground truth) streams.

    Accepts (a) a recurrent-oracle rescore artifact (per_prompt[].positions[]
    .served_token_id = the forced reference); (b) a recurrent-oracle smoke/
    generate artifact with gen ids; (c) a capture-q / speed artifact with
    records[].served_token_ids or served_streams; (d) a plain JSON list of
    id-lists. Returns (streams, source_kind, reference_fingerprint)."""
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(art, list):  # plain list of id-lists
        streams = [[int(x) for x in s] for s in art]
        return streams, "plain_id_lists", stream_fingerprint(streams)
    schema = art.get("schema", "")
    if "recurrent_decode_oracle.rescore" in schema:
        if not art.get("RECURRENT_PATH_ENGAGED", False):
            raise RuntimeError(
                "GAP-2 paired-accept: reference oracle must be RECURRENT_PATH_ENGAGED "
                "(the deployment-correct no-spec greedy ground truth).")
        streams = [[int(p["served_token_id"]) for p in pp["positions"]]
                   for pp in art["per_prompt"]]
        return streams, "recurrent_oracle_rescore", stream_fingerprint(streams)
    if "records" in art:  # capture-q / speed
        streams = [[int(x) for x in r["served_token_ids"]] for r in art["records"]]
        return streams, "capture_q_or_speed_records", stream_fingerprint(streams)
    if "served_streams" in art:
        streams = [[int(x) for x in s] for s in art["served_streams"]]
        return streams, "speed_served_streams", stream_fingerprint(streams)
    raise RuntimeError(f"GAP-2 paired-accept: unrecognized reference artifact {path!r}")


def _arm_verifier_argmax_ids(q_art: dict[str, Any]) -> dict[int, list[int]]:
    """Per-prompt list of the arm's VERIFIER argmax token-id at each served
    position, from the capture-q artifact (the top_logprobs_ids argmax). This is
    the arm's greedy verify decision used by the structural accept reducer."""
    out: dict[int, list[int]] = {}
    for rec in q_art["records"]:
        pid = rec["prompt_id"]
        ids_per_pos = rec.get("top_logprobs_ids") or []
        str_per_pos = rec.get("top_logprobs") or []
        served = rec.get("served_token_ids") or []
        argmaxes: list[int] = []
        for i in range(len(str_per_pos)):
            qid = ids_per_pos[i] if i < len(ids_per_pos) else None
            if qid:
                # argmax = id with the max logprob in the id-keyed q
                best = max(qid.items(), key=lambda kv: kv[1])[0]
                argmaxes.append(int(best) if not best.startswith("unmapped::")
                                else (int(served[i]) if i < len(served) else -1))
            else:
                argmaxes.append(int(served[i]) if i < len(served) else -1)
        out[pid] = argmaxes
    return out


def _structural_accept_on_reference(
    reference: list[int], verifier_argmax: list[int], draft_depth: int,
) -> dict[str, Any]:
    """GREEDY verify of ONE arm against the fixed reference, segmented into spec
    events of `draft_depth` drafted tokens. Per event starting at reference
    position j: the arm accepts reference[j..] while reference[k] equals the
    arm's verifier argmax at k (verifier-confirmed), up to draft_depth; the first
    mismatch ends the run (then +1 bonus token = the verifier's own token). This
    is the depth-D verification efficiency on IDENTICAL content (fork-immune)."""
    n = min(len(reference), len(verifier_argmax))
    j = 0
    events = 0
    accepted_total = 0
    per_event = []
    while j < n:
        acc = 0
        for d in range(draft_depth):
            k = j + d
            if k >= n:
                break
            if reference[k] == verifier_argmax[k]:
                acc += 1
            else:
                break
        events += 1
        accepted_total += acc
        per_event.append(acc)
        # advance by accepted + 1 bonus token (the verifier-committed token).
        j += acc + 1
    return {
        "draft_depth": draft_depth,
        "n_events": events,
        "accepted_total": accepted_total,
        "accept_per_event": (accepted_total / events) if events else None,
        "committed_per_event": ((accepted_total / events) + 1.0) if events else None,
        "per_event_accepts": per_event,
    }


def cmd_paired_accept(args: argparse.Namespace) -> int:
    """GAP-2: paired teacher-forced accept on a COMMON reference trajectory.

    --reference = the deployment-correct ground-truth stream (no-spec recurrent
    oracle greedy). --arm-q = one or more capture-q artifacts (each carrying its
    arm's verifier dist + draft depth). Each arm is scored on the SAME forced
    reference content (fork-immune). mode=structural (CPU). mode=force documents
    the GPU ground-truth hook (orchestrate boots it)."""
    reference, ref_kind, ref_fp = _load_reference_streams(args.reference)
    arms: list[dict[str, Any]] = []
    for q_path in args.arm_q:
        q_art = json.loads(Path(q_path).read_text(encoding="utf-8"))
        if q_art.get("instrument") != "ON":
            raise RuntimeError(f"{q_path}: paired-accept arm-q must be instrument=ON.")
        # the structural verifier-argmax MUST come from the id-keyed q; a non-
        # id-keyed q would silently fall back to the arm's OWN served token,
        # which conflates the served stream with the verify decision and makes
        # the structural accept meaningless. Fail loud (class-9) unless --allow-
        # served-fallback is explicitly set (and then flag it on the record).
        q_id_keyed = bool(q_art.get("q_id_keyed", False)) or all(
            r.get("top_logprobs_ids") for r in q_art["records"])
        if not q_id_keyed and not args.allow_served_fallback:
            raise RuntimeError(
                f"class-9 FAIL-LOUD [{q_path}]: arm-q is NOT id-keyed (no "
                "top_logprobs_ids) -> the verifier argmax would fall back to the "
                "arm's served token (conflates served-stream with verify-decision). "
                "Re-run capture-q with the GAP-1 re-key, or pass --allow-served-"
                "fallback to score the SERVED stream as the proxy (labelled)."
            )
        arm = q_art.get("arm")
        is_tree = q_art.get("mode") == "tree_mtp"
        # draft depth D: native MTP-N = N; tree = len(TREE) spine depth used for
        # the linear-chain structural proxy (the spine; the branch superset edge
        # needs mode=force, documented).
        if is_tree and q_art.get("tree"):
            import ast
            depth = len(ast.literal_eval(q_art["tree"]))
        elif arm and arm.startswith("native_e"):
            depth = int(arm[len("native_e"):])
        else:
            depth = int(args.default_depth)
        argmax_by_pid = _arm_verifier_argmax_ids(q_art)
        per_prompt = []
        agg_events = 0
        agg_accept = 0
        for pid, ref in enumerate(reference):
            vargmax = argmax_by_pid.get(pid)
            if vargmax is None:
                continue
            r = _structural_accept_on_reference(ref, vargmax, depth)
            r["prompt_id"] = pid
            per_prompt.append(r)
            agg_events += r["n_events"]
            agg_accept += r["accepted_total"]
        arms.append({
            "arm": arm,
            "q_artifact": q_path,
            "draft_depth": depth,
            "is_tree_spine_proxy": is_tree,
            "verifier_argmax_source": ("id_keyed_q" if q_id_keyed else "served_fallback_PROXY"),
            "served_stream_fingerprint": q_art.get("served_stream_fingerprint"),
            "structural_accept_per_event": (agg_accept / agg_events) if agg_events else None,
            "structural_committed_per_event": (
                (agg_accept / agg_events) + 1.0) if agg_events else None,
            "n_events": agg_events,
            "accepted_total": agg_accept,
            "per_prompt": per_prompt,
        })
    result = {
        "schema": "fr13.measure.paired_accept.v1",
        "kind": "paired_accept",
        "mode": "structural",
        "reference_artifact": args.reference,
        "reference_kind": ref_kind,
        "reference_fingerprint": ref_fp,
        "reference_note": (
            "the deployment-correct ground-truth stream (no-spec recurrent oracle "
            "greedy). BOTH arms verify THIS SAME fixed sequence -> fork-immune."
        ),
        "structural_note": (
            "mode=structural: GREEDY verify of each arm's captured verifier argmax "
            "against the fixed reference, segmented into depth-D spec events. The "
            "TREE arm's number here is the SPINE (linear-chain) proxy; the branch "
            "superset edge (a sibling holding the reference token after a spine "
            "miss) needs mode=force on the live tree verifier (GPU, orchestrate "
            "boots it). Use this for the apples-to-apple break-even; do NOT cross-"
            "compare with the free-running deployment-accept (cmd_speed), which is "
            "trajectory-variable (the floor, bug-class #12)."
        ),
        "deployment_vs_paired": {
            "paired_accept": "apples-to-apple STRUCTURAL edge on a COMMON reference (this)",
            "deployment_accept": "free-running, trajectory-bound floor (cmd_speed.accept_per_event)",
            "rule": "NEVER cross-compare; paired drives the break-even, deployment is the floor",
        },
        "arms": arms,
        "ts": time.time(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "reference_kind": ref_kind,
        "reference_fingerprint": ref_fp,
        "arms": [{"arm": a["arm"], "draft_depth": a["draft_depth"],
                  "structural_accept_per_event": a["structural_accept_per_event"],
                  "n_events": a["n_events"]} for a in arms],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# DIAG-RESIDUE — OFF-vs-ON s/fwd instrument tax                                #
# --------------------------------------------------------------------------- #
def cmd_diag_residue(args: argparse.Namespace) -> int:
    off = json.loads(Path(args.off).read_text(encoding="utf-8"))
    on = json.loads(Path(args.on).read_text(encoding="utf-8"))
    if off.get("instrument") != "OFF":
        raise RuntimeError("diag-residue --off must be an OFF (clean) speed record.")
    # the ON record's s/fwd is whatever its raw_counter_delta yields
    on_md = on.get("raw_counter_delta", {})
    on_drafts = on_md.get(M_DRAFTS, 0.0)
    on_s_fwd = (on_md.get(M_DECODE_S, 0.0) / on_drafts) if on_drafts > 0 else None
    off_s_fwd = off.get("s_per_fwd")
    tax = None
    if off_s_fwd and on_s_fwd:
        tax = (on_s_fwd - off_s_fwd) / off_s_fwd
    result = {
        "schema": "fr13.measure.diag_residue.v1",
        "arm": off.get("arm"),
        "off_s_per_fwd": off_s_fwd,
        "on_s_per_fwd": on_s_fwd,
        "instrument_tax_frac": tax,
        "tax_expectation": "<=0.025 (46e89f22), MEASURED not assumed",
        "tax_within_expectation": (tax is not None and tax <= 0.025),
        "note": (
            "OFF s/fwd is the deployment speed; ON s/fwd is ONLY for this tax. "
            "Speed verdicts NEVER use the ON number."
        ),
        "ts": time.time(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# RECONCILE — measured vs banked historic numbers                             #
# --------------------------------------------------------------------------- #
BANKED = {
    "native_e5": {"s_per_fwd": 0.218160, "accept_per_event": 3.161290,
                  "decode_tps": 18.93, "src": "FR13_B1_CURRENT_GATE_BIND"},
    "cat9": {"s_per_fwd": 0.2248, "accept_per_event": 3.18,
             "src": "FR13_B1_FIX3_GATE_BIND"},
}


def cmd_reconcile(args: argparse.Namespace) -> int:
    rows = []
    for path in args.speed:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        if rec.get("kind") != "speed" or rec.get("instrument") != "OFF":
            raise RuntimeError(f"{path}: reconcile only consumes OFF speed records.")
        arm = rec["arm"]
        banked = BANKED.get(arm)
        row = {
            "arm": arm,
            "batch_size": rec.get("batch_size"),
            "served_stream_fingerprint": rec.get("served_stream_fingerprint"),
            "measured_s_per_fwd": rec.get("s_per_fwd"),
            "measured_accept_per_event": rec.get("accept_per_event"),
            "banked_s_per_fwd": banked.get("s_per_fwd") if banked else None,
            "banked_accept_per_event": banked.get("accept_per_event") if banked else None,
            "banked_src": banked.get("src") if banked else None,
        }
        if banked and rec.get("s_per_fwd"):
            row["s_per_fwd_delta_frac"] = (
                (rec["s_per_fwd"] - banked["s_per_fwd"]) / banked["s_per_fwd"])
        if banked and rec.get("accept_per_event") is not None:
            row["accept_delta_abs"] = rec["accept_per_event"] - banked["accept_per_event"]
            row["reproduces_banked_accept"] = (
                abs(row["accept_delta_abs"]) <= args.accept_tol)
            row["accept_note"] = (
                "accept is trajectory-bound (bug-class #12): a match means this "
                "boot's greedy stream is the gold trajectory; a miss is a "
                "same-prefill fork, NOT a behavior regression — check the "
                "served_stream_fingerprint, do not treat as lossless verdict."
            )
        rows.append(row)
    result = {
        "schema": "fr13.measure.reconcile.v1",
        "accept_tol": args.accept_tol,
        "rows": rows,
        "ts": time.time(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


# =========================================================================== #
# DEPLOYMENT REGIME (the CANONICAL path) — real SWE-Verified + codex            #
# =========================================================================== #
# The raw-/v1/completions handrolled regime above (speed/capture-q on            #
# prompts_swe4.json) is DEPRECATED: sending the handrolled prompt as a RAW       #
# string with NO chat template is OFF-DISTRIBUTION for this chat/thinking model. #
# The native E5 served stream on prompt 0 REPEATS [271,248068,271,248069,271,40] #
# = "\n<think>\n</think>\nI" (a degenerate empty-<think></think> loop, verified  #
# output/fr13_measure/native_e5_q_temp06_on.json) — that off-distribution        #
# degeneration (NOT a kernel bug) tanked native accept to ~1.589 and forked the  #
# stream cross-boot. The no-spec oracle ranks the COHERENT continuation correct  #
# by ~11 nats, so the real model decode is coherent; only the raw-prompt spec    #
# boots degenerate. THE FIX (user): measure on the DEPLOYMENT regime = the codex #
# agent loop on real SWE-bench-Verified tasks, chat-templated via /v1/responses, #
# multi-turn, real tool calls. The big-denom ALREADY proved this regime faithful #
# + representative (codex on astropy-12907: native ~= cat9, 13.99% vs 13.55%     #
# clear-margin flips, NO degenerate loop, spec-vs-non-spec CONFIRMED).            #
#                                                                                 #
# This block ORCHESTRATES the proven big-denom machinery (it does NOT re-invent  #
# it): the deployment harness (fr13_bigdenom_swe_serve.sh) boots the arm + runs  #
# scripts/run_swe_bench_q36_a.py (the codex loop), which ALREADY brackets        #
# /metrics per task into vllm_metrics_pre.txt / vllm_metrics_post.txt. So the    #
# DEPLOYMENT-regime s/fwd + accept/event is the SAME raw-counter delta basis as  #
# cmd_speed, only on the real codex trajectory (no degenerate fork). Lossless on #
# the deployment trajectory is the big-denom rescore (proxy pair-dump -> no-spec #
# RECURRENT decode oracle -> clear-margin flip rate + Wilson CI) consolidated by #
# fr13_bigdenom_rescore_consolidate.py; cmd_deploy_lossless reads that artifact. #
# --------------------------------------------------------------------------- #
def _scrape_metrics_file(path: str) -> dict[str, float]:
    """Parse a Prometheus /metrics SNAPSHOT FILE (vllm_metrics_pre/post.txt
    bracketed per task by run_swe_bench_q36_a.py) into the raw counters. Same
    parse as _scrape() but from a captured file instead of a live HTTP poll."""
    out = {c: 0.0 for c in COUNTERS}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            try:
                out[name] += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return out


def _find_task_dirs(out_root: str) -> list[Path]:
    """Locate per-task dirs under a run_swe_bench_q36_a out-root. The layout is
    <out_root>/<dataset_tag>/per_task/<instance_id>/ (each holding the bracketed
    vllm_metrics_pre.txt / vllm_metrics_post.txt). Accepts an out-root, a
    dataset dir, or a swe_out dir."""
    root = Path(out_root)
    dirs = sorted(root.glob("**/per_task/*"))
    return [d for d in dirs if (d / "vllm_metrics_pre.txt").exists()
            and (d / "vllm_metrics_post.txt").exists()]


def cmd_deploy_speed(args: argparse.Namespace) -> int:
    """DEPLOYMENT-regime SPEED (OFF, regime=deployment): s/fwd + accept/event +
    committed + derived TPS on the REAL codex trajectory, from the per-task
    /metrics brackets run_swe_bench_q36_a.py already captured.

    THE SAME truthful basis as cmd_speed (raw-counter delta, banned bases
    blocked, class-9 engagement asserted) — only the trajectory differs: this is
    the real SWE-Verified + codex deployment stream (chat-templated /v1/responses,
    multi-turn, tool calls), NOT the off-distribution raw-/v1/completions prompts.
    No degenerate fork: accept/event here is the DEPLOYMENT-TRAJECTORY number.

    --out-root = a run_swe_bench_q36_a out-root (e.g. output/fr13_bigdenom_swe/
    cat9_a/swe_out). --batch-size labels which co-residency regime the arm was
    booted in (B=1 MAX_NUM_SEQS=1 deployment, or B=4). --expected-tok-per-draft
    is the class-9 engagement gate (5 for native E5, len(TREE) for a tree arm)."""
    assert_speed_basis(args.basis)
    task_dirs = _find_task_dirs(args.out_root)
    if not task_dirs:
        raise RuntimeError(
            f"class-9 FAIL-LOUD [deploy-speed]: no per-task /metrics brackets "
            f"under {args.out_root!r} (vllm_metrics_pre/post.txt). The codex "
            "deployment run did not produce a bracketed task -- nothing to reduce."
        )
    # Aggregate the raw-counter delta OVER ALL tasks (the deployment workload).
    agg = {c: 0.0 for c in COUNTERS}
    per_task: list[dict[str, Any]] = []
    for d in task_dirs:
        before = _scrape_metrics_file(str(d / "vllm_metrics_pre.txt"))
        after = _scrape_metrics_file(str(d / "vllm_metrics_post.txt"))
        md = _delta(after, before)
        for c in COUNTERS:
            agg[c] += md[c]
        drafts = md[M_DRAFTS]
        tpot_sum_d = md.get(M_TPOT_SUM, 0.0)
        fwd_gpu_d = md.get(M_DECODE_FWD_GPU_S, 0.0)
        # MATCHED denominator (drafts on the pure-decode steps the timer measured),
        # NOT the global M_DRAFTS (which adds mixed-step drafts the numerator excludes
        # => prefill-load-dependent confound). None if the matched counter is absent
        # (timer off / pre-fix run) -> re-derive from the sidecar.
        fwd_gpu_drafts_d = md.get(M_DECODE_FWD_GPU_DRAFTS, 0.0)
        per_task.append({
            "instance_id": d.name,
            "drafts": drafts,
            "tok_per_draft": (md[M_DRAFT_TOK] / drafts) if drafts > 0 else None,
            "s_per_fwd": (md[M_DECODE_S] / drafts) if drafts > 0 else None,
            # FR13_SFWD_GPU_TIMER: prefill-INDEPENDENT decode-forward GPU time per
            # spec event = pure-decode GPU sec / drafts-on-those-pure-decode-steps.
            "s_per_fwd_gpu": (fwd_gpu_d / fwd_gpu_drafts_d) if (fwd_gpu_drafts_d > 0 and fwd_gpu_d > 0) else None,
            "accept_per_event": (md[M_ACCEPTED] / drafts) if drafts > 0 else None,
            "per_request_decode_tps": (md.get(M_TPOT_COUNT, 0.0) / tpot_sum_d) if tpot_sum_d > 0 else None,
        })

    # class-9 engagement: tok/draft over the WHOLE deployment workload == expected.
    spec_like = {"expected_tok_per_draft": args.expected_tok_per_draft, "arm": args.arm}
    eng = assert_engaged(spec_like, agg, context=f"deploy-speed {args.arm} B={args.batch_size}")

    drafts = agg[M_DRAFTS]
    s_fwd = agg[M_DECODE_S] / drafts if drafts > 0 else None
    # FR13_SFWD_GPU_TIMER: the PREFILL-INDEPENDENT decode-forward GPU time per spec
    # event = d(fr13_decode_forward_gpu_seconds_total)/d(fr13_decode_forward_gpu_
    # drafts_total) -- the MATCHED denominator (drafts on the pure-decode steps the
    # timer measured), NOT the global spec_decode_num_drafts_total. The global counter
    # adds drafts from mixed prefill+decode steps the timer excludes (~49% at B=4
    # deployment), which scales with prefill load and is arm-dependent => dividing by
    # it reintroduces the very prefill confound the metric removes. Both numerator and
    # this denominator are pure-decode-only. None unless the server booted timer ON
    # with the matched-denominator synthetic lines (else re-derive from the sidecar).
    agg_fwd_gpu = agg.get(M_DECODE_FWD_GPU_S, 0.0)
    agg_fwd_gpu_drafts = agg.get(M_DECODE_FWD_GPU_DRAFTS, 0.0)
    agg_fwd_gpu_steps = agg.get(M_DECODE_FWD_GPU_STEPS, 0.0)
    s_fwd_gpu = (agg_fwd_gpu / agg_fwd_gpu_drafts) if (agg_fwd_gpu_drafts > 0 and agg_fwd_gpu > 0) else None
    # per-pure-decode-FORWARD GPU time (matches the metric name + the banked B=1 0.218
    # per-forward), reported alongside the per-spec-event s_fwd_gpu.
    s_fwd_gpu_per_forward = (agg_fwd_gpu / agg_fwd_gpu_steps) if (agg_fwd_gpu_steps > 0 and agg_fwd_gpu > 0) else None
    # events-per-step (FIX 2026-07-22, fullstep_alignment_ratio ~0.52-0.54 root
    # cause): drafter_gpu_ms_per_step / committer_gpu_ms_per_step are genuinely
    # PER PHYSICAL STEP (_Fr13SpanTimer.end() increments n_spans by 1 once per
    # wrapped call -- one call per decode step, regardless of how many co-
    # resident requests it served; see class docstring "no pure-decode gate,
    # the wrapped op runs once per spec decode step"). s_fwd_gpu is genuinely
    # PER REQUEST-EVENT (_Fr13SfwdGpuTimer increments n_drafts by n_reqs, the
    # co-resident request count, each pure-decode step; verified at the
    # `self._n_drafts += int(n_reqs)` call site). Summing a per-event term with
    # two per-step terms without normalizing mixes scales whenever
    # events_per_step > 1 (B>1 co-residency) -- this WAS the confirmed root
    # cause of fullstep_alignment_ratio's persistent ~0.52-0.54 (derived
    # understated because the per-step drafter+committer costs were counted at
    # full step-cost per event instead of amortized across the events sharing
    # that step). events_per_step is the EXACT measured ratio for the SAME
    # pure-decode-step window s_fwd_gpu/s_fwd_gpu_per_forward are computed
    # over (not the coarser wall-clock effective_concurrency estimate computed
    # later from a different basis).
    events_per_step = (
        (agg_fwd_gpu_drafts / agg_fwd_gpu_steps)
        if (agg_fwd_gpu_steps and agg_fwd_gpu_steps > 0) else None
    )
    # FR13_DFWD/CFWD_GPU_TIMER: per-arm component GPU totals (drafter propose /
    # committer rejection-sampler dispatch) + per-step ms, from the synthetic
    # bracket lines (sidecar-sourced). null when a timer was off / the brackets
    # predate the wiring (no crash). Totals share the per-task-bracket overlap
    # caveat of the other summed aggregates at B>1; ms-per-step is a ratio, so
    # the overlap cancels.
    agg_drafter_s = agg.get(M_DRAFTER_GPU_S, 0.0)
    agg_drafter_spans = agg.get(M_DRAFTER_GPU_SPANS, 0.0)
    agg_committer_s = agg.get(M_COMMITTER_GPU_S, 0.0)
    agg_committer_spans = agg.get(M_COMMITTER_GPU_SPANS, 0.0)
    drafter_gpu_seconds = agg_drafter_s if agg_drafter_s > 0 else None
    drafter_gpu_ms_per_step = (
        (agg_drafter_s / agg_drafter_spans) * 1000.0
        if (agg_drafter_s > 0 and agg_drafter_spans > 0) else None
    )
    committer_gpu_seconds = agg_committer_s if agg_committer_s > 0 else None
    committer_gpu_ms_per_step = (
        (agg_committer_s / agg_committer_spans) * 1000.0
        if (agg_committer_s > 0 and agg_committer_spans > 0) else None
    )
    accept_per_event = agg[M_ACCEPTED] / drafts if drafts > 0 else None
    committed_per_event = (accept_per_event + 1.0) if accept_per_event is not None else None
    derived_tps = (committed_per_event / s_fwd) if (s_fwd and committed_per_event) else None
    # derived GPU TPS uses the prefill-independent per-forward time (the kernel-
    # speed view; the wall-span derived_tps stays as the concurrency-summed basis).
    derived_tps_gpu = (committed_per_event / s_fwd_gpu) if (s_fwd_gpu and committed_per_event) else None
    # TRUE full-step compute TPS: committed / (drafter + verify + committer) per decode step.
    # derived_tps_gpu above is verify-ONLY (optimistic; blind to the drafter). For the merged
    # skip/fill drafter MODES the drafter time is exactly what changes, so this drafter-INCLUSIVE
    # basis is the right speed metric to compare arms. (Excludes host gaps -> still a compute-basis
    # upper bound, but it captures the drafter delta the verify-only metric cannot.)
    # UNIT FIX (see events_per_step above): drafter/committer per-step costs are amortized across
    # events_per_step before adding to the already-per-event s_fwd_gpu, so every term here is on
    # the SAME per-request-event basis (matching committed_per_event's own basis). None (not a
    # silently-mixed-unit number) when events_per_step isn't available (timer off / no steps
    # counted) -- a basis-mismatched number is worse than no number.
    _drafter_s_step = (drafter_gpu_ms_per_step / 1000.0) if drafter_gpu_ms_per_step else 0.0
    _committer_s_step = (committer_gpu_ms_per_step / 1000.0) if committer_gpu_ms_per_step else 0.0
    if events_per_step and events_per_step > 0:
        _fullstep_s = (s_fwd_gpu or 0.0) + (_drafter_s_step + _committer_s_step) / events_per_step
    else:
        _fullstep_s = None
    derived_tps_fullstep_gpu = (
        (committed_per_event / _fullstep_s)
        if (_fullstep_s and _fullstep_s > 0 and committed_per_event) else None
    )
    # FR13_STEP_WALL alignment (user gate): the derived fullstep TPS is a
    # compute-basis upper bound blind to overhead OUTSIDE verify+drafter+
    # committer. Report the MEASURED wall TPS on the same per-event basis and
    # the residual bucket; a large residual or misalignment invalidates any
    # cross-arm verdict made on the derived number alone.
    agg_wall_s = agg.get(M_STEP_WALL_S, 0.0)
    agg_wall_drafts = agg.get(M_STEP_WALL_DRAFTS, 0.0)
    agg_wall_steps = agg.get(M_STEP_WALL_STEPS, 0.0)
    wall_s_per_event = (
        (agg_wall_s / agg_wall_drafts)
        if (agg_wall_s > 0 and agg_wall_drafts > 0) else None
    )
    measured_tps_fullstep_wall = (
        (committed_per_event / wall_s_per_event)
        if (wall_s_per_event and committed_per_event) else None
    )
    overhead_other_ms_per_event = (
        (wall_s_per_event - _fullstep_s) * 1000.0
        if (wall_s_per_event is not None and _fullstep_s) else None
    )
    fullstep_alignment_ratio = (
        (derived_tps_fullstep_gpu / measured_tps_fullstep_wall)
        if (derived_tps_fullstep_gpu and measured_tps_fullstep_wall) else None
    )

    # NON-deflated per-STREAM decode rate: count/sum of vLLM
    # request_time_per_output_token = 1/avg(per-request mean-TPOT). Per-request
    # normalized (NOT the concurrency-summed gen_tok/decode_s_sum basis FR10 flagged).
    tpot_sum = agg[M_TPOT_SUM]
    per_request_decode_tps = (agg[M_TPOT_COUNT] / tpot_sum) if tpot_sum > 0 else None

    # Aggregate throughput (REPORTING-ONLY, not a kernel-speed basis): total
    # generation_tokens over the UNION measurement window / union wall. The per-task
    # brackets OVERLAP at B>1, so use the earliest-pre -> latest-post SINGLE delta
    # (NOT the summed agg[M_GEN_TOK], which double-counts the overlapping windows).
    aggregate_decode_tps = None
    aggregate_window_wall_s = None
    union_decode_s = None  # earliest-pre -> latest-post single delta (NOT the summed
    # agg[M_DECODE_S], which double-counts the OVERLAPPING per-task brackets at B>1).
    try:
        earliest_pre = min((d / "vllm_metrics_pre.txt" for d in task_dirs),
                           key=lambda p: p.stat().st_mtime)
        latest_post = max((d / "vllm_metrics_post.txt" for d in task_dirs),
                          key=lambda p: p.stat().st_mtime)
        wall = latest_post.stat().st_mtime - earliest_pre.stat().st_mtime
        pre_m = _scrape_metrics_file(str(earliest_pre))
        post_m = _scrape_metrics_file(str(latest_post))
        gen0, gen1 = pre_m.get(M_GEN_TOK, 0.0), post_m.get(M_GEN_TOK, 0.0)
        union_decode_s = post_m.get(M_DECODE_S, 0.0) - pre_m.get(M_DECODE_S, 0.0)
        if wall > 0 and gen1 > gen0:
            aggregate_window_wall_s = wall
            aggregate_decode_tps = (gen1 - gen0) / wall
    except OSError:
        pass

    # prefill_frac: how much GPU-wall went to prefill vs decode. HIGH prefill (long
    # re-prefilled codex contexts) drags the gen/wall aggregate down WITHOUT being a
    # decode slowdown - the workload confound behind "why is our aggregate low" (user
    # 2026-06-16). request_decode_time EXCLUDES prefill, so per_request_decode_tps is
    # unaffected; only aggregate_decode_tps (gen/wall) absorbs it. (A RATIO -> the
    # bracket-overlap over-count cancels, so the summed agg is fine here.)
    prefill_frac = (agg[M_PREFILL_S] / agg[M_DECODE_S]) if agg[M_DECODE_S] > 0 else None
    # effective_concurrency: UNION decode-stream-seconds / union wall = avg co-resident
    # streams over the window (bounded by batch_size). aggregate_decode_tps ~=
    # per_request_decode_tps x eff_conc, separating batch-fullness from per-stream (the
    # 39.9-vs-ours decomposition; lower eff_conc = drained batch / fewer tasks). MUST
    # use union_decode_s, NOT summed agg[M_DECODE_S] (the latter gives >batch_size).
    effective_concurrency = (
        (union_decode_s / aggregate_window_wall_s)
        if (union_decode_s and aggregate_window_wall_s and aggregate_window_wall_s > 0)
        else None
    )
    # CROSS-CHECK (superseded 2026-07-22 the old rg1 "_norm" double-correction --
    # that field divided _fullstep_s by effective_concurrency AGAIN on top of the
    # events_per_step normalization now baked into _fullstep_s itself, which would
    # double-amortize the drafter/committer terms). events_per_step (GPU-timer-
    # native, exact for the pure-decode-step window s_fwd_gpu/drafter/committer
    # were measured over) and effective_concurrency (wall-clock, union decode-
    # seconds / union wall, a DIFFERENT independent estimate of average co-
    # residency) SHOULD roughly agree; a ratio far from 1 flags that one of the
    # two concurrency estimates is off (e.g. a workload whose batch occupancy
    # during pure-decode steps differs from its overall wall-clock occupancy).
    events_per_step_vs_effective_concurrency_ratio = (
        (events_per_step / effective_concurrency)
        if (events_per_step and effective_concurrency and effective_concurrency > 0)
        else None
    )
    # per_stream_decode_rate = union gen / union decode-seconds = the EXACT
    # decomposition factor: aggregate_decode_tps == per_stream_decode_rate x
    # effective_concurrency (identity, both from the same union deltas). This is the
    # ratio-of-sums per-stream rate; it differs slightly from per_request_decode_tps
    # (the per-request-mean TPOT basis) - use THIS one for the aggregate decomposition.
    _union_gen = None
    try:
        _union_gen = post_m.get(M_GEN_TOK, 0.0) - pre_m.get(M_GEN_TOK, 0.0)
    except (NameError, TypeError):
        _union_gen = None
    per_stream_decode_rate = (
        (_union_gen / union_decode_s)
        if (_union_gen and union_decode_s and union_decode_s > 0) else None
    )

    rec = {
        "schema": "fr13.measure.deploy_speed.v1",
        "kind": "speed",
        "instrument": "OFF",
        "regime": "deployment",  # real SWE-Verified + codex (the canonical path)
        "regime_note": (
            "DEPLOYMENT trajectory: s/fwd + accept/event over the REAL codex agent "
            "loop on SWE-bench-Verified tasks (chat-templated /v1/responses, "
            "multi-turn, tool calls), from the per-task /metrics brackets. NO "
            "degenerate raw-prompt fork; accept is on the deployment trajectory."
        ),
        "label": f"deploy_speed_{args.arm}_b{args.batch_size}",
        "arm": args.arm,
        "batch_size": args.batch_size,
        "out_root": args.out_root,
        "n_tasks": len(task_dirs),
        "task_instance_ids": [d.name for d in task_dirs],
        "engagement": eng,
        "speed_basis": "d(request_decode_time_seconds_sum)/d(spec_decode_num_drafts_total)",
        "s_per_fwd": s_fwd,
        "s_per_fwd_note": (
            "WALL-SPAN basis (request_decode_time = last_token_ts - first_token_ts, "
            "vLLM v1/metrics/stats.py: 'preemptions during decode are included'). "
            "PREFILL-CONFOUNDED at B>1: the decode window absorbs co-resident "
            "chunked-prefill the scheduler interleaves into it (high prefill_frac) "
            "+ idle. Use s_per_fwd_gpu for the prefill-independent decode-kernel time."
        ),
        # FR13_SFWD_GPU_TIMER: prefill-INDEPENDENT decode-forward GPU time per spec
        # event. None unless the server booted with FR13_SFWD_GPU_TIMER=1.
        "s_per_fwd_gpu": s_fwd_gpu,
        "s_per_fwd_gpu_per_forward": s_fwd_gpu_per_forward,
        "s_per_fwd_gpu_basis": "d(fr13_decode_forward_gpu_seconds_total)/d(fr13_decode_forward_gpu_drafts_total) [MATCHED pure-decode drafts; NOT global spec_decode_num_drafts_total]",
        "s_per_fwd_gpu_note": (
            "PREFILL-INDEPENDENT: GPU-active time in the PURE-DECODE model-forward "
            "(tree TREE_ATTN / MTP verify forward) only, summed over uniform-decode "
            "steps via async cuda events (gpu_model_runner.execute_model). Divided by "
            "the MATCHED draft count (drafts on those same pure-decode steps), NOT the "
            "global spec_decode_num_drafts_total: the global counter adds drafts from "
            "mixed prefill+decode steps the timer excludes (~49% at B=4 deployment), a "
            "prefill-load-dependent, arm-dependent confound that would defeat prefill-"
            "independence. s_per_fwd_gpu_per_forward = per pure-decode FORWARD (matches "
            "the banked B=1 ~0.218). None => timer-OFF (default; byte-identical) OR a "
            "pre-fix run whose brackets lack the matched-denominator lines (re-derive "
            "from the sidecar fwd_gpu/n_drafts_in_timed_steps)."
        ),
        "derived_tps_gpu": derived_tps_gpu,
        "derived_tps_gpu_note": (
            "DERIVED = committed_per_event / s_per_fwd_gpu = decode-kernel TPS "
            "(prefill-independent). None unless the timer was on. VERIFY-ONLY -- blind to the "
            "drafter; use derived_tps_fullstep_gpu to compare drafter MODES."
        ),
        "derived_tps_fullstep_gpu": derived_tps_fullstep_gpu,
        "derived_tps_fullstep_gpu_note": (
            "committed_per_event / (drafter + verify + committer) per decode step -- the "
            "drafter-INCLUSIVE compute TPS (captures the merged skip/fill drafter modes' effect, "
            "which the verify-only derived_tps_gpu cannot). Excludes host gaps. FIXED 2026-07-22: "
            "drafter/committer GPU-ms are measured PER PHYSICAL STEP (_Fr13SpanTimer: one span per "
            "wrapped call, regardless of co-resident request count) while s_fwd_gpu is measured PER "
            "REQUEST-EVENT (_Fr13SfwdGpuTimer: n_drafts += co-resident request count); summing them "
            "unnormalized mixed scales whenever events_per_step>1 and was the confirmed root cause "
            "of fullstep_alignment_ratio sitting at ~0.52-0.54 regardless of arm. Now amortized by "
            "the measured events_per_step (see that field) before summing -- all terms on the same "
            "per-event basis. None (not a silently-mixed number) when events_per_step is unavailable."
        ),
        "events_per_step": events_per_step,
        "events_per_step_note": (
            "d(fr13_decode_forward_gpu_drafts_total)/d(fr13_decode_forward_gpu_steps_total) -- "
            "average co-resident spec-decode requests per PHYSICAL pure-decode step, measured over "
            "the EXACT same step window s_fwd_gpu/drafter/committer GPU timers cover (more precise "
            "than the wall-clock effective_concurrency estimate below, which covers the whole task "
            "window incl. prefill/idle). The normalizer that fixes derived_tps_fullstep_gpu."
        ),
        # FR13_STEP_WALL (user gate): MEASURED wall twin + residual. A speed
        # verdict quoting derived_tps_fullstep_gpu MUST also quote these; if
        # fullstep_alignment_ratio drifts far from 1 the derived basis is
        # missing real overhead and the verdict must use the measured number.
        "measured_tps_fullstep_wall": measured_tps_fullstep_wall,
        "measured_tps_fullstep_wall_note": (
            "committed_per_event / MEASURED wall per event (start-to-start deltas "
            "between consecutive pure-decode steps, idle-capped, chain broken on "
            "mixed/prefill steps). Includes ALL step cost: host glue, sampler, "
            "packer, scheduler gap."
        ),
        "wall_s_per_event": wall_s_per_event,
        "wall_steps_measured": agg_wall_steps or None,
        "overhead_other_ms_per_event": overhead_other_ms_per_event,
        "overhead_other_note": (
            "wall_s_per_event - _fullstep_s (now basis-matched, both per-event -- see "
            "derived_tps_fullstep_gpu_note). The genuine non-component overhead per event: host "
            "glue, sampler, packer, scheduler gap that the GPU-only components don't cover. Was "
            "'RAW basis-mismatched' pre-fix; the old _norm twin was a compounding double-correction "
            "(divided an already-mixed number by effective_concurrency again) and has been replaced "
            "by events_per_step_vs_effective_concurrency_ratio, a genuine cross-check instead."
        ),
        "events_per_step_vs_effective_concurrency_ratio": events_per_step_vs_effective_concurrency_ratio,
        "events_per_step_vs_effective_concurrency_ratio_note": (
            "events_per_step (GPU-timer-native, pure-decode-step window) / effective_concurrency "
            "(wall-clock, whole-task window). Two independent estimates of average co-residency; "
            "should be roughly 1. Far from 1 flags that pure-decode-step occupancy differs "
            "meaningfully from the task's overall wall-clock occupancy (e.g. heavy prefill "
            "interleaving) -- investigate before trusting either normalization."
        ),
        "fullstep_alignment_ratio": fullstep_alignment_ratio,
        "fullstep_alignment_ratio_note": (
            "derived_tps_fullstep_gpu / measured_tps_fullstep_wall. Post-fix (2026-07-22) this "
            "should sit close to but somewhat below 1 -- the residual gap is REAL host overhead "
            "(overhead_other_ms_per_event) that the GPU-only derived basis structurally excludes, "
            "not a unit-mismatch artifact. A ratio still far below 1 (e.g. <0.8) after the fix "
            "warrants investigating overhead_other_ms_per_event directly."
        ),
        # FR13_DFWD/CFWD_GPU_TIMER component spans (where the tree overhead
        # lives per reference_tree_tps_overhead_bound): per-arm totals +
        # per-step ms. null => timer OFF (default; byte-identical) or the
        # brackets lack the sidecar-synthesized lines.
        "drafter_gpu_seconds": drafter_gpu_seconds,
        "drafter_gpu_ms_per_step": drafter_gpu_ms_per_step,
        "committer_gpu_seconds": committer_gpu_seconds,
        "committer_gpu_ms_per_step": committer_gpu_ms_per_step,
        "component_gpu_note": (
            "FR13_DFWD/CFWD_GPU_TIMER spans: drafter = propose_draft_token_ids "
            "(all D spine forwards); committer = the spec-decode rejection-"
            "sampler dispatch in _sample (accept/LCP/bonus decision + commit; "
            "includes the host committer loop's packed DtoH+sync when "
            "FR13_GPU_COMMITTER=0). Async cuda-event spans summed over the "
            "per-task brackets (d(vllm:fr13_drafter/committer_gpu_seconds_"
            "total)); ms_per_step = seconds/spans*1000, one span per spec "
            "decode step. Totals share the per-task-bracket overlap caveat at "
            "B>1; the ms-per-step ratio does not."
        ),
        "accept_per_event": accept_per_event,
        "accept_per_event_note": (
            "DEPLOYMENT-TRAJECTORY accept (real codex loop). B-DEPENDENT; valid "
            "for this batch_size on this workload. NOT the off-distribution raw-"
            "prompt accept (which forked degenerate)."
        ),
        "committed_per_event": committed_per_event,
        "derived_tps": derived_tps,
        "derived_tps_note": (
            "DERIVED = committed_per_event / s_fwd = generation_tokens / "
            "request_decode_time_seconds_sum = the CONCURRENCY-SUMMED basis "
            "(decode-seconds summed over co-resident requests). FR10 flagged this "
            "as NOT directly E5-comparable at B>1. Use per_request_decode_tps "
            "(per-stream rate) or aggregate_decode_tps (GPU throughput) instead."
        ),
        "per_request_decode_tps": per_request_decode_tps,
        "per_request_decode_tps_note": (
            "NON-deflated per-STREAM decode rate = count/sum of vLLM "
            "request_time_per_output_token (= 1/avg per-request mean-TPOT). "
            "Per-request normalized (NOT concurrency-summed); includes the real "
            "co-residency cost at B>1. The FR10-recommended per-request basis "
            "(cross-references the pre-kernel B=4 per-request ~28 synthetic / "
            "B=1 ~18 deployment)."
        ),
        "aggregate_decode_tps": aggregate_decode_tps,
        "aggregate_window_wall_s": aggregate_window_wall_s,
        "aggregate_decode_tps_note": (
            "END-TO-END throughput = total generation_tokens / UNION wall (earliest-"
            "pre -> latest-post). The wall is IDLE- AND PREFILL-INCLUSIVE, so this is "
            "the SAME basis as the fr9 steptrace decode_tps=39.9 baseline (gen/wall) "
            "and is the ONLY field comparable to it - but ONLY at matched task-count + "
            "concurrency + prefill_frac (see effective_concurrency + prefill_frac). It "
            "is NOT decode capacity (a high-prefill / drained-batch run reads low here "
            "without being slower per forward). For per-forward/decode speed use s/fwd "
            "or per_request_decode_tps; aggregate = per_stream_decode_rate x "
            "effective_concurrency (exact identity)."
        ),
        "per_stream_decode_rate": per_stream_decode_rate,
        "effective_concurrency": effective_concurrency,
        "prefill_frac": prefill_frac,
        "prefill_frac_note": (
            "request_prefill_time_seconds_sum / request_decode_time_seconds_sum = the "
            "workload's prefill-vs-decode GPU-time ratio. HIGH (e.g. 0.39 for the 4 "
            "astropy tasks vs 0.108 for fr9's 16-task set) drags aggregate_decode_tps "
            "down WITHOUT a per-forward slowdown. The driver behind aggregate gaps; "
            "MATCH it before comparing aggregates across runs."
        ),
        "raw_counter_delta_aggregate": agg,
        "per_task": per_task,
        "ts": time.time(),
    }
    assert_no_mode_mix([rec])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in (
        "arm", "regime", "batch_size", "n_tasks", "s_per_fwd", "s_per_fwd_gpu",
        "accept_per_event", "committed_per_event", "derived_tps", "derived_tps_gpu",
        "per_request_decode_tps", "aggregate_decode_tps",
        "effective_concurrency", "prefill_frac", "instrument")},
        indent=2))
    return 0


def cmd_deploy_lossless(args: argparse.Namespace) -> int:
    """DEPLOYMENT-regime LOSSLESS verdict (ON, regime=deployment): the within-
    floor lossless gate on the REAL codex trajectory.

    This CONSUMES the big-denom rescore consolidation
    (fr13.bigdenom_rescore_consolidate.v1, produced by
    fr13_bigdenom_phase3_rescore.sh: proxy pair-dump -> byte-exact detok src ->
    no-spec RECURRENT decode oracle -> clear-margin flip rate + Wilson CI). Each
    arm is scored vs ITS OWN no-spec recurrent oracle on its OWN served codex
    stream; native-E5 is the within-floor BAR (feedback_fr13_lossless_compare_
    target: US vs native-E5 vs no-spec-decode, never a proxy).

    The verdict: cat9 is lossless-within-floor iff its clear-margin flip rate is
    NOT statistically above native-E5's (Wilson CIs OVERLAP, i.e. cat9 is not
    separated-above native). A high absolute flip rate that BOTH arms share is the
    deployment self-noise floor, NOT a cat9 defect (scalar_metric blindspot: the
    per-token clear-margin definition is the binding instrument, not bag-TV)."""
    cons = json.loads(Path(args.consolidated).read_text(encoding="utf-8"))
    if cons.get("schema") != "fr13.bigdenom_rescore_consolidate.v1":
        raise RuntimeError(
            f"class-9 FAIL-LOUD [deploy-lossless]: --consolidated must be a "
            f"fr13.bigdenom_rescore_consolidate.v1 artifact, got {cons.get('schema')!r}."
        )
    nv = cons.get("non_vacuity", {})
    if not nv.get("oracle_engaged_both", False):
        raise RuntimeError(
            "class-9 FAIL-LOUD [deploy-lossless]: the big-denom rescore oracle was "
            "NOT engaged on both arms (RECURRENT_PATH_ENGAGED / recurrent_decode_"
            "calls == 0) -- the lossless verdict would be vacuous."
        )
    if not nv.get("denominator_is_validated_roundtrip_tokens", False):
        raise RuntimeError(
            "class-12 FAIL-LOUD [deploy-lossless]: the flip-rate denominator is NOT "
            "the round-trip-validated token count (text-length / silent re-count) -- "
            "refusing to bind a lossless verdict on a fictional denominator."
        )
    cat9 = cons["cat9"]
    native = cons["native"]
    # within-floor: cat9 NOT separated-above native (Wilson CIs overlap).
    cat9_above_native = bool(cons.get("ci_separated_cat9_above_native", False))
    within_floor = not cat9_above_native
    result = {
        "schema": "fr13.measure.deploy_lossless.v1",
        "kind": "lossless",
        "instrument": "ON",
        "regime": "deployment",
        "compare_target": cons.get("compare_target"),
        "clear_margin_def": cons.get("clear_margin_def"),
        "source_consolidated": args.consolidated,
        # the 4 lossless numbers ON the deployment trajectory ---------------
        "cat9_clear_margin_rate_ci": cat9.get("rate_ci_str"),
        "native_clear_margin_rate_ci": native.get("rate_ci_str"),
        "cat9_clear_margin_rate": cat9.get("clear_margin_rate"),
        "native_clear_margin_rate": native.get("clear_margin_rate"),
        "cat9_n_positions": cat9.get("total_positions_rescored"),
        "native_n_positions": native.get("total_positions_rescored"),
        "cat9_above_native_separated": cat9_above_native,
        "within_floor_verdict": "LOSSLESS_within_floor" if within_floor else "ABOVE_floor",
        "within_floor": within_floor,
        "within_proc_determinism_both": cons.get("within_proc_determinism_both"),
        "non_vacuity": nv,
        "verdict_note": (
            "DEPLOYMENT-regime lossless: each arm vs its OWN no-spec RECURRENT "
            "decode oracle on its OWN codex served stream; native-E5 is the BAR. "
            "cat9 lossless-within-floor iff its clear-margin flip rate is not "
            "Wilson-separated ABOVE native's (shared rate = deployment self-noise "
            "floor, not a cat9 defect). This is the binding per-token instrument "
            "(reference_scalar_metric_per_token_blindspot), not bag-TV."
        ),
        "ts": time.time(),
    }
    assert_no_mode_mix([result])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "regime", "cat9_clear_margin_rate_ci", "native_clear_margin_rate_ci",
        "within_floor_verdict", "within_proc_determinism_both", "instrument")},
        indent=2))
    return 0


# --------------------------------------------------------------------------- #
# DEPLOYMENT temp-0.6 DRIFT reduce (CPU) — q (spec verify) vs p (recurrent),    #
# both forced onto the SAME deployment served stream, step/id-aligned.          #
# --------------------------------------------------------------------------- #
def _q_deploy_by_pos(q_art: dict[str, Any], pid: int, q_path: str) -> dict[int, dict[str, float]]:
    """{step -> {str(token_id): logprob}} of the SPEC VERIFY top-K q for one
    prompt, from the deployment capture-q-deploy q-sinks (verify_topk_*). Keyed
    by token id (id-keyed by construction)."""
    out: dict[int, dict[str, float]] = {}
    pp = {p["prompt_id"]: p for p in q_art.get("per_prompt", [])}.get(pid)
    candidates: list[Path] = []
    if pp and pp.get("qsink"):
        candidates.append(Path(pp["qsink"]))
    qsink_dir = q_art.get("qsink_dir")
    if qsink_dir:
        candidates.append(Path(qsink_dir) / f"p{pid}_rep0.jsonl")
    candidates.append(Path(q_path).parent / f"{q_art.get('arm','')}_qsinks" / f"p{pid}_rep0.jsonl")
    for sink in candidates:
        if sink.exists():
            for line in sink.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids = rec.get("verify_topk_ids") or []
                lps = rec.get("verify_topk_logprobs") or []
                if ids and lps:
                    out[rec["step"]] = {str(int(t)): float(v) for t, v in zip(ids, lps)}
            if out:
                return out
    return out


def _p_deploy_by_pos(p_art: dict[str, Any], pid: int, p_path: str) -> dict[int, dict[str, float]]:
    """{step -> {str(token_id): logprob}} of the RECURRENT oracle top-K p for one
    prompt. Reuses the SAME sink/positions resolution as the canonical
    _p_topk_by_pos (bigdenom <arm>_sinks/pN_rep0.jsonl oracle_topk_*), but ALSO
    accepts the bigdenom default sink layout (sink_dir absent in the artifact but
    the on-disk sinks present next to it)."""
    out = _p_topk_by_pos(p_art, pid, p_path)
    if out:
        return out
    # bigdenom layout: <consolidated_dir>/<arm>_sinks/pN_rep0.jsonl. The rescore
    # artifact may have sink_dir=None (older runs); locate the sink next to it.
    arm = p_art.get("arm", "")
    base = Path(p_path).parent / f"{arm}_sinks"
    sink = base / f"p{pid}_rep0.jsonl"
    if sink.exists():
        for line in sink.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = rec.get("oracle_topk_ids") or []
            lps = rec.get("oracle_topk_logprobs") or []
            if ids and lps:
                out[rec["step"]] = {str(int(t)): float(v) for t, v in zip(ids, lps)}
    return out


def cmd_deploy_temp06_drift(args: argparse.Namespace) -> int:
    """DEPLOYMENT-regime temp-0.6 distributional DRIFT (ON, regime=deployment):
    per-position TV(softmax(q/0.6), softmax(p/0.6)) on the REAL codex served
    stream, each-arm-vs-its-OWN oracle, vs the DEPTH-MATCHED native floor.

    q = the SPEC VERIFY top-K (capture-q-deploy: forced-decode the deployment
        served stream through the spec serve; verify_topk_*; ID-KEYED).
    p = the no-spec RECURRENT decode oracle top-K (the bigdenom rescore of the
        SAME served stream; oracle_topk_*; RECURRENT_PATH_ENGAGED).
    q and p are forced onto the IDENTICAL served stream (same <arm>_src.json), so
    they are step/id-aligned BY CONSTRUCTION -- no re-key, no string match, no
    streamed-logprob off-by-one (the trust-ledger #1 gap the off-distribution
    temp06-drift could only reach via a re-key).

    The DEPLOYMENT difference vs the off-distribution temp06-drift:
      * q is the verify dist on the PINNED deployment stream (not a free-running
        raw-prompt generation that degenerates into the <think></think> loop);
      * p comes from the big-denom recurrent rescore (the canonical deployment
        oracle), read from its per-step full-top-K sinks.
    PASS = the arm's drift is NOT separated above its DEPTH-MATCHED native floor
    (cat9/depth-5 -> native-E5 floor; 3-3-3/depth-3 -> native-E3 floor). Pair the
    scalar with the per-position vector (reference_scalar_metric_per_token_blindspot)."""
    q_art = json.loads(Path(args.q).read_text(encoding="utf-8"))
    p_art = json.loads(Path(args.p).read_text(encoding="utf-8"))
    if q_art.get("schema") != "fr13.recurrent_decode_oracle.capture_q_deploy.v1":
        raise RuntimeError(
            "class-9 FAIL-LOUD [deploy-temp06-drift]: --q must be a "
            "fr13.recurrent_decode_oracle.capture_q_deploy.v1 artifact (the SPEC "
            f"verify q forced onto the deployment stream), got {q_art.get('schema')!r}."
        )
    if not q_art.get("SPEC_VERIFY_ENGAGED", False):
        raise RuntimeError(
            "class-9 FAIL-LOUD [deploy-temp06-drift]: the q capture did NOT engage "
            "the spec verify forward (SPEC_VERIFY_ENGAGED != True) -- vacuous q."
        )
    if p_art.get("schema") != "fr13.recurrent_decode_oracle.rescore.v1":
        raise RuntimeError(
            "class-9 FAIL-LOUD [deploy-temp06-drift]: --p must be a "
            "fr13.recurrent_decode_oracle.rescore.v1 artifact (the no-spec "
            f"recurrent deployment oracle), got {p_art.get('schema')!r}."
        )
    if not p_art.get("RECURRENT_PATH_ENGAGED", False):
        raise RuntimeError(
            "class-9 FAIL-LOUD [deploy-temp06-drift]: p oracle does not assert "
            "RECURRENT_PATH_ENGAGED -- wrong/chunked oracle frame, refusing."
        )
    if q_art.get("src") != p_art.get("src"):
        # q and p MUST be the SAME served stream or the per-position align is a
        # fiction (different stream -> different forced tokens at each step).
        raise RuntimeError(
            "class-12 FAIL-LOUD [deploy-temp06-drift]: q.src != p.src "
            f"({q_art.get('src')!r} vs {p_art.get('src')!r}) -- q and p must be "
            "forced onto the IDENTICAL deployment served stream to be aligned."
        )
    temp = args.temp
    # The served stream pins which token is forced at each step; align q[step] to
    # p[step] by step index (both forced the same served id at that step).
    n_prompts = max(
        (pp["prompt_id"] for pp in q_art.get("per_prompt", [])), default=-1) + 1

    forks: list[dict[str, Any]] = []
    all_tv: list[float] = []
    all_kl: list[float] = []
    over_floor_total = 0
    n_q_only = 0   # steps with q but no p (align miss)
    n_p_only = 0   # steps with p but no q (align miss)
    n_id_mismatch_served = 0  # served id absent from one side's support
    for pid in range(n_prompts):
        q_by = _q_deploy_by_pos(q_art, pid, args.q)
        p_by = _p_deploy_by_pos(p_art, pid, args.p)
        if not q_by or not p_by:
            forks.append({"prompt_id": pid,
                          "note": f"no {'q' if not q_by else 'p'} top-K for prompt"})
            continue
        steps = sorted(set(q_by) & set(p_by))
        n_q_only += len(set(q_by) - set(p_by))
        n_p_only += len(set(p_by) - set(q_by))
        per_pos: list[dict[str, Any]] = []
        for st in steps:
            qmap, pmap = q_by[st], p_by[st]
            q_at = _softmax_over_support_at_temp(qmap, temp)
            p_at = _softmax_over_support_at_temp(pmap, temp)
            overlap = set(q_at) & set(p_at)
            tv = _tv(q_at, p_at)
            kl = _kl(p_at, q_at)
            entry = {
                "pos": st,
                "tv_q_p_at_temp": tv,
                "kl_p_q_at_temp": kl,
                "n_support_overlap": len(overlap),
                "q_support": len(q_at),
                "p_support": len(p_at),
                "align_status": "ok",
            }
            per_pos.append(entry)
            all_tv.append(tv)
            all_kl.append(kl)
            if args.per_position_floor is not None and tv > args.per_position_floor:
                over_floor_total += 1
        forks.append({
            "prompt_id": pid,
            "n_positions": len(per_pos),
            "mean_tv": (statistics.fmean([e["tv_q_p_at_temp"] for e in per_pos])
                        if per_pos else None),
            "max_tv": (max(e["tv_q_p_at_temp"] for e in per_pos) if per_pos else None),
            "positions": per_pos if args.dump_positions else per_pos[:5],
        })

    # depth-matched native floor verdict (if a floor was supplied).
    arm_p95 = (sorted(all_tv)[int(0.95 * (len(all_tv) - 1))] if all_tv else None)
    above_floor = None
    if args.native_floor_p95 is not None and arm_p95 is not None:
        above_floor = arm_p95 > args.native_floor_p95
    result = {
        "schema": "fr13.measure.deploy_temp06_drift.v1",
        "kind": "drift",
        "instrument": "ON",
        "regime": "deployment",
        "label": f"deploy_temp06_drift_{q_art.get('arm')}",
        "arm": q_art.get("arm"),
        "depth_matched_native_floor_p95": args.native_floor_p95,
        "depth_match_note": (
            "feedback_depth_matched_accept_compare: a depth-D tree's TV floor is "
            "native MTP-D (cat9/depth-5 -> native-E5; 3-3-3/depth-3 -> native-E3), "
            "NOT E5 for every arm. Supply --native-floor-p95 = the p95 of the "
            "DEPTH-MATCHED native deploy-temp06-drift run."
        ),
        "temp": temp,
        "q_artifact": args.q,
        "p_artifact": args.p,
        "src": q_art.get("src"),
        "aligned_by": "served-stream step index (q.src == p.src, id-keyed)",
        "per_position_floor": args.per_position_floor,
        "mean_tv_q_p_at_temp": (statistics.fmean(all_tv) if all_tv else None),
        "p95_tv_q_p_at_temp": arm_p95,
        "max_tv_q_p_at_temp": (max(all_tv) if all_tv else None),
        "mean_kl_p_q_at_temp": (statistics.fmean(all_kl) if all_kl else None),
        "n_positions_scored": len(all_tv),
        "n_q_only_steps": n_q_only,
        "n_p_only_steps": n_p_only,
        "over_floor_count": over_floor_total,
        "arm_p95_above_native_floor": above_floor,
        "within_floor_verdict": (
            None if above_floor is None
            else ("ABOVE_floor" if above_floor else "WITHIN_floor")),
        "non_vacuity": {
            "q_spec_verify_engaged": q_art.get("SPEC_VERIFY_ENGAGED"),
            "p_recurrent_engaged": p_art.get("RECURRENT_PATH_ENGAGED"),
            "q_p_same_served_stream": (q_art.get("src") == p_art.get("src")),
            "q_id_keyed": q_art.get("q_id_keyed"),
            "n_positions_scored_gt0": len(all_tv) > 0,
        },
        "interpretation_note": (
            "DEPLOYMENT temp-0.6 drift: q (SPEC VERIFY top-K) vs p (no-spec "
            "RECURRENT oracle), BOTH forced onto the SAME codex served stream "
            "(q.src == p.src) so per-position alignment is exact. Each arm vs its "
            "OWN oracle; native-of-matching-depth is the floor. PAIR the scalar "
            "with the per-position vector (over_floor_count locates the drift); a "
            "high shared TV is the deployment self-noise floor, NOT a defect."
        ),
        "forks": forks,
        "ts": time.time(),
    }
    assert_no_mode_mix([result])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "arm", "regime", "temp", "mean_tv_q_p_at_temp", "p95_tv_q_p_at_temp",
        "max_tv_q_p_at_temp", "n_positions_scored", "over_floor_count",
        "within_floor_verdict", "instrument")}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common_capture(sp):
        sp.add_argument("--arm", required=True,
                        help="native_eN (N=3..8) or a tree name (needs --tree)")
        sp.add_argument("--tree", default=None,
                        help="caterpillar TREE for tree arms, e.g. cat9 list")
        sp.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--prompts-file", default=CANONICAL_PROMPTS)
        sp.add_argument("--seed", type=int, default=CANONICAL_SEED)
        sp.add_argument("--max-tokens", type=int, default=CANONICAL_MAX_TOKENS)
        sp.add_argument("--request-timeout", type=float, default=900.0)
        sp.add_argument("--wait-health", type=float, default=0.0)
        sp.add_argument("--out", required=True)

    s = sub.add_parser("speed", help="OFF-mode s/fwd + accept (deployment speed)")
    common_capture(s)
    s.add_argument("--batch-size", type=int, default=1,
                   help="1 = SEQUENTIAL B=1 (default); 4 = co-resident B=4")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--top-p", type=float, default=1.0)
    s.add_argument("--basis", default="decode_seconds",
                   help="speed basis; banned: tps/accept/wall/req_elapsed")
    s.add_argument("--dump-streams", action="store_true")
    s.set_defaults(func=cmd_speed)

    q = sub.add_parser("capture-q",
                       help="ON-mode full-stream spec verify dist q (top-K log_softmax)")
    common_capture(q)
    q.add_argument("--temperature", type=float, default=0.6)
    q.add_argument("--top-p", type=float, default=0.95)
    q.add_argument("--top-k", type=int, default=20)
    q.add_argument("--tokenizer", default=CANONICAL_TOKENIZER,
                   help="served-model tokenizer for the GAP-1 decoded-string->id re-key")
    q.add_argument("--no-rekey", action="store_true",
                   help="skip the id re-key (string-keyed q only; temp06-drift will be vacuous)")
    q.set_defaults(func=cmd_capture_q)

    d = sub.add_parser("temp06-drift",
                       help="CPU reduce: TV(softmax(q/0.6),softmax(p/0.6)) per position")
    d.add_argument("--q", required=True, help="capture-q artifact (instrument=ON)")
    d.add_argument("--p", required=True,
                   help="recurrent oracle rescore (RECURRENT_PATH_ENGAGED)")
    d.add_argument("--temp", type=float, default=0.6)
    d.add_argument("--per-position-floor", type=float, default=None,
                   help="native-vs-native per-position TV floor; over-floor count")
    d.add_argument("--tokenizer", default=CANONICAL_TOKENIZER,
                   help="served-model tokenizer for on-the-fly re-key of OLD q artifacts")
    d.add_argument("--no-rekey", action="store_true",
                   help="forbid on-the-fly re-key (require q already id-keyed)")
    d.add_argument("--dump-positions", action="store_true")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_temp06_drift)

    b = sub.add_parser("bag-tv",
                       help="multi-seed realized bag-TV with p95 native floor")
    b.add_argument("--native", nargs="+", required=True,
                   help="native temp-0.6 capture-q/speed artifacts (N seeds)")
    b.add_argument("--cat", nargs="+", default=None,
                   help="cat9 temp-0.6 artifacts (same seeds)")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_bag_tv)

    pa = sub.add_parser("paired-accept",
                        help="GAP-2: fork-immune paired accept on a COMMON reference trajectory")
    pa.add_argument("--reference", required=True,
                    help="deployment-correct ground-truth stream (no-spec recurrent "
                         "oracle greedy rescore/smoke, or any served-stream artifact)")
    pa.add_argument("--arm-q", nargs="+", required=True,
                    help="capture-q artifacts (instrument=ON) for the arms to score")
    pa.add_argument("--default-depth", type=int, default=5,
                    help="draft depth fallback when the arm name/tree does not encode it")
    pa.add_argument("--allow-served-fallback", action="store_true",
                    help="score the arm's SERVED stream as the verify proxy when the "
                         "q is not id-keyed (labelled served_fallback_PROXY; the "
                         "real structural edge needs id-keyed q)")
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=cmd_paired_accept)

    dr = sub.add_parser("diag-residue", help="OFF-vs-ON s/fwd instrument tax")
    dr.add_argument("--off", required=True, help="OFF speed record")
    dr.add_argument("--on", required=True, help="ON capture (carries raw_counter_delta)")
    dr.add_argument("--out", required=True)
    dr.set_defaults(func=cmd_diag_residue)

    rc = sub.add_parser("reconcile", help="measured OFF speed vs banked historic")
    rc.add_argument("--speed", nargs="+", required=True,
                    help="OFF speed records to reconcile")
    rc.add_argument("--accept-tol", type=float, default=0.05)
    rc.add_argument("--out", required=True)
    rc.set_defaults(func=cmd_reconcile)

    # ----- DEPLOYMENT REGIME (the CANONICAL path) ----------------------------
    ds = sub.add_parser(
        "deploy-speed",
        help="CANONICAL: deployment-regime s/fwd + accept on the real codex "
             "trajectory (per-task /metrics brackets from run_swe_bench_q36_a)")
    ds.add_argument("--arm", required=True, help="cat9 / native_e5 / ... (label)")
    ds.add_argument("--out-root", required=True,
                    help="a run_swe_bench_q36_a out-root (e.g. "
                         "output/fr13_bigdenom_swe/cat9_a/swe_out)")
    ds.add_argument("--expected-tok-per-draft", type=float, required=True,
                    help="class-9 engagement gate: 5 for native E5, len(TREE) for a tree")
    ds.add_argument("--batch-size", type=int, default=1,
                    help="co-residency regime the arm was booted in (1 or 4)")
    ds.add_argument("--basis", default="decode_seconds",
                    help="speed basis; banned: tps/accept/wall/req_elapsed")
    ds.add_argument("--out", required=True)
    ds.set_defaults(func=cmd_deploy_speed)

    dl = sub.add_parser(
        "deploy-lossless",
        help="CANONICAL: deployment-regime lossless verdict (within-floor) from "
             "the big-denom rescore consolidation (no-spec recurrent oracle flips)")
    dl.add_argument("--consolidated", required=True,
                    help="fr13.bigdenom_rescore_consolidate.v1 artifact "
                         "(output/fr13_bigdenom_rescore/consolidated.json)")
    dl.add_argument("--out", required=True)
    dl.set_defaults(func=cmd_deploy_lossless)

    dt = sub.add_parser(
        "deploy-temp06-drift",
        help="CANONICAL: deployment-regime temp-0.6 TV(q,p) drift on the real "
             "codex served stream (q = spec verify capture-q-deploy, p = no-spec "
             "recurrent bigdenom rescore; same stream, step/id-aligned)")
    dt.add_argument("--q", required=True,
                    help="fr13.recurrent_decode_oracle.capture_q_deploy.v1 (the "
                         "spec verify q forced onto the deployment served stream)")
    dt.add_argument("--p", required=True,
                    help="fr13.recurrent_decode_oracle.rescore.v1 (the no-spec "
                         "recurrent deployment oracle on the SAME served stream)")
    dt.add_argument("--temp", type=float, default=0.6)
    dt.add_argument("--per-position-floor", type=float, default=None,
                    help="per-position TV floor; over_floor_count locates the drift")
    dt.add_argument("--native-floor-p95", type=float, default=None,
                    help="DEPTH-MATCHED native deploy-temp06-drift p95 (the floor "
                         "this arm is judged against; depth-5->E5, depth-3->E3)")
    dt.add_argument("--dump-positions", action="store_true")
    dt.add_argument("--out", required=True)
    dt.set_defaults(func=cmd_deploy_temp06_drift)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
