#!/usr/bin/env python3
"""FR14 fused draft top-k: offline byte-exact selection gate + stage microbench.

TWO JOBS, ONE HARNESS -- the campaign's standard offline shape.

  1. THE GATE (byte-exact selection).  The deployed K0 drafter head runs, five
     times per decode step, on the [rows, 248320] bf16 logits row that
     `compute_logits` just wrote:

         draft_token_ids      = logits.argmax(dim=-1)
         _fr10_wide_topk[pos] = torch.topk(logits, 3, dim=-1).indices

     The candidate replaces BOTH with one launch of
     `torch.ops.fr14_fused_draft_topk.select_out`.  This gate asserts the
     candidate emits the IDENTICAL int64 token ids in the IDENTICAL order,
     compared as RAW BYTES (`.view(torch.uint8)`) so no NaN/sign laundering is
     possible, over many seeds at the REAL geometry (V = 248320, k = 3,
     rows 1..4, bf16) and -- the part that actually decides the lever's tier --
     over ADVERSARIAL TIE cases that the random draws cannot be trusted to
     produce: exact ties at the top value, planted at adjacent indices, at
     maximal index spread, at index 0, at index V-1, and in numbers that
     straddle k.

     PRE-REGISTERED VERDICT RULE (written before the data exists):
       * 0 raw-byte mismatches over every case, every seed, every rows, every
         `blocks_per_row`  -> Tier-A parity, the lever may be integrated.
       * ANY mismatch on a NON-tie case                 -> the kernel is wrong.
       * ANY mismatch confined to TIE cases             -> tie-break parity is
         NOT established; the honest report is STOP, not "approximately equal".
       * torch itself disagreeing with torch across repeats on a tie case
         -> the deployed selection is not a function of the logits, exact
         parity is undefinable, and the honest report is STOP.

     Anti-vacuity: a POWERED NEGATIVE CONTROL runs in every configuration -- a
     deliberately corrupted candidate that MUST mismatch.  If it does not, the
     comparison is not looking at anything and the run is void.

  1b. THE CUDA-GRAPH REPLAY GATE.  The deployed drafter captures its four
     post-root head reads as ONE graph, so the kernel runs from a replay with
     its scratch address baked in and its atomic ticket reused.  Four captured
     selects are replayed with fresh logits every time and every replay is
     compared against the eager ATen reference, then the ticket is checked back
     at zero.  This is the integration risk the case sweep cannot see.

  2. THE MICROBENCH (old vs new, real geometry).  CUDA-event timing of the
     deployed two-ATen-op selection versus the single fused launch, plus the
     pure write cost of materialising the logits row, so the "materialisation"
     surface can be priced in BYTES and in LATENCY separately instead of being
     argued.  Reported per head read and scaled by the 5 head reads/step that
     the fixed32 drafter's own capture-count invariant pins
     (`_fr13_dfwd_top3_capture_calls == 4` plus >= 1 root call).

  This is a SYNTHETIC-TENSOR kernel benchmark.  It is not acceptance-valid, it
  is not a step-envelope measurement, and it emits neither `step_wall_ms` nor
  `s_per_fwd_gpu`.  It carries no TPS, floor or acceptance claim.

Run (GPU must be free; `docker ps` empty):

  docker run --rm --gpus all \
    -v /home/mark/shared/lumoFlyWheel-nvfp4-port-20260816:/workspace -w /workspace \
    -e PYTHONPATH=/workspace/src --entrypoint python3 \
    vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
    results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe.py \
      --so /workspace/output/<build>/fr14_dfwd_full_topk_sm121a.abi3.so \
      --json results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe_result.json

Exit 0 = PASS, 3 = gate FAIL (mismatch or void control), 1 = crash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Real deployed geometry.  Every constant here is pinned elsewhere in the tree
# and is echoed into the artifact so a reviewer can see what was actually run.
# ---------------------------------------------------------------------------
VOCAB = 248_320          # config text_config.vocab_size; fr13_fixed32_work_census.TAW_VOCAB_SIZE
TOPK = 3                 # fr13_fixed32_topology.SAMPLER_MAX_FANOUT
HEAD_READS_PER_STEP = 5  # 1 root + 4 graph-captured loop passes
ROWS_SERVED = 1          # the B1 arm; the gate additionally sweeps 2..4
ROWS_SWEEP = (1, 2, 3, 4)
BLOCKS_SWEEP = (1, 8, 32, 64, 121)
BLOCKS_DEFAULT = 64
LOGITS_DTYPE = torch.bfloat16   # apply_nvfp4_linear returns x.dtype (bf16)

BENCH_REPS = 200
BENCH_WARMUP = 25


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw(tensor: torch.Tensor) -> bytes:
    """Bytes, not values -- NaN and -0.0 cannot launder a mismatch past this."""
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def sha16(tensor: torch.Tensor) -> str:
    return hashlib.sha256(raw(tensor)).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def make_case(name: str, rows: int, seed: int, device: torch.device) -> torch.Tensor:
    """Build one [rows, VOCAB] bf16 logits block."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def base(scale: float) -> torch.Tensor:
        return (
            torch.randn(rows, VOCAB, generator=generator, dtype=torch.float32) * scale
        ).to(LOGITS_DTYPE).to(device)

    if name == "random_realistic":
        # Draft logits at this checkpoint sit in roughly +-30; bf16 has 8 bits
        # of mantissa, so the top of a 248k draw already ties naturally.
        return base(8.0)
    if name == "random_wide":
        return base(64.0)
    if name == "random_narrow":
        # Compresses everything into few bf16 codes -> ties everywhere.
        return base(0.02)
    if name == "coarse_quantised":
        # Only ~64 distinct bf16 codes in play: maximal natural tie density.
        values = torch.randint(0, 64, (rows, VOCAB), generator=generator)
        return (values.to(torch.float32) * 0.5).to(LOGITS_DTYPE).to(device)
    if name == "all_equal":
        return torch.full((rows, VOCAB), 1.5, dtype=LOGITS_DTYPE, device=device)
    if name == "all_neg_inf":
        return torch.full(
            (rows, VOCAB), float("-inf"), dtype=LOGITS_DTYPE, device=device
        )
    if name == "signed_zero":
        # +-0.0 compare EQUAL as floats but differ in bits; a radix select and
        # a value comparison can disagree here.  Reported, and gated.
        logits = torch.full((rows, VOCAB), -3.0, dtype=LOGITS_DTYPE, device=device)
        for row in range(rows):
            for offset, value in enumerate((0.0, -0.0, 0.0, -0.0, -0.0, 0.0)):
                logits[row, 4096 + offset * 37 + row] = value
        return logits
    if name.startswith("plateau_"):
        # plateau_<n>_<stride>_<offset>: n exactly-tied maxima on a stride, plus
        # a distinct runner-up group so k=3 straddles two value groups.
        _, n_s, stride_s, offset_s = name.split("_")
        n, stride, offset = int(n_s), int(stride_s), int(offset_s)
        logits = torch.full((rows, VOCAB), -20.0, dtype=LOGITS_DTYPE, device=device)
        top = torch.tensor(9.0, dtype=LOGITS_DTYPE).item()
        second = torch.tensor(8.0, dtype=LOGITS_DTYPE).item()
        for row in range(rows):
            shift = row * 3
            plateau = sorted(
                {(offset + shift + i * stride) % VOCAB for i in range(n)}
            )
            runner = [(offset + shift + 500 + i * 13) % VOCAB for i in range(3)]
            index = torch.tensor(plateau, device=device, dtype=torch.long)
            logits[row].index_fill_(0, index, top)
            for r in runner:
                if r not in plateau:
                    logits[row, r] = second
        return logits
    if name == "nan_present":
        logits = base(8.0)
        for row in range(rows):
            logits[row, 12_345 + row] = float("nan")
        return logits
    raise ValueError(f"unknown case {name!r}")


def plateau_case_names() -> tuple[str, ...]:
    """Adversarial tie sweep: plateau size x stride x offset.

    The strides deliberately straddle every seam the kernel can split on --
    the 8-wide uint4 vector, the 256-thread CTA, and the CTA grid stride -- and
    the offsets place the plateau at index 0, mid-row, and the last index.
    """
    names = []
    for n in (2, 3, 4, 5, 8, 17, 64, 1000):
        for stride in (1, 2, 7, 256, 257, 1024, 4096, 30011):
            for offset in (0, 1, 255, 12_345, VOCAB - 1):
                names.append(f"plateau_{n}_{stride}_{offset}")
    return tuple(names)


TIE_CASES = plateau_case_names() + (
    "all_equal",
    "all_neg_inf",
    "coarse_quantised",
    "random_narrow",
    "signed_zero",
)
PLAIN_CASES = ("random_realistic", "random_wide")
DIAGNOSTIC_CASES = ("nan_present",)


# ---------------------------------------------------------------------------
# Reference (what ships today) and candidate
# ---------------------------------------------------------------------------
def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """EXACTLY the two ATen calls the deployed K0 drafter makes per head read."""
    spine = logits.argmax(dim=-1)
    wide = torch.topk(logits, TOPK, dim=-1).indices
    return spine, wide


def candidate(
    logits: torch.Tensor, scratch: torch.Tensor, blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = logits.shape[0]
    spine = torch.empty(rows, dtype=torch.int64, device=logits.device)
    wide = torch.empty((rows, TOPK), dtype=torch.int64, device=logits.device)
    torch.ops.fr14_fused_draft_topk.select_out(spine, wide, logits, scratch, blocks)
    return spine, wide


def scratch_for(rows: int, blocks: int, device: torch.device) -> torch.Tensor:
    numel = int(torch.ops.fr14_fused_draft_topk.scratch_numel(rows, blocks))
    return torch.zeros(numel, dtype=torch.int64, device=device)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def run_gate(device: torch.device, seeds: int) -> dict:
    cases: list[dict] = []
    mismatches = 0
    tie_mismatches = 0
    plain_mismatches = 0
    torch_self_disagreements = 0
    evaluated = 0
    controls_all_fire = True
    argmax_ne_topk0 = 0
    argmax_ne_topk0_plain = 0
    plain_evaluated = 0

    all_cases = list(PLAIN_CASES) + list(TIE_CASES)
    for rows in ROWS_SWEEP:
        scratches = {b: scratch_for(rows, b, device) for b in BLOCKS_SWEEP}
        for case_name in all_cases:
            deterministic_case = case_name not in PLAIN_CASES
            n_seeds = 1 if deterministic_case and "random" not in case_name else seeds
            for seed in range(n_seeds):
                logits = make_case(case_name, rows, 1000 + 7919 * seed, device)

                ref_spine, ref_wide = reference(logits)
                # Is the DEPLOYED selection even a function of the logits?
                # Repeat the reference; any drift means exact parity is
                # undefinable and the pre-registered rule says STOP.
                for _ in range(3):
                    again_spine, again_wide = reference(logits)
                    if raw(again_spine) != raw(ref_spine) or raw(again_wide) != raw(
                        ref_wide
                    ):
                        torch_self_disagreements += 1
                        break

                per_blocks = {}
                case_mismatch = 0
                for blocks in BLOCKS_SWEEP:
                    cand_spine, cand_wide = candidate(logits, scratches[blocks], blocks)
                    spine_ok = raw(cand_spine) == raw(ref_spine)
                    wide_ok = raw(cand_wide) == raw(ref_wide)
                    if not (spine_ok and wide_ok):
                        case_mismatch += 1
                    per_blocks[str(blocks)] = {
                        "spine_byte_equal": spine_ok,
                        "wide_byte_equal": wide_ok,
                        "candidate_wide_sha16": sha16(cand_wide),
                    }

                # POWERED NEGATIVE CONTROL: corrupt one index and require the
                # comparison to catch it.  Without this the gate could be
                # comparing nothing at all.
                control = ref_wide.clone()
                control[0, TOPK - 1] = (int(control[0, TOPK - 1].item()) + 1) % VOCAB
                control_fires = raw(control) != raw(ref_wide)

                mismatches += case_mismatch
                if case_mismatch:
                    if case_name in TIE_CASES:
                        tie_mismatches += case_mismatch
                    else:
                        plain_mismatches += case_mismatch

                argmax_matches_rank0 = bool(torch.equal(ref_spine, ref_wide[:, 0]))
                if not argmax_matches_rank0:
                    argmax_ne_topk0 += 1
                    if case_name in PLAIN_CASES:
                        argmax_ne_topk0_plain += 1
                if case_name in PLAIN_CASES:
                    plain_evaluated += 1

                record = {
                    "case": case_name,
                    "rows": rows,
                    "seed": seed,
                    "is_tie_case": case_name in TIE_CASES,
                    "reference_wide_sha16": sha16(ref_wide),
                    "argmax_equals_topk_rank0": argmax_matches_rank0,
                    "mismatch_configs": case_mismatch,
                    "negative_control_fires": control_fires,
                }
                controls_all_fire = controls_all_fire and control_fires
                evaluated += 1
                if case_mismatch or case_name in PLAIN_CASES:
                    record["reference_spine_row0"] = int(ref_spine[0].item())
                    record["reference_wide_row0"] = [
                        int(v) for v in ref_wide[0].tolist()
                    ]
                    record["blocks"] = per_blocks
                    cases.append(record)
                elif len(cases) < 40:
                    # keep a readable sample of the passing bulk
                    cases.append(record)

    # Diagnostics only -- NaN in a draft logit means the model is broken; the
    # result is reported, never gated on.
    diagnostics = []
    for case_name in DIAGNOSTIC_CASES:
        logits = make_case(case_name, ROWS_SERVED, 4242, device)
        scratch = scratch_for(ROWS_SERVED, BLOCKS_DEFAULT, device)
        ref_spine, ref_wide = reference(logits)
        cand_spine, cand_wide = candidate(logits, scratch, BLOCKS_DEFAULT)
        diagnostics.append(
            {
                "case": case_name,
                "spine_byte_equal": raw(cand_spine) == raw(ref_spine),
                "wide_byte_equal": raw(cand_wide) == raw(ref_wide),
                "reference_wide_row0": [int(v) for v in ref_wide[0].tolist()],
                "candidate_wide_row0": [int(v) for v in cand_wide[0].tolist()],
                "gated": False,
            }
        )

    return {
        "cases_recorded": cases,
        "cases_recorded_note": (
            "every mismatching case and every plain (non-tie) case is recorded "
            "in full; passing tie cases are summarised by the counters"
        ),
        "diagnostics": diagnostics,
        "cases_evaluated": evaluated,
        "total_configs": evaluated * len(BLOCKS_SWEEP),
        "mismatch_total": mismatches,
        "mismatch_on_tie_cases": tie_mismatches,
        "mismatch_on_plain_cases": plain_mismatches,
        "torch_self_disagreements": torch_self_disagreements,
        "negative_control_all_fire": controls_all_fire,
        "deployed_argmax_ne_topk_rank0_cases": argmax_ne_topk0,
        "deployed_argmax_ne_topk_rank0_plain_cases": argmax_ne_topk0_plain,
        "plain_cases_evaluated": plain_evaluated,
        "gate_pass": (
            mismatches == 0
            and torch_self_disagreements == 0
            and controls_all_fire
            and evaluated > 0
        ),
    }


# ---------------------------------------------------------------------------
# CUDA-graph replay gate
# ---------------------------------------------------------------------------
def run_graph_gate(device: torch.device, replays: int = 24) -> dict:
    """Four captured selects, replayed many times, must stay byte-exact.

    This is the integration risk the case sweep cannot see.  The deployed
    drafter captures its four post-root head reads as ONE CUDA graph, so the
    kernel runs from a replay with its scratch buffer's address baked in.  The
    multi-CTA path uses an atomic ticket in that scratch to elect a final
    merging block, and the elected block resets the ticket to zero on its way
    out -- if that self-clean were wrong, replay 1 would pass and replay 2 would
    hang or produce a stale answer.  So: capture, then replay with fresh logits
    every time, and compare every replay against the eager ATen reference.
    """
    rows = ROWS_SERVED
    blocks = BLOCKS_DEFAULT
    static_logits = torch.zeros(rows, VOCAB, dtype=LOGITS_DTYPE, device=device)
    static_spine = torch.zeros(4, rows, dtype=torch.int64, device=device)
    static_wide = torch.zeros(4, rows, TOPK, dtype=torch.int64, device=device)
    scratch = scratch_for(rows, blocks, device)

    def four_selects():
        for level in range(4):
            torch.ops.fr14_fused_draft_topk.select_out(
                static_spine[level], static_wide[level], static_logits, scratch, blocks
            )

    # warm up on a side stream, exactly as the drafter graph does
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        four_selects()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        four_selects()

    mismatches = 0
    per_replay = []
    for replay in range(replays):
        source = make_case(
            "coarse_quantised" if replay % 2 else "random_realistic",
            rows,
            77_000 + replay,
            device,
        )
        static_logits.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        ref_spine, ref_wide = reference(static_logits)
        ok = True
        for level in range(4):
            if raw(static_spine[level]) != raw(ref_spine) or raw(
                static_wide[level]
            ) != raw(ref_wide):
                ok = False
        if not ok:
            mismatches += 1
        per_replay.append({"replay": replay, "byte_equal_all_four_levels": ok})

    # the ticket must be back at zero for the next launch
    ticket_tail = scratch[rows * blocks * TOPK :].tolist()
    return {
        "replays": replays,
        "levels_per_replay": 4,
        "blocks_per_row": blocks,
        "mismatching_replays": mismatches,
        "scratch_ticket_tail_after_run": ticket_tail,
        "ticket_self_cleaned": all(int(v) == 0 for v in ticket_tail),
        "per_replay": per_replay,
        "graph_gate_pass": mismatches == 0
        and all(int(v) == 0 for v in ticket_tail),
    }


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def time_us(fn, reps: int = BENCH_REPS, warmup: int = BENCH_WARMUP) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return {
        "us_min": round(samples[0], 3),
        "us_p05": round(samples[int(0.05 * len(samples))], 3),
        "us_p50": round(statistics.median(samples), 3),
        "us_mean": round(statistics.fmean(samples), 3),
        "reps": reps,
    }


def run_bench(device: torch.device) -> dict:
    out: dict = {}
    for rows in (1, 4):
        logits = make_case("random_realistic", rows, 31337, device)
        sink_src = torch.randn(rows, VOCAB, dtype=torch.float32, device=device).to(
            LOGITS_DTYPE
        )
        sink_dst = torch.empty_like(logits)
        spine = torch.empty(rows, dtype=torch.int64, device=device)
        wide = torch.empty((rows, TOPK), dtype=torch.int64, device=device)

        key = f"rows{rows}"
        out[key] = {
            "old_argmax_only": time_us(lambda: logits.argmax(dim=-1)),
            "old_topk_only": time_us(lambda: torch.topk(logits, TOPK, dim=-1)),
            "old_selection_pair": time_us(lambda: reference(logits)),
            "logits_row_copy": time_us(lambda: sink_dst.copy_(sink_src)),
        }
        for blocks in BLOCKS_SWEEP:
            scratch = scratch_for(rows, blocks, device)

            def run(blocks=blocks, scratch=scratch):
                torch.ops.fr14_fused_draft_topk.select_out(
                    spine, wide, logits, scratch, blocks
                )

            out[key][f"new_fused_blocks{blocks}"] = time_us(run)

        best_key = min(
            (k for k in out[key] if k.startswith("new_fused_blocks")),
            key=lambda k: out[key][k]["us_p50"],
        )
        old_p50 = out[key]["old_selection_pair"]["us_p50"]
        new_p50 = out[key][best_key]["us_p50"]
        out[key]["best_blocks"] = int(best_key.replace("new_fused_blocks", ""))
        out[key]["saving_us_per_head_read_p50"] = round(old_p50 - new_p50, 3)
        out[key]["saving_ms_per_step_p50"] = round(
            (old_p50 - new_p50) * HEAD_READS_PER_STEP / 1000.0, 4
        )
        out[key]["old_ms_per_step_p50"] = round(
            old_p50 * HEAD_READS_PER_STEP / 1000.0, 4
        )
        out[key]["new_ms_per_step_p50"] = round(
            new_p50 * HEAD_READS_PER_STEP / 1000.0, 4
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", required=True, help="built fr14 fused top-k .so")
    parser.add_argument("--json", default=None)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device", file=sys.stderr)
        return 1
    torch.ops.load_library(args.so)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)

    report: dict = {
        "schema": "fr14.fused_draft_topk.offline_probe.v1",
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "analysis_only": True,
        "acceptance_valid": False,
        "step_envelope_claim": False,
        "performance_measurement": "kernel_microbenchmark_synthetic",
        "so": args.so,
        "so_sha256": sha256_file(args.so),
        "torch_version": torch.__version__,
        "device": {
            "name": properties.name,
            "sms": properties.multi_processor_count,
            "capability": f"{properties.major}.{properties.minor}",
            "l2_bytes": getattr(properties, "L2_cache_size", None),
        },
        "geometry": {
            "vocab": VOCAB,
            "topk": TOPK,
            "rows_served": ROWS_SERVED,
            "rows_swept": list(ROWS_SWEEP),
            "blocks_per_row_swept": list(BLOCKS_SWEEP),
            "head_reads_per_step": HEAD_READS_PER_STEP,
            "logits_dtype": str(LOGITS_DTYPE),
            "logits_row_bytes": VOCAB * 2,
            "logits_bytes_per_step": VOCAB * 2 * HEAD_READS_PER_STEP,
        },
        "reference_ops": [
            "logits.argmax(dim=-1)",
            "torch.topk(logits, 3, dim=-1).indices",
        ],
    }

    if not args.skip_gate:
        report["gate"] = run_gate(device, args.seeds)
        report["graph_gate"] = run_graph_gate(device)
    if not args.skip_bench:
        report["bench_us"] = run_bench(device)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="ascii") as handle:
            handle.write(text + "\n")
    if args.skip_gate:
        return 0
    passed = report["gate"]["gate_pass"] and report["graph_gate"]["graph_gate_pass"]
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
