"""FR14 lane 3 -- measure the cfwd single-row-softmax claim instead of asserting it.

WHY THIS EXISTS
---------------
``scripts/fr13_device_multidraft_kernel.py`` walks the tree-accept (TAW) rejection
sampler one level at a time. At every level it calls ``_device_softmax_row`` on a
SINGLE full-vocabulary logits row::

    p_row = _device_softmax_row(target_logits[st + children[0]])   # torch.softmax([V])
    p     = p_row / p_row.sum()                                    # per-row renorm

The deployed hydra27_fixed32 B1 route
(``taw.route = fixed32_pytorch_exact_float_triton_integer_commit``) issues
``full_vocab_softmax_calls = 24`` of these per step (12 walk levels x {self, target}),
each over ``vocab_size = 248320`` fp32.

nsys (``output/fr13_fixed32_b1_nsys_20260818T001018Z``) measures every one of those
launches at grid ``(1, 1, 1)``: ATen's ``cunn_SoftMaxForward`` assigns ONE BLOCK PER
ROW, so a one-row softmax occupies exactly ONE SM. Measured 26 instances/step
(24 TAW + 2 engine) totalling 2.244 ms of GPU-BUSY time, i.e. ~86 us each to move
~1 MB -- about 34 GB/s, roughly 8x off the ~273 GB/s the rest of cfwd achieves.
It is the one op class in cfwd that is NOT bandwidth-bound; it is occupancy-bound.

Batching the rows into one ``[A, V]`` softmax makes the grid ``(A, 1, 1)``: the same
per-row block algorithm, A blocks in parallel, ~one row's latency for all of them.

THE ONLY THING BLOCKING THAT IS AN UNMEASURED CLAIM. The byte-identity contract in
``fr13_device_multidraft_kernel.py`` (``_fr13_dm_depthsync_walk``) says:

    "single-row softmax (never batched: a stacked [A,V] softmax could shift p by 1 ULP)"

"could" is a hypothesis, not a measurement, and the whole campaign's doctrine is that
a hypothesis in a docstring does not get to veto a lever. This probe settles it on the
deployed shape, dtype, GPU and torch build.

WHAT IS MEASURED
----------------
G1  BITWISE  -- is ``torch.softmax(X[rows], -1)[i]`` bit-for-bit equal to
    ``torch.softmax(X[rows[i]], -1)`` at V=248320 fp32, over many trials, many
    batch widths, and adversarial logit distributions? Compared as int32 bit
    patterns, not as floats: "allclose" is not the question being asked.
G2  SUM-SHAPE -- is the FOLLOW-ON renormalisation ``p.sum()`` shape-sensitive?
    This is the op that actually carries a reduction-order dependence, and it is
    why ``_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2`` exists. If G1 passes and G2
    fails, the correct lever is softmax-only: batch the softmax, leave every
    normalisation exactly where and how the incumbent does it.
T1  TIMING   -- 24 single-row softmax calls vs 1 batched call, CUDA-event timed.
B1  SUB-LEVER B CEILING -- achievable bandwidth for a 48-layer GDN state copy
    (48 x 48 vheads x 128 x 128 fp32 = 151 MB), which is what a spine-state
    checkpoint commit would have to pay, priced against the measured 4.42 ms
    that the accepted-path replay costs today.

No arithmetic in the serving path is changed by running this. It allocates its own
tensors and reports numbers.

RUN
---
    docker run --rm --gpus all -v <repo>:/workspace -w /workspace \
        vllm/vllm-openai@sha256:3dbe092e... \
        python3 scripts/fr14_cfwd_softmax_batching_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch


SCHEMA = "fr14.cfwd_softmax_batching_probe.v1"

# Deployed hydra27_fixed32 B1 geometry, read out of the live work census
# (output/fr13_fixed32_b1_nsys_20260818T001018Z/.../fr13_fixed32_work_census.jsonl):
#   taw.vocab_size = 248320, taw.self_shape = taw.target_shape = [31, 248320] fp32,
#   taw.self_rows = taw.target_rows = 12, taw.loop_iterations = 12.
VOCAB = 248320
PHYSICAL_DRAFTS = 31
SOFTMAX_CALLS_PER_STEP = 24

# GDN committer geometry (fr10_gdn_tree_kernel / nsys grid (1,4,48)):
# 48 layers, 48 value heads, dv=128, dk=128, fp32 state bank.
GDN_LAYERS = 48
GDN_VHEADS = 48
GDN_DV = 128
GDN_DK = 128


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int]:
    """Bitwise comparison of two fp32 tensors. Returns (equal, differing count).

    Compared through an int32 view so that -0.0 vs +0.0 and any NaN payload
    difference is caught. ``torch.equal`` would call +0.0 == -0.0 equal; that is
    not the question.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    if a.dtype != torch.float32:
        raise TypeError(f"expected fp32, got {a.dtype}")
    diff = a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)
    return bool(not diff.any()), int(diff.sum().item())


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max |ULP| gap, so a FAIL reports its magnitude instead of just failing."""
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    bi = b.contiguous().view(torch.int32).to(torch.int64)
    return int((ai - bi).abs().max().item())


def _logit_generators(device, generator):
    """Input regimes. A lever that only survives N(0,1) has not been tested."""

    def normal(rows):
        return torch.randn(rows, VOCAB, device=device, dtype=torch.float32,
                           generator=generator)

    def wide(rows):
        # Large dynamic range -> the max-subtraction and the exp sum both matter.
        return torch.randn(rows, VOCAB, device=device, dtype=torch.float32,
                           generator=generator) * 40.0

    def peaked(rows):
        # One dominant logit: the sum is then a long accumulation of denormal-ish
        # terms onto one large term, the worst case for order sensitivity.
        x = torch.randn(rows, VOCAB, device=device, dtype=torch.float32,
                        generator=generator) * 2.0
        x[:, 0] += 60.0
        return x

    def ties(rows):
        # Massive tie structure: many exactly-equal exponentials to accumulate.
        x = torch.zeros(rows, VOCAB, device=device, dtype=torch.float32)
        x[:, ::2] = 1.0
        x[:, 0] = 8.0
        return x

    def bf16_cast(rows):
        # The walk casts to fp32 before the softmax; cover a bf16 source.
        x = torch.randn(rows, VOCAB, device=device, dtype=torch.bfloat16,
                        generator=generator)
        return x.to(torch.float32)

    return {
        "normal": normal,
        "wide": wide,
        "peaked": peaked,
        "ties": ties,
        "bf16_cast": bf16_cast,
    }


def gate_g1_bitwise(device, *, trials: int, widths: tuple[int, ...]) -> dict:
    """G1: batched [A,V] softmax vs single-row [V] softmax, bit for bit."""
    generator = torch.Generator(device=device).manual_seed(20260818)
    regimes = _logit_generators(device, generator)
    cases = []
    all_pass = True
    for regime_name, make in regimes.items():
        for width in widths:
            for trial in range(trials):
                logits = make(PHYSICAL_DRAFTS)
                # The walk indexes ONE row at a time out of the [31, V] block.
                rows = torch.randperm(PHYSICAL_DRAFTS, device=device,
                                      generator=generator)[:width]
                # CANDIDATE: one batched softmax over the gathered rows.
                batched = torch.softmax(logits[rows].to(torch.float32), dim=-1)
                # REFERENCE: exactly what _device_softmax_row does today.
                ok = True
                differing = 0
                ulp = 0
                for i in range(width):
                    ref = torch.softmax(
                        logits[int(rows[i])].to(torch.float32), dim=-1
                    )
                    equal, ndiff = _bit_equal(ref, batched[i])
                    if not equal:
                        ok = False
                        differing += max(ndiff, 0)
                        ulp = max(ulp, _max_ulp(ref, batched[i]))
                if not ok:
                    all_pass = False
                cases.append({
                    "regime": regime_name,
                    "width": int(width),
                    "trial": int(trial),
                    "bitwise_equal": bool(ok),
                    "differing_elements": int(differing),
                    "max_ulp": int(ulp),
                })
    return {
        "gate": "G1_batched_softmax_bitwise_equals_single_row",
        "verdict": "PASS" if all_pass else "FAIL",
        "vocab": VOCAB,
        "widths": [int(w) for w in widths],
        "trials_per_cell": trials,
        "cases_run": len(cases),
        "cases_failed": sum(1 for c in cases if not c["bitwise_equal"]),
        "failures": [c for c in cases if not c["bitwise_equal"]][:20],
    }


def gate_g2_sum_shape(device, *, trials: int, widths: tuple[int, ...]) -> dict:
    """G2: is the FOLLOW-ON renormalisation sum shape-sensitive?

    The incumbent computes ``p / p.sum()`` on a 1-D [V] tensor. A batched
    candidate that also batched the renormalisation would compute
    ``P / P.sum(-1, keepdim=True)`` on [A, V]. torch's reduction splits the work
    differently for those two shapes, so this is where a real ULP difference can
    live. Measuring it tells us whether the lever must stay softmax-only.
    """
    generator = torch.Generator(device=device).manual_seed(20260819)
    regimes = _logit_generators(device, generator)
    cases = []
    any_diff = False
    for regime_name, make in regimes.items():
        for width in widths:
            for trial in range(trials):
                probs = torch.softmax(make(PHYSICAL_DRAFTS), dim=-1)
                rows = torch.randperm(PHYSICAL_DRAFTS, device=device,
                                      generator=generator)[:width]
                block = probs[rows].contiguous()
                batched_sum = block.sum(-1)
                ok = True
                ulp = 0
                for i in range(width):
                    ref_sum = block[i].sum()
                    equal, _ = _bit_equal(ref_sum.reshape(1),
                                          batched_sum[i].reshape(1))
                    if not equal:
                        ok = False
                        ulp = max(ulp, _max_ulp(ref_sum.reshape(1),
                                                batched_sum[i].reshape(1)))
                if not ok:
                    any_diff = True
                cases.append({
                    "regime": regime_name,
                    "width": int(width),
                    "trial": int(trial),
                    "bitwise_equal": bool(ok),
                    "max_ulp": int(ulp),
                })
    return {
        "gate": "G2_batched_rowsum_vs_per_row_sum",
        # This gate is DIAGNOSTIC, not pass/fail: it explains why the lever is
        # scoped to the softmax only. "differs" is the expected, useful answer.
        "batched_rowsum_differs": bool(any_diff),
        "cases_run": len(cases),
        "cases_differing": sum(1 for c in cases if not c["bitwise_equal"]),
        "max_ulp_observed": max((c["max_ulp"] for c in cases), default=0),
    }


def gate_g3_composite_b1(device, *, trials: int) -> dict:
    """G3: the WHOLE per-level expression, at the deployed B=1 shapes.

    G1 compares softmax outputs in isolation. What the serving path actually
    computes at every walk level is (``fr13_device_multidraft_kernel.py``, the
    ``fixed32_pytorch_exact_float_triton_integer_commit`` walk)::

        REFERENCE (self_prob_cache is None):
            self_indices = starts + current.clamp(...)          # shape [B]
            self_prob = torch.softmax(X[self_indices].to(f32), dim=-1)   # [B, V]
            self_prob = self_prob / self_prob.sum(dim=-1, keepdim=True)

        CANDIDATE (self_prob_cache is not None):
            cache = torch.softmax(X[reachable].to(f32), dim=-1)  # [R, V], once
            self_prob = cache[self_indices]                      # [B, V]
            self_prob = self_prob / self_prob.sum(dim=-1, keepdim=True)

    At B=1 the reference softmax input is a ``[1, V]`` tensor -- ONE row, ONE
    block, one SM. Note the renormalisation is textually IDENTICAL in both
    branches and operates on the same ``[B, V]`` shape either way, so the
    shape-sensitive reduction G2 found is NOT crossed by this lever. This gate
    proves the composite, not a component, at exactly B=1.
    """
    generator = torch.Generator(device=device).manual_seed(20260821)
    regimes = _logit_generators(device, generator)
    reachable_rows = 12  # taw.self_rows == taw.target_rows == 12, live census
    cases = []
    all_pass = True
    for regime_name, make in regimes.items():
        for trial in range(trials):
            logits = make(PHYSICAL_DRAFTS)
            reachable = torch.randperm(PHYSICAL_DRAFTS, device=device,
                                       generator=generator)[:reachable_rows]
            cache = torch.softmax(logits[reachable].to(torch.float32), dim=-1)
            for slot in range(reachable_rows):
                node = reachable[slot].reshape(1)          # self_indices, [B=1]
                # REFERENCE: exactly the deployed expression.
                ref = torch.softmax(logits[node].to(torch.float32), dim=-1)
                ref = ref / ref.sum(dim=-1, keepdim=True)
                # CANDIDATE: cache lookup, same renormalisation.
                cand = cache[torch.tensor([slot], device=device)]
                cand = cand / cand.sum(dim=-1, keepdim=True)
                equal, ndiff = _bit_equal(ref, cand)
                if not equal:
                    all_pass = False
                    cases.append({
                        "regime": regime_name, "trial": trial, "slot": slot,
                        "bitwise_equal": False, "differing_elements": ndiff,
                        "max_ulp": _max_ulp(ref, cand),
                    })
    return {
        "gate": "G3_composite_per_level_expression_at_B1",
        "verdict": "PASS" if all_pass else "FAIL",
        "batch_size": 1,
        "reachable_rows": reachable_rows,
        "comparisons": len(regimes) * trials * reachable_rows,
        "failures": cases[:20],
    }


def gate_g4_rank_dispatch(device, *, trials: int) -> dict:
    """G4: does softmax dispatch differently for [V] vs [1, V]?

    The reference path softmaxes a ``[1, V]`` tensor (it indexes with a length-1
    index tensor, which keeps the batch axis). Some of this probe's other gates
    build a 1-D ``[V]`` row. If those two disagreed, every comparison mixing them
    would be measuring the wrong thing. Checked rather than assumed.
    """
    generator = torch.Generator(device=device).manual_seed(20260822)
    regimes = _logit_generators(device, generator)
    mismatches = 0
    total = 0
    for _name, make in regimes.items():
        for _trial in range(trials):
            logits = make(PHYSICAL_DRAFTS)
            for row in (0, PHYSICAL_DRAFTS // 2, PHYSICAL_DRAFTS - 1):
                flat = torch.softmax(logits[row].to(torch.float32), dim=-1)
                kept = torch.softmax(
                    logits[row:row + 1].to(torch.float32), dim=-1
                )
                equal, _ = _bit_equal(flat, kept.reshape(-1))
                total += 1
                if not equal:
                    mismatches += 1
    return {
        "gate": "G4_softmax_rank1_vs_rank2_dispatch",
        "verdict": "PASS" if mismatches == 0 else "FAIL",
        "comparisons": total,
        "mismatches": mismatches,
    }


def gate_g5_cumsum_determinism(device, *, trials: int) -> dict:
    """G5: is the B=1 reference walk's own cumsum run-to-run deterministic?

    ``fr13_device_multidraft_kernel.py`` justifies
    ``_FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2`` with this claim::

        "At B=1 the reference operator itself is not reproducible (cumsum over a
         single [1, V] row is run-to-run non-deterministic on this device), so no
         byte-exact batched candidate can exist there and the walk stays
         unbatched."

    That is a strong statement about the DEPLOYED path -- B=1 is what hydra27
    serves -- so it is worth a measurement rather than a citation. Whichever way
    it lands it scopes the softmax lever's gate:

      * deterministic  -> an end-to-end byte gate of the B=1 walk is possible;
      * NON-deterministic -> the incumbent is already non-reproducible at B=1,
        the softmax lever cannot be judged by an end-to-end byte comparison, and
        the correct gate is the stage-wise one (G3) plus the observation that the
        lever does not touch the cumsum at all.

    Runs the same cumsum repeatedly on identical input and bit-compares.
    """
    generator = torch.Generator(device=device).manual_seed(20260823)
    results = {}
    for width in (1, 2, 4, 12, 31):
        probs = torch.softmax(
            torch.randn(width, VOCAB, device=device, dtype=torch.float32,
                        generator=generator),
            dim=-1,
        )
        probs = probs / probs.sum(dim=-1, keepdim=True)
        first = torch.cumsum(probs, dim=-1)
        mismatch = 0
        max_ulp = 0
        for _ in range(trials * 5):
            again = torch.cumsum(probs, dim=-1)
            equal, _ = _bit_equal(first, again)
            if not equal:
                mismatch += 1
                max_ulp = max(max_ulp, _max_ulp(first, again))
        results[f"width_{width}"] = {
            "repeats": trials * 5,
            "nondeterministic_repeats": mismatch,
            "max_ulp": max_ulp,
        }
    b1 = results["width_1"]
    return {
        "gate": "G5_cumsum_run_to_run_determinism",
        "claim_under_test": (
            "cumsum over a single [1, V] row is run-to-run non-deterministic on "
            "this device (fr13_device_multidraft_kernel.py, the stated reason "
            "for _FR13_FIXED32_TAW_PINNED_MIN_BATCH = 2)"
        ),
        "b1_nondeterministic": bool(b1["nondeterministic_repeats"] > 0),
        "claim_supported": bool(b1["nondeterministic_repeats"] > 0),
        "by_width": results,
    }


def _time_cuda(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Median-ish ms per iteration, CUDA-event timed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def bench_t1_softmax(device) -> dict:
    """T1: 24 single-row softmax calls vs 1 batched call at the deployed shape."""
    generator = torch.Generator(device=device).manual_seed(20260820)
    logits = torch.randn(PHYSICAL_DRAFTS, VOCAB, device=device,
                         dtype=torch.float32, generator=generator)
    rows = torch.arange(SOFTMAX_CALLS_PER_STEP, device=device) % PHYSICAL_DRAFTS

    def incumbent():
        for i in range(SOFTMAX_CALLS_PER_STEP):
            torch.softmax(logits[int(rows[i])].to(torch.float32), dim=-1)

    def batched_all():
        torch.softmax(logits.to(torch.float32), dim=-1)

    def batched_reachable():
        torch.softmax(logits[rows].to(torch.float32), dim=-1)

    def single_row():
        torch.softmax(logits[0].to(torch.float32), dim=-1)

    one = _time_cuda(single_row)
    inc = _time_cuda(incumbent)
    b_all = _time_cuda(batched_all)
    b_reach = _time_cuda(batched_reachable)
    row_bytes = VOCAB * 4
    return {
        "bench": "T1_softmax_batching",
        "vocab": VOCAB,
        "softmax_calls_per_step": SOFTMAX_CALLS_PER_STEP,
        "single_row_ms": one,
        "single_row_effective_gbps": (2 * row_bytes / 1e9) / (one / 1e3),
        "incumbent_24_single_rows_ms": inc,
        "batched_all31_ms": b_all,
        "batched_24_gathered_ms": b_reach,
        "saving_vs_batched_all31_ms": inc - b_all,
        "saving_vs_batched_24_ms": inc - b_reach,
    }


def bench_b1_state_copy(device) -> dict:
    """B1: what a spine-state checkpoint commit would cost, at best.

    Sub-lever B replaces the accepted-path GDN replay with a checkpointed state.
    Whatever else it does, the commit must land 48 layers of recurrent state into
    the state bank. This prices that floor as a pure device-to-device copy so the
    lever can be judged against the 4.42 ms/step the replay costs today (nsys,
    48 launches of fused_sigmoid_gating_delta_rule_update_kernel).
    """
    per_layer_elems = GDN_VHEADS * GDN_DV * GDN_DK
    src = torch.randn(GDN_LAYERS, per_layer_elems, device=device,
                      dtype=torch.float32)
    dst = torch.empty_like(src)
    total_bytes = src.numel() * 4

    def copy_all():
        dst.copy_(src)

    def copy_per_layer():
        for layer in range(GDN_LAYERS):
            dst[layer].copy_(src[layer])

    single = _time_cuda(copy_all)
    per_layer = _time_cuda(copy_per_layer)
    return {
        "bench": "B1_gdn_state_copy_floor",
        "layers": GDN_LAYERS,
        "state_bytes_total": int(total_bytes),
        "state_mib_total": total_bytes / 2**20,
        "one_copy_ms": single,
        "one_copy_gbps": (2 * total_bytes / 1e9) / (single / 1e3),
        "per_layer_48_copies_ms": per_layer,
        "measured_replay_ms_per_step_nsys": 4.42,
        "note": (
            "one_copy_ms is the FLOOR for any checkpoint-commit that must move "
            "the state; the checkpoint WRITE during sfwd costs at least the same "
            "again per checkpointed spine node."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3,
                        help="trials per (regime, width) cell for G1/G2")
    parser.add_argument("--out", default="", help="write the JSON report here")
    parser.add_argument("--skip-bench", action="store_true",
                        help="gates only, no timing (for a shared GPU)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FR14 cfwd softmax probe requires CUDA", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.cuda.init()

    widths = (2, 3, 4, 8, 12, 16, 24, 31)
    report = {
        "schema": SCHEMA,
        "analysis_only": True,
        "acceptance_valid": False,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "platform": platform.platform(),
        },
        "deployed_geometry": {
            "vocab_size": VOCAB,
            "physical_drafts": PHYSICAL_DRAFTS,
            "walk_levels": 12,
            "full_vocab_softmax_calls_per_step": SOFTMAX_CALLS_PER_STEP,
            "source": (
                "output/fr13_fixed32_b1_nsys_20260818T001018Z/"
                "hydra27_fixed32_b1_nsys_f32_20260818T001018Z/logs/"
                "fr13_fixed32_work_census.jsonl (taw.*)"
            ),
        },
    }

    t0 = time.time()
    report["G1"] = gate_g1_bitwise(device, trials=args.trials, widths=widths)
    report["G2"] = gate_g2_sum_shape(device, trials=args.trials, widths=widths)
    report["G3"] = gate_g3_composite_b1(device, trials=args.trials)
    report["G4"] = gate_g4_rank_dispatch(device, trials=args.trials)
    report["G5"] = gate_g5_cumsum_determinism(device, trials=args.trials)
    if not args.skip_bench:
        report["T1"] = bench_t1_softmax(device)
        report["B1"] = bench_b1_state_copy(device)
    verdicts = {name: report[name]["verdict"] for name in ("G1", "G3", "G4")}
    report["gate_verdicts"] = verdicts
    report["overall"] = "PASS" if set(verdicts.values()) == {"PASS"} else "FAIL"
    report["elapsed_s"] = round(time.time() - t0, 2)

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
