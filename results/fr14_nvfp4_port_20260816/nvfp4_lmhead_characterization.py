#!/usr/bin/env python3
"""FR14 lane 5A: characterise the RadixArk NVFP4 ``lm_head`` against the BF16
reference head, in weight space and (given real hidden states) in logit space.

Two phases, both CPU-only:

``--phase weights``
    Dequantises the 4-tensor ModelOpt NVFP4 head out of the RadixArk shard and
    compares it, row by row, against the BF16 ``lm_head.weight`` shipped by the
    FP8-3.8 baseline checkpoint -- which is the SAME base model's *unquantised*
    head (proved here, not assumed: an FP8-per-channel head has at most 256
    distinct magnitudes per row, and this one has ~2^15).  Arm A's head
    (FP8-per-channel dequantised to BF16 by ``lmhead_surgery.py``) is measured
    against the same reference as a CALIBRATION CONTROL: the campaign already
    serves that head, so its error is the precedent any NVFP4 error is judged
    against.

``--phase logits --hidden <npz>``
    Given real pre-``lm_head`` hidden states captured from a live serve, forms
    the logits under each head and reports max-abs delta, argmax-flip rate and
    top-k overlap.  Chunked over rows so the 248,320-column logit matrix never
    materialises for more than one chunk at a time.

DEQUANT DIRECTION IS DERIVED, NOT ASSUMED.  ModelOpt stores ``weight_scale_2``
in reciprocal form here (1.27e-4), so the dequant is
``w = e2m1(nibble) * fp8(block_scale) * weight_scale_2``; the script asserts the
resulting amax lands within 2x of the reference head's amax, which the opposite
direction misses by ~8 orders of magnitude.

The on-disk block scale is the LINEAR [out, in/16] layout -- swizzling happens
in ``process_weights_after_loading``, inside the engine -- so no de-swizzle is
applied here.  ``radixark_dvk_dequant_check.py`` covers the swizzled path.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

RADIXARK = "/home/mark/shared/models/qwen3.8-27b-nvfp4-radixark"
REF_BF16 = "/home/mark/shared/models/qwen3.8-27b-fp8/outside.safetensors"
ARM_A = "/home/mark/shared/models/qwen3.8-27b-nvfp4/model.safetensors"

VOCAB = 248_320
HIDDEN = 5_120
BLOCK = 16

# vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:18-20
kE2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# e4m3 (float8_e4m3fn) -> float32 lookup, built by construction rather than by
# relying on a numpy/torch float8 dtype being present on the host.
def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) else 1.0
        e = (b >> 3) & 0x0F
        m = b & 0x07
        if e == 0:
            v = (m / 8.0) * (2.0 ** -6)
        elif e == 0x0F and m == 0x07:
            v = float("nan")  # e4m3fn: S.1111.111 is NaN, no infinities
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        t[b] = s * v
    return t


E4M3 = _e4m3_table()


def _st_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _st_read(path: str, name: str, np_dtype) -> np.ndarray:
    hdr, base = _st_header(path)
    v = hdr[name]
    a, b = v["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        buf = f.read(b - a)
    return np.frombuffer(buf, dtype=np_dtype).reshape(v["shape"])


def _find_shard(root: str, tensor: str) -> str:
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".safetensors"):
                continue
            p = os.path.join(dirpath, fn)
            hdr, _ = _st_header(p)
            if tensor in hdr:
                return p
    raise SystemExit(f"tensor {tensor!r} not found under {root}")


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 (exact: bf16 is f32's top 16 bits)."""
    out = np.zeros(raw.shape + (2,), dtype=np.uint16)
    out[..., 1] = raw
    return out.view(np.float32).reshape(raw.shape)


def read_bf16(path: str, name: str) -> np.ndarray:
    return _st_read(path, name, np.uint16)  # left as raw bits; caller converts


def break_fp4_bytes(packed: np.ndarray) -> np.ndarray:
    """Replica of vllm nvfp4_emulation_utils.break_fp4_bytes."""
    m, n = packed.shape
    flat = packed.reshape(-1)
    high = (flat & 0xF0) >> 4
    low = flat & 0x0F
    combined = np.stack((low, high), axis=1).reshape(-1)
    signs = (combined & 0x08).astype(bool)
    abs_vals = (combined & 0x07).astype(np.int64)
    values = kE2M1[abs_vals] * np.where(signs, -1.0, 1.0).astype(np.float32)
    return values.reshape(m, n * 2)


class NvFp4Head:
    """Lazy, row-chunked dequant of the on-disk NVFP4 head."""

    def __init__(self, root: str = RADIXARK):
        shard = _find_shard(root, "lm_head.weight")
        self.shard = shard
        hdr, base = _st_header(shard)
        self._hdr, self._base = hdr, base
        self.packed_shape = tuple(hdr["lm_head.weight"]["shape"])
        self.scale_shape = tuple(hdr["lm_head.weight_scale"]["shape"])
        self.weight_scale_2 = float(_st_read(shard, "lm_head.weight_scale_2", np.float32).reshape(-1)[0])
        self.input_scale = float(_st_read(shard, "lm_head.input_scale", np.float32).reshape(-1)[0])
        self.packed = _st_read(shard, "lm_head.weight", np.uint8)
        self.scale_bits = _st_read(shard, "lm_head.weight_scale", np.uint8)

    def rows(self, lo: int, hi: int) -> np.ndarray:
        f4 = break_fp4_bytes(self.packed[lo:hi])           # [n, 5120] f32
        sf = E4M3[self.scale_bits[lo:hi]] * self.weight_scale_2  # [n, 320]
        n, k = f4.shape
        return (f4.reshape(n, k // BLOCK, BLOCK) * sf[..., None]).reshape(n, k)


def _row_chunks(total: int, chunk: int):
    for lo in range(0, total, chunk):
        yield lo, min(lo + chunk, total)


# ---------------------------------------------------------------- phase 1


def phase_weights(chunk: int, out_path: str) -> int:
    res: dict = {"schema": "fr14.lane5a.lmhead_weight_characterization.v1"}

    head = NvFp4Head()
    res["nvfp4"] = {
        "shard": os.path.relpath(head.shard, RADIXARK),
        "packed_shape": list(head.packed_shape),
        "scale_shape": list(head.scale_shape),
        "weight_scale_2": head.weight_scale_2,
        "input_scale": head.input_scale,
        "dequant": "w = e2m1(nibble) * e4m3(block_scale) * weight_scale_2",
    }

    ref_path = REF_BF16
    ref_bits = read_bf16(ref_path, "lm_head.weight")
    assert tuple(ref_bits.shape) == (VOCAB, HIDDEN), ref_bits.shape

    armA_bits = read_bf16(ARM_A, "lm_head.weight")

    # --- is the reference head actually unquantised? -----------------------
    # An FP8-per-channel head dequantised to BF16 has <=256 distinct magnitudes
    # per row (one FP8 grid scaled by one per-row scale).  A native BF16 head
    # has thousands.  Checked on 8 sampled rows of each candidate.
    rng = np.random.default_rng(20260818)
    probe_rows = np.sort(rng.choice(VOCAB, size=8, replace=False))
    def distinct_mags(bits: np.ndarray) -> list[int]:
        out = []
        for r in probe_rows:
            v = np.abs(bf16_to_f32(bits[r]))
            out.append(int(np.unique(v).size))
        return out
    res["distinct_magnitudes_per_row"] = {
        "probe_rows": probe_rows.tolist(),
        "reference_bf16_head": distinct_mags(ref_bits),
        "arm_a_fp8_derived_head": distinct_mags(armA_bits),
        "note": "<=256 means an fp8-per-channel grid; thousands means native BF16",
    }

    # --- accumulators ------------------------------------------------------
    acc = {
        name: dict(
            sq_err=0.0, sq_ref=0.0, max_abs=0.0, max_rel_row=0.0,
            dot=0.0, n=0, worst_row=-1, min_row_cos=2.0, min_cos_row=-1,
        )
        for name in ("nvfp4", "arm_a_fp8")
    }
    amax_ref = 0.0
    amax_nvfp4 = 0.0

    for lo, hi in _row_chunks(VOCAB, chunk):
        ref = bf16_to_f32(ref_bits[lo:hi]).astype(np.float32)
        amax_ref = max(amax_ref, float(np.abs(ref).max()))
        cands = {
            "nvfp4": head.rows(lo, hi),
            "arm_a_fp8": bf16_to_f32(armA_bits[lo:hi]).astype(np.float32),
        }
        amax_nvfp4 = max(amax_nvfp4, float(np.abs(cands["nvfp4"]).max()))
        ref_row_norm = np.linalg.norm(ref, axis=1)
        for name, w in cands.items():
            a = acc[name]
            d = w - ref
            a["sq_err"] += float((d.astype(np.float64) ** 2).sum())
            a["sq_ref"] += float((ref.astype(np.float64) ** 2).sum())
            m = float(np.abs(d).max())
            if m > a["max_abs"]:
                a["max_abs"] = m
                a["worst_row"] = int(lo + np.abs(d).max(axis=1).argmax())
            rn = np.linalg.norm(d, axis=1) / np.maximum(ref_row_norm, 1e-30)
            a["max_rel_row"] = max(a["max_rel_row"], float(rn.max()))
            cos = (w * ref).sum(axis=1) / np.maximum(
                np.linalg.norm(w, axis=1) * ref_row_norm, 1e-30
            )
            j = int(cos.argmin())
            if float(cos[j]) < a["min_row_cos"]:
                a["min_row_cos"] = float(cos[j])
                a["min_cos_row"] = int(lo + j)
            a["n"] += w.size

    # Dequant-direction proof: the multiply form must land near the reference
    # amax; the divide form would be ~1/weight_scale_2^2 (~6e7) times larger.
    res["amax"] = {
        "reference_bf16_head": amax_ref,
        "nvfp4_dequantised": amax_nvfp4,
        "ratio": amax_nvfp4 / amax_ref,
        "divide_form_would_give": amax_nvfp4 / (head.weight_scale_2 ** 2),
    }
    if not (0.5 <= amax_nvfp4 / amax_ref <= 2.0):
        res["verdict"] = "FAIL: dequant direction / reference-head mismatch"
        print(json.dumps(res, indent=1))
        return 1

    for name, a in acc.items():
        res[name if name != "nvfp4" else "nvfp4_vs_reference"] = {
            "relative_frobenius_error": (a["sq_err"] / a["sq_ref"]) ** 0.5,
            "rmse": (a["sq_err"] / a["n"]) ** 0.5,
            "max_abs_weight_delta": a["max_abs"],
            "max_abs_delta_row": a["worst_row"],
            "max_relative_row_error": a["max_rel_row"],
            "min_row_cosine": a["min_row_cos"],
            "min_row_cosine_row": a["min_cos_row"],
            "elements": a["n"],
        }
    res.pop("arm_a_fp8", None)
    res["arm_a_fp8_vs_reference_CONTROL"] = {
        "relative_frobenius_error": (acc["arm_a_fp8"]["sq_err"] / acc["arm_a_fp8"]["sq_ref"]) ** 0.5,
        "rmse": (acc["arm_a_fp8"]["sq_err"] / acc["arm_a_fp8"]["n"]) ** 0.5,
        "max_abs_weight_delta": acc["arm_a_fp8"]["max_abs"],
        "max_abs_delta_row": acc["arm_a_fp8"]["worst_row"],
        "max_relative_row_error": acc["arm_a_fp8"]["max_rel_row"],
        "min_row_cosine": acc["arm_a_fp8"]["min_row_cos"],
        "min_row_cosine_row": acc["arm_a_fp8"]["min_cos_row"],
    }
    res["nvfp4_over_fp8_error_ratio"] = (
        res["nvfp4_vs_reference"]["relative_frobenius_error"]
        / res["arm_a_fp8_vs_reference_CONTROL"]["relative_frobenius_error"]
    )
    res["verdict"] = "PASS"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


# ---------------------------------------------------------------- phase 2


def load_capture(path: str, max_rows: int):
    """Read the lane-5A capture triple written by fr14_lane5a_capture_patch.py."""
    meta = json.load(open(path + ".meta.json"))
    hidden = int(meta["hidden"])
    h = np.fromfile(path, dtype=np.float32)
    rows = h.size // hidden
    h = h[: rows * hidden].reshape(rows, hidden)
    top = np.fromfile(path + ".top.bin", dtype=np.int32)
    ntop = top.size // 3
    top = top[: ntop * 3].reshape(ntop, 3)
    n = min(rows, ntop)
    dev_argmax = top[:n, 0].astype(np.int64)
    dev_top = top[:n, 1:].view(np.float32).reshape(n, 2)
    if n > max_rows:
        # Keep a deterministic spread rather than only the prefill head.
        sel = np.linspace(0, n - 1, max_rows).astype(np.int64)
    else:
        sel = np.arange(n)
    return h[:n][sel], dev_argmax[sel], dev_top[sel], meta, n, sel


def phase_logits(hidden_path: str, chunk: int, out_path: str, topk: int,
                 max_rows: int) -> int:
    h, dev_argmax, dev_top, meta, n_total, sel = load_capture(hidden_path, max_rows)
    h = h.astype(np.float32)
    assert h.ndim == 2 and h.shape[1] == HIDDEN, h.shape
    T = h.shape[0]

    head = NvFp4Head()
    ref_bits = read_bf16(REF_BF16, "lm_head.weight")
    armA_bits = read_bf16(ARM_A, "lm_head.weight")

    logit_ref = np.zeros((T, VOCAB), dtype=np.float32)
    logit_q = np.zeros((T, VOCAB), dtype=np.float32)
    logit_a = np.zeros((T, VOCAB), dtype=np.float32)
    for lo, hi in _row_chunks(VOCAB, chunk):
        ref = bf16_to_f32(ref_bits[lo:hi]).astype(np.float32)
        logit_ref[:, lo:hi] = h @ ref.T
        logit_q[:, lo:hi] = h @ head.rows(lo, hi).T
        logit_a[:, lo:hi] = h @ bf16_to_f32(armA_bits[lo:hi]).astype(np.float32).T

    # Token strings straight out of vocab.json -- no transformers dependency on
    # the host, and the byte-level BPE spelling (Ġ for a leading space) is
    # exactly what makes a "flip" between " the" and "the" legible as harmless.
    try:
        _vocab = json.load(open(os.path.join(RADIXARK, "vocab.json")))
        _inv = {v: k for k, v in _vocab.items()}
    except Exception:
        _inv = {}

    def _tok(i: int) -> str:
        return _inv.get(i, f"<id:{i}>")

    def _softmax(x, temp):
        z = x / temp
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # The reference head's own top-1 margin: how DECIDED each position was
    # before any quantisation touched it.  A flip at margin 0.06 is a coin the
    # reference itself was flipping; a flip at margin 3.0 is an error.  Reported
    # separately, because a single pooled flip rate conflates the two and the
    # conflation always flatters whichever side you happen to be defending.
    part_ref = np.partition(logit_ref, -2, axis=1)
    margin_ref = part_ref[:, -1] - part_ref[:, -2]
    am_ref = logit_ref.argmax(axis=1)
    p_ref_1 = _softmax(logit_ref, 1.0)
    p_ref_06 = _softmax(logit_ref, 0.6)

    def compare(name, lg):
        d = lg - logit_ref
        am = lg.argmax(axis=1)
        flipped = am != am_ref
        flips = int(flipped.sum())
        kk = min(topk, VOCAB)
        ov = []
        for t in range(T):
            # kk SMALLEST of -x == kk LARGEST of x.  Getting this backwards
            # silently measures the overlap of the bottom-k tokens instead.
            a = set(np.argpartition(-logit_ref[t], kk)[:kk].tolist())
            b = set(np.argpartition(-lg[t], kk)[:kk].tolist())
            ov.append(len(a & b) / kk)
        buckets = {}
        for lo, hi in ((0.0, 0.25), (0.25, 1.0), (1.0, 3.0), (3.0, np.inf)):
            m = (margin_ref >= lo) & (margin_ref < hi)
            n = int(m.sum())
            buckets[f"margin_{lo}_{hi}"] = {
                "rows": n,
                "flips": int((flipped & m).sum()),
                "flip_rate": float((flipped & m).sum() / n) if n else None,
            }
        p_1 = _softmax(lg, 1.0)
        p_06 = _softmax(lg, 0.6)
        return {
            "max_abs_logit_delta": float(np.abs(d).max()),
            "mean_abs_logit_delta": float(np.abs(d).mean()),
            "p99_abs_logit_delta": float(np.percentile(np.abs(d), 99)),
            "logit_std_reference": float(logit_ref.std()),
            "argmax_flips": flips,
            "argmax_flip_rate": flips / T,
            "argmax_flips_by_reference_margin": buckets,
            "confident_flip_rate_margin_ge_1": (
                buckets["margin_1.0_3.0"]["flips"] + buckets["margin_3.0_inf"]["flips"]
            ) / max(1, buckets["margin_1.0_3.0"]["rows"] + buckets["margin_3.0_inf"]["rows"]),
            f"top{kk}_overlap_mean": float(np.mean(ov)),
            f"top{kk}_overlap_min": float(np.min(ov)),
            "total_variation_T1_mean": float(0.5 * np.abs(p_1 - p_ref_1).sum(axis=1).mean()),
            "total_variation_T1_max": float(0.5 * np.abs(p_1 - p_ref_1).sum(axis=1).max()),
            "total_variation_T0.6_mean": float(0.5 * np.abs(p_06 - p_ref_06).sum(axis=1).mean()),
            "total_variation_T0.6_max": float(0.5 * np.abs(p_06 - p_ref_06).sum(axis=1).max()),
            "reference_top1_margin_mean": float(margin_ref.mean()),
            "reference_top1_margin_min": float(margin_ref.min()),
            "flip_positions": np.nonzero(flipped)[0].tolist()[:64],
            "flip_reference_margins": margin_ref[flipped].tolist()[:64],
            # What the flip actually SWAPPED.  A pooled flip rate cannot tell
            # " the" -> "the" from "return" -> "raise"; the token strings can.
            "flip_tokens_reference_to_candidate": [
                [_tok(int(am_ref[i])), _tok(int(am[i])), float(margin_ref[i])]
                for i in np.nonzero(flipped)[0].tolist()[:48]
            ],
        }

    # ---------------------------------------------------------- KERNEL CHECK
    # The offline NVFP4 head model is only worth something if it reproduces the
    # DEVICE kernel's own decision on the same hidden states.  The engine banked
    # its argmax next to every captured row precisely so this can be asserted
    # rather than assumed.
    off_argmax = logit_q.argmax(axis=1)
    agree = int((off_argmax == dev_argmax).sum())
    dev_margin = dev_top[:, 0] - dev_top[:, 1]
    kernel_check = {
        "rows_compared": int(T),
        "offline_nvfp4_argmax_equals_device_argmax": agree,
        "agreement_rate": agree / T,
        "device_top1_margin_mean": float(dev_margin.mean()),
        "device_top1_margin_min": float(dev_margin.min()),
        "disagreement_rows": np.nonzero(off_argmax != dev_argmax)[0].tolist()[:32],
        "disagreement_device_margins": dev_margin[off_argmax != dev_argmax].tolist()[:32],
    }

    res = {
        "schema": "fr14.lane5a.lmhead_logit_characterization.v2",
        "hidden_states": {
            "path": hidden_path,
            "rows_used": int(T),
            "rows_captured": int(n_total),
            "capture_meta": meta,
            "abs_mean": float(np.abs(h).mean()),
            "abs_max": float(np.abs(h).max()),
        },
        "device_kernel_check": kernel_check,
        "nvfp4_vs_reference": compare("nvfp4", logit_q),
        "arm_a_fp8_vs_reference_CONTROL": compare("arm_a_fp8", logit_a),
    }
    res["verdict"] = "PASS"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("weights", "logits"), required=True)
    ap.add_argument("--hidden", default="")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=512,
                    help="rows of the capture to characterise (each costs a "
                         "248,320-wide fp32 logit row per head)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.phase == "weights":
        return phase_weights(a.chunk, a.out)
    if not a.hidden:
        raise SystemExit("--phase logits requires --hidden")
    return phase_logits(a.hidden, a.chunk, a.out, a.topk, a.max_rows)


if __name__ == "__main__":
    sys.exit(main())
