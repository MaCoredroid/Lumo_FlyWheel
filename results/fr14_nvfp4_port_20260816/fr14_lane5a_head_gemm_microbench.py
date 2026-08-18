#!/usr/bin/env python3
"""FR14 lane 5A: the verifier lm_head GEMM, BF16 vs NVFP4, on the real weights.

RUNS INSIDE THE PINNED CONTAINER with a GPU.  Loads NOTHING but the head: the
RadixArk 4-tensor NVFP4 set (715,161,608 B) and the FP8-3.8 baseline's BF16
``lm_head.weight`` (2,542,796,800 B).  No engine, no KV reservation, no model
body -- so the number it produces is the head GEMM and only the head GEMM.

WHAT IS BEING TESTED.  The campaign's floor prices this GEMM purely by bytes at
273 GB/s: 2.620 ms NVFP4 vs 9.314 ms BF16, a 6.695 ms delta.  Two things can
falsify that: the NVFP4 kernel could be compute-bound rather than byte-bound at
these shapes (then the delta is smaller than the byte ratio promises), or the
BF16 baseline could be running well under roofline (then the delta is larger
than the port deserves credit for).  So this reports ACHIEVED BANDWIDTH for
both, next to the pinned bytes, and lets the ratio fall where it falls.

The NVFP4 side goes through vLLM's OWN ``ModelOptNvFp4LinearMethod`` --
``create_weights`` -> real tensors copied in -> ``process_weights_after_loading``
(which is what installs the swizzled block scale and selects
``FlashInferCutlassNvFp4LinearKernel``) -> ``apply``.  Re-implementing the GEMM
would measure a kernel the serve does not run.

Batch sweep because ``compute_logits`` is called with the accepted-row count,
not with 1: a term that is byte-bound at M=1 and compute-bound at M=64 is a
different lever than the floor thinks it is.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

RADIXARK = "/models/qwen3.8-27b-nvfp4-radixark"
REF_BF16 = "/models/qwen3.8-27b-fp8/outside.safetensors"

VOCAB = 248_320
HIDDEN = 5_120
NVFP4_HEAD_BYTES = 715_161_608
BF16_HEAD_BYTES = 2_542_796_800
PINNED_BW_GBPS = 273.0  # scripts/fr13_hardware_floor_ledger.py root_64k formula


def _st_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _st_tensor(path: str, name: str) -> torch.Tensor:
    hdr, base = _st_header(path)
    v = hdr[name]
    a, b = v["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        buf = bytearray(f.read(b - a))
    dt = {
        "U8": torch.uint8,
        "I8": torch.int8,
        "F8_E4M3": torch.float8_e4m3fn,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }[v["dtype"]]
    t = torch.frombuffer(buf, dtype=dt)
    return t.reshape(v["shape"]) if v["shape"] else t.reshape(())


def _find_shard(root: str, tensor: str) -> str:
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".safetensors"):
                p = os.path.join(dirpath, fn)
                if tensor in _st_header(p)[0]:
                    return p
    raise SystemExit(f"{tensor!r} not found under {root}")


def timed(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--out", default="/cap/head_gemm_microbench.json")
    ap.add_argument("--dist-port", type=int, default=51797)
    a = ap.parse_args()

    dev = torch.device("cuda")
    res: dict = {
        "schema": "fr14.lane5a.head_gemm_microbench.v1",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "pinned": {
            "nvfp4_head_bytes": NVFP4_HEAD_BYTES,
            "bf16_head_bytes": BF16_HEAD_BYTES,
            "assumed_bandwidth_GBps": PINNED_BW_GBPS,
            "nvfp4_floor_ms": NVFP4_HEAD_BYTES / PINNED_BW_GBPS / 1e6,
            "bf16_floor_ms": BF16_HEAD_BYTES / PINNED_BW_GBPS / 1e6,
            "floor_delta_ms": (BF16_HEAD_BYTES - NVFP4_HEAD_BYTES) / PINNED_BW_GBPS / 1e6,
        },
        "iters": a.iters,
    }

    # ------------------------------------------------------------ BF16 arm
    w_bf16 = _st_tensor(REF_BF16, "lm_head.weight").to(dev, non_blocking=True)
    assert tuple(w_bf16.shape) == (VOCAB, HIDDEN), w_bf16.shape
    res["bf16_weight_bytes_resident"] = w_bf16.numel() * w_bf16.element_size()

    # ------------------------------------------------------------ NVFP4 arm
    shard = _find_shard(RADIXARK, "lm_head.weight")
    packed = _st_tensor(shard, "lm_head.weight").to(dev)
    wscale = _st_tensor(shard, "lm_head.weight_scale").to(dev)
    wscale2 = _st_tensor(shard, "lm_head.weight_scale_2").to(dev)
    iscale = _st_tensor(shard, "lm_head.input_scale").to(dev)
    res["nvfp4_on_disk"] = {
        "weight": [list(packed.shape), str(packed.dtype)],
        "weight_scale": [list(wscale.shape), str(wscale.dtype)],
        "weight_scale_2": float(wscale2.reshape(-1)[0]),
        "input_scale": float(iscale.reshape(-1)[0]),
    }

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    # vLLM's parameter classes read the TP rank at construction, so a
    # single-rank group has to exist even though nothing here is distributed.
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{a.dist_port}",
        local_rank=0,
        backend="gloo",
    )

    vcfg = VllmConfig()
    with set_current_vllm_config(vcfg):
        # initialize_model_parallel itself reads the current vLLM config, so it
        # has to be inside the context, not before it.
        initialize_model_parallel(tensor_model_parallel_size=1)
        cfg = None
        for kwargs in (
            dict(is_checkpoint_nvfp4_serialized=True, kv_cache_quant_algo=None,
                 exclude_modules=[], group_size=16),
            dict(is_checkpoint_nvfp4_serialized=True, kv_cache_quant_method=None,
                 exclude_modules=[], group_size=16),
        ):
            try:
                cfg = ModelOptNvFp4Config(**kwargs)
                break
            except TypeError as e:
                last = e
        if cfg is None:
            raise SystemExit(f"could not construct ModelOptNvFp4Config: {last}")
        method = ModelOptNvFp4LinearMethod(cfg)

        layer = torch.nn.Module()
        layer.to(dev)
        method.create_weights(
            layer,
            input_size_per_partition=HIDDEN,
            output_partition_sizes=[VOCAB],
            input_size=HIDDEN,
            output_size=VOCAB,
            params_dtype=torch.bfloat16,
        )
        layer = layer.to(dev)
        with torch.no_grad():
            layer.weight.data.copy_(packed)
            layer.weight_scale.data.copy_(wscale.view(torch.uint8).view(wscale.shape).view(torch.float8_e4m3fn)
                                          if layer.weight_scale.dtype == torch.float8_e4m3fn
                                          else wscale.view(layer.weight_scale.dtype))
            layer.weight_scale_2.data.copy_(wscale2.reshape(layer.weight_scale_2.shape))
            layer.input_scale.data.copy_(iscale.reshape(layer.input_scale.shape))
        method.process_weights_after_loading(layer)
        res["nvfp4_kernel"] = type(getattr(method, "kernel", None)).__name__
        res["nvfp4_resident_bytes"] = sum(
            p.numel() * p.element_size()
            for n, p in layer.named_parameters(recurse=False)
        ) + sum(
            b.numel() * b.element_size() for n, b in layer.named_buffers(recurse=False)
        )

        # ------------------------------------------------------- correctness
        # Same input through both heads: the microbench must be timing a GEMM
        # that computes approximately the right thing, not a no-op.
        h_chk = torch.randn(8, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.05
        with torch.no_grad():
            o_q = method.apply(layer, h_chk).float()
            o_r = torch.nn.functional.linear(h_chk, w_bf16).float()
        res["sanity"] = {
            "bf16_logit_std": float(o_r.std()),
            "nvfp4_logit_std": float(o_q.std()),
            "argmax_agreement_on_random_input": float((o_q.argmax(-1) == o_r.argmax(-1)).float().mean()),
            "note": "random input, not real hidden states -- shape/scale check only",
        }

        # ------------------------------------------------------------- sweep
        rows = []
        for M in [int(x) for x in a.batches.split(",")]:
            h = torch.randn(M, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.05
            with torch.no_grad():
                ms_bf16 = timed(lambda: torch.nn.functional.linear(h, w_bf16), a.iters)
                ms_nvfp4 = timed(lambda: method.apply(layer, h), a.iters)
            rows.append({
                "M": M,
                "bf16_ms": ms_bf16,
                "nvfp4_ms": ms_nvfp4,
                "delta_ms": ms_bf16 - ms_nvfp4,
                "bf16_achieved_GBps": BF16_HEAD_BYTES / ms_bf16 / 1e6,
                "nvfp4_achieved_GBps": NVFP4_HEAD_BYTES / ms_nvfp4 / 1e6,
                "bf16_pct_of_pinned_bw": BF16_HEAD_BYTES / ms_bf16 / 1e6 / PINNED_BW_GBPS,
                "nvfp4_pct_of_pinned_bw": NVFP4_HEAD_BYTES / ms_nvfp4 / 1e6 / PINNED_BW_GBPS,
                "speedup": ms_bf16 / ms_nvfp4,
            })
            print(f"  M={M:3d}  bf16 {ms_bf16:7.3f} ms  nvfp4 {ms_nvfp4:7.3f} ms  "
                  f"delta {ms_bf16 - ms_nvfp4:7.3f} ms  x{ms_bf16/ms_nvfp4:.2f}")
        res["sweep"] = rows

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
