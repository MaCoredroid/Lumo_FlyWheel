"""FR14 checkpoint surgery #3 (RadixArk aggressive arm): drop the FP8 KV cache
declaration so the engine keeps a BF16 KV cache and TREE_ATTN stays legal.

Analogue of `kv_scheme_surgery.py` (unsloth arm) but the MECHANISM IS DIFFERENT
and had to be re-derived against the pinned image
(`vllm/vllm-openai@sha256:3dbe092e...c38e776`, vllm 0.19.2rc1.dev134+gfe9c3d6c5).
Do not assume the unsloth write-up applies.

WHERE IT BITES
--------------
The unsloth arm was refused inside `Attention.__init__`
(`model_executor/layers/attention/attention.py:232`)::

    kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)

That is a lookup on the *config object*, and `CompressedTensorsConfig` has such
an attribute. `ModelOptMixedPrecisionConfig` does NOT -- verified:
`hasattr(cfg, "kv_cache_scheme") is False`. So that road is closed here.

The RadixArk arm is forced FP8 one layer EARLIER, straight out of the raw HF
config dict, before any model code runs
(`engine/arg_utils.py:1616` -> `utils/torch_utils.py:279`)::

    resolved_cache_dtype = resolve_kv_cache_dtype_string(self.kv_cache_dtype,
                                                         model_config)
    ...
    cache_config = CacheConfig(..., cache_dtype=resolved_cache_dtype, ...)

`resolve_kv_cache_dtype_string` only acts when the user asked for ``auto`` --
which the fixed32 stack does -- and then reads
``model_config.hf_config.quantization_config`` and calls
``get_kv_cache_quant_algo_string``. That function, for any
``quant_method.startswith("modelopt")``, accepts EITHER key::

    kv_algo = (quantization_inner.get("kv_cache_scheme")
               or quant_cfg.get("kv_cache_scheme")
               or quantization_inner.get("kv_cache_quant_algo")
               or quant_cfg.get("kv_cache_quant_algo"))

RadixArk's `config.json` carries ``quantization_config.kv_cache_scheme =
{"dynamic": false, "num_bits": 8, "type": "float"}``, which that function maps
dict -> ``"fp8"`` -> ``MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP["fp8"]`` ->
``"fp8_e4m3"``. So ``--kv-cache-dtype auto`` silently becomes a GLOBAL
``cache_dtype="fp8_e4m3"``, and
``TreeAttentionBackend.supported_kv_cache_dtypes`` is
``["auto", "float16", "bfloat16"]`` (`v1/attention/backends/tree_attn.py:34-38`)
-- the same TREE_ATTN refusal as the unsloth arm, reached by a different road.

The same key is ALSO what makes `ModelOptQuantConfigBase._from_config`
(modelopt.py:286-306) set ``kv_cache_quant_method = "FP8"``, which is what makes
``ModelOptMixedPrecisionConfig.get_quant_method`` (modelopt.py:2164-2167) hand
every `Attention` layer a `ModelOptFp8KVCacheMethod`. Removing the one key
closes both doors at once.

WHY THIS IS RIGHT, NOT A WORKAROUND
-----------------------------------
* RadixArk's FP8 KV is UNCALIBRATED BY THEIR OWN ADMISSION. `qualification.json`
  records ``fp8_kv_cache: {"dtype": "fp8_e4m3", "scales_calibrated": false,
  "warning": "Using FP8 KV cache but no scaling factors provided. Defaulting to
  scaling factors of 1.0. This may lead to less accurate results!"}``, and the
  shards contain ZERO k_scale/v_scale tensors (verified over all 2194
  safetensors headers). The declaration buys nothing but precision loss.
* The fixed32 campaign serves BF16 KV BY CONTRACT
  (`FR13_FULL_ATTN_KV_FP8=0`; `FIXED32_B4_KV_CACHE_MEMORY_BYTES` is sized for
  4 KV heads x head_dim 256 x bf16). An fp8 cache would silently halve the pool
  arithmetic under a constant the contract pins.
* The whole Hydra27/Tail23 stack IS TREE_ATTN; no fixed32 configuration
  survives an fp8 KV cache.
* `--kv-cache-dtype bfloat16` would NOT be an equivalent fix: it changes
  `expected_pid1_argv`, and it leaves `kv_cache_quant_method="FP8"` in place so
  every `Attention` layer still gets a `ModelOptFp8KVCacheMethod`.

CONSEQUENCE: none that is lost. There are no calibrated KV scale tensors to
strand -- unlike the unsloth surgery, which stranded 32 real ones. The removed
blocks are recorded verbatim in the sidecars, so this is fully reversible.

Both files are edited so they cannot disagree: `config.json`'s embedded
`quantization_config` is what vLLM actually reads for this checkpoint
(`config/model.py:935`), but `hf_quant_config.json` is the declared fallback
(`ModelOptQuantConfigBase.get_config_filenames()`) and would re-attach the
FP8 KV method if it were ever consulted alone.

Usage (idempotent; exits 0 and changes nothing if already applied)::

    python3 results/fr14_nvfp4_port_20260816/radixark_kv_surgery.py [MODEL_ROOT]

Writes, per edited file:
  * ``<file>``                              (kv declaration removed)
  * ``<file>.pre_kv_surgery.bak``           (the pre-surgery bytes)
  * ``<model>/.lumo_radixark_kv_surgery.json`` (provenance + removed blocks)

After running it the contract's model block MUST be re-pinned in the same
commit: both files' shas move.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")

# (filename, json.dumps kwargs, trailing bytes) -- each verified to round-trip
# the untouched file BYTE-EXACTLY before any edit is written, so the diff is
# exactly the removed block and nothing else.
SERIALISATION = {
    "config.json": ({"indent": 2, "ensure_ascii": False}, "\n"),
    "hf_quant_config.json": ({"indent": 4, "ensure_ascii": False}, ""),
}

# The keys that make vLLM pick an FP8 KV cache, per file.
KV_KEYS = ("kv_cache_scheme", "kv_cache_quant_algo")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quant_block(obj, filename):
    """Return the dict that holds the kv declaration for this file."""
    if filename == "config.json":
        return obj.get("quantization_config")
    return obj.get("quantization")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        print(f"model root not found: {root}", file=sys.stderr)
        return 2

    sidecar_path = root / ".lumo_radixark_kv_surgery.json"
    report: dict[str, object] = {
        "surgery": (
            "removed the FP8 KV cache declaration so resolve_kv_cache_dtype_string "
            "leaves cache_dtype='auto' (bf16) and TREE_ATTN stays a legal backend, "
            "and so ModelOpt stops attaching ModelOptFp8KVCacheMethod to Attention"
        ),
        "reason": (
            "engine/arg_utils.py:1616 -> utils/torch_utils.py:279 maps "
            "quantization_config.kv_cache_scheme {'dynamic':false,'num_bits':8,"
            "'type':'float'} to cache_dtype 'fp8_e4m3' even under "
            "--kv-cache-dtype auto; TreeAttentionBackend.supported_kv_cache_dtypes "
            "is ('auto','float16','bfloat16')"
        ),
        "upstream_evidence": (
            "RadixArk qualification.json: fp8_kv_cache.scales_calibrated=false, "
            "'Defaulting to scaling factors of 1.0'; 0 k_scale/v_scale tensors "
            "across all 2194 safetensors headers -- nothing is stranded"
        ),
        "model_root": str(root),
        "files": {},
    }

    changed = False
    for filename, (dump_kwargs, tail) in SERIALISATION.items():
        path = root / filename
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw, object_pairs_hook=collections.OrderedDict)

        # Fail closed: refuse to rewrite a file we cannot reproduce byte-exactly.
        if json.dumps(obj, **dump_kwargs) + tail != raw:
            print(
                f"{filename}: serialisation round-trip is not byte-exact -- "
                "refusing to rewrite",
                file=sys.stderr,
            )
            return 3

        block = quant_block(obj, filename)
        if not isinstance(block, dict):
            print(f"{filename}: no quantization block found", file=sys.stderr)
            return 4

        removed = {k: block.pop(k) for k in KV_KEYS if k in block}
        entry: dict[str, object] = {
            "removed": removed,
            "sha256_before": sha256_text(raw),
            "size_before": len(raw.encode("utf-8")),
        }
        if not removed:
            entry["status"] = "already-absent"
            entry["sha256_after"] = entry["sha256_before"]
            entry["size_after"] = entry["size_before"]
            report["files"][filename] = entry  # type: ignore[index]
            continue

        new_raw = json.dumps(obj, **dump_kwargs) + tail
        backup = path.with_suffix(path.suffix + ".pre_kv_surgery.bak")
        if not backup.exists():
            backup.write_text(raw, encoding="utf-8")
        path.write_text(new_raw, encoding="utf-8")
        changed = True

        entry["status"] = "removed"
        entry["backup"] = backup.name
        entry["sha256_after"] = sha256_text(new_raw)
        entry["size_after"] = len(new_raw.encode("utf-8"))
        report["files"][filename] = entry  # type: ignore[index]

    report["changed"] = changed
    sidecar_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
