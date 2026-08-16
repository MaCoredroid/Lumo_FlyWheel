"""FR14 checkpoint surgery #2: drop `quantization_config.kv_cache_scheme`.

WHY (first fixed32 GPU boot, 2026-08-16T19:40Z, evidence
`output/fr14_b1_probe_20260816T193907Z/tail6_fixed32_b1probe/container_docker_logs.txt`):

    ValueError: Selected backend AttentionBackendEnum.TREE_ATTN is not valid
    for this configuration. Reason: ['kv_cache_dtype not supported']

raised from `qwen3_next.py:672 Attention(...)` -> `attention.py:299
get_attn_backend(...)` -> `platforms/cuda.py:303`.

MECHANISM (vLLM 0.19.2rc1.dev134, `model_executor/layers/attention/attention.py`
lines 231-239 -- verified against the pinned image's source):

    # llm-compressor mdls need to set cache_dtype to "fp8" manually.
    kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
    if kv_cache_scheme is not None:
        kv_cache_dtype = "fp8"
        calculate_kv_scales = False
        if cache_config is not None:
            cache_config.cache_dtype = "fp8"
            cache_config.calculate_kv_scales = False

The unsloth repack ships a calibrated static per-tensor FP8 KV scheme. Its mere
PRESENCE forces `kv_cache_dtype = "fp8"` for every `Attention` layer,
UNCONDITIONALLY overriding the engine's `--kv-cache-dtype auto`.
`TreeAttentionBackend.supported_kv_cache_dtypes` is `("auto","float16",
"bfloat16")` (`v1/attention/backends/tree_attn.py:32-36`), so the fixed32 stack's
mandatory TREE_ATTN backend is refused before a single weight is read.

WHY THIS IS THE RIGHT FIX, not a workaround:

* The fixed32 campaign serves BF16 KV BY CONTRACT. `FR13_FULL_ATTN_KV_FP8=0` is
  exported by `scripts/fr13_fixed32_floor_timers_seq.sh`, and
  `FIXED32_B4_KV_CACHE_MEMORY_BYTES` (46 GiB / 176k tokens) is sized for
  4 KV heads x head_dim 256 x bf16. An FP8 KV cache would silently halve the
  pool arithmetic under a constant the contract pins.
* The whole Hydra27/Tail23 tree-speculative stack IS TREE_ATTN. There is no
  fixed32 configuration that survives an fp8 KV cache.
* The per-layer escape hatch (`--kv-cache-dtype-skip-layers`) does NOT work: the
  block above mutates `cache_config.cache_dtype` GLOBALLY before the per-layer
  skip is consulted, so the engine would still size and label the cache as fp8.
  It would also change `expected_pid1_argv`.

CONSEQUENCE, stated so it is never mistaken for a silent loss: the checkpoint's
32 calibrated `self_attn.{k,v}_scale` tensors (16 full-attn layers) become
unused. They are NOT deleted from the shard. `maybe_remap_kv_scale_name`
(`model_loader/weight_utils.py:1504-1598`) remaps them to
`...self_attn.attn.{k,v}_scale`, finds no such parameter, logs
"k_scale is not loaded" and returns None -- the loader SKIPS them, it does not
fail. Re-instating the scheme restores them exactly; the removed block is
recorded verbatim in the sidecar below, so this surgery is fully reversible.

Usage (idempotent; exits 0 and changes nothing if already applied)::

    python3 results/fr14_nvfp4_port_20260816/kv_scheme_surgery.py

Writes:
  * `<model>/config.json`                          (kv_cache_scheme removed)
  * `<model>/config.json.pre_kv_scheme_surgery.bak` (the pre-surgery bytes)
  * `<model>/.lumo_kv_scheme_surgery.json`          (provenance + the removed block)

After running it the contract's model block MUST be re-pinned in the same
commit (`scripts/fr14_gen_model_manifest.py`): config.json's sha moves and two
files join the pinned set.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4")
CONFIG = MODEL_ROOT / "config.json"
BACKUP = MODEL_ROOT / "config.json.pre_kv_scheme_surgery.bak"
SIDECAR = MODEL_ROOT / ".lumo_kv_scheme_surgery.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    raw = CONFIG.read_text(encoding="utf-8")
    obj = json.loads(raw, object_pairs_hook=collections.OrderedDict)
    quant = obj.get("quantization_config")
    if not isinstance(quant, dict):
        print("no quantization_config in config.json", file=sys.stderr)
        return 2
    if "kv_cache_scheme" not in quant:
        print("kv_cache_scheme already absent -- nothing to do")
        return 0
    removed = quant.pop("kv_cache_scheme")

    # The file's own serialisation convention, verified byte-exact by round-trip
    # before the edit: json.dumps(indent=2, ensure_ascii=False), NO trailing
    # newline. Keeping it means the diff is exactly the removed block.
    new_raw = json.dumps(obj, indent=2, ensure_ascii=False)

    if not BACKUP.exists():
        BACKUP.write_text(raw, encoding="utf-8")
    CONFIG.write_text(new_raw, encoding="utf-8")

    manifest = {
        "surgery": (
            "quantization_config.kv_cache_scheme removed so the engine keeps "
            "kv_cache_dtype=auto (bf16) and TREE_ATTN is a valid backend"
        ),
        "reason": (
            "vLLM attention.py forces kv_cache_dtype='fp8' whenever "
            "quant_config.kv_cache_scheme is not None; "
            "TreeAttentionBackend.supported_kv_cache_dtypes is "
            "('auto','float16','bfloat16')"
        ),
        "removed_kv_cache_scheme": removed,
        "unused_after_surgery": (
            "32 tensors: model.language_model.layers.{11,15,19,...,63}"
            ".self_attn.{k,v}_scale -- retained in the shard, skipped by "
            "maybe_remap_kv_scale_name with a 'not loaded' warning"
        ),
        "config_sha256_before": sha256_text(raw),
        "config_sha256_after": sha256_text(new_raw),
        "config_size_before": len(raw.encode("utf-8")),
        "config_size_after": len(new_raw.encode("utf-8")),
        "backup": BACKUP.name,
    }
    SIDECAR.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
