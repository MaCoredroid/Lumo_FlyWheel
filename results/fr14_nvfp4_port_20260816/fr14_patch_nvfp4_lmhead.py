#!/usr/bin/env python3
"""FR14 boot-time patcher: route ``ParallelLMHead`` through the ModelOpt NVFP4
linear method so the RadixArk aggressive checkpoint's 4-bit LM head can load.

RUNS INSIDE THE CONTAINER, against the pinned image's site-packages, before
``vllm serve``.  The image itself is never rebuilt and its digest never moves:
the edits land in the container's ephemeral writable layer and die with
``docker rm -f``.

WHY A FILE PATCH AND NOT A MONKEYPATCH
--------------------------------------
vLLM v1 runs the EngineCore/model worker in a *separate* process, so an
in-process monkeypatch applied by a wrapper entrypoint would never reach the
process that actually builds the model.  The two mechanisms that do reach it
are (a) a ``sitecustomize`` on ``PYTHONPATH``, which would have to either
eagerly ``import vllm`` at interpreter start -- ordering-hazardous, since vLLM
sets CUDA/NCCL env vars in children *before* importing torch -- or install an
import hook, and (b) editing the source the child imports.  (b) is fewer moving
parts, is what the FR10 patcher already does for this same image, and makes the
production port a copy-paste of the anchors below.

THE FOUR GAPS (all verified statically against
vllm 0.19.2rc1.dev134+gfe9c3d6c5, image sha256:3dbe092e...c38e776)
-----------------------------------------------------------------
G1  ``qwen3_5.py:501`` and ``qwen3_5_mtp.py:379`` build
    ``ParallelLMHead(vocab, hidden, prefix=...)`` with **no** ``quant_config``
    argument.  ``VocabParallelEmbedding.__init__`` (vocab_parallel_embedding.py
    :270-274) then does ``if quant_config is None: quant_method =
    UnquantizedEmbeddingMethod()``.  This is config-agnostic: NO quantization
    scheme of any kind can reach the head.  It is the same root cause as the
    compressed-tensors/unsloth "no parameter named lm_head.weight_scale"
    failure -- the ParallelLMHead dispatch at compressed_tensors.py:184 exists
    but is unreachable dead code for this model family.

G2  ModelOpt has no ``ParallelLMHead`` branch at all.  Both
    ``ModelOptQuantConfigBase.get_quant_method`` (modelopt.py:179-214) and
    ``ModelOptMixedPrecisionConfig.get_quant_method`` (modelopt.py:2159-2198)
    dispatch only on ``Attention``/``MLAAttention``, ``LinearBase`` and
    ``FusedMoE``.  ``ParallelLMHead`` is a ``VocabParallelEmbedding``, not a
    ``LinearBase``, so it falls through to ``return None``.  Fixing G1 alone
    would still yield ``UnquantizedEmbeddingMethod``.

G3  Key/prefix mismatch.  ``hf_quant_config.json`` keys the head as the bare
    string ``"lm_head"``, but under the Qwen3.5-VL wrapper
    (``Qwen3_5ForConditionalGeneration`` -> ``language_model`` ->
    ``Qwen3_5ForCausalLM``) the runtime module prefix is
    ``"language_model.lm_head"``.  ``apply_vllm_mapper`` remaps
    ``quantized_layers`` with ``WeightsMapper.apply_dict``, whose
    ``orig_to_new_prefix`` entry is ``"lm_head."`` -- WITH a trailing dot --
    and ``"lm_head".startswith("lm_head.")`` is False, so the key is never
    remapped.  ``_resolve_quant_algo("language_model.lm_head")`` therefore
    misses on all three of its strategies and returns None.  (The body keys
    ``model.language_model.*`` DO remap correctly, which is why only the head
    is affected.)

G4  Scalar-shape mismatch on the two global scales.  On disk
    ``lm_head.input_scale`` and ``lm_head.weight_scale_2`` are 0-dim (``[]``)
    F32 -- as are all 401/193 of their body counterparts.  The body ones load
    fine because ``LinearBase`` uses the vLLM-parameter-aware loader, but
    ``ModelOptNvFp4LinearMethod.create_weights`` allocates them as
    ``PerTensorScaleParameter(torch.empty(1))`` -- shape ``[1]`` -- and a
    ParallelLMHead's params carry ``VocabParallelEmbedding.weight_loader``,
    whose ``output_dim is None`` branch (vocab_parallel_embedding.py:440-443)
    asserts ``param.data.shape == loaded_weight.shape``.  ``[1] != []`` ->
    AssertionError.  Patched with a numel-preserving reshape only.

WHAT LANDS AFTER THE PATCH (create_weights vs the on-disk ledger)
----------------------------------------------------------------
    param                 allocated                   on disk
    weight                U8   [248320, 2560]         U8   [248320, 2560]
    weight_scale          F8E4M3 [248320, 320]        F8E4M3 [248320, 320]
    weight_scale_2        F32  [1]      <- reshaped   F32  []
    input_scale           F32  [1]      <- reshaped   F32  []
Vocab 248320 needs no padding (248320 % 64 == 0) and the NVFP4 scale swizzle
tiles exactly (248320 % 128 == 0, 320 % 4 == 0), so no pad path is taken.
``LogitsProcessor._get_logits`` calls ``lm_head.quant_method.apply(...)``
(logits_processor.py:96), so the logits GEMM becomes the NVFP4 cutlass kernel.

FAIL-CLOSED: with ``FR14_REQUIRE_NVFP4_LMHEAD=1`` in the environment, a head
that does NOT resolve to ``ModelOptNvFp4LinearMethod`` raises at construction
instead of silently serving an unquantized/half-loaded head.  Every anchor is
required to be present exactly once; a missing or duplicated anchor aborts the
patch (and therefore the boot) rather than half-applying.

Idempotent: each edit is guarded by its own sentinel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SP = Path("/usr/local/lib/python3.12/dist-packages/vllm")
QWEN3_5_PATH = SP / "model_executor/models/qwen3_5.py"
QWEN3_5_MTP_PATH = SP / "model_executor/models/qwen3_5_mtp.py"
MODELOPT_PATH = SP / "model_executor/layers/quantization/modelopt.py"
VPE_PATH = SP / "model_executor/layers/vocab_parallel_embedding.py"

applied: dict[str, str] = {}


def _replace_once(path: Path, needle: str, replacement: str, tag: str) -> str:
    """fr10 literal-anchor idiom: the anchor must occur exactly once."""
    text = path.read_text()
    n = text.count(needle)
    if n != 1:
        raise RuntimeError(
            f"FR14 lm_head patch: anchor {tag!r} occurs {n} times in {path} "
            "(expected exactly 1) -- refusing to patch"
        )
    path.write_text(text.replace(needle, replacement, 1))
    return "patched"


# --------------------------------------------------------------------------
# G1 (+ fail-closed assertion): pass quant_config into ParallelLMHead.
# Identical anchor in the target model and in the MTP drafter -- the drafter
# loads the TARGET's lm_head tensors (qwen3_5_mtp.py:441 passes any name
# containing "lm_head" straight through), so it needs the same routing.
# --------------------------------------------------------------------------
LMHEAD_NEEDLE = """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
"""

LMHEAD_REPLACEMENT = """                # FR14_LMHEAD_QUANT_ROUTE: upstream omits quant_config here, which
                # pins the head to UnquantizedEmbeddingMethod for EVERY quant
                # scheme (vocab_parallel_embedding.py:270-274).
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                _fr14_qm = type(self.lm_head.quant_method).__name__
                logger.info("FR14_LMHEAD_QUANT_ROUTE lm_head quant_method=%s", _fr14_qm)
                if (
                    os.environ.get("FR14_REQUIRE_NVFP4_LMHEAD") == "1"
                    and _fr14_qm != "ModelOptNvFp4LinearMethod"
                ):
                    raise RuntimeError(
                        "FR14_REQUIRE_NVFP4_LMHEAD=1 but lm_head resolved to "
                        f"{_fr14_qm}; refusing to serve a head that did not "
                        "route through NVFP4"
                    )
"""

for _p, _key in ((QWEN3_5_PATH, "qwen3_5"), (QWEN3_5_MTP_PATH, "qwen3_5_mtp")):
    _t = _p.read_text()
    if "FR14_LMHEAD_QUANT_ROUTE" in _t:
        applied[_key] = "already-patched"
        continue
    if "\nimport os\n" not in _t:
        _replace_once(_p, "import typing\n", "import os\nimport typing\n", f"{_key}:import os")
    applied[_key] = _replace_once(_p, LMHEAD_NEEDLE, LMHEAD_REPLACEMENT, f"{_key}:lm_head")


# --------------------------------------------------------------------------
# G2 + G3: give ModelOptMixedPrecisionConfig a ParallelLMHead branch, with a
# basename fallback for the un-remapped bare "lm_head" key.
# --------------------------------------------------------------------------
MODELOPT_NEEDLE = """            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            return None

        return None
"""

MODELOPT_REPLACEMENT = """            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            return None

        # FR14_LMHEAD_NVFP4: upstream modelopt.py dispatches only on
        # LinearBase / FusedMoE / Attention, so ParallelLMHead -- which is a
        # VocabParallelEmbedding -- falls through to `return None` and the head
        # silently becomes UnquantizedEmbeddingMethod.  compressed_tensors.py
        # :184 has the equivalent branch; modelopt has none.  Imported locally
        # to keep modelopt.py's import graph unchanged.
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead as _FR14ParallelLMHead,
        )

        if isinstance(layer, _FR14ParallelLMHead):
            _fr14_algo = quant_algo
            if _fr14_algo is None:
                # The checkpoint keys the head as the bare string "lm_head",
                # but under the Qwen3.5-VL wrapper the runtime prefix is
                # "language_model.lm_head".  WeightsMapper.orig_to_new_prefix
                # carries "lm_head." (trailing dot), and
                # "lm_head".startswith("lm_head.") is False, so apply_dict
                # never remaps the key.  Retry on the basename.
                _fr14_algo = self._resolve_quant_algo(prefix.rsplit(".", 1)[-1])
            logger.info(
                "FR14_LMHEAD_NVFP4 ParallelLMHead prefix=%s resolved_algo=%s",
                prefix,
                _fr14_algo,
            )
            if _fr14_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if _fr14_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            return None

        return None
"""

_t = MODELOPT_PATH.read_text()
if "FR14_LMHEAD_NVFP4" in _t:
    applied["modelopt"] = "already-patched"
else:
    applied["modelopt"] = _replace_once(
        MODELOPT_PATH, MODELOPT_NEEDLE, MODELOPT_REPLACEMENT, "modelopt:get_quant_method"
    )


# --------------------------------------------------------------------------
# G4: 0-dim global scales vs [1]-shaped PerTensorScaleParameter.
# numel-preserving reshape ONLY -- any real shape disagreement still asserts.
# --------------------------------------------------------------------------
VPE_NEEDLE = """        if output_dim is None:
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return
"""

VPE_REPLACEMENT = """        if output_dim is None:
            # FR14_SCALAR_SCALE_RESHAPE: ModelOpt writes the NVFP4 global
            # scales (input_scale, weight_scale_2) as 0-dim tensors, while
            # PerTensorScaleParameter allocates them as shape [1].  A
            # numel-preserving reshape is the whole fix; a genuine shape
            # disagreement still trips the assert below.
            if (
                param.data.shape != loaded_weight.shape
                and param.data.numel() == loaded_weight.numel()
            ):
                loaded_weight = loaded_weight.reshape(param.data.shape)
            assert param.data.shape == loaded_weight.shape
            param.data.copy_(loaded_weight)
            return
"""

_t = VPE_PATH.read_text()
if "FR14_SCALAR_SCALE_RESHAPE" in _t:
    applied["vocab_parallel_embedding"] = "already-patched"
else:
    applied["vocab_parallel_embedding"] = _replace_once(
        VPE_PATH, VPE_NEEDLE, VPE_REPLACEMENT, "vpe:scalar_scale"
    )


# Compile-check every file we touched: a syntax error here must kill the boot,
# not surface as a confusing import failure 40 seconds later.
import py_compile  # noqa: E402

for _p in (QWEN3_5_PATH, QWEN3_5_MTP_PATH, MODELOPT_PATH, VPE_PATH):
    py_compile.compile(str(_p), doraise=True)

print(json.dumps({"fr14_lmhead_patch": applied}, indent=1))
sys.exit(0)
