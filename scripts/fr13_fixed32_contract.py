#!/usr/bin/env python3
"""Pinned external and runtime contract for fixed-32 floor campaigns."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fr13_fixed32_topology import FIXED32_CHOICES, PHYSICAL_DRAFTS

EXTERNAL_SCHEMA = "fr13-fixed32-external-manifest-v1"
RUNTIME_SCHEMA = "fr13-fixed32-runtime-attestation-v1"
CANONICAL_FORMAT = "utf8-json-sort-keys-compact-v1"
RUNTIME_ATTESTATION_MODE = 0o644

QWEN_VISIBLE_MAX_OUTPUT_TOKENS = 32_768
QWEN_COMPACTION_MAX_OUTPUT_TOKENS = 20_000
# qwen-code's ``web_fetch`` is not a retrieval tool: after it fetches the URL
# it runs a ``runSideQuery`` model call (purpose "web-fetch", maxAttempts 1,
# the main model, the ordinary 32768 max_tokens) to extract the answer from the
# fetched bytes, and returns only that call's text. The side query never enters
# the agent's chat history, so it emits no trace assistant record -- but vLLM
# serves, bills and histograms it and our own ingress ledger records it. These
# are the two -- and in 0.19.4 the only two -- terminal displays of
# ``executeDirectFetch``; the success one is reached only after the side query
# resolves, so it is exactly one completed engine request, and the error one is
# reached from the outer catch, so it is none.
QWEN_WEB_FETCH_TOOL_NAME = "web_fetch"
QWEN_WEB_FETCH_INPUT_FIELDS = frozenset({"url", "prompt", "format"})
QWEN_WEB_FETCH_INPUT_REQUIRED_FIELDS = ("url", "prompt")
QWEN_WEB_FETCH_SUCCESS_TEMPLATE = "Content from {url} processed successfully."
QWEN_WEB_FETCH_ERROR_PREFIX = "Error: "
# qwen-code renders a client-side failure as an ordinary assistant TEXT
# record carrying this banner, with all-zero usage, and then closes the
# session with subtype="success" / is_error=false / num_turns=1. Nothing
# in that shape distinguishes it from a served turn, so the validator used
# to count it as ONE completed model request when the engine served ZERO --
# it manufactured evidence of traffic that never happened, which is the one
# direction a fail-closed counter must never fail in. Observed 2026-08-17 in
# both fr14_b1_probe_* traces:
#   "[API Error: EngineCore encountered an issue. See stack trace ...]"
#   "[API Error: Connection error. (cause: fetch failed)]"   <- never left
#                                                               the client
# We cannot tell from the trace alone whether the engine served it, so we
# REFUSE rather than guess a number in either direction.
QWEN_API_ERROR_TEXT_PREFIX = "[API Error:"
# qwen-code validates a tool call against its JSON schema BEFORE executing
# it. A web_fetch missing a required parameter is rejected here, with
# is_error=True and this ajv-shaped message, and never reaches
# executeDirectFetch -- so it fetches nothing and issues no runSideQuery,
# and owes ZERO completed engine requests. Observed 2026-08-17 in
# fr14_b1_stock_20260817T020534Z astropy__astropy-13236 line 159:
#   tool_use  web_fetch {"url": "https://raw.githubusercontent.com/..."}
#   result    "params must have required property 'prompt'"  is_error=True
QWEN_TOOL_SCHEMA_REJECTION_PREFIX = "params must have required property"
QWEN_COMPACTION_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-compaction-metrics-v1"
)
QWEN_CAMPAIGN_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-campaign-metrics-v1"
)
QWEN_CAMPAIGN_TASK_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-campaign-task-metrics-v1"
)
QWEN_CAMPAIGN_TOKEN_LEDGER_SCHEMA = (
    "fr13-fixed32-ingress-ledger-token-usage-v1"
)
FIXED32_INGRESS_LEDGER_RECORD_SCHEMA = "fr13.fixed32.ingress-ledger-record.v1"
_FIXED32_LEDGER_USAGE_KEYS = ("prompt_tokens", "completion_tokens")
_FIXED32_LEDGER_USAGE_EVENTS = frozenset(
    {"request_complete", "logical_complete"}
)
_FIXED32_LEDGER_COMPLETION_EVENTS = {
    "proxy": "logical_complete",
    "engine": "request_complete",
}
# What the two reconciliation bases are called in published evidence. The
# ledger basis is the engine's own per-request count, joined to the digest
# chain; the trace basis is qwen-code's self-report, which under-credits its
# own hidden requests (rejected compactions, discarded first turns, delegated
# sub-agents reporting 0/0) and is therefore a structural witness only.
QWEN_TOKEN_BASIS_LEDGER = "fixed32_ingress_ledger_usage"
QWEN_TOKEN_BASIS_TRACE = "qwen_trace_result_usage"

_QWEN_COMPACTION_FAILURE_TEXT_RE = re.compile(
    r"\[API Error: Context is too large to send safely after automatic "
    r"compression\. Estimated prompt tokens: ([1-9][0-9]*); hard limit: "
    r"([1-9][0-9]*); compression status: "
    r"COMPRESSION_FAILED_EMPTY_SUMMARY\. Start a new session or reduce the "
    r"resumed history before continuing\.\]"
)

IMAGE_REFERENCE = (
    "vllm/vllm-openai@"
    "sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
)
IMAGE_ID = "sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc"
IMAGE_OS = "linux"
IMAGE_ARCHITECTURE = "arm64"
VLLM_VERSION = "0.19.2rc1.dev134+gfe9c3d6c5"

class ContractError(RuntimeError):
    """Raised when a fixed-32 contract value is not exact."""


NSYS_PROFILE_BINARY = Path(
    "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
)
NSYS_PROFILE_OUTPUT = Path("/logs/fr13_fixed32_b1_real_swe")
# The B4 width-4 attribution writes its own report so a width-4 capture can
# never be mistaken for, or overwrite, the B1 one.
NSYS_PROFILE_OUTPUT_WIDTH4 = Path("/logs/fr13_b4_width4_real_swe")
NSYS_PROFILE_OUTPUTS = (NSYS_PROFILE_OUTPUT, NSYS_PROFILE_OUTPUT_WIDTH4)


def nsys_profile_prefix(
    *,
    deferred_capture: bool = False,
    capture_output: Path | str | None = None,
) -> tuple[str, ...]:
    """The exact PID1 profiler argv a wrapped fixed32 server must present.

    Two capture gates exist and they are mutually exclusive:

    * WALL-GATED (default) -- `--delay 1200 --duration 300`, the canonical B1
      attribution window. Unchanged, and what every existing caller gets.
    * STEP-GATED (`deferred_capture=True`) -- `--start-later=true`, collecting
      nothing until an external `nsys start` arrives. Required by the B4
      width-4 profile, whose window is defined in absolute forward-step
      indices: the admit->first-step hydration lag is ~118 steps, so a fixed
      wall delay cannot name the step range it lands on
      (results/fr13_b4_refill_citable_20260812/width4_window.md §6).

    `--delay`/`--duration` are OMITTED entirely in the deferred shape rather
    than set alongside `--start-later`: nsys documents `--start-later` as
    overriding `--delay`, and a surviving `--duration` would silently re-impose
    a wall bound on a step-gated capture.

    The attestation stays exact in both shapes -- this returns one specific
    argv, never a pattern -- so a wrapped PID1 is still matched element by
    element and an unexpected profiler invocation is still refused.
    """
    if capture_output is None:
        capture_output = NSYS_PROFILE_OUTPUT
    output = Path(capture_output)
    if output not in NSYS_PROFILE_OUTPUTS:
        raise ContractError(f"fixed32 nsys capture output is not pinned: {output}")
    gate: tuple[str, ...] = (
        ("--start-later=true",)
        if deferred_capture
        else ("--delay", "1200", "--duration", "300")
    )
    return (
        str(NSYS_PROFILE_BINARY),
        "profile",
        "--session-new=%q{LUMO_NSYS_SESSION_NAME}",
        *gate,
        "--trace=cuda,cuda-sw,nvtx",
        "--cuda-graph-trace=node",
        "--cuda-flush-interval",
        "100",
        "--discard-environment=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "-o",
        str(output),
    )

# The canonical wall-gated B1 prefix, unchanged. Kept as a module constant
# because callers and tests pin it by identity.
NSYS_PROFILE_PREFIX = nsys_profile_prefix()

FA2_REPO_RELATIVE = (
    "output/auto_research/"
    "qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-"
    "20260504T053925Z/cutlass_source_workspace/vllm-source/build/"
    "lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
)
FA2_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
FA2_SIZE = 299_183_936
QROW16_FA2_SHA256 = (
    "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
)
QROW16_FA2_SIZE = 299_507_792
QROW16_DIVFREE_FA2_SHA256 = (
    "106e54d1c82ec7ce7576cbb44bb4aa2342b2985bb58e97aeeca5503275bee3e2"
)
QROW16_DIVFREE_FA2_SIZE = 299_491_544
QROW32_B1_SPLIT2_FA2_SHA256 = (
    "a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a"
)
QROW32_B1_SPLIT2_FA2_SIZE = 300_154_616
QROW32_B1_VISIBILITY_FA2_SHA256 = (
    "c5ab32a6ae4e615f1e77a4997db5429152053c549e761fb11d90b33bb3959a79"
)
QROW32_B1_VISIBILITY_FA2_SIZE = 300_200_192
QROW32_B1_GQA_PAIR_FA2_SHA256 = (
    "3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae"
)
QROW32_B1_GQA_PAIR_FA2_SIZE = 299_815_552
# FR14 lane 4 split-K. TIER-B: the context walk is split four ways and
# re-reduced, so this arm is not byte-identical to the incumbent by design and
# can never pass the raw-byte gate. It serves only under a validated Tier-B
# qualification credential (Mark, pass 64), never as a promoted default. The
# SASS digests are pinned beside the artifact because THIS arm's .so sha is not
# rebuild-reproducible -- four links from one closure gave two .so hashes at an
# identical size -- so the digests are what attest the kernel reproduced, and
# the baseline digest is what keeps "the split-K header edits are inert at
# Split=false" a measurement rather than a claim.
QROW32_B1_SPLITK_FA2_SHA256 = (
    "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
)
QROW32_B1_SPLITK_FA2_SIZE = 300_123_792
QROW32_B1_SPLITK_SOURCE_CLOSURE_SHA256 = (
    "4ed00909cef7ea83849f897018ea4f6a14119b8d160927af426938920c170878"
)
QROW32_B1_SPLITK_SASS_DIGEST_SHA256 = (
    "3f24d70dce2ff70ad9209bad5af2a93cc39453df529cb298e4476cbfbfd80b9e"
)
QROW32_B1_SPLITK_BASELINE_SASS_DIGEST_SHA256 = (
    "fa01f98840420b9c0177d06297aacabb0ed5e00c674511fdaa4aa618c3473470"
)
QROW32_B1_SPLITK_NUM_SPLITS = 4
# The arms that may present a TIER-B credential instead of byte identity. This
# tuple is the whole allowance: an arm not named here gets the byte-exact
# Tier-A path unchanged, and nothing byte-gated becomes easier because this
# tuple exists.
QROW32_B1_TIER_B_ARMS = ("gqa_pair_splitk",)
QROW32_B4_FA2_SHA256 = (
    "77f3fb22c19d0eb2ac0ec28230cf9401221425692a505efde62aa838760d81ce"
)
QROW32_B4_FA2_SIZE = 299_876_120
QROW32_B4_GQA_PAIR_FA2_SHA256 = (
    "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
)
QROW32_B4_GQA_PAIR_FA2_SIZE = 299_813_360
QROW32_B4_VISIBILITY_FA2_SHA256 = (
    "805635d6881dbf73287d66c10541880b7cf93bcb6bf7b04e50efd3d32728b0aa"
)
QROW32_B4_VISIBILITY_FA2_SIZE = 299_810_632
CONTAINER_FA2_SOURCE = Path("/tmp/fr13_fork_fa2.so")
CONTAINER_FA2_DESTINATION = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
)

MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")
# FR14 served-model-name. The FR13 name was "qwen3.6-27b" (bare, no quant
# suffix). FR14 keeps the quantisation IN the name on purpose: the FP8-3.8
# baseline arm (/models/qwen3.8-27b-fp8) is a structurally identical serve of
# the same base model, so a bare "qwen3.8-27b" would let an FP8 boot satisfy
# every Prometheus/trace comparison meant for the NVFP4 arm. With the suffix
# a mis-pointed serve 404s on the very first request instead of quietly
# producing a decomposition-proof QC readout.
MODEL_SERVED_NAME = "qwen3.8-27b-nvfp4-radixark"
# The canonical exact4 prompts need enough live KV capacity to admit four
# concurrent real requests; 20 GiB capped the scheduler at physical B2.
#
# 40 GiB was still too small to hold four of them RESIDENT. The engine reports
# "GPU KV cache size: 153,600 tokens" at 40 GiB, against a four-task working set
# of ~4 x 40k = ~160k tokens, so the pool cannot cover the batch it is booted
# for and the tail of the oldest prefix is evicted to admit the newest. The
# 16-task refill diagnostic measured the consequence directly: once the pool
# runs full-width, APC hit rate collapses to 33-40% (40.5% at 23:19:38) while KV
# utilisation pins at 75-83%
# (output/fr13_b4_refill_diag_20260808T230623Z/analysis/apc_timeline.txt).
# 46 GiB restores ~176k tokens, which covers the working set with margin.
#
# This is the sizing lever, not FR13_SPEC_BLOCKS_CAP. That flag is a TRAP: the
# lever it named was measured BELOW the no-lever baseline (cap 29.62 vs 32.14
# tps) and its implementation was excised on 2026-07-25 in dce60d18c -- 101
# lines covering the env read, the mamba patch, the consumer width caps and the
# preflight. Nothing reads the env, and since 2026-08-09 (d96d36200) neither
# the launcher nor fr13_required_tree_flags.sh advertises it any more.
# FR13_LEVER_REDESIGN.md already routes the cache-hit-rate concern here, to
# pool sizing, instead.
#
# FR13_MAMBA_SPEC_BLOCKS_CDIV (2026-08-09) started in the same territory and
# was BLOCKED for the same structural reason -- num_speculative_blocks counts
# mamba STATE SLOTS, one per draft node, not a token range, so the physical
# narrowing alone short-fed the per-node consumers. That objection was answered
# on 2026-08-10 by pairing the narrowing with the col-aliased scratch table
# (9d8095ea0), which keeps every logical spec-window column at its full
# num_spec + 1 width over two physical pages, so the flag is now PROMOTED to
# the fixed32 B4 default (749f83af6). It is fixed32-only: the fail-loud guard
# 4b3c7f8d4 refuses it otherwise. See fr13_required_tree_flags.sh for the
# per-node consumer audit and fr10_phase4_patch_vllm_tree_gdn.py's
# _fr13_assert_mamba_spec_blocks_cdiv_slot_demand /
# _fr13_assert_mamba_spec_blocks_cdiv_coherent for the fail-loud preflights.
# It is a page-reclaim lever, not a pool-sizing one; this constant stays the
# sizing lever.
#
# Raising this DOES NOT re-profile memory: vLLM logs "reserved 40.0 GiB memory
# for KV Cache as specified by kv_cache_memory_bytes config and skipped memory
# profiling. This does not respect the gpu_memory_utilization config", so the
# 0.70 GPU_UTIL is not the binding constraint. The measured headroom is
# "Initial free memory 104.25 GiB", so 46 GiB leaves ~58 GiB for weights and
# activations, and the B4 container cap stays at 112g.
FIXED32_B4_KV_CACHE_MEMORY_BYTES = 46 * 1024**3
# FR14 ARM B (2026-08-16) MODEL BLOCK -- regenerated wholesale, not edited.
#
# The served checkpoint is RadixArk/Qwen3.8-27B-NVFP4 @
# 554ebba9b5f1b79dc11246341960360e6ef05ef4, the AGGRESSIVE side of the FR14
# bytes ablation: NVFP4 W4A4-g16 across the MLP stack, FP8 attention/GDN
# projections, BF16 GDN conv/in_proj + norms, and -- the term that actually
# pays -- an NVFP4 lm_head (0.715 GB against the 2.543 GB BF16 head arm A was
# forced into). Arm A (unsloth, /models/qwen3.8-27b-nvfp4) is still on disk;
# re-serving it is a REVERT of this constant-train commit, because the floor
# literals it needs are spread across ~50 timing instruments.
#
# LAYOUT DIFFERENCES from arm A that this block encodes:
#   * THREE shards, model-0000{1,2,3}-of-00003.safetensors, not one
#     model.safetensors -- and the 15 `mtp.*` drafter tensors live INSIDE
#     shard 3, not in a separate model_mtp.safetensors. Anything that located
#     the drafter by FILENAME had to be re-keyed to locate it BY TENSOR NAME
#     (scripts/fr13_hardware_floor_ledger.py._classify_tensor).
#   * 26 pinned names: 22 upstream files + ".lumo_pinned_revision" + the KV
#     surgery sidecar + its two ".bak" files. No layers-{0..63}.safetensors.
#
# File-set semantics are unchanged from FR13/arm A: _pinned_model_files walks
#   sorted(path.name for path in model_root.iterdir() if path.is_file())
# so a metadata DIRECTORY such as .cache/huggingface is excluded by
# construction, while dot-FILES are members -- FR13 already pinned
# ".gitattributes" on exactly that rule.
#
# THE THREE MUTATIONS this checkpoint carries over its pinned upstream
# revision, each with in-dir provenance that the contract pins DELIBERATELY so
# a silent deletion fails rather than passes:
#   1. ".lumo_radixark_kv_surgery.json" + "config.json.pre_kv_surgery.bak" +
#      "hf_quant_config.json.pre_kv_surgery.bak" -- quantization_config
#      .kv_cache_scheme (and hf_quant_config's kv_cache_quant_algo, so the two
#      files cannot disagree) removed. RadixArk is forced to FP8 KV one layer
#      EARLIER than the unsloth arm was: arg_utils.py:1616 -> torch_utils.py
#      :279 maps the raw HF key to cache_dtype "fp8_e4m3" even under
#      --kv-cache-dtype auto, and TREE_ATTN accepts only auto/float16/bfloat16.
#      Nothing is stranded -- their own qualification.json admits
#      scales_calibrated=false and there are ZERO k_scale/v_scale tensors in
#      all 2,194 headers. Script: results/fr14_nvfp4_port_20260816/
#      radixark_kv_surgery.py.
#   2. "tokenizer_config.json" normalised to the official 3.8 file. RadixArk
#      ships a 1,121-byte conversion-tool stub with no added_tokens_decoder, no
#      chat_template, and pad_token="<|im_end|>" -- the chat format's STOP
#      token. The original is archived OUTSIDE the model dir at
#      /home/mark/shared/models/_fr14_orig_nvfp4_fp8head/
#      tokenizer_config.json.radixark.bak so the pinned name set stays at the
#      26 the checkpoint ships; script + record are in
#      results/fr14_nvfp4_port_20260816/radixark_tokenizer_normalize.py.
#      tokenizer.json / vocab.json / merges.txt / chat_template.jinja were
#      already byte-identical to official 3.8 and were NOT touched.
#   3. NO lm_head surgery. Arm A needed one; arm B does not -- the NVFP4 head
#      loads and generates through the boot-time loader patch
#      (scripts/fr14_patch_nvfp4_lmhead.py, fail-closed under
#      FR14_REQUIRE_NVFP4_LMHEAD=1). Its absence from this file set is the
#      evidence that the 4-bit head is served as shipped.
#
# Regenerate with:
#   python3 scripts/fr14_gen_model_manifest.py \
#     --model-root /models/qwen3.8-27b-nvfp4-radixark --emit-python <path>
# and verify an existing pin with the same script's --check.
MODEL_AUXILIARY_FILES = (
    ".gitattributes",
    ".lumo_pinned_revision",
    ".lumo_radixark_kv_surgery.json",
    ".quant_summary.txt",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "config.json.pre_kv_surgery.bak",
    "conversion-manifest.json",
    "generation_config.json",
    "hf_quant_config.json",
    "hf_quant_config.json.pre_kv_surgery.bak",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "qualification-criteria.json",
    "qualification.json",
    "tensor-audit.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
MODEL_FILES = tuple(sorted(MODEL_AUXILIARY_FILES))
MODEL_CANONICAL_SHA256 = (
    "7e89afacd7351493508a358b7d83e43f141111736d19142bf89c5698033fe84f"
)
MODEL_TEXT_CONFIG_VOCAB_SIZE = 248_320
# FR14 tokenizer-identity pin. vocab_size alone cannot catch a vocabulary
# REORDERING, and the K64 draft-vocab block map
# (scripts/fr13_dvk_subset_blocks.json, 512 measured 128-id blocks) indexes
# lm_head rows by token id -- a remap would keep every boot assertion green and
# only show up as a silently degraded accept rate, which this campaign would
# then mis-attribute to NVFP4 quality loss.
#
# tokenizer.json is NOT comparable across 3.6/3.8 by file bytes: the tokenizers
# library changed how merges serialise ("\u0120 t" strings became
# ["\u0120","t"] lists) and 3.8 adds 7 audio/TTS special tokens inside the
# reserved 248044->248320 padding range plus a pre_tokenizer regex tweak. But
# vocab.json IS byte-identical across qwen3.6-27b-fp8, qwen3.8-27b-fp8 and
# qwen3.8-27b-nvfp4-radixark (verified 2026-08-16), so its sha256 is the robust
# id-mapping pin and it is the same sha FR13 pinned for the 3.6 dir. Asserting
# it here is what carries the DVK block map across the model swap.
MODEL_VOCAB_JSON_SHA256 = (
    "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
)
MODEL_FILE_RECORDS = (
    (
        ".gitattributes",
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    (
        ".lumo_pinned_revision",
        41,
        "4d8d0f1fb6eabdbf5527798d2ae245254d67c92bfe9124dd4ecca6f547850f53",
    ),
    (
        ".lumo_radixark_kv_surgery.json",
        1651,
        "fc35f2510d54ca3a5cad9312b4b29683ccc0157f7e8cee9bdc568ef92b65b5bd",
    ),
    (
        ".quant_summary.txt",
        314291,
        "5920198f1770fc91c0f8032108e6d861ce7e0ef196eca140c9681231d8d99967",
    ),
    (
        "LICENSE",
        11544,
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    ),
    (
        "README.md",
        4574,
        "3cfd18e07422e6eff20fcdf8dcdb3c864976dcef0e1e0f2c3f8bf788603203c2",
    ),
    (
        "chat_template.jinja",
        8952,
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    ),
    (
        "config.json",
        72904,
        "08abd8e204ecae324108f41acc1bac10549a3cd0728737ecfa0fc94b21bbac73",
    ),
    (
        "config.json.pre_kv_surgery.bak",
        73003,
        "7ff41ec6f96ad50efea3c92751cd261b63839d39936eb6e6ffc9066db8672740",
    ),
    (
        "conversion-manifest.json",
        23587,
        "c71e938cadabd25b2d6ec6b1bd15afecb618674cd8004cf1ecf04e28badbbf55",
    ),
    (
        "generation_config.json",
        214,
        "a4cef85934ea1fdcb207944dbc6eee70dbbf16806874428556ae33023336c0a4",
    ),
    (
        "hf_quant_config.json",
        53711,
        "f1ac5f2c91c307559544c17d2435fc845e6fa5d4f8828a4707c89f446fd6eb78",
    ),
    (
        "hf_quant_config.json.pre_kv_surgery.bak",
        53749,
        "0f39e8cd23abdfb79adc89ac1b19acad990aa6ac32973f9ab0a67d1e3449535c",
    ),
    (
        "merges.txt",
        3353259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    (
        "model-00001-of-00003.safetensors",
        9965652544,
        "fbcdb5ba1cdda462b5f38592d071e772c4d398afea61a0aa9188b32d1a239a79",
    ),
    (
        "model-00002-of-00003.safetensors",
        9985757064,
        "db6146a5464fb0a891181b93c81593f0ca65c602eb14120a1c2b1b09bca11f85",
    ),
    (
        "model-00003-of-00003.safetensors",
        1970287672,
        "d3cfb92742e30c8b46564665791dbe0a86ed64cfc02b1275081530793c0c9581",
    ),
    (
        "model.safetensors.index.json",
        214866,
        "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
    ),
    (
        "preprocessor_config.json",
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    (
        "qualification-criteria.json",
        1145,
        "8e3b57367245354acb040a133e63df8eb1f4aef787e036156b944dfda217352a",
    ),
    (
        "qualification.json",
        2717,
        "7be9d60606fc9590ca7b5717018e12050deaa6d7d93abb5e8cad0845240983a8",
    ),
    (
        "tensor-audit.json",
        539,
        "8a7801e2b46298432a129689879c9e4f8c69444e0b71b4c971470c2747794679",
    ),
    (
        "tokenizer.json",
        12809320,
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    ),
    (
        "tokenizer_config.json",
        17928,
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    ),
    (
        "video_preprocessor_config.json",
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    (
        "vocab.json",
        6722759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
)

ARCTIC_VERSION = "0.1.2"
ARCTIC_SDIST_SHA256 = "e4f4d5a4f25c5ba2b0d1641d9a66f0d38cab5859ff3571eb4c59898bce0dca50"
ARCTIC_SDIST_URL = (
    "https://files.pythonhosted.org/packages/a9/c9/"
    "9ade0a7ec01f98b5340f1d0e3699f0fb2a686fe1c8594f1e7055607b3d0e/"
    "arctic_inference-0.1.2.tar.gz"
)
ARCTIC_PINNED_REQUIREMENT = (
    f"arctic-inference @ {ARCTIC_SDIST_URL}#sha256={ARCTIC_SDIST_SHA256}"
)


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def expected_model_file_records() -> list[dict[str, Any]]:
    return [
        {"path": path, "size": size, "sha256": sha256}
        for path, size, sha256 in MODEL_FILE_RECORDS
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixed32_tree_text() -> str:
    return repr(list(FIXED32_CHOICES))


def speculative_config_text() -> str:
    return json.dumps(
        {
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": PHYSICAL_DRAFTS,
            "speculative_token_tree": fixed32_tree_text(),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def expected_pid1_argv(concurrency: int) -> list[str]:
    if concurrency not in (1, 4):
        raise ContractError(f"fixed32 concurrency must be 1 or 4, got {concurrency}")
    argv = [
        "/usr/bin/python3",
        "/usr/local/bin/vllm",
        "serve",
        str(MODEL_ROOT),
        "--served-model-name",
        MODEL_SERVED_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "9950",
        "--max-num-seqs",
        str(concurrency),
        "--gpu-memory-utilization",
        "0.70",
        "--max-model-len",
        "131072",
        "--seed",
        "0",
    ]
    if concurrency == 4:
        argv.extend(
            [
                "--kv-cache-memory-bytes",
                str(FIXED32_B4_KV_CACHE_MEMORY_BYTES),
            ]
        )
    argv.extend(
        [
        "--attention-backend",
        "TREE_ATTN",
        "--gdn-prefill-backend",
        "triton",
        "--chat-template",
        "/workspace/docker/chat_templates/qwen3-openai-codex.jinja",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--reasoning-parser",
        "qwen3",
        "--speculative-config",
        speculative_config_text(),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--mamba-block-size",
        "1024",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--max-num-batched-tokens",
        "4096",
        "--block-size",
        "1024",
        "--long-prefill-token-threshold",
        "1024",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
        "--middleware",
        "lumo_flywheel_serving.inference_proxy.Fixed32EngineIngressMiddleware",
        ]
    )
    return argv


def expected_process_pid1_argv(
    concurrency: int,
    *,
    attribution_only: bool,
    eager_diagnostic: bool = False,
    graph_diagnostic: bool = False,
    streamk_eager_diagnostic: bool = False,
    sfwd_byte_diagnostic: bool = False,
    deferred_capture: bool = False,
    capture_output: object = None,
) -> list[str]:
    if type(attribution_only) is not bool:
        raise ContractError("fixed32 attribution-only selector must be boolean")
    if type(eager_diagnostic) is not bool:
        raise ContractError("fixed32 eager-diagnostic selector must be boolean")
    if type(graph_diagnostic) is not bool:
        raise ContractError("fixed32 graph-diagnostic selector must be boolean")
    if type(streamk_eager_diagnostic) is not bool:
        raise ContractError(
            "fixed32 Stream-K eager-diagnostic selector must be boolean"
        )
    if type(sfwd_byte_diagnostic) is not bool:
        raise ContractError(
            "fixed32 SFWD byte-diagnostic selector must be boolean"
        )
    if sum(
        (eager_diagnostic, graph_diagnostic, streamk_eager_diagnostic)
    ) > 1:
        raise ContractError(
            "fixed32 process diagnostics are mutually exclusive"
        )
    # The SFWD conv/post-prep and prior-reuse byte gates are eager kernel byte
    # diagnostics (fr13_run_b1_sfwd_conv_postprep_gate.sh,
    # fr13_run_b4_sfwd_embedded_gate_live_gate.sh,
    # fr13_run_b1_sfwd_prior_reuse_gate.sh). They are legal at B1 and B4 and
    # they ride EITHER the stock wave or a B1 CUTLASS byte wave, so they
    # compose with the Stream-K B1 eager selector instead of excluding it —
    # both selectors demand the identical trailing '--enforce-eager'. They
    # never compose with the graph diagnostic, which is the one non-eager
    # selector.
    if sfwd_byte_diagnostic and graph_diagnostic:
        raise ContractError(
            "fixed32 process diagnostics are mutually exclusive"
        )
    if attribution_only and (
        eager_diagnostic or streamk_eager_diagnostic or sfwd_byte_diagnostic
    ):
        raise ContractError(
            "fixed32 eager diagnostic cannot be attribution-only"
        )
    if attribution_only and graph_diagnostic:
        raise ContractError(
            "fixed32 graph diagnostic cannot be attribution-only"
        )
    if eager_diagnostic and concurrency != 4:
        raise ContractError(
            "fixed32 eager diagnostic requires concurrency 4"
        )
    if graph_diagnostic and concurrency != 4:
        raise ContractError(
            "fixed32 graph diagnostic requires concurrency 4"
        )
    if streamk_eager_diagnostic and concurrency != 1:
        raise ContractError(
            "fixed32 Stream-K eager diagnostic requires concurrency 1"
        )
    vllm_argv = expected_pid1_argv(concurrency)
    if eager_diagnostic or streamk_eager_diagnostic or sfwd_byte_diagnostic:
        vllm_argv = [*vllm_argv, "--enforce-eager"]
    if not attribution_only:
        # An unwrapped server has no profiler prefix at all, so asking for a
        # capture shape here is a caller bug, not a no-op. Refuse BEFORE the
        # early return -- placing this after it silently accepted the
        # contradiction.
        if deferred_capture or capture_output is not None:
            raise ContractError(
                "fixed32 capture shape requires attribution-only mode"
            )
        return vllm_argv
    prefix = nsys_profile_prefix(
        deferred_capture=deferred_capture, capture_output=capture_output
    )
    return [*prefix, "vllm", *vllm_argv[2:]]


def validate_process_pid1_argv(
    argv: object,
    concurrency: int,
    *,
    attribution_only: bool,
    eager_diagnostic: bool = False,
    graph_diagnostic: bool = False,
    streamk_eager_diagnostic: bool = False,
    sfwd_byte_diagnostic: bool = False,
    deferred_capture: bool = False,
    capture_output: object = None,
) -> list[str]:
    expected = expected_process_pid1_argv(
        concurrency,
        attribution_only=attribution_only,
        eager_diagnostic=eager_diagnostic,
        graph_diagnostic=graph_diagnostic,
        streamk_eager_diagnostic=streamk_eager_diagnostic,
        sfwd_byte_diagnostic=sfwd_byte_diagnostic,
        deferred_capture=deferred_capture,
        capture_output=capture_output,
    )
    if argv != expected:
        raise ContractError(f"fixed32 PID1 argv mismatch: {argv!r}")
    return expected


def _fixed32_trace_message(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") == "assistant":
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message
    if event.get("type") == "message" and event.get("role") == "assistant":
        return event
    return None


def _fixed32_nonempty_text_record(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and bool(item["text"].strip())
        for item in content
    )


def _fixed32_qwen_reasoning_only_record(message: dict[str, Any]) -> bool:
    """True when an assistant record carries reasoning content and nothing else.

    Qwen may legally close a task on a reasoning-only turn: the record holds
    only ``thinking`` blocks, so it contributes no visible text and no
    ``tool_use``. The engine still served that logical model request -- the
    turn appears in the engine's request metrics -- so the campaign policy
    counts it as served and it must reconcile like any other response group.
    """
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(item, dict) and item.get("type") == "thinking"
        for item in content
    )


def _fixed32_qwen_api_error_record(message: dict[str, Any]) -> bool:
    """True when an assistant record is a client-side ``[API Error: ...]`` banner.

    See QWEN_API_ERROR_TEXT_PREFIX. Such a record is qwen-code narrating its own
    failure, not a model response: the engine may have served nothing at all.
    """
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    for item in content:
        if (
            not isinstance(item, dict)
            or item.get("type") != "text"
            or not isinstance(item.get("text"), str)
        ):
            continue
        text = item["text"].lstrip()
        if not text.startswith(QWEN_API_ERROR_TEXT_PREFIX):
            continue
        # The local compression-failure terminal is ALSO an [API Error: ...]
        # banner, but it is a fully modelled shape with its own accounting
        # (_fixed32_qwen_synthetic_compaction_failure + the failed-compaction
        # split): it names its own token counts and is matched exactly. It is
        # evidence, not an unaccountable failure, so it is not refused here.
        if _QWEN_COMPACTION_FAILURE_TEXT_RE.fullmatch(item["text"]) is not None:
            continue
        return True
    return False


def _fixed32_qwen_blank_text_record(message: dict[str, Any]) -> bool:
    """True when an assistant record carries only whitespace-only text blocks.

    Qwen may trail its closing reasoning with a text record that is pure
    whitespace -- observed live as ``"\\n\\n"``. The record contributes no
    visible text and no ``tool_use``, so it can never be read as a
    submission; it only shows that the turn emitted an empty message. It is
    never sufficient on its own (see the caller), because a bare whitespace
    record is indistinguishable from a degenerate empty response.
    """
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and not item["text"].strip()
        for item in content
    )


def _fixed32_qwen_synthetic_compaction_failure(
    group: list[tuple[dict[str, Any], dict[str, Any], str, int]],
    *,
    result: dict[str, Any],
) -> bool:
    """Recognize the exact local Qwen compression-failure terminal."""
    if len(group) != 1:
        return False
    event, message, event_id, _event_index = group[0]
    if set(event) != {
        "type",
        "uuid",
        "session_id",
        "parent_tool_use_id",
        "message",
    } or set(message) != {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "usage",
    }:
        return False
    content = message.get("content")
    usage = message.get("usage")
    if (
        event.get("type") != "assistant"
        or event.get("parent_tool_use_id") is not None
        or message.get("id") != event_id
        or message.get("type") != "message"
        or message.get("role") != "assistant"
        or message.get("model") != MODEL_SERVED_NAME
        or message.get("stop_reason") is not None
        or usage != {"input_tokens": 0, "output_tokens": 0}
        or not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        return False
    text = content[0]["text"]
    match = _QWEN_COMPACTION_FAILURE_TEXT_RE.fullmatch(text)
    if match is None or int(match.group(1)) <= int(match.group(2)):
        return False
    return result.get("result") == text


def fixed32_trace_session_id(instance_id: str) -> str:
    if not isinstance(instance_id, str) or not instance_id:
        raise ContractError("fixed32 trace instance ID must be nonempty")
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"urn:fr13:fixed32:trace-session:{instance_id}",
        )
    )


def _fixed32_qwen_group_request_id(event_ids: list[str]) -> str:
    payload = json.dumps(
        event_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"qwen-assistant-group-sha256:{hashlib.sha256(payload).hexdigest()}"


def _fixed32_qwen_hidden_agent_terminal_request_id(
    *,
    agent_tool_use_id: str,
    child_event_ids: list[str],
    outer_tool_result_event_id: str,
) -> str:
    payload = json.dumps(
        {
            "agent_tool_use_id": agent_tool_use_id,
            "child_event_ids": child_event_ids,
            "outer_tool_result_event_id": outer_tool_result_event_id,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "qwen-hidden-agent-terminal-sha256:"
        f"{hashlib.sha256(payload).hexdigest()}"
    )


def _fixed32_qwen_hidden_web_fetch_request_id(
    *,
    web_fetch_tool_use_id: str,
    tool_result_event_id: str,
    url: str,
) -> str:
    payload = json.dumps(
        {
            "tool_result_event_id": tool_result_event_id,
            "url": url,
            "web_fetch_tool_use_id": web_fetch_tool_use_id,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "qwen-hidden-web-fetch-sha256:"
        f"{hashlib.sha256(payload).hexdigest()}"
    )


def _fixed32_qwen_hidden_compaction_request_id(
    *,
    previous_group_event_ids: list[str],
    intervening_event_ids: list[str],
    next_group_event_ids: list[str],
    previous_input_tokens: int,
    next_input_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "previous_group_event_ids": previous_group_event_ids,
            "intervening_event_ids": intervening_event_ids,
            "next_group_event_ids": next_group_event_ids,
            "previous_input_tokens": previous_input_tokens,
            "next_input_tokens": next_input_tokens,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "qwen-hidden-compaction-sha256:"
        f"{hashlib.sha256(payload).hexdigest()}"
    )


def _fixed32_qwen_hidden_failed_compaction_request_id(
    *,
    result_event_id: str,
    trace_event_ids_sha256: str,
    metric_evidence_sha256: str,
    ordinal: int,
) -> str:
    payload = json.dumps(
        {
            "metric_evidence_sha256": metric_evidence_sha256,
            "ordinal": ordinal,
            "result_event_id": result_event_id,
            "trace_event_ids_sha256": trace_event_ids_sha256,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "qwen-hidden-failed-compaction-sha256:"
        f"{hashlib.sha256(payload).hexdigest()}"
    )


def _fixed32_qwen_metric_labels(
    *,
    finished_reason: str | None = None,
    le: str | None = None,
) -> str:
    fields = ['engine="0"']
    if finished_reason is not None:
        fields.append(f'finished_reason="{finished_reason}"')
    if le is not None:
        fields.append(f'le="{le}"')
    fields.append(f'model_name="{MODEL_SERVED_NAME}"')
    return ",".join(fields)


# --------------------------------------------------------------------------- #
# engine completion classes                                                    #
# --------------------------------------------------------------------------- #
# Every completed logical model request terminates in exactly one vLLM
# finished_reason, and the algebra needs a category for each one that
# LEGITIMATELY occurs.  Until 2026-08-12 it had only one, "stop", and pinned the
# other four at zero.
#
#   stop    the model emitted its stop condition.  The overwhelming majority.
#
#   length  the model reached its max_tokens cap (32768 visible / 20000
#           compaction) and its response was truncated.  LEGAL AND ACCOUNTED,
#           not tolerated: the engine served the request to completion, the
#           decode work is fully counted in generation_tokens and in the step
#           wall, and the request occupies its histogram bucket exactly like any
#           other (max_tokens_count / le_50000 / le_inf / max_tokens_sum all
#           reconcile to the digit).  What the agent received was a truncated
#           assistant message -- an ordinary outcome of real agent traffic on
#           long turns, not a defect and not a measurement error.
#
#   abort / error / repetition  DEFECTS.  Still pinned at zero.  An aborted or
#           errored request means the engine did not serve what was asked, and a
#           repetition stop means vLLM's degenerate-output detector fired.  None
#           of those may appear in evidence-grade traffic.
#
# WHY THIS ONLY SURFACED NOW.  The class is rare, so it is a function of scale.
# An exact4 arm serves roughly 100 logical requests; the first 16-task pool arm
# served 390 and produced exactly one length termination (in
# astropy__astropy-14369, the last-closing bracket).  The 4-task campaigns the
# algebra was validated on never drew one.  The count is published in the metric
# evidence so a reader can see how much truncated traffic a campaign contained
# rather than having it silently absorbed into a "non_stop" bucket.
#
#   abort   ADDED 2026-08-13 as a CONDITIONAL class, for the FR13 campaign
#           per-task budget cap only.  Killing a capped agent kills its
#           in-flight logical request, and vLLM finishes that request with
#           finished_reason="abort".  The class is legal ONLY when the campaign
#           declares exactly how many tasks it capped, and then the counter must
#           equal that number EXACTLY -- it is an accounted category with
#           corroborating evidence from a different source (the per-task runner
#           records), not a tolerance.  With no declared cap it stays pinned at
#           zero exactly as before, so every pre-2026-08-13 arm reconciles
#           byte-for-byte identically.
#
#           An aborted request is NOT a completion: the agent never received a
#           response, so it appears in no trace and is in no `completed` count.
#           It IS a finished request, so it occupies a max_tokens histogram
#           bucket like any other -- which is why the histogram identity gains
#           the capped term while the completion identity does not.
#
#   error / repetition  DEFECTS.  Still pinned at zero, unconditionally.  An
#           errored request means the engine failed to serve what was asked, and
#           a repetition stop means vLLM's degenerate-output detector fired.
#           Neither is ever legal, capped campaign or not.
QWEN_TERMINAL_COMPLETION_REASONS = ("stop", "length")
QWEN_CAPPED_COMPLETION_REASON = "abort"
QWEN_FORBIDDEN_COMPLETION_REASONS = ("error", "repetition")


def _fixed32_qwen_completion_classes(
    deltas: dict[str, int],
    *,
    completed: int,
    scope: str,
    capped_requests: int = 0,
) -> dict[str, int]:
    """Split completed engine requests into terminal classes, or fail loud.

    One implementation for both the single-task and the campaign-union path:
    they had the identical clause and drifting them apart is how a class ends up
    legal in one and forbidden in the other.

    ``capped_requests`` is the number of logical requests the FR13 campaign
    budget cap is DECLARED to have aborted -- one per capped task, corroborated
    by that task's own runner record.  It defaults to 0, which is the pre-cap
    behaviour exactly: abort pinned at zero and the histogram equal to the
    completion count.

    Every failure names the measured numbers.  A gate that only says "do not
    reconcile" makes the next run guess -- which is exactly what the first
    pool16 pass had to do.
    """
    if isinstance(capped_requests, bool) or type(capped_requests) is not int:
        raise ContractError(
            f"fixed32 qwen {scope} capped request count must be an int, "
            f"got {capped_requests!r}"
        )
    if capped_requests < 0:
        raise ContractError(
            f"fixed32 qwen {scope} capped request count must not be negative, "
            f"got {capped_requests}"
        )
    counts = {
        reason: deltas[f"request_success_{reason}"]
        for reason in QWEN_TERMINAL_COMPLETION_REASONS
    }
    forbidden = {
        reason: deltas[f"request_success_{reason}"]
        for reason in QWEN_FORBIDDEN_COMPLETION_REASONS
        if deltas[f"request_success_{reason}"] != 0
    }
    if forbidden:
        raise ContractError(
            f"fixed32 qwen {scope} engine completion metrics do not reconcile: "
            "forbidden completion reasons present ("
            + ", ".join(f"{k}={v}" for k, v in sorted(forbidden.items()))
            + "); error/repetition mean the engine did not serve what was "
            "asked and may never appear in evidence-grade traffic"
        )
    aborted = deltas[f"request_success_{QWEN_CAPPED_COMPLETION_REASON}"]
    if aborted != capped_requests:
        raise ContractError(
            f"fixed32 qwen {scope} engine completion metrics do not reconcile: "
            f"abort={aborted} but the campaign declares capped_requests="
            f"{capped_requests}"
            + (
                "; an abort with no declared budget cap means the engine did not"
                " serve what was asked"
                if capped_requests == 0
                else "; every budget-capped task must abort exactly one logical"
                " request, and every abort must be a declared cap"
            )
        )
    counts[QWEN_CAPPED_COMPLETION_REASON] = aborted
    terminal_total = sum(
        counts[reason] for reason in QWEN_TERMINAL_COMPLETION_REASONS
    )
    # Finished requests, not completions: a capped abort never reaches the agent
    # and so is in no trace, but vLLM still finished it and still histogrammed
    # its max_tokens.
    expected_histogram = completed + capped_requests
    histogram = {
        key: deltas[key]
        for key in ("max_tokens_count", "max_tokens_le_inf", "max_tokens_le_50000")
    }
    mismatched = {k: v for k, v in histogram.items() if v != expected_histogram}
    if mismatched or terminal_total != completed:
        raise ContractError(
            f"fixed32 qwen {scope} engine completion metrics do not reconcile: "
            f"completed={completed} but "
            + ", ".join(
                f"{reason}={counts[reason]}"
                for reason in QWEN_TERMINAL_COMPLETION_REASONS
            )
            + f" (terminal total {terminal_total})"
            + (
                "; histogram "
                + ", ".join(f"{k}={v}" for k, v in sorted(mismatched.items()))
                + f" against expected {expected_histogram} "
                f"(completed {completed} + capped {capped_requests})"
                if mismatched
                else ""
            )
        )
    return counts


def _fixed32_qwen_metric_snapshot(
    raw: bytes,
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(raw, bytes) or not raw:
        raise ContractError(
            f"fixed32 qwen {label} metrics must be nonempty bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(
            f"fixed32 qwen {label} metrics are not UTF-8"
        ) from error

    expected: dict[tuple[str, str], str] = {
        (
            "vllm:prompt_tokens_total",
            _fixed32_qwen_metric_labels(),
        ): "prompt_tokens",
        (
            "vllm:generation_tokens_total",
            _fixed32_qwen_metric_labels(),
        ): "generation_tokens",
        (
            "vllm:request_params_max_tokens_count",
            _fixed32_qwen_metric_labels(),
        ): "max_tokens_count",
        (
            "vllm:request_params_max_tokens_sum",
            _fixed32_qwen_metric_labels(),
        ): "max_tokens_sum",
    }
    for reason in ("stop", "length", "abort", "error", "repetition"):
        expected[
            (
                "vllm:request_success_total",
                _fixed32_qwen_metric_labels(finished_reason=reason),
            )
        ] = f"request_success_{reason}"
    for le, key in (
        ("10000.0", "max_tokens_le_10000"),
        ("20000.0", "max_tokens_le_20000"),
        ("50000.0", "max_tokens_le_50000"),
        ("+Inf", "max_tokens_le_inf"),
    ):
        expected[
            (
                "vllm:request_params_max_tokens_bucket",
                _fixed32_qwen_metric_labels(le=le),
            )
        ] = key

    target_names = {name for name, _labels in expected}
    values: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            series, value_text = stripped.rsplit(None, 1)
        except ValueError:
            series = stripped
            value_text = ""
        if "{" in series and series.endswith("}"):
            name, labels = series[:-1].split("{", 1)
        else:
            name, labels = series, ""
        if name not in target_names:
            continue
        key = expected.get((name, labels))
        if key is None:
            if (
                name == "vllm:request_params_max_tokens_bucket"
                and labels.startswith('engine="0",le="')
                and labels.endswith(f'",model_name="{MODEL_SERVED_NAME}"')
            ):
                continue
            raise ContractError(
                f"fixed32 qwen {label} metric {name} labels differ"
            )
        if key in values:
            raise ContractError(
                f"fixed32 qwen {label} metric {name} is duplicated"
            )
        try:
            value = Decimal(value_text)
        except InvalidOperation as error:
            raise ContractError(
                f"fixed32 qwen {label} metric {name} is malformed"
            ) from error
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ContractError(
                f"fixed32 qwen {label} metric {name} is not a "
                "nonnegative integer"
            )
        values[key] = int(value)
    missing = sorted(set(expected.values()) - set(values))
    if missing:
        raise ContractError(
            f"fixed32 qwen {label} metrics are missing {missing}"
        )
    return values


def _fixed32_qwen_compaction_metric_evidence(
    *,
    events: list[dict[str, Any]],
    result: dict[str, Any],
    normal_request_count: int,
    successful_compaction_count: int,
    synthetic_compaction_failure_terminal: bool,
    unobservable_compaction_boundaries: int,
    expected_completed_logical_model_requests: int,
    metrics_pre: bytes,
    metrics_post: bytes,
) -> tuple[dict[str, Any], int]:
    if (
        type(expected_completed_logical_model_requests) is not int
        or expected_completed_logical_model_requests <= 0
    ):
        raise ContractError(
            "fixed32 qwen expected completed request count is invalid"
        )
    before = _fixed32_qwen_metric_snapshot(metrics_pre, label="pre")
    after = _fixed32_qwen_metric_snapshot(metrics_post, label="post")
    deltas: dict[str, int] = {}
    for key in sorted(before):
        if after[key] < before[key]:
            raise ContractError(
                f"fixed32 qwen metric {key} decreased across task"
            )
        deltas[key] = after[key] - before[key]

    completed = expected_completed_logical_model_requests
    completion_classes = _fixed32_qwen_completion_classes(
        deltas, completed=completed, scope="task"
    )
    if deltas["max_tokens_le_10000"] != 0:
        raise ContractError(
            "fixed32 qwen max-token histogram has an unpinned low request"
        )

    total_compactions = deltas["max_tokens_le_20000"]
    expected_max_tokens_sum = (
        normal_request_count * QWEN_VISIBLE_MAX_OUTPUT_TOKENS
        + total_compactions * QWEN_COMPACTION_MAX_OUTPUT_TOKENS
    )
    if (
        total_compactions < successful_compaction_count
        or normal_request_count + total_compactions != completed
        or deltas["max_tokens_sum"] != expected_max_tokens_sum
    ):
        # Name every measured number. The FR14 bring-up burned a full
        # diagnosis pass on this clause because it said only "does not
        # reconcile" -- the numbers below would have said "the trace is one
        # 32768 request short" in one line.
        raise ContractError(
            "fixed32 qwen 32768/20000 max-token algebra does not reconcile: "
            f"trace normal={normal_request_count} + "
            f"le_20000_compactions={total_compactions} against engine "
            f"completed={completed} (trace-visible successful compactions "
            f"{successful_compaction_count}); max_tokens_sum="
            f"{deltas['max_tokens_sum']} against expected "
            f"{expected_max_tokens_sum}, a shortfall of "
            f"{deltas['max_tokens_sum'] - expected_max_tokens_sum}"
        )

    result_usage = result.get("usage")
    if not isinstance(result_usage, dict):
        raise ContractError("fixed32 qwen result usage is missing")
    aggregate_input = result_usage.get("input_tokens")
    aggregate_output = result_usage.get("output_tokens")
    aggregate_total = result_usage.get("total_tokens")
    if (
        type(aggregate_input) is not int
        or aggregate_input < 0
        or type(aggregate_output) is not int
        or aggregate_output < 0
        or type(aggregate_total) is not int
        or aggregate_total != aggregate_input + aggregate_output
        or aggregate_input != deltas["prompt_tokens"]
        or aggregate_output != deltas["generation_tokens"]
    ):
        raise ContractError(
            "fixed32 qwen aggregate and vLLM token usage do not reconcile"
        )

    visible_input = 0
    visible_output = 0
    for event in events:
        message = _fixed32_trace_message(event)
        if message is None:
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            raise ContractError(
                "fixed32 qwen assistant usage is missing"
            )
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            type(input_tokens) is not int
            or input_tokens < 0
            or type(output_tokens) is not int
            or output_tokens < 0
        ):
            raise ContractError(
                "fixed32 qwen assistant token usage is invalid"
            )
        visible_input += input_tokens
        visible_output += output_tokens
    hidden_input = aggregate_input - visible_input
    hidden_output = aggregate_output - visible_output
    if (
        hidden_input < 0
        or hidden_output < 0
        or (total_compactions > 0 and (hidden_input <= 0 or hidden_output <= 0))
    ):
        raise ContractError(
            "fixed32 qwen hidden compaction token usage is invalid"
        )

    failed_compactions = total_compactions - successful_compaction_count
    # A compaction inside a delegated (sub-agent) conversation can never show
    # up as a top-level input-token drop, so demand trace-visible or synthetic
    # evidence only for compactions beyond the unobservable boundaries the
    # trace actually contains. The exact 32768/20000 algebra above already
    # pins every engine request.
    if (
        failed_compactions > 0
        and successful_compaction_count <= 0
        and synthetic_compaction_failure_terminal is not True
        and failed_compactions > unobservable_compaction_boundaries
    ):
        raise ContractError(
            "fixed32 qwen failed compactions lack a trace-visible "
            "successful compaction or exact synthetic failure terminal"
        )
    evidence = {
        "schema": QWEN_COMPACTION_METRIC_SCHEMA,
        "metrics_pre_sha256": hashlib.sha256(metrics_pre).hexdigest(),
        "metrics_post_sha256": hashlib.sha256(metrics_post).hexdigest(),
        "completed_engine_requests": completed,
        "normal_visible_max_output_tokens": (
            QWEN_VISIBLE_MAX_OUTPUT_TOKENS
        ),
        "compaction_max_output_tokens": (
            QWEN_COMPACTION_MAX_OUTPUT_TOKENS
        ),
        "normal_requests": normal_request_count,
        "successful_compaction_requests": successful_compaction_count,
        "failed_compaction_requests": failed_compactions,
        "total_compaction_requests": total_compactions,
        "unobservable_compaction_boundaries": (
            unobservable_compaction_boundaries
        ),
        "max_tokens_count": deltas["max_tokens_count"],
        "max_tokens_sum": deltas["max_tokens_sum"],
        "max_tokens_le_10000": deltas["max_tokens_le_10000"],
        "max_tokens_le_20000": deltas["max_tokens_le_20000"],
        "max_tokens_le_50000": deltas["max_tokens_le_50000"],
        "max_tokens_le_inf": deltas["max_tokens_le_inf"],
        "request_success_stop": completion_classes["stop"],
        # Truncated-at-max_tokens completions, counted rather than absorbed.
        # Published so a reader can see how much of a campaign's traffic ran to
        # its output cap; forbidden reasons are proven zero by
        # _fixed32_qwen_completion_classes, so this IS the whole non-stop
        # remainder and request_success_non_stop stays exact.
        "request_success_length": completion_classes["length"],
        "request_success_non_stop": completion_classes["length"],
        "prompt_tokens": deltas["prompt_tokens"],
        "generation_tokens": deltas["generation_tokens"],
        "visible_prompt_tokens": visible_input,
        "visible_generation_tokens": visible_output,
        "hidden_prompt_tokens": hidden_input,
        "hidden_generation_tokens": hidden_output,
    }
    return evidence, failed_compactions


def _fixed32_qwen_user_tool_result(
    event: dict[str, Any],
) -> tuple[str, bool] | None:
    if event.get("type") != "user":
        return None
    message = event.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return None
    item = content[0]
    if not isinstance(item, dict) or item.get("type") != "tool_result":
        return None
    tool_use_id = item.get("tool_use_id")
    is_error = item.get("is_error")
    if (
        not isinstance(tool_use_id, str)
        or not tool_use_id
        or type(is_error) is not bool
    ):
        return None
    return tool_use_id, is_error


def _fixed32_qwen_group_input_tokens(
    group: list[tuple[dict[str, Any], dict[str, Any], str, int]],
) -> int | None:
    positive_values: set[int] = set()
    for _event, message, _event_id, _event_index in group:
        value = message["usage"].get("input_tokens")
        if type(value) is not int or value < 0:
            raise ContractError(
                "fixed32 qwen assistant input-token usage is invalid"
            )
        if value > 0:
            positive_values.add(value)
    if len(positive_values) > 1:
        raise ContractError(
            "fixed32 qwen assistant group input-token usage differs"
        )
    return next(iter(positive_values), None)


def _fixed32_qwen_unobservable_compaction_boundaries(
    assistant_groups: list[
        list[tuple[dict[str, Any], dict[str, Any], str, int]]
    ],
) -> int:
    """Count response-group boundaries where a compaction cannot be seen.

    ``_fixed32_qwen_hidden_compaction_requests`` infers a successful
    compaction from an input-token drop between consecutive *top-level*
    response groups. Delegated (sub-agent) conversations report
    ``{"input_tokens": 0, "output_tokens": 0}`` on every assistant record, so
    a compaction performed inside one is structurally invisible to that
    detector no matter how large the delegated context grows. Each adjacent
    pair of such unobservable groups within one delegated conversation is one
    boundary a compaction can legitimately hide behind; the count bounds how
    many unattributed compactions the engine histogram may report.
    """
    boundaries = 0
    previous_parent: str | None = None
    previous_unobservable = False
    for group in assistant_groups:
        parent_tool_use_id = group[0][0].get("parent_tool_use_id")
        unobservable = (
            parent_tool_use_id is not None
            and _fixed32_qwen_group_input_tokens(group) is None
        )
        if (
            unobservable
            and previous_unobservable
            and parent_tool_use_id == previous_parent
        ):
            boundaries += 1
        previous_parent = parent_tool_use_id
        previous_unobservable = unobservable
    return boundaries


def _fixed32_qwen_hidden_compaction_requests(
    events: list[dict[str, Any]],
    *,
    top_level_groups: list[
        list[tuple[dict[str, Any], dict[str, Any], str, int]]
    ],
) -> list[tuple[int, str]]:
    hidden_requests: list[tuple[int, str]] = []
    for previous_group, next_group in zip(
        top_level_groups,
        top_level_groups[1:],
    ):
        previous_input_tokens = _fixed32_qwen_group_input_tokens(
            previous_group
        )
        next_input_tokens = _fixed32_qwen_group_input_tokens(next_group)
        if (
            previous_input_tokens is None
            or next_input_tokens is None
            or next_input_tokens >= previous_input_tokens
        ):
            continue

        expected_tool_ids = [
            item["id"]
            for _event, message, _event_id, _event_index in previous_group
            for item in message["content"]
            if item.get("type") == "tool_use"
        ]
        boundary_start = previous_group[-1][3] + 1
        boundary_end = next_group[0][3]
        intervening_events = events[boundary_start:boundary_end]
        observed_tool_ids: list[str] = []
        for event in intervening_events:
            tool_result = _fixed32_qwen_user_tool_result(event)
            if (
                event.get("parent_tool_use_id") is not None
                or tool_result is None
            ):
                raise ContractError(
                    "fixed32 qwen input-usage drop is not bounded by "
                    "top-level tool results"
                )
            observed_tool_ids.append(tool_result[0])
        if (
            not expected_tool_ids
            or observed_tool_ids != expected_tool_ids
        ):
            raise ContractError(
                "fixed32 qwen input-usage drop tool results do not reconcile"
            )

        hidden_requests.append(
            (
                boundary_end - 1,
                _fixed32_qwen_hidden_compaction_request_id(
                    previous_group_event_ids=[
                        record[2] for record in previous_group
                    ],
                    intervening_event_ids=[
                        event["uuid"] for event in intervening_events
                    ],
                    next_group_event_ids=[
                        record[2] for record in next_group
                    ],
                    previous_input_tokens=previous_input_tokens,
                    next_input_tokens=next_input_tokens,
                ),
            )
        )
    return hidden_requests


def _fixed32_qwen_hidden_web_fetch_requests(
    events: list[dict[str, Any]],
    *,
    tool_use_records: dict[str, dict[str, Any]],
) -> list[tuple[int, str]]:
    """Count the ordinary model request each ``web_fetch`` hides.

    THE HOLE THIS CLOSES. ``_fixed32_qwen_hidden_compaction_requests`` knows
    the 20000-max_tokens compaction the agent hides, and
    ``_fixed32_qwen_hidden_agent_terminal_requests`` knows the sub-agent's
    final turn. Neither knows the third hidden class: a TOOL that calls the
    model. qwen-code 0.19.4's ``web_fetch`` fetches the URL and then issues a
    ``runSideQuery`` completion at the ordinary 32768 max_tokens to extract the
    answer from the fetched bytes, returning only that call's text as the tool
    result. FR13's 236 banked 3.6 traces never once called the tool, so the
    32768/20000 algebra never had to account for it. The first 3.8 arm called
    it on its second task (astropy__astropy-13033) and the validator did what
    it is built to do: 17 trace-visible requests against 18 engine requests, so
    it failed closed on traffic it could not see.

    This does not skip that traffic -- it counts it, off evidence the trace
    itself carries. ``executeDirectFetch`` has exactly two terminal returns.
    The success display is emitted only on the line AFTER ``runSideQuery``
    resolves, so it proves exactly one completed engine request; it embeds the
    caller's own ``url`` parameter, so it cannot be forged by a tool result
    that did not come from this invocation. The ``Error: `` display comes from
    the outer catch, which is reached when the fetch or the side query fails,
    and a failed side query is an error/abort completion that
    ``_fixed32_qwen_completion_classes`` already proves is zero -- so it
    accounts for no completed request. Any other closure is unaccountable and
    fails closed, as does any ``web_fetch`` whose invocation or result is not
    exactly one well-formed pair. The engine's own max-token histogram and our
    ingress ledger remain the meters; this only lets the trace name the
    request they already counted.
    """
    hidden_requests: list[tuple[int, str]] = []
    for tool_use_id, origin in tool_use_records.items():
        if origin["name"] != QWEN_WEB_FETCH_TOOL_NAME:
            continue
        params = origin["input"]
        if not isinstance(params, dict):
            raise ContractError("fixed32 qwen web_fetch input is not an object")
        if not set(params) <= QWEN_WEB_FETCH_INPUT_FIELDS:
            raise ContractError(
                "fixed32 qwen web_fetch input contains unknown fields"
            )
        missing_required = [
            field
            for field in QWEN_WEB_FETCH_INPUT_REQUIRED_FIELDS
            if not isinstance(params.get(field), str)
            or not params[field].strip()
        ]
        if "format" in params and not isinstance(params["format"], str):
            raise ContractError(
                "fixed32 qwen web_fetch format selector is invalid"
            )

        result_indices = [
            event_index
            for event_index, event in enumerate(events[:-1])
            if (_fixed32_qwen_user_tool_result(event) or (None,))[0]
            == tool_use_id
        ]
        if len(result_indices) != 1:
            raise ContractError(
                "fixed32 qwen web_fetch has no unique owner tool result"
            )
        result_index = result_indices[0]
        result_event = events[result_index]
        if (
            "parent_tool_use_id" not in result_event
            or result_event["parent_tool_use_id"]
            != origin["parent_tool_use_id"]
            or result_index <= origin["event_index"]
        ):
            raise ContractError(
                "fixed32 qwen web_fetch owner tool result is invalid"
            )
        _result_tool_use_id, is_error = _fixed32_qwen_user_tool_result(
            result_event
        )
        content = result_event["message"]["content"][0].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ContractError(
                "fixed32 qwen web_fetch tool result content is empty"
            )
        if missing_required:
            # SCHEMA REJECTION, not unaccountable traffic. The call never
            # reached executeDirectFetch, so it fetched nothing and issued no
            # runSideQuery: it owes zero completed engine requests. Still
            # fail-closed -- the trace must SHOW the rejection. A malformed
            # invocation that somehow came back successful is exactly the
            # unaccountable case this validator exists for, and still raises.
            if is_error is not True or not content.startswith(
                QWEN_TOOL_SCHEMA_REJECTION_PREFIX
            ):
                raise ContractError(
                    "fixed32 qwen web_fetch "
                    f"{missing_required[0]} is empty or invalid"
                )
            continue
        if is_error is True and content.startswith(
            QWEN_WEB_FETCH_ERROR_PREFIX
        ):
            # The outer catch: the fetch or the side query failed, so no
            # completed engine request is owed for this invocation.
            continue
        if is_error is not False or content != (
            QWEN_WEB_FETCH_SUCCESS_TEMPLATE.format(url=params["url"])
        ):
            raise ContractError(
                "fixed32 qwen web_fetch closure is neither the processed "
                "display nor a fetch error"
            )
        hidden_requests.append(
            (
                result_index,
                _fixed32_qwen_hidden_web_fetch_request_id(
                    web_fetch_tool_use_id=tool_use_id,
                    tool_result_event_id=result_event["uuid"],
                    url=params["url"],
                ),
            )
        )
    hidden_requests.sort(key=lambda record: record[0])
    return hidden_requests


def _fixed32_qwen_user_text(event: dict[str, Any]) -> bool:
    if event.get("type") != "user":
        return False
    message = event.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return False
    item = content[0]
    return (
        isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and bool(item["text"].strip())
    )


def _fixed32_qwen_tool_descends_from(
    tool_use_records: dict[str, dict[str, Any]],
    tool_use_id: str,
    ancestor_tool_use_id: str,
) -> bool:
    current_tool_use_id: str | None = tool_use_id
    visited: set[str] = set()
    while current_tool_use_id is not None:
        if current_tool_use_id == ancestor_tool_use_id:
            return True
        if current_tool_use_id in visited:
            raise ContractError("fixed32 qwen tool ancestry contains a cycle")
        visited.add(current_tool_use_id)
        record = tool_use_records.get(current_tool_use_id)
        if record is None:
            return False
        current_tool_use_id = record["parent_tool_use_id"]
    return False


def _fixed32_qwen_agent_outer_result_is_async(
    event: dict[str, Any],
) -> bool:
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return False
    item = content[0]
    if not isinstance(item, dict):
        return False
    result_content = item.get("content")
    return isinstance(result_content, str) and (
        result_content.startswith("Background agent launched successfully.")
        or result_content.startswith("Fork started")
        or result_content.startswith("Teammate ")
    )


def _fixed32_qwen_agent_outer_result_is_failure(
    event: dict[str, Any],
) -> bool:
    message = event.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return False
    item = content[0]
    if not isinstance(item, dict):
        return False
    result_content = item.get("content")
    return isinstance(result_content, str) and (
        result_content.startswith("Failed to run subagent:")
        or result_content.startswith("Subagent execution failed.")
        or result_content.startswith("Agent was cancelled by the user.")
        or result_content.startswith(
            "(subagent produced no model-visible output)"
        )
    )


def _fixed32_qwen_hidden_agent_terminal_requests(
    events: list[dict[str, Any]],
    *,
    assistant_groups: list[
        list[tuple[dict[str, Any], dict[str, Any], str, int]]
    ],
    tool_use_records: dict[str, dict[str, Any]],
    nested_error_index: int | None,
    nested_error_parent_tool_use_id: str | None,
) -> list[tuple[int, str]]:
    groups_by_parent: dict[
        str,
        list[list[tuple[dict[str, Any], dict[str, Any], str, int]]],
    ] = {}
    for group in assistant_groups:
        parent_tool_use_id = group[0][0].get("parent_tool_use_id")
        if isinstance(parent_tool_use_id, str):
            groups_by_parent.setdefault(parent_tool_use_id, []).append(group)
    for parent_tool_use_id in groups_by_parent:
        origin = tool_use_records.get(parent_tool_use_id)
        if origin is None or origin["name"] != "agent":
            raise ContractError(
                "fixed32 qwen nested response has a non-agent parent"
            )

    agent_sessions: dict[str, dict[str, Any]] = {}
    for agent_tool_use_id, origin in tool_use_records.items():
        if origin["name"] != "agent":
            continue
        params = origin["input"]
        if not isinstance(params, dict):
            raise ContractError("fixed32 qwen agent input is not an object")

        outer_result_indices: list[int] = []
        for event_index, event in enumerate(events[:-1]):
            tool_result = _fixed32_qwen_user_tool_result(event)
            if (
                tool_result is not None
                and tool_result[0] == agent_tool_use_id
            ):
                outer_result_indices.append(event_index)
        if len(outer_result_indices) != 1:
            raise ContractError(
                "fixed32 qwen agent has no unique owner tool result"
            )
        outer_result_index = outer_result_indices[0]
        outer_result_event = events[outer_result_index]
        if (
            "parent_tool_use_id" not in outer_result_event
            or outer_result_event["parent_tool_use_id"]
            != origin["parent_tool_use_id"]
            or outer_result_index <= origin["event_index"]
        ):
            raise ContractError(
                "fixed32 qwen agent owner tool result is invalid"
            )
        outer_result = _fixed32_qwen_user_tool_result(outer_result_event)
        if outer_result is None:
            raise ContractError("fixed32 qwen agent tool result is malformed")
        outer_result_content = outer_result_event["message"]["content"][0].get(
            "content"
        )
        if (
            not isinstance(outer_result_content, str)
            or not outer_result_content.strip()
        ):
            raise ContractError(
                "fixed32 qwen agent tool result content is empty"
            )

        prompt_indices = [
            event_index
            for event_index, event in enumerate(events[:-1])
            if event.get("parent_tool_use_id") == agent_tool_use_id
            and _fixed32_qwen_user_text(event)
        ]
        descendant_event_indices = [
            event_index
            for event_index, event in enumerate(events[:-1])
            if isinstance(event.get("parent_tool_use_id"), str)
            and _fixed32_qwen_tool_descends_from(
                tool_use_records,
                event["parent_tool_use_id"],
                agent_tool_use_id,
            )
        ]

        if not prompt_indices:
            if descendant_event_indices or outer_result[1] is not True:
                raise ContractError(
                    "fixed32 qwen agent has no provable setup-error closure"
                )
            agent_sessions[agent_tool_use_id] = {
                "outer_result_index": outer_result_index,
                "prompt_index": None,
                "hidden": False,
            }
            continue
        allowed_fields = {
            "description",
            "isolation",
            "name",
            "prompt",
            "run_in_background",
            "subagent_type",
        }
        if not set(params) <= allowed_fields:
            raise ContractError(
                "fixed32 qwen agent input contains unknown fields"
            )
        for field in ("description", "prompt"):
            if (
                not isinstance(params.get(field), str)
                or not params[field].strip()
            ):
                raise ContractError(
                    f"fixed32 qwen agent {field} is empty or invalid"
                )
        if (
            "run_in_background" in params
            and type(params["run_in_background"]) is not bool
        ):
            raise ContractError(
                "fixed32 qwen agent background selector is invalid"
            )
        for field in ("isolation", "name", "subagent_type"):
            if field in params and not isinstance(params[field], str):
                raise ContractError(
                    f"fixed32 qwen agent {field} selector is invalid"
                )
        if (
            "subagent_type" in params
            and not params["subagent_type"].strip()
        ):
            raise ContractError(
                "fixed32 qwen agent subagent_type selector is invalid"
            )
        subagent_type = params.get("subagent_type")
        if params.get("run_in_background") is True or (
            isinstance(subagent_type, str)
            and subagent_type.strip().lower() == "fork"
        ):
            raise ContractError(
                "fixed32 qwen asynchronous agent invocation is unsupported"
            )
        if "isolation" in params:
            raise ContractError(
                "fixed32 qwen isolated agent invocation is unsupported"
            )
        if isinstance(params.get("name"), str) and params["name"]:
            raise ContractError(
                "fixed32 qwen teammate agent invocation is unsupported"
            )
        if len(prompt_indices) != 1:
            raise ContractError(
                "fixed32 qwen agent initial prompt is missing or duplicated"
            )
        prompt_index = prompt_indices[0]
        prompt_text = events[prompt_index]["message"]["content"][0]["text"]
        error_boundary = (
            nested_error_index is not None
            and outer_result_index == nested_error_index + 1
        )
        if (
            prompt_index <= origin["event_index"]
            or prompt_index >= outer_result_index
            or outer_result[1]
            or not isinstance(params.get("prompt"), str)
            or not params["prompt"].strip()
            or prompt_text != params["prompt"]
            or _fixed32_qwen_agent_outer_result_is_async(outer_result_event)
            or (
                _fixed32_qwen_agent_outer_result_is_failure(
                    outer_result_event
                )
                and not error_boundary
            )
        ):
            raise ContractError(
                "fixed32 qwen foreground agent closure is invalid"
            )

        if any(
            origin["event_index"] < event_index < prompt_index
            for event_index in descendant_event_indices
        ):
            raise ContractError(
                "fixed32 qwen agent activity precedes its initial prompt"
            )
        if any(
            event_index > outer_result_index
            for event_index in descendant_event_indices
        ):
            raise ContractError(
                "fixed32 qwen agent continues after its owner result"
            )

        for event_index in range(prompt_index, outer_result_index):
            if event_index == nested_error_index:
                if (
                    nested_error_parent_tool_use_id is None
                    or not _fixed32_qwen_tool_descends_from(
                        tool_use_records,
                        nested_error_parent_tool_use_id,
                        agent_tool_use_id,
                    )
                ):
                    raise ContractError(
                        "fixed32 qwen agent error boundary is not in its subtree"
                    )
                continue
            event_parent = events[event_index].get("parent_tool_use_id")
            if (
                not isinstance(event_parent, str)
                or not _fixed32_qwen_tool_descends_from(
                    tool_use_records,
                    event_parent,
                    agent_tool_use_id,
                )
            ):
                raise ContractError(
                    "fixed32 qwen agent session is not a serial subtree"
                )

        if (
            error_boundary
            and nested_error_parent_tool_use_id != agent_tool_use_id
        ):
            raise ContractError(
                "fixed32 qwen agent error boundary belongs to another tool"
            )
        agent_sessions[agent_tool_use_id] = {
            "outer_result_index": outer_result_index,
            "prompt_index": prompt_index,
            "error_boundary": error_boundary,
            # The stream exposes child tool rounds but returns the child's
            # final assistant text only through the successful owner result.
            "hidden": not error_boundary,
        }

    if nested_error_index is not None:
        boundary_origin = tool_use_records.get(
            nested_error_parent_tool_use_id or ""
        )
        if (
            (boundary_origin is None or boundary_origin["name"] != "agent")
            and events[nested_error_index + 1].get("parent_tool_use_id")
            is not None
        ):
            raise ContractError(
                "fixed32 qwen nested error boundary transition is invalid"
            )

    for agent_tool_use_id, session in agent_sessions.items():
        prompt_index = session["prompt_index"]
        if prompt_index is None:
            continue
        outer_result_index = session["outer_result_index"]
        cursor = prompt_index + 1
        nested_groups = groups_by_parent.get(agent_tool_use_id, [])
        for nested_group in nested_groups:
            if nested_group[0][3] != cursor:
                raise ContractError(
                    "fixed32 qwen agent response groups are not contiguous"
                )
            expected_tool_ids: list[str] = []
            for _event, message, _event_id, _event_index in nested_group:
                expected_tool_ids.extend(
                    item["id"]
                    for item in message["content"]
                    if item.get("type") == "tool_use"
                )
            if not expected_tool_ids:
                raise ContractError(
                    "fixed32 qwen agent response group has no tool call"
                )
            cursor = nested_group[-1][3] + 1
            for expected_tool_id in expected_tool_ids:
                expected_record = tool_use_records[expected_tool_id]
                if expected_record["name"] == "agent":
                    child_session = agent_sessions[expected_tool_id]
                    child_start = (
                        child_session["prompt_index"]
                        if child_session["prompt_index"] is not None
                        else child_session["outer_result_index"]
                    )
                    if child_start != cursor:
                        raise ContractError(
                            "fixed32 qwen nested agent transition is invalid"
                        )
                    cursor = child_session["outer_result_index"] + 1
                    continue
                if cursor >= outer_result_index:
                    raise ContractError(
                        "fixed32 qwen agent tool result is missing"
                    )
                tool_result_event = events[cursor]
                tool_result = _fixed32_qwen_user_tool_result(
                    tool_result_event
                )
                if (
                    tool_result_event.get("parent_tool_use_id")
                    != agent_tool_use_id
                    or tool_result is None
                    or tool_result[0] != expected_tool_id
                ):
                    raise ContractError(
                        "fixed32 qwen agent tool results do not reconcile"
                    )
                cursor += 1

        if session["error_boundary"]:
            if cursor != nested_error_index:
                raise ContractError(
                    "fixed32 qwen agent error transition is invalid"
                )
            cursor += 1
        if cursor != outer_result_index:
            raise ContractError(
                "fixed32 qwen agent terminal transition is invalid"
            )

    hidden_requests: list[tuple[int, str]] = []
    for agent_tool_use_id, session in agent_sessions.items():
        if not session["hidden"]:
            continue
        prompt_index = session["prompt_index"]
        outer_result_index = session["outer_result_index"]
        if prompt_index is None:
            raise ContractError(
                "fixed32 qwen hidden agent request has no initial prompt"
            )
        hidden_requests.append(
            (
                outer_result_index,
                _fixed32_qwen_hidden_agent_terminal_request_id(
                    agent_tool_use_id=agent_tool_use_id,
                    child_event_ids=[
                        events[event_index]["uuid"]
                        for event_index in range(
                            prompt_index,
                            outer_result_index,
                        )
                    ],
                    outer_tool_result_event_id=events[outer_result_index][
                        "uuid"
                    ],
                ),
            )
        )
    return hidden_requests


def _validate_fixed32_qwen_nested_error_boundary(
    events: list[dict[str, Any]],
    *,
    result_index: int,
    session_id: str,
    final_result_uuid: str,
) -> str:
    result = events[result_index]
    usage = result.get("usage")
    error = result.get("error")
    if (
        result.get("subtype") != "error_during_execution"
        or result.get("is_error") is not True
        or type(result.get("num_turns")) is not int
        or result["num_turns"] != 0
        or type(result.get("duration_ms")) is not int
        or result["duration_ms"] != 0
        or type(result.get("duration_api_ms")) is not int
        or result["duration_api_ms"] != 0
        or result.get("permission_denials") != []
        or result.get("session_id") != session_id
        or "result" in result
        or "parent_tool_use_id" in result
    ):
        raise ContractError(
            "fixed32 qwen nested error boundary state is invalid"
        )
    if (
        not isinstance(usage, dict)
        or set(usage) != {"input_tokens", "output_tokens"}
        or type(usage["input_tokens"]) is not int
        or usage["input_tokens"] != 0
        or type(usage["output_tokens"]) is not int
        or usage["output_tokens"] != 0
    ):
        raise ContractError(
            "fixed32 qwen nested error boundary usage is not zero"
        )
    if (
        not isinstance(error, dict)
        or set(error) != {"message"}
        or not isinstance(error["message"], str)
        or not error["message"].strip()
    ):
        raise ContractError(
            "fixed32 qwen nested error boundary message is invalid"
        )
    result_uuid = result.get("uuid")
    if (
        not isinstance(result_uuid, str)
        or not result_uuid
        or result_uuid == final_result_uuid
    ):
        raise ContractError(
            "fixed32 qwen nested/final result identities are invalid"
        )
    if result_index == 0 or result_index + 1 >= len(events) - 1:
        raise ContractError(
            "fixed32 qwen nested error boundary position is invalid"
        )
    nested_user = events[result_index - 1]
    top_level_user = events[result_index + 1]
    next_parent = top_level_user.get("parent_tool_use_id")
    if (
        nested_user.get("type") != "user"
        or not isinstance(nested_user.get("parent_tool_use_id"), str)
        or not nested_user["parent_tool_use_id"]
        or top_level_user.get("type") != "user"
        or "parent_tool_use_id" not in top_level_user
        or (
            next_parent is not None
            and (not isinstance(next_parent, str) or not next_parent)
        )
    ):
        raise ContractError(
            "fixed32 qwen nested error boundary transition is invalid"
        )
    return nested_user["parent_tool_use_id"]


def validate_fixed32_trace_model_requests(
    events: list[dict[str, Any]],
    *,
    expected_session_id: str | None = None,
    expected_completed_logical_model_requests: int | None = None,
    metrics_pre: bytes | None = None,
    metrics_post: bytes | None = None,
) -> dict[str, Any]:
    """Reconcile legacy terminals or pinned Qwen assistant response groups."""
    if not events or any(not isinstance(event, dict) for event in events):
        raise ContractError("fixed32 trace events must be nonempty objects")
    metric_arguments = (
        expected_completed_logical_model_requests,
        metrics_pre,
        metrics_post,
    )
    if any(value is not None for value in metric_arguments) and any(
        value is None for value in metric_arguments
    ):
        raise ContractError(
            "fixed32 qwen compaction metrics require count, pre, and post"
        )

    terminal_records: list[
        tuple[int, dict[str, Any], dict[str, Any], str]
    ] = []
    result_records: list[tuple[int, dict[str, Any]]] = []
    for index, event in enumerate(events):
        if event.get("type") == "result":
            result_records.append((index, event))
        message = _fixed32_trace_message(event)
        if message is None or message.get("stop_reason") is None:
            continue
        response_id = message.get("id")
        if (
            not isinstance(response_id, str)
            or not response_id
            or not isinstance(message.get("usage"), dict)
        ):
            raise ContractError(
                "fixed32 terminal assistant record lacks response ID or usage"
            )
        terminal_records.append((index, event, message, response_id))

    if not result_records:
        if metrics_pre is not None:
            raise ContractError(
                "fixed32 compaction metric evidence requires a Qwen result"
            )
        response_ids = [record[3] for record in terminal_records]
        if not response_ids or len(response_ids) != len(set(response_ids)):
            raise ContractError(
                "fixed32 legacy terminal response IDs are empty or duplicated"
            )
        return {
            "trace_format": "legacy_terminal_records",
            "completed_logical_model_requests": len(response_ids),
            "model_request_ids": response_ids,
            "hidden_terminal_model_requests": 0,
            "hidden_compaction_model_requests": 0,
            "engine_id_joinable": True,
        }

    if (
        len(result_records) > 2
        or result_records[-1][0] != len(events) - 1
    ):
        raise ContractError(
            "fixed32 qwen trace requires one final result and at most one "
            "nested error boundary"
        )
    result = result_records[-1][1]
    num_turns = result.get("num_turns")
    if (
        result.get("subtype") != "success"
        or result.get("is_error") is not False
        or type(num_turns) is not int
        or num_turns <= 0
    ):
        raise ContractError("fixed32 qwen result terminal state is invalid")
    for key in ("uuid", "session_id"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ContractError(f"fixed32 qwen result {key} is invalid")
    for key in ("duration_ms", "duration_api_ms"):
        value = result.get(key)
        if type(value) is not int or value < 0:
            raise ContractError(f"fixed32 qwen result {key} is invalid")
    if not isinstance(result.get("usage"), dict):
        raise ContractError("fixed32 qwen result evidence is incomplete")
    # PERMISSION DENIALS ARE NORMAL, and under the no-net agent settings they
    # are EXPECTED: qwen-code enforces the web_fetch deny rule against
    # equivalent shell commands, so a `curl https://...` comes back
    # "denied by permission rules" and lands here. Observed 2026-08-17 in
    # fr14_b1_stock_20260817T031507Z astropy__astropy-13236.
    #
    # A denial costs the request ledger NOTHING: the assistant's tool_use is
    # already counted in its own group, the denial is delivered as an ordinary
    # paired tool_result, and no model request is hidden behind it -- unlike
    # web_fetch, the denied tool never runs and never calls the model. So the
    # count is unaffected and the old `!= []` was refusing evidence it did not
    # need to refuse.
    #
    # Still fail-closed on SHAPE: each entry must name the tool it denied and
    # the tool_use it belongs to, and that tool_use must be one this trace
    # actually contains -- a denial referring to an unknown call would mean the
    # trace is not a complete record of its own session.
    denials = result.get("permission_denials")
    if not isinstance(denials, list):
        raise ContractError("fixed32 qwen result permission denials are invalid")
    denied_tool_use_ids: list[str] = []
    for denial in denials:
        if (
            not isinstance(denial, dict)
            or not isinstance(denial.get("tool_name"), str)
            or not denial["tool_name"]
            or not isinstance(denial.get("tool_use_id"), str)
            or not denial["tool_use_id"]
        ):
            raise ContractError(
                "fixed32 qwen result permission denial record is invalid"
            )
        denied_tool_use_ids.append(denial["tool_use_id"])

    result_session_id = result["session_id"]
    if (
        expected_session_id is not None
        and result_session_id != expected_session_id
    ):
        raise ContractError(
            "fixed32 qwen result session does not bind to the task"
        )

    nested_error_index: int | None = None
    nested_error_parent_tool_use_id: str | None = None
    if len(result_records) == 2:
        nested_error_index = result_records[0][0]
        nested_error_parent_tool_use_id = (
            _validate_fixed32_qwen_nested_error_boundary(
                events,
                result_index=nested_error_index,
                session_id=result_session_id,
                final_result_uuid=result["uuid"],
            )
        )

    qwen_event_ids = [event.get("uuid") for event in events]
    if (
        any(not isinstance(event_id, str) or not event_id for event_id in qwen_event_ids)
        or len(qwen_event_ids) != len(set(qwen_event_ids))
    ):
        raise ContractError(
            "fixed32 qwen event identities are empty or duplicated"
        )

    tool_use_ids: set[str] = set()
    assistant_groups: list[
        list[tuple[dict[str, Any], dict[str, Any], str, int]]
    ] = []
    tool_use_records: dict[str, dict[str, Any]] = {}
    previous_was_assistant = False
    for event_index, event in enumerate(events[:-1]):
        event_type = event.get("type")
        if event_type not in {"system", "user", "assistant", "result"}:
            raise ContractError(
                "fixed32 qwen pre-result event type is invalid"
            )
        if event.get("session_id") != result_session_id:
            raise ContractError(
                "fixed32 qwen pre-result session identity differs"
            )

        if event_type == "result":
            if event_index != nested_error_index:
                raise ContractError(
                    "fixed32 qwen pre-final result is not the nested error boundary"
                )
            previous_was_assistant = False
            continue

        parent_tool_use_id = event.get("parent_tool_use_id")
        if parent_tool_use_id is not None:
            if (
                not isinstance(parent_tool_use_id, str)
                or not parent_tool_use_id
            ):
                raise ContractError(
                    "fixed32 qwen parent tool identity is invalid"
                )
            if parent_tool_use_id not in tool_use_ids:
                raise ContractError(
                    "fixed32 qwen event has an unknown or non-ancestral parent tool"
                )

        if event_type != "assistant":
            previous_was_assistant = False
            continue
        message = _fixed32_trace_message(event)
        if message is None:
            raise ContractError("fixed32 qwen assistant record is malformed")
        event_id = event.get("uuid")
        if (
            not isinstance(event_id, str)
            or not event_id
            or message.get("id") != event_id
            or not isinstance(message.get("usage"), dict)
        ):
            raise ContractError(
                "fixed32 qwen assistant session or event identity differs"
            )
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise ContractError(
                "fixed32 qwen assistant content is empty or invalid"
            )
        if _fixed32_qwen_api_error_record(message):
            raise ContractError(
                "fixed32 qwen assistant record is a client-side API error "
                "banner, so the trace cannot say whether the engine served it"
            )
        event_tool_ids: list[str] = []
        event_tool_id_set: set[str] = set()
        for item in content:
            if not isinstance(item, dict):
                raise ContractError(
                    "fixed32 qwen assistant content item is invalid"
                )
            if item.get("type") != "tool_use":
                continue
            tool_id = item.get("id")
            if (
                not isinstance(tool_id, str)
                or not tool_id
                or tool_id in tool_use_ids
                or tool_id in event_tool_id_set
            ):
                raise ContractError(
                    "fixed32 qwen tool-use identity is empty or duplicated"
                )
            event_tool_ids.append(tool_id)
            event_tool_id_set.add(tool_id)
            tool_use_records[tool_id] = {
                "event_index": event_index,
                "name": item.get("name"),
                "input": item.get("input"),
                "parent_tool_use_id": parent_tool_use_id,
            }
        stop_reason = message.get("stop_reason")
        if stop_reason not in {None, "tool_use"}:
            raise ContractError(
                "fixed32 qwen assistant stop reason is invalid"
            )
        if (stop_reason == "tool_use") != bool(event_tool_ids):
            raise ContractError(
                "fixed32 qwen tool-use terminal/content evidence differs"
            )
        tool_use_ids.update(event_tool_ids)

        record = (event, message, event_id, event_index)
        if previous_was_assistant:
            assistant_groups[-1].append(record)
        else:
            assistant_groups.append([record])
        previous_was_assistant = True

    if not assistant_groups or events[-2].get("type") != "assistant":
        raise ContractError(
            "fixed32 qwen trace has no final assistant response group"
        )

    top_level_groups: list[
        list[tuple[dict[str, Any], dict[str, Any], str, int]]
    ] = []
    request_records: list[tuple[int, str]] = []
    synthetic_compaction_failure_terminal = False
    for group_index, group in enumerate(assistant_groups):
        parent_ids = {record[0].get("parent_tool_use_id") for record in group}
        if len(parent_ids) != 1:
            raise ContractError(
                "fixed32 qwen contiguous assistant group changes parent identity"
            )
        parent_tool_use_id = next(iter(parent_ids))
        if parent_tool_use_id is None:
            top_level_groups.append(group)

        terminal_seen = False
        terminal_count = 0
        nonempty_text_count = 0
        for _event, message, _event_id, _event_index in group:
            if message.get("stop_reason") == "tool_use":
                terminal_seen = True
                terminal_count += 1
            elif terminal_seen:
                raise ContractError(
                    "fixed32 qwen assistant group continues after a terminal record"
                )
            if _fixed32_nonempty_text_record(message):
                nonempty_text_count += 1

        is_final_group = group_index == len(assistant_groups) - 1
        if is_final_group:
            # A final group is canonical when it closes on exactly one
            # nonempty text record. Qwen may instead close on a
            # reasoning-only turn, whose records carry ``thinking`` blocks
            # and nothing else; that turn was still served by the engine, so
            # it is accepted here and counted below like any other group.
            #
            # That closing turn may also trail its reasoning with a
            # whitespace-only text record -- observed live as ``"\n\n"``
            # ending a 434-event trajectory that had already produced a real
            # 2216-byte patch. The blank record carries no visible text, so
            # it still cannot be read as a submission, and the reasoning in
            # the same group remains the positive evidence that the engine
            # served the turn. A final group carrying no reasoning at all
            # stays invalid: a bare whitespace record on its own is
            # indistinguishable from a degenerate empty response.
            silent_records = [
                message
                for _event, message, _event_id, _event_index in group
            ]
            reasoning_only_final_group = nonempty_text_count == 0 and (
                any(
                    _fixed32_qwen_reasoning_only_record(message)
                    for message in silent_records
                )
                and all(
                    _fixed32_qwen_reasoning_only_record(message)
                    or _fixed32_qwen_blank_text_record(message)
                    for message in silent_records
                )
            )
            if (
                parent_tool_use_id is not None
                or terminal_count != 0
                or (nonempty_text_count != 1 and not reasoning_only_final_group)
            ):
                raise ContractError(
                    "fixed32 qwen final assistant response group is invalid"
                )
            synthetic_compaction_failure_terminal = (
                _fixed32_qwen_synthetic_compaction_failure(
                    group,
                    result=result,
                )
            )
        elif terminal_count == 0:
            raise ContractError(
                "fixed32 qwen non-final assistant response group is incomplete"
            )

        if not synthetic_compaction_failure_terminal:
            request_records.append(
                (
                    group[0][3],
                    _fixed32_qwen_group_request_id(
                        [record[2] for record in group]
                    ),
                )
            )

    # PERMISSION-DENIAL JOIN. Deferred to here because tool_use_ids is only
    # complete after the collection loop above. A denial that names a tool_use
    # this trace does not contain would mean the trace is not a complete record
    # of its own session -- exactly the condition that makes an independent
    # count meaningless -- so it fails closed even though a denial costs the
    # ledger nothing.
    for denied_tool_use_id in denied_tool_use_ids:
        if denied_tool_use_id not in tool_use_ids:
            raise ContractError(
                "fixed32 qwen result permission denial names an unknown "
                "tool use"
            )

    if len(top_level_groups) != num_turns:
        raise ContractError(
            "fixed32 qwen result turn count and top-level response groups "
            "do not reconcile"
        )
    hidden_requests = _fixed32_qwen_hidden_agent_terminal_requests(
        events,
        assistant_groups=assistant_groups,
        tool_use_records=tool_use_records,
        nested_error_index=nested_error_index,
        nested_error_parent_tool_use_id=(
            nested_error_parent_tool_use_id
        ),
    )
    request_records.extend(hidden_requests)
    # Tool-internal model calls. These are ORDINARY 32768-max_tokens requests,
    # not compactions, so they join request_records before the compaction split
    # below and land in normal_request_count where the algebra expects them.
    hidden_web_fetch_requests = _fixed32_qwen_hidden_web_fetch_requests(
        events,
        tool_use_records=tool_use_records,
    )
    request_records.extend(hidden_web_fetch_requests)
    hidden_compaction_requests = _fixed32_qwen_hidden_compaction_requests(
        events,
        top_level_groups=top_level_groups,
    )
    request_records.extend(hidden_compaction_requests)
    failed_compaction_requests: list[tuple[int, str]] = []
    compaction_metric_evidence: dict[str, Any] | None = None
    if metrics_pre is not None:
        normal_request_count = (
            len(request_records) - len(hidden_compaction_requests)
        )
        (
            compaction_metric_evidence,
            failed_compaction_count,
        ) = _fixed32_qwen_compaction_metric_evidence(
            events=events,
            result=result,
            normal_request_count=normal_request_count,
            successful_compaction_count=len(hidden_compaction_requests),
            synthetic_compaction_failure_terminal=(
                synthetic_compaction_failure_terminal
            ),
            unobservable_compaction_boundaries=(
                _fixed32_qwen_unobservable_compaction_boundaries(
                    assistant_groups
                )
            ),
            expected_completed_logical_model_requests=(
                expected_completed_logical_model_requests
            ),
            metrics_pre=metrics_pre,
            metrics_post=metrics_post,
        )
        existing_request_count = len(request_records)
        if (
            existing_request_count + failed_compaction_count
            != expected_completed_logical_model_requests
        ):
            raise ContractError(
                "fixed32 qwen metric-proven request count does not reconcile"
            )
        evidence_sha256 = hashlib.sha256(
            json.dumps(
                compaction_metric_evidence,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        event_ids_sha256 = hashlib.sha256(
            json.dumps(
                qwen_event_ids,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        failed_compaction_requests = [
            (
                len(events) - 1,
                _fixed32_qwen_hidden_failed_compaction_request_id(
                    result_event_id=result["uuid"],
                    trace_event_ids_sha256=event_ids_sha256,
                    metric_evidence_sha256=evidence_sha256,
                    ordinal=ordinal,
                ),
            )
            for ordinal in range(failed_compaction_count)
        ]
        request_records.extend(failed_compaction_requests)
    request_records.sort(key=lambda record: record[0])
    response_ids = [record[1] for record in request_records]
    if len(response_ids) != len(set(response_ids)):
        raise ContractError(
            "fixed32 qwen response group identities are duplicated"
        )
    return {
        "trace_format": "qwen_result",
        "completed_logical_model_requests": len(response_ids),
        "model_request_ids": response_ids,
        "hidden_terminal_model_requests": len(hidden_requests),
        "hidden_web_fetch_model_requests": len(hidden_web_fetch_requests),
        "hidden_compaction_model_requests": (
            len(hidden_compaction_requests)
            + len(failed_compaction_requests)
        ),
        "hidden_successful_compaction_model_requests": len(
            hidden_compaction_requests
        ),
        "hidden_failed_compaction_model_requests": len(
            failed_compaction_requests
        ),
        "synthetic_compaction_failure_terminal": (
            synthetic_compaction_failure_terminal
        ),
        "qwen_compaction_metric_evidence": compaction_metric_evidence,
        "engine_id_joinable": False,
    }


def fixed32_ingress_ledger_token_usage(
    raw: bytes,
    *,
    role: str = "proxy",
) -> dict[str, Any]:
    """Sum tamper-evident per-request token usage off a fixed32 ingress ledger.

    THE METER THIS REPLACES. Campaign token reconciliation used to close
    against qwen-code's self-reported ``result.usage``. That is a third-party
    agent's own accounting and it under-credits its own hidden requests: a
    rejected compaction the engine billed and the agent discarded, a retried
    first turn, a delegated sub-agent that reports ``0/0`` on every record.
    On the 2026-08-15 width-4 screen that cost 189,780 prompt and 5,654
    generation tokens across 3 of 32 task-instances -- deterministic failure
    at n=16, and pure luck at n=4.

    The ledger is the other meter: the proxy terminates every completion, so
    it records the ENGINE's own count per request, on the same SHA-256 chain
    that carries the request identities. Editing a token count invalidates
    the chain from that row onward.

    FAIL-CLOSED ON EMPTINESS. ``vllm_request_metrics.jsonl`` recorded nothing
    for months and no gate noticed, because "absent" and "empty" were
    indistinguishable from "fine". Here they are not: a ledger written before
    the usage fields existed carries the keys on NO row (``absent``, and the
    caller may fall back); a ledger written by a proxy that has them carries
    them on EVERY row (``present``), and if none of its completions is metered
    that is a raised ContractError, never a silent fallback.
    """
    if role not in _FIXED32_LEDGER_COMPLETION_EVENTS:
        raise ContractError("fixed32 ingress ledger role is invalid")
    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise ContractError(
            "fixed32 ingress ledger is empty or truncated"
        )
    completion_event = _FIXED32_LEDGER_COMPLETION_EVENTS[role]

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(
                    f"fixed32 ingress ledger row has duplicate key {key!r}"
                )
            result[key] = value
        return result

    rows: list[dict[str, Any]] = []
    try:
        for line in raw.decode("utf-8").splitlines():
            row = json.loads(line, object_pairs_hook=reject_duplicate_pairs)
            if not isinstance(row, dict):
                raise ContractError("fixed32 ingress ledger row is not an object")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"fixed32 ingress ledger JSONL is invalid: {error}"
        ) from error
    if not rows:
        raise ContractError("fixed32 ingress ledger is empty or truncated")

    usage_present = all(key in rows[0] for key in _FIXED32_LEDGER_USAGE_KEYS)
    previous = "0" * 64
    prompt_tokens = 0
    completion_tokens = 0
    usage_rows = 0
    completion_rows = 0
    unmetered_completions = 0
    for sequence, row in enumerate(rows):
        present = [key in row for key in _FIXED32_LEDGER_USAGE_KEYS]
        if any(present) != all(present) or all(present) is not usage_present:
            # No writer produces a ledger whose rows disagree about the
            # schema. One that does was assembled, not recorded.
            raise ContractError(
                "fixed32 ingress ledger mixes token usage schemas"
            )
        claimed = row.get("record_sha256")
        unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if (
            row.get("schema") != FIXED32_INGRESS_LEDGER_RECORD_SCHEMA
            or type(row.get("seq")) is not int
            or row["seq"] != sequence
            or row.get("role") != role
            or row.get("prev_sha256") != previous
            or not isinstance(claimed, str)
            or hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            != claimed
        ):
            raise ContractError(
                f"fixed32 ingress ledger chain differs at record {sequence}"
            )
        previous = claimed
        event = row.get("event")
        is_completion = event == completion_event
        if is_completion:
            completion_rows += 1
        if not usage_present:
            continue
        values = [row[key] for key in _FIXED32_LEDGER_USAGE_KEYS]
        metered = [value is not None for value in values]
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            )
            for value in values
        ):
            raise ContractError(
                f"fixed32 ingress ledger token usage is invalid at record "
                f"{sequence}"
            )
        if any(metered) and not is_completion:
            raise ContractError(
                f"fixed32 ingress ledger meters a non-completion event at "
                f"record {sequence}"
            )
        if any(metered) != all(metered):
            raise ContractError(
                f"fixed32 ingress ledger token usage is half-recorded at "
                f"record {sequence}"
            )
        if all(metered):
            usage_rows += 1
            prompt_tokens += values[0]
            completion_tokens += values[1]
        elif is_completion and row.get("outcome") == "completed":
            unmetered_completions += 1

    if usage_present and completion_rows and not usage_rows:
        raise ContractError(
            "fixed32 ingress ledger records no token usage: the meter wrote "
            f"{completion_rows} completions and metered none of them"
        )
    if usage_present and unmetered_completions:
        raise ContractError(
            "fixed32 ingress ledger token usage is incomplete: "
            f"{unmetered_completions} of {completion_rows} completions carry "
            "no token counts"
        )
    return {
        "schema": QWEN_CAMPAIGN_TOKEN_LEDGER_SCHEMA,
        "role": role,
        "records": len(rows),
        "chain_head_sha256": previous,
        "token_usage_schema": "present" if usage_present else "absent",
        "completion_records": completion_rows,
        "token_usage_records": usage_rows,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": completion_tokens,
    }


def validate_fixed32_qwen_campaign_metrics(
    tasks: list[dict[str, Any]],
    *,
    metrics_pre: bytes,
    metrics_post: bytes,
    ingress_ledger: bytes | None = None,
    ingress_ledger_role: str = "proxy",
) -> dict[str, Any]:
    """Reconcile one global Prometheus window across concurrent Qwen tasks."""
    if not isinstance(tasks, list) or len(tasks) < 2:
        raise ContractError(
            "fixed32 qwen campaign metrics require at least two tasks"
        )
    expected_task_keys = {
        "instance_id",
        "expected_session_id",
        "expected_completed_logical_model_requests",
        "events",
        # ADDED 2026-08-13. Required, not optional: the completion algebra now
        # has a conditional class whose legality depends on this declaration, so
        # a caller that forgets it must fail loud rather than silently declare
        # zero caps and turn a real abort into a refusal nobody can explain.
        "budget_capped",
    }
    task_inputs: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    capped_requests = 0
    capped_instance_ids: list[str] = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) != expected_task_keys:
            raise ContractError(
                "fixed32 qwen campaign task input is not exact"
            )
        instance_id = task["instance_id"]
        expected_session_id = task["expected_session_id"]
        completed = task["expected_completed_logical_model_requests"]
        events = task["events"]
        budget_capped = task["budget_capped"]
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in seen_instance_ids
            or expected_session_id != fixed32_trace_session_id(instance_id)
            or type(completed) is not int
            or completed <= 0
            or not isinstance(events, list)
            or not isinstance(budget_capped, bool)
        ):
            raise ContractError(
                "fixed32 qwen campaign task identity or count is invalid"
            )
        seen_instance_ids.add(instance_id)
        if budget_capped:
            capped_requests += 1
            capped_instance_ids.append(instance_id)
        task_inputs.append(task)
    task_inputs.sort(key=lambda task: task["instance_id"])

    before = _fixed32_qwen_metric_snapshot(metrics_pre, label="campaign pre")
    after = _fixed32_qwen_metric_snapshot(metrics_post, label="campaign post")
    deltas: dict[str, int] = {}
    for key in sorted(before):
        if after[key] < before[key]:
            raise ContractError(
                f"fixed32 qwen metric {key} decreased across campaign"
            )
        deltas[key] = after[key] - before[key]

    analyses: dict[str, dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    completed_total = 0
    normal_total = 0
    successful_compaction_total = 0
    failed_compaction_total = 0
    result_prompt_total = 0
    result_generation_total = 0
    visible_prompt_total = 0
    visible_generation_total = 0
    for task in task_inputs:
        instance_id = task["instance_id"]
        events = task["events"]
        base = validate_fixed32_trace_model_requests(
            events,
            expected_session_id=task["expected_session_id"],
        )
        if base.get("trace_format") != "qwen_result":
            raise ContractError(
                "fixed32 qwen campaign task trace is not a Qwen result"
            )
        expected_completed = task[
            "expected_completed_logical_model_requests"
        ]
        base_completed = base["completed_logical_model_requests"]
        successful_compactions = base.get(
            "hidden_successful_compaction_model_requests",
            base.get("hidden_compaction_model_requests", 0),
        )
        failed_compactions = expected_completed - base_completed
        normal_requests = base_completed - successful_compactions
        # A failed compaction may be absent from the task trace. Its task-auth
        # gap is admitted here only if the campaign algebra below proves it.
        if (
            type(base_completed) is not int
            or type(successful_compactions) is not int
            or successful_compactions < 0
            or normal_requests <= 0
            or failed_compactions < 0
        ):
            raise ContractError(
                "fixed32 qwen campaign trace/task-auth counts do not reconcile"
            )

        result = events[-1]
        result_usage = result.get("usage")
        if not isinstance(result_usage, dict):
            raise ContractError("fixed32 qwen campaign result usage is missing")
        aggregate_input = result_usage.get("input_tokens")
        aggregate_output = result_usage.get("output_tokens")
        aggregate_total = result_usage.get("total_tokens")
        if (
            type(aggregate_input) is not int
            or aggregate_input < 0
            or type(aggregate_output) is not int
            or aggregate_output < 0
            or type(aggregate_total) is not int
            or aggregate_total != aggregate_input + aggregate_output
        ):
            raise ContractError(
                "fixed32 qwen campaign aggregate token usage is invalid"
            )
        visible_input = 0
        visible_output = 0
        for event in events:
            message = _fixed32_trace_message(event)
            if message is None:
                continue
            usage = message.get("usage")
            if not isinstance(usage, dict):
                raise ContractError(
                    "fixed32 qwen campaign assistant usage is missing"
                )
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if (
                type(input_tokens) is not int
                or input_tokens < 0
                or type(output_tokens) is not int
                or output_tokens < 0
            ):
                raise ContractError(
                    "fixed32 qwen campaign assistant token usage is invalid"
                )
            visible_input += input_tokens
            visible_output += output_tokens
        hidden_input = aggregate_input - visible_input
        hidden_output = aggregate_output - visible_output
        total_compactions = successful_compactions + failed_compactions
        if (
            hidden_input < 0
            or hidden_output < 0
            or (
                total_compactions > 0
                and (hidden_input <= 0 or hidden_output <= 0)
            )
        ):
            raise ContractError(
                "fixed32 qwen campaign hidden compaction token usage is invalid"
            )

        event_ids_sha256 = hashlib.sha256(
            json.dumps(
                [event["uuid"] for event in events],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        base_request_ids_sha256 = hashlib.sha256(
            json.dumps(
                base["model_request_ids"],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        row = {
            "instance_id": instance_id,
            "expected_completed_engine_requests": expected_completed,
            "trace_completed_requests_before_failed_compactions": (
                base_completed
            ),
            "normal_requests": normal_requests,
            "successful_compaction_requests": successful_compactions,
            "failed_compaction_requests": failed_compactions,
            "total_compaction_requests": total_compactions,
            "result_prompt_tokens": aggregate_input,
            "result_generation_tokens": aggregate_output,
            "visible_prompt_tokens": visible_input,
            "visible_generation_tokens": visible_output,
            "hidden_prompt_tokens": hidden_input,
            "hidden_generation_tokens": hidden_output,
            "trace_event_ids_sha256": event_ids_sha256,
            "base_model_request_ids_sha256": base_request_ids_sha256,
            "synthetic_compaction_failure_terminal": base.get(
                "synthetic_compaction_failure_terminal",
                False,
            ),
        }
        task_rows.append(row)
        analyses[instance_id] = {
            "base": base,
            "events": events,
            "result": result,
            "row": row,
        }
        completed_total += expected_completed
        normal_total += normal_requests
        successful_compaction_total += successful_compactions
        failed_compaction_total += failed_compactions
        result_prompt_total += aggregate_input
        result_generation_total += aggregate_output
        visible_prompt_total += visible_input
        visible_generation_total += visible_output

    total_compactions = (
        successful_compaction_total + failed_compaction_total
    )
    completion_classes = _fixed32_qwen_completion_classes(
        deltas,
        completed=completed_total,
        scope="campaign",
        capped_requests=capped_requests,
    )
    if deltas["max_tokens_le_10000"] != 0:
        raise ContractError(
            "fixed32 qwen campaign max-token histogram has an unpinned low request"
        )
    # A budget-capped abort was admitted with one of the two pinned max_tokens
    # values, and which one is not knowable from the task record -- the agent was
    # killed mid-request and never reported. It is knowable from the meters:
    # the le_20000 bucket over-counts by exactly the number of capped requests
    # that were compactions. Solve for that split and then require the token SUM
    # to reconcile exactly against it. No slack: a split that does not close the
    # sum identity to the token is refused.
    capped_compaction = deltas["max_tokens_le_20000"] - total_compactions
    capped_visible = capped_requests - capped_compaction
    if (
        capped_compaction < 0
        or capped_visible < 0
        or normal_total + total_compactions != completed_total
        or deltas["max_tokens_sum"]
        != (
            (normal_total + capped_visible) * QWEN_VISIBLE_MAX_OUTPUT_TOKENS
            + (total_compactions + capped_compaction)
            * QWEN_COMPACTION_MAX_OUTPUT_TOKENS
        )
    ):
        raise ContractError(
            "fixed32 qwen campaign 32768/20000 max-token algebra does not "
            f"reconcile: normal={normal_total} compactions={total_compactions} "
            f"completed={completed_total} capped={capped_requests} "
            f"le_20000={deltas['max_tokens_le_20000']} "
            f"sum={deltas['max_tokens_sum']} "
            f"(capped split visible={capped_visible} "
            f"compaction={capped_compaction})"
        )
    # ---------------------------------------------------------------- #
    # Which meter closes the token identity                             #
    # ---------------------------------------------------------------- #
    # The branch is decided by FIELD PRESENCE in the ingress ledger, never by
    # task count. A ledger written before the usage fields existed takes the
    # qwen-trace path so every historical artifact still validates, byte for
    # byte. A ledger written by a proxy that has them takes the ledger-sum
    # path -- and if that ledger metered nothing, the reader above has already
    # raised, because a meter recording nothing must fail the audit rather
    # than fall back to the meter it was built to replace.
    ledger_usage: dict[str, Any] | None = None
    if ingress_ledger is not None:
        ledger_usage = fixed32_ingress_ledger_token_usage(
            ingress_ledger,
            role=ingress_ledger_role,
        )
        if ledger_usage["token_usage_schema"] != "present":
            ledger_usage = None
    if ledger_usage is not None:
        if (
            deltas["prompt_tokens"] != ledger_usage["prompt_tokens"]
            or deltas["generation_tokens"] != ledger_usage["generation_tokens"]
        ):
            raise ContractError(
                "fixed32 qwen campaign ingress ledger and vLLM token usage do "
                "not reconcile: vLLM "
                f"{deltas['prompt_tokens']}/{deltas['generation_tokens']} vs "
                f"ledger {ledger_usage['prompt_tokens']}/"
                f"{ledger_usage['generation_tokens']} over "
                f"{ledger_usage['token_usage_records']} metered requests"
            )
    elif (
        deltas["prompt_tokens"] != result_prompt_total
        or deltas["generation_tokens"] != result_generation_total
    ):
        raise ContractError(
            "fixed32 qwen campaign aggregate and vLLM token usage do not reconcile"
        )

    metric_evidence = {
        "schema": QWEN_CAMPAIGN_METRIC_SCHEMA,
        "metrics_pre_sha256": hashlib.sha256(metrics_pre).hexdigest(),
        "metrics_post_sha256": hashlib.sha256(metrics_post).hexdigest(),
        "task_count": len(task_rows),
        "task_ids": [row["instance_id"] for row in task_rows],
        "completed_engine_requests": completed_total,
        "normal_visible_max_output_tokens": QWEN_VISIBLE_MAX_OUTPUT_TOKENS,
        "compaction_max_output_tokens": QWEN_COMPACTION_MAX_OUTPUT_TOKENS,
        "normal_requests": normal_total,
        "successful_compaction_requests": successful_compaction_total,
        "failed_compaction_requests": failed_compaction_total,
        "total_compaction_requests": total_compactions,
        "max_tokens_count": deltas["max_tokens_count"],
        "max_tokens_sum": deltas["max_tokens_sum"],
        "max_tokens_le_10000": deltas["max_tokens_le_10000"],
        "max_tokens_le_20000": deltas["max_tokens_le_20000"],
        "max_tokens_le_50000": deltas["max_tokens_le_50000"],
        "max_tokens_le_inf": deltas["max_tokens_le_inf"],
        "request_success_stop": completion_classes["stop"],
        # Truncated-at-max_tokens completions, counted rather than absorbed.
        # Published so a reader can see how much of a campaign's traffic ran to
        # its output cap; error/repetition are proven zero by
        # _fixed32_qwen_completion_classes, so length + abort IS the whole
        # non-stop remainder and request_success_non_stop stays exact.
        "request_success_length": completion_classes["length"],
        # Budget-capped aborts. Zero unless the campaign declared a cap, and
        # then exactly one per capped task -- the count is cross-checked against
        # the per-task runner records, not merely tolerated.
        "request_success_abort": completion_classes[
            QWEN_CAPPED_COMPLETION_REASON
        ],
        "request_success_non_stop": (
            completion_classes["length"]
            + completion_classes[QWEN_CAPPED_COMPLETION_REASON]
        ),
        "budget_capped_tasks": capped_requests,
        "budget_capped_task_ids": sorted(capped_instance_ids),
        # How each capped abort's max_tokens was admitted, solved from the
        # le_20000 bucket and then required to close the token sum exactly.
        "budget_capped_visible_requests": capped_visible,
        "budget_capped_compaction_requests": capped_compaction,
        "prompt_tokens": deltas["prompt_tokens"],
        "generation_tokens": deltas["generation_tokens"],
        "visible_prompt_tokens": visible_prompt_total,
        "visible_generation_tokens": visible_generation_total,
        "hidden_prompt_tokens": result_prompt_total - visible_prompt_total,
        "hidden_generation_tokens": (
            result_generation_total - visible_generation_total
        ),
        "tasks": task_rows,
    }
    # Which meter closed the identity is part of the claim, so it is published
    # INSIDE the hashed evidence -- and with it the gap the agent's own report
    # would have left, so the under-crediting is on the record rather than
    # merely absorbed. Added only on the ledger path: on the compatibility
    # path the evidence, and therefore its digest, stays byte-exact with every
    # proof written before this existed.
    token_reconciliation: dict[str, Any] = {
        "basis": QWEN_TOKEN_BASIS_TRACE,
        "vllm_prompt_tokens": deltas["prompt_tokens"],
        "vllm_generation_tokens": deltas["generation_tokens"],
        "qwen_trace_prompt_tokens": result_prompt_total,
        "qwen_trace_generation_tokens": result_generation_total,
        "qwen_trace_prompt_token_gap": (
            deltas["prompt_tokens"] - result_prompt_total
        ),
        "qwen_trace_generation_token_gap": (
            deltas["generation_tokens"] - result_generation_total
        ),
    }
    if ledger_usage is not None:
        token_reconciliation.update(
            {
                "basis": QWEN_TOKEN_BASIS_LEDGER,
                "ledger_schema": ledger_usage["schema"],
                "ledger_role": ledger_usage["role"],
                "ledger_records": ledger_usage["records"],
                "ledger_chain_head_sha256": ledger_usage["chain_head_sha256"],
                "ledger_completion_records": ledger_usage[
                    "completion_records"
                ],
                "ledger_token_usage_records": ledger_usage[
                    "token_usage_records"
                ],
                "ledger_prompt_tokens": ledger_usage["prompt_tokens"],
                "ledger_generation_tokens": ledger_usage["generation_tokens"],
            }
        )
        metric_evidence["token_reconciliation"] = dict(token_reconciliation)
    metric_evidence_sha256 = hashlib.sha256(
        json.dumps(
            metric_evidence,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    task_results: dict[str, dict[str, Any]] = {}
    campaign_request_ids: set[str] = set()
    for instance_id in metric_evidence["task_ids"]:
        analysis = analyses[instance_id]
        base = analysis["base"]
        row = analysis["row"]
        failed_request_ids = [
            _fixed32_qwen_hidden_failed_compaction_request_id(
                result_event_id=analysis["result"]["uuid"],
                trace_event_ids_sha256=row["trace_event_ids_sha256"],
                metric_evidence_sha256=metric_evidence_sha256,
                ordinal=ordinal,
            )
            for ordinal in range(row["failed_compaction_requests"])
        ]
        request_ids = [*base["model_request_ids"], *failed_request_ids]
        if (
            len(request_ids) != row["expected_completed_engine_requests"]
            or len(request_ids) != len(set(request_ids))
            or campaign_request_ids.intersection(request_ids)
        ):
            raise ContractError(
                "fixed32 qwen campaign request identities do not reconcile"
            )
        campaign_request_ids.update(request_ids)
        task_metric_evidence = {
            "schema": QWEN_CAMPAIGN_TASK_METRIC_SCHEMA,
            "campaign_metric_evidence_sha256": metric_evidence_sha256,
            **{
                key: value
                for key, value in row.items()
                if key != "instance_id"
            },
        }
        task_results[instance_id] = {
            **base,
            "completed_logical_model_requests": len(request_ids),
            "model_request_ids": request_ids,
            "hidden_compaction_model_requests": row[
                "total_compaction_requests"
            ],
            "hidden_successful_compaction_model_requests": row[
                "successful_compaction_requests"
            ],
            "hidden_failed_compaction_model_requests": row[
                "failed_compaction_requests"
            ],
            "qwen_compaction_metric_evidence": task_metric_evidence,
            "qwen_campaign_metric_evidence_sha256": (
                metric_evidence_sha256
            ),
        }
    return {
        "schema": QWEN_CAMPAIGN_METRIC_SCHEMA,
        "metric_evidence": metric_evidence,
        "metric_evidence_sha256": metric_evidence_sha256,
        "tasks": task_results,
        # Always returned, on both paths, so a caller can see which meter ran
        # without reaching into the hashed evidence to find out.
        "token_reconciliation": token_reconciliation,
    }


def _exact_file_record(path: Path, *, display_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"required regular file is missing or symlinked: {path}")
    stat = path.stat()
    return {
        "path": display_path,
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"{path}: non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{path}: JSON root must be an object")
    return payload


def _pinned_model_files(
    model_root: Path,
    *,
    expected_files: tuple[str, ...] = MODEL_FILES,
    expected_digest: str = MODEL_CANONICAL_SHA256,
    expected_vocab_size: int = MODEL_TEXT_CONFIG_VOCAB_SIZE,
    expected_records: tuple[tuple[str, int, str], ...] | None = MODEL_FILE_RECORDS,
    expected_vocab_json_sha256: str = MODEL_VOCAB_JSON_SHA256,
) -> tuple[list[dict[str, Any]], str]:
    if not model_root.is_dir() or model_root.is_symlink():
        raise ContractError(f"model root is missing or symlinked: {model_root}")
    actual_model_names = tuple(
        sorted(path.name for path in model_root.iterdir() if path.is_file())
    )
    if actual_model_names != expected_files:
        raise ContractError(
            "model file set differs from the fixed32 contract: "
            f"{actual_model_names} != {expected_files}"
        )

    config = _strict_json_object(model_root / "config.json")
    text_config = config.get("text_config")
    vocab_size = (
        text_config.get("vocab_size") if isinstance(text_config, dict) else None
    )
    if type(vocab_size) is not int or vocab_size != expected_vocab_size:
        raise ContractError(
            "model config text_config.vocab_size mismatch: "
            f"{vocab_size!r} != {expected_vocab_size}"
        )
    # FR14: vocab_size is a WIDTH check, not an IDENTITY check. The K64
    # draft-vocab block map indexes lm_head rows by token id, so a reordered
    # vocabulary of the same width would pass every boot assertion and only
    # show up as a degraded accept rate. vocab.json is byte-identical across
    # the 3.6 and 3.8 checkpoints, so pin its digest right here, beside the
    # width check, where a model swap cannot route around it.
    vocab_json_sha256 = sha256_file(model_root / "vocab.json")
    if vocab_json_sha256 != expected_vocab_json_sha256:
        raise ContractError(
            "model tokenizer vocab.json digest mismatch (token id mapping "
            "moved; the K64 draft-vocab block map is no longer valid): "
            f"{vocab_json_sha256} != {expected_vocab_json_sha256}"
        )

    model_files = [
        _exact_file_record(model_root / name, display_path=name)
        for name in expected_files
    ]
    if expected_records is not None:
        pinned_records = [
            {"path": path, "size": size, "sha256": sha256}
            for path, size, sha256 in expected_records
        ]
        if model_files != pinned_records:
            mismatch = next(
                (
                    (observed, pinned)
                    for observed, pinned in zip(
                        model_files, pinned_records, strict=False
                    )
                    if observed != pinned
                ),
                None,
            )
            raise ContractError(f"model file identity mismatch: {mismatch!r}")
    model_digest = hashlib.sha256(canonical_bytes(model_files)).hexdigest()
    if model_digest != expected_digest:
        raise ContractError(
            f"model canonical digest mismatch: {model_digest} != {expected_digest}"
        )
    return model_files, model_digest


def _docker_image_record() -> dict[str, Any]:
    proc = subprocess.run(
        ["docker", "image", "inspect", IMAGE_REFERENCE],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ContractError(
            f"cannot inspect pinned image: rc={proc.returncode} stderr={proc.stderr!r}"
        )
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise ContractError("docker image inspect returned invalid JSON") from error
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ContractError("docker image inspect did not return exactly one image")
    row = rows[0]
    record = {
        "reference": IMAGE_REFERENCE,
        "id": row.get("Id"),
        "repo_digests": sorted(row.get("RepoDigests") or []),
        "os": row.get("Os"),
        "architecture": row.get("Architecture"),
    }
    expected = {
        "reference": IMAGE_REFERENCE,
        "id": IMAGE_ID,
        "repo_digests": [IMAGE_REFERENCE],
        "os": IMAGE_OS,
        "architecture": IMAGE_ARCHITECTURE,
    }
    if record != expected:
        raise ContractError(f"pinned image identity mismatch: {record} != {expected}")
    return record


def build_external_manifest(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    fa2_path = (repo / FA2_REPO_RELATIVE).resolve(strict=True)
    expected_fa2_path = repo / FA2_REPO_RELATIVE
    if fa2_path != expected_fa2_path:
        raise ContractError(
            f"forked FA2 realpath mismatch: {fa2_path} != {expected_fa2_path}"
        )
    fa2 = _exact_file_record(fa2_path, display_path=FA2_REPO_RELATIVE)
    if fa2["size"] != FA2_SIZE or fa2["sha256"] != FA2_SHA256:
        raise ContractError(f"forked FA2 identity mismatch: {fa2}")

    model_files, model_digest = _pinned_model_files(MODEL_ROOT)
    payload: dict[str, Any] = {
        "schema": EXTERNAL_SCHEMA,
        "canonical_format": CANONICAL_FORMAT,
        "image": _docker_image_record(),
        "forked_fa2": fa2,
        "model": {
            "root": str(MODEL_ROOT),
            "file_count": len(model_files),
            "files": model_files,
            "canonical_sha256": model_digest,
        },
        "arctic_source": {
            "version": ARCTIC_VERSION,
            "url": ARCTIC_SDIST_URL,
            "sha256": ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()
    return payload


def validate_external_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("external manifest must be an object")
    recorded_digest = payload.get("overall_canonical_sha256")
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    expected_digest = hashlib.sha256(canonical_bytes(digest_payload)).hexdigest()
    if recorded_digest != expected_digest:
        raise ContractError("external manifest canonical digest mismatch")
    if payload.get("schema") != EXTERNAL_SCHEMA:
        raise ContractError("external manifest schema mismatch")
    if payload.get("canonical_format") != CANONICAL_FORMAT:
        raise ContractError("external manifest canonical format mismatch")
    if payload.get("image") != {
        "reference": IMAGE_REFERENCE,
        "id": IMAGE_ID,
        "repo_digests": [IMAGE_REFERENCE],
        "os": IMAGE_OS,
        "architecture": IMAGE_ARCHITECTURE,
    }:
        raise ContractError("external manifest image identity mismatch")
    if payload.get("forked_fa2") != {
        "path": FA2_REPO_RELATIVE,
        "size": FA2_SIZE,
        "sha256": FA2_SHA256,
    }:
        raise ContractError("external manifest FA2 identity mismatch")
    model = payload.get("model")
    model_rows = model.get("files") if isinstance(model, dict) else None
    if (
        not isinstance(model, dict)
        or model.get("root") != str(MODEL_ROOT)
        or model.get("file_count") != len(MODEL_FILES)
        or not isinstance(model_rows, list)
        or any(not isinstance(row, dict) for row in model_rows)
        or [row.get("path") for row in model_rows] != list(MODEL_FILES)
    ):
        raise ContractError("external manifest model identity is incomplete")
    model_digest = hashlib.sha256(canonical_bytes(model_rows)).hexdigest()
    if (
        model_rows != expected_model_file_records()
        or model.get("canonical_sha256") != MODEL_CANONICAL_SHA256
        or model_digest != MODEL_CANONICAL_SHA256
    ):
        raise ContractError(
            "external manifest model digest is not the pinned canonical digest"
        )
    if payload.get("arctic_source") != {
        "version": ARCTIC_VERSION,
        "url": ARCTIC_SDIST_URL,
        "sha256": ARCTIC_SDIST_SHA256,
    }:
        raise ContractError("external manifest Arctic source mismatch")
    return payload


def _distribution_files_record(distribution_name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    rows = []
    for relative in sorted(distribution.files or [], key=str):
        relative_text = str(relative)
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        rows.append(
            {
                "path": relative_text,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "files": rows,
        "canonical_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }


def _expected_runtime_fa2_identity(
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    env = os.environ if env is None else env
    live = env.get("FR13_FA2_QROW16_LIVE_PAGED_AB", "0")
    production = env.get("FR13_FA2_QROW16_PRODUCTION", "0")
    qrow32_b1_live = env.get("FR13_FA2_QROW32_B1_LIVE_AB_ARM", "")
    qrow32_b1_production = env.get("FR13_FA2_QROW32_B1_PRODUCTION_ARM", "")
    qrow32_b4_live = env.get("FR13_FA2_QROW32_LIVE_PAGED_AB", "0")
    qrow32_b4_arm = env.get("FR13_FA2_QROW32_LIVE_PAGED_AB_ARM", "")
    qrow32_b4_timing = env.get("FR13_FA2_QROW32_B4_TIMING_ARM", "")
    qrow32_b4_production = env.get("FR13_FA2_QROW32_B4_PRODUCTION_ARM", "")
    for name, value in (
        ("FR13_FA2_QROW16_LIVE_PAGED_AB", live),
        ("FR13_FA2_QROW16_PRODUCTION", production),
        ("FR13_FA2_QROW32_LIVE_PAGED_AB", qrow32_b4_live),
    ):
        if value not in {"0", "1"}:
            raise ContractError(f"{name} must be exactly 0 or 1")
    if live == "1" and production == "1":
        raise ContractError("qrow16 live and production selectors are mutually exclusive")
    if qrow32_b1_live not in (
        {"", "nosplit", "split2", "visibility", "gqa_pair"}
        | set(QROW32_B1_TIER_B_ARMS)
    ):
        raise ContractError(
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty, nosplit, split2, "
            "visibility, gqa_pair, or a tier-b arm "
            f"({', '.join(QROW32_B1_TIER_B_ARMS)})"
        )
    # Tier-B arms are LIVE-only. Mark's pass-64 ruling grants serving on a
    # Tier-B credential and withholds promoted-default until exact16 QC parity,
    # so the production allowlist deliberately does NOT gain the tier-b arms --
    # widening it is a separate decision with a separate gate.
    if qrow32_b1_production not in {"", "nosplit", "gqa_pair"}:
        raise ContractError(
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty, nosplit, or gqa_pair"
        )
    if qrow32_b1_live and qrow32_b1_production:
        raise ContractError(
            "qrow32 B1 live and production selectors are mutually exclusive"
        )
    if (qrow32_b1_live or qrow32_b1_production) and (
        live == "1" or production == "1" or qrow32_b4_live == "1"
    ):
        raise ContractError("qrow16 and qrow32 B1 selectors are mutually exclusive")
    if qrow32_b4_arm not in {"", "qrow32", "gqa_pair", "visibility"}:
        raise ContractError("qrow32 B4 live arm is invalid")
    if (qrow32_b4_live == "1") != bool(qrow32_b4_arm):
        raise ContractError("qrow32 B4 live gate and arm must be enabled together")
    if qrow32_b4_live == "1" and (live == "1" or production == "1"):
        raise ContractError("qrow16 and qrow32 B4 selectors are mutually exclusive")
    if qrow32_b4_timing not in {"", "stock_dispatch", "gqa_pair"}:
        raise ContractError(
            "FR13_FA2_QROW32_B4_TIMING_ARM must be empty, stock_dispatch, "
            "or gqa_pair"
        )
    if qrow32_b4_production not in {"", "gqa_pair"}:
        raise ContractError(
            "FR13_FA2_QROW32_B4_PRODUCTION_ARM must be empty or gqa_pair"
        )
    # The timing pair is a single-variable delta: both arms load the identical
    # pinned GQA-pair binary and differ only in whether the served decode call
    # carries the sentinel. A NAMED timing arm must therefore agree exactly
    # with the served kernel.
    #
    # PROMOTION 2026-08-14 (Mark's B4 default flip, the B4 analogue of the B1
    # flip 99a511319) introduced the one configuration that is not a pair:
    # production SERVING carries no timing arm at all. So the disagreement is
    # checked only when a timing arm was named. `stock_dispatch` beside a
    # gqa_pair serve is still refused and a gqa_pair timing arm beside a stock
    # serve is still refused; what is now legal is exactly the empty timing arm
    # beside the promoted production arm. This mirrors, byte for byte in
    # intent, the launcher clause it backs up -- the two must not drift, or the
    # host would admit a serve the in-container attestation then rejects.
    if qrow32_b4_timing and (
        (qrow32_b4_timing == "gqa_pair") != (qrow32_b4_production == "gqa_pair")
    ):
        raise ContractError(
            "qrow32 B4 timing and production arms must agree on the served kernel"
        )
    if (qrow32_b4_timing or qrow32_b4_production) and (
        live == "1"
        or production == "1"
        or qrow32_b4_live == "1"
        or qrow32_b1_live
        or qrow32_b1_production
    ):
        raise ContractError(
            "qrow32 B4 timing and other private FA2 selectors are mutually exclusive"
        )
    # The binary pin follows the ARM, not the pair. A promoted production serve
    # loads the same sealed .so the timing pair loaded, so it must declare the
    # same identity -- otherwise the flip would have created a serving shape
    # that reaches the credential with an unpinned binary.
    if qrow32_b4_timing or qrow32_b4_production:
        if env.get("FR13_FA2_QROW32_SO_SHA256", "") != QROW32_B4_GQA_PAIR_FA2_SHA256:
            raise ContractError(
                "qrow32 B4 runtime FA2 declaration is not the pinned "
                "GQA-pair candidate"
            )
        return QROW32_B4_GQA_PAIR_FA2_SIZE, QROW32_B4_GQA_PAIR_FA2_SHA256
    if qrow32_b1_live or qrow32_b1_production:
        declared_sha256 = env.get("FR13_FA2_QROW32_B1_SO_SHA256", "")
        # Resolve the arm that decides which binary is loaded. A production
        # launch has no live arm, so keying on the live arm alone would check
        # a GQA-pair production run against the split2/incumbent pins and
        # demand the wrong .so.
        qrow32_b1_pin_arm = qrow32_b1_live or qrow32_b1_production
        if qrow32_b1_pin_arm == "gqa_pair":
            if not QROW32_B1_GQA_PAIR_FA2_SHA256 or not QROW32_B1_GQA_PAIR_FA2_SIZE:
                raise ContractError(
                    "qrow32 B1 GQA-pair binary is not pinned: fill "
                    "QROW32_B1_GQA_PAIR_FA2_SHA256 and "
                    "QROW32_B1_GQA_PAIR_FA2_SIZE from the build attestation "
                    "before running this arm"
                )
            expected = (
                QROW32_B1_GQA_PAIR_FA2_SIZE,
                QROW32_B1_GQA_PAIR_FA2_SHA256,
            )
        elif qrow32_b1_pin_arm in QROW32_B1_TIER_B_ARMS:
            # Arm S (promotion A/B, 2026-08-18) refused here: the split-K pins
            # landed in the launcher's bash pin case but this resolver had no
            # entry, so a split-K launch fell through to split2's identity and
            # died at "binary identity is not qualified". A fall-through
            # default that silently answers for an arm it was not written for
            # is the defect; naming every arm is the fix.
            expected = (QROW32_B1_SPLITK_FA2_SIZE, QROW32_B1_SPLITK_FA2_SHA256)
        else:
            expected = (
                (QROW32_B1_VISIBILITY_FA2_SIZE, QROW32_B1_VISIBILITY_FA2_SHA256)
                if qrow32_b1_pin_arm == "visibility"
                else (QROW32_B1_SPLIT2_FA2_SIZE, QROW32_B1_SPLIT2_FA2_SHA256)
            )
        if declared_sha256 != expected[1]:
            raise ContractError(
                "qrow32 B1 runtime FA2 declaration is not the pinned candidate"
            )
        return expected
    if qrow32_b4_live == "1":
        identities = {
            "qrow32": (QROW32_B4_FA2_SIZE, QROW32_B4_FA2_SHA256),
            "gqa_pair": (
                QROW32_B4_GQA_PAIR_FA2_SIZE,
                QROW32_B4_GQA_PAIR_FA2_SHA256,
            ),
            "visibility": (
                QROW32_B4_VISIBILITY_FA2_SIZE,
                QROW32_B4_VISIBILITY_FA2_SHA256,
            ),
        }
        expected = identities[qrow32_b4_arm]
        if env.get("FR13_FA2_QROW32_SO_SHA256", "") != expected[1]:
            raise ContractError(
                "qrow32 B4 runtime FA2 declaration is not the pinned candidate"
            )
        return expected
    if live == "1":
        declared_sha256 = env.get("FR13_FA2_QROW16_SO_SHA256", "")
        if declared_sha256 != QROW16_DIVFREE_FA2_SHA256:
            raise ContractError(
                "qrow16 live runtime FA2 declaration is not the pinned division-free candidate"
            )
        return QROW16_DIVFREE_FA2_SIZE, QROW16_DIVFREE_FA2_SHA256
    if production == "1":
        declared_sha256 = env.get("FR13_FA2_QROW16_SO_SHA256", "")
        if declared_sha256 != QROW16_FA2_SHA256:
            raise ContractError(
                "qrow16 production runtime FA2 declaration is not the qualified candidate"
            )
        return QROW16_FA2_SIZE, QROW16_FA2_SHA256
    return FA2_SIZE, FA2_SHA256


def _require_built_runtime_fa2_identity(
    source: dict[str, Any],
    destination: dict[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    expected_size, expected_sha256 = _expected_runtime_fa2_identity(env)
    for record, expected_path in (
        (source, str(CONTAINER_FA2_SOURCE)),
        (destination, str(CONTAINER_FA2_DESTINATION)),
    ):
        if record != {
            "path": expected_path,
            "size": expected_size,
            "sha256": expected_sha256,
        }:
            raise ContractError(f"container FA2 identity mismatch: {record}")


def build_runtime_attestation() -> dict[str, Any]:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    import vllm

    source = _exact_file_record(
        CONTAINER_FA2_SOURCE, display_path=str(CONTAINER_FA2_SOURCE)
    )
    destination = _exact_file_record(
        CONTAINER_FA2_DESTINATION, display_path=str(CONTAINER_FA2_DESTINATION)
    )
    _require_built_runtime_fa2_identity(source, destination)
    if source["sha256"] != destination["sha256"]:
        raise ContractError("mounted and installed FA2 binaries differ")
    arctic = _distribution_files_record("arctic-inference")
    if arctic["version"] != ARCTIC_VERSION:
        raise ContractError(f"Arctic version mismatch: {arctic['version']}")
    if vllm.__version__ != VLLM_VERSION:
        raise ContractError(f"vLLM version mismatch: {vllm.__version__}")
    payload: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "canonical_format": CANONICAL_FORMAT,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "vllm": {
            "version": vllm.__version__,
            "module_path": str(Path(vllm.__file__).resolve()),
        },
        "forked_fa2": {
            "source": source,
            "destination": destination,
            "byte_identical": True,
        },
        "arctic": {
            **arctic,
            "cache_class_module": SuffixDecodingCache.__module__,
            "cache_class_qualname": SuffixDecodingCache.__qualname__,
            "pinned_source_url": ARCTIC_SDIST_URL,
            "pinned_source_sha256": ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        canonical_bytes(payload)
    ).hexdigest()
    return payload


def validate_runtime_attestation(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("runtime attestation must be an object")
    recorded_digest = payload.get("overall_canonical_sha256")
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    if recorded_digest != hashlib.sha256(canonical_bytes(digest_payload)).hexdigest():
        raise ContractError("runtime attestation canonical digest mismatch")
    if payload.get("schema") != RUNTIME_SCHEMA:
        raise ContractError("runtime attestation schema mismatch")
    if payload.get("canonical_format") != CANONICAL_FORMAT:
        raise ContractError("runtime attestation canonical format mismatch")
    if (payload.get("vllm") or {}).get("version") != VLLM_VERSION:
        raise ContractError("runtime attestation vLLM version mismatch")
    fa2 = payload.get("forked_fa2")
    if not isinstance(fa2, dict) or fa2.get("byte_identical") is not True:
        raise ContractError("runtime attestation does not prove FA2 byte identity")
    source = fa2.get("source")
    destination = fa2.get("destination")
    known_identities = {
        (FA2_SIZE, FA2_SHA256),
        (QROW16_FA2_SIZE, QROW16_FA2_SHA256),
        (QROW16_DIVFREE_FA2_SIZE, QROW16_DIVFREE_FA2_SHA256),
        (QROW32_B1_SPLIT2_FA2_SIZE, QROW32_B1_SPLIT2_FA2_SHA256),
        (QROW32_B1_VISIBILITY_FA2_SIZE, QROW32_B1_VISIBILITY_FA2_SHA256),
        (QROW32_B4_FA2_SIZE, QROW32_B4_FA2_SHA256),
        (QROW32_B4_GQA_PAIR_FA2_SIZE, QROW32_B4_GQA_PAIR_FA2_SHA256),
        (QROW32_B4_VISIBILITY_FA2_SIZE, QROW32_B4_VISIBILITY_FA2_SHA256),
    }
    if QROW32_B1_GQA_PAIR_FA2_SHA256 and QROW32_B1_GQA_PAIR_FA2_SIZE:
        known_identities.add(
            (QROW32_B1_GQA_PAIR_FA2_SIZE, QROW32_B1_GQA_PAIR_FA2_SHA256)
        )
    known_identities.add(
        (QROW32_B1_SPLITK_FA2_SIZE, QROW32_B1_SPLITK_FA2_SHA256)
    )
    for key, record, expected_path in (
        ("source", source, str(CONTAINER_FA2_SOURCE)),
        ("destination", destination, str(CONTAINER_FA2_DESTINATION)),
    ):
        if (
            not isinstance(record, dict)
            or record.get("path") != expected_path
            or (record.get("size"), record.get("sha256")) not in known_identities
        ):
            raise ContractError(f"runtime attestation {key} FA2 mismatch")
    if (
        source.get("size") != destination.get("size")
        or source.get("sha256") != destination.get("sha256")
    ):
        raise ContractError("runtime attestation FA2 source/destination mismatch")
    arctic = payload.get("arctic")
    if (
        not isinstance(arctic, dict)
        or arctic.get("version") != ARCTIC_VERSION
        or arctic.get("pinned_source_url") != ARCTIC_SDIST_URL
        or arctic.get("pinned_source_sha256") != ARCTIC_SDIST_SHA256
        or arctic.get("cache_class_module") != "arctic_inference.suffix_decoding.cache"
        or arctic.get("cache_class_qualname") != "SuffixDecodingCache"
        or not isinstance(arctic.get("files"), list)
        or not arctic["files"]
    ):
        raise ContractError("runtime attestation Arctic identity mismatch")
    if (
        arctic.get("canonical_sha256")
        != hashlib.sha256(canonical_bytes(arctic["files"])).hexdigest()
    ):
        raise ContractError("runtime attestation Arctic file digest mismatch")
    return payload


def atomic_write_json(
    path: Path,
    payload: object,
    *,
    mode: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        if mode is not None:
            os.fchmod(handle.fileno(), mode)
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _expect_contract_error(
    label: str,
    expected_text: str,
    callback: Any,
) -> None:
    try:
        callback()
    except ContractError as error:
        if expected_text not in str(error):
            raise AssertionError(
                f"{label}: wrong error {error!r}; expected {expected_text!r}"
            ) from error
    else:
        raise AssertionError(f"{label}: tamper unexpectedly passed")


def run_self_test() -> None:
    golden_records = expected_model_file_records()
    if tuple(row["path"] for row in golden_records) != MODEL_FILES:
        raise AssertionError("golden model records do not match MODEL_FILES")
    golden_digest = hashlib.sha256(canonical_bytes(golden_records)).hexdigest()
    if golden_digest != MODEL_CANONICAL_SHA256:
        raise AssertionError(
            "golden model records do not produce MODEL_CANONICAL_SHA256"
        )

    fixture_files = ("config.json", "weights.bin")
    with tempfile.TemporaryDirectory(prefix="fr13-fixed32-contract-test-") as raw:
        model_root = Path(raw) / "model"
        model_root.mkdir()
        config_path = model_root / "config.json"
        weight_path = model_root / "weights.bin"
        config_path.write_text(
            '{"text_config":{"vocab_size":248320}}\n',
            encoding="utf-8",
        )
        weight_path.write_bytes(b"fixed32-model-fixture")

        def fixture_digest() -> str:
            records = [
                _exact_file_record(model_root / name, display_path=name)
                for name in fixture_files
            ]
            return hashlib.sha256(canonical_bytes(records)).hexdigest()

        expected_digest = fixture_digest()
        records, observed_digest = _pinned_model_files(
            model_root,
            expected_files=fixture_files,
            expected_digest=expected_digest,
            expected_records=None,
        )
        if observed_digest != expected_digest or len(records) != len(fixture_files):
            raise AssertionError("valid pinned-model fixture did not round-trip")

        weight_path.write_bytes(b"fixed32-model-tamper")
        _expect_contract_error(
            "model content tamper",
            "model canonical digest mismatch",
            lambda: _pinned_model_files(
                model_root,
                expected_files=fixture_files,
                expected_digest=expected_digest,
                expected_records=None,
            ),
        )
        weight_path.write_bytes(b"fixed32-model-fixture")

        config_path.write_text(
            '{"text_config":{"vocab_size":248319}}\n',
            encoding="utf-8",
        )
        wrong_vocab_digest = fixture_digest()
        _expect_contract_error(
            "model vocab tamper",
            "text_config.vocab_size mismatch",
            lambda: _pinned_model_files(
                model_root,
                expected_files=fixture_files,
                expected_digest=wrong_vocab_digest,
                expected_records=None,
            ),
        )

        config_path.write_text(
            '{"text_config":{"vocab_size":true}}\n',
            encoding="utf-8",
        )
        bool_vocab_digest = fixture_digest()
        _expect_contract_error(
            "boolean model vocab tamper",
            "text_config.vocab_size mismatch",
            lambda: _pinned_model_files(
                model_root,
                expected_files=fixture_files,
                expected_digest=bool_vocab_digest,
                expected_records=None,
            ),
        )

        config_path.write_text(
            '{"text_config":{"vocab_size":248320,"vocab_size":248320}}\n',
            encoding="utf-8",
        )
        duplicate_vocab_digest = fixture_digest()
        _expect_contract_error(
            "duplicate model vocab key",
            "duplicate JSON key",
            lambda: _pinned_model_files(
                model_root,
                expected_files=fixture_files,
                expected_digest=duplicate_vocab_digest,
                expected_records=None,
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    external = subparsers.add_parser("external-manifest")
    external.add_argument("--repo", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    runtime = subparsers.add_parser("runtime-attestation")
    runtime.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("self-test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "self-test":
            run_self_test()
            print("PASS fr13_fixed32_contract self-test")
            return 0
        if args.command == "external-manifest":
            payload = build_external_manifest(args.repo)
            validate_external_manifest(payload)
        else:
            payload = build_runtime_attestation()
            validate_runtime_attestation(payload)
        atomic_write_json(
            args.output,
            payload,
            mode=(
                RUNTIME_ATTESTATION_MODE
                if args.command == "runtime-attestation"
                else None
            ),
        )
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL fixed32 contract: {error}", file=sys.stderr)
        return 2
    print(payload["overall_canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
