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
QWEN_COMPACTION_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-compaction-metrics-v1"
)
QWEN_CAMPAIGN_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-campaign-metrics-v1"
)
QWEN_CAMPAIGN_TASK_METRIC_SCHEMA = (
    "fr13-fixed32-qwen-campaign-task-metrics-v1"
)

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

NSYS_PROFILE_BINARY = Path(
    "/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys"
)
NSYS_PROFILE_OUTPUT = Path("/logs/fr13_fixed32_b1_real_swe")
NSYS_PROFILE_PREFIX = (
    str(NSYS_PROFILE_BINARY),
    "profile",
    "--session-new=%q{LUMO_NSYS_SESSION_NAME}",
    "--delay",
    "1200",
    "--duration",
    "300",
    "--trace=cuda,cuda-sw,nvtx",
    "--cuda-graph-trace=node",
    "--cuda-flush-interval",
    "100",
    "--discard-environment=true",
    "--sample=none",
    "--cpuctxsw=none",
    "--force-overwrite=true",
    "-o",
    str(NSYS_PROFILE_OUTPUT),
)

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
# The B1 GQA-pair candidate is not built yet. An empty pin is a hard refusal in
# _expected_runtime_fa2_identity, never a skipped check.
QROW32_B1_GQA_PAIR_FA2_SHA256 = ""
QROW32_B1_GQA_PAIR_FA2_SIZE = 0
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

MODEL_ROOT = Path("/models/qwen3.6-27b-fp8")
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
# FR13_MAMBA_SPEC_BLOCKS_CDIV (2026-08-09) is the same territory and is BLOCKED
# for the same structural reason: num_speculative_blocks counts mamba STATE
# SLOTS, one per draft node, not a token range, so it cannot be ceil-divided by
# mamba_block_size. See fr13_required_tree_flags.sh for the four per-node
# consumers and fr10_phase4_patch_vllm_tree_gdn.py's
# _fr13_assert_mamba_spec_blocks_cdiv_slot_demand for the fail-loud preflight.
#
# Raising this DOES NOT re-profile memory: vLLM logs "reserved 40.0 GiB memory
# for KV Cache as specified by kv_cache_memory_bytes config and skipped memory
# profiling. This does not respect the gpu_memory_utilization config", so the
# 0.70 GPU_UTIL is not the binding constraint. The measured headroom is
# "Initial free memory 104.25 GiB", so 46 GiB leaves ~58 GiB for weights and
# activations, and the B4 container cap stays at 112g.
FIXED32_B4_KV_CACHE_MEMORY_BYTES = 46 * 1024**3
MODEL_AUXILIARY_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "config.json.lumo_pre_fp8_fix.bak",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "mtp.safetensors",
    "outside.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
MODEL_FILES = tuple(
    sorted(
        (
            *MODEL_AUXILIARY_FILES,
            *(f"layers-{index}.safetensors" for index in range(64)),
        )
    )
)
MODEL_CANONICAL_SHA256 = (
    "95f8bdf8693e2b7581a27dd05494e3016d5b7d2e150de4d4e9beaead6253fc3d"
)
MODEL_TEXT_CONFIG_VOCAB_SIZE = 248_320
MODEL_FILE_RECORDS = (
    (
        ".gitattributes",
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    (
        "LICENSE",
        11343,
        "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    ),
    (
        "README.md",
        62868,
        "1d812aa4505b65e4893130b61d8e60498f15f751ed180e60dab0650823f2dced",
    ),
    (
        "chat_template.jinja",
        7764,
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
    ),
    (
        "config.json",
        21854,
        "f78c412bfdec65a88c8aa2a031d39c2fda32e3377ae48a77f971bc40a4f095df",
    ),
    (
        "config.json.lumo_pre_fp8_fix.bak",
        3662,
        "b27522b99ab9fee733c48405687ca23ccdf49845f957c3b602e5322c3194fe3f",
    ),
    (
        "configuration.json",
        51,
        "2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0",
    ),
    (
        "generation_config.json",
        202,
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    ),
    (
        "layers-0.safetensors",
        383865448,
        "5a6c052e37a754549e8f81c6fb32b050419f0fc2f71598817e08d0446b3f309d",
    ),
    (
        "layers-1.safetensors",
        383865448,
        "9a2efd1048386e560c8241c7b972e213b516f0cbeafc4de6f486bd7faddb8c39",
    ),
    (
        "layers-10.safetensors",
        383865472,
        "ebc4fd84671c2ac2d8d7b8da3707ed5d30738771a22eecc883d707521ed513a0",
    ),
    (
        "layers-11.safetensors",
        372313760,
        "9daf0f0b763489c0869061669293f93213d6060b1b0c88061b53bb97f9082cd6",
    ),
    (
        "layers-12.safetensors",
        383865472,
        "922729fcaedaece39e7b0fba137e3b900fbc3fe6bd8a85b16a215658be86938f",
    ),
    (
        "layers-13.safetensors",
        383865472,
        "61c0a820b09d58e45e516c56532007155b0fd214c11fa8906640c9217da95c84",
    ),
    (
        "layers-14.safetensors",
        383865472,
        "c2faab3a2ecddde0e762b3870c75f6fd83886edbabbd99ac40d641122d29e652",
    ),
    (
        "layers-15.safetensors",
        372313760,
        "e75fbe4de60c0420912b91f379c29c95a49644feee6fe8a0bd3dd638ae07ea2b",
    ),
    (
        "layers-16.safetensors",
        383865472,
        "84776e1d0a62c40730a4bc0f0e2c68242ba9a6150975d8992747fb6ecc0ebfcc",
    ),
    (
        "layers-17.safetensors",
        383865472,
        "c7e32717b97f9553d62734f01bed4cc970dcf6bd3d5ba67c7704290656523d7e",
    ),
    (
        "layers-18.safetensors",
        383865472,
        "1c6aed5f416dee34bb7ccfc322e0aa33f192b40afeab4b69fce784b2b349e253",
    ),
    (
        "layers-19.safetensors",
        372313760,
        "63e1b8af41de9851ed279f9d4f42566febf81d13ae47e469ea1009a26c6b64e0",
    ),
    (
        "layers-2.safetensors",
        383865448,
        "c2709097f77e8ec7204f908f54d7b84343139163e600815789124cbf15a709c6",
    ),
    (
        "layers-20.safetensors",
        383865472,
        "c5945b2e40f3d97cb01d0cedcb60ea36f61ab585179ab0c02fdeec3d360f724d",
    ),
    (
        "layers-21.safetensors",
        383865472,
        "c8a7696ad5cd1be016cfc553351ee4b588f3d972084dd47028db8dba9b113870",
    ),
    (
        "layers-22.safetensors",
        383865472,
        "08481a6a46057d0bcdcd2598d4f67d5f821725fc483800cd360c2d7047907e08",
    ),
    (
        "layers-23.safetensors",
        372313760,
        "13b2ba6536c77330fa1c40e91c4c707d5086803299e0f57dd03142bb28faaaa4",
    ),
    (
        "layers-24.safetensors",
        383865472,
        "0f7855ee6b018e707d015f445f7e8b9542003104dbd0436e83e8c0dec73bdd17",
    ),
    (
        "layers-25.safetensors",
        383865472,
        "fe49c034e095263e51b391895ccb5098741e06d75a6788769beb72fd50708846",
    ),
    (
        "layers-26.safetensors",
        383865472,
        "14f086ff505e9df89c2fb938c947b0d023f6f19d3cecdc7f03a70caf36ea6ed4",
    ),
    (
        "layers-27.safetensors",
        372313760,
        "4fa89a43e7196b18e6c67478aeedc1b3ab4853fa3bc4305d4ac73341b70e7e24",
    ),
    (
        "layers-28.safetensors",
        383865472,
        "f10aecd4f9800052bbf1ad2cca939eb44dc205d42ff546096ce9212bd6eae411",
    ),
    (
        "layers-29.safetensors",
        383865472,
        "633058fe6f1ec434c79e5c50d04c83b1955d52ca1889b1adca0e2db1cad9fa16",
    ),
    (
        "layers-3.safetensors",
        372313744,
        "a874cf17d3480894aa13eb34568e1025f10c4d6b96f483cd93be583849312a0f",
    ),
    (
        "layers-30.safetensors",
        383865472,
        "f9af47cefbdb300129c25eee1127c50629d9a2eba7f862a58a430a2cb378cf55",
    ),
    (
        "layers-31.safetensors",
        372313760,
        "2a2e87f569b5da06a87b83df628cb27f491f3dbed55ca82054820cb382cffd2e",
    ),
    (
        "layers-32.safetensors",
        383865472,
        "ca63c9dc522a1f0efd39e2cd7f5fb0efdb4274eabce5c15cc9aab11463cb3bb8",
    ),
    (
        "layers-33.safetensors",
        383865472,
        "bb8eb220efa6a599a4ee29914d59ca26fe13fe64ec6f9a5ccff300432abc572b",
    ),
    (
        "layers-34.safetensors",
        383865472,
        "6e08147ff5c8e0d5d7207d62ede78d6194f1453165cfab940743ff327bb0e6e5",
    ),
    (
        "layers-35.safetensors",
        372313760,
        "f4391ffd8245e648eeaad69954721f106ea0423a56f9abee921d3d2880d98c03",
    ),
    (
        "layers-36.safetensors",
        383865472,
        "5b077d63785d5c2b20ab5c0b8895b08941af08bb5685a1b6a5a3292c3eac638f",
    ),
    (
        "layers-37.safetensors",
        383865472,
        "d38c28d8c86cc147e5edc1935dec46687a6b3a8bc0d712e829003dd9cd4fab7d",
    ),
    (
        "layers-38.safetensors",
        383865472,
        "1628a41fa5d3dbcf2ffde423530a9e94e7228e247e922e2269bd348071a76365",
    ),
    (
        "layers-39.safetensors",
        372313760,
        "889c827bbff3fa6b78558dcdfd24670894b8a450395ec418e7a4f3202789e0b1",
    ),
    (
        "layers-4.safetensors",
        383865448,
        "9eb31d85023776894347a56101ee067f369d0d3c024f1331f5c7cba1f45b6df4",
    ),
    (
        "layers-40.safetensors",
        383865472,
        "5f90d2b2d16e5c51a078f21eed843d80c22af71401669577d274ea7f2c1ce95f",
    ),
    (
        "layers-41.safetensors",
        383865472,
        "80877a16835f7311a112ad0bbaf1d19997142ea3e1df95ffe446fd52ad1f80d8",
    ),
    (
        "layers-42.safetensors",
        383865472,
        "c1bf34d1a3e4d36b342c5c680bb12c4ce74f02530dff33b735fad27470ef2594",
    ),
    (
        "layers-43.safetensors",
        372313760,
        "10eceff499a24e0072a40b960671f00fc5012d7451f22cb6883f358e085ce276",
    ),
    (
        "layers-44.safetensors",
        383865472,
        "673543b94e59003265ae648bba11389c6735ee3264ec164108b6db99f1219587",
    ),
    (
        "layers-45.safetensors",
        383865472,
        "77b8c7a9cc451a378ee8d6927ce1c4b9664adb83c4786440eec099591dc98f32",
    ),
    (
        "layers-46.safetensors",
        383865472,
        "9ee06b8029b7812644bd5fb01b8bfea7b640353769d094ae59d8c82d986a8b7b",
    ),
    (
        "layers-47.safetensors",
        372313760,
        "6a8e0ddb389485965cd30bd0a90d1be5e1acb65cb6311ce337199f268c3120f6",
    ),
    (
        "layers-48.safetensors",
        383865472,
        "70bd74dcdf4ea8b09e1ae3463977a05aba73f2c02397491bb641a0a3ca6832a3",
    ),
    (
        "layers-49.safetensors",
        383865472,
        "a3481f971433ecc9041f44a555e443ff124e037601b36eab0bb3057946e041e4",
    ),
    (
        "layers-5.safetensors",
        383865448,
        "a8c9ea40638fd66bd8ab21940df215e7921b4af04646d758d74425be011a9b6f",
    ),
    (
        "layers-50.safetensors",
        383865472,
        "1aa60d9cb7f867a2033be0b1ff053049f09a6623826dd0d37daed3281479b875",
    ),
    (
        "layers-51.safetensors",
        372313760,
        "511f64772c1a5a15292eb1c48166b9d206088aed283185e10877fc43308097bd",
    ),
    (
        "layers-52.safetensors",
        383865472,
        "92c35a4e850b4c0b87a1eab42c47996504e062a8a11699ada20fcea0b88f36c7",
    ),
    (
        "layers-53.safetensors",
        383865472,
        "ab48cad204502ddbf17f5ee2d5e52457baac965300cc15e8a25f35c843b4d526",
    ),
    (
        "layers-54.safetensors",
        383865472,
        "2804cee95e18e9e516b9490c36ea7f41553a3fcb4455cf44a361d6086ed214a3",
    ),
    (
        "layers-55.safetensors",
        372313760,
        "543841981255fce51f0d84fe2db4de580594f6c6d6a590ae47ffbdbd01211571",
    ),
    (
        "layers-56.safetensors",
        383865472,
        "9847da2610e67b1f0ff39d8d103c7814f95bb8ede4204d5ae3055a8589bb0313",
    ),
    (
        "layers-57.safetensors",
        383865472,
        "32fe173c033e46648bc71bc8e5036565234c0c13ac66acb2ad84c9cbb426b90e",
    ),
    (
        "layers-58.safetensors",
        383865472,
        "c9858a0c4bd8cb980829a45d3f9ca52a0bd506c6a0775eb179d3e5ebfaae7867",
    ),
    (
        "layers-59.safetensors",
        372313760,
        "3c30fd14e883c15af50e2f647c2d0b8f3a2fca96904e4b37cc8bb1956a133606",
    ),
    (
        "layers-6.safetensors",
        383865448,
        "ad8b704be0cde8a774c4a49af59336d2b48b574446c1eddff8ba35d20d05ae69",
    ),
    (
        "layers-60.safetensors",
        383865472,
        "1f24aa7a99e72f658089462b258613b11d06d6e11733ff5d5f864c663a190f92",
    ),
    (
        "layers-61.safetensors",
        383865472,
        "0b083f4afca3538f87545d4fc50817292fc80061a731a5f0235f5c406be87b45",
    ),
    (
        "layers-62.safetensors",
        383865472,
        "b1e4c80f95241cc7d269d7cf6a35291c8b455f2d4f1381e4eafbfccb35add125",
    ),
    (
        "layers-63.safetensors",
        372313760,
        "7e27ec71df803eb38ba93aead54daaf1110d3f7124416443a30a3c81abc1c805",
    ),
    (
        "layers-7.safetensors",
        372313744,
        "2acb7e8d035d730a1c81c36c3b20ca15020b89d2bcd4462259f2ecb2a8dcb828",
    ),
    (
        "layers-8.safetensors",
        383865448,
        "81ebe095241562b2e2ace8875e2771f0cdb68e76286a18506ea19a749b8de9b4",
    ),
    (
        "layers-9.safetensors",
        383865448,
        "fdd926f3e8c335eb21a2b41cf635cb241f337a5d47310ef41ae3d43bf53410c3",
    ),
    (
        "merges.txt",
        3353259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    (
        "model.safetensors.index.json",
        137335,
        "6d19a4e607604c1ac631f810a56e6084c892b4cb0251c530c6a24fc877f9fb4b",
    ),
    (
        "mtp.safetensors",
        477202224,
        "9557770331f1f648eb96039f2a1e7cdc5742fe15c0f5777c43063b2c12a60f4c",
    ),
    (
        "outside.safetensors",
        6007102112,
        "27a91100d904f6acc1c86d08675691199e1fe2da5da613106fb01ae4809b3ac1",
    ),
    (
        "preprocessor_config.json",
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    (
        "tokenizer.json",
        12807982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    (
        "tokenizer_config.json",
        16718,
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
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


class ContractError(RuntimeError):
    """Raised when a fixed-32 contract value is not exact."""


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
        "qwen3.6-27b",
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
        return vllm_argv
    return [*NSYS_PROFILE_PREFIX, "vllm", *vllm_argv[2:]]


def validate_process_pid1_argv(
    argv: object,
    concurrency: int,
    *,
    attribution_only: bool,
    eager_diagnostic: bool = False,
    graph_diagnostic: bool = False,
    streamk_eager_diagnostic: bool = False,
    sfwd_byte_diagnostic: bool = False,
) -> list[str]:
    expected = expected_process_pid1_argv(
        concurrency,
        attribution_only=attribution_only,
        eager_diagnostic=eager_diagnostic,
        graph_diagnostic=graph_diagnostic,
        streamk_eager_diagnostic=streamk_eager_diagnostic,
        sfwd_byte_diagnostic=sfwd_byte_diagnostic,
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
        or message.get("model") != "qwen3.6-27b"
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
    fields.append('model_name="qwen3.6-27b"')
    return ",".join(fields)


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
                and labels.endswith('",model_name="qwen3.6-27b"')
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
    if (
        deltas["max_tokens_count"] != completed
        or deltas["max_tokens_le_inf"] != completed
        or deltas["max_tokens_le_50000"] != completed
        or deltas["request_success_stop"] != completed
        or any(
            deltas[f"request_success_{reason}"] != 0
            for reason in ("length", "abort", "error", "repetition")
        )
    ):
        raise ContractError(
            "fixed32 qwen engine completion metrics do not reconcile"
        )
    if deltas["max_tokens_le_10000"] != 0:
        raise ContractError(
            "fixed32 qwen max-token histogram has an unpinned low request"
        )

    total_compactions = deltas["max_tokens_le_20000"]
    if (
        total_compactions < successful_compaction_count
        or normal_request_count + total_compactions != completed
        or deltas["max_tokens_sum"]
        != (
            normal_request_count * QWEN_VISIBLE_MAX_OUTPUT_TOKENS
            + total_compactions * QWEN_COMPACTION_MAX_OUTPUT_TOKENS
        )
    ):
        raise ContractError(
            "fixed32 qwen 32768/20000 max-token algebra does not reconcile"
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
        "request_success_stop": deltas["request_success_stop"],
        "request_success_non_stop": sum(
            deltas[f"request_success_{reason}"]
            for reason in ("length", "abort", "error", "repetition")
        ),
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
    if (
        not isinstance(result.get("usage"), dict)
        or result.get("permission_denials") != []
    ):
        raise ContractError("fixed32 qwen result evidence is incomplete")

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
            reasoning_only_final_group = nonempty_text_count == 0 and all(
                _fixed32_qwen_reasoning_only_record(message)
                for _event, message, _event_id, _event_index in group
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


def validate_fixed32_qwen_campaign_metrics(
    tasks: list[dict[str, Any]],
    *,
    metrics_pre: bytes,
    metrics_post: bytes,
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
    }
    task_inputs: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != expected_task_keys:
            raise ContractError(
                "fixed32 qwen campaign task input is not exact"
            )
        instance_id = task["instance_id"]
        expected_session_id = task["expected_session_id"]
        completed = task["expected_completed_logical_model_requests"]
        events = task["events"]
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in seen_instance_ids
            or expected_session_id != fixed32_trace_session_id(instance_id)
            or type(completed) is not int
            or completed <= 0
            or not isinstance(events, list)
        ):
            raise ContractError(
                "fixed32 qwen campaign task identity or count is invalid"
            )
        seen_instance_ids.add(instance_id)
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
    if (
        deltas["max_tokens_count"] != completed_total
        or deltas["max_tokens_le_inf"] != completed_total
        or deltas["max_tokens_le_50000"] != completed_total
        or deltas["request_success_stop"] != completed_total
        or any(
            deltas[f"request_success_{reason}"] != 0
            for reason in ("length", "abort", "error", "repetition")
        )
    ):
        raise ContractError(
            "fixed32 qwen campaign engine completion metrics do not reconcile"
        )
    if deltas["max_tokens_le_10000"] != 0:
        raise ContractError(
            "fixed32 qwen campaign max-token histogram has an unpinned low request"
        )
    if (
        deltas["max_tokens_le_20000"] != total_compactions
        or normal_total + total_compactions != completed_total
        or deltas["max_tokens_sum"]
        != (
            normal_total * QWEN_VISIBLE_MAX_OUTPUT_TOKENS
            + total_compactions * QWEN_COMPACTION_MAX_OUTPUT_TOKENS
        )
    ):
        raise ContractError(
            "fixed32 qwen campaign 32768/20000 max-token algebra does not reconcile"
        )
    if (
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
        "request_success_stop": deltas["request_success_stop"],
        "request_success_non_stop": sum(
            deltas[f"request_success_{reason}"]
            for reason in ("length", "abort", "error", "repetition")
        ),
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
    if qrow32_b1_live not in {"", "nosplit", "split2", "visibility", "gqa_pair"}:
        raise ContractError(
            "FR13_FA2_QROW32_B1_LIVE_AB_ARM must be empty, nosplit, split2, "
            "visibility, or gqa_pair"
        )
    if qrow32_b1_production not in {"", "nosplit"}:
        raise ContractError(
            "FR13_FA2_QROW32_B1_PRODUCTION_ARM must be empty or nosplit"
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
    # carries the sentinel. The two declarations must therefore agree exactly.
    if (qrow32_b4_timing == "gqa_pair") != (qrow32_b4_production == "gqa_pair"):
        raise ContractError(
            "qrow32 B4 timing and production arms must agree on the served kernel"
        )
    if qrow32_b4_timing and (
        live == "1"
        or production == "1"
        or qrow32_b4_live == "1"
        or qrow32_b1_live
        or qrow32_b1_production
    ):
        raise ContractError(
            "qrow32 B4 timing and other private FA2 selectors are mutually exclusive"
        )
    if qrow32_b4_timing:
        if env.get("FR13_FA2_QROW32_SO_SHA256", "") != QROW32_B4_GQA_PAIR_FA2_SHA256:
            raise ContractError(
                "qrow32 B4 timing runtime FA2 declaration is not the pinned "
                "GQA-pair candidate"
            )
        return QROW32_B4_GQA_PAIR_FA2_SIZE, QROW32_B4_GQA_PAIR_FA2_SHA256
    if qrow32_b1_live or qrow32_b1_production:
        declared_sha256 = env.get("FR13_FA2_QROW32_B1_SO_SHA256", "")
        if qrow32_b1_live == "gqa_pair":
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
        else:
            expected = (
                (QROW32_B1_VISIBILITY_FA2_SIZE, QROW32_B1_VISIBILITY_FA2_SHA256)
                if qrow32_b1_live == "visibility"
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
