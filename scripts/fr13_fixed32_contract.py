#!/usr/bin/env python3
"""Pinned external and runtime contract for fixed-32 floor campaigns."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fr13_fixed32_topology import FIXED32_CHOICES, PHYSICAL_DRAFTS

EXTERNAL_SCHEMA = "fr13-fixed32-external-manifest-v1"
RUNTIME_SCHEMA = "fr13-fixed32-runtime-attestation-v1"
CANONICAL_FORMAT = "utf8-json-sort-keys-compact-v1"

IMAGE_REFERENCE = (
    "vllm/vllm-openai@"
    "sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
)
IMAGE_ID = "sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc"
IMAGE_OS = "linux"
IMAGE_ARCHITECTURE = "arm64"
VLLM_VERSION = "0.19.2rc1.dev134+gfe9c3d6c5"

FA2_REPO_RELATIVE = (
    "output/auto_research/"
    "qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-"
    "20260504T053925Z/cutlass_source_workspace/vllm-source/build/"
    "lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
)
FA2_SHA256 = "97fa2519739b3f976debb8377f8829cf3a167b410d1770bb42db390f8c5c0ae1"
FA2_SIZE = 301_219_928
CONTAINER_FA2_SOURCE = Path("/tmp/fr13_fork_fa2.so")
CONTAINER_FA2_DESTINATION = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
)

MODEL_ROOT = Path("/models/qwen3.6-27b-fp8")
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
    return [
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
    ]


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


def build_runtime_attestation() -> dict[str, Any]:
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    import vllm

    source = _exact_file_record(
        CONTAINER_FA2_SOURCE, display_path=str(CONTAINER_FA2_SOURCE)
    )
    destination = _exact_file_record(
        CONTAINER_FA2_DESTINATION, display_path=str(CONTAINER_FA2_DESTINATION)
    )
    for record in (source, destination):
        if record["size"] != FA2_SIZE or record["sha256"] != FA2_SHA256:
            raise ContractError(f"container FA2 identity mismatch: {record}")
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
    for key in ("source", "destination"):
        record = fa2.get(key)
        if (
            not isinstance(record, dict)
            or record.get("size") != FA2_SIZE
            or record.get("sha256") != FA2_SHA256
        ):
            raise ContractError(f"runtime attestation {key} FA2 mismatch")
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


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
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
        atomic_write_json(args.output, payload)
    except (ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL fixed32 contract: {error}", file=sys.stderr)
        return 2
    print(payload["overall_canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
