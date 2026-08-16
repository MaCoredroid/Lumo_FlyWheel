#!/usr/bin/env python3
"""FR14 arm B: normalise the RadixArk checkpoint's tokenizer periphery to official 3.8.

WHY. RadixArk ships a 1,121-byte ``tokenizer_config.json`` produced by their
conversion tool, not the upstream Qwen one. Against
``/models/qwen3.8-27b-fp8/tokenizer_config.json`` (17,928 B, the official 3.8
file, byte-identical to what the conservative unsloth arm already serves) it is
missing:

* ``added_tokens_decoder`` -- all 33 special-token records. Transformers rebuilds
  the special-token map from this block; without it the ``<|im_start|>`` /
  ``<|im_end|>`` / tool-call specials lose their declared ``special``/``lstrip``/
  ``rstrip`` semantics on the slow path and on every offline tokenizer probe.
* ``chat_template`` -- absent entirely (RadixArk carries only the sidecar
  ``chat_template.jinja``). Our serve line passes ``--chat-template`` explicitly,
  so this is not load-bearing for the engine, but the SWE runner and every
  tokenizer-side probe resolve the template from here.
* ``additional_special_tokens`` / ``extra_special_tokens`` / ``add_bos_token``.

and, worse than missing, it sets ``pad_token = "<|im_end|>"`` where official 3.8
sets ``pad_token = "<|endoftext|>"``. ``<|im_end|>`` is the STOP token of the
chat format; conflating pad with stop is the kind of thing that surfaces as a
truncation artefact under batching rather than as a loud error.

``tokenizer.json``, ``vocab.json`` and ``merges.txt`` already match official 3.8
byte-for-byte (asserted below), so no token ID moves -- this normalisation
touches only the periphery, and the contract's ``MODEL_VOCAB_JSON_SHA256`` pin
(the thing that carries the K64 DVK block map across a model swap) is unaffected
by construction.

``chat_template.jinja`` is ALREADY byte-identical to official 3.8
(``c3cf9e34...``), so it is verified and left alone rather than copied.

MUST RUN BEFORE the contract's model block is regenerated: the manifest pins
every file's sha256, so normalising afterwards would invalidate the pin.

The original is archived OUTSIDE the model directory, alongside the other FR14
provenance in ``/home/mark/shared/models/_fr14_orig_nvfp4_fp8head/``. Deliberately
not a sidecar inside the model dir: the pinned file set stays at the 26 names the
checkpoint actually ships (22 upstream + ``.lumo_pinned_revision`` + the KV
surgery sidecar + its two ``.bak`` files), and the tokenizer provenance lives with
the tokenizer provenance from the conservative arm.

(The hardlink view ``qwen3.8-27b-nvfp4-radixark-asshipped/``, used for the SGLang
native calibration, keeps its own link to the original inode and is deliberately
NOT normalised -- that view exists to serve the checkpoint exactly as published.
This script therefore unlinks before writing, never overwrites in place.)

    python3 radixark_tokenizer_normalize.py [--check] [--record <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")
OFFICIAL_ROOT = Path("/home/mark/shared/models/qwen3.8-27b-fp8")
ARCHIVE_ROOT = Path("/home/mark/shared/models/_fr14_orig_nvfp4_fp8head")

# Copied from official 3.8.
COPY_FILES = ("tokenizer_config.json",)
# Asserted identical to official 3.8, never written.
VERIFY_IDENTICAL = (
    "chat_template.jinja",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the normalisation is already applied; exit 2 if not",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="write the JSON provenance record here (outside the model dir)",
    )
    args = parser.parse_args()

    problems: list[str] = []
    for name in VERIFY_IDENTICAL:
        ours = sha256_file(MODEL_ROOT / name)
        theirs = sha256_file(OFFICIAL_ROOT / name)
        if ours != theirs:
            problems.append(
                f"{name} differs from official 3.8: {ours} != {theirs} -- this "
                "script normalises the PERIPHERY only; a token-id difference is "
                "a STOP, not something to paper over"
            )
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 2

    copied: dict[str, dict[str, object]] = {}
    for name in COPY_FILES:
        dst = MODEL_ROOT / name
        src = OFFICIAL_ROOT / name
        before = sha256_file(dst)
        want = sha256_file(src)
        entry: dict[str, object] = {
            "official_sha256": want,
            "official_size": src.stat().st_size,
            "observed_sha256": before,
            "observed_size": dst.stat().st_size,
        }
        if before == want:
            entry["action"] = "already-normalised"
            copied[name] = entry
            continue
        if args.check:
            print(f"FAIL: {name} is not normalised ({before} != {want})", file=sys.stderr)
            return 2
        archive = ARCHIVE_ROOT / f"{name}.radixark.bak"
        ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, archive)
        if sha256_file(archive) != before:
            raise SystemExit(f"archive copy of {name} did not reproduce its sha256")
        # UNLINK, do not overwrite: these files are hardlinked into the
        # as-shipped calibration view and an in-place write would mutate it.
        dst.unlink()
        shutil.copy2(src, dst)
        after = sha256_file(dst)
        if after != want:
            raise SystemExit(f"{name} copy did not reproduce the official sha256")
        entry["action"] = "copied-from-official-3.8"
        entry["archived_to"] = str(archive)
        entry["archived_sha256"] = before
        copied[name] = entry

    record = {
        "schema": "fr14.radixark_tokenizer_normalize.v1",
        "model_root": str(MODEL_ROOT),
        "official_source": str(OFFICIAL_ROOT),
        "archive_root": str(ARCHIVE_ROOT),
        "pinned_file_count_unchanged": len(
            [path for path in MODEL_ROOT.iterdir() if path.is_file()]
        ),
        "verified_identical": {
            name: sha256_file(MODEL_ROOT / name) for name in VERIFY_IDENTICAL
        },
        "normalised": copied,
    }
    rendered = json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if args.record is not None:
        args.record.write_text(rendered, encoding="ascii")
    print(rendered, end="")
    if args.check:
        print("radixark tokenizer normalisation check PASSED", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
