#!/usr/bin/env python3
"""Verify and install the pinned fixed32 B1 FP8 quant runtime extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


VLLM_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
BINARY_SCHEMA = "fr13.fixed32.b1_fp8_quant_regcache.binary.v1"
CONTAINER_SOURCE = Path("/tmp/fr13_fp8_quant_regcache.abi3.so")
CONTAINER_DESTINATION = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
)
SELECTORS = frozenset({"0", "byte_ab", "1"})
HEX = frozenset("0123456789abcdef")
REQUIRED_BINARY_TOKENS = (
    b"FR13_FIXED32_B1_FP8_QUANT_REGCACHE",
    b"fr13_fixed32_b1_fp8_quant_regcache_r32k5120_kernel",
    b"fr13.fixed32.b1_fp8_quant_regcache.byte_ab.v1",
    b"/logs/fr13_fixed32_b1_fp8_quant_regcache.byte_ab.jsonl",
    b"comparison_sampled",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase commit")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{label} is empty")
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_binary(
    path: Path, expected_sha256: str, expected_size: int | None = None
) -> dict[str, Any]:
    expected_sha256 = _require_sha256(expected_sha256, "candidate binary")
    raw = regular_bytes(path, "candidate binary")
    if not raw.startswith(b"\x7fELF"):
        raise ValueError("candidate binary is not ELF")
    if len(raw) < 64 * 1024:
        raise ValueError("candidate binary is implausibly small")
    if _sha256(raw) != expected_sha256:
        raise ValueError("candidate binary SHA-256 mismatch")
    if expected_size is not None and len(raw) != expected_size:
        raise ValueError("candidate binary size mismatch")
    missing = [token.decode("ascii") for token in REQUIRED_BINARY_TOKENS if token not in raw]
    if missing:
        raise ValueError("candidate binary markers are missing: " + ",".join(missing))
    return {
        "path": str(path),
        "regular": True,
        "symlink": False,
        "bytes": len(raw),
        "sha256": expected_sha256,
    }


def _write_new(path: Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    temporary.write_bytes(raw)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def install_binary(
    *,
    source: Path,
    destination: Path,
    attestation: Path,
    selector: str,
    expected_sha256: str,
    patch_source: Path,
    source_commit: str,
    production_sidecar: Path | None = None,
    expected_production_sidecar_sha256: str | None = None,
    smoke_load: bool = False,
) -> dict[str, Any]:
    if selector not in SELECTORS:
        raise ValueError("FP8 quant selector must be 0, byte_ab, or 1")
    source_commit = _require_commit(source_commit, "runtime source")
    source_identity = validate_binary(source, expected_sha256)
    patch_raw = regular_bytes(patch_source, "FP8 quant patch source")
    production_sidecar_sha256 = None
    if selector == "1":
        if production_sidecar is None or expected_production_sidecar_sha256 is None:
            raise ValueError("production selector requires a pinned PASS sidecar")
        from fr13_fp8_quant_regcache_pass import verify_sidecar

        sidecar = verify_sidecar(
            sidecar_path=production_sidecar,
            expected_sidecar_sha256=expected_production_sidecar_sha256,
            candidate_so=source,
            expected_candidate_sha256=expected_sha256,
            patch_source=patch_source,
        )
        if sidecar["qualified_source_commit"] != source_commit:
            raise ValueError("production PASS source commit mismatch")
        production_sidecar_sha256 = expected_production_sidecar_sha256
    elif production_sidecar is not None or expected_production_sidecar_sha256 is not None:
        raise ValueError("non-production selector forbids a PASS sidecar")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(temporary, 0o555)
    os.replace(temporary, destination)
    destination_identity = validate_binary(destination, expected_sha256)
    if destination_identity["bytes"] != source_identity["bytes"]:
        raise ValueError("installed candidate size changed")
    if smoke_load:
        import torch

        torch.ops.load_library(str(destination))
        for op_name in ("per_token_group_fp8_quant", "cutlass_scaled_mm"):
            if not hasattr(torch.ops._C, op_name):
                raise ValueError(f"installed candidate did not register {op_name}")

    payload = {
        "schema": BINARY_SCHEMA,
        "status": "INSTALLED",
        "selector": selector,
        "production_enabled": selector == "1",
        "diagnostic_enabled": selector == "byte_ab",
        "vllm_commit": VLLM_COMMIT,
        "source_commit": source_commit,
        "patch_source_sha256": _sha256(patch_raw),
        "candidate_sha256": expected_sha256,
        "candidate_bytes": source_identity["bytes"],
        "source": source_identity,
        "destination": destination_identity,
        "installed_mode": "0555",
        "smoke_load_passed": smoke_load,
        "production_sidecar_sha256": production_sidecar_sha256,
    }
    _write_new(attestation, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-binary")
    verify.add_argument("binary", type=Path)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--expected-size", type=int)

    install = subparsers.add_parser("install")
    install.add_argument("--source", type=Path, required=True)
    install.add_argument("--destination", type=Path, required=True)
    install.add_argument("--attestation", type=Path, required=True)
    install.add_argument("--selector", required=True)
    install.add_argument("--expected-sha256", required=True)
    install.add_argument("--patch-source", type=Path, required=True)
    install.add_argument("--source-commit", required=True)
    install.add_argument("--production-sidecar", type=Path)
    install.add_argument("--expected-production-sidecar-sha256")
    install.add_argument("--smoke-load", action="store_true")
    args = parser.parse_args()

    if args.command == "verify-binary":
        result = validate_binary(args.binary, args.expected_sha256, args.expected_size)
    else:
        result = install_binary(
            source=args.source,
            destination=args.destination,
            attestation=args.attestation,
            selector=args.selector,
            expected_sha256=args.expected_sha256,
            patch_source=args.patch_source,
            source_commit=args.source_commit,
            production_sidecar=args.production_sidecar,
            expected_production_sidecar_sha256=(
                args.expected_production_sidecar_sha256
            ),
            smoke_load=args.smoke_load,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
