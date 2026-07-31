#!/usr/bin/env python3
"""Verify and install the pinned fixed32 CUTLASS Stream-K extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


CANDIDATE_SHA256 = "fa9395754b13de26dbed38dfc551614dbb109058764426564dcbb3c77fdd6ea9"
CANDIDATE_SIZE = 111_383_840
CANDIDATE_SELECTORS = frozenset({"streamk_coop128", "streamk_coop128_byte_ab"})
CONTAINER_SOURCE = Path("/tmp/fr13_cutlass_wave.abi3.so")
CONTAINER_DESTINATION = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_candidate(path: Path) -> dict[str, object]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"candidate is not a regular non-symlink file: {path}")
    if info.st_size != CANDIDATE_SIZE:
        raise ValueError(f"candidate size mismatch: {info.st_size} != {CANDIDATE_SIZE}")
    digest = sha256_file(path)
    if digest != CANDIDATE_SHA256:
        raise ValueError(f"candidate SHA-256 mismatch: {digest} != {CANDIDATE_SHA256}")
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_production_qualification(
    sidecar: Path,
    expected_sidecar_sha256: str,
    candidate: Path,
    patch_source: Path,
) -> dict[str, object]:
    import fr13_cutlass_streamk_pass as qualification

    return qualification.verify_sidecar(
        sidecar,
        expected_sidecar_sha256,
        candidate,
        patch_source,
    )


def install_candidate(
    source: Path,
    destination: Path,
    attestation: Path,
    selector: str,
    *,
    production_sidecar: Path | None = None,
    expected_production_sidecar_sha256: str | None = None,
    patch_source: Path = Path("scripts/fr13_patch_cutlass_fixed32_wave.py"),
) -> dict[str, object]:
    if selector not in CANDIDATE_SELECTORS:
        raise ValueError(f"unsupported candidate selector: {selector!r}")
    source_identity = verify_candidate(source)
    production_enabled = selector == "streamk_coop128"
    qualification: dict[str, object] | None = None
    if production_enabled:
        if production_sidecar is None or expected_production_sidecar_sha256 is None:
            raise ValueError(
                "Stream-K production install requires a pinned production sidecar"
            )
        qualification_record = _verify_production_qualification(
            production_sidecar,
            expected_production_sidecar_sha256,
            source,
            patch_source,
        )
        qualification = {
            "sidecar_sha256": expected_production_sidecar_sha256,
            "live_result_sha256": qualification_record["live_result_sha256"],
            "candidate_sha256": qualification_record["candidate_sha256"],
            "patch_source_sha256": qualification_record["patch_source_sha256"],
            "qualification_source_commit": qualification_record[
                "qualification_source_commit"
            ],
        }
    elif (
        production_sidecar is not None or expected_production_sidecar_sha256 is not None
    ):
        raise ValueError("diagnostic Stream-K install forbids production credentials")
    destination_info = destination.lstat()
    if destination.is_symlink() or not stat.S_ISREG(destination_info.st_mode):
        raise ValueError(
            f"destination is not an existing regular non-symlink file: {destination}"
        )

    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{destination.name}.fr13.", dir=destination.parent
    )
    temporary = Path(temporary_raw)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            os.fchmod(writer.fileno(), 0o555)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    installed_identity = verify_candidate(destination)
    installed_mode = stat.S_IMODE(destination.lstat().st_mode)
    if installed_mode != 0o555:
        raise ValueError(f"installed candidate mode mismatch: {installed_mode:#o}")
    payload: dict[str, object] = {
        "schema": "fr13.fixed32.cutlass_streamk_binary.v2",
        "selector": selector,
        "source": source_identity,
        "destination": installed_identity,
        "installed_mode": "0555",
        "production_enabled": production_enabled,
    }
    if qualification is not None:
        payload["qualification"] = qualification
    _write_json(attestation, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("candidate", type=Path)
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--source", type=Path, required=True)
    install_parser.add_argument("--destination", type=Path, required=True)
    install_parser.add_argument("--attestation", type=Path, required=True)
    install_parser.add_argument("--selector", required=True)
    install_parser.add_argument("--production-pass-sidecar", type=Path)
    install_parser.add_argument("--expected-production-pass-sha256")
    install_parser.add_argument(
        "--patch-source",
        type=Path,
        default=Path("scripts/fr13_patch_cutlass_fixed32_wave.py"),
    )
    args = parser.parse_args()

    if args.command == "verify":
        payload = verify_candidate(args.candidate)
    else:
        payload = install_candidate(
            args.source,
            args.destination,
            args.attestation,
            args.selector,
            production_sidecar=args.production_pass_sidecar,
            expected_production_sidecar_sha256=(args.expected_production_pass_sha256),
            patch_source=args.patch_source,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
