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


CANDIDATE_SHA256 = "f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d"
CANDIDATE_SIZE = 111_417_328
WIDE256_CANDIDATE_SHA256 = (
    "f7d5c01ca79829fbfff4c93949d057bd740905165b0b6793b3c0007629add962"
)
WIDE256_CANDIDATE_SIZE = 112_481_752
B4_M128_CANDIDATE_SHA256 = (
    "895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f"
)
B4_M128_CANDIDATE_SIZE = 112_698_512
COOP128_SELECTORS = frozenset({"streamk_coop128", "streamk_coop128_byte_ab"})
WIDE256_SELECTORS = frozenset(
    {"streamk_force_wide256", "streamk_force_wide256_byte_ab"}
)
B4_M128_SELECTORS = frozenset({"persistent_b4_m128", "persistent_b4_m128_byte_ab"})
CANDIDATE_SELECTORS = COOP128_SELECTORS | WIDE256_SELECTORS | B4_M128_SELECTORS
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


def candidate_identity(selector: str) -> tuple[str, int, str]:
    if selector in COOP128_SELECTORS:
        return CANDIDATE_SHA256, CANDIDATE_SIZE, "streamk_coop128"
    if selector in WIDE256_SELECTORS:
        return WIDE256_CANDIDATE_SHA256, WIDE256_CANDIDATE_SIZE, "streamk_force_wide256"
    if selector in B4_M128_SELECTORS:
        return B4_M128_CANDIDATE_SHA256, B4_M128_CANDIDATE_SIZE, "persistent_b4_m128"
    raise ValueError(f"unsupported candidate selector: {selector!r}")


def verify_candidate(
    path: Path, selector: str = "streamk_coop128"
) -> dict[str, object]:
    expected_sha256, expected_size, candidate_family = candidate_identity(selector)
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"candidate is not a regular non-symlink file: {path}")
    if info.st_size != expected_size:
        raise ValueError(f"candidate size mismatch: {info.st_size} != {expected_size}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"candidate SHA-256 mismatch: {digest} != {expected_sha256}")
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
        "candidate_family": candidate_family,
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
    selector: str,
    fixed32_mode: str,
    draft_vocab_blocks: Path | None = None,
) -> dict[str, object]:
    if selector not in {
        "streamk_coop128",
        "streamk_force_wide256",
        "persistent_b4_m128",
    }:
        raise ValueError(f"unsupported production candidate selector: {selector!r}")
    if selector == "persistent_b4_m128":
        import fr13_cutlass_b4_pass as qualification
    else:
        import fr13_cutlass_streamk_pass as qualification

    kwargs = {"fixed32_mode": fixed32_mode} if selector == "persistent_b4_m128" else {}
    if selector == "persistent_b4_m128" and draft_vocab_blocks is not None:
        kwargs["draft_vocab_blocks"] = draft_vocab_blocks
    return qualification.verify_sidecar(
        sidecar,
        expected_sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector=selector,
        **kwargs,
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
    fixed32_mode: str = "hydra27_fixed32",
    draft_vocab_blocks: Path | None = None,
) -> dict[str, object]:
    if selector not in CANDIDATE_SELECTORS:
        raise ValueError(f"unsupported candidate selector: {selector!r}")
    source_identity = verify_candidate(source, selector)
    production_enabled = selector in {
        "streamk_coop128",
        "streamk_force_wide256",
        "persistent_b4_m128",
    }
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
            selector,
            fixed32_mode,
            draft_vocab_blocks,
        )
        qualification = {
            "sidecar_sha256": expected_production_sidecar_sha256,
            "live_result_sha256": qualification_record["live_result_sha256"],
            "candidate_sha256": qualification_record["candidate_sha256"],
            "patch_source_sha256": qualification_record["patch_source_sha256"],
            "qualification_source_commit": qualification_record[
                "qualification_source_commit"
            ],
            "qualification_task_marker": qualification_record[
                "qualification_task_marker"
            ],
            "real_task_arm_sha256": qualification_record["real_task_arm_sha256"],
            "container_env_sha256": qualification_record["container_env_sha256"],
            "qualified_draft_vocab_root": qualification_record[
                "qualified_draft_vocab_root"
            ],
            "qualified_draft_vocab_k": qualification_record["qualified_draft_vocab_k"],
            "mandatory_weight_bytes": qualification_record["mandatory_weight_bytes"],
            "mandatory_weight_floor_ms": qualification_record[
                "mandatory_weight_floor_ms"
            ],
            "one_sided_u95_cap_ms": qualification_record["one_sided_u95_cap_ms"],
        }
        for key in (
            "qualified_eager_builder_capacity",
            "qualified_topology",
            "qualified_comparison_call_limit",
        ):
            if key in qualification_record:
                qualification[key] = qualification_record[key]
        if qualification_record.get("qualification_profile") == "k64_root":
            for key in (
                "qualification_profile",
                "qualified_draft_vocab_blocks",
                "qualified_draft_vocab_blocks_sha256",
                "qualified_fixed_rows",
                "qualified_projection_nk",
            ):
                qualification[key] = qualification_record[key]
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

    installed_identity = verify_candidate(destination, selector)
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
        "candidate_family": source_identity["candidate_family"],
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
    verify_parser.add_argument("--selector", default="streamk_coop128")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--source", type=Path, required=True)
    install_parser.add_argument("--destination", type=Path, required=True)
    install_parser.add_argument("--attestation", type=Path, required=True)
    install_parser.add_argument("--selector", required=True)
    install_parser.add_argument("--production-pass-sidecar", type=Path)
    install_parser.add_argument("--expected-production-pass-sha256")
    install_parser.add_argument(
        "--fixed32-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
        default="hydra27_fixed32",
    )
    install_parser.add_argument(
        "--patch-source",
        type=Path,
        default=Path("scripts/fr13_patch_cutlass_fixed32_wave.py"),
    )
    install_parser.add_argument("--draft-vocab-blocks", type=Path)
    args = parser.parse_args()

    if args.command == "verify":
        payload = verify_candidate(args.candidate, args.selector)
    else:
        payload = install_candidate(
            args.source,
            args.destination,
            args.attestation,
            args.selector,
            production_sidecar=args.production_pass_sidecar,
            expected_production_sidecar_sha256=(args.expected_production_pass_sha256),
            patch_source=args.patch_source,
            fixed32_mode=args.fixed32_mode,
            draft_vocab_blocks=args.draft_vocab_blocks,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
