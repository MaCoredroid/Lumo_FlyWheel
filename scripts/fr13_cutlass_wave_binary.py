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
    "503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a"
)
WIDE256_CANDIDATE_SIZE = 113_079_680
STATIC_PERSISTENT_B1_CANDIDATE_SHA256 = (
    "88c50e7d1b6060c2bcec68f50985a1db47b43d299b574edfbfc32cac1ce68742"
)
STATIC_PERSISTENT_B1_CANDIDATE_SIZE = 113_383_800
DIVISOR_STATIC_B1_CANDIDATE_SHA256 = (
    "338e89d062c2b1ac40909dbc8d64d4ab6b0def9fd86988c9e395e8244606a9f6"
)
DIVISOR_STATIC_B1_CANDIDATE_SIZE = 113_837_288
IDENTITY_STAGE2_CANDIDATE_SHA256 = (
    "fda710872a88d29b8c5d04f463f4e1f3149f919c494437fb49f5fb9620bb0c92"
)
IDENTITY_STAGE2_CANDIDATE_SIZE = 114_909_032
IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SHA256 = (
    "bab51a0f346fe3230e351004732a0cc41f1bd6c0732b238e3ae592f07f47e208"
)
IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SIZE = 115_315_576
IDENTITY_B4_CANDIDATE_SHA256 = (
    "d7771d5a95a34d6072a796d520e8f2fa500aeccc900d57e1477941b966ea77a9"
)
IDENTITY_B4_CANDIDATE_SIZE = 116_284_480
IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256 = (
    "c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29"
)
IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE = 117_488_608
B4_M128_CANDIDATE_SHA256 = (
    "895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f"
)
B4_M128_CANDIDATE_SIZE = 112_698_512
STATIC_B4_M128_CANDIDATE_SHA256 = (
    "9c63ed03ad73640293ba544fc5acad9047dcf9e202854d86f83a7ba4ca5a7d39"
)
STATIC_B4_M128_CANDIDATE_SIZE = 113_010_008
STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256 = (
    "7ab2c3223366f4591fc2324a47c805aa0a1e9d4a106743af4256d4089054a2dc"
)
STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE = 5_404
STATIC_B4_M128_RESOURCE_CREDENTIAL_SCHEMA = (
    "fr13.fixed32.cutlass_b4_m128_static_host_build.v1"
)
COOP128_SELECTORS = frozenset({"streamk_coop128", "streamk_coop128_byte_ab"})
WIDE256_SELECTORS = frozenset(
    {"streamk_force_wide256", "streamk_force_wide256_byte_ab"}
)
STATIC_PERSISTENT_B1_SELECTORS = frozenset(
    {"static_persistent_stocktile", "static_persistent_stocktile_byte_ab"}
)
DIVISOR_STATIC_B1_SELECTORS = frozenset(
    {"divisor_static_stocktile", "divisor_static_stocktile_byte_ab"}
)
IDENTITY_STAGE2_SELECTORS = frozenset(
    {"identity_stage2_static", "identity_stage2_static_byte_ab"}
)
IDENTITY_STAGE2_PINGPONG_B1_SELECTORS = frozenset(
    {
        "identity_stage2_pingpong_b1",
        "identity_stage2_pingpong_b1_byte_ab",
    }
)
IDENTITY_STOCKSHAPE_B4_SELECTORS = frozenset(
    {"identity_stockshape_b4", "identity_stockshape_b4_byte_ab"}
)
IDENTITY_STOCKSHAPE_STAGE2_B4_SELECTORS = frozenset(
    {
        "identity_stockshape_stage2_b4",
        "identity_stockshape_stage2_b4_byte_ab",
    }
)
IDENTITY_DIVISOR_B4_SELECTORS = frozenset(
    {"identity_divisor_b4", "identity_divisor_b4_byte_ab"}
)
B4_M128_SELECTORS = frozenset({"persistent_b4_m128", "persistent_b4_m128_byte_ab"})
STATIC_B4_M128_SELECTORS = frozenset(
    {"persistent_b4_m128_static", "persistent_b4_m128_static_byte_ab"}
)
CANDIDATE_SELECTORS = (
    COOP128_SELECTORS
    | WIDE256_SELECTORS
    | STATIC_PERSISTENT_B1_SELECTORS
    | DIVISOR_STATIC_B1_SELECTORS
    | IDENTITY_STAGE2_SELECTORS
    | IDENTITY_STAGE2_PINGPONG_B1_SELECTORS
    | IDENTITY_STOCKSHAPE_B4_SELECTORS
    | IDENTITY_STOCKSHAPE_STAGE2_B4_SELECTORS
    | IDENTITY_DIVISOR_B4_SELECTORS
    | B4_M128_SELECTORS
    | STATIC_B4_M128_SELECTORS
)
PRODUCTION_SELECTORS = frozenset(
    {"streamk_coop128", "streamk_force_wide256", "persistent_b4_m128"}
)
INSTALLABLE_SELECTORS = CANDIDATE_SELECTORS - {
    "static_persistent_stocktile",
    "divisor_static_stocktile",
    "identity_stage2_static",
    "identity_stage2_pingpong_b1",
    "identity_stockshape_b4",
    "identity_stockshape_stage2_b4",
    "identity_divisor_b4",
    "persistent_b4_m128_static",
}
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
    if selector in STATIC_PERSISTENT_B1_SELECTORS:
        return (
            STATIC_PERSISTENT_B1_CANDIDATE_SHA256,
            STATIC_PERSISTENT_B1_CANDIDATE_SIZE,
            "static_persistent_stocktile",
        )
    if selector in DIVISOR_STATIC_B1_SELECTORS:
        return (
            DIVISOR_STATIC_B1_CANDIDATE_SHA256,
            DIVISOR_STATIC_B1_CANDIDATE_SIZE,
            "divisor_static_stocktile",
        )
    if selector in IDENTITY_STAGE2_SELECTORS:
        return (
            IDENTITY_STAGE2_CANDIDATE_SHA256,
            IDENTITY_STAGE2_CANDIDATE_SIZE,
            "identity_stage2_static",
        )
    if selector in IDENTITY_STAGE2_PINGPONG_B1_SELECTORS:
        return (
            IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SHA256,
            IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SIZE,
            "identity_stage2_pingpong_b1",
        )
    if selector in IDENTITY_STOCKSHAPE_B4_SELECTORS:
        return (
            IDENTITY_B4_CANDIDATE_SHA256,
            IDENTITY_B4_CANDIDATE_SIZE,
            "identity_stockshape_b4",
        )
    if selector in IDENTITY_STOCKSHAPE_STAGE2_B4_SELECTORS:
        return (
            IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256,
            IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE,
            "identity_stockshape_stage2_b4",
        )
    if selector in IDENTITY_DIVISOR_B4_SELECTORS:
        return (
            IDENTITY_B4_CANDIDATE_SHA256,
            IDENTITY_B4_CANDIDATE_SIZE,
            "identity_divisor_b4",
        )
    if selector in B4_M128_SELECTORS:
        return B4_M128_CANDIDATE_SHA256, B4_M128_CANDIDATE_SIZE, "persistent_b4_m128"
    if selector in STATIC_B4_M128_SELECTORS:
        return (
            STATIC_B4_M128_CANDIDATE_SHA256,
            STATIC_B4_M128_CANDIDATE_SIZE,
            "persistent_b4_m128_static",
        )
    raise ValueError(f"unsupported candidate selector: {selector!r}")


def verify_candidate(
    path: Path,
    selector: str = "streamk_coop128",
    *,
    resource_credential: Path | None = None,
    expected_resource_credential_sha256: str | None = None,
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
    result: dict[str, object] = {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
        "candidate_family": candidate_family,
    }
    if selector in STATIC_B4_M128_SELECTORS:
        if (
            resource_credential is None
            or expected_resource_credential_sha256 is None
        ):
            raise ValueError(
                "static M128 candidate requires a pinned resource credential"
            )
        result["resource_credential"] = verify_static_m128_resource_credential(
            resource_credential,
            expected_resource_credential_sha256,
        )
    elif (
        resource_credential is not None
        or expected_resource_credential_sha256 is not None
    ):
        raise ValueError(
            "non-static CUTLASS candidate forbids a static-M128 resource credential"
        )
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key in resource credential: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value in resource credential: {value}")


def verify_static_m128_resource_credential(
    path: Path,
    expected_sha256: str,
) -> dict[str, object]:
    if expected_sha256 != STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256:
        raise ValueError("static M128 resource-credential SHA-256 is not pinned")
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError(
            f"resource credential is not a regular non-symlink file: {path}"
        )
    if info.st_size != STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE:
        raise ValueError(
            "static M128 resource-credential size mismatch: "
            f"{info.st_size} != {STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE}"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"static M128 resource-credential SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("static M128 resource credential is not ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("static M128 resource credential must be a JSON object")

    source = payload.get("source")
    candidate = payload.get("candidate")
    host_build = payload.get("host_build")
    outputs = payload.get("outputs")
    resources = payload.get("resource_audit")
    sass = payload.get("sass_audit")
    abi = payload.get("abi_audit")
    if not all(
        isinstance(value, dict)
        for value in (source, candidate, host_build, outputs, resources, sass, abi)
    ):
        raise ValueError("static M128 resource credential lacks audited sections")
    assert isinstance(source, dict)
    assert isinstance(candidate, dict)
    assert isinstance(host_build, dict)
    assert isinstance(outputs, dict)
    assert isinstance(resources, dict)
    assert isinstance(sass, dict)
    assert isinstance(abi, dict)
    candidate_binary = outputs.get("candidate_binary")
    if not isinstance(candidate_binary, dict):
        raise ValueError("static M128 resource credential lacks binary identity")

    required = {
        "schema": STATIC_B4_M128_RESOURCE_CREDENTIAL_SCHEMA,
        "status": "host_compile_link_audit_pass_default_off",
        "acceptance_valid": False,
        "performance_claim": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"static M128 resource credential {key} mismatch"
            )
    nested_required = (
        (source, "patch_source_sha256", "977c0204d03d022bd3f4b745ad4a0bad8ec36d7bf82ac1c6f82aa42a62094fab"),
        (source, "vllm_commit", "fe9c3d6c5f66c873d196800384ed6880687b9e52"),
        (source, "patched_dispatch_sha256", "446771039af31a2ae386b917540be2a018fdc8d947c001030696ec9a6608a4c4"),
        (source, "cutlass_commit", "da5e086dab31d63815acafdac9a9c5893b1c69e2"),
        (candidate, "selector", "persistent_b4_m128_static"),
        (candidate, "diagnostic_selector", "persistent_b4_m128_static_byte_ab"),
        (candidate, "default_enabled", False),
        (host_build, "result", "pass"),
        (host_build, "candidate_object_observed_gencode", "arch=compute_121a,code=sm_121a"),
        (host_build, "gpu_runtime_used", False),
        (host_build, "docker_used", False),
        (candidate_binary, "sha256", STATIC_B4_M128_CANDIDATE_SHA256),
        (candidate_binary, "bytes", STATIC_B4_M128_CANDIDATE_SIZE),
        (resources, "result", "pass"),
        (resources, "incumbent_resource_records", 307),
        (resources, "candidate_resource_records", 309),
        (resources, "removed_or_changed_incumbent_records", 0),
        (resources, "added_candidate_records", 2),
        (resources, "candidate_registers_per_thread", 168),
        (resources, "candidate_stack_bytes_per_thread", 0),
        (resources, "candidate_local_bytes_per_thread", 0),
        (resources, "candidate_static_shared_bytes_per_cta", 1024),
        (resources, "candidate_constant0_bytes", 2688),
        (resources, "candidate_parameter_bytes", 1792),
        (resources, "threads_per_cta", 384),
        (resources, "warps_per_cta", 12),
        (resources, "detected_spills", False),
        (sass, "performance_inference_allowed", False),
        (abi, "classification", "additive_static_audit"),
        (abi, "removed_defined_symbols", 0),
        (abi, "added_or_removed_undefined_symbols", 0),
        (abi, "dt_needed_equal", True),
        (abi, "runpath_equal", True),
    )
    for section, key, expected in nested_required:
        if section.get(key) != expected:
            raise ValueError(
                f"static M128 resource credential {key} mismatch"
            )
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "schema": payload["schema"],
        "candidate_sha256": candidate_binary["sha256"],
        "candidate_bytes": candidate_binary["bytes"],
        "resource_records": resources["candidate_resource_records"],
        "registers_per_thread": resources["candidate_registers_per_thread"],
        "stack_bytes_per_thread": resources["candidate_stack_bytes_per_thread"],
        "local_bytes_per_thread": resources["candidate_local_bytes_per_thread"],
        "static_shared_bytes_per_cta": resources[
            "candidate_static_shared_bytes_per_cta"
        ],
        "constant0_bytes": resources["candidate_constant0_bytes"],
        "parameter_bytes": resources["candidate_parameter_bytes"],
        "threads_per_cta": resources["threads_per_cta"],
        "warps_per_cta": resources["warps_per_cta"],
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
            os.fchmod(handle.fileno(), 0o444)
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
    resource_credential: Path | None = None,
    expected_resource_credential_sha256: str | None = None,
) -> dict[str, object]:
    if selector not in CANDIDATE_SELECTORS:
        raise ValueError(f"unsupported candidate selector: {selector!r}")
    if selector not in INSTALLABLE_SELECTORS:
        if selector in {
            "static_persistent_stocktile",
            "divisor_static_stocktile",
            "identity_stage2_static",
            "identity_stage2_pingpong_b1",
        }:
            raise ValueError(
                "static B1 production remains unavailable until the K64/root "
                "raw-byte gate passes"
            )
        raise ValueError(
            "static M128 production remains unavailable until Tail23 and Hydra27 "
            "raw-byte gates pass"
        )
    source_identity = verify_candidate(
        source,
        selector,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=expected_resource_credential_sha256,
    )
    production_enabled = selector in PRODUCTION_SELECTORS
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
        raise ValueError("diagnostic CUTLASS install forbids production credentials")
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

    installed_identity = verify_candidate(
        destination,
        selector,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=expected_resource_credential_sha256,
    )
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
    verify_parser.add_argument("--resource-credential", type=Path)
    verify_parser.add_argument("--expected-resource-credential-sha256")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--source", type=Path, required=True)
    install_parser.add_argument("--destination", type=Path, required=True)
    install_parser.add_argument("--attestation", type=Path, required=True)
    install_parser.add_argument("--selector", required=True)
    install_parser.add_argument("--production-pass-sidecar", type=Path)
    install_parser.add_argument("--expected-production-pass-sha256")
    install_parser.add_argument("--resource-credential", type=Path)
    install_parser.add_argument("--expected-resource-credential-sha256")
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
    args = parser.parse_args()

    if args.command == "verify":
        payload = verify_candidate(
            args.candidate,
            args.selector,
            resource_credential=args.resource_credential,
            expected_resource_credential_sha256=(
                args.expected_resource_credential_sha256
            ),
        )
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
            resource_credential=args.resource_credential,
            expected_resource_credential_sha256=(
                args.expected_resource_credential_sha256
            ),
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
