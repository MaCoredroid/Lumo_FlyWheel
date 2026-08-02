#!/usr/bin/env python3
"""Derive the fixed32 Qwen 0.19.4 bundle with a 256-call turn cap.

The source is the previously accepted, fully content-addressed bundle.  This
script never edits that source.  It verifies the complete source tree, copies it
to a new directory, changes one exact-hash-pinned JavaScript constant, and then
verifies the complete derived tree before publishing it atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from typing import Any


SCHEMA = "fr13-qwen-agent-bundle-manifest-v1"
DERIVATION_SCHEMA = "fr13-fixed32-qwen-bundle-derivation-v1"
ROOTS = ["**"]
QWEN_CODE_VERSION = "0.19.4"
SOURCE_MANIFEST_SHA256 = (
    "2643d1d64c03887654794d9bd00a88fbf9ced7362e034557cf196b8a37e744bc"
)
DERIVED_MANIFEST_SHA256 = (
    "594cac41e2d5ed505e0646f318b263ff70e200bcffe97326fe1c042fdc220516"
)
DEFAULT_SOURCE = pathlib.Path.home() / "qwen_agent_bundle"
DEFAULT_OUTPUT = pathlib.Path.home() / (
    "qwen_agent_bundle-" + DERIVED_MANIFEST_SHA256
)

PACKAGE_RELATIVE_PATH = (
    "npm/lib/node_modules/@qwen-code/qwen-code/package.json"
)
CAP_CHUNK_RELATIVE_PATH = (
    "npm/lib/node_modules/@qwen-code/qwen-code/chunks/chunk-BFG6OZN7.js"
)
SOURCE_CAP = 100
DERIVED_CAP = 256
SOURCE_NEEDLE = b"var TURN_TOOL_CALL_CAP = 100;"
DERIVED_NEEDLE = b"var TURN_TOOL_CALL_CAP = 256;"
CAP_CHUNK_BYTES = 5_451_144
SOURCE_CAP_CHUNK_SHA256 = (
    "99b24b00ea2caed61b109f68f4191a86f20afde8d1fffec0dd7c9dc06b0596fd"
)
DERIVED_CAP_CHUNK_SHA256 = (
    "d61b71c03180822e875976a721a856144b70ae8b7ff687910021a5cb91a7db89"
)

SUMMARY = {
    "entry_count": 10_499,
    "directory_count": 1_514,
    "regular_file_count": 8_970,
    "symlink_count": 15,
    "regular_file_bytes": 327_941_291,
    "executable_regular_file_count": 93,
    "manifest_bytes": 2_057_964,
}
COMMON_ENTRYPOINTS = {
    "bin/qwen": {
        "path": "bin/qwen",
        "type": "file",
        "mode": "0755",
        "bytes": 217,
        "sha256": (
            "286a61bd49fd103d0ea29a8d971030b60ac0a6e7f19b292bdf9b39858e1161e2"
        ),
    },
    "node/bin/node": {
        "path": "node/bin/node",
        "type": "file",
        "mode": "0755",
        "bytes": 124_835_376,
        "sha256": (
            "93956de2e59480474a7b46571da1651180b1a050cdf32641ebec4ce6e478e068"
        ),
    },
    "npm/bin/qwen": {
        "path": "npm/bin/qwen",
        "type": "symlink",
        "mode": "0777",
        "target": "../lib/node_modules/@qwen-code/qwen-code/cli-entry.js",
    },
    "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js": {
        "path": "npm/lib/node_modules/@qwen-code/qwen-code/cli-entry.js",
        "type": "file",
        "mode": "0755",
        "bytes": 777,
        "sha256": (
            "98335eda2e0eaa737640cb5d43da032dee457ff7931c429f972ba3ff8a695d3a"
        ),
    },
}
REQUIRED_ENTRYPOINTS = [*COMMON_ENTRYPOINTS, CAP_CHUNK_RELATIVE_PATH]


class BundleDerivationError(RuntimeError):
    """The source or derived bundle violated its pinned contract."""


@dataclass(frozen=True)
class BundleScan:
    observation: dict[str, Any]
    manifest: dict[str, Any]
    captured_files: dict[str, bytes]


def _cap_entry(sha256: str) -> dict[str, Any]:
    return {
        "path": CAP_CHUNK_RELATIVE_PATH,
        "type": "file",
        "mode": "0644",
        "bytes": CAP_CHUNK_BYTES,
        "sha256": sha256,
    }


def _expected_observation(*, derived: bool) -> dict[str, Any]:
    entrypoints = dict(COMMON_ENTRYPOINTS)
    entrypoints[CAP_CHUNK_RELATIVE_PATH] = _cap_entry(
        DERIVED_CAP_CHUNK_SHA256 if derived else SOURCE_CAP_CHUNK_SHA256
    )
    return {
        "qwen_code_version": QWEN_CODE_VERSION,
        "bundle_tree": {
            "schema": SCHEMA,
            "roots": ROOTS,
            "summary": SUMMARY,
            "entrypoints": entrypoints,
            "manifest_sha256": (
                DERIVED_MANIFEST_SHA256 if derived else SOURCE_MANIFEST_SHA256
            ),
        },
    }


SOURCE_EXPECTED = _expected_observation(derived=False)
DERIVED_EXPECTED = _expected_observation(derived=True)


def _require_ascii(value: str, *, label: str) -> str:
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise BundleDerivationError(
            f"{label} must be ASCII: {value!r}"
        ) from error
    return value


def _stamp(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_xattrs(path: pathlib.Path) -> None:
    attributes = os.listxattr(path, follow_symlinks=False)
    if attributes:
        raise BundleDerivationError(
            f"Qwen bundle entry has extended attributes: {path}"
        )


def _manifest_payload(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], bytes]:
    content_bytes = sum(entry.get("bytes", 0) for entry in entries)
    manifest = {
        "schema": SCHEMA,
        "roots": ROOTS,
        "entry_count": len(entries),
        "content_bytes": content_bytes,
        "entries": entries,
    }
    raw = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return manifest, raw


def _summary(entries: list[dict[str, Any]], manifest_raw: bytes) -> dict[str, int]:
    return {
        "entry_count": len(entries),
        "directory_count": sum(
            entry["type"] == "directory" for entry in entries
        ),
        "regular_file_count": sum(
            entry["type"] == "file" for entry in entries
        ),
        "symlink_count": sum(
            entry["type"] == "symlink" for entry in entries
        ),
        "regular_file_bytes": sum(entry.get("bytes", 0) for entry in entries),
        "executable_regular_file_count": sum(
            entry["type"] == "file"
            and bool(int(entry["mode"], 8) & 0o111)
            for entry in entries
        ),
        "manifest_bytes": len(manifest_raw),
    }


def scan_bundle(raw_root: pathlib.Path) -> BundleScan:
    root_input = raw_root.expanduser()
    try:
        root_lstat = root_input.lstat()
    except OSError as error:
        raise BundleDerivationError(
            f"cannot inspect Qwen bundle root {root_input}: {error}"
        ) from error
    if not stat.S_ISDIR(root_lstat.st_mode) or stat.S_ISLNK(root_lstat.st_mode):
        raise BundleDerivationError(
            f"Qwen bundle root must be a real directory: {root_input}"
        )
    root = root_input.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    stamps: dict[pathlib.Path, tuple[int, ...]] = {}
    captured_files: dict[str, bytes] = {}

    def relative_path(path: pathlib.Path) -> str:
        relative = "." if path == root else path.relative_to(root).as_posix()
        return _require_ascii(relative, label="Qwen bundle relative path")

    def record(path: pathlib.Path) -> None:
        relative = relative_path(path)
        before = path.lstat()
        _reject_xattrs(path)
        mode = f"{stat.S_IMODE(before.st_mode):04o}"
        if stat.S_ISDIR(before.st_mode):
            entry: dict[str, Any] = {
                "path": relative,
                "type": "directory",
                "mode": mode,
            }
        elif stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1:
                raise BundleDerivationError(
                    f"Qwen bundle hardlink is forbidden: {relative}"
                )
            digest = hashlib.sha256()
            chunks: list[bytes] | None = (
                []
                if relative in {PACKAGE_RELATIVE_PATH, CAP_CHUNK_RELATIVE_PATH}
                else None
            )
            byte_count = 0
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                opened = os.fstat(descriptor)
                if _stamp(opened) != _stamp(before):
                    raise BundleDerivationError(
                        f"Qwen bundle file changed before hashing: {relative}"
                    )
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                if _stamp(os.fstat(descriptor)) != _stamp(opened):
                    raise BundleDerivationError(
                        f"Qwen bundle file changed while hashing: {relative}"
                    )
            finally:
                os.close(descriptor)
            entry = {
                "path": relative,
                "type": "file",
                "mode": mode,
                "bytes": byte_count,
                "sha256": digest.hexdigest(),
            }
            if chunks is not None:
                captured_files[relative] = b"".join(chunks)
        elif stat.S_ISLNK(before.st_mode):
            if before.st_nlink != 1:
                raise BundleDerivationError(
                    f"Qwen bundle symlink has multiple links: {relative}"
                )
            target = _require_ascii(
                os.readlink(path),
                label="Qwen bundle symlink target",
            )
            if pathlib.PurePath(target).is_absolute():
                raise BundleDerivationError(
                    f"Qwen bundle symlink target must be relative: {relative}"
                )
            resolved_target = (path.parent / target).resolve(strict=True)
            try:
                resolved_target.relative_to(root)
            except ValueError as error:
                raise BundleDerivationError(
                    f"Qwen bundle symlink escapes its root: {relative}"
                ) from error
            entry = {
                "path": relative,
                "type": "symlink",
                "mode": mode,
                "target": target,
            }
        else:
            raise BundleDerivationError(
                f"unsupported Qwen bundle entry type: {relative}"
            )
        after = path.lstat()
        _reject_xattrs(path)
        if _stamp(after) != _stamp(before):
            raise BundleDerivationError(
                f"Qwen bundle entry changed during inspection: {relative}"
            )
        entries.append(entry)
        stamps[path] = _stamp(after)

    def walk_error(error: OSError) -> None:
        raise error

    def enumerate_paths() -> list[str]:
        observed = ["."]
        for current, dirnames, filenames in os.walk(
            root,
            topdown=True,
            onerror=walk_error,
            followlinks=False,
        ):
            for name in dirnames + filenames:
                _require_ascii(name, label="Qwen bundle path component")
            dirnames.sort(key=lambda name: name.encode("ascii"))
            filenames.sort(key=lambda name: name.encode("ascii"))
            current_path = pathlib.Path(current)
            if current_path != root:
                observed.append(relative_path(current_path))
            for name in list(dirnames):
                child = current_path / name
                if stat.S_ISLNK(child.lstat().st_mode):
                    observed.append(relative_path(child))
                    dirnames.remove(name)
            for name in filenames:
                observed.append(relative_path(current_path / name))
        return sorted(observed, key=lambda value: value.encode("ascii"))

    record(root)
    for current, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        for name in dirnames + filenames:
            _require_ascii(name, label="Qwen bundle path component")
        dirnames.sort(key=lambda name: name.encode("ascii"))
        filenames.sort(key=lambda name: name.encode("ascii"))
        current_path = pathlib.Path(current)
        if current_path != root:
            record(current_path)
        for name in list(dirnames):
            child = current_path / name
            if stat.S_ISLNK(child.lstat().st_mode):
                record(child)
                dirnames.remove(name)
        for name in filenames:
            record(current_path / name)

    entries.sort(key=lambda entry: entry["path"].encode("ascii"))
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise BundleDerivationError("duplicate path in Qwen bundle manifest")
    entry_by_path = {entry["path"]: entry for entry in entries}
    try:
        entrypoints = {
            relative: entry_by_path[relative]
            for relative in REQUIRED_ENTRYPOINTS
        }
    except KeyError as error:
        raise BundleDerivationError(
            f"missing Qwen bundle entrypoint: {error.args[0]}"
        ) from error

    for path, expected_stamp in stamps.items():
        _reject_xattrs(path)
        if _stamp(path.lstat()) != expected_stamp:
            raise BundleDerivationError(
                f"Qwen bundle tree changed during inspection: {path}"
            )
    if enumerate_paths() != paths:
        raise BundleDerivationError(
            "Qwen bundle path set changed during inspection"
        )

    manifest, manifest_raw = _manifest_payload(entries)
    try:
        package = json.loads(
            captured_files[PACKAGE_RELATIVE_PATH].decode("utf-8")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleDerivationError(
            "Qwen package metadata is missing or invalid"
        ) from error
    observation = {
        "qwen_code_version": package.get("version"),
        "bundle_tree": {
            "schema": SCHEMA,
            "roots": ROOTS,
            "summary": _summary(entries, manifest_raw),
            "entrypoints": entrypoints,
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
    }
    return BundleScan(observation, manifest, captured_files)


def _require_observation(
    observation: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if observation != expected:
        observed_manifest = observation.get("bundle_tree", {}).get(
            "manifest_sha256"
        )
        expected_manifest = expected["bundle_tree"]["manifest_sha256"]
        raise BundleDerivationError(
            f"{label} bundle differs from its canonical tree: "
            f"expected={expected_manifest} observed={observed_manifest!r}"
        )


def _derive_chunk(source: bytes) -> bytes:
    if len(source) != CAP_CHUNK_BYTES:
        raise BundleDerivationError(
            f"Qwen cap chunk byte count differs: {len(source)}"
        )
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_CAP_CHUNK_SHA256:
        raise BundleDerivationError(
            f"Qwen cap chunk preimage differs: {digest}"
        )
    if source.count(SOURCE_NEEDLE) != 1 or source.count(DERIVED_NEEDLE) != 0:
        raise BundleDerivationError(
            "Qwen cap chunk does not contain exactly one unpatched constant"
        )
    derived = source.replace(SOURCE_NEEDLE, DERIVED_NEEDLE)
    if len(derived) != len(source):
        raise BundleDerivationError("Qwen cap patch changed the chunk length")
    if hashlib.sha256(derived).hexdigest() != DERIVED_CAP_CHUNK_SHA256:
        raise BundleDerivationError("Qwen cap chunk postimage differs")
    changed = [
        index
        for index, (before, after) in enumerate(zip(source, derived, strict=True))
        if before != after
    ]
    needle_offset = source.index(SOURCE_NEEDLE)
    cap_offset = needle_offset + SOURCE_NEEDLE.index(b"100")
    if changed != [cap_offset, cap_offset + 1, cap_offset + 2]:
        raise BundleDerivationError(
            "Qwen cap patch changed bytes outside the three-digit constant"
        )
    return derived


def _simulate_derived(source_scan: BundleScan) -> BundleScan:
    source_chunk = source_scan.captured_files[CAP_CHUNK_RELATIVE_PATH]
    derived_chunk = _derive_chunk(source_chunk)
    entries = [dict(entry) for entry in source_scan.manifest["entries"]]
    changed_paths: list[str] = []
    for entry in entries:
        if entry["path"] == CAP_CHUNK_RELATIVE_PATH:
            if entry != _cap_entry(SOURCE_CAP_CHUNK_SHA256):
                raise BundleDerivationError(
                    "Qwen cap manifest entry does not match its source preimage"
                )
            entry["sha256"] = hashlib.sha256(derived_chunk).hexdigest()
            changed_paths.append(entry["path"])
    if changed_paths != [CAP_CHUNK_RELATIVE_PATH]:
        raise BundleDerivationError(
            "Qwen cap derivation did not change exactly one manifest entry"
        )
    manifest, manifest_raw = _manifest_payload(entries)
    entry_by_path = {entry["path"]: entry for entry in entries}
    observation = {
        "qwen_code_version": QWEN_CODE_VERSION,
        "bundle_tree": {
            "schema": SCHEMA,
            "roots": ROOTS,
            "summary": _summary(entries, manifest_raw),
            "entrypoints": {
                relative: entry_by_path[relative]
                for relative in REQUIRED_ENTRYPOINTS
            },
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
    }
    return BundleScan(
        observation,
        manifest,
        {
            PACKAGE_RELATIVE_PATH: source_scan.captured_files[
                PACKAGE_RELATIVE_PATH
            ],
            CAP_CHUNK_RELATIVE_PATH: derived_chunk,
        },
    )


def _write_chunk_exact(path: pathlib.Path, payload: bytes) -> None:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise BundleDerivationError(
            "derived Qwen cap chunk is not a single-link regular file"
        )
    temporary = path.with_name(f".{path.name}.fr13-{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        stat.S_IMODE(before.st_mode),
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, stat.S_IMODE(before.st_mode), follow_symlinks=False)
    os.replace(temporary, path)


def _derivation_report(*, status: str, source: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    return {
        "schema": DERIVATION_SCHEMA,
        "status": status,
        "qwen_code_version": QWEN_CODE_VERSION,
        "source": {
            "path": str(source),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "patch": {
            "path": CAP_CHUNK_RELATIVE_PATH,
            "source_cap": SOURCE_CAP,
            "derived_cap": DERIVED_CAP,
            "source_sha256": SOURCE_CAP_CHUNK_SHA256,
            "derived_sha256": DERIVED_CAP_CHUNK_SHA256,
            "changed_manifest_entries": [CAP_CHUNK_RELATIVE_PATH],
        },
        "derived": {
            "path": str(output),
            "manifest_sha256": DERIVED_MANIFEST_SHA256,
        },
    }


def derive_bundle(
    *,
    source: pathlib.Path,
    output: pathlib.Path,
    dry_run: bool,
) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().absolute()
    if source == output:
        raise BundleDerivationError("source and derived bundle paths must differ")

    source_scan = scan_bundle(source)
    _require_observation(
        source_scan.observation,
        SOURCE_EXPECTED,
        label="source",
    )
    simulated = _simulate_derived(source_scan)
    _require_observation(
        simulated.observation,
        DERIVED_EXPECTED,
        label="simulated derived",
    )
    if dry_run:
        return _derivation_report(
            status="verified-dry-run",
            source=source,
            output=output,
        )

    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            raise BundleDerivationError(
                f"derived bundle path is not a real directory: {output}"
            )
        _require_observation(
            scan_bundle(output).observation,
            DERIVED_EXPECTED,
            label="existing derived",
        )
        return _derivation_report(
            status="verified-existing",
            source=source,
            output=output,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging, symlinks=True, copy_function=shutil.copy2)
        shutil.copystat(source, staging, follow_symlinks=False)
        _write_chunk_exact(
            staging / CAP_CHUNK_RELATIVE_PATH,
            simulated.captured_files[CAP_CHUNK_RELATIVE_PATH],
        )
        derived_scan = scan_bundle(staging)
        _require_observation(
            derived_scan.observation,
            DERIVED_EXPECTED,
            label="derived",
        )
        if derived_scan.manifest != simulated.manifest:
            raise BundleDerivationError(
                "derived bundle manifest differs from the one-file simulation"
            )
        os.rename(staging, output)
    finally:
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging)

    _require_observation(
        scan_bundle(output).observation,
        DERIVED_EXPECTED,
        label="published derived",
    )
    return _derivation_report(
        status="created",
        source=source,
        output=output,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify and simulate the exact derivation without creating output",
    )
    args = parser.parse_args(argv)
    try:
        report = derive_bundle(
            source=args.source,
            output=args.output,
            dry_run=args.dry_run,
        )
    except (BundleDerivationError, OSError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
