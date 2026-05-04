from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


REQUIRED_COMMANDS = ("nvcc", "cmake", "ninja", "ccache", "gcc", "g++")
REQUIRED_SOURCE_FILES = (
    "CMakeLists.txt",
    "setup.py",
    "requirements/build.txt",
    "csrc/quantization/w8a8/cutlass/Epilogues.md",
    "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu",
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_kernels.hpp",
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu",
    "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu",
)


def _vllm_package_dir() -> Path | None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def verify_cutlass_build_surface(
    *,
    source_dir: Path,
    package_dir: Path | None = None,
    require_toolchain: bool = True,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    package_dir = package_dir.resolve() if package_dir is not None else _vllm_package_dir()

    missing_commands = [
        command for command in REQUIRED_COMMANDS if require_toolchain and shutil.which(command) is None
    ]
    missing_source_files = [
        rel_path for rel_path in REQUIRED_SOURCE_FILES if not (source_dir / rel_path).is_file()
    ]

    package_extension_files: list[str] = []
    if package_dir is not None and package_dir.is_dir():
        package_extension_files = sorted(path.name for path in package_dir.glob("_C*.so"))

    missing: list[str] = []
    missing.extend(f"command:{command}" for command in missing_commands)
    missing.extend(f"source:{rel_path}" for rel_path in missing_source_files)
    if package_dir is None or not package_dir.is_dir():
        missing.append("package:vllm")
    elif not package_extension_files:
        missing.append("package:vllm/_C*.so")

    payload: dict[str, object] = {
        "ok": not missing,
        "source_dir": str(source_dir),
        "package_dir": str(package_dir) if package_dir is not None else None,
        "required_commands": list(REQUIRED_COMMANDS),
        "required_source_files": list(REQUIRED_SOURCE_FILES),
        "package_extension_files": package_extension_files,
        "missing": missing,
    }
    if missing:
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the vLLM container has both the installed binary wheel and "
            "an editable/rebuildable CUTLASS scaled-mm source surface."
        )
    )
    parser.add_argument("--source-dir", default="/opt/vllm-source")
    parser.add_argument("--package-dir")
    parser.add_argument("--skip-toolchain", action="store_true")
    args = parser.parse_args()

    payload = verify_cutlass_build_surface(
        source_dir=Path(args.source_dir),
        package_dir=Path(args.package_dir) if args.package_dir else None,
        require_toolchain=not args.skip_toolchain,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
