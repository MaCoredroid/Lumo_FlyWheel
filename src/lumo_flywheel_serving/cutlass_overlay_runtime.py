from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OVERLAY_CONSTANT = "CUTLASS_FP8_GEMM_OVERLAY_BOOTSTRAP"
OVERLAY_RUNTIME_SCHEMA = "l0c.fp8_gemm.cutlass_runtime_overlay.v1"
DEFAULT_TARGET_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.cutlass"
LEGACY_TARGET_MODULE = "vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass"
NO_RUNTIME_EFFECT_REASON = "l0c_fp8_gemm_cutlass_overlay_no_runtime_effect"


@dataclass(frozen=True)
class CutlassOverlayMaterialization:
    runtime_dir: Path
    config_path: Path
    sitecustomize_path: Path
    effective_hash: str
    replacements_count: int
    target_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_dir": str(self.runtime_dir),
            "config_path": str(self.config_path),
            "sitecustomize_path": str(self.sitecustomize_path),
            "effective_hash": self.effective_hash,
            "replacements_count": self.replacements_count,
            "target_modules": list(self.target_modules),
        }


def load_cutlass_overlay_bootstrap(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == OVERLAY_CONSTANT for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f"{OVERLAY_CONSTANT} must be a mapping: {source_path}")
            return value
    raise ValueError(f"{OVERLAY_CONSTANT} missing from {source_path}")


def runtime_overlay_config(bootstrap: dict[str, Any]) -> dict[str, Any]:
    raw = bootstrap.get("runtime_overlay")
    if not isinstance(raw, dict):
        raise ValueError("CUTLASS overlay bootstrap missing runtime_overlay mapping")
    config = dict(raw)
    if config.get("schema") != OVERLAY_RUNTIME_SCHEMA:
        raise ValueError(
            f"CUTLASS runtime overlay schema mismatch: {config.get('schema')!r}"
        )
    targets = config.get("target_modules")
    if targets is None:
        config["target_modules"] = [DEFAULT_TARGET_MODULE, LEGACY_TARGET_MODULE]
    elif not isinstance(targets, list) or not all(isinstance(item, str) and item for item in targets):
        raise ValueError("CUTLASS runtime overlay target_modules must be a non-empty string list")
    replacements = config.get("source_replacements", [])
    if not isinstance(replacements, list):
        raise ValueError("CUTLASS runtime overlay source_replacements must be a list")
    normalized_replacements: list[dict[str, str]] = []
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict):
            raise ValueError(f"CUTLASS runtime overlay replacement {index} is not a mapping")
        before = replacement.get("before")
        after = replacement.get("after")
        label = replacement.get("label", f"replacement_{index}")
        if not isinstance(before, str) or not before:
            raise ValueError(f"CUTLASS runtime overlay replacement {index} missing before text")
        if not isinstance(after, str):
            raise ValueError(f"CUTLASS runtime overlay replacement {index} missing after text")
        if not isinstance(label, str) or not label:
            raise ValueError(f"CUTLASS runtime overlay replacement {index} missing label")
        normalized_replacements.append({"label": label, "before": before, "after": after})
    config["source_replacements"] = normalized_replacements
    return config


def effective_runtime_overlay_hash(config: dict[str, Any]) -> str:
    effective = {
        "schema": config.get("schema"),
        "target_modules": list(config.get("target_modules") or []),
        "source_replacements": [
            {
                "label": item["label"],
                "before_sha256": hashlib.sha256(item["before"].encode("utf-8")).hexdigest(),
                "after_sha256": hashlib.sha256(item["after"].encode("utf-8")).hexdigest(),
            }
            for item in config.get("source_replacements", [])
        ],
    }
    return hashlib.sha256(json.dumps(effective, sort_keys=True).encode("utf-8")).hexdigest()


def materialize_cutlass_overlay_runtime(
    *,
    overlay_source_path: str | Path,
    output_dir: str | Path,
) -> CutlassOverlayMaterialization:
    bootstrap = load_cutlass_overlay_bootstrap(overlay_source_path)
    if bootstrap.get("runtime_wired") is not True:
        raise ValueError("CUTLASS overlay bootstrap runtime_wired is not true")
    if bootstrap.get("backend") != "cutlass":
        raise ValueError("CUTLASS overlay bootstrap backend must be 'cutlass'")
    config = runtime_overlay_config(bootstrap)
    effective_hash = effective_runtime_overlay_hash(config)

    runtime_dir = Path(output_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / "cutlass_fp8_gemm_overlay.json"
    config_payload = {
        "schema": config["schema"],
        "effective_hash": effective_hash,
        "target_modules": list(config["target_modules"]),
        "source_replacements": list(config["source_replacements"]),
    }
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    sitecustomize_path = runtime_dir / "sitecustomize.py"
    sitecustomize_path.write_text(_sitecustomize_source(), encoding="utf-8")
    return CutlassOverlayMaterialization(
        runtime_dir=runtime_dir,
        config_path=config_path,
        sitecustomize_path=sitecustomize_path,
        effective_hash=effective_hash,
        replacements_count=len(config["source_replacements"]),
        target_modules=tuple(str(item) for item in config["target_modules"]),
    )


def _sitecustomize_source() -> str:
    return textwrap.dedent(
        r'''
        from __future__ import annotations

        import importlib.abc
        import importlib.machinery
        import json
        import os
        import sys
        from pathlib import Path


        CONFIG_ENV = "LUMO_FP8_GEMM_CUTLASS_OVERLAY_CONFIG"
        STRICT_ENV = "LUMO_FP8_GEMM_CUTLASS_OVERLAY_STRICT"


        def _load_config():
            raw = os.environ.get(CONFIG_ENV) or os.environ.get("VLLM_" + CONFIG_ENV)
            if not raw:
                return None
            path = Path(raw)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                if os.environ.get(STRICT_ENV, "1").lower() not in {"0", "false", "no"}:
                    raise
                return None
            return payload


        class _CutlassOverlayLoader(importlib.abc.Loader):
            def __init__(self, wrapped, fullname, config):
                self._wrapped = wrapped
                self._fullname = fullname
                self._config = config

            def create_module(self, spec):
                create_module = getattr(self._wrapped, "create_module", None)
                if create_module is None:
                    return None
                return create_module(spec)

            def exec_module(self, module):
                get_source = getattr(self._wrapped, "get_source", None)
                if get_source is None:
                    return self._wrapped.exec_module(module)
                source = get_source(self._fullname)
                if source is None:
                    return self._wrapped.exec_module(module)
                applied = []
                for replacement in self._config.get("source_replacements", []):
                    before = replacement["before"]
                    after = replacement["after"]
                    label = replacement["label"]
                    if before not in source:
                        raise RuntimeError(
                            "CUTLASS overlay replacement anchor not found: "
                            f"{label} in {self._fullname}"
                        )
                    source = source.replace(before, after, 1)
                    applied.append(label)
                code = compile(source, getattr(self._wrapped, "path", self._fullname), "exec")
                module.__dict__["__lumo_cutlass_overlay__"] = {
                    "effective_hash": self._config.get("effective_hash"),
                    "applied_replacements": applied,
                }
                exec(code, module.__dict__)


        class _CutlassOverlayFinder(importlib.abc.MetaPathFinder):
            def __init__(self, config):
                self._config = config
                self._targets = set(config.get("target_modules", []))

            def find_spec(self, fullname, path=None, target=None):
                if fullname not in self._targets:
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec is None or spec.loader is None:
                    return None
                spec.loader = _CutlassOverlayLoader(spec.loader, fullname, self._config)
                return spec


        _config = _load_config()
        if _config is not None:
            sys.meta_path.insert(0, _CutlassOverlayFinder(_config))
        '''
    ).lstrip()
