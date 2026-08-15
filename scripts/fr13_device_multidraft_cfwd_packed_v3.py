#!/usr/bin/env python3
"""Load the credential-bound device module with the reviewed CFWD v3 overlay."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "fr13_device_multidraft_kernel.py"
BASE_SHA256 = "6e1f09f55327428a8a4b9cfdb885dcdba4c7457f58fb2b7d8b5169083dba6cf2"
OVERLAY_PATH = SCRIPT_DIR / "fr13_cfwd_logit_direct_packed_runtime_overlay.py"
OVERLAY_SHA256 = "65be6890516ebd2e5d6296abb25c69c0f731732bedea7029a2ca47210921a330"
NODE_TRUST_OVERLAY_PATH = (
    SCRIPT_DIR / "fr13_cfwd_packed_walk_node_trust_runtime_overlay.py"
)
NODE_TRUST_OVERLAY_SHA256 = (
    "b6790fe8626cc3877e8ebaab8415a827a2ca7275248247efc2b433b9c1a0425b"
)
ACTIVE_DEPTH_OVERLAY_PATH = (
    SCRIPT_DIR / "fr13_cfwd_packed_walk_active_depth_runtime_overlay.py"
)
ACTIVE_DEPTH_OVERLAY_SHA256 = (
    "8c8ef918c09102244587ba3fc46339b86cd8448bcb62db6ba04035713c07caee"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load packed CFWD runtime dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


if _sha256(BASE_PATH) != BASE_SHA256:
    raise RuntimeError("credential-bound device module identity drifted")
if _sha256(OVERLAY_PATH) != OVERLAY_SHA256:
    raise RuntimeError("packed CFWD runtime overlay identity drifted")
if _sha256(NODE_TRUST_OVERLAY_PATH) != NODE_TRUST_OVERLAY_SHA256:
    raise RuntimeError("packed-walk node-trust runtime overlay identity drifted")
if _sha256(ACTIVE_DEPTH_OVERLAY_PATH) != ACTIVE_DEPTH_OVERLAY_SHA256:
    raise RuntimeError("packed-walk active-depth runtime overlay identity drifted")

_base = _load("_fr13_device_multidraft_credential_bound_base", BASE_PATH)
_overlay = _load("_fr13_cfwd_logit_direct_packed_runtime_overlay", OVERLAY_PATH)
_overlay.install(_base)
_node_trust_overlay = _load(
    "_fr13_cfwd_packed_walk_node_trust_runtime_overlay",
    NODE_TRUST_OVERLAY_PATH,
)
_node_trust_overlay.install(_base)
_active_depth_overlay = _load(
    "_fr13_cfwd_packed_walk_active_depth_runtime_overlay",
    ACTIVE_DEPTH_OVERLAY_PATH,
)
_active_depth_overlay.install(_base)

# Export the base module's API. Its function objects retain the base namespace,
# where the installer replaced only the reviewed CFWD symbols.
for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value
