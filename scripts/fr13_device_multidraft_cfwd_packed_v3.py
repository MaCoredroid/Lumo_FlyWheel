#!/usr/bin/env python3
"""Load the credential-bound device module with the reviewed CFWD v3 overlay."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "fr13_device_multidraft_kernel.py"
BASE_SHA256 = "088454e0605c5d41aee7b385c6d0ff66e6a7ddb999a9697258762d0aac9fe166"
OVERLAY_PATH = SCRIPT_DIR / "fr13_cfwd_logit_direct_packed_runtime_overlay.py"
OVERLAY_SHA256 = "30e08c690d4253ce8a051a04a69ff26ab1ea292ecf8190be30de5ccfbb7ccc44"
NODE_TRUST_OVERLAY_PATH = (
    SCRIPT_DIR / "fr13_cfwd_packed_walk_node_trust_runtime_overlay.py"
)
NODE_TRUST_OVERLAY_SHA256 = (
    "2a19b24edba9b0c16410dcbd3637f2a7cb2338536d0f230a766e3d225d0fd90c"
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

_base = _load("_fr13_device_multidraft_credential_bound_base", BASE_PATH)
_overlay = _load("_fr13_cfwd_logit_direct_packed_runtime_overlay", OVERLAY_PATH)
_overlay.install(_base)
_node_trust_overlay = _load(
    "_fr13_cfwd_packed_walk_node_trust_runtime_overlay",
    NODE_TRUST_OVERLAY_PATH,
)
_node_trust_overlay.install(_base)

# Export the base module's API. Its function objects retain the base namespace,
# where the installer replaced only the reviewed CFWD symbols.
for _name, _value in vars(_base).items():
    if not _name.startswith("__"):
        globals()[_name] = _value
del _name, _value
