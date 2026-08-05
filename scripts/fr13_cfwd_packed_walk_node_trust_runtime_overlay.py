#!/usr/bin/env python3
"""Install the default-off trusted-node walk on the reviewed CFWD v3 route."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


SELECTOR_ENV = "FR13_CFWD_PACKED_WALK_NODE_TRUST_BYTE_AB"
RUNTIME_SCHEMA = "fr13.fixed32.cfwd_packed_walk.node_trust.runtime.v1"
BASE_CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
BASE_CANDIDATE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
)
BASE_CANDIDATE_SOURCE_SHA256 = (
    "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
)
BASE_INTEGRATION_SOURCE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
)
BASE_INTEGRATION_SOURCE_SHA256 = (
    "5c30860712e9766fd397b3e90e2ea203ad4ee2a89302d4a3c3c0e412452e4e07"
)
NODE_TRUST_CANDIDATE = "fixed32_cfwd_packed_walk_node_trust_v1"
NODE_TRUST_SOURCE_SCHEMA = "fr13.fixed32.cfwd_packed_walk.node_trust.v1"
NODE_TRUST_SOURCE_SHA256 = (
    "07cd03173ab1a6e6b9aa597d9c912475034f5b8100c2c57d819b2b7bbcf3bc37"
)
NODE_TRUST_SOURCE_NAME = "fr13_cfwd_packed_walk_node_trust_kernel.py"
MODE = "hydra27_fixed32"
BATCHES = (1, 4)
PHYSICAL_DRAFTS = 31
PHYSICAL_ROWS = 32
WALK_CAP = 12
OUTPUT_CAPACITY = 32
PATH_CAPACITY = 16

_RUNTIME_MODULE: ModuleType | None = None
_REFERENCE_WALK = None
_NODE_TRUST_MODULE: ModuleType | None = None


def _selector_raw() -> str:
    raw = os.environ.get(SELECTOR_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise RuntimeError(f"{SELECTOR_ENV} must be exactly 0 or 1")
    return raw


def runtime_contract(*, installed: bool) -> dict[str, object]:
    return {
        "schema": RUNTIME_SCHEMA,
        "candidate": NODE_TRUST_CANDIDATE,
        "candidate_schema": NODE_TRUST_SOURCE_SCHEMA,
        "candidate_source_sha256": NODE_TRUST_SOURCE_SHA256,
        "base_candidate": BASE_CANDIDATE,
        "base_candidate_schema": BASE_CANDIDATE_SCHEMA,
        "base_candidate_source_sha256": BASE_CANDIDATE_SOURCE_SHA256,
        "base_integration_source_schema": BASE_INTEGRATION_SOURCE_SCHEMA,
        "base_integration_source_sha256": BASE_INTEGRATION_SOURCE_SHA256,
        "selector_env": SELECTOR_ENV,
        "candidate_default_off": True,
        "reference_always_served": True,
        "shadow_comparator": "_fr13_cfwd_logit_direct_compare",
        "mode": MODE,
        "batches": list(BATCHES),
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
        "walk_levels": WALK_CAP,
        "output_capacity": OUTPUT_CAPACITY,
        "path_capacity": PATH_CAPACITY,
        "installed": bool(installed),
    }


def _producer_contract(mode: str) -> dict[str, object]:
    return {
        "candidate": BASE_CANDIDATE,
        "candidate_schema": BASE_CANDIDATE_SCHEMA,
        "candidate_source_sha256": BASE_CANDIDATE_SOURCE_SHA256,
        "integration_source_schema": BASE_INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": BASE_INTEGRATION_SOURCE_SHA256,
        "mode": mode,
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
    }


def _select(topology, entry: dict[str, Any]) -> str:
    if _selector_raw() == "0":
        return "packed_v3"
    if _RUNTIME_MODULE is None:
        raise RuntimeError("FR13 packed-walk node-trust overlay is not installed")
    mode = str(entry.get("mode", ""))
    batch = int(entry.get("batch_size", 0))
    if (
        mode != MODE
        or batch not in BATCHES
        or int(topology.PHYSICAL_DRAFTS) != PHYSICAL_DRAFTS
        or int(topology.PHYSICAL_ROWS) != PHYSICAL_ROWS
        or int(topology.WALK_CAP) != WALK_CAP
        or int(topology.OUTPUT_PUBLISH_CAPACITY) != OUTPUT_CAPACITY
        or int(topology.ACCEPTED_PATH_CAPACITY) != PATH_CAPACITY
    ):
        raise RuntimeError(
            "FR13 packed-walk node trust requires exact Hydra27 physical32 B1/B4"
        )
    selector = _RUNTIME_MODULE._fr13_cfwd_logit_direct_selector(
        mode=mode,
        batch_size=batch,
    )
    if selector != "diagnostic":
        raise RuntimeError(
            "FR13 packed-walk node trust requires CFWD diagnostic shadow mode"
        )
    return "node_trust"


def _fr13_cfwd_logit_direct_walk_cuda(
    topology,
    entry: dict[str, Any],
    bonus_flat,
    decisions: tuple[Any, ...],
) -> tuple[Any, Any, Any, Any, Any]:
    """Select the trusted walk only inside the existing CFWD shadow route."""
    if _REFERENCE_WALK is None:
        raise RuntimeError("FR13 packed-walk node-trust reference is unavailable")
    if _select(topology, entry) == "packed_v3":
        return _REFERENCE_WALK(topology, entry, bonus_flat, decisions)
    if _NODE_TRUST_MODULE is None:
        raise RuntimeError("FR13 packed-walk node-trust source is unavailable")
    _NODE_TRUST_MODULE.launch_packed_walk_node_trust(
        producer_contract=_producer_contract(str(entry["mode"])),
        mode=str(entry["mode"]),
        self_token=decisions[0],
        event=decisions[1],
        bonus_token=bonus_flat,
        output_tokens=entry["output_tokens"],
        output_lens=entry["output_lens"],
        accepted_path_rows=entry["accepted_path_rows"],
        accepted_lens=entry["accepted_lens"],
        last_row=entry["last_row"],
    )
    return (
        entry["output_tokens"],
        entry["output_lens"],
        entry["accepted_path_rows"],
        entry["accepted_lens"],
        entry["last_row"],
    )


def _load_node_trust(path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("FR13 packed-walk node-trust source is not regular")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != NODE_TRUST_SOURCE_SHA256:
        raise RuntimeError("FR13 packed-walk node-trust source identity drifted")
    name = "_fr13_cfwd_packed_walk_node_trust_kernel"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("FR13 packed-walk node-trust source cannot load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    if (
        getattr(module, "CANDIDATE", None) != NODE_TRUST_CANDIDATE
        or getattr(module, "CANDIDATE_SCHEMA", None) != NODE_TRUST_SOURCE_SCHEMA
    ):
        raise RuntimeError("FR13 packed-walk node-trust candidate identity drifted")
    return module


def install(module: ModuleType) -> dict[str, object]:
    """Install only when explicitly armed against the exact packed-v3 base."""
    global _NODE_TRUST_MODULE, _REFERENCE_WALK, _RUNTIME_MODULE
    armed = _selector_raw() == "1"
    if not armed:
        return runtime_contract(installed=False)
    if (
        os.environ.get("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "0").strip() != "1"
        or os.environ.get("FR13_CFWD_LOGIT_DIRECT_PRODUCTION", "0").strip()
        != "0"
    ):
        raise RuntimeError(
            "FR13 packed-walk node trust requires CFWD diagnostic shadow mode"
        )
    if (
        getattr(module, "_FR13_CFWD_LOGIT_DIRECT_CANDIDATE", None)
        != BASE_CANDIDATE
        or getattr(module, "_FR13_CFWD_LOGIT_DIRECT_SCHEMA", None)
        != BASE_CANDIDATE_SCHEMA
        or getattr(module, "_FR13_CFWD_LOGIT_DIRECT_SOURCE_SHA256", None)
        != BASE_CANDIDATE_SOURCE_SHA256
        or module._fr13_cfwd_logit_direct_integration_source_contract()
        != {
            "integration_source_schema": BASE_INTEGRATION_SOURCE_SCHEMA,
            "integration_source_sha256": BASE_INTEGRATION_SOURCE_SHA256,
        }
    ):
        raise RuntimeError("FR13 packed-walk node trust requires exact CFWD v3")
    if getattr(module, "_FR13_CFWD_PACKED_WALK_NODE_TRUST_INSTALLED", False):
        return runtime_contract(installed=True)
    source = Path(__file__).resolve().with_name(NODE_TRUST_SOURCE_NAME)
    candidate = _load_node_trust(source)
    _RUNTIME_MODULE = module
    _REFERENCE_WALK = module._fr13_cfwd_logit_direct_walk_cuda
    _NODE_TRUST_MODULE = candidate
    module._fr13_cfwd_logit_direct_walk_cuda = (
        _fr13_cfwd_logit_direct_walk_cuda
    )
    module._fr13_cfwd_packed_walk_node_trust_runtime_contract = (
        lambda: runtime_contract(installed=True)
    )
    module._FR13_CFWD_PACKED_WALK_NODE_TRUST_INSTALLED = True
    return runtime_contract(installed=True)
