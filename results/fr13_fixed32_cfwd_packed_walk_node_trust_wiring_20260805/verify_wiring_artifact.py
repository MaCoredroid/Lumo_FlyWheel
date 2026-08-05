#!/usr/bin/env python3
"""Verify source pins and fail-closed wiring for the node-trust shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = {
    "candidate": "scripts/fr13_cfwd_packed_walk_node_trust_kernel.py",
    "runtime_overlay": (
        "scripts/fr13_cfwd_packed_walk_node_trust_runtime_overlay.py"
    ),
    "cfwd_runtime_wrapper": "scripts/fr13_device_multidraft_cfwd_packed_v3.py",
    "launcher": "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "runtime_manifest": "scripts/fr13_runtime_manifest.py",
    "base_live_runner": "scripts/fr13_run_b1_cfwd_logit_direct_live_gate.sh",
    "wiring_runner": (
        "scripts/fr13_run_b1_cfwd_packed_walk_node_trust_live_gate.sh"
    ),
    "wiring_test": (
        "tests/test_fr13_fixed32_cfwd_packed_walk_node_trust_wiring.py"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact = args.artifact.resolve()
    contract = json.loads(
        (artifact / "runtime_contract.json").read_text(encoding="ascii")
    )
    if (
        contract.get("schema")
        != "fr13.fixed32.cfwd_packed_walk.node_trust.wiring.v1"
        or contract.get("status") != "static_ready_for_real_swe_byte_gate"
        or contract.get("candidate_default_off") is not True
        or contract.get("reference_always_served") is not True
        or contract.get("runtime_speedup_claimed") is not False
        or contract.get("gpu_execution") is not False
        or contract.get("shadow_comparator")
        != "_fr13_cfwd_logit_direct_compare"
        or contract.get("admission", {}).get("mode") != "hydra27_fixed32"
        or contract.get("admission", {}).get("batches") != [1, 4]
        or contract.get("admission", {}).get("physical_rows") != 32
    ):
        raise SystemExit("node-trust wiring contract drifted")
    observed = {name: _sha256(repo / path) for name, path in FILES.items()}
    if observed != contract.get("source_sha256"):
        raise SystemExit("node-trust wiring source hashes drifted")
    overlay = (repo / FILES["runtime_overlay"]).read_text(encoding="ascii")
    wrapper = (repo / FILES["cfwd_runtime_wrapper"]).read_text(encoding="ascii")
    runner = (repo / FILES["wiring_runner"]).read_text(encoding="ascii")
    launcher = (repo / FILES["launcher"]).read_text(encoding="utf-8")
    selector = contract["selector_env"]
    if not (
        'os.environ.get(SELECTOR_ENV, "0")' in overlay
        and observed["runtime_overlay"] in wrapper
        and f"{selector}=1" in runner
        and f'-e {selector}="${selector}"' in launcher
        and "_fr13_cfwd_logit_direct_compare" in contract["shadow_comparator"]
    ):
        raise SystemExit("node-trust default-off runner wiring drifted")
    print(
        json.dumps(
            {
                "schema": (
                    "fr13.fixed32.cfwd_packed_walk.node_trust.wiring.verify.v1"
                ),
                "status": "PASS",
                "source_bindings": len(observed),
                "candidate_default_off": True,
                "admitted_batches": [1, 4],
                "reference_always_served": True,
                "gpu_execution": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
