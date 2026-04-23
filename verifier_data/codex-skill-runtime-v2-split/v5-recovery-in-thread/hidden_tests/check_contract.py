from __future__ import annotations

import importlib.util
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / "_shared" / "contract_checks.py"
spec = importlib.util.spec_from_file_location("csr_contract_checks", SHARED)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
run_checks = module.run_checks
