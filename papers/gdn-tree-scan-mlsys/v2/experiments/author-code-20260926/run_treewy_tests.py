"""Run unchanged author component tests through a package-import adapter.

This is kernel/reference qualification, not an installed vLLM integration test.
The adapter binds only the pinned modules and does not alter kernel arithmetic,
test inputs, tolerances, or assertions. Full layer tests need the author's stack.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import triton

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

def bind(name, file):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            sys.modules[parent] = pkg
    spec = importlib.util.spec_from_file_location(name, ROOT / "treewy" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

bind("vllm.v1.spec_decode.tree_spec", "tree_spec.py")
bind("vllm.third_party.flash_linear_attention.ops.tree_wy_ref", "tree_wy_ref.py")
bind("vllm.third_party.flash_linear_attention.ops.tree_wy_triton", "tree_wy_triton.py")

environment = dict(torch=torch.__version__, triton=triton.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                   capability=torch.cuda.get_device_capability(),
                   scope="Unchanged author kernel/reference tests; import adapter only")
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
tests = [str(ROOT / "treewy/test_tree_wy_chain_capture.py")]
for test in ["test_tree_capture_forward_matches_reference",
             "test_tree_capture_commits_branch_leaf_state",
             "test_tree_capture_sentinel_skips_commit",
             "test_maskfree_equals_masked_on_a_chain",
             "test_tree_capture_replays_with_new_leaf"]:
    tests.append(str(ROOT / "treewy/test_treewy_tree_capture.py") + "::" + test)
for test in ["test_tree_wy_matches_naive", "test_chain_is_special_case",
             "test_batched_tree_wy_matches_naive"]:
    tests.append(str(ROOT / "treewy/test_tree_wy_ref.py") + "::" + test)
raise SystemExit(pytest.main(["-v", "--tb=short", "--junitxml=" + str(OUT / "treewy-author.xml"), *tests]))
