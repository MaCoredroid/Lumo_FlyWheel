"""Import adapter for the unchanged pinned author benchmark.

Reports its numerical characterization and isolated verifier timings. This is
not a LumoTree comparison, a commit-inclusive benchmark, or a serving result.
"""
import importlib.util
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = "sglang.srt.layers.attention.fla"
for i in range(1, len(PREFIX.split(".")) + 1):
    name = ".".join(PREFIX.split(".")[:i])
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    sys.modules[name] = pkg
for filename in ["gdn_tree_fused", "gdn_tree_triton"]:
    name = PREFIX + "." + filename
    spec = importlib.util.spec_from_file_location(name, ROOT / "weaver" / (filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
runpy.run_path(str(ROOT / "weaver/bench_gdn_tree_bf16.py"), run_name="__main__")
