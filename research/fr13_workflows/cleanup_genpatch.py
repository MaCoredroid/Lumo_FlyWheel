#!/usr/bin/env python3
"""Offline: run a patcher against a PRISTINE vllm copy, dump patched target files.
GPU-free (patch fns are pure string rewrites). Used for the cleanup OFF-equivalence diff.
Usage: cleanup_genpatch.py <patcher.py> <pristine_vllm_dir> <out_dir>
"""
import importlib.util, os, sys, shutil, tempfile
from pathlib import Path

patcher_path, vsrc, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
work = Path(tempfile.mkdtemp(prefix="patchwork_"))
shutil.copytree(vsrc, work / "vllm")
os.environ.setdefault("FR13_APC_ZERO_MAMBA_ON_ALLOC", "0")
os.environ.setdefault("FR13_APC_PREFILL_CARRY_TRACE", "0")

spec = importlib.util.spec_from_file_location("patcher", patcher_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

inbase = "/usr/local/lib/python3.12/dist-packages/vllm"
for name in dir(m):
    v = getattr(m, name)
    if name.endswith("_PATH") and isinstance(v, Path):
        s = str(v)
        if inbase in s:
            setattr(m, name, work / "vllm" / s.split(inbase + "/", 1)[1])

try:
    rc = m.main()
except SystemExit as e:
    rc = e.code
except Exception as e:
    print(f"[gen] main raised {type(e).__name__}: {e}", file=sys.stderr); rc = -99
print(f"[gen] patcher main rc={rc}", file=sys.stderr)

out_dir.mkdir(parents=True, exist_ok=True)
targets = ["model_executor/layers/mamba/gdn_linear_attn.py",
           "v1/attention/backends/gdn_attn.py",
           "v1/worker/gpu_model_runner.py",
           "v1/worker/mamba_utils.py",
           "v1/core/single_type_kv_cache_manager.py"]
for t in targets:
    src = work / "vllm" / t
    if src.is_file():
        dst = out_dir / t.replace("/", "__")
        dst.write_text(src.read_text())
        print(f"[gen] wrote {dst.name} ({dst.stat().st_size}B)", file=sys.stderr)
shutil.rmtree(work, ignore_errors=True)
