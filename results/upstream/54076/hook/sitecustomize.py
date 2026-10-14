# NOTE: this file shadows /usr/lib/python3.12/sitecustomize.py (which only installs the
# apport crash handler). That preamble is replicated verbatim below so shadowing changes nothing.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

# Item E / #54076 — ADDITIVE, LOGGING-ONLY instrumentation.
# Placed on PYTHONPATH (NOT inside the venv; nothing under .venv is modified).
#
# It wraps two functions. Each wrapper calls the ORIGINAL first, unchanged, and then only
# reads attributes and prints. It assigns nothing on any vLLM object and never raises
# (every print is inside try/except). It therefore cannot change engine behaviour.
#
# Why it is needed: stock vLLM 0.28.0 logs `cache_config.block_size` only in the config repr
# emitted BEFORE KV-cache initialisation. The value that decides #54076 is the one written
# AFTER, at vllm/v1/engine/core.py:322-324
#     vllm_config.cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups)
# and read at vllm/v1/core/sched/scheduler.py:392
#     block_size = self.cache_config.block_size          # inside _mamba_block_aligned_split
# No stock INFO or DEBUG line prints it. EngineCore.get_kv_cache_group_metadata (core.py:419)
# would, but it is dead code in 0.28.0 (no caller, no HTTP route).
#
# The hook prints id(self.cache_config) so that the object it reports is demonstrably the
# same object `_mamba_block_aligned_split` reads through `self.cache_config`.
import os
import sys

_TAG = "[[E54076]]"
_TARGETS = ("vllm.v1.core.sched.scheduler", "vllm.v1.engine.core")


def _emit(*parts):
    try:
        sys.stderr.write(f"{_TAG} pid={os.getpid()} " + " ".join(str(p) for p in parts) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _dump_scheduler(sched):
    try:
        cc = sched.cache_config
        _emit("SCHEDULER_CACHE_CONFIG",
              f"id(cache_config)={id(cc):#x}",
              f"cache_config.block_size={cc.block_size}",
              f"cache_config.mamba_block_size={cc.mamba_block_size}",
              f"cache_config.mamba_cache_mode={cc.mamba_cache_mode}",
              f"cache_config.enable_prefix_caching={cc.enable_prefix_caching}",
              f"cache_config.mamba_page_size_padded={cc.mamba_page_size_padded}",
              f"cache_config.num_gpu_blocks={cc.num_gpu_blocks}")
    except Exception as ex:
        _emit("SCHEDULER_CACHE_CONFIG_ERROR", repr(ex))
    try:
        _emit("SCHEDULER_FIELDS",
              f"scheduler.block_size={getattr(sched, 'block_size', None)}",
              f"scheduler.hash_block_size={getattr(sched, 'hash_block_size', None)}",
              f"need_mamba_block_aligned_split={getattr(sched, 'need_mamba_block_aligned_split', None)}",
              f"mamba_partial_cache_hit={getattr(sched, 'mamba_partial_cache_hit', None)}",
              f"has_mamba_layers={getattr(sched, 'has_mamba_layers', None)}",
              f"use_eagle={getattr(sched, 'use_eagle', None)}")
    except Exception as ex:
        _emit("SCHEDULER_FIELDS_ERROR", repr(ex))
    try:
        groups = sched.kv_cache_config.kv_cache_groups
        mamba_bs = None
        for i, g in enumerate(groups):
            spec = g.kv_cache_spec
            _emit("KV_GROUP", f"idx={i}",
                  f"spec={type(spec).__name__}",
                  f"block_size={spec.block_size}",
                  f"page_size_bytes={getattr(spec, 'page_size_bytes', None)}",
                  f"prefix_cacheable={getattr(spec, 'prefix_cacheable', 'ABSENT_IN_0.28.0')}",
                  f"n_layers={len(g.layer_names)}",
                  f"first_layer={g.layer_names[0] if g.layer_names else None}")
            if type(spec).__name__ == "MambaSpec":
                mamba_bs = spec.block_size
        cbs = sched.cache_config.block_size
        if mamba_bs is not None:
            _emit("VERDICT",
                  f"cache_config.block_size={cbs}",
                  f"MambaSpec.block_size={mamba_bs}",
                  f"divergent={cbs < mamba_bs}")
        else:
            _emit("VERDICT", f"cache_config.block_size={cbs}", "MambaSpec.block_size=NONE_FOUND")
    except Exception as ex:
        _emit("KV_GROUP_ERROR", repr(ex))


def _patch_scheduler(mod):
    try:
        cls = mod.Scheduler
        if getattr(cls, "_e54076_wrapped", False):
            return
        orig = cls.__init__

        def __init__(self, *a, **kw):
            orig(self, *a, **kw)          # original, unchanged
            _dump_scheduler(self)          # read-only
        __init__.__wrapped__ = orig
        cls.__init__ = __init__
        cls._e54076_wrapped = True
        _emit("HOOK_INSTALLED", "Scheduler.__init__")
    except Exception as ex:
        _emit("HOOK_ERROR", "Scheduler", repr(ex))


def _patch_core(mod):
    try:
        cls = mod.EngineCore
        if getattr(cls, "_e54076_wrapped", False):
            return
        orig = cls._initialize_kv_caches

        def _initialize_kv_caches(self, vllm_config):
            out = orig(self, vllm_config)   # original, unchanged
            try:
                cc = vllm_config.cache_config
                _emit("POST_INIT_KV_CACHES",
                      f"id(cache_config)={id(cc):#x}",
                      f"cache_config.block_size={cc.block_size}",
                      f"cache_config.mamba_block_size={cc.mamba_block_size}",
                      f"n_groups={len(out.kv_cache_groups)}")
                for i, g in enumerate(out.kv_cache_groups):
                    s = g.kv_cache_spec
                    _emit("POST_INIT_GROUP", f"idx={i}", f"spec={type(s).__name__}",
                          f"block_size={s.block_size}")
            except Exception as ex:
                _emit("POST_INIT_ERROR", repr(ex))
            return out
        _initialize_kv_caches.__wrapped__ = orig
        cls._initialize_kv_caches = _initialize_kv_caches
        cls._e54076_wrapped = True
        _emit("HOOK_INSTALLED", "EngineCore._initialize_kv_caches")
    except Exception as ex:
        _emit("HOOK_ERROR", "EngineCore", repr(ex))


_PATCHERS = {
    "vllm.v1.core.sched.scheduler": _patch_scheduler,
    "vllm.v1.engine.core": _patch_core,
}


class _PostImportFinder:
    """Standard post-import hook: delegate to the real finder, then wrap exec_module."""

    def find_spec(self, name, path=None, target=None):
        if name not in _TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:
                spec = None
            if spec is None or spec.loader is None:
                continue
            loader = spec.loader
            orig_exec = loader.exec_module
            patcher = _PATCHERS[name]

            def exec_module(module, _orig=orig_exec, _p=patcher):
                _orig(module)   # original module execution, unchanged
                _p(module)
            try:
                loader.exec_module = exec_module
            except Exception:
                pass
            return spec
        return None


try:
    sys.meta_path.insert(0, _PostImportFinder())
    _emit("SITECUSTOMIZE_LOADED", "argv0=" + (sys.argv[0] if sys.argv else "?"))
except Exception:
    pass
