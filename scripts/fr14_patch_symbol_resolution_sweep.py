#!/usr/bin/env python3
"""Resolve every symbol the FA2 patcher injects, without a GPU or a vLLM.

THE DEFECT CLASS THIS ENDS. Sites 18 and 25 are the same shape in two different
installers: a fragment that CALLS a symbol is injected under one condition
while the blob that DEFINES it is injected under another, and the two
conditions disagree. Site 18 was the serving call site without its cuda_graph
hook; site 25 was the cuda_graph hook without the blob that defines the
function it imports. Each was found by booting a GPU server for minutes and
reading the traceback.

Both are static facts about the patcher's output, so this checks them
statically: build a scratch engine tree carrying only the anchors the patcher
needs, run the real `patch_installed_vllm` against it once per arm mode, then
AST-walk every patched file and require that every `_fr13_*` symbol REFERENCED
(called, or named in an injected `from ... import`) is DEFINED somewhere the
patch actually put it.

Run for ALL THREE modes -- tier-A production, tier-B serve, and the live-A/B
shadow -- so the detector cannot itself be one-sided, which is the mistake it
exists to catch.

Usage:
  fr14_patch_symbol_resolution_sweep.py [--json out.json] [--keep-tree DIR]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr13_patch_fa2_tree_bias.py"

# The three arm modes. Every one of them is swept, because a detector that
# only knew the mode currently under repair would have passed on the day
# site 18 shipped.
MODES = {
    "tier_a_production": {"fixed32_query_tile32_b1_production": True},
    "tier_b_serve": {"fixed32_query_tile32_b1_tier_b_serve": True},
    "live_ab_shadow": {"fixed32_query_tile32_b1_live_ab": True},
}

# Symbols the injected code reaches for in modules THIS patcher never touches.
# They belong to the GDN patcher (scripts/fr10_phase4_patch_vllm_tree_gdn.py),
# so they are legitimately unresolvable here and are named rather than ignored
# -- an unadjudicated dangling symbol must stay a failure.
FOREIGN_SYMBOLS = {
    "_fr13_fixed32_observed_event_active":
        "produced by the GDN patcher; hard attribute access in live_replay",
    "_fr13_fixed32_capture_begin": "produced by the GDN patcher",
}

# ---------------------------------------------------------------- the stubs
#
# Each file carries exactly the anchors the patcher searches for, at the
# indentation it searches for them at, and nothing else. They are not
# pretending to be vLLM; they are the minimum surface the patcher binds to.

FLASH_ATTN_INTERFACE = '''import torch

DEFAULT_FA_VERSION = 2


def flash_attn_varlen_func(
    q,
    k,
    v,
    out=None,
    cu_seqlens_q=None,
    max_seqlen_q=0,
    seqused_k=None,
    max_seqlen_k=0,
    softmax_scale=None,
    causal=False,
    alibi_slopes=None,
    window_size=None,
    block_table=None,
    softcap=0.0,
    scheduler_metadata=None,
    fa_version=DEFAULT_FA_VERSION,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=0,
    return_softmax_lse=False,
    dropout_p=0.0,
    s_aux=None,
    cp_world_size=1,
):
    if fa_version == 2:
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            leftpad_k,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            num_splits,
            None,
        )
    return out
'''

TREE_ATTN = '''import ast
import os

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

logger = None


def _get_depth_counts(x):
    return x


class TreeAttentionImpl:
    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None
    ):
        num_decode_tokens = 0
        num_actual_tokens = 0
        key_cache = kv_cache
        value_cache = kv_cache
        descale_shape = ()
        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
        if decode_meta := attn_metadata.decode_metadata:
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
        return output
'''

FLASH_ATTN_BACKEND = '''import os

import torch


class FlashAttentionImpl:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None):
        num_actual_tokens = 0
        key_cache = kv_cache
        value_cache = kv_cache
        cu_seqlens_q = None
        max_seqlen_q = 0
        seqused_k = None
        max_seqlen_k = 0
        sliding_window_size = None
        block_table = None
        scheduler_metadata = None
        q_descale = k_descale = v_descale = None
        if True:
            if True:
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
        return output
'''

BATCH_INVARIANT = '''import os


class AttentionBackendEnum:
    FLASH_ATTN = "FLASH_ATTN"
    TRITON_ATTN = "TRITON_ATTN"
    TREE_ATTN = "TREE_ATTN"


def get_decode_invariant_backends():
    decode_invariant_backends = [
        AttentionBackendEnum.FLASH_ATTN,  # best supported backend
        AttentionBackendEnum.TRITON_ATTN,
    ]
    return decode_invariant_backends
'''

CUDA_GRAPH = '''import torch


class CUDAGraphWrapper:
    def replay(self, entry):
        entry.cudagraph.replay()
        return entry

    def capture(self, entry, cudagraph):
        if True:
            entry.cudagraph = cudagraph
        return entry
'''

STUBS = {
    "vllm/vllm_flash_attn/flash_attn_interface.py": FLASH_ATTN_INTERFACE,
    "vllm/v1/attention/backends/tree_attn.py": TREE_ATTN,
    "vllm/v1/attention/backends/flash_attn.py": FLASH_ATTN_BACKEND,
    "vllm/model_executor/layers/batch_invariant.py": BATCH_INVARIANT,
    "vllm/compilation/cuda_graph.py": CUDA_GRAPH,
}


def build_engine_tree(root: Path) -> Path:
    """A scratch site-packages carrying only the patcher's anchors."""
    for relative, text in STUBS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _patcher():
    spec = importlib.util.spec_from_file_location("fr14_sym_patcher", PATCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fr14_sym_patcher", module)
    spec.loader.exec_module(module)
    return module


def _fr13_names(node_iter):
    return {n for n in node_iter if n.startswith("_fr13_") or n.startswith("_FR13_")}


def analyse(path: Path):
    """(defined, referenced, imported_from) for one patched file."""
    tree = ast.parse(path.read_text())
    defined: set[str] = set()
    referenced: set[str] = set()
    imported: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name.startswith("_fr13") or alias.name.startswith("_FR13"):
                    # An IMPORT IS NOT A DEFINITION. Counting it as one is
                    # precisely the site-25 blind spot: `from tree_attn import
                    # _fr13_..._capture_end` looks locally satisfied while the
                    # blob that defines it was never inserted. The import is
                    # recorded as an obligation to be discharged against the
                    # module it names.
                    imported[alias.name] = node.module or ""
                    continue
                defined.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)

    return _fr13_names(defined), _fr13_names(referenced), imported


MODULE_TO_RELATIVE = {
    "vllm.v1.attention.backends.tree_attn":
        "vllm/v1/attention/backends/tree_attn.py",
    "vllm.v1.attention.backends.flash_attn":
        "vllm/v1/attention/backends/flash_attn.py",
    "vllm.vllm_flash_attn.flash_attn_interface":
        "vllm/vllm_flash_attn/flash_attn_interface.py",
    "vllm.compilation.cuda_graph": "vllm/compilation/cuda_graph.py",
    "vllm.model_executor.layers.batch_invariant":
        "vllm/model_executor/layers/batch_invariant.py",
}


def sweep_mode(mode: str, params: dict, keep: Path | None = None):
    patcher = _patcher()
    workdir = Path(tempfile.mkdtemp(prefix=f"fr14_symsweep_{mode}_"))
    try:
        root = build_engine_tree(workdir / "site-packages")
        patcher.patch_installed_vllm(root, **params)

        per_file = {
            relative: analyse(root / relative) for relative in STUBS
        }
        all_defined = {
            relative: defined for relative, (defined, _r, _i) in per_file.items()
        }

        dangling = []
        cross_file = []
        resolved = 0
        for relative, (defined, referenced, imported) in per_file.items():
            for name in sorted(referenced):
                if name in FOREIGN_SYMBOLS:
                    continue
                if name in defined:
                    resolved += 1
                    continue
                # an injected `from <module> import _fr13_x`
                source = imported.get(name)
                if source is not None:
                    target = MODULE_TO_RELATIVE.get(source)
                    if target and name in all_defined.get(target, set()):
                        resolved += 1
                        cross_file.append(
                            {"symbol": name, "referenced_in": relative,
                             "defined_in": target}
                        )
                        continue
                    dangling.append({
                        "symbol": name, "referenced_in": relative,
                        "imported_from": source,
                        "reason": "imported from a module that does not define it",
                    })
                    continue
                dangling.append({
                    "symbol": name, "referenced_in": relative,
                    "reason": "referenced with no definition in the patched tree",
                })
        if keep is not None:
            shutil.copytree(root, keep / mode, dirs_exist_ok=True)
        return {
            "mode": mode,
            "params": params,
            "files_patched": sorted(STUBS),
            "symbols_resolved": resolved,
            "cross_file_edges": sorted(
                cross_file, key=lambda e: (e["symbol"], e["referenced_in"])
            ),
            "dangling": dangling,
            "foreign_adjudicated": sorted(FOREIGN_SYMBOLS),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def sweep(keep: Path | None = None):
    return {
        "schema": "fr14.patch_symbol_resolution.v1",
        "modes": [sweep_mode(mode, params, keep) for mode, params in MODES.items()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--keep-tree", type=Path)
    args = ap.parse_args()
    report = sweep(args.keep_tree)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n")
    failed = False
    for row in report["modes"]:
        status = "OK  " if not row["dangling"] else "DANGLING"
        print(
            f"[{status}] {row['mode']:<20s} "
            f"{row['symbols_resolved']:>4d} symbols resolved, "
            f"{len(row['cross_file_edges'])} cross-file, "
            f"{len(row['dangling'])} dangling"
        )
        for edge in row["cross_file_edges"]:
            print(f"         cross-file: {edge['symbol']} "
                  f"({edge['referenced_in'].split('/')[-1]} -> "
                  f"{edge['defined_in'].split('/')[-1]})")
        for bad in row["dangling"]:
            failed = True
            print(f"         DANGLING: {bad['symbol']} in "
                  f"{bad['referenced_in']}: {bad['reason']}")
    print()
    print(f"{len(report['modes'])} arm modes swept")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
