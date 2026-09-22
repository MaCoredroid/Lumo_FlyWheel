#!/usr/bin/env python3
"""Emit the FROZEN 18-cell E1 order (e1/E1_FREEZE.md): 3 arms x 2 batch conditions x 3 paired blocks, balanced arm order
per block and alternating batch order per arm position. Deterministic; the output's sha256 is the campaign's order record.
Usage: e1_make_cells.py <out.json>"""
import json, sys, hashlib
BLOCKS = [["native-5", "native-11", "tree"], ["tree", "native-5", "native-11"], ["native-11", "tree", "native-5"]]
BATCH_ORDER = [["B1", "B4"], ["B4", "B1"], ["B1", "B4"]]   # first arm B1 then B4, second B4 then B1, third B1 then B4
TREE_SPEC = '{"method":"qwen3_5_mtp","num_speculative_tokens":9,"speculative_token_tree":"[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"}'   # exact descriptor of the qualified tree pilots (docker env SPEC_CONFIG of out-20260922T084319Z / out-20260922T082815Z)
SPEC = {"native-5": '{"method":"qwen3_5_mtp","num_speculative_tokens":5}', "native-11": '{"method":"qwen3_5_mtp","num_speculative_tokens":11}', "tree": TREE_SPEC}
ARMS = {"native-5": {"launcher": "e1_native_launch.v2.sh", "descriptor": "native chain-5 — PATCHED-RUNNER NATIVE BASELINE: stock MTP method (naive_mtp decode mode, FR10_ENABLE_TREE_GDN=0, no tree descriptor; stock conv/update + rejection fallback) with the same recorder/timer instrumentation as the tree arm; tree-branch guards exclude the baked replay/conv paths (not a global feature-OFF claim)", "num_speculative_tokens": 5},
        "native-11": {"launcher": "e1_native_launch.v2.sh", "descriptor": "native chain-11 — LONGER-CHAIN CONTROL (not depth-matched to cat10); same patched-runner native baseline route as native-5", "num_speculative_tokens": 11},
        "tree": {"launcher": "e7a_capture_launch.v7.sh", "descriptor": "cat10 tree (parents [-1,0,1,1,2,2,4,4,6,6], 10 nodes, max depth 5), TREE_ATTN, FR13_REPLAY_ROUTE=1, EAGER_PACK=1, TREE_CONV_FUSED=1, RUNROW_INIT=1, KV policy B (remap 1 / reorder 0 / syncfree 1)", "num_speculative_tokens": 9}}
FROZEN = {"temperature": 0, "seed": 20260921, "max_tokens_timed": 128, "max_tokens_warmup": 32, "ignore_eos": False, "enforce_eager": 1, "prefix_cache": "OFF (FR13_ENABLE_APC=0; --no-enable-prefix-caching verified post-boot)",
          "scheduling": "explicit --no-async-scheduling (VLLM_SYNC_SCHED=1)", "gpu_util": 0.6, "max_model_len": 16384, "attention_backend": {"tree": "TREE_ATTN", "native-5": "FLASH_ATTN", "native-11": "FLASH_ATTN"},
          "kv_policy_tree": {"FR13_ATTN_KV_REMAP": "1", "FR13_SLOT_REORDER": "0", "FR13_KV_REMAP_SYNCFREE": "1"}, "api_token_ids": "return_tokens_as_token_ids=True (mandatory)", "spec_config_exact": SPEC, "launchers": {"tree": "e7a_capture_launch.v7.sh", "native": "e1_native_launch.v2.sh (patched-runner native baseline)"}, "no_heavy_capture": "FR10_METRICS=0; no draft/LCP/sampler traces; no capture sinks; recorder + forward timer only",
          "prompts": "the 8 frozen PILOT prefixes of frozen_prefixes.json (sha-bound); B1 = 8 sequential single requests; B4 = cohort A (pilot 1-4) then cohort B (pilot 5-8), submitted together",
          "warmup": {"B1": "two untimed single requests of pilot prompt 1, 32 tokens", "B4": "one untimed ACTUAL four-request cohort of pilot prompts 1-4, 32 tokens each"},
          "coverage_floors": {"B1": "all 8 prompts each >= 1 valid retained interval (N1 >= 8)", "B4": "each of the 2 cohorts >= 1 complete four-request same-cohort interval (N4 >= 2)"},
          "precision_target": 0.10, "idle_policy": "retain every valid pure same-cohort interval after warm-up; >1.5 s reported as diagnostic only"}
cells = []; n = 0
for b, arms in enumerate(BLOCKS):
    for pos, arm in enumerate(arms):
        for batch in BATCH_ORDER[pos]:
            n += 1; cells.append({"index": n, "block": b + 1, "position_in_block": pos + 1, "arm": arm, "batch": batch, "max_num_seqs": (1 if batch == "B1" else 4), **ARMS[arm], "spec_config": SPEC[arm], "status": "PLANNED (not qualified — not timed until the route qualification manifest says so)"})
doc = {"schema": "e1.cells.v1", "frozen_from": "e1/E1_FREEZE.md", "n_cells": len(cells), "frozen_settings": FROZEN, "cells": cells}
s = json.dumps(doc, indent=1); open(sys.argv[1], "w").write(s); print("cells:", len(cells), "sha256", hashlib.sha256(s.encode()).hexdigest()[:16])
