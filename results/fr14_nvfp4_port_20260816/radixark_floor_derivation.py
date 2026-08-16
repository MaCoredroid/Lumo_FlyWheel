#!/usr/bin/env python3
"""Re-derive the FR14 hardware floor for the RadixArk aggressive NVFP4 arm from
the ACTUAL on-disk tensor ledger. No FR13 or FR14 constant is scaled: every byte
term below is a sum of real safetensors tensor spans read out of
/models/qwen3.8-27b-nvfp4-radixark by radixark_ledger.py.

Emits radixark_floor_ledger.json (schema-parallel to floor_ledger.json, the
unsloth arm's) and radixark_floor_derivation.md (the arithmetic, shown).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = json.load(open(HERE / "radixark_ledger.json"))
BW = 273_000_000_000  # bytes/s, the FR13 measured GB10 unified-memory bandwidth

rows = LEDGER["rows"]
by_name = {r["name"]: r for r in rows}


def total(pred):
    return sum(r["bytes"] for r in rows if pred(r["name"]))


# ---------------------------------------------------------------- components
# Same semantics as the unsloth arm's floor_ledger.json: "model.language_model.*
# minus embed_tokens", i.e. the 64 decoder layers + the final norm. Verified:
# layer indices 0..63 (64 distinct) plus model.language_model.norm.weight.
TARGET = total(lambda n: n.startswith("model.language_model.") and "embed_tokens" not in n)

# lm_head as RadixArk actually ships it: the ModelOpt NVFP4 4-tensor set.
NVFP4_HEAD = total(lambda n: n.startswith("lm_head"))

# The BF16 head the unsloth arm ended up serving (its FP8 head had to be
# dequantised to load at all), kept here purely as the comparison baseline.
BF16_HEAD = by_name["model.language_model.embed_tokens.weight"]["bytes"]  # same [248320,5120] bf16

# FP8 per-channel head = the plan-B fallback. Corroborated by the unsloth
# lm_head surgery's own byte delta: that surgery grew the shard by
# 1_270_901_784 B replacing the FP8 head with the BF16 one, so the FP8 head was
# 2_542_796_800 - 1_270_901_784 = 1_271_895_016 B on disk (the 24 B residual is
# safetensors header text, not payload).
FP8_HEAD = 248320 * 5120 * 1 + 248320 * 1 * 2  # weight F8_E4M3 + weight_scale BF16

# MTP: 15 tensors, all BF16, byte-identical in count and shape to the unsloth arm.
MTP_PER_PASS = total(lambda n: n.startswith("mtp."))
MTP_PASSES = 5  # 1 initial + 4 post-root-graph, the production invariant

# K64 draft head = a 65536-row slice of lm_head.
K64 = 65536
HID = 5120
DRAFT_BF16 = K64 * HID * 2
# Row-sliced straight out of the NVFP4 head: the block scales are [out, in/16],
# so index_select on rows works for the weight AND the scale.
DRAFT_NVFP4 = K64 * (HID // 2) + K64 * (HID // 16)
DRAFT_FP8 = K64 * HID * 1 + K64 * 2

# MTP FP8 requant lever (projection, not a built artefact): mirror the served
# 3.6 MTP recipe -- FP8 e4m3 weights with 128x128 F32 block scales on the 2-D
# matrices, 1-D norms left BF16.
mtp_rows = [r for r in rows if r["name"].startswith("mtp.")]
mtp_fp8 = 0
for r in mtp_rows:
    if len(r["shape"]) == 2:
        o, i = r["shape"]
        mtp_fp8 += o * i  # e4m3
        mtp_fp8 += -(-o // 128) * (-(-i // 128)) * 4  # F32 block scales
    else:
        mtp_fp8 += r["bytes"]  # norms stay BF16
MTP_FP8_PER_PASS = mtp_fp8

components = {
    "target_model": {
        "bytes": TARGET,
        "source": "summed tensor spans of model.language_model.* minus embed_tokens (64 decoder layers + final norm)",
        "tensors": sum(1 for r in rows if r["name"].startswith("model.language_model.") and "embed_tokens" not in r["name"]),
    },
    "nvfp4_head": {
        "bytes": NVFP4_HEAD,
        "source": "summed tensor spans of lm_head.* as shipped (ModelOpt NVFP4 4-tensor set)",
        "tensors": {r["name"]: {"dtype": r["dtype"], "shape": r["shape"], "bytes": r["bytes"]}
                    for r in rows if r["name"].startswith("lm_head")},
    },
    "fp8_head": {
        "bytes": FP8_HEAD,
        "source": "plan-B: lm_head dequantised to FP8 per-channel (weight F8_E4M3 [248320,5120] + weight_scale BF16 [248320,1]); corroborated by the unsloth lm_head surgery byte delta",
        "hypothetical": True,
    },
    "full_bf16_head": {
        "bytes": BF16_HEAD, "rows": 248320, "hidden_size": HID, "element_bytes": 2,
        "source": "the unsloth arm's served head, kept as the comparison baseline",
        "hypothetical": True,
    },
    "mtp_forward": {
        "bytes": MTP_PER_PASS * MTP_PASSES, "bytes_per_pass": MTP_PER_PASS,
        "passes": MTP_PASSES, "initial_passes": 1, "post_root_graph_passes": 4,
        "source": "summed tensor spans of mtp.* (15 tensors, all BF16)",
    },
    "mtp_forward_fp8_requant": {
        "bytes": MTP_FP8_PER_PASS * MTP_PASSES, "bytes_per_pass": MTP_FP8_PER_PASS,
        "passes": MTP_PASSES,
        "source": "PROJECTION: 2-D mtp matrices at FP8 e4m3 + 128x128 F32 block scales, 1-D norms left BF16",
        "hypothetical": True,
    },
    "draft_64k_bf16_head": {"bytes": DRAFT_BF16, "rows": K64, "hidden_size": HID, "element_bytes": 2},
    "draft_64k_nvfp4_head": {
        "bytes": DRAFT_NVFP4, "rows": K64,
        "source": "65536-row slice of the NVFP4 head: 65536*2560 packed U8 + 65536*320 F8_E4M3 block scales (row-aligned by construction)",
    },
    "draft_64k_fp8_head": {"bytes": DRAFT_FP8, "rows": K64, "hypothetical": True},
}

# ---------------------------------------------------------------- scenarios
def scen(name, terms, formula, **extra):
    b = sum(terms)
    d = {"component_formula": formula, "mandatory_weight_bytes": b,
         "mandatory_weight_floor_ms": b * 1000 / BW, "nonweight_costs_included": False}
    d.update(extra)
    return name, d


scenarios = dict([
    scen("root_64k_nvfp4_head_bf16_draft",
         [TARGET, NVFP4_HEAD, MTP_PER_PASS * MTP_PASSES, 5 * DRAFT_BF16],
         "target_model + nvfp4_head + mtp_forward + 5 * draft_64k_bf16_head",
         drafter_head_bytes=5 * DRAFT_BF16, shippable_now=True,
         note="PHASE 1 -- what this arm can serve the moment the lm_head patch boots: DVK dequants the sliced rows to BF16, leaving the sealed BF16 GEMV units and the 128-block map untouched."),
    scen("root_64k_nvfp4_head_nvfp4_draft",
         [TARGET, NVFP4_HEAD, MTP_PER_PASS * MTP_PASSES, 5 * DRAFT_NVFP4],
         "target_model + nvfp4_head + mtp_forward + 5 * draft_64k_nvfp4_head",
         drafter_head_bytes=5 * DRAFT_NVFP4, shippable_now=False,
         note="PHASE 2 -- needs an FP4 draft-head GEMV unit; its own byte gate."),
    scen("root_64k_nvfp4_head_nvfp4_draft_mtp_fp8",
         [TARGET, NVFP4_HEAD, MTP_FP8_PER_PASS * MTP_PASSES, 5 * DRAFT_NVFP4],
         "target_model + nvfp4_head + mtp_forward_fp8_requant + 5 * draft_64k_nvfp4_head",
         drafter_head_bytes=5 * DRAFT_NVFP4, shippable_now=False,
         note="PHASE 2 + the MTP-FP8 local-requant lever stacked on top."),
    scen("root_64k_fp8_head_bf16_draft",
         [TARGET, FP8_HEAD, MTP_PER_PASS * MTP_PASSES, 5 * DRAFT_BF16],
         "target_model + fp8_head + mtp_forward + 5 * draft_64k_bf16_head",
         drafter_head_bytes=5 * DRAFT_BF16, shippable_now=False,
         note="PLAN B -- if NVFP4 lm_head cannot serve, dequant the head to FP8 per-channel (NOT BF16)."),
    scen("root_64k_bf16_head_bf16_draft",
         [TARGET, BF16_HEAD, MTP_PER_PASS * MTP_PASSES, 5 * DRAFT_BF16],
         "target_model + full_bf16_head + mtp_forward + 5 * draft_64k_bf16_head",
         drafter_head_bytes=5 * DRAFT_BF16, shippable_now=False,
         note="WORST CASE -- the unsloth arm's head treatment applied to the aggressive backbone; shows how little the backbone alone buys."),
    scen("legacy_target_plus_verifier_head",
         [TARGET, NVFP4_HEAD],
         "target_model + nvfp4_head",
         is_full_speculative_step_floor=False),
])

UNSLOTH_ROOT64K_MS = 102.479937172  # floor_ledger.json, the conservative arm

out = {
    "schema": "fr13.speculative_step_weight_ledger.v3",
    "campaign": "fr14_nvfp4_port_20260816",
    "arm": "radixark_aggressive_nvfp4",
    "model_root": LEDGER["model_dir"],
    "revision": "554ebba9b5f1b79dc11246341960360e6ef05ef4",
    "bandwidth_bytes_per_s": BW,
    "formula": "floor_ms = mandatory_weight_bytes * 1000 / bandwidth_bytes_per_s",
    "derivation": ("every byte term is a SUM of real safetensors tensor spans in "
                   "/models/qwen3.8-27b-nvfp4-radixark (2194 tensors, 21,921,428,072 B "
                   "total, == the repo's own qualification.json "
                   "output_indexed_payload_bytes); no FR13 or FR14 constant was scaled "
                   "and the staged design-note estimate was not trusted"),
    "ledger_total_bytes": LEDGER["total_bytes"],
    "ledger_tensor_count": LEDGER["n_tensors"],
    "excluded_from_floor": {
        "embed_tokens_bytes": by_name["model.language_model.embed_tokens.weight"]["bytes"],
        "vision_tower_bytes": LEDGER["buckets"]["visual"]["bytes"],
        "why": "neither is read on a decode step",
    },
    "production_invariants": {
        "drafter_head_passes_per_event": 5,
        "initial_mtp_forward_passes_per_event": 1,
        "post_root_graph_mtp_forward_passes_per_event": 4,
        "total_mtp_forward_passes_per_event": MTP_PASSES,
    },
    "components": components,
    "scenarios": scenarios,
    "comparison": {
        "unsloth_conservative_root_64k_ms": UNSLOTH_ROOT64K_MS,
        "deltas_ms_vs_unsloth": {
            k: round(UNSLOTH_ROOT64K_MS - v["mandatory_weight_floor_ms"], 4)
            for k, v in scenarios.items() if k != "legacy_target_plus_verifier_head"
        },
    },
    "slo_cap_provisional": True,
}
json.dump(out, open(HERE / "radixark_floor_ledger.json", "w"), indent=1, sort_keys=True)


# ---------------------------------------------------------------- markdown
def gb(b):
    return f"{b/1e9:.4f}"


def ms(b):
    return b * 1000 / BW


L = []
A = L.append
A("# FR14 — RadixArk aggressive NVFP4 arm: hardware floor, re-derived from the on-disk ledger")
A("")
A("Generated by `radixark_floor_derivation.py` from `radixark_ledger.json`, which is")
A("built by reading the **safetensors headers only** of")
A("`/models/qwen3.8-27b-nvfp4-radixark` @ `554ebba9b5f1b79dc11246341960360e6ef05ef4`.")
A("")
A("**No constant was scaled.** The staged design note's estimate was not trusted; every")
A("term below is a sum of real tensor spans. The ledger's own total is")
A(f"`{LEDGER['total_bytes']:,}` B across `{LEDGER['n_tensors']}` tensors, which equals the repo's")
A("`qualification.json:checkpoint.output_indexed_payload_bytes` exactly — the ledger and")
A("the publisher's audit agree byte-for-byte.")
A("")
A(f"Bandwidth: **{BW:,} B/s** (the FR13 measured GB10 unified-memory figure, unchanged).")
A("")
A(f"    floor_ms = mandatory_weight_bytes × 1000 / {BW:,}")
A("")
A("## 1. The ledger, bucketed")
A("")
A("| bucket | tensors | bytes | GB | in the floor? |")
A("|---|---:|---:|---:|---|")
order = [("target", "**yes** — the 64 decoder layers + final norm"),
         ("lm_head", "**yes** — the verifier head, read once per step"),
         ("mtp", "**yes** — ×5 passes per speculative event"),
         ("embed", "no — not read on a decode step"),
         ("visual", "no — not read on a decode step")]
for k, why in order:
    b = LEDGER["buckets"][k]
    A(f"| `{k}` | {b['tensors']} | {b['bytes']:,} | {gb(b['bytes'])} | {why} |")
A(f"| **total** | **{LEDGER['n_tensors']}** | **{LEDGER['total_bytes']:,}** | **{gb(LEDGER['total_bytes'])}** | |")
A("")
A("Target-bucket composition by dtype — this is what the aggressive recipe actually did:")
A("")
A("| dtype | tensors | bytes | GB |")
A("|---|---:|---:|---:|")
for k, v in LEDGER["target_bytes_by_dtype"].items():
    A(f"| `{k}` | {v['tensors']} | {v['bytes']:,} | {gb(v['bytes'])} |")
A("")
A("Reading that table: `U8` is the packed NVFP4 weight payload — 192 layers here, 193")
A("counting the head, matching `tensor-audit.json`'s `NVFP4: 193`. The 400 `F8_E4M3`")
A("tensors split into 208 FP8 weights (== `tensor-audit.json`'s `FP8: 208`) and 192 NVFP4")
A("per-16 block scales. The 800 `F32` tensors are all 0-dim scalar global scales — 3.2 kB")
A("in total, i.e. noise — and with the head's 2 they make 802 = 401 `input_scale` + 193")
A("`weight_scale_2` + 208 FP8 per-tensor `weight_scale`. `BF16` is what the recipe left")
A("unconverted (GDN `in_proj_a`/`in_proj_b`, conv, norms).")
A("")
A("## 2. Components")
A("")
A("| component | bytes | GB | how it is obtained |")
A("|---|---:|---:|---|")
A(f"| `target_model` | {TARGET:,} | {gb(TARGET)} | sum of `model.language_model.*` minus `embed_tokens` — layer indices 0..63 (64 distinct) + `model.language_model.norm.weight`; same semantics as the unsloth arm's ledger |")
A(f"| `nvfp4_head` | {NVFP4_HEAD:,} | {gb(NVFP4_HEAD)} | sum of the 4 shipped `lm_head.*` tensors |")
A(f"| `mtp_forward` (per pass) | {MTP_PER_PASS:,} | {gb(MTP_PER_PASS)} | sum of the 15 `mtp.*` tensors, all BF16 |")
A(f"| `draft_64k_bf16_head` | {DRAFT_BF16:,} | {gb(DRAFT_BF16)} | 65536 × 5120 × 2 |")
A(f"| `draft_64k_nvfp4_head` | {DRAFT_NVFP4:,} | {gb(DRAFT_NVFP4)} | 65536 × 2560 (packed U8) + 65536 × 320 (F8 block scales) |")
A(f"| `fp8_head` *(plan B)* | {FP8_HEAD:,} | {gb(FP8_HEAD)} | 248320×5120×1 + 248320×1×2 |")
A(f"| `full_bf16_head` *(baseline)* | {BF16_HEAD:,} | {gb(BF16_HEAD)} | 248320 × 5120 × 2 |")
A(f"| `mtp_forward_fp8_requant` *(projection)* | {MTP_FP8_PER_PASS:,} | {gb(MTP_FP8_PER_PASS)} | 2-D mtp matrices at e4m3 + 128×128 F32 block scales; 1-D norms left BF16 |")
A("")
A("### The head, tensor by tensor")
A("")
A("| tensor | dtype | shape | bytes |")
A("|---|---|---|---:|")
for r in LEDGER["lm_head_tensors"]:
    A(f"| `{r['name']}` | {r['dtype']} | {r['shape']} | {r['bytes']:,} |")
A(f"| **total** | | | **{NVFP4_HEAD:,}** |")
A("")
A(f"That is **{gb(NVFP4_HEAD)} GB against {gb(BF16_HEAD)} GB** for the BF16 head the")
A(f"conservative arm ended up serving — a **{gb(BF16_HEAD-NVFP4_HEAD)} GB** saving on the")
A(f"verifier head alone, worth **{ms(BF16_HEAD-NVFP4_HEAD):.3f} ms** of floor.")
A("")
A("The `fp8_head` figure is not a guess: the unsloth lm_head surgery's own manifest")
A("records the shard growing by `1,270,901,784` B when it replaced the FP8 head with the")
A(f"BF16 one, so that FP8 head was `2,542,796,800 - 1,270,901,784 = 1,271,895,016` B on")
A(f"disk. The computed `{FP8_HEAD:,}` differs by 24 B — safetensors header text, not payload.")
A("")
A("### The DVK slice is row-aligned by construction")
A("")
A("The NVFP4 block scales are `[out, in/16]` = `[248320, 320]`, so a 65536-row")
A("`index_select` selects consistent rows of the weight **and** the scale. The K64 draft")
A("head therefore slices out of the NVFP4 head with no repacking:")
A(f"`65536×2560 + 65536×320 = {DRAFT_NVFP4:,}` B versus `{DRAFT_BF16:,}` B for the BF16 read.")
A("")
A("## 3. Scenarios")
A("")
A("| scenario | mandatory weight bytes | GB | **floor ms** | vs unsloth 102.480 | shippable now |")
A("|---|---:|---:|---:|---:|---|")
for k, v in scenarios.items():
    if k == "legacy_target_plus_verifier_head":
        continue
    d = UNSLOTH_ROOT64K_MS - v["mandatory_weight_floor_ms"]
    sn = "**yes**" if v.get("shippable_now") else "no"
    A(f"| `{k}` | {v['mandatory_weight_bytes']:,} | {gb(v['mandatory_weight_bytes'])} | **{v['mandatory_weight_floor_ms']:.3f}** | −{d:.3f} ms | {sn} |")
A("")
A("Arithmetic, spelled out:")
A("")
for k, v in scenarios.items():
    if k == "legacy_target_plus_verifier_head":
        continue
    A(f"* **`{k}`** — {v['component_formula']}")
    A(f"  = `{v['mandatory_weight_bytes']:,}` B = {gb(v['mandatory_weight_bytes'])} GB")
    A(f"  → `{v['mandatory_weight_bytes']:,} × 1000 / {BW:,}` = **{v['mandatory_weight_floor_ms']:.3f} ms**")
    if v.get("note"):
        A(f"  · {v['note']}")
    A("")
A("## 4. Where the ceiling lands")
A("")
A("The conservative unsloth arm is pinned at **102.480 ms** (`floor_ledger.json`,")
A("`root_64k_five_64k_draft_heads`). Against that:")
A("")
p1 = scenarios["root_64k_nvfp4_head_bf16_draft"]["mandatory_weight_floor_ms"]
p2 = scenarios["root_64k_nvfp4_head_nvfp4_draft"]["mandatory_weight_floor_ms"]
p3 = scenarios["root_64k_nvfp4_head_nvfp4_draft_mtp_fp8"]["mandatory_weight_floor_ms"]
pb = scenarios["root_64k_fp8_head_bf16_draft"]["mandatory_weight_floor_ms"]
wc = scenarios["root_64k_bf16_head_bf16_draft"]["mandatory_weight_floor_ms"]
A(f"* **Phase 1, shippable the moment the lm_head patch boots: {p1:.3f} ms** "
  f"(−{UNSLOTH_ROOT64K_MS-p1:.3f} ms, −{100*(UNSLOTH_ROOT64K_MS-p1)/UNSLOTH_ROOT64K_MS:.1f}%).")
A(f"* Phase 2 (FP4 draft-head GEMV): **{p2:.3f} ms** (−{UNSLOTH_ROOT64K_MS-p2:.3f} ms). "
  "This is the low-80s band the directive asked for.")
A(f"* Phase 2 + MTP-FP8 requant: **{p3:.3f} ms**.")
A(f"* Plan B (FP8 head instead of NVFP4): **{pb:.3f} ms** — {pb-p1:.3f} ms worse than Phase 1, "
  f"i.e. the NVFP4 head is worth {pb-p1:.3f} ms over the FP8 fallback (0.557 GB).")
A(f"* Worst case (BF16 head, as the unsloth arm was forced into): **{wc:.3f} ms** — only "
  f"{UNSLOTH_ROOT64K_MS-wc:.3f} ms better than the conservative arm. **The aggressive backbone "
  "alone buys almost nothing; nearly all of the win is the head.** That is why the lm_head "
  "loader patch is the whole ballgame for this arm.")
A("")
A("## 5. What is NOT in these numbers")
A("")
A("* The honest-floor per-request geometry term (**+7.117 ms at C=18k**) is unchanged and")
A("  additive; it is geometry-only and carries over from FR13 untouched.")
A("* KV-cache traffic, activations, and every non-weight cost: `nonweight_costs_included`")
A("  is `false` in every scenario, as in the unsloth ledger.")
A("* `embed_tokens` (2.543 GB) and the vision tower (0.921 GB) are excluded — neither is")
A("  read on a decode step.")
A("* Acceptance. Floor ms is per *step*; TPS also needs the measured acceptance length.")
A("  RadixArk's own SGLang qualification measured MTP acceptance **2.775** (chain EAGLE")
A("  3/1/4, GB300 ×4) — a prior, not a gate, for Hydra27's tree.")
open(HERE / "radixark_floor_derivation.md", "w").write("\n".join(L) + "\n")

print(f"target_model        {TARGET:>15,}  {gb(TARGET)} GB")
print(f"nvfp4_head          {NVFP4_HEAD:>15,}  {gb(NVFP4_HEAD)} GB")
print(f"mtp_per_pass        {MTP_PER_PASS:>15,}  {gb(MTP_PER_PASS)} GB")
print(f"draft_64k_bf16      {DRAFT_BF16:>15,}  {gb(DRAFT_BF16)} GB")
print(f"draft_64k_nvfp4     {DRAFT_NVFP4:>15,}  {gb(DRAFT_NVFP4)} GB")
print(f"fp8_head            {FP8_HEAD:>15,}  {gb(FP8_HEAD)} GB")
print(f"mtp_fp8_per_pass    {MTP_FP8_PER_PASS:>15,}  {gb(MTP_FP8_PER_PASS)} GB")
print()
for k, v in scenarios.items():
    print(f"{k:<44} {v['mandatory_weight_bytes']:>15,} B  {v['mandatory_weight_floor_ms']:>9.3f} ms")
