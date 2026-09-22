# B4 duplicate-path C4 review

2026-09-22. Read-only raw-artifact/source inspection and CPU-only execution of one extracted production function. No GPU, inference, source edits or new campaign.

**Verdict: the original C4 failure is a false deterministic-path requirement. The recorded alternative is legal under the current sampler. Neither FIRST nor LAST matching sibling is guaranteed.** The v3b hard-coded LAST repair fits this sample but misstates the implementation; replace it with validation of the actual published path against independent candidate IDs and target argmaxes. No new model capture is needed.

The case is **p085, not p095**. B4’s request order differs from cohort-file order. Remote run:
`/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all/`.

## Exact raw binding and decision

- Forward index **7** (zero-based) = recorder seq **8**, physical_step_id **2**.
- `final_logits.call2.pt` / `layer_hidden.call2.pt`: both capture_call_index=2; block **3**, global rows **30–39**.
- Recorder request `cmpl-822fbbe30317f121-0-9234a7af` uniquely maps to `cohort/req_2_p085/capture_request.json`, response ID `cmpl-822fbbe30317f121`, prefix p085, 4096 prompt tokens.
- Verified candidates are draft record **idx=6** (the preceding forward), request row **3**, ts=1790066970.1844. Per-request spec record at ts=1790066970.9665 reports acc=2.
- State/operand payloads are `language_model_model_layers_62_linear_attn.row3.step2.pt`: matching layer/row/step; authoritative row=h0_row=118, h0_col=0. Published accepted path=[1,3], accepted_len=2.
- Hidden root position=4100 in all three position axes = 4096 + five previously emitted tokens − 1; full depth offsets match cat10.

Parents: `[-1,0,1,1,2,2,4,4,6,6]`.

| GDN node | Draft candidate (root has none) | Captured full-vocabulary argmax |
|---|---:|---:|
| 0 | — | 13 |
| 1 | 13 | 561 |
| 2 | 561 | 8057 |
| 3 | 561 | 8057 |
| 4 | 8057 | 2099 |
| 5 | 829 | 22258 |
| 6 | 4021 | 1436 |
| 7 | 579 | 2099 |
| 8 | 2528 | 1070 |
| 9 | 1436 | 1070 |

Both edges of the published path pass independently: node0 argmax13 equals draft(node1), and node1 argmax561 equals draft(node3). Node3 is a leaf; its argmax8057 is the bonus. The exact recorder output is **[13,561,8057]**. This is not terminal clipping: these are the complete accepted-plus-bonus outputs for the chosen path.

The alternative first-child path [1,2,4] would output [13,561,8057,2099]. Node2 and node3 logits are **bitwise equal** in this capture (max absolute difference 0). The shorter legal path changes accepted length/work per step, not the greedy token prefix in this observed decision. No maximal-length path contract is implemented.

Independent corroboration: `logs/tree_sampler_debug.jsonl.commit`, line **40**, records req=3, sampler node=2 (root-excluding index, hence GDN node3), committed_token=561, accepted=true, child_drafts=[561,561], child_probs=[1,1], overlap_mass=2. The preceding line records acceptance of token13. Leaf bonus is established by the captured logits and output ledger; that leaf branch does not emit the same per-child trace entry.

## Actual selection contract

The preserved loaded `rejection_sampler.py:3778–3791` routes greedy sampling into the unified committer with all_greedy=True. Lines `2649–2652` choose the default device implementation; the run log records its engagement. Lines `2737–2771` load `/workspace/scripts/fr13_device_multidraft_kernel.py` and pass all_greedy through.

That script’s actual code, rather than its overstated comments, determines duplicate selection:

- `1089`: all_greedy makes each target/self probability row one-hot at its argmax.
- `915–927`: candidate overlaps are normalized into source weights, then **torch.multinomial chooses the source**.
- For duplicate candidates [561,561] with p(561)=1, source weights are [1/2,1/2]; `930–937` gives acceptance probability1 for either source.
- `1149–1165`: the sampled source selects the next tree node. At a leaf, `1125–1136` emits the selected node’s self-row bonus.

The script is clean relative to recorded HEAD; current and HEAD SHA-256 both equal `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9`. The loaded caller is preserved, but this dynamically imported script was **not separately copied into loaded_backend**; the source binding here is its recorded import path plus unchanged checked-out HEAD, corroborated by the actual commit trace. It is not a new claim of captured GPU RNG state.

I extracted and executed the exact `device_multidraft_node_step` AST on CPU with p(561)=1, candidates [561,561], seeds0–11. Source0 was selected for seeds1,3,4,6,9,11; source1 for seeds0,2,5,7,8,10. Every result was token561/accepted=true, and every generator state advanced. Thus “FIRST”, “LAST”, “longest path”, and the source docstring’s “ZERO rng consumption” do not describe this code.

The existing dup-sibling gate’s 0/2000 result uses generators=None, creating the same default generator state for every request. Its observed FIRST choice is not a proof over other generators/devices and must not override the actual categorical-source rule. Preserve the test result with that narrower interpretation; no serving implementation change is required for the observed legal path.

## Minimal gate repair

Validate the recorded path **as a legal greedy path**, without choosing it from a fabricated tie rule:

1. Start at root0. For every published next node, independently require the correct parent edge and draft(next)==argmax(current); reject wrong candidate IDs or wrong edges.
2. Derive each emitted token from the captured parent argmax. At the published final node, require no matching child remains (or an explicit real sampler depth cap); otherwise the path stopped too early.
3. Derive the final correction/bonus from that node’s argmax and compare the complete list with the output ledger. Retain the separately reviewed final-request-only API truncation rule.
4. With duplicate matches, either matching child is legal. Do not accept arbitrary paths merely because their emitted strings coincide. Retain request/position/state identity checks unchanged.

Two positive CPU controls should accept this leaf path and its valid longer first-child alternative; existing wrong-token, wrong-parent, early-stop and wrong-bonus negatives must reject. This is a bounded offline checker correction using existing data, not a request to re-run the model or broaden stochastic testing.

Keep `capture_gate.v3.json` (original FIRST-path failure) and `capture_gate.v3b.json` (LAST-path provisional pass) immutable. Add the corrected interpretation/output separately. This review covers the C4 duplicate case; it does not replace agent2’s API formatter identity proof or certify other gate checks.

## Reviewed identities and reproduction

The inspected helper/gate pair was `e7b_ledger_align.py` SHA `6cbb9491d260ef369f360b8d8e73c32aef0a6aa5947dcd9fce299a3766edfaf5` and `e7b_b4_capture_gate.py` SHA `f49246b62c8398a5503f4a6e60226e1f6483e428f7b31831995398d0ffac53f3`. The helper and device-script bytes were stable throughout the independent raw/CPU reduction.

```text
279474c5323637feaa13457c1abe42ff63a4e3aa897eaa425c767c54ecbd79dd  logs/final_logits.call2.pt
6bc8b75556a5bbc47abb16bcde5cd900c66e52f01217a5a734931d9aa1e5a554  logs/layer_hidden.call2.pt
c3fe7d0085d0960e6fe852f400c2bfb13031a185930b49c2e5e5fdf99349af82  logs/e7b_state/language_model_model_layers_62_linear_attn.row3.step2.pt
0662ce92d571c57272b3054ea8114e5dfb5ead47dfaab45f1d744e0358ff5369  logs/e7b_operands/language_model_model_layers_62_linear_attn.row3.step2.pt
a970cd4a6612eb466700697f644b1db997647f4ac6b901525ad1080d7ca9e57e  cohort/req_2_p085/capture_request.json
69907abdaa6174e4fade1bac99f8d8d8a2da64b258b91bd50edb537f8ec6e2d9  capture_gate.v3.json
add3a8a6b11ea638b9a7e7b83085c5e1c1d8fbbf0f22dbc82b8d73150b3f87d2  capture_gate.v3b.json
cf410506441426904020ac756f60daa23f9d42ba5fca236b37499b16f26ae40b  logs/e1_events.jsonl
e2f3b8db9234e35afe23f822545edd629e24d06a2ffa14fe236e93f457779f0e  logs/fr10_mtp_draft_trace.jsonl
cf13b1e21f9689724fdf9482257f71557be7744bd0a3271e6478ebb63c622a42  logs/per_req_spec_trace.jsonl
110371ca5ca623d04d5e47ac2f3227592880daf4544e54913d3d6016c68b2bb0  loaded_backend/rejection_sampler.py
```

Minimal source-control reproduction, from the remote checkout with CUDA_VISIBLE_DEVICES empty and python3 -B:

```python
import ast, torch
from pathlib import Path
p = Path("scripts/fr13_device_multidraft_kernel.py")
node = next(n for n in ast.parse(p.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "device_multidraft_node_step")
ns = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), ns)
probs = torch.zeros(600); probs[561] = 1
for seed in range(12):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    print(seed, ns["device_multidraft_node_step"](
        probs, torch.tensor([561, 561]), generator=gen))
```

