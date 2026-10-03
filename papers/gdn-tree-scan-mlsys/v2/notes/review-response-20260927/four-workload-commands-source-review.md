# Four rendered workload commands and Lumo argv — source review

**PASS for this bounded source scope. No concrete blocker found.** No rendered shell command, launcher, agent, Docker/container, network call, evaluator or GPU operation was executed. The package remains `DRAFT_NOT_FROZEN`; this report is not route qualification, runtime admission or WP authorization.

## Rendered commands

Independent AST replay of the pinned historical **pure renderer functions** reproduced the public prompt and all four command files byte-for-byte after exactly the documented transformations. No benchmark-driver entry point or scientific module was imported. The reconstruction used the preserved source SHA `a924716508292e5a86ae7cde1ee09794dd229665f2a42b958f26cd06fba224c9`.

- Public input contains exactly `instance_id`, `repo`, `base_commit`, `problem_statement`, `version`; no gold patch, hidden test patch, outcome or retry prompt is consumed. It selects `scikit-learn__scikit-learn-9288`, base `3eacf948e0f95ef957862568d87ce082f378e186`, matching the predetermined SCOPE and pinned image metadata. `_write_agents_md` consumes this allowlist (`agent_v2.py:36–46`; historical source:6959–7021). The operator is the original first-attempt prompt, with `/workspace/` replaced by `/testbed/`.
- The fixed order is AR, CHAIN_MTP, SGLANG_EAGLE, LUMOTREE: one task, one attempt per arm, no additional repeats. Each command has exactly one Qwen invocation. After normalizing only container name, owned output directory and distinct deterministic session UUID, all four token arrays are identical. Session UUIDs are not sampling seeds.
- Identical command settings include response cap 32768, stream idle timeout 600000 ms, PATH, `/testbed` working directory, common model name and proxy endpoint. The common pinned image is `swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-9288@sha256:702434646ba19a69bf216770efdbc2010f64aad9f1162be14abec1bb52399662`; known image ID `sha256:fae30c4fa5ad6588291ec3e3e4c33a36879f234d50ff5d00eb145e8b1ccc46ce`. The existing known-lock receipt binds the base metadata and identical tracked blobs. This check does not re-inspect the host image. `git diff HEAD` preserves the historical image-environment convention rather than introducing a new checkout/reset.
- All four mount the exact bundle directory ending `594cac41e2d5ed505e0646f318b263ff70e200bcffe97326fe1c042fdc220516` read-only at `/opt/qwen`, and use `--network=fr14-agent-isolated`. The draft separately binds the expected internal network identity/subnet/gateway and requires live admission rechecks. Rendering alone proves neither actual isolation nor actual bundle bytes.
- The only post-render command changes are omission of explicit empty `BASH_ENV`, `LD_LIBRARY_PATH`, `LD_PRELOAD`, `NODE_OPTIONS`, `NODE_PATH`, and replacement of host-inherited `OPENAI_API_KEY` with the declared nonsecret local placeholder. Every omission and replacement matches `RENDER-RECEIPT.json` (`render_four_commands_v1.py:23–34`). Runtime absence checks remain necessary; omitting an entry does not itself override an arbitrary image environment. No secret values from unrelated environments were read or printed.

Actual proxy sampling, compaction, agent/evaluator budgets, seed delivery and settings precedence remain governed by the separately reviewed runtime checks. This source review does not infer them from session IDs or from a command-rendering receipt.

## Lumo serve argv

The Lumo draft's **46-token argv** exactly matches the checked static expansion of the selected host profile and launcher defaults. Source anchors: `lumotree_workload_owned_v2_4.sh:77–84` (profile), 676–677 (model), 3796–3802 (31 physical drafts/exact tree), 4376–4462 (APC), 4508–4519 (graph/KV options), 8388–8397 (serve assembly).

Verified: model/tokenizer/name; seed 0; max sequences 1; GPU utilization **text `0.70`**; max length 131072; TREE_ATTN/triton; explicit template/parser options; exact tree and speculative method; APC/chunked prefill; 1024 attention and mamba block sizes; recurrent-cache float32; batched-token limit 4096; long-prefill threshold 1024; FULL_AND_PIECEWISE graphs. **No `--mamba-cache-mode` occurs in the launcher or draft argv.** The candidate v2.2 receipt agrees on these scientific settings (0.7 numerically). Current explicit template and disabled ingress middleware are documented workload changes, not falsely asserted to be identical to the historical candidate's entire argv.

This is static source expansion, not a shell execution smoke or an observation of actual process argv. Unset optional inputs are evaluated at their checked launcher defaults; an ambient override is not approved by this note. Exact image/argv, loaded sources, actual allocated precision, named-probe evidence and environment still require the future owned caller's admission checks. The rendered commands/receipt do not themselves authorize a task attempt.

## Reproduction and identities

Preserved snapshot: `p0/monitor/review-response-20260927/four-command-independent-v1/` (18 original files). Independent check:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/four-command-independent-v1/check.py
```

All four exact replays, public/operator hashes, cross-arm normalization, task/order/image metadata checks and Lumo argv/receipt checks PASS. The pure prompt renderer wrote only to a temporary directory. No supplied command was executed.

| Source / artifact | SHA-256 |
|---|---|
| `render_four_commands_v1.py` | `6a5de3615f9d3d88fa8720758d72a52ecbebd514d82ee95ee43ca90e499ef0f7` |
| `runtime-freeze.DRAFT.json` | `739c1ada30b714c201a036469c7be6cbea8b6ab886bfa9f78b255ee07735d278` |
| `lumotree_workload_owned_v2_4.sh` | `02ef55ea8a83f49a8ace3605a9d482f22c65728c88f7d7dceeed4edb5a76f46d` |
| `lumotree-workload-host-env.v1.2.env` | `c961494bee18b1538494c30aeeccd7b4fd93f58728607d4efe4360c409d125ff` |
| `candidate-launch-env.v2.2.json` | `ac3719d00ec69d03d2bc23966ee23eeb3d8929e962a88dd3e96b05f168c42bc4` |
| `AR.command` | `ecc08e04fba86ee725d1a0e81ea9b28c47ca775bfa4dbe293af789921ddfc87a` |
| `SGLANG_EAGLE.command` | `09f57025e21f7ae309b365fbfcce5958f1ba1e77ba27fd586260bfb771a64e62` |
| `LUMOTREE.command` | `a1b73630b5cb594504ca53fadb77d9d10428fb24d75acc0c503647bb1736f571` |
| `CHAIN_MTP.command` | `8a295e5526f567998a08e4ff002ba9d16924dec1282c47b73e3ad8749f42b385` |
| `RENDER-RECEIPT.json` | `ed9bb93a127f0ee1e2b351d30f614d62b7869aaea9a150cacd53d609202d4e3b` |
| `AGENTS.md` | `061ef7751ca8e3c5f1265ad1b437a827a5370eff276486fa61fe627a1266ad95` |

During final identity checking, the parent updated the draft freeze. Both snapshots are preserved. Latest captured draft SHA `b56b8d327200ee45651695e6acf40958e8931173eb0bf1c5c2f13a7464b814fc` has unchanged common settings, all four argv arrays, and task/arm/ordinal order. Its differences add source/collector/schedule/observed-route bindings, replace configuration IDs accordingly, and set boot instrumentation paths to null for the caller-owned producer. These metadata changes do not alter this command/argv verdict; their wider integration is outside this bounded check. No final-freeze approval is implied.

- `p0/monitor/review-response-20260927/four-command-independent-v1/SNAPSHOT.json` — SHA-256 `a22dcbe9cb4e7a997a7531e9708a302a9e37d5deaba486362f95b4b4120f9ee4`.
- `p0/monitor/review-response-20260927/four-command-independent-v1/check.py` — SHA-256 `f8ea33cc5f61da43424e313bae196af50e057cfca3df358a372a33308917eab9`.
- `p0/monitor/review-response-20260927/four-command-independent-v1/results.json` — SHA-256 `7d2ad327bfa83033c46a9fe1b863e9a4d4077a685003e5fae31b67daa3166f7e`.
- `p0/monitor/review-response-20260927/four-command-independent-v1/runtime-freeze.latest-draft.json` — SHA-256 `b56b8d327200ee45651695e6acf40958e8931173eb0bf1c5c2f13a7464b814fc`.
- `p0/monitor/review-response-20260927/four-command-independent-v1/latest-draft-delta.json` — SHA-256 `97e35a989e42a289b099f6bf72df3895f6705b125eea8b1defb8efd07808d216`.

The other 17 original files were unchanged at final check. No implementation or gate file was modified.
