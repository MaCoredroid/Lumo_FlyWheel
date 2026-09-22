# #54076 reply to jschmied's 2026-09-22 reproduction + question — v2 (Codex replacement verbatim: NO-GO on "unchanged in source" — 382970ee6c adds the prefix-cacheable filter, packed-group bypass and minimum-page scaling; GO on this text; AWAITING MARK GO)
> jschmied (2026-09-22T15:24Z) ran our config on his 0.28.1rc1.dev524 build: identical 800/200 lines,
> "It reproduces, and my 2026-09-20 comment here was wrong." He asks: "If a current-main run is what
> you need, say so — I have the aarch64 nightly wheel for 1ea7c63f4 on the box." He also notes that
> `eagle_aux_hidden_state_layer_ids` must be nested under `draft_model_config.hf_config` (our comment's
> inline list was shorthand; our argv.json in the bundle used the nested form).
> Purpose: answer his direct question (yes, a main run closes the only gap we could not close), and
> confirm the nested form. ≤90 words. Answering an explicit question = not an unsolicited ask.

---

Thanks for the rerun. Yes, please run the offered `1ea7c63f4` nightly with this config: it would check whether the startup geometry persists on that revision. At `382970ee6c`, source inspection supported this hidden-state reduction path and its inclusion in the prefix-cacheable minimum; we did not execute main. This still leaves runtime correctness untested. You're right about nesting: our archived `argv.json` uses `draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids`; my comment abbreviated the config. AI assistance was used.
