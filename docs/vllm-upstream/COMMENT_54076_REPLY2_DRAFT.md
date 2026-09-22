# #54076 reply to jschmied's 2026-09-22 reproduction + question — v1 (my draft; Codex check pending; AWAITING MARK GO)
> jschmied (2026-09-22T15:24Z) ran our config on his 0.28.1rc1.dev524 build: identical 800/200 lines,
> "It reproduces, and my 2026-09-20 comment here was wrong." He asks: "If a current-main run is what
> you need, say so — I have the aarch64 nightly wheel for 1ea7c63f4 on the box." He also notes that
> `eagle_aux_hidden_state_layer_ids` must be nested under `draft_model_config.hf_config` (our comment's
> inline list was shorthand; our argv.json in the bundle used the nested form).
> Purpose: answer his direct question (yes, a main run closes the only gap we could not close), and
> confirm the nested form. ≤90 words. Answering an explicit question = not an unsolicited ask.

---

Thanks for the rerun. Yes: a current-main run would settle the one part we could only read. At `382970ee6c` the hidden-state block-size reduction and the prefix-cacheable minimum are unchanged in source, but nothing on our side executed main. On the form: our runs used the nested `draft_model_config.hf_config` shape (see `argv.json` in the pinned bundle); the inline list in my comment was shorthand. AI assistance was used.
