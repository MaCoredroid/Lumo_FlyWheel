# #53142 reply to Karl0007's 2026-09-22 fixture-adaptation report — v2 (Codex replacement verbatim: "which column is restored" imprecise; "not on initialisation seams" too categorical; GO; AWAITING MARK GO)
> Karl0007 (#55507 author) adapted our P8 restore-fidelity fixture to #55507's lazy binding, ran it on
> the tree of our branch: #53798 as-is 10 passed; #55507 as written 2 FAILED (seed column 6709 vs 121;
> unequal-geometry restore into the wrong block); #55507 + fallback fix Karl0007/vllm@adc7d30
> (`_mamba_spec.block_size` else `cache_config.mamba_block_size or cache_config.block_size`) 10 passed.
> Confirms item D's flagged unbound-fallback divisor; reachable when a request is admitted before the
> first preprocess (KV-connector-restored prefix). Notes the fixture discriminates on the seeding path,
> not the initialisation seams, and cannot run against base main unmodified (no set_kv_cache_config).
> Offers to push the adapted fixture + arm patches to a branch.
> Purpose: answer the offer (yes), agree the scope statement, state what we did not run. ≤90 words.

---

Yes, please push the adapted fixture and the arm patches to a branch and link it here. Your reported failures match the unbound-fallback divisor we flagged above. The fixture checks seeding and restored state bytes; the adapted comparison does not establish equivalence of the initialization paths. Having the branch available would make the adaptation and fix independently reviewable. We have not run your adapted arms or `adc7d30`. AI assistance was used.
