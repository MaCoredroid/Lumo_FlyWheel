# Corrections to RESULT.md / E_54076_bootcheck.md (independent Codex check, 2026-09-21)

The measured values, the arithmetic, the hook's logging-only behaviour and the main line references
were verified. The following statements in the agent's report are corrected here rather than edited
in place:

1. **Hook wording.** `hook/sitecustomize.py` DOES monkeypatch two methods (`Scheduler.__init__`,
   `EngineCore._initialize_kv_caches`) with wrappers that call the originals and then read/print
   geometry. The report's phrase "assigns nothing on any vLLM object" is inaccurate; "additive,
   logging-only wrappers" is the accurate description. Matching `id(cache_config)` at both sites was
   confirmed in all four arms.
2. **Arm-4 log line numbers.** The decisive lines in `runs/arm4_extract/server.log` (inside the
   tarball) are 1923, 2530/2548, 2596 and 2608–2615, not the line numbers quoted in the report
   (which came from an earlier extraction).
3. **"Only one difference matters" on main is unsupported.** Besides the `prefix_cacheable` filter,
   main also differs in a packed-group bypass and a minimum-page scaling step. For this candidate,
   equal non-hidden pages skip packing and 10240 B < 3,276,800 B skips scaling, so the hidden-state
   reduction path is preserved. Main remains unexecuted; no universal cross-version claim is made.
4. **"Dead code."** `EngineCore.get_kv_cache_group_metadata` (0.28.0 `core.py:419`) has no in-tree
   caller found; "dead code" overstates it.
5. **Scope of the negative result.** The DFlash arm shows that this particular target + drafter pair
   on 0.28.0 yields 832/832 and does not reproduce the PR body's 816/1648 geometry. It does not prove
   that no drafter configuration can produce a divergence.
6. **Characterisation of arm 4.** `extract_hidden_states` + `ExampleHiddenStatesConnector` is the
   documented hidden-states extraction workflow for drafter-training data, not ordinary speculative
   serving, and it is not a guarantee for every hybrid model.
