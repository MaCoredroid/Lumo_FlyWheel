# Four-attempt source assembly: independent bounded review

Source/CPU review only; no implementation edits, Docker, SSH, GPU, model, agent, evaluator or gate action. WP remains closed, 0/4. Commands and full argv are being reviewed separately.

The submitted draft was named as `f8945159feb3112e1fc86ae3de48bd6be9d746ed63700de6f78ab95acbd66c23`. The parent repaired the first finding during review; the retained snapshot and executable controls bind the resulting draft **`b56b8d327200ee45651695e6acf40958e8931173eb0bf1c5c2f13a7464b814fc`** and generator **`112c1fbc029749ba6c5baf5279024ba89edac9dd928ff777fc2a7caa3c42161a`**. These remain drafts. Evidence and controls: `p0/monitor/review-response-20260927/source-assembly-independent-20260929T0201Z/`.

## Concrete corrections

1. **SGLang container paths — corrected in retained successor.** The initial generator used `/workspace/...` for the shared template and launcher source, but the SG launcher mounts neither that repository nor that template path. `collectors_v3.py:110–126,208` reads all source-manifest members from the actual server process root. The corrected draft names the actual `/lumotree/chat_template/...` template and installed `/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py` route source. The launcher remains coordinator source/boot identity; this does not claim it is installed inside SG. The CPU control confirms these corrected paths, and the source launcher mount supports the template path.

2. **Missing terminal collector role — still reproduced in snapshot.** The assembled `collectors.json` does not approve `collectors_v3.py` for `agent_completion`. Its `_agent_terminal` creates exactly that terminal and calls `e.terminal` (`collectors_v3.py:355–363`); `e3_preflight_v5.py:169` requires the producer's role to match the terminal kind. The real `approved_collector` function refuses the assembled manifest for this role. Add `agent_completion` to that exact collector source's roles, regenerate the collector manifest/evidence/freeze hashes. Existing observation roles pass the control.

3. **Incomplete mutable Lumo runtime source closure.** The installed, prepared `gdn_linear_attn.py:72–75` imports `lumo_flywheel_serving.fr13_sfwd_state_fusion_production`, `fr13_replay_conv_remap`, `fr13_ex2_silu`, and `fr13_tree_conv_fused` at module scope. None of those `/workspace/src/lumo_flywheel_serving/*.py` files is in the three Lumo manifests. The AST control reproduces all four omissions. Unlike installed unmodified image sources, these bind-mounted numerical sources are not covered by the immutable image ID. Include their local dependency closure at actual container paths in the runtime source observation, or connect an existing complete frozen package manifest to that same actual-byte validation. Do not claim that hashing the importing installed file binds the imported module bytes.

4. **Observed route value type mismatch.** The assembled vLLM `observed_route_contract` expects JSON boolean `prefix_caching: true`. `runtime_probe_producer_v1_2.py:329` returns the raw regex string `"True"`/`"False"` and line 838 copies it into observed route. `runtime_projection_v1_2.py:149–158` compares canonical JSON exactly, so all three vLLM arms would refuse the intended enabled-cache policy. The retained parser control reproduces `"True" != true`. Normalize the known log token to an actual boolean while preserving missing as missing, test both True/False/missing, and repin the producer. Do not use generic `bool(string)`.

## Checks that pass within this scope

- All **105** assembled evidence files hash-match; all **11** manifests pass the real adapter's member-byte validation. The evidence bytes are retained for repeatable controls.
- Each configuration ID equals the canonical digest of its record. All three vLLM arms share exactly the same LM-head patch/template manifest; AR/chain use the retained native CPU-prepared source set and Lumo uses its separate post-patch set. This checks identities, not numerical equivalence.
- The schedule filename is `SCHEDULE-v2.1.json`; the recorded schedule hash equals its **canonical JSON digest**. The assembled order is exactly AR, CHAIN_MTP, SGLANG_EAGLE, LUMOTREE for scikit-learn__scikit-learn-9288. No old eight-row schedule is reintroduced.
- All three `bundle_sources` hashes match their actual allocation/request/projection modules. They are source expectations, not fabricated worker observations; their existing dependency seals remain necessary.
- `boot.instrumentation_path=None` is consistent with `boot_v3_3.py:428`: the boot phase waits for health without requiring a fictional pre-existing probe file. The caller's reviewed owned probe then supplies real instrumentation before agent admission. This does not waive the probe or baseline equality checks.
- Qualification hashes remain null, approval/time fields remain unresolved, and the draft remains explicitly unfrozen. No scientific or runtime approval is inferred from these CPU checks.

The bounded disposition is **source draft needs the three remaining corrections above**. No extra workload task, seed, cap, repeat, experiment or broad redesign is requested.
