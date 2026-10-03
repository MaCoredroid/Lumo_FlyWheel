# Native corpus reducer repair: one remaining provenance closure

29 September 2026. Source/CPU-only review of reducer SHA-256 **`d240befad2a424c86674031b0d724adf4c1a56ba68c4c2da714f021ddbbdef10`**, tests **`5138f7bf38a81e0e12c00235a0b76d812563cb8a3d3df8474438c2f05738214b`**. The immutable snapshot, dependency hashes and controls are under `p0/monitor/review-response-20260927/native-corpus-repair-independent-review/`. This does not approve a launch or cover subsequent wrapper/source edits.

**The prior numerical/domain/metadata fixes are closed. One narrow actual-execution provenance check remains before reducer acceptance.** All seven supplied CPU methods passed independently. A separate control alters only a temporary driver-source copy: the frozen corpus comparison rejects it before processing runs. No model, GPU, Docker, remote operation or gate change occurred; no raw smoke payload was reread in this pass because parent is separately running that integration fixture.

## Closed findings

- Exact native GDN and attention group checks now accompany complete layer/geometry/content validation (`snapshot`, lines 135–179). Missing/wrong group controls refuse. Source logit dtype is explicitly BF16 while the archived complete vector remains FP32 (`logits`, 182–201); unsupported/missing dtype refuses.
- Fixture canonical content is recomputed using the generator's serialization; unique prefix/path pairs complete the 84-case Cartesian denominator (66–93). False canonical and duplicate-pair controls refuse.
- Terminal status requires exact driver and cleanup success (278–280). Per-request actual sent/usage prefix counts and case/repeat/sealed identity are checked (345–348); aggregate verdict binds the run/job and all 168 requests (326–330).
- The corpus now pins reducer/dependency sources and image, and reduction checks approved gate/launch bindings, actual image and distinct container IDs, generated-runner patch content and FA2 installation (295–325). Source drift is rejected. These materially improve provenance over the first draft.
- O0/O1 pointer, stride and storage-offset continuity is checked within each request (223–227). Physical state-row or KV block IDs are correctly not required to be equal across independent observations. Raw finite, smallest-ID tie, exact per-case A/B/R2/common-O0 and separate packed-control rules remain unchanged.

## Required final closure: bind the actual configuration to its generator

At reducer lines 311–317, only `Q1_REF_HOOKS_JOB_SHA256` and the packed flag are checked in actual Docker `Config.Env`. Actual `Cmd` is compared with the saved `engine-config.json`, but the arm configuration is not independently regenerated from the pinned generator. `RestartCount` is also not checked at lines 303–310.

`probe_provenance.py` executes the exact relevant reducer AST statements, using the actual frozen config module's pure `arm_config()` output as a positive fixture. The positive fixture passes. Each of the following alterations also passes those statements:

1. Actual `OMP_NUM_THREADS=7`, while the generated configuration declares 1.
2. Actual `PYTHONPATH=/unbound/source`, while the generated configuration declares the campaign tools and source directory.
3. Actual `RestartCount=1`, inconsistent with one fresh engine lifetime.

The results are saved in `PROVENANCE-PROBES.json`. This is explicitly a **predicate-level source test**, not a claimed full-corpus or GPU result. The existing shell may prevent those runtime configurations; the remaining issue is that the reducer does not independently reject contradictory actual execution evidence.

Minimal repair: regenerate the selected pure `arm_config(arm, run_id, bound_log_dir, job_sha)` from the hash-pinned generator and compare it with the saved arm configuration and recorded actual launch. Validate every declared runtime environment entry (and the pinned image defaults where relevant), rather than just two keys; reject contradictory duplicate environment entries. Require `RestartCount == 0` and a consistent completed engine identity. Bind the actual launch argv/script and selected command digest if those artifacts are used. This does not require invoking the full config builder, hashing model weights again, launching an engine, changing a numerical rule, or expanding the experiment population. The other reviewer owns shell lifecycle; no additional shell-lifecycle review is requested here.

The reducer continues to write a characterization result with exit zero even if `primary_baseline_qualified` is false. Keep gate consumption tied to the explicit eligibility field and exact coverage, not process exit alone. The accepted smoke integration test can close raw-data compatibility separately; it must retain its one-case/A/R2 scope.
