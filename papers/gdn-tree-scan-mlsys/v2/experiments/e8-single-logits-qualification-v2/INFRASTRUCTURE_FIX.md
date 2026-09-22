# Qualification infrastructure revision 2 — 2026-09-22

The preserved first attempt `out-20260922T2119Z-e8-single-logits` failed before Docker creation, model boot, or API requests. Its source manifest remains `45da91477bf238e99f70c2201bc722059427ff9cced399754171a4371c41efcf`. The first generator matched only the prefix of a comment and inserted a newline, exposing `, NOT HARDCODED. This read` as shell code. Bash syntax checking did not detect this valid but unintended command.

This sibling changes only the generator's complete-comment-line replacement and the corresponding generated launcher, plus this record, the execution regression test, and the resulting file manifest. The shim, head gate, run verifier, qualification driver, original CPU controls, all frozen repository/E1 dependencies, and all qualification criteria remain byte-identical. `e8_launch.v1.sh` retains its filename for the unchanged driver's reference; the sibling path and manifest distinguish the corrected revision. No failed correctness test is being retried: no numerical gate ran in the first attempt. A corrected infrastructure attempt requires the parent's separate authorization after review.

Run the new Linux execution control without a GPU or Docker container:

```sh
CUDA_VISIBLE_DEVICES='' python3 -B test_launcher_execution.py --original ../e8-single-logits
```

The test executes the actual generated outer shell, real stage verifier, model-directory identity logic against a temporary fake model, and captured inner `bash -lc` body. Strict executable stubs capture Docker arguments, replace only the two allowlisted memory recovery/check Python snippets, and replace the in-container patcher/shims/vLLM. It asserts patcher → E8 arm/qualification shim → E1 recorder shim → vLLM order, actual arm/selfcheck env, source overlays, Cat10 specification, seed, B1, eager, cache-off and synchronous argv. It also executes the unmodified preserved first stage and requires the specific detached-comment failure before any Docker call. No real memory recovery, Docker, model import, or inference occurs.

The sibling's `README.md` is an unchanged historical copy of the first-stage plan; use this revision path when invoking the unchanged `e8_qualify.py`. Timing remains disabled. The provisional timing paragraph is not a frozen timing campaign; a separate sibling will state its pre-data estimator supersession before timing can be authorized.
