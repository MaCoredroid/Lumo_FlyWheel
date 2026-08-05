# FR13 CFWD target-capture scope correction

This artifact records the failed real B1 Hydra27 fixed32 K64/root1 composed gate
launched from source commit `8c59e28f4afa56edb526562df78fd72c39561616`.
The run loaded the target and drafter, completed PIECEWISE capture, and reached
FULL target-model capture. It failed before service readiness or task traffic at
`fr13_fixed32_cfwd_logit_direct_capture_end` with
`FR13 CFWD logit-direct capture binding drift`.

The target-model FULL graph does not contain the rejection-sampler CFWD commit.
The old lifecycle incorrectly required one CFWD call inside target capture and
disabled byte-comparison counting immediately after target replay, before the
external commit. The correction is isolated to the CFWD runtime overlay and
vLLM patch wiring:

- bind the already-prewarmed CFWD state to the target graph identity;
- require zero CFWD calls inside target capture, then seal the single external
  committer route;
- keep comparison counting enabled through the rejection-sampler commit; and
- disable counting immediately after that commit returns.

The credential-bound base device source remains unchanged at SHA-256
`088454e0605c5d41aee7b385c6d0ff66e6a7ddb999a9697258762d0aac9fe166`.
This artifact makes no GPU correctness or speed claim; a fresh real SWE-Verified
B1 gate is required.
