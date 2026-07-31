# FR13 fixed32 qrow16 production exact4 B1 closeout

Status: **REJECTED AS A TWO-ARM CAMPAIGN**

The Hydra arm is a valid canonical SWE-Verified exact4 B1 arm. It measured
`232.779790071 ms/event` full-step wall and `24.718146718` full-wall TPS with
qrow16 engaged on all 16 target-attention layers. Against the valid Hydra
baseline, that is a `6.246844167 ms/event` (`2.613451%`) observed wall
improvement.

This remains far outside the corrected hardware-floor requirement:

- corrected mandatory-weight floor: `119.658015414 ms/event`
- one-sided U95 acceptance cap: `137.606717726 ms/event`
- Hydra qrow16 point ratio: `1.945375655x`
- Hydra point gap to cap: `95.173072345 ms/event`

The Tail arm is not valid evidence for acceptance or tuning. Its final task,
`astropy__astropy-13398`, was stopped by Qwen's always-on loop detector after
96 turns and exited `1`. The harness correctly rejected the missing terminal
`runner_metadata.json`; no formal two-arm floor gate was produced. The Tail
performance JSON is retained only for failure forensics and is explicitly
marked ineligible in `measurement_summary.json`.

Both arms did prove qrow16 production engagement. Both final flushes completed
with zero pending scorer, drafter, and committer samples. Runtime and external
manifests are byte-identical from launch to end, Docker is empty, and no GPU
compute process remains.

The paired campaign is not being rerun immediately. A one-sided U95 cannot
rescue a point already `95.173 ms` above the cap, so the next GPU time goes to
larger kernel candidates and the missing B4 gate. Exact16 remains blocked
until a meaningful floor-ratio breakthrough.

No raw prompts, responses, patches, or benchmark workspaces are published in
this artifact.
