# B4 persistent-M128 post-teardown evidence handoff

The Tail23 K64 exact-four persistent-M128 byte diagnostic completed all four
real SWE-Verified tasks from source
`2f76c744510384b780899b7f3b5a1dda74efc903`. The serving process produced 320
comparisons across all five required B4 projection shapes, with zero mismatched
comparisons and zero differing bytes.

The original launcher stopped after container teardown because the live marker
was root-owned mode `0400` and the installed-binary attestation was root-owned
mode `0600`, while the reducer ran as the host user. This was a post-run
permission defect, not a kernel mismatch or task failure.

The completed evidence was reduced by replaying the exact pinned reducer under
root. The resulting live verdict has SHA-256
`58e8da4d84ffcc2934e8a5327b7c11e8431a9c44324ca177e6cfcdec2a343c9e`
and status `pass`; its production credential was issued without rerunning any
task. This recovery is qualification evidence only and contains no timing or
hardware-floor claim.

Code commit `ef8397ae8a8e8b8dd7b75039a4451d5e1ed401c3` fixes future launches
before any additional GPU spend.
It preflights passwordless privilege, validates that both artifacts are regular
single-link root-owned files with their exact modes, transfers ownership after
teardown without following symlinks, revalidates stable identity and content,
and records a reduced handoff receipt. The byte reducer itself is unchanged.

This directory contains reduced aggregate metadata only. It contains no
prompts, responses, patches, traces, raw logs, credentials, process identities,
or container identities.
