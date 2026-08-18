# The SASS-digest reproducibility credential

Mark's ruling, 2026-08-18. Implemented in
`scripts/fr13_build_fa2_qrow32_gqa_pair_b1_sm121a.sh`. Validation evidence is the
FR14 TreeAttention-v2 build-environment proof, which is what surfaced the problem.

## The problem, measured

The campaign pins the candidate `.so` sha256 at six hard-fail sites. That value is
**not reproducible across rebuilds from byte-identical source.** nvcc stamps its own
driver PID inside the build container into host-side symbol and name-table entries
(`tmpxft_<pid>_<counter>`), which propagates into roughly 87 kB of symbol and
relocation bytes.

Three builds of the promoted B1 GQA-pair unit, from a source closure that reproduced
byte-for-byte every time (`172b5e71…`), inside the same pinned image:

| build | nvcc module id | SASS digest | `.so` sha256 | `.so` bytes |
|---|---|---|---|---|
| 2026-08-10 (sealed) | `tmpxft_00000009` | `fa01f988…` | `3560cdc0…` **(the pin)** | 299 815 552 |
| FR14 rebuild run 1 | `tmpxft_00000009` | `fa01f988…` | `3560cdc0…` **= pin** | 299 815 552 |
| FR14 rebuild run 2 | `tmpxft_0000000a` | `fa01f988…` | `454135ce…` **≠ pin** | 299 815 552 |

Device code is fully deterministic — one SASS digest across all three, and identical
`REG:243 STACK:0 LOCAL:0`. Run 1 matched the pinned `.so` hash by landing the same
container PID as the sealed build. **That is a coin flip, not a proof**, and before this
change a rebuild that lost the flip was indistinguishable from a rebuild that had
genuinely produced a different kernel.

## The split

The two values answer different questions, and conflating them is what made a
PID-shifted rebuild look like a corruption:

| credential | question it answers | reproducible? | where enforced |
|---|---|---|---|
| **SASS digest** (new) | *Did this rebuild reproduce the sealed **kernel**?* | **yes** | asserted in the build script, before the link, exit 96 |
| **`.so` sha256 + size** (unchanged) | *Is the artifact about to be staged the one that was **gated**?* | no | the six pin sites, hard-fail, exactly as before |

The SASS digest is asserted **before the link** deliberately: if the rebuild did not
reproduce the sealed device code there is no reason to spend a link on it, and no
half-credentialed artifact is left lying around to be staged by mistake.

The dump is environment-independent by inspection — it carries arch, code version, host
OS class and the disassembly, and no host path, timestamp, `BUILD` directory or `tmpxft`
module id. It is bound to the pinned image, which fixes the `cuobjdump`/`nvdisasm` that
produce it.

## Nothing became easier to pass

This is a credential-**strengthening** change in the sanctioned family, and the diff
shows it:

- **One new hard-fail added** (exit 96). It can only reject builds the old script
  accepted; it can never admit one the old script rejected.
- **No executable check removed or weakened.** The only deleted lines in the whole diff
  are two `echo` guidance lines, replaced by expanded guidance.
- **The six `.so` pin sites are untouched.** A staged `.so` whose sha does not match what
  was gated still refuses to serve, and a tree that re-pins only some of the six is still
  rejected at launch.
- The script now explicitly forbids the obvious wrong move: *do not re-pin
  `SASS_DIGEST_SHA256` to make a build pass.* If the device code moved, the toolchain or
  the flags moved, and that is the finding — not a value to refresh.

## Self-test (container-free; run during the arm-B nsys window)

The assertion block was extracted verbatim from the script, so the test cannot drift
from the code, and exercised against the real artifacts:

- **Positive:** all three builds above PASS — including run 2, whose `.so` differs from
  the pin. That is the case the credential exists for: kernel identical, binary
  re-containered.
- **Negative:** a SASS dump altered by a single instruction mnemonic is refused with
  exit 96 and a message that names the likely cause (toolchain/flag divergence, since the
  source closure is separately proven).
- `bash -n` clean.

A full end-to-end rebuild was not re-run: the script requires zero containers and arm B's
nsys attribution chain held one throughout this window. The assertion operates on
`$BUILD/<obj>.sm121a.sass`, which the preceding step already produces and which was
verified for all three builds, so the end-to-end path is covered by the artifacts rather
than by a fourth compile.

## Applicability

Written for the gqa_pair B1 builder because that is where the evidence is. The same split
applies to every sibling FA2 build script that pins a `.so` sha256 — each would need its
own pinned SASS digest, taken from a build already qualified. Not done here; flagged.
