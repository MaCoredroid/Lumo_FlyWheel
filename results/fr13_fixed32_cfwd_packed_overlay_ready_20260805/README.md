# Packed CFWD Credential-Preserving Overlay Readiness

Status: `CPU_READY_UNMEASURED`

This artifact binds the packed-CFWD v3 runtime overlay prepared at source
commit `16a5ff62def58483d8ef170f03eb9f39bcfd9854`. The canonical
`fr13_device_multidraft_kernel.py` bytes are restored exactly to SHA-256
`088454e0605c5d41aee7b385c6d0ff66e6a7ddb999a9697258762d0aac9fe166`,
so the existing TAW B4/B1 credential chain remains valid without weakening its
whole-file binding.

The fail-closed loader verifies the canonical file and generated overlay before
installation. The installer replaces only the reviewed packed-CFWD kernels,
helpers, and CFWD identity fields. It checks that every TAW source function
object and the TAW source contract remain unchanged. Host validation observed:

- TAW contract: `fr13-fixed32-taw-all-parent-v7`, SHA-256 `998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290`.
- Packed CFWD contract: `fr13.fixed32.cfwd_logit_direct.integration_source.v2`, SHA-256 `a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c`.
- Packed candidate source SHA-256: `5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c`.
- Focused CFWD/U8/TAW host suite: `94 passed`.

No GPU, Docker, service, or SWE-Verified task was run for this artifact. It
makes no runtime correctness, performance, timing, floor-acceptance, or
production claim.

After the existing TAW credential is available, two real B1 boots remain the
minimum: one stock-FA2 FULL-graph boot composing CFWD packed v3 with DFWD U8,
and one separate eager Qrow16 SFWD byte-gate boot. SFWD cannot share the first
boot because its eager/Qrow16 contract conflicts with FULL graph and stock FA2.
