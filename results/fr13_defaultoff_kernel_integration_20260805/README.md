# FR13 default-off kernel integration

This artifact records the CPU/static integration of three independently
reviewed, default-off kernel candidates onto main commit
`fee4cfd2b4de6fca5108da32122973ab720295a6`:

- fixed32 CFWD packed-walk node trust: `e7c8bac9b296e6fd3de1c9fef00418c83266e231`
- B4 K64 M4 reused-weight draft head: `77817d7a31a05b07a7f5a1b29e5ee9f7faeeb65d`
- fixed32 SFWD embedded gate CTA: `db73940140147c91b6431914f352ad0bda20e4d1`

The integration preserves main's CFWD capture-scope contract
`421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b`
and the disabled-U8 credential clearing in the shared B1 runner. The node-trust
overlay, wrapper, and real-task runner were rebound fail-closed to that current
contract. All three new selectors remain exactly default off.

No GPU API, Docker command, service, real task, timing, or acceptance run was
used for this artifact. Runtime qualification remains required before any
candidate can become production eligible.
