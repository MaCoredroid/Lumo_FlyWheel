# Fixed32 native key-group CFWD pinned-image build

Status: **the exact full vLLM extension is rebuilt and source-bound inside the
pinned serving image; the real K64/root1 B1 byte gate is pending**.

## Build result

The canonical native key-group candidate was built as the full
`vllm/_C.abi3.so` target against vLLM commit
`fe9c3d6c5f66c873d196800384ed6880687b9e52`. The build ran without GPU access
inside the pinned serving image at digest
`sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`.
Its host ABI came from GLIBC 2.35 and GCC 11.4; CUDA device compilation used
NVCC 13.0.88 for `sm_121a`.

The runtime image's Python 3.12 development surface, bundled CUDA 13 NVRTC,
headers, and libraries were used directly. Source attestation used a read-only
Git 2.39.5 executable whose own maximum GLIBC requirement is 2.34. No host
compiler, host CUDA headers, package install, or dependency fetch participated
in the final build.

Binding issuance verified the exact patched vLLM graph, made the candidate CUDA
source newer than the prior object and extension, rebuilt that object, relinked
the full extension, and bound all three timestamps and byte identities.

| Surface | Bytes | SHA256 |
| --- | ---: | --- |
| Canonical CUDA source | 19,955 | `1c1a9813410dcf15bcbb4d23bec71ee16ddcd7e2dbe3b1a3698e58f71bd96985` |
| Rebuilt candidate object | 6,261,768 | `5ac9344a80432cd5f15bbd45f6682cff3afe9f84392666a923d56550dbc99714` |
| Full `_C.abi3.so` | 201,407,632 | `bd5f38ff19f3ebb08f00fce2498746cfa111a7819fb63b3b12ccf4dc2edbc9e4` |
| Private binding | 1,817 | `0345f08be7f702053d3a719a10a01fdf33d7bf5494a6e600294ee5099eb290fb` |

## ABI and codegen

The linked extension requires at most GLIBC 2.34, GLIBCXX 3.4.21, and CXXABI
1.3.9. The enforced runtime ceiling is GLIBC 2.35. Its runpath includes the
pinned image's Torch, bundled CUDA 13, and CUDA toolkit library directories.

The exact rebuilt candidate object passes the frozen SM121a codegen contract:
64 registers per thread, zero stack and local bytes, 7,592 reported shared
bytes, and zero `LDL`, `STL`, or `CALL`. The pinned arithmetic and shuffle SASS
counts also match.

## Qualification boundary

- Focused selector, source, binary, committer, and B1 gate suite: `99 passed`.
- Host binding verification, object codegen checker, Python byte compilation,
  Ruff, runtime-manifest self-test, JSON parsing, checksums, and whitespace
  checks: pass.
- Build teardown census: zero Docker containers and zero GPU compute processes.
- No CUDA kernel launch, synthetic or probe timing, real SWE-Verified task,
  throughput measurement, hardware-floor result, or production authorization
  is represented here.
- The next step is the separately coordinated real K64/root1 B1 all-depth byte
  gate. B1 timing and canonical exact4 B4 qualification remain pending.

This directory contains reduced build facts only. It excludes binaries,
objects, prompts, model inputs or outputs, requests, responses, patches, raw
logs, environment dumps, credentials, process identities, and container
identities.

From this directory, verify the manifest and checksums with:

```bash
python3 -m json.tool manifest.json >/dev/null
sha256sum --check SHA256SUMS
```
