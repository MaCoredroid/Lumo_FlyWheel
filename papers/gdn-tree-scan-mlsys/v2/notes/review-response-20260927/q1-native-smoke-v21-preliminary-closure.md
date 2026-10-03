# Native-smoke wrapper v2.1 preliminary F1 closure

This note preserves the intermediate snapshot; it is not final freeze approval. Original review remains unchanged. Scientific scope and seven previously approved source files are not reopened.

Exact reviewed wrapper: `fb0ec877dd3137d933538a0802d877443a25ed952379eda6862e224a23f3a788`.
Cleanup library: `9c4904be64fb0b8495498cd9c54be9904d25e14ffb64976f92d96d22cbde58be`.
Shipped mock test source: `a6c452a58745f00e4b0c1399089b9eef16200f5774035b4bfb36c4ed91e4b31c`.
Copies are saved under `native-smoke-v21-closure-snapshot/` next to this report.

Seven shipped mocked-shell test functions pass independently. They exercise the actual cleanup library and isolated actual trap, with shell mocks instead of Docker or NVIDIA tools. Two composition/failure controls remain necessary to close original F1:

1. Docker enumeration failure is interpreted as confirmed absence. In the shipped mock replace the `ps` branch with `exit 1`, then call the unchanged `stop_engine` with pipefail enabled. Actual result is `rc=0`, state `no_owned_container`. Read the query result and its exit status separately; unavailable daemon/query cannot attest stopped/absent.
2. On failed cleanup, exact `finish_run` plus exact launcher EXIT trap runs stop twice. Actual combined control: driver rc0, stop rc1, status running, running true -> terminal rc8, two stop calls, `engine_cleanup.err`150 bytes when receipt is written and300 bytes afterward. Receipt no longer authenticates the final error artifact. Make finalization have a single owner or a finalized guard. Test the combined finish+EXIT path, including driver failure plus cleanup failure and boot/early-abort failure, with no after-receipt writes.

The gate includes the cleanup-library digest, but sources the library before checking it. Source after successful hash validation so no executable dependency is loaded before its identity gate. Keep no-action dry run side-effect free.

No GPU, remote mutation, implementation edit, gate change, or new scientific acceptance criterion was introduced. Parent and sole worker have these precise closure requests.
