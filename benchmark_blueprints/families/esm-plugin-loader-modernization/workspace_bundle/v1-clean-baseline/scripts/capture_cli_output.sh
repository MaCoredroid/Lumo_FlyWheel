
#!/usr/bin/env bash
set -euo pipefail

npm run build >/dev/null
mkdir -p artifacts

{
  echo '$ node dist/src/index.mjs --help'
  node dist/src/index.mjs --help
  echo
  echo '$ node dist/src/index.mjs good-default'
  node dist/src/index.mjs good-default
  echo
  echo '$ node dist/src/index.mjs good-named'
  node dist/src/index.mjs good-named
  echo
  echo '$ node dist/src/index.mjs --list'
  node dist/src/index.mjs --list
} > artifacts/cli_capture.txt
