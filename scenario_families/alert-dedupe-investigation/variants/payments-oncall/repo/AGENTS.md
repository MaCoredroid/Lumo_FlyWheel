An integration regression in the payments on-call alert stream started after
the dedupe logic was simplified.

Read `logs/failure.log`, reproduce the problem through the tests, and fix the
dedupe key so alerts are collapsed by the correct time window and environment.
The code should preserve distinct incidents instead of merging them together.

Do not delete the failing integration coverage.
