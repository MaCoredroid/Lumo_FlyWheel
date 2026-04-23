## Abandoned patch

An earlier attempt started moving the worker to a staged `processing` write with
per-row commits. The author left before proving rollback behavior. Treat this as
context only, not as a directive to keep the partial transaction pattern.
