# Native CFWD B1 pre-task identity rejection

The real SWE-Verified B1 arm reached server health and opened authenticated
ingress, but the immutable pre-task boundary rejected the candidate before any
model request ran. The installed v4 binary and runtime boundary agreed on the
v4 CUDA source identity; the host boundary validators still expected the
retired v3 source identity.

This artifact is an infrastructure rejection, not a kernel correctness or
performance result. It contains no task prompt, model request or response,
patch, raw log, environment, process identifier, or container identifier.

The readiness repair updates both fail-closed boundary validators to the
canonical v4 source identity and adds a regression assertion tying them to the
candidate contract. A new real SWE-Verified B1 arm is still required.
