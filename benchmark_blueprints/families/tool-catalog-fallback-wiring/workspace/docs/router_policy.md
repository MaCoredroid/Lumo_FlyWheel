Fallback selection must be capability-aware and policy-aware.

When a preferred tool is configured for a capability, use it only if it both
supports that capability and its policy appears in the router's
`eligible_fallback_policies` list. If that preferred tool is unavailable or
ineligible, inspect only the configured fallback list for the requested
capability, discard any tool that lacks the capability, discard any tool whose
policy is not eligible, and then prefer the safest remaining policy before
using list order as a tie-breaker.
