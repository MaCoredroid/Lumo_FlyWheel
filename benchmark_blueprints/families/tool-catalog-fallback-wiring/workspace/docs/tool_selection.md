Use `chrome-devtools` as the preferred tool for `browser.read` when it is
present, advertises that capability, and is allowed by the configured router
policy.

When the preferred tool is missing or no longer supports the requested
capability, inspect the configured fallback list for that capability, discard
tools that do not advertise the capability, discard tools whose safety policy
is not in `eligible_fallback_policies`, and then prefer safer policies before
falling back to catalog order.

Verification examples:
- `select_tool("browser.read", "chrome-devtools")` returns `chrome-devtools`.
- `select_tool("browser.read", "missing-browser-tool")` returns
  `browser_snapshot`, because `http_fetch` does not support `browser.read`.
- `select_tool("docs.lookup", None)` returns `http_fetch`.
