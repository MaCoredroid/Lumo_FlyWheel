Inventory Ops uses this repo to publish a queue handoff during shift change, but operators still only get JSON even though the handoff doc now assumes a Markdown view.

There is already some groundwork in the codebase for the handoff layout, and the failing tests are the fastest way to see what never got finished. Complete the Markdown path, keep the existing JSON behavior working, and make sure the docs describe what the CLI actually supports.
