# Tool Routing

- `github.list_pull_requests` is owned by GitHub and uses `github_list_pull_requests_handler`.
- `gmail.search_threads` is owned by Gmail and uses `gmail_search_threads_handler`.
- Routing resolves exact canonical tool ids from the registry or discovery metadata instead of `search_*` prefix matching.
- Fallback order is deterministic per tool: `github.list_pull_requests` prefers `github` before `gmail`, and `gmail.search_threads` prefers `gmail` before `github`.
- A connector can only receive a fallback when it advertises the same canonical tool id; generic GitHub search helpers do not satisfy Gmail thread search requests.
