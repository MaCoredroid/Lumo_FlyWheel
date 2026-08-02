# Verification

- Focused descriptorless and launcher tests: 22 passed.
- Python byte compilation and `git diff --check`: passed.
- Primary fresh-cache SM121a B1/B4 compiles: passed.
- Independent fresh-cache SM121a B1/B4 rebuilds: passed.
- B1/B4 and primary/rebuild cubin, PTX, SASS, and resource identity: passed.
- Static SASS improved from 383 to 382; all resource and memory classes held.
- Stack/local/LDL/STL/calls: all zero.
- Runtime byte gate, timing, and floor acceptance: not run.
