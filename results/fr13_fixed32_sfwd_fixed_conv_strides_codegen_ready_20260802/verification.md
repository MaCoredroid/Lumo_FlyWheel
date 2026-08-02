# Verification

- Focused descriptorless and launcher tests: 22 passed.
- Python byte compilation and `git diff --check`: passed.
- Primary fresh-cache SM121a B1/B4 compiles: passed.
- Independent fresh-cache SM121a B1/B4 rebuilds: passed.
- B1/B4 cubin, PTX, SASS, and resource identity: passed.
- Primary/rebuild cubin, PTX, SASS, resource, and summary identity: passed.
- Registers/static/encoded SASS improved from 55/391/408 to 54/383/400.
- Stack/local/LDL/STL/calls: all zero.
- Runtime byte gate, timing, and floor acceptance: not run.
