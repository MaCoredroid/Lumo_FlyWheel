## Codex CLI invocation prompt

"Read the task prompt at /workspace/AGENTS.md and complete it in this workspace. Edit the source files directly to implement the fix. Do not write a diff file -- modify the files in place so that running pytest passes the tests described in the prompt."

## AGENTS.md (workspace/astropy__astropy-7166)

# SWE-Bench task: astropy__astropy-7166

**Repo:** `astropy/astropy`  
**Base commit:** `26d147868f8a891a6009a25cd6a8576d2e1bd747`  
**Version:** `1.3`  

## Problem statement

InheritDocstrings metaclass doesn't work for properties
Inside the InheritDocstrings metaclass it uses `inspect.isfunction` which returns `False` for properties.


## Required behavior

Implement the fix described in the problem statement by editing the source files in this workspace. Do NOT modify any test files. The hidden grader will apply its own test patch and run the test suite; your code must make those tests pass without breaking existing ones.

