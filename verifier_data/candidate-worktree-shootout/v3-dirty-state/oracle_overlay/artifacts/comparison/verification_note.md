# Verification Note

- Candidate A Worktree: `/tmp/v3-dirty-state-candidate-a`
- Candidate B Worktree: `/tmp/v3-dirty-state-candidate-b`
- Candidate A Validation Command: `python -m pytest -q tests/test_cli.py`
- Candidate B Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
- Final Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
