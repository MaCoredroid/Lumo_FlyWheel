# Verification Note

- Candidate A Worktree: `/tmp/v1-clean-baseline-candidate-a`
- Candidate B Worktree: `/tmp/v1-clean-baseline-candidate-b`
- Candidate A Validation Command: `python -m pytest -q tests/test_cli.py`
- Candidate B Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
- Final Validation Command: `python -m pytest -q tests/test_cli.py tests/test_service.py`
