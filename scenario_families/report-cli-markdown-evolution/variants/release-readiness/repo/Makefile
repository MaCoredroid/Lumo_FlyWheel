.PHONY: install test lint typecheck ci

install:
	pip install -e ".[test]"

test:
	pytest -q

lint:
	ruff check src tests

typecheck:
	mypy src

ci: lint typecheck test
