.PHONY: help current-help visible-tests

help:
	python src/release_preview/cli.py --help

current-help:
	python src/release_preview/cli.py generate --help

visible-tests:
	pytest -q tests/test_release_preview_cli.py
