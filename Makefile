.PHONY: install dev test lint compile build-gui build-desktop

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

compile:
	python -m compileall src tests

build-gui: build-desktop

build-desktop:
	python scripts/build_desktop.py
