.PHONY: install dev test lint compile build-gui

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

build-gui:
	pyinstaller packaging/audiocover-gui.spec --clean --noconfirm
