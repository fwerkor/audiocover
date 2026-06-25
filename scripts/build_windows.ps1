$ErrorActionPreference = "Stop"
python -m pip install -U pip
python -m pip install -e ".[build]"
python scripts/build_desktop.py
