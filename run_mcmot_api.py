#!/usr/bin/env python3
"""Run the MCMOT FastAPI backend."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
VENV_ROOT = os.path.join(PROJECT_ROOT, ".venv")

if os.path.exists(VENV_PYTHON):
    current_prefix = os.path.realpath(sys.prefix)
    expected_prefix = os.path.realpath(VENV_ROOT)
    if current_prefix != expected_prefix:
        os.execv(VENV_PYTHON, [VENV_PYTHON, __file__, *sys.argv[1:]])

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import uvicorn


def main() -> None:
    uvicorn.run("mcmot.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

