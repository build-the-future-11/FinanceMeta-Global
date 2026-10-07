#!/usr/bin/env python3
"""Compile the prepared manuscript in a complete ACM TeX environment."""
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
executable = shutil.which('pdflatex')
if executable is None:
    raise SystemExit('Install a TeX distribution with the ACM template dependencies first.')
for _ in range(2):
    subprocess.run([str(Path(executable).absolute()), '-halt-on-error', '-interaction=nonstopmode', '-no-shell-escape', 'manuscript.tex'], cwd=root, check=True, timeout=120)
