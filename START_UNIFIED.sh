#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -f ../NBody_Integrated_Ubuntu/.venv/bin/activate ]]; then
  source ../NBody_Integrated_Ubuntu/.venv/bin/activate
fi
python integrated_gui.py
