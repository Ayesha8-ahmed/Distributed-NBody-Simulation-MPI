#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Installing Ubuntu requirements (sudo may ask for password)..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip libopenmpi-dev openmpi-bin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
printf '\nSetup complete. Run: bash run_demo.sh\n'
