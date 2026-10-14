#!/bin/bash
# setup.sh — One-command project setup
# Usage: bash setup.sh

set -e

echo ""
echo "  graph-rca › setting up environment..."
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ venv created"
else
    echo "  ✓ venv already exists"
fi

# Activate and install
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "  ✓ dependencies installed"
echo ""
echo "  activate with:  source venv/bin/activate"
echo "  then run:       python pipeline/monitor.py"
echo ""
