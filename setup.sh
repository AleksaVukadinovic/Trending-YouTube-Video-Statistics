#!/usr/bin/env bash

set -e  # stop on first error

PYTHON_VERSION=3.11
VENV_NAME=venv

echo "🔧 Installing Python $PYTHON_VERSION..."

sudo apt update
sudo apt install -y \
    python$PYTHON_VERSION \
    python$PYTHON_VERSION-venv \
    python$PYTHON_VERSION-dev

SCRIPT_DIR="$(pwd)"

echo "Creating virtual environment in $SCRIPT_DIR/$VENV_NAME"
python$PYTHON_VERSION -m venv "$SCRIPT_DIR/$VENV_NAME"

echo "Activating virtual environment"
source "$SCRIPT_DIR/$VENV_NAME/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "No requirements.txt found, skipping dependency install"
fi

echo "Done!"
echo "To activate later, run:"
echo "source $VENV_NAME/bin/activate"
