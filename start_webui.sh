#!/bin/sh
set -eu

# Resolve paths from the script, including when started from another directory.
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -x "chatbot-env/bin/python" ]; then
    echo "Virtual environment missing. Follow the setup instructions in Readme.md." >&2
    exit 1
fi
exec "chatbot-env/bin/python" webui.py
