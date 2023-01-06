#!/usr/bin/env zsh
set -e

pip install sphinx sphinx-autobuild myst-parser

echo
echo "Tools have been installed."
echo "Run ./scripts/doc_serve.sh to start the doc server."
echo "You should also modify docs/conf.py to change basic informations."
