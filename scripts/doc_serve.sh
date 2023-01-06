#!/usr/bin/env zsh
# sphinx-quickstart docs --ext-autodoc --ext-
set -e

CODE_SRC="src"

# generate api docs
sphinx-apidoc -o docs/source ${CODE_SRC} -f
# build html pages
sphinx-autobuild -b html --port 8080 --watch ${CODE_SRC} docs docs/_build/html \
    --pre-build "sphinx-apidoc -o docs/source ${CODE_SRC} -f"
