#!/usr/bin/env bash

source .venv/bin/activate

jupyter nbconvert "$1"/notebook.ipynb --to markdown --output README

git add .
git commit -m 'initial commit'
git push

deactivate
