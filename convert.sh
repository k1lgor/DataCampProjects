#!/usr/bin/env bash

jupyter nbconvert "$1"/notebook.ipynb --to markdown

git add .
git commit -m 'initial commit'
git push
