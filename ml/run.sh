#!/bin/sh
# The purpose of this script is to run the ml package as a package

root="$(dirname $0)"

cd "$root/../"

python -i -m ml "$@"