#!/bin/sh

root="$(dirname $0)"

cd "$root/../"

python -m ml "$@"