#!/usr/bin/env sh
echo -n "username: "
read USERNM

scp -r "$USERNM"@137.120.136.27:~/dexterous-robot-hand/docs/benchmarks/* docs/benchmarks/
scp -r "$USERNM"@137.120.136.27:~/dexterous-robot-hand/docs/figures/benchmarking_* docs/benchmarks/
