#!/usr/bin/env sh
echo -n "username: "
read USERNM

scp -r "$USERNM"@192.168.26.181:~/dexterous-robot-hand/docs/benchmarks/workstation_* docs/benchmarks/
