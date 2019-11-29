#!/usr/bin/env sh
ID=$1

echo -n "username: "
read USERNM

scp -r "$USERNM"@137.120.136.27:~/dexterous-robot-hand/monitor/experiments/"$ID" monitor/experiments/
