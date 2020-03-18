#!/usr/bin/env sh
ID=$1

echo -n "username: "
read USERNM

scp -r "$USERNM"@192.168.26.181:~/dexterous-robot-hand/monitor/static/experiments/"$ID" monitor/static/experiments/
scp -r "$USERNM"@192.168.26.181:~/dexterous-robot-hand/storage/saved_models/states/"$ID" storage/saved_models/states/
