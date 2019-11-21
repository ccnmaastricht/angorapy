#!/usr/bin/env bash

uid=$(id -u -n)

# permissions for monitoring
usermod -a -G www-data "$uid"
chmod +x monitor/delete_experiments.py
chgrp www-data monitor/experiments/
chmod g+rwxs monitor/experiments/