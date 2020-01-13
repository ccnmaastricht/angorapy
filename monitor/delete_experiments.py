#!/usr/bin/env python
"""Delete empty experiments."""
import json
import os
import re
import shutil

exp_dir = "experiments/"
experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

deleted = 0
for path in experiment_paths:
    if re.match("[0-9]+", str(path.split("/")[-1])):
        if os.path.isfile(os.path.join(path, "progress.json")):
            try:
                with open(os.path.join(path, "progress.json"), "r") as f:
                    progress = json.load(f)
            except Exception:
                # delete corrupted
                shutil.rmtree(path)
                deleted += 1
                continue

            if "rewards" in progress and len(progress["rewards"]) < 2:
                shutil.rmtree(path)
                deleted += 1

print("DONE")
print(deleted)
