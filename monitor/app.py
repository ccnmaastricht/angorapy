import json
import os
import re
import shutil

import flask
from flask import request
from flask_jsglue import JSGlue

from utilities.const import PATH_TO_EXPERIMENTS

app = flask.Flask(__name__)
jsglue = JSGlue(app)


@app.route("/")
def overview():
    """Write Overview page."""
    exp_dir = PATH_TO_EXPERIMENTS
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    experiments = {}
    envs_available = set()
    for path in experiment_paths:
        eid_m = re.match("[0-9]+", str(path.split("/")[-1]))
        if eid_m:
            eid = eid_m.group(0)

            if os.path.isfile(os.path.join(path, "progress.json")):
                with open(os.path.join(path, "progress.json"), "r") as f:
                    progress = json.load(f)

                with open(os.path.join(path, "meta.json"), "r") as f:
                    meta = json.load(f)

                # for vn, vector in progress.items():
                #     replace()


                reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
                    meta["environment"]["reward_threshold"])
                iterations = len(progress["rewards"]["mean"])
                experiments.update({
                    eid: {
                        "env": meta["environment"]["name"],
                        "date": meta["date"],
                        "iterations": iterations,
                        "max_reward": max(progress["rewards"]["mean"]) if iterations > 0 else "N/A",
                        "is_success": False if iterations == 0 else ("maybe" if reward_threshold is None else max(
                            progress["rewards"]["mean"]) > reward_threshold)
                    }
                })

                envs_available.add(meta["environment"]["name"])

    return flask.render_template("overview.html", exps=experiments, envs_available=envs_available)


@app.route("/experiment/<int:exp_id>", methods=("POST", "GET"))
def show_experiment(exp_id):
    """Show experiment of given ID."""
    experiment_paths = sorted([int(p) for p in os.listdir(f"{PATH_TO_EXPERIMENTS}")])
    current_index = experiment_paths.index(exp_id)

    path = f"{PATH_TO_EXPERIMENTS}{exp_id}"
    with open(os.path.join(path, "progress.json"), "r") as f:
        progress = json.load(f)

    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    return flask.render_template("experiment.html", info={
        "env": meta["environment"]["name"],
        "current_id": exp_id,
        "next_id": experiment_paths[current_index + 1] if current_index != len(experiment_paths) - 1 else None,
        "prev_id": experiment_paths[current_index - 1] if current_index != 0 else None,
        "hps": meta["hyperparameters"],
        "env_meta": meta["environment"],
    })


@app.route("/_clear_all_empty")
def clear_all_empty():
    """Delete all experiments stored that have less than 2 episodes finished."""
    experiment_paths = [os.path.join(PATH_TO_EXPERIMENTS, p) for p in os.listdir(PATH_TO_EXPERIMENTS)]

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

                if "rewards" in progress and len(progress["rewards"]["mean"]) < 2:
                    shutil.rmtree(path)
                    deleted += 1

    return {"deleted": deleted}


@app.route('/delete_experiment', methods=['GET', 'POST'])
def delete_experiment():
    """Delete an experiment of posted id."""
    if request.method == "POST":
        try:
            eid = request.json['id']
            eid = int(eid)  # make sure its a number
            shutil.rmtree(os.path.join(PATH_TO_EXPERIMENTS, str(eid)))
        except Exception:
            return {"success": False}
    else:
        return {"success": False}

    return {"success": True}
