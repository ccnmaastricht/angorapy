import json
import os
import re
import shutil

import flask
from flask import request, Blueprint, send_from_directory
from flask_jsglue import JSGlue

from agent.ppo import PPOAgent
from utilities.const import PATH_TO_EXPERIMENTS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

agents = Blueprint("agents", __name__, static_folder="../storage/saved_models/states")
exps = Blueprint("exps", __name__, static_folder="../storage/experiments")

app = flask.Flask(__name__, )
app.register_blueprint(agents)
app.register_blueprint(exps)

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
                            progress["rewards"]["mean"]) > reward_threshold),
                        "bookmark": meta["bookmark"] if "bookmark" in meta else False,
                        "config_name": meta["config"] if "config" in meta else "unknown",
                    }
                })

                envs_available.add(meta["environment"]["name"])

    return flask.render_template("overview.html", exps=experiments, envs_available=envs_available)


@app.route("/benchmarks")
def benchmarks():
    """Write Benchmark page."""
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
                            progress["rewards"]["mean"]) > reward_threshold),
                        "bookmark": meta["bookmark"] if "bookmark" in meta else False
                    }
                })

                envs_available.add(meta["environment"]["name"])

    return flask.render_template("overview.html", exps=experiments, envs_available=envs_available)


@app.route("/bookmark", methods=("POST", "GET"))
def bookmark():
    """Bookmark an experiment."""
    if request.method == "POST":
        try:
            exp_id = int(request.json['id'])
            path = f"{PATH_TO_EXPERIMENTS}{exp_id}"

            with open(os.path.join(path, "meta.json"), "r") as f:
                meta = json.load(f)
                current_status = "bookmark" in meta and meta["bookmark"]

            meta.update({"bookmark": not current_status})

            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump(meta, f)

        except Exception as e:
            return {"success": e.__repr__()}
    else:
        return {"success": "no post"}

    return {"success": "success"}


@app.route("/experiment/<int:exp_id>", methods=("POST", "GET"))
def show_experiment(exp_id):
    """Show experiment of given ID."""
    experiment_paths = sorted([int(p) for p in os.listdir(f"{PATH_TO_EXPERIMENTS}")])
    current_index = experiment_paths.index(exp_id)

    path = f"{PATH_TO_EXPERIMENTS}/{exp_id}"
    with open(os.path.join(path, "progress.json"), "r") as f:
        progress = json.load(f)

    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    return flask.render_template("experiment.html", info={
        "env": meta["environment"]["name"],
        "config": meta["config"] if "config" in meta else None,
        "current_id": exp_id,
        "next_id": experiment_paths[current_index + 1] if current_index != len(experiment_paths) - 1 else None,
        "prev_id": experiment_paths[current_index - 1] if current_index != 0 else None,
        "hps": meta["hyperparameters"],
        "env_meta": meta["environment"],
        "iterations": meta["iterations"] if "iterations" in meta else None
    })


@app.route("/expfile/<int:exp_id>/<path:filename>")
def expfile(exp_id, filename):
    path = os.path.abspath(os.path.join(PATH_TO_EXPERIMENTS, str(exp_id)))
    return send_from_directory(path, filename, as_attachment=True)


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


@app.route("/_clear_all_short")
def clear_all_short():
    """Delete all experiments stored that have less than 10 cycles finished."""
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

                if "rewards" in progress and len(progress["rewards"]["mean"]) <= 10:
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


@app.route("/evaluate", methods=("POST", "GET"))
def evaluate():
    """Evaluate an agent."""
    if request.method == "POST":
        try:
            agent = PPOAgent.from_agent_state(request.json['id'])
            evaluation_stats = agent.evaluate(10, save=True)

            return {"results": evaluation_stats._asdict()}

        except Exception as e:
            return {"success": e.__repr__()}

    return {"success": "success"}
