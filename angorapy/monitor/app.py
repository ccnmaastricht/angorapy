import json
import os
import re
import shutil
import sys
from json import JSONDecodeError

from bokeh import embed
from tqdm import tqdm

from angorapy.models import MODELS_AVAILABLE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy.utilities.monitor.statistical_plots import plot_episode_box_plots, plot_per_receptor_mean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import flask
from flask import request, Blueprint, send_from_directory
from flask_jsglue import JSGlue

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.common.const import PATH_TO_EXPERIMENTS, BASE_SAVE_PATH
from angorapy.utilities.monitor.training_plots import plot_memory_usage, plot_execution_times, plot_preprocessor, \
    plot_reward_progress, plot_loss, plot_length_progress, plot_distribution, compare_reward_progress, \
    grouped_reward_progress, group_preview, plot_aux_perf_progress
from angorapy.utilities.statistics import ignore_none

from angorapy.monitor import app


agents = Blueprint("agents", __name__, static_folder="storage/saved_models/states")
exps = Blueprint("exps", __name__, static_folder="storage/experiments")

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
    for exp_path in tqdm(experiment_paths):

        eid_m = re.match("[0-9]+", str(exp_path.split("/")[-1]))
        if eid_m:
            eid = eid_m.group(0)
            model_path = os.path.join(BASE_SAVE_PATH, eid)

            if os.path.isfile(os.path.join(exp_path, "progress.json")):
                with open(os.path.join(exp_path, "progress.json"), "r") as f:
                    progress = json.load(f)

                try:
                    with open(os.path.join(exp_path, "meta.json"), "r") as f:
                        meta = json.load(f)
                except:
                    pass

                agent_parameters = {}
                model_available = False
                if os.path.isfile(os.path.join(model_path, "best/weights.index")):
                    model_available = True

                    with open(os.path.join(model_path, "best/parameters.json")) as f:
                        agent_parameters = json.load(f)

                reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
                    meta["environment"]["reward_threshold"])
                iterations = len(progress["rewards"]["mean"])
                max_performance = ignore_none(max, progress["rewards"]["mean"])
                is_success = False
                if iterations >= 0 and reward_threshold is None:
                    is_success = "maybe"
                elif max_performance is not None and max_performance > reward_threshold:
                    is_success = True

                architecture = "Any" if "architecture" not in meta["hyperparameters"] else meta["hyperparameters"]["architecture"]
                model = "Any" if "model" not in meta["hyperparameters"] else meta["hyperparameters"]["model"]
                if model_available:
                    architecture = agent_parameters.get("builder_function_name", architecture)
                    for mname in MODELS_AVAILABLE:
                        if mname in architecture:
                            architecture = mname

                experiments.update({
                    eid: {
                        "env": meta["environment"]["name"],
                        "date": meta["date"],
                        "host": meta["host"] if "host" in meta else "unknown",
                        "iterations": iterations,
                        "max_reward": max_performance if iterations > 0 else "N/A",
                        "is_success": is_success,
                        "bookmark": meta["bookmark"] if "bookmark" in meta else False,
                        "config_name": meta["config"] if "config" in meta else "unknown",
                        "reward": meta["reward_function"] if "reward_function" in meta else "None",
                        "model": architecture + "[" + model + "]",
                        "model_available": model_available,
                        "gatherer": meta["hyperparameters"].get("gatherer", "default"),
                        "policy": meta["hyperparameters"].get("distribution", "default")
                    }
                })

                envs_available.add(meta["environment"]["name"])

    return flask.render_template("overview.html", exps=experiments, envs_available=envs_available)


@app.route("/groups")
def view_groups():
    """Build group view page."""
    exp_dir = PATH_TO_EXPERIMENTS
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    experiments_by_groups = {}
    envs_available = set()
    for exp_path in experiment_paths:

        eid_m = re.match("[0-9]+", str(exp_path.split("/")[-1]))
        if eid_m:
            eid = eid_m.group(0)
            model_path = os.path.join(BASE_SAVE_PATH, eid)

            if os.path.isfile(os.path.join(exp_path, "progress.json")):
                with open(os.path.join(exp_path, "progress.json"), "r") as f:
                    progress = json.load(f)

                with open(os.path.join(exp_path, "meta.json"), "r") as f:
                    try:
                        meta = json.load(f)
                    except JSONDecodeError as jse:
                        continue

                exp_group = meta.get("experiment_group", "n/a")

                if not exp_group in experiments_by_groups.keys():
                    experiments_by_groups[exp_group] = {}

                model_available = False
                if os.path.isfile(os.path.join(model_path, "best/weights.index")):
                    model_available = True

                reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
                    meta["environment"]["reward_threshold"])
                iterations = len(progress["rewards"]["mean"])
                max_performance = ignore_none(max, progress["rewards"]["mean"])
                is_success = False
                if iterations >= 0 and reward_threshold is None:
                    is_success = "maybe"
                elif max_performance is not None and max_performance > reward_threshold:
                    is_success = True

                architecture = "Any" if "architecture" not in meta["hyperparameters"] else meta["hyperparameters"]["architecture"]
                model = "Any" if "model" not in meta["hyperparameters"] else meta["hyperparameters"]["model"]

                experiments_by_groups[exp_group].update({
                    eid: {
                        "env": meta["environment"]["name"],
                        "date": meta["date"],
                        "host": meta["host"] if "host" in meta else "unknown",
                        "iterations": iterations,
                        "max_reward": max_performance if iterations > 0 else "N/A",
                        "is_success": is_success,
                        "bookmark": meta["bookmark"] if "bookmark" in meta else False,
                        "config_name": meta["config"] if "config" in meta else "unknown",
                        "reward": meta["reward_function"] if "reward_function" in meta else "None",
                        "model": architecture + "[" + model + "]",
                        "model_available": model_available
                    }
                })

                envs_available.add(meta["environment"]["name"])

    return flask.render_template("groups.html", exps=experiments_by_groups, envs_available=envs_available)


@app.route("/group/", methods=("POST", "GET"))
def show_group():
    """Analyze the summarized results of a group of experiments."""
    ids = request.args.getlist('ids')
    group_name = request.args.get('group')
    info = {
        "ids": ids,
        "group_name": group_name,
        "plots": {}
    }

    progress_reports = {}
    metas = {}
    environments = []

    for exp_id in ids:
        path = f"{PATH_TO_EXPERIMENTS}/{exp_id}"
        with open(os.path.join(path, "progress.json"), "r") as f:
            progress = json.load(f)

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        progress_reports.update({f"{exp_id}": progress})
        metas[str(exp_id)] = meta

        environments.append(meta["environment"]["name"])

    environments = set(environments)
    environment_reward_threshold = None
    if len(environments) == 1:
        environment_reward_threshold = meta["environment"].get("reward_threshold", None)  # take last exps threshold

    info["plots"]["reward_grouped"] = grouped_reward_progress(
        {id: (group_name, progress["rewards"]) for id, progress in progress_reports.items()},
        environment_reward_threshold
    )

    return flask.render_template("group_analysis.html", info=info)


@app.route("/make_group_preview", methods=("POST", "GET"))
def make_group_preview():
    """Analyze the summarized results of a group of experiments."""
    group_names = request.json['names']
    group_names = group_names.split(",")

    exp_dir = PATH_TO_EXPERIMENTS
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    environments = {}
    reward_thresholds = {}
    experiments_by_groups = {}
    envs_available = set()
    for exp_path in experiment_paths:

        eid_m = re.match("[0-9]+", str(exp_path.split("/")[-1]))
        if eid_m:
            eid = eid_m.group(0)
            model_path = os.path.join(BASE_SAVE_PATH, eid)

            if os.path.isfile(os.path.join(exp_path, "progress.json")):
                with open(os.path.join(exp_path, "progress.json"), "r") as f:
                    progress = json.load(f)

                with open(os.path.join(exp_path, "meta.json"), "r") as f:
                    try:
                        meta = json.load(f)
                    except JSONDecodeError as jserr:
                        continue

                exp_group = meta.get("experiment_group", "n/a")

                if exp_group not in group_names:
                    continue

                reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(meta["environment"]["reward_threshold"])

                if not exp_group in experiments_by_groups.keys():
                    experiments_by_groups[exp_group] = {}
                    reward_thresholds[exp_group] = reward_threshold
                    environments[exp_group] = meta["environment"]["name"]

                envs_available.add(meta["environment"]["name"])

                experiments_by_groups[exp_group].update({
                    eid: progress
                })

    return flask.render_template("update_group_preview.html",
                                 plot=group_preview(experiments_by_groups,
                                                    alt_titles=environments,
                                                    reward_thresholds=reward_thresholds))


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
                        "host": meta["host"] if "host" in meta else "unknown",
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
    experiment_paths = sorted([int(p) for p in os.listdir(f"{PATH_TO_EXPERIMENTS}") if re.match("^[0-9]*$", p)])
    current_index = experiment_paths.index(exp_id)

    path = f"{PATH_TO_EXPERIMENTS}/{exp_id}"
    with open(os.path.join(path, "progress.json"), "r") as f:
        progress = json.load(f)

    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    info = dict(
        env=meta["environment"]["name"],
        config=meta["config"] if "config" in meta else None,
        host=meta["host"] if "host" in meta else "unknown",
        current_id=exp_id,
        next_id=experiment_paths[current_index + 1] if current_index != len(experiment_paths) - 1 else None,
        prev_id=experiment_paths[current_index - 1] if current_index != 0 else None,
        hps=meta["hyperparameters"],
        env_meta=meta["environment"],
        reward_function=meta["reward_function"] if "reward_function" in meta else {},
        iterations=meta["iterations"] if "iterations" in meta else None
    )

    plots = {"normalization": {}}

    stats_path = os.path.join(path, "statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)

        info.update(dict(
            statistics=stats
        ))

        if "used_memory" in stats:
            plots["mem_usage"] = embed.components(plot_memory_usage(stats["used_memory"]))

        if "cycle_timings" in stats:
            plots["timings"] = embed.components(plot_execution_times(stats["cycle_timings"],
                                                    stats.get("optimization_timings"),
                                                    stats.get("gathering_timings")))

    if "RewardNormalizationTransformer" in progress["preprocessors"]:
        plots["normalization"]["reward"] = embed.components(plot_preprocessor(progress["preprocessors"]["RewardNormalizationTransformer"]))

    if "StateNormalizationTransformer" in progress["preprocessors"]:
        plots["normalization"]["state"] = embed.components(plot_preprocessor(progress["preprocessors"]["StateNormalizationTransformer"]))

    reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
        meta["environment"]["reward_threshold"])

    plots["reward_distribution"] = embed.components(plot_distribution(progress["rewards"]["last_cycle"], "Rewards (Last Cycle)", color=0))
    plots["length_distribution"] = embed.components(plot_distribution(progress["lengths"]["last_cycle"], "Episode Lengths (Last Cycle)", color=1))
    cycles_loaded = []
    if "loaded_at" in stats.keys():
        cycles_loaded = stats["loaded_at"]
    plots["reward_progress"] = embed.components(plot_reward_progress(progress["rewards"], cycles_loaded, reward_threshold=reward_threshold))
    plots["length_progress"] = embed.components(plot_length_progress(progress["lengths"], cycles_loaded))
    plots["policy_loss"] = embed.components(plot_loss(progress["ploss"], progress["rewards"]["mean"], "Policy Loss", color_id=0))
    plots["value_loss"] = embed.components(plot_loss(progress["vloss"], progress["rewards"]["mean"], "Value Loss", color_id=1))
    plots["entropies"] = embed.components(plot_loss(progress["entropies"], progress["rewards"]["mean"], "Entropy", color_id=2))

    if "per_receptor_mean" in stats.keys():
        plots["per_receptor_mean"] = embed.components(plot_per_receptor_mean(stats["per_receptor_mean"]))

    plots["auxiliary_plots"] = {}
    if "auxiliary_performances" in stats.keys():
        for aux_perf_key, aux_perf_values in stats["auxiliary_performances"].items():
            plots["auxiliary_plots"][aux_perf_key] = embed.components(
                plot_aux_perf_progress(aux_perf_values, cycles_loaded, perf_name=aux_perf_key))

    info.update(dict(
        plots=plots
    ))

    return flask.render_template("experiment.html", info=info)


@app.route("/compare/", methods=("POST", "GET"))
def show_comparison():
    """Compare experiments of given IDs."""
    ids = request.args.getlist('ids')

    info = {
        "ids": ids,
        "plots": {}
    }

    progress_reports = {}
    metas = {}
    for exp_id in ids:
        path = f"{PATH_TO_EXPERIMENTS}/{exp_id}"
        with open(os.path.join(path, "progress.json"), "r") as f:
            progress = json.load(f)

        with open(os.path.join(path, "meta.json"), "r") as f:
            try:
                meta = json.load(f)
            except:
                continue

        progress_reports.update({f"{exp_id}": progress})
        metas[str(exp_id)] = meta

    info["plots"]["reward"] = embed.components(compare_reward_progress(
        {id: progress["rewards"] for id, progress in progress_reports.items()},
        None
    ))

    info["plots"]["reward_grouped"] = embed.components(grouped_reward_progress(
        {id: (metas[str(id)]["hyperparameters"]["epochs_per_cycle"], progress["rewards"]) for id, progress in progress_reports.items() if metas[str(id)]["hyperparameters"]["epochs_per_cycle"] > 30},
        None
    ))

    return flask.render_template("compare.html", info=info)


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

                if "rewards" in progress and len(progress["rewards"]["mean"]) <= 30:
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


@app.route('/delete_experiments', methods=['GET', 'POST'])
def delete_experiments():
    """Delete multiple experiments of posted ids."""
    if request.method == "POST":
        try:
            ids = request.json['ids']
            ids = ids.split(",")
            ids = list(map(int, ids))

            for id in ids:
                shutil.rmtree(os.path.join(PATH_TO_EXPERIMENTS, str(id)))
        except Exception:
            return {"success": False}
    else:
        return {"success": False}

    return {"success": True}


@app.route('/regroup_experiments', methods=['GET', 'POST'])
def regroup_experiments():
    """Assign new group to multiple experiments of posted ids."""
    if request.method == "POST":
        try:
            ids = request.json['ids']
            ids = ids.split(",")
            ids = list(map(int, ids))
            new_group = request.json['group']

            for id in ids:
                with open(os.path.join(PATH_TO_EXPERIMENTS, str(id), "meta.json"), "r") as f:
                    data = json.load(f)

                data.update({"experiment_group": new_group})

                with open(os.path.join(PATH_TO_EXPERIMENTS, str(id), "meta.json"), "w") as f:
                    json.dump(data, f)

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
            evaluation_stats, _ = agent.evaluate(10, save=True)

            return {"results": evaluation_stats._asdict()}

        except Exception as e:
            return {"success": e.__repr__()}

    return {"success": "success"}
