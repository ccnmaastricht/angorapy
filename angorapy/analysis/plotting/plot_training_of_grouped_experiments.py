import glob
import os
import json

experiment_dir = "../../../storage/experiments/"

experiment_group_inclusion_filters = {
    # all filters are "only if in" filters, i.e. if the experiment has any of the values in the filter, it will be
    # included, otherwise it will be excluded

    "group_one": {
        "architecture": ["build_fpn_v2_models"],
    },

    "group_two": {
        "architecture": ["build_fpn_v3_models"],
    },
}

# get ids of all experiments in the storage by globbing the experiment folders
experiment_ids = sorted([os.path.basename(path) for path in glob.glob(f"{experiment_dir}*")
                  if os.path.isfile(path + "/meta.json")])

# load experiment group data
group_ids = {}
for group_name, group_filters in experiment_group_inclusion_filters.items():
    group_ids[group_name] = []
    for experiment_id in experiment_ids:
        try:
            with open(f"{experiment_dir}{experiment_id}/meta.json", "r") as f:
                experiment = json.load(f)
        except json.JSONDecodeError as e:
            # print(f"Skipping experiment {experiment_id} because it has corrupted meta.json!")
            continue

        include_experiment = True
        for filter_name, filter_values in group_filters.items():
            if experiment["hyperparameters"][filter_name] not in filter_values:
                # print(f"Excluding experiment {experiment_id} because it does not match filter {filter_name}: {experiment['hyperparameters'][filter_name]}!={filter_values}")
                include_experiment = False
                break

        if include_experiment:
            group_ids[group_name].append(experiment_id)

# plot it
