"""Maker functions that generate model builder from parameters."""
import itertools
from collections import OrderedDict
from functools import partial

from models.shadow import build_shadow_brain_v1
from models.simple import build_ffn_models, build_rnn_models

arguments = OrderedDict(sorted([
    ("model", ["simple", "shadow"]),
    ("model_type", ["ffn", "rnn", "lstm", "gru"]),
    ("shared", [True, False]),
], key=lambda x: x[0]))

combinations = list(itertools.product(*arguments.values()))
parameter_dicts = list(map(lambda c: dict(zip(arguments.keys(), c)), combinations))

for pd in parameter_dicts:
    if pd["model"] == "simple":
        base_function = build_ffn_models if pd["model_type"] == "ffn" else build_rnn_models
    elif pd["model"] == "shadow":
        base_function = build_shadow_brain_v1
    else:
        raise ValueError("Unknown model name registered.")

    name_parts = pd.copy()
    name_parts["shared"] = "shared" if name_parts["shared"] else "distinct"

    func_name = f"build_{'_'.join(map(str, name_parts.values()))}_models"
    pd.pop("model")
    globals()[func_name] = partial(base_function, **pd)
    globals()[func_name].__name__ = func_name


def get_model_type(model_builder) -> str:
    """Get short name of type of a model builder."""
    for mt in arguments["model_type"]:
        if mt in model_builder.__name__:
            return mt

    return "unknown"
