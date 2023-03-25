"""Maker functions that generate model builder from parameters."""
import functools
import itertools
from collections import OrderedDict
from functools import partial

from angorapy.common.activations import LiF
from angorapy.models.shadow import build_shadow_brain_base
from angorapy.models.shadow_v2 import build_shadow_v2_brain_base
from angorapy.models.simple import build_simple_models, build_deeper_models, build_wider_models

MODELS_AVAILABLE = ["simple", "deeper", "wider", "shadow", "shadowlif", "shadowv2"]

arguments = OrderedDict(sorted([
    ("model", MODELS_AVAILABLE),
    ("model_type", ["ffn", "rnn", "lstm", "gru"]),
    ("blind", [True, False]),
    ("shared", [True, False]),
], key=lambda x: x[0]))

combinations = list(itertools.product(*arguments.values()))
parameter_dicts = list(map(lambda c: dict(zip(arguments.keys(), c)), combinations))

for pd in parameter_dicts:
    if pd["model"] == "simple":
        base_function = build_simple_models
    elif pd["model"] == "shadow":
        base_function = build_shadow_brain_base
    elif pd["model"] == "shadowv2":
        base_function = build_shadow_v2_brain_base
    elif pd["model"] == "shadowlif":
        base_function = functools.partial(build_shadow_brain_base, activation=LiF)
    elif pd["model"] == "deeper":
        base_function = build_deeper_models
    elif pd["model"] == "wider":
        base_function = build_wider_models
    else:
        raise ValueError("Unknown model name registered.")

    name_parts = pd.copy()
    name_parts["shared"] = "shared" if name_parts["shared"] else "distinct"
    name_parts["blind"] = "blind" if name_parts["blind"] else ""

    func_name = f"build_{'_'.join([str(val) for n, val in sorted(name_parts.items(), key=lambda x: x[0]) if val])}_models"
    pd.pop("model")
    globals()[func_name] = partial(base_function, **pd)
    globals()[func_name].__name__ = func_name


def get_model_type(model_builder) -> str:
    """Get short name of type of a model builder."""
    for mt in arguments["model_type"]:
        if mt in model_builder.__name__:
            return mt

    return "unknown"


if __name__ == '__main__':
    print("\n".join(f for f in locals() if f.startswith("build_")))