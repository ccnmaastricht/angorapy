"""A collection of builder functions for models usable with different envs in PPO."""
import inspect
from typing import Callable

import itertools
from collections import OrderedDict
from functools import partial

# registry for base models
MODELS_AVAILABLE = {}
MODEL_BUILDERS = {}


def register_model(model_name):
    def inner(func):
        assert all(p in inspect.signature(func).parameters for p in ["env", "distribution", "bs", "sequence_length"]), \
            (f"Model builder {func.__name__} must have the following parameters: env, distribution, bs, sequence_length"
             f"but has {inspect.signature(func).parameters}")

        MODELS_AVAILABLE[model_name] = func

        arguments = OrderedDict(sorted([
            ("model_type", ["ffn", "rnn", "lstm", "gru"]),
            ("blind", [True, False]),
            ("shared", [True, False]),
        ], key=lambda x: x[0]))

        combinations = list(itertools.product(*arguments.values()))
        parameter_dicts = list(map(lambda c: dict(zip(arguments.keys(), c)), combinations))

        for pd in parameter_dicts:
            name_parts = pd.copy()
            name_parts["shared"] = "shared" if name_parts["shared"] else "distinct"
            name_parts["blind"] = "blind" if name_parts["blind"] else ""

            func_name = f"build_{model_name}_{'_'.join([str(val) for n, val in sorted(name_parts.items(), key=lambda x: x[0]) if val])}_models"
            MODEL_BUILDERS[func_name] = partial(func, **pd)
            MODEL_BUILDERS[func_name].__name__ = func_name

        return func

    return inner


def get_model_type(model_builder) -> str:
    """Get short name of type of model builder."""
    return model_builder.keywords["model_type"]


def get_model_builder(model="simple", model_type: str = "ffn", shared: bool = False, blind: bool = True) -> Callable:
    """Get a builder function for a model with the described parameters."""
    # TODO shared seems not to work yet
    params = locals()
    params["shared"] = "shared" if params["shared"] else "distinct"
    params["blind"] = "blind" if params["blind"] else ""
    params.pop("model")

    params = [str(val) for key, val in sorted(params.items(), key=lambda x: x[0]) if val != ""]
    return MODEL_BUILDERS[f"build_{model}_{'_'.join(params)}_models"]


# def register_model(builder):
#     globals().update({builder.__name__: builder})