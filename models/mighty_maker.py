"""Maker functions that generate model builder from parameters."""
import itertools
from collections import OrderedDict
from functools import partial

from models.simple import build_ffn_models, build_rnn_models

arguments = OrderedDict(sorted([
    ("model_type", ["ffn", "rnn", "lstm", "gru"]),
    ("shared", [True, False]),
], key=lambda x: x[0]))

combinations = list(itertools.product(*arguments.values()))
parameter_dicts = list(map(lambda c: dict(zip(arguments.keys(), c)), combinations))

for pd in parameter_dicts:
    base_function = build_ffn_models if pd["model_type"] == "ffn" else build_rnn_models
    func_name = f"build_{'_'.join(map(str, pd.values()))}_models"
    globals()[func_name] = partial(base_function, **pd)
    globals()[func_name].__name__ = func_name

