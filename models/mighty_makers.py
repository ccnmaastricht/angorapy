"""Maker functions that generate model builder from parameters."""
import itertools
from collections import OrderedDict

arguments = OrderedDict([
    ("model_types", ["ffn", "rnn"]),
    ("test", ["a", "b"]),
])


def _bind_generic_model_builders():
    combinations = list(itertools.product(*arguments.values()))
    parameter_dicts = list(map(lambda c: dict(zip(arguments.keys(), c)), combinations))

    for pd in parameter_dicts:
        pass


def make_generic_model_builder(model_type: str):
    pass


if __name__ == '__main__':
    _bind_generic_model_builders()
