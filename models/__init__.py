"""A collection of builder functions for models usable with different environments in PPO."""

from models.convolutional import *
from models.simple import *
from models.shadow import *
from models.components import *

from models.mighty_maker import *


def get_model_builder(model_type: str, shared: bool):
    # TODO shared seems not to work yet
    params = [str(val) for key, val in sorted(locals().items(), key=lambda x: x[0])]
    return globals()[f"build_{'_'.join(params)}_models"]
