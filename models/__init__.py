"""A collection of builder functions for models usable with different environments in PPO."""

from models.convolutional import *
from models.simple import *
from models.shadow import *
from models.components import *

from models.mighty_maker import *


def get_model_builder(model_type: str, shared: bool):
    """Get a builder function for a model with the described parameters."""
    # TODO shared seems not to work yet
    params = locals()
    params["shared"] = "shared" if params["shared"] else "distinct"
    params = [str(val) for key, val in sorted(params.items(), key=lambda x: x[0])]

    return globals()[f"build_{'_'.join(params)}_models"]
