"""A collection of builder functions for models usable with different environments in PPO."""

from models.components import *
from models.convolutional import *
from models.mighty_maker import *
from models.shadow import *
from models.simple import *


def get_model_builder(model="simple", model_type: str = "ffn", shared: bool = True):
    """Get a builder function for a model with the described parameters."""
    # TODO shared seems not to work yet
    params = locals()
    params["shared"] = "shared" if params["shared"] else "distinct"
    params = [str(val) for key, val in sorted(params.items(), key=lambda x: x[0])]

    return globals()[f"build_{'_'.join(params)}_models"]
