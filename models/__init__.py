"""A collection of builder functions for models usable with different environments in PPO."""
from typing import Callable

from models.components import _build_fcn_component, _build_encoding_sub_model
from models.convolutional import _build_visual_decoder, _build_visual_encoder
from models.mighty_maker import *
from models.shadow import _build_visual_encoder, build_shadow_brain_models, build_shadow_brain_base
from models.simple import build_ffn_models, build_deeper_models, build_rnn_models, _build_encoding_sub_model, \
    build_simple_models


def get_model_builder(model="simple", model_type: str = "ffn", shared: bool = True, blind: bool = False) -> Callable:
    """Get a builder function for a model with the described parameters."""
    # TODO shared seems not to work yet
    params = locals()
    params["shared"] = "shared" if params["shared"] else "distinct"
    params["blind"] = "blind" if params["blind"] else ""

    params = [str(val) for key, val in sorted(params.items(), key=lambda x: x[0]) if val != ""]

    return globals()[f"build_{'_'.join(params)}_models"]
