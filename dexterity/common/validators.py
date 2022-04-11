import tensorflow as tf

from dexterity.common.wrappers import BaseWrapper, make_env
from dexterity.utilities.error import IncompatibleModelException


def validate_env_model_compatibility(env: BaseWrapper, model: tf.keras.Model) -> bool:
    """Validate whether the environments states can be processed by the model."""
    model_inputs = [ip.name for ip in model.inputs]
    env_outputs = list(env.reset().dict().keys())

    if not sorted(model_inputs) == sorted(env_outputs):
        raise IncompatibleModelException(
            f"The model with inputs {model_inputs} cannot handle this environment's states with senses {env_outputs}."
        )

    return True


if __name__ == '__main__':
    from environments import *
    from common.policies import BetaPolicyDistribution
    from dexterity.models import build_shadow_brain_base, build_simple_models

    e = make_env("LunarLanderContinuous-v2")
    _, _, m = build_simple_models(e, BetaPolicyDistribution(e), bs=16)
    print(validate_env_model_compatibility(e, m))

    e = make_env("ReachAbsolute-v0")
    _, _, m = build_shadow_brain_base(e, BetaPolicyDistribution(e), bs=16)
    print(validate_env_model_compatibility(e, m))

