from angorapy import make_task
from angorapy.common.policies import BetaPolicyDistribution, MultiCategoricalPolicyDistribution, \
    CategoricalPolicyDistribution
from angorapy.utilities.model_utils import is_recurrent_model
from angorapy.utilities.core import flatten

from angorapy.models import get_model_builder


def perform_test_on_model(model_name):
    cont_env = make_task("LunarLanderContinuous-v2")
    cont_distr = BetaPolicyDistribution(cont_env)

    discrete_env = make_task("LunarLander-v2")
    discrete_distr = CategoricalPolicyDistribution(discrete_env)

    multi_discrete_env = make_task("ManipulateBlockDiscrete-v0")
    multi_discrete_distr = MultiCategoricalPolicyDistribution(multi_discrete_env)

    build_model = get_model_builder(model=model_name, model_type="ffn", shared=False)
    build_shared_model = get_model_builder(model=model_name, model_type="ffn", shared=True)
    build_recurrent_model = get_model_builder(model=model_name, model_type="lstm", shared=False)
    build_shared_recurrent_model = get_model_builder(model=model_name, model_type="lstm", shared=True)

    for env, distr in [(cont_env, cont_distr), (discrete_env, discrete_distr), (multi_discrete_env, multi_discrete_distr)]:
        for model in [build_model, build_shared_model, build_recurrent_model, build_shared_recurrent_model]:
            _, _, joint = model(env, distr, bs=1, sequence_length=1)
            state = env.reset()[0]
            prepared_state = state.with_leading_dims(time=is_recurrent_model(joint)).dict_as_tf()
            policy_out = flatten(joint(prepared_state, training=False))

            predicted_distribution_parameters, value = policy_out[:-1], policy_out[-1]
            # from the action distribution sample an action and remember both the action and its probability
            action, action_probability = distr.act(*predicted_distribution_parameters)


def test_simple():
    """Test simple model."""
    perform_test_on_model("simple")


def test_wider():
    """Test simple model."""
    perform_test_on_model("wider")


def test_deeper():
    """Test deeper model."""
    perform_test_on_model("deeper")
