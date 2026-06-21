"""Smoke tests for the built-in model builders in :mod:`angorapy.models`.

These tests exercise the model *constructors* (``simple``, ``wider``, ``deeper``)
against the full matrix of configurations the agent supports, checking that each
builds a working policy/value network whose forward pass produces parameters a
policy distribution can actually sample from. The matrix crosses:

* **action spaces / distributions** — continuous (Beta), discrete (Categorical),
  multi-discrete (MultiCategorical), and an asymmetric-observation variant;
* **architectures** — separate vs. shared policy/value bodies, feed-forward vs.
  recurrent (LSTM).

This is an integration-level shape/compatibility check: the value is in catching
a builder that produces a model incompatible with a given observation space,
action head, or recurrence setting — failures that otherwise only surface deep
inside training.
"""
from angorapy import make_task
from angorapy.common.policies import BetaPolicyDistribution, MultiCategoricalPolicyDistribution, \
    CategoricalPolicyDistribution
from angorapy.utilities.model_utils import is_recurrent_model
from angorapy.utilities.core import flatten

from angorapy.models import get_model_builder


def perform_test_on_model(model_name):
    """Build ``model_name`` in every supported configuration and run one forward pass.

    For each (environment, distribution) pair and each of the four architecture
    variants (plain, shared, recurrent, shared-recurrent), this builds the joint
    model, feeds it a single reset observation (with a leading time axis when the
    model is recurrent), and samples an action from the predicted distribution
    parameters. Reaching the end without an exception means the builder produced a
    model whose output shapes are compatible with that observation space, action
    head, and recurrence mode.
    """
    cont_env = make_task("LunarLanderContinuous-v2")
    cont_distr = BetaPolicyDistribution(cont_env)

    discrete_env = make_task("LunarLander-v2")
    discrete_distr = CategoricalPolicyDistribution(discrete_env)

    multi_discrete_env = make_task("ManipulateBlockDiscrete-v0")
    multi_discrete_distr = MultiCategoricalPolicyDistribution(multi_discrete_env)

    asymmetric_env = make_task("ManipulateBlockDiscreteAsymmetric-v0")
    asymmetric_env_distr = MultiCategoricalPolicyDistribution(asymmetric_env)

    build_model = get_model_builder(model=model_name, model_type="ffn", shared=False)
    build_shared_model = get_model_builder(model=model_name, model_type="ffn", shared=True)
    build_recurrent_model = get_model_builder(model=model_name, model_type="lstm", shared=False)
    build_shared_recurrent_model = get_model_builder(model=model_name, model_type="lstm", shared=True)

    for env, distr in [
        (cont_env, cont_distr),
        (discrete_env, discrete_distr),
        (multi_discrete_env, multi_discrete_distr),
        (asymmetric_env, asymmetric_env_distr)
    ]:
        for model in [build_model, build_shared_model, build_recurrent_model, build_shared_recurrent_model]:
            _, _, joint = model(env, distr, bs=1, sequence_length=1)
            state = env.reset()[0]
            prepared_state = state.with_leading_dims(time=is_recurrent_model(joint)).dict_as_tf()
            policy_out = flatten(joint(prepared_state, training=False))

            predicted_distribution_parameters, value = policy_out[:-1], policy_out[-1]
            # from the action distribution sample an action and remember both the action and its probability
            action, action_probability = distr.act(*predicted_distribution_parameters)


def test_simple():
    """The ``simple`` builder produces a usable model in every configuration.

    Rationale
        ``simple`` is the default architecture used across most tasks; this
        guards that it stays compatible with all action spaces and the
        feed-forward/recurrent and separate/shared variants.
    """
    perform_test_on_model("simple")


def test_wider():
    """The ``wider`` builder produces a usable model in every configuration.

    Rationale
        ``wider`` shares ``simple``'s structure with larger layers; testing it
        catches width-dependent shape regressions in the builder.
    """
    perform_test_on_model("wider")


def test_deeper():
    """The ``deeper`` builder produces a usable model in every configuration.

    Rationale
        ``deeper`` adds layers on top of ``simple``; this guards against
        depth-dependent wiring errors (e.g. an incorrectly chained block).
    """
    perform_test_on_model("deeper")
