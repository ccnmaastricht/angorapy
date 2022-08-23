from typing import List

import numpy as np
import sklearn.linear_model
import tensorflow as tf
import tqdm
from scipy.spatial import transform
from sklearn.linear_model import Ridge

from angorapy.analysis.investigators import base_investigator
from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.senses import stack_sensations
from angorapy.common.wrappers import BaseWrapper
from angorapy.environments.rotations import quat2mat
from angorapy.utilities.util import flatten, stack_dicts


class Predictability(base_investigator.Investigator):
    """Investigate predictability of information from the activity in different regions of the network."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)
        self._data = None
        self.prepared = False

        self.n_states = np.nan

    def prepare(self, env: BaseWrapper, layers: List[str], n_states=1000):
        """Prepare samples to predict from."""
        state, info = env.reset(return_info=True)

        state_collection = []
        activation_collection = []
        other_collection = []
        prev_achieved_goals = [info["achieved_goal"]] * 50
        prev_object_quats = [info["achieved_goal"][3:]] * 10
        for _ in tqdm.tqdm(range(n_states)):
            state_collection.append(state)
            other = {
                "reward": info.get("original_reward", 0),
            }

            prepared_state = state.with_leading_dims(time=self.is_recurrent).dict()
            activations = self.get_layer_activations(layers, prepared_state)
            activation_collection.append(activations)

            probabilities = flatten(activations["output"])
            action, _ = self.distribution.act(*probabilities)

            observation, reward, done, info = env.step(action)

            other["fingertip_positions"] = env.unwrapped.get_fingertip_positions()
            other["translation"] = env.unwrapped._goal_distance(info["achieved_goal"], prev_achieved_goals[-1])
            other["translation_to_10"] = env.unwrapped._goal_distance(info["achieved_goal"], prev_achieved_goals[-10])
            other["translation_to_50"] = env.unwrapped._goal_distance(info["achieved_goal"], prev_achieved_goals[-50])
            other["object_orientation"] = env.unwrapped.data.jnt('object:joint').qpos.copy()[3:]

            prev_quat = transform.Rotation.from_quat(prev_object_quats[-1])
            prev_quat_10 = transform.Rotation.from_quat(prev_object_quats[-10])
            current_quat = transform.Rotation.from_quat(other["object_orientation"])
            change_in_orientation = (prev_quat_10 * current_quat.inv()).as_quat()

            other["rotation_matrix"] = quat2mat((prev_quat * current_quat.inv()).as_quat()).flatten()  # todo
            other["rotation_matrix_last_10"] = quat2mat(change_in_orientation).flatten()  # todo
            other["current_rotational_axis"] = 0  # todo

            other_collection.append(other)

            if done:
                state, info = env.reset(return_info=True)
                self.network.reset_states()
            else:
                state = observation
                prev_achieved_goals.append(info["achieved_goal"])

                if len(prev_achieved_goals) > 50:
                    prev_achieved_goals.pop(0)

        stacked_other_collection = stack_dicts(other_collection)
        stacked_other_collection["translation"][:-1] = stacked_other_collection["translation"][1:]
        stacked_other_collection["translation_to_10"][:-10] = stacked_other_collection["translation_to_10"][10:]
        stacked_other_collection["translation_to_50"][:-50] = stacked_other_collection["translation_to_50"][50:]

        stacked_other_collection["translation"][-1:] = stacked_other_collection["translation"][-1]
        stacked_other_collection["translation_to_10"][-10:] = stacked_other_collection["translation"][-1]
        stacked_other_collection["translation_to_50"][-50:] = stacked_other_collection["translation"][-1]

        self._data = tf.data.Dataset.from_tensor_slices({
            **stack_dicts(activation_collection),
            **stack_sensations(state_collection).dict(),
            **stacked_other_collection
        }).shuffle(n_states, reshuffle_each_iteration=False)

        self.train_data, self.val_data = self._data.skip(int(0.2 * n_states)), self._data.take(int(0.2 * n_states))
        self.prepared = True

    def measure(self, source_layer: str, target_information: str):
        """Measure the predictability of target_information based on the information in source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."
        assert source_layer in list(self.train_data.element_spec.keys()) + ["noise"]
        assert target_information in list(self.train_data.element_spec.keys())

        predictor = Ridge()

        base_shape = np.squeeze(list(self._data.as_numpy_iterator().__next__().values())[0]).shape
        X = list(self.train_data.map(lambda x: (tf.random.normal(base_shape) if source_layer == "noise" else tf.squeeze(x[source_layer]))).as_numpy_iterator())
        Y = list(self.train_data.map(lambda x: (tf.random.normal(base_shape) if source_layer == "noise" else tf.squeeze(x[target_information]))).as_numpy_iterator())
        X_test = list(self.val_data.map(lambda x: (tf.random.normal(base_shape) if source_layer == "noise" else tf.squeeze(
            x[source_layer]))).as_numpy_iterator())
        Y_test = list(self.val_data.map(lambda x: (tf.random.normal(base_shape) if source_layer == "noise" else tf.squeeze(
            x[target_information]))).as_numpy_iterator())

        predictor.fit(X, Y)

        print(f"{target_information} from {source_layer} has an R2 of {predictor.score(X_test, Y_test)}.")

    def measure_nn(self, source_layer: str, target_information: str):
        """Measure the predictability of target_information based on the information in source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."
        assert source_layer in self.list_layer_names(only_para_layers=False) + ["noise"]
        assert target_information in self._data.as_numpy_iterator().__next__().keys()

        output_dim = self._data.as_numpy_iterator().__next__()[target_information].shape[-1]

        predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_dim),
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        predictor.compile(
            optimizer,
            loss="mse",
            metrics=["mae"]
        )

        base_shape = list(self._data.as_numpy_iterator().__next__().values())[0].shape
        map_fnc = lambda x: (tf.random.normal(base_shape) if source_layer == "noise" else x[source_layer], x[target_information])
        history = predictor.fit(
            self.train_data.map(map_fnc),
            batch_size=128,
            epochs=30,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            ],
            validation_data=self.val_data.map(map_fnc)
        )

        print(
            f"Trained predictability of {target_information} from {source_layer} to a loss of "
            f"{min(history.history['val_loss'])}.\n\n")
