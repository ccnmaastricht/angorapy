from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys

sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
from analysis.chiefinvestigation import Chiefinvestigator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateActivitySingleRun(ThreeDScene):

    def setup(self):
        agent_id = 1590500032  # cartpole-v1
        chiefinvesti = Chiefinvestigator(agent_id)

        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])

    def construct(self):

        RUN_TIME = 0.7

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations_over_episode)
        activations_transformed = activations_transformed * 0.5
        shape = Polygon(*activations_transformed, color=BLUE, width=0.1)
        dot = Dot(activations_transformed[0, :], color=RED, size=1.2)

        self.play(ShowCreation(shape),
                  ShowCreation(dot),
                  run_time=RUN_TIME)

        for i in range(200):

            new_dot = Dot(activations_transformed[i, :], color=RED, size=1.2)
            self.play(Transform(dot, new_dot))
            self.wait(0.1)
# TODO: add input and predictions to animation as numbers
