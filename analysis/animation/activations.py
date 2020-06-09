from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append("/home/raphael/Code/dexterous-robot-hand/")
from analysis.chiefinvestigation import Chiefinvestigator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateActivityCartPole(ThreeDScene):

    def setup(self):
        agent_id = 1590500032  # cartpole-v1
        chiefinvesti = Chiefinvestigator(agent_id)

        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode, images = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])


        fig = plt.figure()

        im = plt.imshow(images[0])
        plt.axis('off')

        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(images[i])
            return [im]

        anim = FuncAnimation(fig, animate, frames=200, interval=25, blit=True)
        anim.save('test_video.mp4', dpi=500)

    def construct(self):

        RUN_TIME = 0.5
        TRANSFORM_RUN_TIME = 0.3

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations_over_episode)
        activations_transformed = activations_transformed * 0.45
        shape = Polygon(*activations_transformed, color=BLUE, width=0.1)
        dot = Dot(activations_transformed[0, :], color=RED, size=1.2)

        action_text = TextMobject("push left", color=RED).to_edge(RIGHT)
        self.play(ShowCreation(shape),
                  ShowCreation(dot),
                  ShowCreation(action_text),
                  run_time=RUN_TIME)

        for i in range(2):
            if self.actions_over_episode[i] == 0:
                new_dot = Dot(activations_transformed[i, :], color=RED, size=1.2)
                new_action_tet = TextMobject("push left", color=RED).to_edge(RIGHT)
            else:
                new_dot = Dot(activations_transformed[i, :], color=GREEN, size=1.2)
                new_action_tet = TextMobject("push right", color=GREEN).to_edge(RIGHT)
            self.play(Transform(dot, new_dot),
                      Transform(action_text, new_action_tet),
                      run_time=TRANSFORM_RUN_TIME)
            self.wait(0.1)


class AnimateActivityMountainCar(ThreeDScene):

    def setup(self):
        agent_id = 1585557832 # mountaincar
        chiefinvesti = Chiefinvestigator(agent_id)

        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode, images = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])


        fig = plt.figure()

        im = plt.imshow(images[0])
        plt.axis('off')

        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(images[i])
            return [im]

        anim = FuncAnimation(fig, animate, frames=len(self.activations_over_episode)-1, interval=100, blit=True)
        anim.save('test_video.mp4', dpi=200)

    def construct(self):

        RUN_TIME = 0.5
        TRANSFORM_RUN_TIME = 0.1

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations_over_episode)
        activations_transformed = activations_transformed * 0.45
        shape = Polygon(*activations_transformed, color=BLUE, width=0.1)
        dot = Dot(activations_transformed[0, :], color=RED, size=1.2)

        action_text = TextMobject("push left", color=RED).to_edge(RIGHT)
        self.play(ShowCreation(shape),
                  ShowCreation(dot),
                  ShowCreation(action_text),
                  run_time=RUN_TIME)

        for i in range(len(self.activations_over_episode)):
            if self.actions_over_episode[i] == 0:
                new_dot = Dot(activations_transformed[i, :], color=RED, size=1.2)
                new_action_tet = TextMobject("push left", color=RED).to_edge(RIGHT)
            elif self.actions_over_episode[i] == 1:
                new_dot = Dot(activations_transformed[i, :], color=BLUE, size=1.2)
                new_action_tet = TextMobject("no push", color=BLUE).to_edge(RIGHT)
            else:
                new_dot = Dot(activations_transformed[i, :], color=GREEN, size=1.2)
                new_action_tet = TextMobject("push right", color=GREEN).to_edge(RIGHT)
            self.play(Transform(dot, new_dot),
                      Transform(action_text, new_action_tet),
                      run_time=TRANSFORM_RUN_TIME)
            #self.wait(0.3)


class AnimateActivityForeFinger(ThreeDScene):

    def setup(self):
        agent_id, env = 1588151579, 'HandFreeReachFFAbsolute-v0'  # small step reach task
        chiefinvesti = Chiefinvestigator(agent_id, env)

        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                                  layer_names[1])

    def construct(self):

        RUN_TIME = 0.5
        TRANSFORM_RUN_TIME = 0.1

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations_over_episode)
        activations_transformed = activations_transformed * 0.45
        shape = Polygon(*activations_transformed, color=BLUE, width=0.1)
        dot = Dot(activations_transformed[0, :], color=RED, size=1.2)


        self.play(ShowCreation(shape),
                  ShowCreation(dot),
                  run_time=RUN_TIME)

        for i in range(len(self.activations_over_episode)):

            new_dot = Dot(activations_transformed[i, :], color=RED, size=1.2)

            self.play(Transform(dot, new_dot),
                      run_time=TRANSFORM_RUN_TIME)
            # self.wait(0.3)


