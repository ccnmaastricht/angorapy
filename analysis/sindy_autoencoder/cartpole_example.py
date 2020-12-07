import os
import math
import numpy as np
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder.sindy_autoencoder import SindyAutoencoder

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

agent_id = 1585500821  # cartpole-v1
# agent_id = 1586597938# finger tapping

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 5
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
    = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])
print(f"SIMULATED {n_episodes} episodes")

dx = np.gradient(activations_over_all_episodes, axis=0)
training_data = {'x': activations_over_all_episodes,
                 'dx': dx}


layers = [64, 32, 16, 4]
seed = 1
#SA = SindyAutoencoder(layers, poly_order=2, seed=seed, max_epochs=3000,
                      #refinement_epochs=500)

#SA.train(training_data=training_data)

# TODO: for analysis of cartpole controllers
# TODO: compare agent, SINDy model, LQR (and perhaps nonlinear controller)
# TODO: for that get sindymodel to interact meaningfully with environment
# TODO: perhaps retrain agent on continuous environment but first try LQR with discrete
# TODO: make script that runs the same environment and compare performance
# TODO: translate sindy model to LQR and compare (would require sindy model to be sort of linear)
# TODO: use sympy to get symbolic jacobian of sindy function and/or numeric jacobian around 2,0,0,0 state vector
