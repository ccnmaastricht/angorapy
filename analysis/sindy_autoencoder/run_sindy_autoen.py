import os
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
n_episodes = 50
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
    = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])
print(f"SIMULATED {n_episodes} episodes")

dx = np.gradient(activations_over_all_episodes, axis=0)
training_data = {'x': activations_over_all_episodes,
                 'dx': dx}

layers = [64, 32, 16, 4]
seed = 1
SA = SindyAutoencoder(layers, poly_order=2, seed=seed, max_epochs=30,
                      refinement_epochs=5)

SA.train(training_data=training_data)
