import os
import numpy as np
import matplotlib.pyplot as plt
from analysis.chiefinvestigation import Chiefinvestigator
from analysis.sindy_autoencoder.sindy_autoencoder import SindyAutoencoder

os.chdir("../../")  # remove if you want to search for ids in the analysis directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# agent_id = 1607352660  # cartpole-v1
agent_id = 1607352660 # inverted pendulum no vel, continuous action
# agent_id = 1586597938# finger tapping

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 500
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
    = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])
print(f"SIMULATED {n_episodes} episodes")

dx = np.gradient(activations_over_all_episodes, axis=0)
training_data = {'x': activations_over_all_episodes,
                 'dx': dx}

layers = [64, 32, 16, 4]
seed = 1
SA = SindyAutoencoder(layers, poly_order=2, seed=seed, max_epochs=2500,
                      refinement_epochs=300)

SA.train(training_data=training_data)
SA.save_state(filename='InvPendulum')
#SA.load_state(filename='CartPoleSindy')

#a = chiefinvesti.sub_model_to.predict(np.reshape(np.array([1, 0, 0, 0]), (1, 1, 4)))
# evaluation = SA.evaluate_jacobian(a[1].reshape(64, ))
# print(evaluation)

plt.imshow(SA.coefficient_mask * SA.autoencoder['sindy_coefficients'])
plt.show()

#z_next, sindy_predict, x_next = SA.simulate_dynamics(a[1].reshape(64, ), np.zeros((64, )))

