from analysis.chiefinvestigation import Chiefinvestigator
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import os


os.chdir("../../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
agent_id = 1614610158  # 'CartPoleContinuous-v0'  # continuous agent, no normalization

chiefinvesti = Chiefinvestigator(agent_id)

layer_names = chiefinvesti.get_layer_names()
print(layer_names)

# collect data from episodes
n_episodes = 100
activations_over_all_episodes, inputs_over_all_episodes, actions_over_all_episodes, states_all_episodes, done \
    = chiefinvesti.get_data_over_episodes(n_episodes, "policy_recurrent_layer", layer_names[1])

a = ['x', 'x_dot', 'theta', 'theta_dot']
fig = plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(states_all_episodes[:1000, i])
    plt.title(a[i])
plt.show()

t = np.linspace(0, 1000*0.02, 1000)

X, U = [], []
for i in range(n_episodes):
    X.append(states_all_episodes[i*1000:(i+1)*1000])
    U.append(actions_over_all_episodes[i*1000:(i+1)*1000])

poly_library = ps.PolynomialLibrary(degree=2)
tri_library = ps.FourierLibrary()
combined_library = poly_library + tri_library
model = ps.SINDy(feature_library=combined_library)
model.fit(x=X, t=t, u=U, multiple_trajectories=True)
model.print()

x_dot_pred = model.predict(x=states_all_episodes, u=actions_over_all_episodes)

x_dot = ps.SmoothedFiniteDifference()._differentiate(X[0], t=t)
print(np.mean(np.square((x_dot - x_dot_pred[:1000, :]))))

a = ['x_dot', 'x_ddot', 'theta_dot', 'theta_ddot']
fig = plt.figure()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(x_dot[:, i], label='ground truth')
    plt.plot(x_dot_pred[:1000, i], label='SINDYc')
    plt.legend()
    plt.title(a[i])
plt.show()

