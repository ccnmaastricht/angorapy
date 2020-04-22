# coupled Hopf model
import numpy as np
import matplotlib.pyplot as plt
from analysis.rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder

# parameters
f = .05 # intrinsic frequency
G = .1 # coupling strength
a1 = .9 # bifurcation parameter of region 1
a2 = .0 # bifurcation parameter of region 2
beta = .02 # scaling factor of noise

# coupling
C = np.zeros((2,2)) # adjacency matrix
C[(1,0)] = 1. # region 1 affects region 2
C[(0,1)] = 1. # region 2 affects region 1

# timing
dt = .1 #timestep of 100ms is sufficient since we are simulating BOLD signal
t_0 = 0
t_end = 60*5 # simulate 5 minutes
t_steps = int((t_end-t_0)/dt+1)

# setup
w = f*2.*np.pi # angular frequency
A = [a1,a2]
X = np.zeros((2,t_steps)) + 0.001 # pre-allocated signal of real component
Y = np.zeros((2,t_steps)) + 0.001# pre-allocated signal of imaginary component
dsig = 0 # np.sqrt(dt)*beta # precalculated timestep for noise

# numerical integration
for t in range(1,t_steps):
    Diff = np.tile(X[:,t-1],(2,1))-np.transpose(np.tile(X[:,t-1],(2,1)))
    X[:,t] = X[:,t-1] + dt*((A-pow(X[:,t-1],2)-pow(Y[:,t-1],2))*X[:,t-1]-w*Y[:,t-1] + G*(np.sum(C*Diff,axis=1))) + dsig*np.random.randn(2)
    Diff = np.tile(Y[:,t-1],(2,1))-np.transpose(np.tile(Y[:,t-1],(2,1)))
    Y[:,t] = Y[:,t-1] + dt*((A-pow(X[:,t-1],2)-pow(Y[:,t-1],2))*Y[:,t-1]+w*X[:,t-1] + G*(np.sum(C*Diff,axis=1))) + dsig*np.random.randn(2)



adamfpf = Adamfixedpointfinder(None, 'hopf',
                               max_iters=5000,
                               q_threshold=1e-01,
                               epsilon=5e-4)

fps = adamfpf.find_fixed_points(np.hstack((X[:, 1000], Y[:, 1000])), np.zeros((4, 2)))
fps_outer = adamfpf.find_fixed_points(np.ones(4), np.zeros((4, 2)))

plt.plot(X[0,:], Y[0, :],'r')
# plt.plot(X[1,:],'g')
plt.plot(fps[0]['x'][:2], fps[0]['x'][2:], 'k', linewidth=5)
plt.show()