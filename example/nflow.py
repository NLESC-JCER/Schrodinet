import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import constraints


from schrodinet.wavefunction.wave_function_flow_1d import WaveFunctionFlow1D, NFlow
from schrodinet.solver.flow_solver import FlowSolver

import matplotlib.pyplot as plt
from schrodinet.solver.plot import plot_results_1d, plotter1d
from torch.autograd import grad, Variable

import normflow as nf


def pot_func(x):
    k = 0.75*1E-1
    x0 = 2.5
    return -k/(x-x0).abs() + -k/(x+x0).abs()


def sol(x):

    n = len(x)

    pval = np.diag(pot_func(x).detach().numpy().flatten())

    x = x.detach().numpy()
    dx2 = (x[1]-x[0])**2
    H = -0.5/dx2 * (np.eye(n, n, k=-1) - 2 *
                    np.eye(n, n) + np.eye(n, n, k=1)) + pval

    u, v = np.linalg.eigh(H)
    return torch.tensor(np.abs(v[:, 0]))


# def pot_func(pos):
#     '''Potential function desired.'''
#     return 0.5*pos**2


# def sol(pos):
#     '''Analytical solution of the 1D harmonic oscillator.'''
#     return torch.exp(-0.5*pos**2)

q0 = nf.distributions.GaussianMixture(2, 1, loc=[[-2.5], [2.5]])

# create the flow
dim = 1  # number of dimensions
K = 1
flow_transforms = []
for i in range(K):
    flow_transforms += [nf.flows.Planar((dim,), act="leaky_relu")]


flow = NFlow(q0, flow_transforms)
wf = WaveFunctionFlow1D(pot_func, flow)

# optimizer
opt = optim.Adam(flow.parameters(),
                 lr=1E-3, weight_decay=1e-4)


# solver
solver = FlowSolver(wf=wf, optimizer=opt)

domain = {'min': -5., 'max': 5.}
plotter = plotter1d(wf, domain, 100, sol=sol,
                    ylim=(-1, 1.), flow=True)
solver.run(0, 1000, loss='energy-manual', plot=plotter)

# plot the final wave function
plot_results_1d(solver, domain, 100, sol,
                e0=0.5, load='model.pth')
