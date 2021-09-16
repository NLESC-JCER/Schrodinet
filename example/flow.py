import torch
from torch import optim
import numpy as np


from schrodinet.wavefunction.wave_function_flow_1d import WaveFunctionFlow1D, Flow
from schrodinet.solver.flow_solver import FlowSolver

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as T
from pyro.nn import AutoRegressiveNN

import matplotlib.pyplot as plt
from schrodinet.solver.plot import plot_results_1d, plotter1d
from torch.autograd import Variable


class MultiModalGaussian(dist.MixtureSameFamily):

    def __init__(self, mix, comp):
        super().__init__(mix, comp)

    def sample(self, size):
        return super().sample(size).view(-1, 1)

    def prob(self, x):
        return torch.exp(self.log_prob(x))

# create the potential


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


def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*pos**2


def sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)


# create the flow
dim = 1  # number of dimensions
K = 1

# create the flow wave function
q0 = torch.distributions.Uniform(
    torch.tensor([-10.]), torch.tensor([10.]))
q0 = torch.distributions.Normal(
    Variable(torch.ones(dim)), Variable(torch.ones(dim)))

mix = dist.Categorical(torch.ones(2,))
comp = dist.Normal(torch.as_tensor([-2.5, 2.5]), torch.ones(2,))
q1 = MultiModalGaussian(mix, comp)


# arn = AutoRegressiveNN(1, [40], param_dims=[16]*3)
# flow_transforms = T.NeuralAutoregressive(arn, hidden_units=16)


flow_transforms = T.Spline(1, count_bins=3, bound=1)
# flow_transforms = T.spline_coupling(1, count_bins=4, bound=3)
# flow_transforms = T.Planar(1)
flow = Flow(q0, [flow_transforms])
wf = WaveFunctionFlow1D(pot_func, flow)

# optimizer
opt = optim.Adam(flow_transforms.parameters(), lr=0.05)

# solver
solver = FlowSolver(wf=wf, optimizer=opt)

domain = {'min': -5., 'max': 5.}
plotter = plotter1d(wf, domain, 100, sol=sol,
                    ylim=(-1, 1.), flow=True)
solver.run(500, 1000, loss='variance', plot=plotter)

# plot the final wave function
plot_results_1d(solver, domain, 100, sol,
                e0=0.5, load='model.pth')
