from matplotlib import scale
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import constraints
from torch.distributions.transforms import ExpTransform


from schrodinet.wavefunction.wave_function_flow_1d import WaveFunctionFlow1D, Flow, GaussianTransform
from schrodinet.solver.flow_solver import FlowSolver

import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import AutoRegressiveNN

import matplotlib.pyplot as plt
from schrodinet.solver.plot import plot_results_1d, plotter1d
from torch.autograd import grad, Variable

from pyro.distributions.transforms.spline import ConditionedSpline
from pyro.distributions.torch_transform import TransformModule


# create the potential
def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*pos**2

# analytic solution


def sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)


# create the flow
dim = 1  # number of dimensions
K = 1


# create base dist
mu = torch.tensor([2.], requires_grad=True)
sigma = torch.tensor([1.], requires_grad=True)
q0 = torch.distributions.Normal(mu, sigma)


# create 0 layer flow
flow = Flow(q0, [])


wf = WaveFunctionFlow1D(pot_func, flow)

# optimizer
opt = optim.Adam([q0.loc, q0.scale], lr=0.05)


# solver
solver = FlowSolver(wf=wf, optimizer=opt)

domain = {'min': -5., 'max': 5.}
plotter = plotter1d(wf, domain, 100, sol=sol,
                    xlim=(-5, 5), ylim=(0, 1.), flow=True)
solver.run(250, 1000, loss='energy-manual', plot=plotter)

# plot the final wave function
plot_results_1d(solver, domain, 100, sol,
                e0=0.5, load='model.pth', flow=True)
