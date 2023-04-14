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


class ErfTransform(T.Transform):
    r"""
    Transform via the mapping :math:`y = \erfinv(x)`.
    """
    domain = constraints.interval(-1.0, 1.0)
    codomain = constraints.real
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, ErfTransform)

    def _call(self, x):
        return torch.erfinv(x)

    def _inverse(self, y):
        return torch.erf(y)

    def log_abs_det_jacobian(self, x, y):
        return x**2


# def pot_func(x):
#     k = 0.75*1E-1
#     x0 = 2.5
#     return -k/(x-x0).abs() + -k/(x+x0).abs()


# def sol(x):

#     n = len(x)

#     pval = np.diag(pot_func(x).detach().numpy().flatten())

#     x = x.detach().numpy()
#     dx2 = (x[1]-x[0])**2
#     H = -0.5/dx2 * (np.eye(n, n, k=-1) - 2 *
#                     np.eye(n, n) + np.eye(n, n, k=1)) + pval

#     u, v = np.linalg.eigh(H)
#     return torch.tensor(np.abs(v[:, 0]))

def pot_func(pos):
    '''Potential function desired.'''
    pos = pos-2.
    return 0.5*(torch.exp(-2.*(pos)) - 2.*torch.exp(-pos)).view(-1, 1)


def sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    pos = pos-2
    vn = torch.exp(-torch.exp(-pos)-0.5*pos)
    return vn / torch.max(vn)


# def pot_func(pos):
#     '''Potential function desired.'''
#     return 0.5*pos**2


# def sol(pos):
#     '''Analytical solution of the 1D harmonic oscillator.'''
#     return torch.exp(-0.5*pos**2)


# create the flow
dim = 1  # number of dimensions
K = 1

# # create the flow wave function
# q0 = torch.distributions.Uniform(
#     torch.tensor([-2.]), torch.tensor([2.]))


mu = torch.tensor([-3.], requires_grad=True)
sigma = torch.tensor([0.5], requires_grad=True)
q0 = torch.distributions.Normal(mu, sigma)


# mix = dist.Categorical(torch.ones(2,))
# comp = dist.Normal(torch.as_tensor([-2.5, 2.5]), torch.ones(2,), 1)
# q0 = MultiModalGaussian(mix, comp)


# flow_transforms = MySpline(
#     1, count_bins=64, bound=5, order="linear")

# flow_transforms = T.spline_coupling(1, count_bins=4, bound=3)
# flow_transforms = T.AffineTransform(0., 1.)
# flow_transforms = T.Planar(1)
# flow_transforms = T.PowerTransform(2)
# a = torch.tensor([0.], requires_grad=True)
# b = torch.tensor([5.], requires_grad=True)
# flow_transforms = GaussianTransform(a, b)
# flow_transforms = T.AffineAutoregressive(AutoRegressiveNN(1, [40]))


loc = torch.tensor([1.], requires_grad=True)
s = torch.tensor([1.], requires_grad=True)
affine_transform = T.AffineTransform(loc=loc, scale=s)
exp_transform = T.ExpTransform()
s = torch.tensor([1.], requires_grad=True)
pow_transform = T.PowerTransform(s)
flow_transform = [affine_transform, exp_transform, pow_transform]

flow = Flow(q0, flow_transform)


wf = WaveFunctionFlow1D(pot_func, flow)

# optimizer
opt = optim.Adam(
    [affine_transform.loc, affine_transform.scale]
    + [q0.loc, q0.scale]
    + [pow_transform.exponent], lr=0.05)
# opt = optim.Adam(
#     list(flow_transforms.parameters()))
# opt = optim.Adam([q0.loc] + [q0.scale], lr=0.05)

# solver
solver = FlowSolver(wf=wf, optimizer=opt)

domain = {'min': -5., 'max': 10.}
plotter = plotter1d(wf, domain, 100, sol=sol,
                    xlim=(-5, 10), ylim=(-1, 1.), flow=True)
solver.run(500, 1000, loss='energy-manual', plot=plotter)

# plot the final wave function
plot_results_1d(solver, domain, 100, sol,
                e0=0.5, load='model.pth', flow=True)

# x = torch.rand(10, 1)
# y = x.clone()
# y.requires_grad = True
# val, gval, hval = wf.flow._get_derivatives(x)

# val = wf(y)
# z = Variable(torch.ones(val.shape))
# ggval = grad(val, y, grad_outputs=z, create_graph=True)[0]

# hess = torch.zeros(ggval.shape[0])
# z = Variable(torch.ones(ggval.shape[0]))

# for idim in range(ggval.shape[1]):

#     tmp = grad(ggval[:, idim], y,
#                grad_outputs=z,
#                only_inputs=True,
#                create_graph=True)[0]

#     hess += tmp[:, idim]
# hess = hess.view(-1, 1)

# print(hess)
# print(hval)
