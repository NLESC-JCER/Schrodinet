import torch
from torch import optim


from schrodinet.wavefunction.wave_function_flow_1d import WaveFunctionFlow1D, Flow
from schrodinet.solver.flow_solver import FlowSolver

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as T
from pyro.nn import AutoRegressiveNN

import matplotlib.pyplot as plt


# create the potential
def fpot(x):
    return 1./(x-1.) + 1./(x+1.)


# create the flow
dim = 1  # number of dimensions
K = 1
flow_transforms = T.Spline(1, count_bins=3)

# create the flow wave function
q0 = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
flow = Flow(q0, [flow_transforms])
wf = WaveFunctionFlow1D(fpot, flow)

# optimizer
opt = optim.Adam(flow_transforms.parameters(), lr=0.05)

# solver
solver = FlowSolver(wf=wf, optimizer=opt)

solver.run(10, 10)


# x = torch.linspace(-5, 5, 100).view(-1, 1)
# x.requires_grad = True
# val = wf.flow.prob(x)

# val.backward(torch.ones_like(val), create_graph=True)
# grad_val = x.grad.clone()

# x.grad.backward(torch.ones_like(x.grad))
# hess_val = x.grad

# plt.plot(x.detach().numpy(), val.detach().numpy())
# plt.plot(x.detach().numpy(), grad_val.detach().numpy())
# plt.plot(x.detach().numpy(), hess_val.detach().numpy())
# plt.show()


# pos = wf.flow.sample([10]).detach()
# eloc = wf.local_energy(pos)
