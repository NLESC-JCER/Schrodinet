import torch
from torch.autograd import grad
from torch import nn
import numpy as np
from schrodinet.wavefunction.wf_base import WaveFunction
from schrodinet.wavefunction.wave_function_1d import WaveFunction1D
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as T
from normflows.core import NormalizingFlow


class GaussianTransform(dist.transforms.Transform):
    r""" Transform via the mapping :math:`y=\exp(\alpha(x-x_0)^2)`
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, mu, alpha):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.alpha = nn.Parameter(alpha)

    def _call(self, x):
        return torch.erfinv(x)

    def _inverse(self, x):
        return torch.erf(x)

    def log_abs_det_jacobian(self, x, y):
        return torch.log(2/torch.sqrt(np.pi) - x**2)

    def first_derivative(self, x):
        return 2/torch.sqrt(np.pi)*torch.exp(-x**2)

    def second_derivative(self, x):
        return -2*x * self.first_derivative(x)

    def third_derivative(self, x):
        return 4/torch.sqrt(np.pi)*torch.exp(-x**2) * (2*x**2-1)

    def parameters(self):
        return [self.mu, self. alpha]


class NFlow(NormalizingFlow):

    def __init__(self, base, transforms):
        super().__init__(base, transforms)
        self.base_dist = self.q0

    def prob(self, x):
        return torch.exp(self.log_prob(x.flatten()))

    def _get_derivatives(self, x):

        # detach pos
        x = x.detach()
        x.requires_grad = True

        # compute prob
        val = self.prob(x)

        # compute grad of prob
        val.backward(torch.ones_like(val), create_graph=True)
        grad_val = x.grad.clone()

        # compute second derivative
        x.grad.backward(torch.ones_like(x.grad), create_graph=True)
        hess_val = x.grad-grad_val

        return val.view(-1, 1), grad_val, hess_val

    def sample(self, num_samples):

        if isinstance(num_samples, list):
            if len(num_samples) == 1:
                return super().sample(num_samples[0])[0]
            else:
                raise ValueError('bla')
        else:
            return super().sample(num_samples)[0]


class Flow(dist.TransformedDistribution):

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__(base_dist, transforms)

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def _get_derivatives(self, x):

        # detach pos
        x = x.detach()
        x.requires_grad = True

        # compute prob
        val = self.prob(x)

        # compute grad of prob
        val.backward(torch.ones_like(val), create_graph=True)
        grad_val = x.grad.clone()

        # compute second derivative
        x.grad.backward(torch.ones_like(x.grad), create_graph=True)
        hess_val = x.grad-grad_val

        return val.view(-1, 1), grad_val, hess_val

    def _get_numerical_derivatives(self, x, eps=1E-6):

        # detach pos
        x = x.detach()
        x = x.repeat(1, 3) + torch.as_tensor([0, eps, -eps])
        shape = x.shape

        all_val = self.prob(x.reshape(-1, 1)).reshape(*shape)

        val = all_val[:, 0]
        grad_val = 0.5/eps * (all_val[:, 1]-all_val[:, 2])
        hess_val = 1./eps/eps * \
            (all_val[:, 1]+all_val[:, 2]-2*all_val[:, 0])
        return val.view(-1, 1), grad_val.view(-1, 1), hess_val.view(-1, 1)


class WaveFunctionFlow1D(WaveFunction):

    def __init__(self, fpot, flow):

        super(WaveFunctionFlow1D, self).__init__(1, 1)
        self.flow = flow

        # book the potential function
        self.user_potential = fpot

    def forward(self, pos):
        return self.flow.prob(pos).view(-1, 1)

    def local_energy(self, pos):
        ''' local energy of the sampling points.'''

        ke = self.kinetic_energy(pos)

        return ke \
            + self.nuclear_potential(pos)  \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def kinetic_energy(self, pos):

        rho, grad_rho, hess_rho = self.flow._get_derivatives(
            pos)

        inv_half_rho = 0.5/rho

        return -0.5*(inv_half_rho * hess_rho - (inv_half_rho * grad_rho)**2)

    def pdf(self, pos):
        '''density of the wave function.'''
        return self.forward(pos)

    def nuclear_potential(self, pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return self.user_potential(pos).flatten().view(-1, 1)

    def electronic_potential(self, pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0

    def nuclear_repulsion(self):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''
        return 0
