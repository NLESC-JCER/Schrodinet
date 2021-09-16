import torch
from torch.autograd import grad
from torch import nn
import numpy as np
from schrodinet.wavefunction.wf_base import WaveFunction
from schrodinet.wavefunction.wave_function_1d import WaveFunction1D
import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as T


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


class Uniform(dist.Uniform):

    def __init__(self, low, high):
        super(Uniform, self).__init__(low, high)

    def prob(x, derivative=0):
        if derivative == 0:
            return torch.exp(super().log_prob(x))
        elif derivative == 1:
            return torch.zeros(x.shape)
        elif derivative == 2:
            return torch.zeros(x.shape)


class Normal(dist.Normal):

    def __init__(self, mu, sigma):
        super(Normal, self).__init__(mu, sigma)

    def prob(x, derivative=0):
        if derivative == 0:
            return torch.exp(super().log_prob(x))
        elif derivative == 1:
            return None
        elif derivative == 2:
            return None


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
        x.grad.backward(torch.ones_like(x.grad))
        hess_val = x.grad

        return val.view(-1, 1), grad_val, hess_val


class WaveFunctionFlow1D(WaveFunction):

    def __init__(self, fpot, flow):

        super(WaveFunctionFlow1D, self).__init__(1, 1)
        self.flow = flow

        # book the potential function
        self.user_potential = fpot

    def forward(self, nsample):

        # get sampling point
        pos = self.flow.sample([nsample])

        # score
        return self.flow.prob(pos)

    def local_energy(self, pos):
        ''' local energy of the sampling points.'''

        ke = self.kinetic_energy(pos)

        return ke \
            + self.nuclear_potential(pos)  \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def kinetic_energy(self, pos):

        rho, grad_rho, hess_rho = self.flow._get_derivatives(pos)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # uni = Uniform(-5., 5.)
    # gt = GaussianTransform(torch.tensor([0.]), torch.tensor([1.]))
    # affine_trans = dist.transforms.AffineTransform(loc=3, scale=0.5)

    # x = uni.sample([1000])
    # target = dist.TransformedDistribution(uni, [gt, affine_trans])
    # y = target.sample([10000])
    # plt.hist(y.numpy(), bins=50)
    # plt.show()
