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


class MySpline(ConditionedSpline, TransformModule):
    r"""
    An implementation of the element-wise rational spline bijections of linear and
    quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020). Rational splines
    are functions that are comprised of segments that are the ratio of two
    polynomials. For instance, for the :math:`d`-th dimension and the :math:`k`-th
    segment on the spline, the function will take the form,

        :math:`y_d = \frac{\alpha^{(k)}(x_d)}{\beta^{(k)}(x_d)},`

    where :math:`\alpha^{(k)}` and :math:`\beta^{(k)}` are two polynomials of
    order :math:`d`. For :math:`d=1`, we say that the spline is linear, and for
    :math:`d=2`, quadratic. The spline is constructed on the specified bounding box,
    :math:`[-K,K]\times[-K,K]`, with the identity function used elsewhere.

    Rational splines offer an excellent combination of functional flexibility whilst
    maintaining a numerically stable inverse that is of the same computational and
    space complexities as the forward operation. This element-wise transform permits
    the accurate represention of complex univariate distributions.

    Example usage:

    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> transform = Spline(10, count_bins=4, bound=3.)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. This is required so we know how
        many parameters to store.
    :type input_dim: int
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, input_dim, count_bins=8, bound=3.0, order="linear"):
        super(MySpline, self).__init__(self._params)

        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order

        self.unnormalized_widths = nn.Parameter(
            torch.ones(self.input_dim, self.count_bins)
        )
        self.unnormalized_heights = nn.Parameter(
            torch.ones(self.input_dim, self.count_bins)
        )
        self.unnormalized_derivatives = nn.Parameter(
            torch.ones(self.input_dim, self.count_bins - 1)
        )

        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(
                torch.rand(self.input_dim, self.count_bins)
            )
        elif self.order != "quadratic":
            raise ValueError(
                "Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(
                    self.order
                )
            )

    def _params(self):
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)
        if self.order == "linear":
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None
        return w, h, d, l


class MultiModalGaussian(dist.MixtureSameFamily):

    def __init__(self, mix, comp):
        super().__init__(mix, comp)

    def sample(self, size):
        return super().sample(size).view(-1, 1)

    def prob(self, x):
        return torch.exp(self.log_prob(x))
