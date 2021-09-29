import torch
from torch.distributions import constraints
import pyro.distributions.transforms as T


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
        return torch.erf(x)

    def log_abs_det_jacobian(self, x, y):
        return x**2
