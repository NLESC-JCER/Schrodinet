import torch
from torch import nn
import torch.nn.functional as F

class RBF_Slater_MultiElec(nn.Module):

    def __init__(self,
                 input_features,
                 output_features,
                 centers,
                 sigma,
                 nelec):
        '''Radial Basis Function Layer in N dimension and M electron

        Args:
            input_features: input side
            output_features: output size
            centers : position of the atoms
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Slater_NELEC_GENERAL, self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(centers)
        self.centers.requires_grad = True
        self.ncenter = len(self.centers)

        # get the standard deviations
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = True

        # wavefunction data
        self.nelec = nelec
        self.ndim = int(self.input_features/self.nelec)

    def forward(self, input):

        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nrbf,Ndim)
        delta = (input.view(-1, self.nelec, 1, self.ndim) -
                 self.centers[None, ...])

        # compute the distance
        # -> (Nbatch,Nelec,Nrbf)
        X = (delta**2).sum(3)
        X = torch.sqrt(X)

        # multiply by the exponent and take the exponential
        # -> (Nbatch,Nelec,Nrbf)
        X = torch.exp(-self.sigma*X)

        return X
