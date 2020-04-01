import torch
from torch import nn
import torch.nn.functional as F

class RBF_Gaussian(nn.Module):

    def __init__(self,
                 input_features,
                 output_features,
                 centers,
                 opt_centers=True,
                 sigma=1.0,
                 opt_sigma=False):
        '''Radial Basis Function Layer in N dimension and 1 electron

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Gaussian, self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.ncenter = len(self.centers)
        self.centers.requires_grad = opt_centers

        # get the standard deviation
        self.sigma = nn.Parameter(sigma*torch.ones(self.ncenter))
        self.sigma.requires_grad = opt_sigma

    def forward(self, input, derivative=0):
        '''Compute the output of the RBF layer'''
        
        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta = (input[:, None, :] - self.centers[None, ...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = (delta**2).sum(2)

        # divide by the determinant of the cov mat
        if derivative == 0:
            X = torch.exp(- X / (2*self.sigma**2))

        elif derivative == 2:
            er = torch.exp(-X/(2*self.sigma**2))
            ndim = delta.shape[-1]
            X =  4 * (delta**2).sum(2) / (2*self.sigma**2)**2 * er \
                - 2.*ndim / (2*self.sigma**2) * er 
            
        return X.view(-1, self.ncenter)


class RBF_Slater(nn.Module):

    def __init__(self,
                 input_features,
                 output_features,
                 centers,
                 opt_centers=True,
                 sigma=1.0,
                 opt_sigma=True):
        '''Radial Basis Function Layer in N dimension abd 1 electron

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Slater, self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.ncenter = len(self.centers)
        self.centers.requires_grad = opt_centers

        # get the standard deviation
        self.sigma = nn.Parameter(sigma*torch.ones(self.ncenter))
        self.sigma.requires_grad = opt_sigma

    def forward(self, input):

        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta = (input[:, None, :] - self.centers[None, ...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = (delta**2).sum(2)
        X = torch.sqrt(X)

        # divide by the determinant of the cov mat
        X = torch.exp(-self.sigma*X)

        return X.view(-1, self.ncenter)



