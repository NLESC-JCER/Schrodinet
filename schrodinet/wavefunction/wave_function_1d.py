import torch
from torch import nn

from schrodinet.wavefunction.wf_base import WaveFunction
from schrodinet.wavefunction.rbf import RBF_Gaussian as RBF


class WaveFunction1D(WaveFunction):

    def __init__(self, fpot, domain, ncenter, nelec=1, ndim=1, fcinit='random',
                 sigma=1.):
        super(WaveFunction1D, self).__init__(nelec, ndim)

        # get the RBF centers
        if not isinstance(ncenter, list):
            ncenter = [ncenter]
        self.centers = torch.linspace(
            domain['min'], domain['max'], ncenter[0]).view(-1, 1)
        self.ncenter = ncenter[0]

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter,
                       centers=self.centers, sigma=sigma,
                       opt_centers=True,
                       opt_sigma=True)

        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)
        self.fc.clip = True

        # initiaize the fc layer
        if fcinit == 'random':
            nn.init.uniform_(self.fc.weight, 0, 1)
        elif isinstance(fcinit, float):
            self.fc.weight.data.fill_(fcinit)

        # book the potential function
        self.user_potential = fpot

    def forward(self, x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1, 1)

    # def kinetic_energy_analytical(self,pos,out=None):
    #     """Fast calculation of the kinetic energy."""
    #     x = self.rbf(pos,der=2)
    #     x = self.fc(x)
    #     return -0.5*x.view(-1,1)

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
