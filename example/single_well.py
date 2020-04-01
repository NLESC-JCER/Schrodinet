import torch
from torch import optim

from schrodinet.sampler.metropolis import Metropolis
from schrodinet.wavefunction.wf_potential import Potential
from schrodinet.solver.solver_potential import SolverPotential
from schrodinet.solver.plot_potential import plot_results_1d, plotter1d


def pot_func(pos):
    '''H type of thing.'''
    k=0.75*1E-1
    if isinstance(pos,torch.Tensor):
        return -k/(torch.abs(pos)+1E-6) 
    else:
        return -k/np.abs(pos) 

def pot_func(pos):
    alpha, beta, gamma, delta = 44.7692410, 29.7636451, -43.2112340, 30.1841596
    r = torch.abs(pos)
    pot = -1./r \
        + 1./r * torch.exp(-alpha*r**2) \
        + alpha * r * torch.exp(-beta*r**2) + \
        gamma * torch.exp(-delta*r**2)
    return pot

# box   
domain, ncenter = {'min': -5., 'max': 5.}, 6

# wavefunction
wf = Potential(pot_func, domain, ncenter, fcinit='random', nelec=1, sigma=1.)
# wf.rbf.centers.data = torch.tensor([0.]*6).view(-1,1)
# wf.rbf.centers.requires_grad = False

# sampler
sampler = Metropolis(nwalkers=2000, nstep=1000,
                     step_size=0.5, nelec=wf.nelec,
                     ndim=wf.ndim, init={'min': -5, 'max': 5})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.05)
# opt = optim.SGD(wf.parameters(),lr=0.05)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.75)

# define solver
solver = SolverPotential(wf=wf, sampler=sampler,
                         optimizer=opt, scheduler=scheduler)


# train the wave function
plotter = plotter1d(wf, domain, 100, sol=None)  # , save='./image/')
solver.run(100, loss='energy-manual', plot=plotter, save='model.pth')

# plot the final wave function
plot_results_1d(solver, domain, 100, None, e0=0.5, load='model.pth')
