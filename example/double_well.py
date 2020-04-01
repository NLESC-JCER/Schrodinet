import torch
from torch import optim

from schrodinet.sampler.metropolis import Metropolis
from schrodinet.wavefunction.wf_potential import Potential
from schrodinet.solver.solver_potential import SolverPotential
from schrodinet.solver.plot_potential import plot_results_1d, plotter1d


def pot_func(pos):
    '''H2+ type of thing.'''
    k=0.75*1E-1
    if isinstance(pos,torch.Tensor):
        return -k/(torch.abs(pos-1.5)+1E-6) + -k/(torch.abs(pos+1.5)+1E-6)
    else:
        return -k/np.abs(pos-1.5) + -k/np.abs(pos+1.5)


# box
domain, ncenter = {'min': -10., 'max': 10.}, 6

# wavefunction
wf = Potential(pot_func, domain, ncenter, fcinit='random', nelec=1, sigma=1.)
# wf.rbf.centers.data = torch.tensor([-1.5]*3 + [1.5]*3).view(-1,1)
# wf.rbf.centers.requires_grad = False

# sampler
sampler = Metropolis(nwalkers=2000, nstep=500,
                     step_size=0.5, nelec=wf.nelec,
                     ndim=wf.ndim, init={'min': -10, 'max': 10})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.15)
# opt = optim.SGD(wf.parameters(),lr=0.05)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

# define solver
solver = SolverPotential(wf=wf, sampler=sampler,
                         optimizer=opt, scheduler=scheduler)


# train the wave function
plotter = plotter1d(wf, domain, 100, sol=None)  # , save='./image/')
solver.run(100, loss='energy-manual', plot=plotter, save='model.pth')

# plot the final wave function
plot_results_1d(solver, domain, 100, None, e0=0.5, load='model.pth')
