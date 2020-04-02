import torch
from torch import optim

from schrodinet.sampler.metropolis import Metropolis
from schrodinet.wavefunction.wf_potential import Potential
from schrodinet.solver.solver_potential import SolverPotential
from schrodinet.solver.plot_potential import plot_results_1d, plotter1d


# def pot_func(pos):
#     '''H type of thing.'''
#     if isinstance(pos,torch.Tensor):
#         return -1./(torch.abs(pos)+1E-6) 
#     else:
#         return -1./np.abs(pos) 

def pot_func(pos):
    '''H with pseudo potential BFK'''
    alpha, beta, gamma, delta = 4.47692410, 2.97636451, -4.32112340, 3.01841596
    r = torch.abs(pos)
    pot = -1./r \
        + 1./r * torch.exp(-alpha*r**2) \
        + alpha * r * torch.exp(-beta*r**2) + \
        gamma * torch.exp(-delta*r**2)
    return pot


# box   
domain, ncenter = {'min': -5., 'max': 5.}, 11

# wavefunction
wf = Potential(pot_func, domain, ncenter, fcinit='random', nelec=1, sigma=1., basis='gaussian')
# wf.rbf.centers.data.fill_(0.)
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

# get the numrical solution                         
x, psi0, e0 = solver.get_numerical_solution()

# train the wave function
plotter = plotter1d(wf, domain, 100, sol={'x':x,'y':psi0}, ymin=-2.5)  
solver.run(300, loss='energy-manual', plot=plotter, save='model.pth')

# plot the final wave function
plot_results_1d(solver, domain, 100, sol={'x':x,'y':psi0}, e0=e0, load='model.pth', ymin=-2.5)
