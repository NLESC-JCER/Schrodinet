import torch
from torch import optim

from schrodinet.sampler.metropolis import Metropolis
from schrodinet.wavefunction.wave_function_1d import WaveFunction1D
from schrodinet.solver.solver import Solver
from schrodinet.solver.plot import plot_results_1d, plotter1d


def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*pos**2


def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)


# box
domain, ncenter = {'min': -5., 'max': 5.}, 11

# wavefunction
wf = WaveFunction1D(pot_func, domain, ncenter, sigma=1.)

# sampler
sampler = Metropolis(nwalkers=10, nstep=200,
                     step_size=1., init=domain)

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.05)
# opt = optim.SGD(wf.parameters(),lr=0.05)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

# define solver
solver = Solver(wf=wf, sampler=sampler,
                optimizer=opt, scheduler=scheduler)


# pos = solver.sample()
# ref = wf.kinetic_energy(pos)
# val = wf.kinetic_energy_density(pos)


# train the wave function
plotter = plotter1d(wf, domain, 100, sol=ho1d_sol)
solver.run(300, loss='energy-manual', plot=plotter, save='model.pth')

# plot the final wave function
plot_results_1d(solver, domain, 100, ho1d_sol,
                e0=0.5, load='model.pth')
