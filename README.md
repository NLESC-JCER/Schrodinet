# Schrodinet

![Build Status](https://travis-ci.com/NLESC-JCER/Schrodinet.svg?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/38b540ecc5464901a5a48a9be037c924)](https://app.codacy.com/gh/NLESC-JCER/Schrodinet?utm_source=github.com&utm_medium=referral&utm_content=NLESC-JCER/Schrodinet&utm_campaign=Badge_Grade_Dashboard)

Quantum Monte-Carlo Simulations of one-dimensional problem using Radial Basis Functions Neural Networks.
<p align="center">
<img src="./pics/morse.gif" title="Optimization of the wave function">
</p>


## Installation

Clone the repo and `pip` insatll the code

```
git clone https://github.com/NLESC-JCER/Schrodinet/
cd Schrodinet
pip install .
```

## Harmonic Oscillator in 1D

The script below illustrates how to optimize the wave function of the one-dimensional harmonic oscillator.

```python
import torch
from torch import optim

from schrodinet.sampler.metropolis import Metropolis
from schrodinet.wavefunction.wf_potential import Potential
from schrodinet.solver.solver_potential import SolverPotential
from schrodinet.solver.plot_potential import plot_results_1d, plotter1d

def pot_func(pos):
    '''Potential function desired.'''
    return 0.5*pos**2


def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)

# Define the domain and the number of RBFs

# wavefunction
wf = Potential(pot_func, domain, ncenter, fcinit='random', nelec=1, sigma=0.5)

# sampler
sampler = Metropolis(nwalkers=1000, nstep=2000,
                     step_size=1., nelec=wf.nelec,
                     ndim=wf.ndim, init={'min': -5, 'max': 5})

# optimizer
opt = optim.Adam(wf.parameters(), lr=0.05)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

# Solver
solver = SolverPotential(wf=wf, sampler=sampler,
                         optimizer=opt, scheduler=scheduler)

# Train the wave function
plotter = plotter1d(wf, domain, 100, sol=ho1d_sol)
solver.run(300, loss='variance', plot=plotter, save='model.pth')

# Plot the final wave function
plot_results_1d(solver, domain, 100, ho1d_sol, e0=0.5, load='model.pth')
```

After otpimization the following trajectory can easily be generated :

<p align="center">
<img src="./pics/ho1d.gif" title="Optimization of the wave function">
</p>

The same procedure can be done on different potentials. The figure below shows the performace of the method on the harmonic oscillator and the morse potential.

<p align="center">
<img src="./pics/rbf1d_summary.png" title="Results of the optimization">
</p>
