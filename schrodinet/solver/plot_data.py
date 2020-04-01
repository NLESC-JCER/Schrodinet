import numpy as np
import matplotlib.pyplot as plt


def plot_observable(obs_dict, e0=None, ax=None):
    '''Plot the observable selected.

    Args:
        obs_dict : dictioanry of observable
    '''
    show_plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_plot = True

    if isinstance(obs_dict, dict):
        data = obs_dict['local_energy']
    else:
        data = np.hstack(np.squeeze(np.array(obs_dict)))

    n = len(data)
    epoch = np.arange(n)

    data = np.array(data)
    idx = np.abs(data - data.mean(axis=0)) < 3. * data.var(axis=0)

    # get the variance
    emax = [np.quantile(e, 0.75) for e in data]
    emin = [np.quantile(e, 0.25) for e in data]
    
    energy = []
    for i, e in enumerate(data):
        energy.append(   e[idx[i,:]].mean() )
    energy = np.array(energy)
   
    # plot
    ax.fill_between(epoch, emin, emax, alpha=0.5, color='#4298f4')
    ax.plot(epoch, energy, color='#144477')
    if e0 is not None:
        ax.axhline(e0, color='black', linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy')

    if show_plot:
        plt.show()
