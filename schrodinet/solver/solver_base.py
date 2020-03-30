import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np

class SolverBase(object):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer

    def resampling(self, ntherm=-1, resample=100, resample_from_last=True,
                   resample_every=1):
        '''Configure the resampling options.'''
        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = resample
        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every

    def observable(self, obs):
        '''Create the observalbe we want to track.'''

        # reset the obs
        self.obs_dict = {}

        for k in obs:
            self.obs_dict[k] = []

        if 'local_energy' not in self.obs_dict:
            self.obs_dict['local_energy'] = []

        if self.task == 'geo_opt' and 'geometry' not in self.obs_dict:
            self.obs_dict['geometry'] = []

        for key, p in zip(self.wf.state_dict().keys(), self.wf.parameters()):
            if p.requires_grad:
                self.obs_dict[key] = []
                self.obs_dict[key+'.grad'] = []

    def sample(self, ntherm=-1, ndecor=100, with_tqdm=True, pos=None):
        ''' sample the wave function.'''

        pos = self.sampler.generate(
            self.wf.pdf, ntherm=ntherm, ndecor=ndecor,
            with_tqdm=with_tqdm, pos=pos)
        pos.requires_grad = True
        return pos

    def get_observable(self, obs_dict, pos, eloc=None, ibatch=None, **kwargs):
        '''compute all the required observable.

        Args :
            obs_dict : a dictionanry with all keys
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        for obs in self. obs_dict.keys():

            if obs == 'local_energy' and eloc is not None:
                data = eloc.cpu().detach().numpy()

                if (ibatch is None) or (ibatch == 0):
                    self.obs_dict[obs].append(data)
                else:
                    self.obs_dict[obs][-1] = np.append(
                        self.obs_dict[obs][-1], data)

            # store variational parameter
            elif obs in self.wf.state_dict():
                layer, param = obs.split('.')
                p = self.wf.__getattr__(layer).__getattr__(param)
                self.obs_dict[obs].append(p.data.clone().numpy())

                if p.grad is not None:
                    self.obs_dict[obs+'.grad'].append(p.grad.clone().numpy())
                else:
                    self.obs_dict[obs+'.grad'].append(torch.zeros_like(p.data))

            # get the method
            elif hasattr(self.wf, obs):
                func = self.wf.__getattribute__(obs)
                data = func(pos)
                if isinstance(data, torch.Tensor):
                    data = data.detach().numpy()
                self.obs_dict[obs].append(data)

    def print_observable(self, cumulative_loss, verbose=False):
        """Print the observalbe to csreen

        Arguments:
            cumulative_loss {float} -- current loss value

        Keyword Arguments:
            verbose {bool} -- print all the observables (default: {False})
        """

        for k in self.obs_dict:

            if k == 'local_energy':

                eloc = self.obs_dict['local_energy'][-1]
                e = np.mean(eloc)
                v = np.var(eloc)
                err = np.sqrt(v/len(eloc))
                print('energy   : %f +/- %f' % (e, err))
                print('variance : %f' % np.sqrt(v))

            elif verbose:
                print(k + ' : ', self.obs_dict[k][-1])
                print('loss %f' % (cumulative_loss))

    def get_wf(self, x):
        '''Get the value of the wave functions at x.'''
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def energy(self, pos=None):
        '''Get the energy of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.energy(pos)

    def variance(self, pos):
        '''Get the variance of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.variance(pos)

    def single_point(self, pos=None, prt=True, ntherm=-1, ndecor=100):
        '''Performs a single point calculation.'''
        if pos is None:
            pos = self.sample(ntherm=ntherm, ndecor=ndecor)

        e, s = self.wf._energy_variance(pos)
        if prt:
            print('Energy   : ', e)
            print('Variance : ', s)
        return pos, e, s

    def save_checkpoint(self, epoch, loss, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.wf.state_dict(),
            'optimzier_state_dict': self.opt.state_dict(),
            'loss': loss
        }, filename)
        return loss

    def sampling_traj(self, pos):
        ndim = pos.shape[-1]
        p = pos.view(-1, self.sampler.nwalkers, ndim)
        el = []
        for ip in tqdm(p):
            el.append(self.wf.local_energy(ip).detach().numpy())
        return {'local_energy': el, 'pos': p}

    def run(self, nepoch, batchsize=None, loss='variance'):
        raise NotImplementedError()
