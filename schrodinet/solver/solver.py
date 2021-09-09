import numpy as np
from types import SimpleNamespace
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from schrodinet.solver.torch_utils import DataSet, Loss, ZeroOneClipper


class Solver(object):

    def __init__(self, wf=None, sampler=None, optimizer=None, scheduler=None):

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer
        self.scheduler = scheduler
        self.task = "wf_opt"

        # esampling
        self.resampling(ntherm=-1,
                        resample=100,
                        resample_from_last=True,
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

    def run(self, nepoch, batchsize=None, save='model.pth',
            loss='variance', plot=None, pos=None, with_tqdm=True):
        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            pos : presampled electronic poition
            obs_dict (dict, {name: []} ) : quantities to be computed during
                                           the training
                                           'name' must refer to a method of
                                            the Solver instance
            ntherm : thermalization of the MC sampling. If negative (-N) takes
                     the last N entries
            resample : number of MC step during the resampling
            resample_from_last (bool) : if true use the previous position as
                                        starting for the resampling
            resample_every (int) : number of epch between resampling
            loss : loss used ('energy','variance' or callable (for supervised)
            plot : None or plotter instance from plot_utils.py to
                   interactively monitor the training
        '''

        # checkpoint file
        self.save_model = save

        # sample the wave function
        pos = self.sample(
            pos=pos, ntherm=self.resample.ntherm, with_tqdm=with_tqdm)

        # determine the batching mode
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf, method=loss)

        # clipper for the fc weights
        clipper = ZeroOneClipper()

        cumulative_loss = []
        min_loss = 1E3

        for n in range(nepoch):
            print('----------------------------------------')
            print('epoch %d' % n)

            cumulative_loss = 0
            for ibatch, data in enumerate(self.dataloader):

                lpos = Variable(data)
                lpos.requires_grad = True

                loss, eloc = self.evaluate_gradient(
                    lpos, self.loss.method)
                cumulative_loss += loss
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)

            if plot is not None:
                plot.drawNow()

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(
                    n, cumulative_loss, self.save_model)

            # get the observalbes
            self.get_observable(
                self.obs_dict, pos, eloc, ibatch=ibatch)
            self.print_observable(cumulative_loss)

            print('----------------------------------------')

            # resample the data
            if (n % self.resample.resample_every == 0) or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach()
                else:
                    pos = None
                pos = self.sample(
                    pos=pos, ntherm=self.resample.ntherm, with_tqdm=False)
                self.dataloader.dataset.data = pos

            if self.scheduler is not None:
                self.scheduler.step()

        # restore the sampler number of step
        self.sampler.nstep = _nstep_save

    def evaluate_gradient(self, lpos, loss):
        """Evaluate the gradient

        Arguments:
            grad {str} -- method of the gradient (auto, manual)
            lpos {torch.tensor} -- positions of the walkers


        Returns:
            tuple -- (loss, local energy)
        """

        if loss in ['variance', 'energy']:
            loss, eloc = self._evaluate_grad_auto(lpos)

        elif loss == 'energy-manual':
            loss, eloc = self._evaluate_grad_manual(lpos)

        else:
            raise ValueError('Error in gradient method')

        if torch.isnan(loss):
            raise ValueError("Nans detected in the loss")

        return loss, eloc

    def _evaluate_grad_auto(self, lpos):
        """Evaluate the gradient using automatic diff of the required loss.

        Arguments:
            lpos {torch.tensor} -- positions of the walkers

        Returns:
            tuple -- (loss, local energy)
        """

        # compute the loss
        loss, eloc = self.loss(lpos)

        # compute local gradients
        self.opt.zero_grad()
        loss.backward()

        return loss, eloc

    def _evaluate_grad_manual(self, lpos):
        """Evaluate the gradient using a low variance method

        Arguments:
            lpos {torch.tensor} -- positions of the walkers

        Returns:
            tuple -- (loss, local energy)
        """

        ''' Get the gradient of the total energy
        dE/dk = < (dpsi/dk)/psi (E_L - <E_L >) >
        '''

        # compute local energy and wf values
        psi = self.wf(lpos)
        eloc = self.wf.local_energy(lpos, wf=psi)
        norm = 1./len(psi)

        # evaluate the prefactor of the grads
        weight = eloc.clone()
        weight -= torch.mean(eloc)
        weight /= psi
        weight *= 2.
        weight *= norm

        # compute the gradients
        self.opt.zero_grad()
        psi.backward(weight)

        return torch.mean(eloc), eloc

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
                    self.obs_dict[obs +
                                  '.grad'].append(p.grad.clone().numpy())
                else:
                    self.obs_dict[obs +
                                  '.grad'].append(torch.zeros_like(p.data))

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
