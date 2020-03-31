import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from schrodinet.solver.solver_base import SolverBase
from schrodinet.solver.torch_utils import DataSet, Loss, ZeroOneClipper


class SolverPotential(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,
                 scheduler=None):
        SolverBase.__init__(self, wf, sampler, optimizer)
        self.scheduler = scheduler
        self.task = "wf_opt"

        # esampling
        self.resampling(ntherm=-1, 
                        resample=100,
                        resample_from_last=True, 
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

    def run(self, nepoch, batchsize=None, save='model.pth',  loss='variance',
            plot=None, pos = None, with_tqdm=True):
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
        pos = self.sample(pos=pos, ntherm=self.resample.ntherm, with_tqdm=with_tqdm)

        # determine the batching mode
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(self.dataset, batch_size=batchsize)

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

                loss, eloc = self.evaluate_gradient(lpos, self.loss.method)
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
            self.get_observable(self.obs_dict, pos, eloc, ibatch=ibatch)
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
        
        if loss in ['variance','energy']:
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