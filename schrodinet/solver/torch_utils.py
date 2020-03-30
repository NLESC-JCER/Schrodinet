import torch
from torch import nn
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]


class Loss(nn.Module):

    def __init__(self, wf, method='variance'):

        super(Loss, self).__init__()
        self.wf = wf
        self.method = method

    def forward(self, pos):

        eloc = self.wf.local_energy(pos)

        if self.method == 'variance':
            loss = torch.var(eloc)

        elif self.method == 'energy':
            loss = torch.mean(eloc)

        elif self.method == 'energy-manual':

            loss = torch.mean(eloc)
            psi = self.wf(pos)
            norm = 1./len(psi)

            # evaluate the prefactor of the grads
            weight = eloc.clone()
            weight -= loss
            weight /= psi
            weight *= 2.
            weight *= norm

            # compute the gradients
            self.opt.zero_grad()
            psi.backward(weight)

        else:
            raise ValueError('method must be variance, energy')

        return loss, eloc


class OrthoReg(nn.Module):
    '''add a penalty to make matrice orthgonal.'''

    def __init__(self, alpha=0.1):
        super(OrthoReg, self).__init__()
        self.alpha = alpha

    def forward(self, W):
        ''' Return the loss : |W x W^T - I|.'''
        return self.alpha * torch.norm(W.mm(W.transpose(0, 1)) - torch.eye(W.shape[0]))


class UnitNormClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w).expand_as(w))


class ZeroOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.sub_(torch.min(w)).div_(torch.norm(w).expand_as(w))
