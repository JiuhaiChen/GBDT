import pdb
import random
from functools import partial
import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init
from sklearn import preprocessing
from submodules import Propagate, PropagateNoPrecond , Attention, KernelPropagate
class LaplacianKernel(object):
    def __init__(self,graph,symmetric=True):
        adjacency=graph.adjacency_matrix()
        if symmetric:
            adjacency=(adjacency+adjacency.transpose(1,0))/2
        laplacian_matrix= torch.diag(torch.matmul(adjacency,torch.ones(adjacency.shape[0])))-adjacency
        self.laplacian_matrix=laplacian_matrix
    def diffusion_kernel(self,sigma):
        # use this to speed up https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html
        eigenValues, eigenVectors=torch.eig(self.laplacian_matrix, eigenvectors=True)
        eigenValues =eigenValues[:,0]
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        eigenValuesTr=torch.exp(sigma**2*eigenValues/2)
        kernel = torch.matmul(torch.matmul(eigenVectors,eigenValuesTr),eigenVectors.transpose(1,0))
        return kernel
    def linear_kernel(self,sigma):
        # use this to speed up https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

        return self.laplacian_matrix
class UnfoldindAndAttention(nn.Module):
    r"""

    Parameters
    ----------
    d : int
        Size of hidden layers.
    alp : float
        The :math:`\alpha` in paper. If equal to :math:`0`, will be automatically decided based
        on other hyper prameters. Default: ``0``.
    lam : float
        The :math:`\lambda` in paper. Default: ``1``.
    prop_step : int
        Number of propagation steps
    attn_aft : int
        Where to do attention. Set to -1 if don't want attention.
    tau : float
        The :math:`\tau` in paper. Default: ``0.2``.
    T : float
        The :math:`T` in paper. If < 0, :math:`T` will be set to `\infty`. Default: ``-1``.
    p : float
        The :math:`p` in paper. Default: ``1``.
    use_eta: bool
        If use eta vector.
    init_att : bool
        If True, add another attention layer before propagation. Default: ``False``.
    attn_dropout : float
        The dropout rate of attention values. Default: ``0.0``.
    precond : str
        If True, use pre conditioning and unormalized laplacian, else not use pre conditioning
        and use normalized laplacian. Default: ``True``
    """

    def __init__(self, alp=0, lam=1, prop_step=5, tau=0.2, p=2, precond=False,kernel=None):

        super().__init__()

        self.alp    = alp if alp > 0 else 1 / (lam + 1) # automatic set alpha
        self.lam    = lam
        self.tau    = tau
        self.p      = p
        self.prop_step = prop_step

        prop_method      = Propagate if precond else PropagateNoPrecond
        if kernel is not None:
            prop_method = KernelPropagate
        self.prop_layers = nn.ModuleList([prop_method() if kernel is None else prop_method(kernel=kernel) for _ in range(prop_step)])
        self.post_step = lambda x:torch.clamp(x, -1, 1)




    def forward(self, g, X, train_mask=None, label=False, error=False):

        Y = X
        g.edata["w"]    = torch.ones(g.number_of_edges(), 1, device = g.device)
        g.ndata["deg"]  = g.in_degrees().float()


        for k, layer in enumerate(self.prop_layers):
            # do unfolding

            Y = layer(g, Y, X, self.alp, self.lam)

            if label == True:
                Y[train_mask] = X[train_mask]
            elif error == True:
                Y = self.post_step(Y)
                Y[train_mask] = X[train_mask]

        return Y







