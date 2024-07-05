import torch
from .kernels import SignatureKernel

def mmd_loss(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel) -> torch.tensor:
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    kernel: kernel to be used e.g. SignatureKernel
    '''

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    K_XX = kernel(X,X)
    K_YY = kernel(Y,Y)
    K_XY = kernel(X,Y)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)

    mmd = (torch.sum(K_XX[~torch.eye(*K_XX.shape,dtype=torch.bool)]) / (n*(n-1))
           + torch.sum(K_YY[~torch.eye(*K_YY.shape, dtype=torch.bool)]) / (m*(m-1))
           - 2*torch.sum(K_XY)/(n*m))

    return mmd