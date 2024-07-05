from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import torch

# Static kernel class and subclasses
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kernel(metaclass=ABCMeta):
    '''
    Base class for static kernels.
    '''

    @abstractmethod
    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.gram_matrix(X, Y)

class LinearKernel(Kernel):

    def __init__(self):
        super().__init__()
        self.static_kernel_type = 'linear'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        return matrix_mult(X, Y, transpose_Y=True)

class RBFKernel(Kernel):

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.static_kernel_type = 'rbf'

    def gram_matrix(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        D2_scaled = squared_euclid_dist(X, Y) / self.sigma**2
        return torch.exp(-D2_scaled)

class RationalQuadraticKernel(Kernel):
    def __init__(self, sigma : float = 1.0, alpha : float = 1.0) -> None:
        super().__init__()
        self.static_kernel_type = 'rq'
        self.alpha = alpha
        self.sigma = sigma

    def gram_matrix(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        D2_scaled = squared_euclid_dist(X, Y) / (2 * self.alpha * self.sigma**2)
        return torch.pow((1 + D2_scaled), -self.alpha)

def matrix_mult(X : torch.Tensor, Y : Optional[torch.Tensor] = None, transpose_X : bool = False, transpose_Y : bool = False) -> torch.Tensor:
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

def squared_norm(X : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.sum(torch.square(X), dim=dim)

def squared_euclid_dist(X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
    X_n2 = squared_norm(X)
    if Y is None:
        D2 = (X_n2[..., :, None] + X_n2[..., None, :]) - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y, dim=-1)
        D2 = (X_n2[..., :, None] + Y_n2[..., None, :]) - 2 * matrix_mult(X, Y, transpose_Y=True)
    return D2

# Signature kernel class and functions
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel():
    def __init__(self, n_levels: int = 5, static_kernel: Optional[Kernel] = None) -> None:
        '''
        Parameters
        ----------
        n_levels: int, default=4
            The number of levels of the signature to keep. Higher order terms are truncated
        static_kernel: Kernel, default=None
            The kernel to use for the static kernel. If None, the linear kernel is used.
        '''

        self.n_levels = n_levels
        self.static_kernel = static_kernel if static_kernel is not None else LinearKernel()

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        M = self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))
        M = torch.diff(torch.diff(M, dim=1), dim=-1) # M[i,j,k,l] = k(X[i,j+1], Y[k,l+1]) - k(X[i,j], Y[k,l+1]) - k(X[i,j+1], Y[k,l]) + k(X[i,j], Y[k,l])
        n_X, n_Y = M.shape[0], M.shape[2]
        K = torch.ones((n_X, n_Y), dtype=M.dtype, device=M.device)
        K += torch.sum(M, dim=(1, -1))
        R = torch.clone(M)
        for _ in range(1, self.n_levels):
            R = M * multi_cumsum(R, axis=(1, -1))
            K += torch.sum(R, dim=(1, -1))

        return K

def multi_cumsum(M: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Computes the exclusive cumulative sum along a given set of axes.

    Args:
        K (torch.Tensor): A matrix over which to compute the cumulative sum
        axis (int or iterable, optional): An axis or a collection of them. Defaults to -1 (the last axis).
    """

    ndim = M.ndim
    axis = [axis] if np.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]

    # create slice for exclusive cumsum (slice off last element along given axis then pre-pad with zeros)
    slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
    M = M[slices]

    for ax in axis:
        M = torch.cumsum(M, dim=ax)

    # pre-pad with zeros along the given axis if exclusive cumsum
    pads = tuple(x for ax in reversed(range(ndim)) for x in ((1, 0) if ax in axis else (0, 0)))
    M = torch.nn.functional.pad(M, pads)

    return M
