"""
Created on Dec 16 2021
@author: Ilan Rosen
Differentiaion matrix operators in polar coordinates,
plus operators with the Jacobian for the Laplacian and Neumann boundary conditions
-   First derivatives d/dr, 1/r * d/dr, 1/r * d/dt
-   Second derivatives d^2/dr^2, 1/r^2 * d^2/dt^2
"""

import numpy as np
import scipy.sparse as sp

def Diff_mat_r(Nr, r):
    '''
        Args:
        Nr : number of points
        r : list of r's
    Returns:
        Dr_1d : d/dr
        rDr_1d : 1/r * d/dr
        D2r_1d : d^2/dr^2
    '''
    
    # First derivative
    Dr_1d = sp.diags([-1, 1], [-1, 1], shape = (Nr,Nr)) # A division by (2*dx) is required later.
    Dr_1d = sp.lil_matrix(Dr_1d)
    Dr_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    Dr_1d[Nr-1,[Nr-3, Nr-2, Nr-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)

    rDr_1d = Dr_1d.T.multiply(1/r).T
    
    # Second derivative
    D2r_1d = sp.diags([1, -2, 1], [-1,0,1], shape = (Nr, Nr)) # division by dx^2 required
    D2r_1d = sp.lil_matrix(D2r_1d)                  
    D2r_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required. 
    D2r_1d[Nr-1,[Nr-4, Nr-3, Nr-2, Nr-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    
    return Dr_1d, rDr_1d, D2r_1d

def Diff_mat_t(Nt):
    '''
    Args:
        Nr : number of points
    Returns:
        Dt_1d : d/dt
        D2t_1d : d^2/dt^2
    '''
    
    # First derivative
    Dt_1d = sp.diags([-1, 1], [-1, 1], shape = (Nt,Nt)) # A division by (2*dx) is required later.
    Dt_1d = sp.lil_matrix(Dt_1d)
    Dt_1d[0,-1] = [-1] # periodic
    Dt_1d[-1,0] = [1] 

    # Second derivative
    D2t_1d = sp.diags([1, -2, 1], [-1,0,1], shape = (Nt, Nt)) # division by dx^2 required
    D2t_1d = sp.lil_matrix(D2t_1d)
    D2t_1d[0, -1] = [1]
    D2t_1d[-1, 0] = [1]               
    
    return Dt_1d, D2t_1d


def Diff_mat_2D_polar(Nr,Nt, r):
    '''
    Args:
        Nr : number of points in radial coordinate
        Nt : number of points in theta coordinate
        r : radial points
    Returns: 
        Finite element matrices for the 2D space, in sparse format
        Dr_2d : d/dr
        rDr_2d : 1/r * d/dr
        d2r_2d : d^2/dr^2
        rDt_2d : 1/r * d/dt
        r2D2t_2d : 1/r^2 * d^2/dt^2
    '''
    # 1D differentiation matrices
    Dr_1d, rDr_1d, D2r_1d = Diff_mat_r(Nr, r)
    Dt_1d, D2t_1d = Diff_mat_t(Nt)


    # Sparse identity matrices
    Ir = sp.eye(Nr)
    It = sp.eye(Nt)
    # Matrix of 1/r's
    Rr = sp.spdiags([1/r], [0], Nr, Nr)
    # Rr = sp.spdiags([1/r, 1/r, 1/r, 1/r, 1/r], [-2, -1, 0, 1, 2], Nr, Nr)
    # Matrix of 1/r^2's
    R2r = sp.spdiags([1/r**2], [0], Nr, Nr)
    # R2r = sp.spdiags([1/r**2, 1/r**2, 1/r**2, 1/r**2, 1/r**2], [-2, -1, 0, 1, 2], Nr, Nr)


    
    # 2D matrix operators from 1D operators using kronecker product
    # Partial derivatives in r
    Dr_2d = sp.kron(It, Dr_1d)
    rDr_2d = sp.kron(It, rDr_1d)
    D2r_2d = sp.kron(It, D2r_1d)
    # Partial derivatives in t, with Jacobian element
    rDt_2d = sp.kron(Dt_1d, Rr)
    r2D2t_2d = sp.kron(D2t_1d, R2r)
    
   
    
    # Return compressed Sparse Row format of the sparse matrices
    return Dr_2d.tocsr(), rDr_2d.tocsr(), D2r_2d.tocsr(), rDt_2d.tocsr(), r2D2t_2d.tocsr()














