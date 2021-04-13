# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:25:28 2021

@author: jzhou

Retractions and exponential maps
"""

import numpy as np
import numpy.linalg as npl



"""---------------------------------------------------------------------------
Exponential map on unit sphere.
Input:
    -x: numpy array, dim=K;
    -Ux: numpy array, dim=K;
Output:
    -res2: numpy array, dim=K;
"""
def ExpUS(x, Ux):
    normUx = npl.norm(Ux)
    res1 = np.cos(normUx)*x + (np.sin(normUx)/normUx)*Ux
    res2 = res1/npl.norm(res1)
    return res2
"""end funtion-------------------------------------------------------------"""



"""---------------------------------------------------------------------------
Exponential map in SPD space:
Input:
    - Sigma  : numpy array, dim=pxp, scatter matrix;
    - USigma : numpy array, dim=pxp, gradient with respect to the current Sigma;
Output:
    - new_Sigma : numpy array, dim=pxp, updated parameter;
"""
def ExpSPD(Sigma, USigma):
    new_Sigma = Sigma + USigma + 0.5 * ( np.dot( USigma, npl.solve(Sigma, USigma) ) )
    return new_Sigma
"""end funtion-------------------------------------------------------------"""



"""---------------------------------------------------------------------------
Retraction map in (0,+inf) for beta
Input:
    -beta: float>0;
    -Ubeta: float;
Output:
    -new_beta: float>0;
"""
def Retbetas(beta, Ubeta):
    new_beta = beta*np.exp(Ubeta/beta)
    return new_beta
"""end funtion-------------------------------------------------------------"""



