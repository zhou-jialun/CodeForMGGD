# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 00:24:10 2021

@author: jzhou

Classic Riemannian Gradients of MGGD
"""

import numpy as np
import numpy.linalg as npl
import scipy.special as ss
import copy

"""---------------------------------------------------------------------------
Classic gradient with respect to mu.
Input:
    -x: numpy array, dim=p;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -Gmu: numpy array, dim=p;
"""
def G_mu( x, mu, Sigma, beta ):
    dlt = np.dot(x-mu, npl.solve(Sigma, x-mu))
    Gmu = beta * np.power(dlt, beta-1) * npl.solve(Sigma, x-mu)
    return Gmu
"""---------------------------------------------------------------------------
Batch Classic gradient with respect to mu.
Input:
    -X: numpy array, dim=Txp;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -BGmu: numpy array, dim=p;
"""
def BG_mus( X, mu, Sigma, beta ):
    T = X.shape[0]
    p = X.shape[1]
    
    aryGmu = np.zeros((T,p))
    for t in range(0,T):
        x = np.copy(X[t,:])
        Gmu = G_mu( x, mu, Sigma, beta )
        aryGmu[t,:] = np.copy(Gmu)
    BGmu = np.mean(aryGmu, axis=0)
    return BGmu
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
Classic Riemannian gradient with respect to Sigmas.
Input:
    -x: numpy array, dim=p;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -GSigma: numpy array, dim=pxp;"""
def G_Sigma( x, mu, Sigma, beta ):
    Sx = np.outer( x-mu, x-mu )
    dlt = np.dot( x-mu, npl.solve(Sigma, x-mu) )
    GSigma = -(1/2)*Sigma + (1/2)*beta*np.power(dlt, beta-1)*Sx
    return GSigma
"""---------------------------------------------------------------------------
Batch classic Riemannian Gradient with respect to Sigmas.
Input:
    -X: numpy array, dim=Txp;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -BGSigma: numpy array, dim=pxp;"""
def BG_Sigma( X, mu, Sigma, beta ):
    T = X.shape[0]
    p = X.shape[1]
    
    aryGSigma = np.zeros((T,p,p))
    for t in range(0,T):
        x = np.copy(X[t,:])
        GSigma = G_Sigma(x, mu, Sigma, beta)            
        aryGSigma[t,:,:] = np.copy(GSigma)
    BGSigma = np.mean(aryGSigma, axis=0)
    return BGSigma
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
Classic gradient with respect to beta.
Input:
    -x: numpy array, dim=p;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -Gbeta: float;"""
def G_beta( x, mu, Sigma, beta ):
    p = mu.shape[0]
    dlt = np.dot(x-mu, npl.solve(Sigma, x-mu))
    p_alpha = (1/beta) * (1+(p/(2*beta))*(ss.polygamma(0,p/(2*beta))+np.log(2)))
    p_h = -(1/2) * np.power(dlt,beta) * np.log(dlt)
    Gbeta = p_alpha + p_h
    return Gbeta
"""---------------------------------------------------------------------------
Batch gradient with respect to betas.
Input:
    -X: numpy array, dim=Txp;
    -mu: numpy array, dim=p;
    -Sigma: numpy array, dim=pxp;
    -beta: float>0;
Output:
    -BGbeta: float; """
def BG_beta( X, mu, Sigma, beta ):
    T = np.size(X,0)
    aryGbeta = np.zeros(T)
    for t in range(0,T):
        x = np.copy(X[t,:])
        Gbeta = G_beta( x, mu, Sigma, beta )
        aryGbeta[t] = copy.deepcopy(Gbeta)
    BGbeta = np.mean(aryGbeta,axis=0)
    return BGbeta
"""end function------------------------------------------------------------"""



