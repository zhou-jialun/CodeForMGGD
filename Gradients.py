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
def BG_mu( X, mu, Sigma, beta ):
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









"""---------------------------------------------------------------------------
Calculate the information constant of mu.
Input:
    -beta: float, shape parameter;
    -p   : int, dimension of random vector;
Output:
    -Imu : float, value of information constant;
"""
def calImu( beta,p ):
    Imu = (2*(beta-1)+p)*(p-2)*ss.gamma((p-2)/(2*beta))/\
        (p*np.power(2,1/beta)*ss.gamma(p/(2*beta)))
    return Imu
"""end function------------------------------------------------------------"""

"""---------------------------------------------------------------------------
Calculate the information constants with respect to Sigma.
Input:
    -beta: float, shape parameter;
    -p   : int, dimension of random vector;
Output:
    -I1: float, value of information constant;
    -I2: float, value of information constant;
"""
def calISigma( beta, p ):
    I1 = (p+2*beta)/(2*(p+2))
    I2 = (beta-1)/(2*(p+2))
    return (I1,I2)
"""end function------------------------------------------------------------"""

"""---------------------------------------------------------------------------
Calculate the coefficient of Information gradient with respect to Sigma.
Input:
    -beta: float, shape parameter;
    -p   : int, dimension of random vector;
Output:
    -J1: float, value of information coefficient;
    -J2: float, value of information coefficient;
"""
def calJSigma( beta, p ):
    J1 = 2*(p+2)/(p+2*beta)
    J2 = 2/beta
    return (J1,J2)
"""end function------------------------------------------------------------"""

"""---------------------------------------------------------------------------
Calculate the information constant of beta.
Input:
    -beta: float, shape parameter;
    -p   : int, dimension of random vector;
Output:
    -Ibeta: float, value of information constant; """
def calIbeta( beta, p ):
    part1 = 1 + np.power(p/(2*beta),2) * ss.polygamma(1, p/(2*beta))\
            + (p/beta)*( np.log(2) + ss.polygamma(0,p/(2*beta)) )
    part2 = (p/(2*beta)) * ( np.power(np.log(2),2) +\
            ss.polygamma(0,1+p/(2*beta))*(np.log(4) +\
            ss.polygamma(0,1+p/(2*beta))) + ss.polygamma(1,1+p/(2*beta)) )
    Ibeta = (part1 + part2)/np.power(beta,2)
    return Ibeta
"""end function------------------------------------------------------------"""








"""---------------------------------------------------------------------------
Information gradient with respect to mu.
Input:
    -Gmu : numpy array, dim=p;
    -Sigma: numpy array, dim=p;
    -Imu: float, information constant;
Output:
    -iGmu: numpy array, dim=p; """
def iG_mu( Gmu, Sigma, Imu ):
    iGmu = (1/Imu) * np.dot( Sigma, Gmu )
    return iGmu
"""end function------------------------------------------------------------"""

"""---------------------------------------------------------------------------
Information Gradient with respect to Sigma.
Input:
    -GSigma: numpy array, dim=pxp;
    -Sigma : numpy array, dim=pxp;
    -J1 : float, information coefficient;
    -J2 : float, information coefficient;
Output:
    -iGSigma: numpy array, dim=pxp; """
def iG_Sigma( GSigma, Sigma, J1, J2 ):
    p = Sigma.shape[0]
    
    G_prl = ( np.trace( npl.solve(Sigma, GSigma) )/p )*Sigma
    G_ort = GSigma - G_prl
    iGSigma = J1*G_ort + J2*G_prl
    
    return iGSigma
"""end function------------------------------------------------------------"""

"""---------------------------------------------------------------------------
Information gradient with respect to beta.
Input:
    -Gbeta : float;
    -Ibeta: float;
Output:
    -iGbeta: float; """
def iG_beta( Gbeta, Ibeta ):
    iGbeta = (1/Ibeta) * Gbeta
    return iGbeta
"""end function------------------------------------------------------------"""

















