# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:28:50 2021

@author: jzhou

This file contains some functions to generate MGGD samples.
"""

import random
import numpy as np
import numpy.linalg as npl
import scipy.linalg as sl
import scipy.special as ss
import copy



"""---------------------------------------------------------------------------
This function generates the samples of MGGD model;
Input:
    -T: int, number of samples;
    -mu: numpy array, dim=p, location parameter;
    -Sigma: numpy array, dim=pxp, scatter matrix;
    -beta: positive float, shape parameter;
Output:
    -X: Txp array, generated samples;
"""
def mggdrand(T,mu,Sigma,beta):
    p = mu.shape[0]
    U = RandSphere( T, p )
    V = np.transpose( np.dot( sl.sqrtm(Sigma) , np.transpose(U) ) )
    var_gamma = np.power( np.random.gamma( p/(2*beta), 2, T ), 1/(2*beta))
    tau = np.transpose( np.tile( var_gamma, [p,1] ) )
    tab_mu = np.tile( mu, [T,1] )
    X = np.multiply( tau, V ) + tab_mu
    
    return X
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
This function generates the samples of the mixture of MGGDs;
Input:
    -T: integer, number of samples;
    -mus: numpy array, dim=Kxp, location parameters;
    -Sigmas: numpy array, dim=Kxpxp, scatter matrices;
    -betas: numpy array, dim=K, shape parameters;
    -w: numpy array, dim=K, the weights;    
Output:
    -X: numpy array, dim=Txp, generated samples;
"""
def mmggdrand(T,w,mus,Sigmas,betas):
    cumsum_w = np.hstack( (np.array([0]),np.cumsum(w)) )
    
    p = mus.shape[1]
    K = w.shape[0]
    
    v_index = np.zeros(int(T))
    for t in range(0,T):
        u = random.random()
        for k in range(0,K):
            if u>cumsum_w[k] and u<cumsum_w[k+1]:
                v_index[t] = copy.deepcopy(k)
                break
    
    X = np.zeros((T,p))
    for t in range(0,T):
        mu_k = np.copy( mus[int(v_index[t]),:] )
        Sigma_k = np.copy( Sigmas[int(v_index[t]),:,:] )
        beta_k = copy.deepcopy( betas[int(v_index[t])] )
        x = mggdrand(1, mu_k, Sigma_k, beta_k)
        X[t,:] = np.copy(x)
    
    return X
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
RandSphere generates random points on unit sphere;
Input:
    - T: integer, number of samples to generate;
    - p: integer, dimension of the random vector;
Output:
    - X: numpy array, dim=Nxp, generated samples; 
"""
def RandSphere(T,p):
    X = np.random.randn(T,p)
    Y = np.power(X,2)
    Z = np.sqrt( np.sum(Y,axis=1) )
    Z = np.transpose(np.tile(Z,[p,1]))
    U = X/Z
    return U
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
cov_matrix generates the random toeplitz matrix
Input:
    - p: integer, dimension of the covariance
    - var_rho: float in (0,1)
Output:
    - res: numpy array, dim=pxp, the generated covariance matrix
"""
def cov_matrix(p,var_rho):
    a = np.array(range(0,p))
    res = sl.toeplitz(var_rho**a)
    return res
"""
cov_matrices generates the random toeplitz matrices
Input:
    - K: integer, number of matrices to generate;
    - p: integer, dimension of the covariance;
    - ary_rho: numpy array, dim=K, values in (0,1);
Output:
    - res: numpy array, dim=Kxpxp, the generated covariance matrix
"""
def cov_matrices(K,p,ary_rho):
    res = np.zeros((K,p,p))
    for k in range(0,K):
        res[k,:,:] = cov_matrix(p,ary_rho[k])
    return res
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
pdf_mggd is the density function of the MGGD.
Input:
    -x: numpy array, dim=p, sample;
    -mu: numpy array, dim=p, location parameter;
    -Sigma: numpy array, dim=pxp, scatter matrix;
    -beta: positive float, shape parameter;
Output:
    -res: positive float in (0,1), the probability
"""
def mggdpdf(x,mu,Sigma,beta):
    p = x.shape[0]
    
    # constant of normalizing
    cn = (ss.gamma(p/2) * beta) / ( np.pi**(p/2) * ss.gamma(p/(2*beta)) * 2**(p/(2*beta)) )
    
    delta = max(np.dot( x-mu, npl.solve( Sigma, x-mu ) ), 1e-12)
    
    detSigma = max( npl.det(Sigma), 1e-12 )
    
    res = cn * ( detSigma**(-0.5) ) * np.exp( -(delta**beta)/2 )
    return res
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
mggdell is the log-likelihood function of the MGGD. i.e. log(mggdpdf(x))
Input:
    -x: numpy array, dim=p, sample;
    -mu: numpy array, dim=p, location parameter;
    -Sigma: numpy array, dim=pxp, scatter matrix;
    -beta: positive float, shape parameter;
Output:
    -res: float, the loglikelihood;
"""
def mggdell(x,mu,Sigma,beta):
    p = x.shape[0]
    alpha = np.log(beta) - np.log(p/(2*beta)) - p*np.log(2)/(2*beta)
    delta = np.dot( x-mu, npl.solve( Sigma, x-mu ) )
    log_det = np.log( npl.det( Sigma ) )
    res = alpha - 0.5*log_det - 0.5*delta**beta
    return res
"""
mggdL is the log-likelihood of the complete dataset, i.e. sum(log(mggdpdf(x)))
Input:
    -X: numpy array, dim=Txp, sample;
    -mu: numpy array, dim=p, location parameter;
    -Sigma: numpy array, dim=pxp, scatter matrix;
    -beta: positive float, shape parameter;
Output:
    -res: real scalar, the loglikelihood;
"""
def mggdL(X,mu,Sigma,beta):
    T = np.size(X,axis=0)
    ary_L = np.zeros(T)
    for t in range(0,T):
        x = np.copy( X[t,:] )
        ary_L[t] = mggdell(x, mu, Sigma, beta)
    res = np.sum(ary_L)
    return res
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
mmggdpdf is the density of the mixture of MGGDs.
Input:
    -x: numpy array, dim=p, sample;
    -w: numpy array, dim=K, the weights;
    -mus: numpy array, dim=Kxp, the location parameters;
    -Sigmas: numpy array, dim=Kxpxp, the scatter matrices;
    -betas: numpy array, dim=K, the shape parameters;
Output:
    -res: positive float in (0,1), the probability;
"""
def mmggdpdf(x, w, mus, Sigmas, betas):
    K = np.size(w)
    vec_prob = np.zeros(K)
    for k in range(0,K):
        vec_prob[k] = w[k] * mggdpdf(x, mus[k,:], Sigmas[k,:,:], betas[k])
    res = np.sum(vec_prob)
    return res
"""end function------------------------------------------------------------"""



"""---------------------------------------------------------------------------
mmggdell calculate the log-likelihood of the mixture of MGGDs, i.e. log(mmggdpdf(x))
Input:
    -x: numpy array, dim=p, sample;
    -w: numpy array, dim=K, the weights;
    -mus: numpy array, dim=Kxp, the location parameters;
    -Sigmas: numpy array, dim=Kxpxp, the scatter matrices;
    -betas: numpy array, dim=K, the shape parameters;
Output:
    -res: float, log-likelihood;
"""
def mmggdell(x, w, mus, Sigmas, betas):
    res = np.log( mmggdpdf(x, w, mus, Sigmas, betas) )
    return res
"""---------------------------------------------------------------------------
mmggdL is the loglikelihood of the complete dataset, i.e. sum( log( mggdpdf(x) ) )
Input:
    -X: numpy array, dim=Txp, sample set;
    -w: numpy array, dim=K, the weights;
    -mus: numpy array, dim=Kxp, the location parameters;
    -Sigmas: numpy array, dim=Kxpxp, the scatter matrices;
    -betas: numpy array, dim=K, the shape parameters;
Output:
    - res: scalar, log-likelihood;
"""
def mmggdL(X, w, mus, Sigmas, betas):
    T = X.shape[0]
    ary_L = np.zeros(T)
    for t in range(0,T):
        x = np.copy(X[t,:])
        ary_L[t] = mmggdell(x, w, mus, Sigmas, betas)
    res = np.sum(ary_L)
    return res
"""end function------------------------------------------------------------"""






