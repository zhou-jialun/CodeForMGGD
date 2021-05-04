# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:42:14 2021

@author: jzhou
"""

import numpy as np
import numpy.linalg as npl
import copy

from generation_of_MGGD import mggdL
from Gradients import BG_mu, BG_Sigma, BG_beta,\
    calImu, calJSigma, calIbeta, iG_mu, iG_Sigma, iG_beta
from Retractions import ExpSPD, RetPRN



"""---------------------------------------------------------------------------
Component-wise Information Gradient method with "backtracking" selected step-
size. 
Input:
    - X: numpy array, dim=Txp, dataset;
    - cas: string, the case of parameter
        - "S" means that (Sigma) is an unknown parameter to be estimated,
            (mu,beta) are regarded as known and should be defined in the "kps"
            tuple.
        - "mS" means that (mu,Sigma) are unknown parameters to be estimated,
            (beta) is regarded as known and should be defined in the "kps"
            tuple..
        - "mSb" means that (mu,Sigma,beta) are unknown parameters to be 
            estimated, the tuple "kps" should be empty.
    - ips: tuple, the initial value of unknown parameters should be put in
        this tuple. This tuple should be defined as follows.
        - for case "S", ips=(ini_Sigma), where ini_Sigma is a numpy array, dim=pxp;
        - for case "mS", ips=(ini_mu,ini_Sigma), where ini_mu is a numpy array,
            dim=p, ini_Sigma is the same as above.
        - for case "mSb", ips=(ini_mu,ini_Sigma,ini_beta), where ini_beta is 
            a positive float, ini_mu and ini_Sigma are the same as above.
    - kps: tuple, the known parameters shoule be put in this tuple. The
        elements inside of this tuple are the same as initial parameters.                
        - for case "S", kps=(knw_mu,knw_beta)
        - for case "mS", kps=(knw_beta)
        - for case "mSb", kps=None
    - max_itr: integer, the maximum number of iterations.
    - fin_err: positive float, the final error to stop iterations.
    - ra: return array, if ra=True, function will return all the list of
        estimators, else ra=False, function will return the final estimator;
Output:
    - 
"""
def mggd_cigbs(X, cas, ips, kps, max_itr, fin_err=1e-6, ra=False ):
    p = X.shape[1]
    crt_err = 1000.0
    crt_itr=0
    
    if cas=="S":
        knw_mu = np.copy( kps[0] )
        knw_beta = copy.deepcopy( kps[1] )
        ini_Sigma = np.copy( ips[0] )
        
        list_Sigma = []
        list_Sigma.append( ini_Sigma )
        while crt_err>fin_err or crt_itr<max_itr:
            crt_Sigma = np.copy( list_Sigma[crt_itr] )
            GSigma = BG_Sigma( X, knw_mu, crt_Sigma, knw_beta )
            (J1, J2) = calJSigma( knw_beta, p )
            iGSigma = iG_Sigma( GSigma, crt_Sigma, J1, J2 )
            ss_Sigma = bkt_Sigma(iGSigma, knw_mu, crt_Sigma, knw_beta, X)
            new_Sigma = ExpSPD( crt_Sigma, ss_Sigma*iGSigma )
            list_Sigma.append( new_Sigma )
            crt_itr = crt_itr+1
            crt_err = npl.norm( crt_Sigma-new_Sigma, 'fro' )
        if ra==True:
            return list_Sigma
        else: # ra==False
            return new_Sigma

    if cas=="mS":
        knw_beta = copy.deepcopy( kps[0] )
        ini_mu = np.copy( ips[0] )
        ini_Sigma = np.copy( ips[1] )
        
        list_mu = []
        list_mu.append( ini_mu )
        list_Sigma = []
        list_Sigma.append( ini_Sigma )
        while crt_err>fin_err or crt_itr<max_itr:
            # reload current estimators:
            crt_mu = np.copy( list_mu[crt_itr] )
            crt_Sigma = np.copy( list_Sigma[crt_itr] )
            # update mu:
            Gmu = BG_mu( X, crt_mu, crt_Sigma, knw_beta )
            Imu = calImu( knw_beta, p )
            iGmu = iG_mu( Gmu, crt_Sigma, Imu )
            ss_mu = bkt_mu(iGmu, crt_mu, crt_Sigma, knw_beta, X)
            new_mu = crt_mu + ss_mu*iGmu
            list_mu.append( new_mu )
            # update Sigma:
            GSigma = BG_Sigma( X, new_mu, crt_Sigma, knw_beta )
            (J1, J2) = calJSigma( knw_beta, p )
            iGSigma = iG_Sigma( GSigma, crt_Sigma, J1, J2 )
            ss_Sigma = bkt_Sigma(iGSigma, new_mu, crt_Sigma, knw_beta, X)
            new_Sigma = ExpSPD( crt_Sigma, ss_Sigma*iGSigma )
            list_Sigma.append( new_Sigma )
            # update hyper parameters:
            crt_itr = crt_itr+1
            crt_err = npl.norm( crt_mu-new_mu ) + npl.norm( crt_Sigma-new_Sigma, 'fro' )
        if ra==True:
            return (list_mu, list_Sigma)
        else: # ra==False
            return (new_mu, new_Sigma)
        
    if cas=="mSb":
        ini_mu = np.copy( ips[0] )
        ini_Sigma = np.copy( ips[1] )
        ini_beta = copy.deepcopy( ips[2] )
        
        list_mu = []
        list_mu.append( ini_mu )
        list_Sigma = []
        list_Sigma.append( ini_Sigma )
        list_beta = []
        list_beta.append( ini_beta )
        while crt_err>fin_err and crt_itr<max_itr:
            print( "current iteration: ", crt_itr )
            # reload current estimators:
            crt_mu = np.copy( list_mu[crt_itr] )
            crt_Sigma = np.copy( list_Sigma[crt_itr] )
            crt_beta = copy.deepcopy( list_beta[crt_itr] )
            # update mu:
            Gmu = BG_mu( X, crt_mu, crt_Sigma, crt_beta )
            Imu = calImu( crt_beta, p )
            iGmu = iG_mu( Gmu, crt_Sigma, Imu )
            ss_mu = bkt_mu(iGmu, crt_mu, crt_Sigma, crt_beta, X)
            new_mu = crt_mu + ss_mu*iGmu
            list_mu.append( new_mu )
            print(" mu updated ")
            # update Sigma:
            GSigma = BG_Sigma( X, new_mu, crt_Sigma, crt_beta )
            (J1, J2) = calJSigma( crt_beta, p )
            iGSigma = iG_Sigma( GSigma, crt_Sigma, J1, J2 )
            ss_Sigma = bkt_Sigma(iGSigma, new_mu, crt_Sigma, crt_beta, X)
            new_Sigma = ExpSPD( crt_Sigma, ss_Sigma*iGSigma )
            list_Sigma.append( new_Sigma )
            print(" Sigma updated ")
            # update beta:
            Gbeta = BG_beta( X, new_mu, new_Sigma, crt_beta )
            Ibeta = calIbeta( crt_beta, p )
            iGbeta = iG_beta( Gbeta, Ibeta )
            ss_beta = bkt_beta(iGbeta, new_mu, new_Sigma, crt_beta, X)
            new_beta = RetPRN( crt_beta, ss_beta*iGbeta )
            list_beta.append( new_beta )
            print(" beta updated ")
            # update hyper parameters:
            crt_itr = crt_itr+1
            crt_err = npl.norm( crt_mu-new_mu ) + npl.norm( crt_Sigma-new_Sigma, 'fro' ) + np.abs( crt_beta-new_beta )
        if ra==True:
            return (list_mu, list_Sigma, list_beta)
        else: # ra==False
            return (new_mu, new_Sigma, new_beta)


    
    
        
    
def bkt_Sigma( GSigma, mu, Sigma, beta, X ):
    T = X.shape[0]
    a = 0.25
    c = 0.5
    ss = 1.0
    
    kld_1 = mggdL( X, mu, Sigma, beta )/T
    kld_2 = mggdL( X, mu, ExpSPD(Sigma,ss*GSigma), beta )/T;
    sn_GSigma = np.trace( np.dot( npl.solve(Sigma,GSigma), npl.solve(Sigma,GSigma) ) )
    while kld_2 < kld_1 - a*ss*sn_GSigma:
        ss = c*ss;
        kld_2 = mggdL( X, mu, ExpSPD(Sigma,ss*GSigma), beta )/T;
    return ss

def bkt_mu( Gmu, mu, Sigma, beta, X ):
    T = X.shape[0]
    a = 0.25
    c = 0.5
    ss = 1.0
    
    kld_1 = mggdL( X, mu, Sigma, beta )/T
    kld_2 = mggdL( X, mu+ss*Gmu, Sigma, beta )/T;
    sn_Gmu = np.dot( Gmu, Gmu )
    while kld_2 < kld_1 - a*ss*sn_Gmu:
        ss = c*ss;
        kld_2 = mggdL( X, mu+ss*Gmu, Sigma, beta )/T;
    return ss

def bkt_beta( Gbeta, mu, Sigma, beta, X ):
    T = X.shape[0]
    a = 0.25
    c = 0.5
    ss = 1.0
    
    kld_1 = mggdL( X, mu, Sigma, beta )/T
    kld_2 = mggdL( X, mu, Sigma, RetPRN(beta, ss*Gbeta) )/T;
    sn_Gbeta = Gbeta*Gbeta
    while kld_2 < kld_1 - a*ss*sn_Gbeta:
        ss = c*ss;
        if ss<1e-12:
            break
        kld_2 = mggdL( X, mu, Sigma, RetPRN(beta, ss*Gbeta) );
    return ss




# mu = np.array([0,0,0])
# Sigma = np.array([[1,0,0],[0,1,0],[0,0,1]])
# beta = 0.5

# kps = ( mu, Sigma, beta )




    # - som: Size Of Mini-batch, int.
    # - epoch: The number of times that the complete dataset is traversed, int.
























