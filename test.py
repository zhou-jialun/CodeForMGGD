# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:32:20 2021

@author: jzhou
"""
import numpy as np

from generation_of_MGGD import mggdrand
from mggd_cig import mggd_cigbs
from Distances import sidist_mu, sidist_Sigma
from Gradients import calImu, calISigma, calIbeta

# define dimension:
p = 3
# define size of dataset:
T = 10000
# define true parameters:
true_mu = np.array([0,1,2])
true_Sigma = np.array( [ [3,2,1],[2,4,0],[1,0,5] ] )
true_beta = 0.85

# generate the samples:
X = mggdrand( T, true_mu, true_Sigma, true_beta )

ini_mu = np.array([0.1,1.5,1.6])
ini_Sigma = np.array([[4,0,0],[0,5,0],[0,0,6]])
ini_beta = 0.8
(list_mu, list_Sigma, list_beta) = mggd_cigbs(X, cas="mSb", ips=(ini_mu,ini_Sigma,ini_beta), kps=(), max_itr=5, fin_err=1e-6, ra=True )

nbitr = len(list_Sigma)
list_err = []
Imu = calImu( true_beta, p )
(I1,I2) = calISigma(true_beta,p)
Ibeta = calIbeta(true_beta, p)
for i in range(0,nbitr):
    list_err.append( sidist_mu(list_mu[i],true_mu,Imu) + sidist_Sigma(list_Sigma[i],true_Sigma,I1,I2) )
err = np.log10( list_err )

import matplotlib.pyplot as plt
x = np.array( range(0,nbitr) )
plt.plot( x, err )

np.save( 'mggd_mSb_test_data.npy', err )





