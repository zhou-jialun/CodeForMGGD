# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:03:12 2021

@author: jzhou
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg as sl



""" square information distance for mu """
def sidist_mu( hat_mu, true_mu, Imu ):
    res = Imu*np.dot(hat_mu-true_mu, hat_mu-true_mu)
    return res
""" end function """



""" square information distance """
def sidist_Sigma( hat_Sigma, true_Sigma, I1, I2 ):
    logSigma = sl.logm(npl.solve(true_Sigma,hat_Sigma))
    tr_prl = np.trace( np.dot(logSigma,logSigma) )
    tr_oth = np.power( np.trace(logSigma), 2 )
    res = I1*tr_prl + I2*tr_oth
    return res
""" end function """



""" square information distance """
def sidist_beta( hat_beta, true_beta, Ibeta ):
    res = Ibeta*np.power(np.log(hat_beta/true_beta), 2)
    return res





















