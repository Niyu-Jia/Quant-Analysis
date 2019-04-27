#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:00:34 2019

@author: niyu
"""
import os
os.chdir("/home/niyu/Documents/796/project/")
import data
import regression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
from numba import jit
from scipy import stats


@jit
def VaR(w,portfolio,factor_list,N):
    """
    w: column vector,weights of portfolio
    coef: an (k+1)xn array of coefficients of n stocks against the k factors,including intercepts
    factor_list: list of factors which will be used for simulation; input eg:[VIX,EIRX,VOL]
    N: trials of simulation
    """
    
    #normalize your weights for sum of 1
    w= [float(i)/sum(w) for i in w]
    all_factor=data.factor(factor_list)
    regressor=regression.regression(portfolio,N)
    
    
    #assign the coef and simulated residuals
    coef=regressor.coef
    matrix_residuals=regressor.simulated_residual.T
    
    #simulate multivariate normal distribution of these factors
    #the multivariate_normal should be a 2_D array of kxN, k is the number of factors
    multivariate_normal=(all_factor.simulation(N)).T
    #print(multivariate_normal)
    
    #portfolio return= intercept+ beta*factor+error
    #assumption: iid error,will be simulated from iid normal
    #directly generate N trials in the matrix
    matrix_intercept=np.array([coef[:,0]]*N).T
    matrix_beta=coef[:,1:]
    
    stock_return=matrix_intercept+np.dot(matrix_beta,multivariate_normal)+matrix_residuals
    
    matrix_w=np.array([w]*N).T
    portfolio_return=np.multiply(matrix_w,stock_return)
    
    #sum of every stock return to get portfolio return simulation
    sum_return=np.sum(portfolio_return,axis=0)
    sort_return=np.sort(sum_return)
    #print(sort_return)
    
    #plt.plot(np.arange(0,N,1),sum_return)
    #cutoff point:
    cutoff=int(N*0.05)
    #use cutoff point's return as the VaR
    var=sort_return[cutoff]
    #print("VaR:",abs(var))
    return abs(var)


def target(w,portfolio,factor_list,N,expected_return,a):
    """
    w: column vector,weights of portfolio
    portfolio: portfolio object
    factor_list: list of factors which will be used for simulation; input eg:[VIX,EIRX,VOL]
    N: trials of simulation
    expected_return: one-column array of expected return of portfolio stocks' return,fields: stock Tickder & expected return
    a: penalty coefficient
    """
    x=portfolio.choose_x
    y=portfolio.choose_y
    expected_return=expected_return[portfolio.Num_list]
    
    #portfolio expected return
    w=np.array( [float(i)/sum(w) for i in w])
    rp=np.dot(w,expected_return)
    var=VaR(w,portfolio,factor_list,N)
    #print("portfolio expected return: ",rp,"  VaR:",var)
    return var-a*rp

