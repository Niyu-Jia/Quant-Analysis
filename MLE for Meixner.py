#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:15:09 2019

@author: niyu
"""
import math
import cmath
from scipy.special import gamma
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt

def data(path):
    """
    processing data and get observation array
    """
    
    raw=pd.read_csv(path)["Adj Close"]
    raw_shift=raw.shift(1)
    
    #log return
    r=[math.log(raw[i]/raw_shift[i]) for i in range(1,len(raw))]
    
    return np.array(r)

def Meixner(para,x):
    """
    para: an array of Meixner parameters
    will return Meixner density for given point x
    """
    
    #a,b,m,d=para[0],para[1],para[2],para[3]
    a=para[0]
    b=para[1]
    m=para[2]
    d=para[3]
    
    term1=(2*cmath.cos(0.5*b))**(2*d)/(2*a*cmath.pi*gamma(2*d))
    term2=cmath.exp((b*(x-m))/a)
    term3=gamma((d+1j*(x-m))/a)
    term4=(abs(term3))**2
    
    pdf=term1*term2*term4
    
    #print(pdf)
    
    return pdf

def LogLikelihood(para,x_array):
    """
    para: an array of Meixner parameters
    x_array: your observation array
    """
    
    n=len(x_array)#number of your observation
    meixner=[Meixner(para,x_array[i]) for i in range(n)]
    
    #log likelihood function
    #LL=cmath.log(np.multiply.reduce(meixner)) #this may cause overflow
    single_LL=[cmath.log(i) for i in meixner if i.real>0]
    LL=sum(single_LL)
    #LL=np.multiply.reduce(meixner)
    return -LL.real

def MLE(x_array):
    """
    Maximizing the loglikelihood and find the best estimation of parameters in Meixner
    """
    #bound of parameters, reference to Meixner distribution
    bound=((0.01, None), (-cmath.pi, cmath.pi),(None,None),(0.01,None))
    
    parameters=minimize(LogLikelihood,x0=[0.1,0.1,0.1,0.1],args=(x_array),bounds=bound)
    
    return parameters


def density(mle_para,x_array):
    density_x=[Meixner(mle_para,i) for i in x_array]
    x=np.arange(0.01,len(density_x),1)
    meixner=[Meixner(mle_para,i) for i in x]
    
    plt.plot(x,meixner)
    #plt.plot(x,)
    
    return None
    
    
    
if __name__=="__main__":
    path="/home/niyu/Documents/794/794Project/SPY.csv"
    x_array=data(path)
    #plt.hist(x_array,bins=50)
    mle_para=MLE(x_array).x
    #density(mel_para,x_array)
    
