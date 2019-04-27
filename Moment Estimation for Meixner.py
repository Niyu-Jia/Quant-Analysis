# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:07:53 2019

@author: boshu
"""

from scipy.special import gamma
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import fsolve, root
import scipy.integrate


import os
os.chdir("/home/niyu/Documents")

_df = pd.read_csv('./794/794Project/SPY.csv')
_df.index = _df.iloc[:,0]
df = _df.iloc[:,5:6] 

def log_return():
    l_r = np.log(df/df.shift(1))
    l_r.fillna(0,inplace = True)
    return l_r

LR = log_return()

class Meixner:
    def __init__(self,parameters):
        """a > 0, −π < b < π, d > 0, and m ∈ R.
        """
        self.a=parameters[0]
        self.b=parameters[1]
        self.d=parameters[2]
        self.m=parameters[3]
    
    def pdf(self,x):
        z = (x-self.m)/self.a
        density = (2*np.cos(self.b/2))**(2*self.d)/(2*self.a*np.pi*gamma(2*self.d))*np.exp(self.b*z)*abs(gamma(self.d+z*1j))**2
        return density
    
    def mean(self):
        return self.m+self.a*self.d*np.tan(self.b/2)
    
    def var(self):
        return self.a**2*self.d/(np.cos(self.b)+1)
    
    def skew(self):
        return np.sin(self.b)*(self.d*(np.cos(self.b)+1))*-0.5
    
    def kurt(self):
        return 3+(2-np.cos(self.b))/self.d

def MomentEstimator():
#    by this data set, the result is:[ 0.01269828  0.58355268  0.83624537 -0.00274537]
    mean = LR.mean()
    var = LR.var()
    skew = LR.skew()
    kurt = LR.kurt()
    
    def fun(x):
        ME = Meixner(x)
        return np.array([float(ME.mean()-mean), float(ME.var()-var), float(ME.skew()-skew), float(ME.kurt()-kurt)])
    
    x0 = np.array([0.01,0,1,0])
    result = fsolve(fun,x0)
   
    return result

def plot():
    sns.set_style('darkgrid')
    sns.set_context('paper')
    sns.distplot(LR,hist = False,label = 'PDF', axlabel = 'daily returns')#,hist=False)

    sample = Meixner(MomentEstimator())
#    
    def PdfPlot():
        x = np.linspace(-0.05,0.05,10000)
        plt.plot(x,sample.pdf(x))
#    sns.set
    PdfPlot()


""" test"""
if __name__ == '__main__':
#    fun()
#    print(df.head())
#    print(RS.head(10))
#    returns()
    
#    print(MomentEstimator())
    plot()
#    y = lambda x: Meixner(np.array([0.012982825,0.12716244,0.57295483,-0.00112426])).pdf(x)
#    print(scipy.integrate.quad(y,-1,100))
#    sample = Meixner(MomentEstimator())
#    sns.pointplot(sample)
#    print(sample.mean())
#    print(sample.pdf())
#    x= np.array([0.0001*i for i in range(-1000,1000)])
#    plt.plot(x,sample.pdf(x))
#    print(fun(np.array([0.01,0,1,0])).shape)
#    df.plot()
#    df.hist(bins = 100)
#    df.subplot()
    
#    plt.plot(df)
    
#    def func(paramlist):
#        x,y=paramlist[0],paramlist[1]
#        return [ x**2+2*y-5,
#                x+y-1 ]
#    s=fsolve(func,[0,0])
#    print(s)