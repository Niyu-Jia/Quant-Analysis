#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:51:16 2019

@author: niyu
"""

"""
In this file we firstly are doing some PCA risk analysis with some stocks data;then we try to construct an optimal portfolio with some chosen risk aversion
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas_datareader.data as web
import datetime
from math import exp,log
from sklearn.decomposition import PCA
import itertools
######################### Preperation for data #################################
codelist=pd.read_csv('/home/codelist.csv')
codelist=codelist
codelist=codelist['Ticker'].tolist()

#interval of data set
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2017, 12, 31)

s=pd.DataFrame()

# downloading data
n=0
while len(s.columns)<100:
    code=codelist[n]
    n=n+1
    try:
        adj_close= np.array(web.DataReader(code, 'yahoo', start, end)['Adj Close'])
        #print(adj_close)
        s[code]=adj_close
        print(len(s.columns),"%")
	
    except:#skip some errors
        print("pass the code ",n)
        pass

s.to_csv('/home/price.csv')


#reload data
s=pd.read_csv('/home/price.csv',index_col=0)
nullcell=s.isnull().sum().sum()#check full matrix 
s_lag=s.shift(1)
R=s/s_lag
r=np.log(R)[1:]


cov=np.cov(r.T)
w, v =np.linalg.eig(cov)
sum(w<0) #total negative number
#w=sorted(w,reverse=True)

accu=list(itertools.accumulate(w))
percentage=accu/sum(w)

w_I=[1/w[i] if percentage[i]<0.9 else 0 for i in range(100)]
eigen_I=np.diag(w_I)
v=np.matrix(v)
CI=np.dot(np.dot(v,eigen_I),v.I)

#PCA 

pca=PCA(svd_solver='full')
pca.fit(cov)
pca.explained_variance_ratio_
pca.singular_values_



c=np.array([1,0.1])
g1=np.array([1 for i in range(100)])
g2=np.array([0 for i in range(100)])
g2[:17]=1

#convariance matrix
C=np.matrix(cov)
G=np.matrix([g1,g2])
mat=np.dot(np.dot(G,C.I),G.T)
mat_inverse=mat.I

R=np.matrix(r.mean(0)).T
# choose your risk aversion coefficient
a=1



right=np.dot(np.dot(G,C.I),R)-np.matrix(2*a*c).T
lamb=np.dot(mat_inverse,right)

weight=(1/(2*a))*(np.dot((R.T-np.dot(lamb.T,G)),C.I))
weight.sum()


##################### use robust C inverse ######################

c=np.array([1,0.1])
g1=np.array([1 for i in range(100)])
g2=np.array([0 for i in range(100)])
g2[:17]=1

#C=np.matrix(cov)
G=np.matrix([g1,g2])
mat=np.dot(np.dot(G,CI),G.T)
mat_inverse=mat.I

R=np.matrix(r.mean(0)).T
# choose your risk aversion coefficient
a=1

right=np.dot(np.dot(G,CI),R)-np.matrix(2*a*c).T
lamb=np.dot(mat_inverse,right)

#deciding the weights
weight=(1/(2*a))*(np.dot((R.T-np.dot(lamb.T,G)),CI))
weight.sum()

