# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:55:17 2019

@author: autol
"""
#from operator import itemgetter
#from itertools import cycle
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale,StandardScaler
#from scipy import sparse,stats
#from decorators import *
#np.set_printoptions(precision=5)
#pd.set_option('display.max_rows', -1)

#%% scale data
class ScaleX:

    def __init__(self,ismean=1,isstd=1):
        self.ismean,self.isstd = ismean,isstd
        self.A = 0
        self.sc = StandardScaler(with_mean=ismean,with_std=isstd)

    def scalef(self,A):
        self.A=A
#        rd = dict(r1=scale(self.A,axis=0,with_mean=self.ismean,with_std=self.isstd),
#             r2=(self.A-np.nanmean(self.A, axis=0))/np.nanstd(self.A, axis=0),
#             r3=self.sc.fit_transform(self.A))
        return self.sc.fit_transform(self.A)

    def inverse_scalef(self,B):
#        return B*np.nanstd(A, axis=0) + np.nanmean(A, axis=0)
        return self.sc.inverse_transform(B)

#ss = ScaleX()
#B = ss.scalef(A);B
#ss.inverse_scalef(B)
##%%
#
#ss = ScaleX()
#X = ss.scalef(X1.copy());X
#ss.inverse_scalef(X)

#%% split Matrix into train and test

#@accepts(A=np.ndarray)
def Matrix_training(A):
    Afill = ~np.isnan(A)
    print(Afill.sum()/A.size)
    Tfill = Afill.copy()
    aa = np.random.choice([0, 1], size=Afill.sum(), p=[1./4, 3./4])
    print(Afill.sum())
    print(aa.sum())
    Tfill[Afill] = aa
    T = A.copy()
    print(Tfill.sum()/A.size)
    T[~Tfill] = np.nan
    print(T)
    print(A)
    return T,A

#%%
def Metrics(Z,Z0,method='mse'):
    if method == 'mse':
        return np.mean((Z-Z0)**2)
    elif method == 'rmse':
        return np.sqrt(np.mean((Z-Z0)**2))
    elif method == 'mae':
        return np.mean(np.abs(Z-Z0))
    elif method == 'sse':
        return np.sum((Z-Z0)**2)
    elif method == 'euclidean':
        return np.linalg.norm(Z-Z0)
    else:
        return 0


