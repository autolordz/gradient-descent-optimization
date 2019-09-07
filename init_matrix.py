# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:09:39 2019

@author: autol
"""
#%%
from scipy import sparse,stats
import numpy as np

#%% Matrix 1
def init_A1(n=5,m=3):
    n,m=int(n),int(m)
    A = sparse.random(n,m, density=.8,
                      data_rvs=stats.randint(1,6).rvs).toarray()
    return A

#%% Matrix 2
def init_A2(n=18,m=15,d=.4,rstat=1212): #密度0~1,1是全部数存在
    np.random.seed(rstat)
    n,m=int(n),int(m)
    d_nan = round((1-d)*n*m)
    A = np.random.randint(1,5,(n,m)).astype(np.float)
    A.ravel()[np.random.choice(A.size,d_nan,replace=0)] = np.nan
    return A

#%% Matrix 3
# [soft impute referense](https://cran.r-project.org/web/packages/softImpute/vignettes/softImpute.html)

def init_A3():
    A = np.array([[0.8654889,0.01565179,0.1747903,np.nan, np.nan],
                    [-0.6004172,np.nan,-0.2119090,np.nan,np.nan],
                    [-0.7169292,np.nan, np.nan,0.06437356,-0.09754133],
                    [0.6965558,-0.50331812,0.5584839 ,1.54375663 ,np.nan],
                    [1.2311610,-0.34232368,-0.8102688 ,-0.82006429 ,-0.13256942],
                    [0.2664415,0.14486388,np.nan,np.nan, -2.24087863]
                    ])
    return A

def init_A4(n=5,m=3):
    n,m=int(n),int(m)
    A = np.random.randint(1,100,n*m).reshape(n,m).astype(np.float)
    return A