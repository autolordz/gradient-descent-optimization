# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:13:19 2019

@author: autol
"""
import numpy as np

def Fill(A,fill='zero'): # impute: "zero", "mean", "median", "min", "random"
        Anas = np.isnan(A)
        X = A.copy()
        if fill == 'zero':
            return np.nan_to_num(X)
        elif fill == 'mean':
            col_fill = np.nanmean(X, axis=0)
        elif fill == 'median':
            col_fill = np.nanmedian(X, axis=0)
        elif fill == 'min':
            col_fill = np.nanmin(X, axis=0)
        elif fill == 'random':
            B = np.full(X.shape,np.nan)
            B[Anas] = np.random.randn(Anas.sum())
            B = B * np.nanstd(X, axis=0) +np.nanmean(X, axis=0)
            X[Anas] = B[Anas]
            return X
        np.copyto(X, col_fill, where=np.isnan(X))
        return X

def Prox(X,λ):
    return np.maximum(X-λ,0) # np.sign(X)

def Frob1(U0,d0,V0,U,d,V): # from github
    denom = (d0 ** 2).sum()
    utu = d * (U.T @ U0)
    vtv = d0 * (V0.T @ V)
    uvprod = (utu @ vtv).diagonal().sum()
    num = denom + (d**2).sum() - 2*uvprod
    return num/max(denom, 1e-9)

def Frob2(Z,Z1): # from textbook
    a = np.trace((Z-Z1).T@(Z-Z1)) # (E**2).sum()
    b = np.trace(Z.T@Z)
    return a/max(b, 1e-9)

def obj1(Z,Z1,xnas,nz): # origin
    E = (Z-Z1)[~xnas]
    return 1./2*(E**2).sum()/nz

def obj2(Z,Z1,xnas,nz,d,λ): # penalty
    E = (Z-Z1)[~xnas]
    return (1./2*(E**2).sum()+λ*d.sum())/nz # d.sum() = np.trace(np.diag(d))

def svd_(Z):
    return np.linalg.svd(Z,full_matrices=0)

def svdk(Z,k):
    U,d,Vt = np.linalg.svd(Z,full_matrices=0)
    U,d,Vt = U[:,:k],d[:k],Vt[:k,:]
    return U,d,Vt

#def Frob1(U0,D0,V0,U,D,V): # from github
#    denom = (D0 ** 2).sum()
#    utu = D * (U.T.dot(U0))
#    vtv = D0 * (V0.T.dot(V))
#    uvprod = utu.dot(vtv).diagonal().sum()
#    num = denom + (D**2).sum() - 2*uvprod
#    return num/max(denom, 1e-9)
