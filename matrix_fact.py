# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:30:41 2019

@author: autol
"""

#%%
from depends import ScaleX
from matrix_fun import Fill,Frob2,obj1,obj2,svdk,svd_,Prox,Frob1
import numpy as np
import time
from init_matrix import init_A1,init_A2,init_A3,init_A4
from sklearn.model_selection import ParameterGrid
from plotxy import plot_gd_xy,iters_matrix_plot

#%% SoftImpute with svda, svds, als implement calculate details inside the functions

def SoftImpute(X=0,k=2,λ=0,
               method='svda',trace=0, isScale=0,
                n_iters=1, final_svd=1, svdw=0,paras='',
                **kwargs):
    time1 = time.time()
    isConv=0
    n,m = X.shape ; xnas = np.isnan(X) ; nz = n*m-xnas.sum()
    if isScale: sc = ScaleX(ismean=1,isstd=1) ;  X = sc.scalef(X);X
    Z = Fill(X.copy(),fill='zero');Z  # Fill fun for fill matrx with 0
    ratio = 0 ; dh=[]
    Z0 = Z.copy() ;  Z1 = 0 ; ratio = 0
    if 'svda' == method:
        print('SVD approximate',kwargs)
        U,d,Vt = np.linalg.svd(Z,full_matrices=0) # full_matrices 只考虑在是否方阵，是则大小不变，否则取min(X.shape)维度
        for i in range(min(k,min(Z.shape))):
            Z1 += d[i] * U[:,i][:,np.newaxis] @ Vt[i][np.newaxis]
            Z[xnas] = Z1[xnas]
            obj = obj1(Z,Z1,xnas,nz)
            ratio = Frob2(Z0,Z1)
            Z0 = Z1.copy()
            dicts = dict(ik=i,obj='%.3e'%obj,ratio='%.3e'%ratio) ; print(dicts)

            dh.append([i,obj,ratio])
            if i>len(d)-2:break
    elif 'svds' == method:
        print('Soft Impute SVD',kwargs)
        svdZ = svdk(Z,k)
        for i in range(n_iters):
            svdZ0 = svdZ
            d = Prox(svdZ[1],λ)
            Z1 = svdZ[0] @ np.diag(d) @ svdZ[2]
            Z[xnas] = Z1[xnas]
            svdZ = svdk(Z,k)
            d1 = Prox(svdZ[1],λ)
            obj = obj2(Z,Z1,xnas,nz,d1,λ)
            ratio = Frob1(svdZ0[0],d,svdZ0[2].T, svdZ[0],d1,svdZ[2].T)
            dicts = dict(ik=i,obj='%.3e'%obj,ratio='%.3e'%ratio) ; print(dicts)
            dh.append([i,obj,ratio])
    elif 'als' == method: # 算法参考[softImpute_als](https://github.com/cran/softImpute/blob/master/R/simpute.als.R)
        print('Soft Impute ALS',kwargs)
        if svdw: # warm start 有初始值
            Z = X.copy() ; J = k  #must have u,d and v components #    J = min(sum(svdw[1]>0)+1,k)
            d = svdw[1] ; JD = sum(d>0)
            print('JD=',JD,'J=',J)
            if JD >= J:
                U = svdw[0][:,:J] ; V = (svdw[2].T)[:,:J] ;  Dsq = d[:J][:,np.newaxis]
            else:
                fill = np.repeat(D[JD-1],J-JD) # impute zeros with last value of D matrix
                Dsq = np.append(D,fill)[:,np.newaxis]
                Ja = J-JD ; U = svdw[0]
                Ua = np.random.normal(size=n*Ja).reshape(n,Ja) # 截断大小
                Ua = Ua - U @ U.T @ Ua
                Ua = svd_(Ua)[0]
                U = np.column_stack((U,Ua))
                V = np.column_stack((svdw[2].T,np.repeat(0,m*Ja).reshape(m,Ja)))
            Z1 = U @ (Dsq*V.T)
            Z[xnas]=Z1[xnas]
#            print('Z=',Z.shape,'Z1=',Z1.shape)
        else: # cool start 冷启动没初始值
#            k = min(sum(svd(Z)[1]>0)+1,k)
            V = np.zeros((m,k))
            U = np.random.normal(size=n*k).reshape(n,k)
            U = svd_(U)[0]
            Dsq = np.repeat(1,k)[:,np.newaxis] # Dsq = D_square = d^2 # we call it Dsq because A=UD and B=VD and AB=U Dsq Vt
            print('Z=',Z.shape,'u=',U.shape,'dsq=',Dsq.shape,'vt=',V.T.shape)
        for i in range(n_iters):
            U0,Dsq0,V0=U,Dsq,V
            # U step
            B = U.T @ Z
            if λ>0:  B = B*(Dsq/(Dsq+λ))
            Bsvd = svd_(B.T)
            V = Bsvd[0] ; Dsq = Bsvd[1][:,np.newaxis] ; U = U @ Bsvd[2].T
            Z1 = U @ (Dsq*V.T)
            Z[xnas] = Z1[xnas]
            obj = obj2(Z,Z1,xnas,nz,Dsq,λ)
            # V step
            A = (Z @ V).T
            if λ>0:  A = A*(Dsq/(Dsq+λ))
            Asvd = svd_(A.T)
            U = Asvd[0] ; Dsq = Asvd[1][:,np.newaxis] ; V = V @ Asvd[2]
            Z1 = U @ (Dsq*V.T)
            Z[xnas] = Z1[xnas]
            # End U V steps
            ratio = Frob1(U0,Dsq0,V0,U,Dsq,V)
#            if ratio>1e-05: break
            dicts = dict(ik=i,obj='%.3e'%obj,ratio='%.3e'%ratio) ; print(dicts)
            dh.append([i,obj,ratio])

    if isScale: Z = sc.inverse_scalef(Z);Z
    time2 = time.time()
    print('All Running time: %s Seconds'%(time2-time1))
    return dict(dh=np.stack(dh),finals=dicts,method=method,Z=Z)

#%%

pgrid = dict(
            method=['svds','als'],
#            λ=[0,.5,1],
#            method=['svda','svds'], #  'svds','als'
#            isScale = [0],
             k=[2,5,7],
#             den_X = [.4,.6],  #np.random.randint(1,100,3) *1/100,
             n_X = [80,100], # np.random.randint(2,100,3),
             )
pgrid = list(ParameterGrid(pgrid));pgrid

rets=[]
for pg in pgrid:
    pg1 = pg.copy()
    n_X = pg.get('n_X',5) ; den_X = pg.get('den_X',.5)
    X = init_A2(n_X,n_X,den_X,rstat=1212);X
#    a = pg1.pop('isScale')
#    X = init_A3();X
    ret = SoftImpute(X,#λ=0,
#                     isScale = a,
                     n_iters = 13,
                     **pg1)
    Z = ret['Z']
    if np.isnan(Z).any():
        print('Z with nan');continue
    rets.append(ret)

iters_matrix_plot(rets,pgrid)


#%%
np.set_printoptions(precision=2,suppress=1)
X = init_A2(10,10,.3,546);print(X)
ret = SoftImpute(X,method='svds',n_iters=25,λ=1)
Z= ret['Z'];Z
plot_gd_xy(ret)

#%%
X = init_A1(4,3);print(X)

#%%
X = init_A2(6,6,.3,546);print(X)

X[np.isnan(X)] = 0

sc = ScaleX(ismean=1,isstd=1)
X = sc.scalef(X);X
#%%
X = init_A2(15,15,.6);X
rets = SoftImpute(X,k=13,n_iters=10)

#%%

#        if i==n_iters:
#            print("Convergence not achieved by",n_iters,"iterations")
#        if λ>0 and final_svd:
#            U = Z @ V
#            sU = svd(U)
#            U = sU[0] ; Dsq = sU[1] ;V = V @ sU[2].T
#            Dsq = Sλ(Dsq,λ)
#            if trace:
#                Z1 = U @ (Dsq*V.T)
#                print("final SVD:", "obj=",obj_(Z,Z1,Dsq).round(5),"\n")

