# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:39:56 2019

@author: autol
"""


#%%
import numpy as np
import time
from gfun import StepClass,ConvClass,JClass,Hessian
from gupdate import UpdateClass

#%% methodtion
#@accepts(w=np.ndarray)
def gradient_descent_f(var,
                        X=0,y=0,w=0,n_iters=1,n_b=10,
                        sgd=0,method='mm10',isStep=0,
                        trace=1,doplot=1,ŋ=0,ŋ_a=1,skipConv=1,
                        **kwargs):
    records = []
    # Shuffle X,y
#    r_index = np.random.RandomState(seed=43).permutation(len(y))
#    X1 = X[r_index,:]
#    w = var.w
#    y1 = y[r_index]
    time1 = time.time()

    He = Hessian(var)
    var.set(dict(A=He.A_(),H=He.H_()))

    Jc = JClass(var,method)
    var.set(dict(gJ=Jc.gJ,J=Jc.Loss,e0=Jc.Loss(w)))

    var.set(dict(θ=w.copy(),
                m=np.zeros(len(w)),v=np.zeros(len(w)),
                t=1,))

    Uc = UpdateClass(var)
    Cc = ConvClass(var)
    Sc = StepClass(var)
    if isStep : #and not method in ['mm52','mm26']
        ŋ = Sc.armijo_i(w,ŋ_a)

    e1 = var.J(w)
    ratio = 0
    n_w,n_y=len(w),len(y)
    records.append([-1,w.copy(),e1,ratio])
    for i in range(n_iters):
        if sgd == 0:
            #if isStep : #and not method in ['mm52','mm26']
            #    ŋ = Sc.armijo_i(w,ŋ_a)
            w = Uc.update_w(w,ŋ=ŋ,i=i)
#            w += -ŋ*2./len(y)*X.T.dot(X.dot(w)-y)
            e1 = var.J(w)
#            e1 = np.mean((X.dot(w)-y)**2)
            isConv,ratio = Cc.Conv(w,e1,ŋ,skipConv)
        elif sgd == 1:
            bb = range(0,n_y,n_b)
            ws = np.zeros(n_w)
            e1s = 0
            for k in bb:
                X_b = X[k:k + n_b]
                y_b = y[k:k + n_b]
#                print('each batch:',len(y_b))
                if len(y_b) ==0:break # 没数据就退出
                w = Uc.update_w(w,ŋ=ŋ,i=i,X=X_b,y=y_b)
                e1s += var.J(w)
                ws += w
            e1 = e1s/len(bb)
            w = ws/len(bb)
            isConv,ratio = Cc.Conv(w,e1,ŋ,skipConv)
        else:
            print('None...');return None
        records.append([i,w.copy(),e1,ratio])
        ret = dict(ik=i,w=w,e1=e1,ratio=ratio)

#        print(ret)
        if isConv>0:break
#        if trace:pass
    print('last: \n',ret)
    if not doplot: print('There\'s no method:',method)
    time2 = time.time()
    print('All Running time: %s Seconds'%(time2-time1))
    rets = dict(wh=np.stack(records),finals=ret,method=method)
    return rets

#%%





