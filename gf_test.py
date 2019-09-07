# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:15:18 2019

@author: autol
"""

#%%
from plotxy import plot_gd_xy,iters_gd_plot,plot_gd_contour
from initdata import init_data,init_data1,data_b,init_data_house
from func import gradient_descent_f
from varclass import VarSetX
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np

#%% Example
n=20
w = np.ones(2);w
X,y=init_data1(n,45,w,b=0);X # eta = 1e-2
#X,y=init_data_house(n,45,w);X #  1e-7
X_b = data_b(X);X_b
y

#%%

B_b = np.linalg.inv(X_b.T.dot(X_b)) @ (X_b.T.dot(y));B_b
B = np.linalg.inv(X.T.dot(X)) @ (X.T.dot(y));B

#%%
#w = np.array([-2.5,-2.5]);w
#w = np.array([0.,0.]);w

A = 2./len(y)*X.T.dot(X) # ŋ=1 # 海森矩阵
J = lambda w: np.mean((X.dot(w)-y)**2) # 目标函数
gJ = lambda w: 2./len(y)*X.T.dot(X.dot(w)-y) # 梯度函数
#A = X.T@X # ŋ=1/n
#J = lambda w: w.dot(A).dot(w)
#gJ = lambda w: A.dot(w)

pgrid =list(ParameterGrid(dict(sgd=[0,1],
                               isStep=[0],
#                              ρ=[.5,5,10],
#                               n_b=[2,5],
#                               ŋ_a=[1], # ŋ_a 要大于1
method=['mm21','mm22','mm23','mm24','mm25'],
#method=['mm31','mm32','mm33','mm34','mm30'],
#method=['mm40','mm41','mm42','mm43','mm44','mm45','mm46'],
#method=['mm51','mm52','mm53','mm54','mm55'],
#method=['mm10'],
#method=['mm90','mm91','mm92','mm93','mm94',],
)))

skwargs = dict(A=A,ŋ=.1,ŋ_a=1,tol=0.05,
               ε=.001,λ=.1,α=.5,γ=0.5,β1=.9,β2=.999)
wws=[];ess=[];rets=[]
for pg in pgrid:
    w0 = w.copy()-np.random.uniform(1,3,2) #任意起点

    kwargs=dict(X=X.copy(),y=y.copy(),
                gJ=gJ,J=J,w=w0,)
    kwargs.update(skwargs) ; kwargs.update(pg) ; var = VarSetX(kwargs)
    ret = gradient_descent_f(var,n_iters=20,skipConv=0,
                             **kwargs)

    ww = np.stack(ret['wh'][:,1])
    es = ret['wh'][:,2]
    wws.append(ww); ess.append(es); rets.append(ret)
    print(ww,es)

#%%

x = np.zeros(len(w));x
x = np.vstack([x, np.amax(X,axis=0)]);x
x_b = data_b(x)
yh = x.dot(B); yh

fig, ax = plt.subplots(figsize = (8,8))
ax.plot(X[:,0],y,'o')
ax.plot(x[:,0],yh,color='b',linewidth=5)

ws = [ww[int(i)] for i in np.linspace(0,len(ww)-1,10)]

for wx in ws:
    yh = x.dot(wx);yh # 画渐近的基准线
    ax.plot(x[:,0],yh,color='r')

ax.set_xlabel('x')
ax.set_ylabel('y')


#%%
plot_gd_contour(J,wws,ess,pgrid,skwargs,B)

#%%
paras = skwargs.copy()
paras.pop('A')
iters_gd_plot(rets,var,pgrid,paras=paras,
              **kwargs)