# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:59:29 2019

@author: autol
"""

#%%
import numpy as np
from  varclass import VarSetX,VarU
from gfun import CordClass,ProxClass

class UpdateClass(VarSetX,VarU):

    def __init__(self,var):
        VarU.__init__(self)
        VarSetX.__init__(self,var)
        self.Cc = CordClass(self.__dict__)
        self.Pc = ProxClass(var)


    def update_w(self,w,ŋ=0,i=0,X=0,y=0): # 更新w函数
        n_w = len(w)
        gJ,Cc,Pc = self.gJ,self.Cc,self.Pc
        if not isinstance(X,np.ndarray):X=self.X
        if not isinstance(y,np.ndarray):y=self.y
        A,H=self.A,self.H
        λ,β1,β2,ε,γ=self.β1,self.β2,self.ε,self.γ,self.λ
        β,θ,μ,ρ=self.β,self.θ,self.μ,self.ρ
        v,G,g0=self.v,self.G,self.g0
        m,γ,t=self.m,self.γ,self.t
        method = self.method

        if method in ['mm10','mm51']: # Origin Ridge
            w += -ŋ*gJ(w)
            #w = gJ_solve(w)
        elif 'mm52' == method:  # Lasso
            j = np.mod(i,n_w)
            w[j] = Cc.Wj_L1(w,j)
        elif 'mm53' == method: # elnet
            j = np.mod(i,n_w) # every time check one index of weight
            w[j] = Cc.Wj_L1L2(w,j)
        elif 'mm54' == method: # ADMM 交替乘子
            w = Cc.ADMM_L1(w)
        elif 'mm55' == method: # FISTA 快速近端L1
            # FISTA reference Sparse Modeling Theory
            # Fast Iterative Soft Thresholding Algorithm
            # a proximal gradient version of Nesterov’s 1983 method
            # reference http://www.stronglyconvex.com/blog/accelerated-proximal-gradient-descent.html
            #print(w,θ,t)
            w1 = Pc.Prox(θ - ŋ*gJ(θ),λ,f='L1') # 这里应该是近端算子prox, L1 的近端算子是 Sλ
            t1 = 1/2*(1 + np.sqrt(1 + 4*t**2))
            θ1 = w1 + ((t-1.)/t1) * (w1-w)
            if np.dot(θ1-w1,w1-w) > 0:
                θ1 = w1.copy()
                t1 = 1.
            w,θ,t = w1,θ1,t1
        elif 'mm90' == method: # R-Cord gFj step descent 循环坐标下降 逐步解
            j = np.mod(i,n_w)
            w[j] += -ŋ*Cc.gJj(w,j)
        elif 'mm91' == method: # R-Cord Wj(w) each w[j] 循环坐标下降 闭式解(最优)
            j = np.mod(i,n_w)
            w[j] = Cc.Wj(w,j)
        elif 'mm92' == method: # S-Cord random w[j] 随机坐标下降
            j = np.random.randint(0,n_w)
            w[j] = Cc.Wj(w,j)
        elif 'mm93' == method: # S-R-Block-Cord 循环块坐标
            group_n=1
            j = Cc.group_w(w,group_n)[np.mod(i,group_n)]
            w[j] += -ŋ*Cc.gJj(w,j)
        elif 'mm94' == method: # S-S-Block-Cord 随机块坐标
            group_n=1
            j = Cc.group_w(w,group_n)[np.random.randint(0,group_n)]
            w[j] += -ŋ*Cc.gJj(w,j)
        elif 'mm21' == method: # Polyak’s Momentum 前定义(更优)
            print(γ,ŋ,v)
            v = γ*v + ŋ*gJ(w)
            w += -v
        elif 'mm22' == method: # Polyak’s Momentum 后定义
            v = γ*v + gJ(w)
            w += -ŋ*v
        elif 'mm23' == method: # Nesterov accelerated gradient NAG 前定义(更优)
            # reference http://ruder.io/optimizing-gradient-descent/index.html#momentum
            # \begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right) \\ \theta &=\theta-v_{t} \end{aligned}
            v = γ*v + ŋ*gJ(w-γ*v)
            w += -v
        elif 'mm24' == method: # NAG 后定义
            v0 = v
            v = γ*v - ŋ*gJ(w)
            w += (v + γ*(v-v0))
        elif 'mm25' == method: # NAG 前定义2 same with mm23
            w1 = w + γ*v
            v = γ*v - ŋ*gJ(w1)
            w += v
        elif 'mm30' == method:
            w += -(H @ gJ(w)) #逆海森矩阵
        elif 'mm31' == method: # Minimizing Along a Line
            g = gJ(w) ; p = -g
            w -= p.dot(p) / p.dot(A).dot(p) * g
        elif 'mm32' == method: # Conjugate Gradient
            w0 = w.copy() ; p0 = -g0
            g = gJ(w)
            if i == 0: p = -g # i=次数
            else: p = -g + (g@g) / (g0 @ g0) * p0
            w = w0 + p.dot(p) / p.dot(A).dot(p) * p
            g0 = g.copy()
        elif 'mm33' == method: # Quasi-Newton(Broyden)
            p = -ŋ*(H @ gJ(w)) # 和速率有关，非直接牛顿法
            q = gJ(w + p) - gJ(w)
            p = p[:,np.newaxis] ; q = q[:,np.newaxis]
#            B += np.outer(Δg-B @ Δw/(Δw @ Δw), Δw)
            H += ((p - H @ q)@ p.T @ H) / (p.T @ H @ q)
#            H += p @ p.T/ p.T @ q - H @ q @ q.T @ H / q.T @ H @ q
            p = p.ravel()
            w = w + p
        elif 'mm40' == method:   #'Adagrad' # 叠加 armijo 比较慢
            g = gJ(w)
            G = G + g*g
            w += -ŋ/np.sqrt(G+ε)*g
        elif 'mm41' == method:  # 'RMSProp' # γ是衰减率 
            g = gJ(w)
            G = γ*G + (1-γ)*g*g
            w += -ŋ/np.sqrt(G+ε)*g
        elif 'mm42' == method: # 'Adadelta'
            g = gJ(w)
            G = γ*G + (1-γ)*g*g
            ŋ = np.sqrt(v+ε)/(np.sqrt(G+ε))
            d = - ŋ*g
            w += d
            v = γ*v + (1-γ)*d*d
        elif 'mm43' == method: # 'Adam' # 转圈速度慢
            g = gJ(w)
            print(1212,w,m,v,g)
            m = β1*m+(1-β1)*g # 更新一阶矩
            v = β2*v+(1-β2)*(g**2) # 更新二阶矩
            mh = m/(1-β1**(i+1)) # 矫正一阶矩
            vh = v/(1-β2**(i+1)) # 矫正二阶矩
            w += - ŋ*mh/(np.sqrt(vh)+ε)
        elif 'mm44' == method: # 'AdaMax' # 转圈速度慢
            g = gJ(w)
            m = β1*m+(1-β1)*g # 更新一阶矩
            v = np.maximum(β2*v,np.abs(g)) # 更新无穷矩
            mh = m/(1-β1**(i+1)) # 矫正一阶矩
            w += -ŋ*mh/(np.sqrt(v)+ε)
        elif 'mm45' == method: # 'Nadam' # 转圈速度慢
            g = gJ(w)
            m = β1*m+(1-β1)*g # 更新一阶矩
            v = β2*v+(1-β2)*g*g  # 更新二阶矩
            mw = β1*m+(1-β1)*g # 更新一阶矩的加速值
            mh = mw/(1-β1**(i+1)) # 矫正一阶矩
            vh = v/(1-β2**(i+1)) # 矫正二阶矩
            w += -ŋ*mh/(np.sqrt(vh)+ε)
        elif 'mm46' == method: # 'AMSGrad' # 非常慢
            print(w,m,v,θ)
            g = gJ(w)
            m = β1*m+(1-β1)*g
            v = β2*v+(1-β2)*g*g
            θ = v/(1-β2**(i+1))
            θ = np.maximum(θ,v)
            w += - ŋ*m/(np.sqrt(θ)+ε)
        self.m,self.γ,self.t = m,γ,t
        self.v,self.G,self.g0 = v,G,g0
        self.β,self.θ,self.μ,self.ρ = β,θ,μ,ρ
        return w.copy()