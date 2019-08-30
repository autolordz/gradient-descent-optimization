# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:01:06 2019

@author: autol
"""

#%%
import numpy as np
from  varclass import VarSetX,VarU

#%%
class Hessian(VarSetX):
    def __init__(self,var):
        self.A=0
        VarSetX.__init__(self,var)
    def A_(self): # Hessian 海森矩阵
        if not isinstance(self.A,np.ndarray):
            self.A = self.X.T @ self.X
        return self.A
    def H_(self): # invert Hessian 海森矩阵逆
        return np.linalg.inv(self.A_())
#%%

class JClass(VarSetX):
    def __init__(self,var,method):
        VarSetX.__init__(self,var)
        self.method = method
        self.Prox = ProxClass(var).Prox
        import inspect
        print(inspect.getsource(self.J))
    def get_method(self):
        return self.method
    #def J(self,w): # OLS
    #    X,y,= self.X,self.y,
    #    return np.mean((X.dot(w)-y)**2) # same as  1./n*(X.dot(w)-y).dot(X.dot(w)-y) or 1./n*(y-X.dot(w)).T.dot(y-X.dot(w))
    def J_L0(self,w): # Best subset selection L0约束 最优部分选取
        return self.J(w) + self.λ*np.linalg.norm(w,ord=0)

    def gJ_solve(self,w): # 闭式解OLS
        X,y,λ = self.X, self.y,self.λ
        if 'mm51' == self.method:
            return np.linalg.inv(X.T.dot(X) + λ*np.eye(len(w))) @ (X.T.dot(y))
        else:
            return np.linalg.inv(X.T.dot(X)) @ (X.T.dot(y))
    def gJ(self,w): # gradient of OLS
        X,y,n_y,λ = self.X, self.y,self.n_y,self.λ
        g = 2./n_y*X.T.dot(X.dot(w)-y)
        if 'mm51' == self.method:
            #return g+2*self.λ*w # Ridge gradient of Ridge  岭回归普通版
            return g+self.Prox(w,λ,f='L22') # 岭回归 近端算子版
        else:
            return g
    def gJv(self,w): # gradient of OLS v-column
        X,y,n_y = self.X, self.y, self.n_y
        return (2./n_y*X.T.dot(X.dot(w)-y))[:,np.newaxis]
    def Loss(self,w):
        λ,α,ε,method = self.λ,self.α,self.ε,self.method
        if 'mm51' == method:
            return self.J(w) + self.λ*w.dot(w) + ε # Ridge L22约束 岭回归
        elif method in ['mm52','mm26']:
            return self.J(w) + self.λ*np.linalg.norm(w,ord=1) + ε  # Lasso L1约束
        elif 'mm53' == method:  # elnet 弹性网约束 SLwSaprity f(4.2)
            return self.J(w) + λ*(1/2*(1-α)*w.dot(w)+α*np.linalg.norm(w,ord=1)) + ε # 1./n_y*(X.dot(w)-y).dot(X.dot(w)-y)
        else:
            return self.J(w) + ε # Metrics(X.dot(w),y) # J(w) from depends import Metrics

class ConvClass(VarSetX,VarU):

    def __init__(self,var):
        VarU.__init__(self)
        VarSetX.__init__(self,var)
        self.w0 = var.w

    def Conv(self,w,e,s,skipConv,flag='r2'):
        e0,e1,w0,w1,ratio0,gJ,ε=self.e0,e,self.w0,w,self.ratio0,self.gJ,self.ε
        rd = dict( # 各种残差
            r1=abs(e0-e1),
            r2=(e0-e1)**2, #最优
            r4=abs(e0-e1)/(np.linalg.norm(w0-w1)+ε),
            r5=np.linalg.norm(w0-w1),
            r6=np.linalg.norm(gJ(w0)-gJ(w1)))
        for k, v in rd.items():
            rd[k]='%.3e'%v
        #print('w=%s,r1=%s s=%.3e'%(w,rd,s))
        ratio=max(float(rd[flag]),1e-14)
        print('w=%s,r=%s '%(w,ratio))
        if not skipConv:
            #if e0-e1 <0 : print("Not converging!!");return 2,ratio
            if ratio < 1e-8: print("Converging!!  1");return 1,ratio # for r2
            if ratio0==ratio: print("Converging!!  2");return 1,ratio
        self.w0,self.e0,self.ratio0= w1.copy(),e1,ratio
        return 0,ratio

class ProxClass(VarSetX):

    def __init__(self,var):
        VarSetX.__init__(self,var)

    def Sλ(self,x,λ=1):
        # soft-thresholding operator 软阈值算子
        return np.sign(x) * np.maximum(np.abs(x)-λ,0)

    def Prox_L1_L22(self,x,λ,α):
        return 1./(1.+2*λ*α)*self.Sλ(x,λ)

    def Prox(self,x,λ,f='L1'):
        if f=='L1': # L1 近端算子就是软阈值算子
            return self.Sλ(x,λ)
        elif f=='L2': # L2 近端算子
            return np.maximum((1 - λ/np.linalg.norm(x)),0)*x
        elif f=='L22': #1/2||x||_2^2
            return 1./(1.+λ)*x
        elif f=='L1L2': #1/2||x||_2^2
            return i/(1+2*λ*α)*self.Prox(x,λ)
        else:
            return x

class CordClass(VarSetX):
    def __init__(self,var):
        VarSetX.__init__(self,var)
        self.ProxC = ProxClass(var)

    def gJj(self,w,j): # Steps Cord 坐标梯度
        X,y,n_y = self.X, self.y, self.n_y
        return 2./n_y * (X.dot(w)-y).dot(X[:,j])

    def gJi(self,w,i): # Steps Cord
        X,y,n_y = self.X, self.y, self.n_y
        return 2./n_y*(X.dot(w)-y).dot(X[:,i])

    def Wj(self,w,j): # Origin Cord 坐标下降
        X,y = self.X, self.y,
        z_j = X[:,j].dot(X[:,j])
        ρ_j = X[:,j].dot(y-X.dot(w))
        return ρ_j/z_j+w[j]

    def Wj_L1(self,w,j): # Lasso Cord L1坐标下降
        X,y,n_y,λ = self.X, self.y, self.n_y,self.λ
        # reference Coordinate Descent - Convex Optimization - Ryan Tibshirani
        # lasso regression by soft-thresholding
        # \beta_{i}=S_{\lambda /\left\|X_{i}\right\|_{2}^{2}}\left(\frac{X_{i}^{T}\left(y-X_{-i} \beta_{-i}\right)}{X_{i}^{T} X_{i}}\right)
        c = 2./n_y
        zj = X[:,j].dot(X[:,j])
        ρj = X[:,j].dot(y-X.dot(w) + X[:,j]*w[j])
        return self.ProxC.Sλ(ρj,λ/c)/zj
#        return self.Sλ(ρj/zj,λ/c/zj)

    def ADMM_L1(self,w): # ρ 越小越快
        X,y,λ,ρ = self.X, self.y,self.λ,max(self.ρ,self.ε)
        θ,μ=self.θ,self.μ
        # 'ADMM for the lasso' reference 'statistical learning with sparsity' formula(5.66)
        #\beta^{t+1}=\left(\mathbf{X}^{T} \mathbf{X}+\rho \mathbf{I}\right)^{-1}\left(\mathbf{X}^{T} \mathbf{y}+\rho \theta^{t}-\mu^{t}\right)
        #print(123123,μ)
        w = np.linalg.inv(X.T.dot(X) + ρ*np.eye(len(w))) @ (X.T.dot(y) + ρ*θ - μ)
        #\theta^{t+1}=\mathcal{S}_{\lambda / \rho}\left(\beta^{t+1}+\mu^{t} / \rho\right)
        #θ = Sλ(np.average(β+μ/ρ),λ/ρ/n_y)
        θ = self.ProxC.Sλ(w+μ/ρ,λ/ρ)
        #\mu^{t+1}=\mu^{t}+\rho\left(\beta^{t+1}-\theta^{t+1}\right)
        μ += ρ*(w-θ)
        self.θ,self.μ = θ,μ
        return w

    def Wj_L1L2(self,w,j): # Elasticnet Cord 弹性网坐标
        X,y,n_y,λ,α = self.X, self.y, self.n_y,self.λ,self.α
        # reference same Lasso Cord L1坐标下降
        c = 2./n_y
        zj = X[:,j].dot(X[:,j]) + (1-α)*λ
        ρj = X[:,j].dot(y-X.dot(w) + X[:,j]*w[j])
        return self.ProxC.Prox(ρj/zj,λ/c/zj) #近端收缩算子

    def group_w(self,w,group_n=1): # Block Coordinate 块坐标下降
        return np.array_split(np.random.permutation(len(w)),group_n)

class StepClass(VarSetX):
    """docstring for ClassName"""
    def __init__(self,var):
        VarSetX.__init__(self,var)
        self.gJi = CordClass(var).gJi
        self.Prox = ProxClass(var).Prox

    def BLS(self,w,F): # 后线性搜索 backtracking_line_search
        gJ,Prox = self.gJ,self.Prox
        def search(x):
            α = 1.0
            β  = 0.9
            while True:
                x_p = Prox(x - α*gJ(x), α) # 近端算子
                G = (1.0/α) * (x - x_p)
                if F(x_p) <= F(x) - α/2 * (G @ G):
                    return α
                else:
                    α = α * β
        return search(w)

    def armijo_i(self,w,ŋ_a):
        J,gJ = self.J,self.gJ
        β = .9;α=.5
        #assert α <= 0.5, "Armijo rule  α is applicable for beta less than 0.5"
        assert β < 1, "Decay factor β has to be less than 1"
        w1,w0 = w.copy(),w.copy()
        s = ŋ_a
        g0 = gJ(w0)
        w1 = w0+s*(-g0)
        print('step0',s)
        #print(111,J(w1),J(w0) + α*s*(-g0).dot(g0)) #gJ(w0).dot(w1-w0)
        #while not np.isnan(J(w1)) and J(w1) > J(w0) + s*α*(-g0).dot(g0):
        while 1:
            if np.isnan(J(w1)) or J(w1) <= J(w0) + s*α*(-g0).dot(g0):break
            #print(222,J(w1),J(w0) + α*s*(-g0).dot(g0))
            s *= β
            if s < 1e-16:
                print('Step s too small')
                break
            w1 = w0+s*(-g0)
        #print(333,J(w1),J(w0) + α*s*(-g0).dot(g0))
        print('step1',s)
        return s

#    def armijo_i1(self,w,ŋ_a):
#        J,gJ,gJi = self.Jc.J,self.Jc.gJ,self.gJi
#        β = .8;α=.5
#        assert α <= 0.5, "Armijo rule  α is applicable for beta less than 0.5"
#        assert β <= 1, "Decay factor β has to be less than 1"
#        ss = []
#        w1,w0 = w.copy(),w.copy()
##        w1 = w.copy()
##        w0 = w.copy()
#        for i in range(len(w0)):
#            s = ŋ_a
#            gg = gJi(w0,i)
#            w1[i] = w0[i]+s*(-gg)
#            g0 = gJ(w0)
#            print('step0',i,s)
#            print(111,i,J(w1),J(w0) + α*s*(-gJ(w0)).dot(gJ(w0))) #gJ(w0).dot(w1-w0)
#            while np.isnan(J(w1)) or J(w1) > J(w0) +  α*s*(-g0).dot(g0):
#                print(222,J(w1),J(w0) + α*s*(-g0).dot(g0))
#                s *= β
#                if s < 1e-16:
#                    print('Step s too small')
#                    break
#                w1[i] = w0[i]+s*(-gg)
#            print(333,i,J(w1),J(w0) + α*s*(-g0).dot(g0)) #gJ(w0).dot(w1-w0)
#            print('step1',i,s)
#            ss.append(s)
#        return np.array(ss)






