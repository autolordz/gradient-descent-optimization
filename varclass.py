# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:13:05 2019

@author: autol
"""

#%%
import numpy as np
class VarU(object):
    def __init__(self,v=0,m=0,θ=0,t=0,β=0,A=0,H=0,
                 g0=0,w0=0,w1=0,G=0,μ=0,ρ=0,
                 e0=np.inf,ratio0=np.inf,
                 ):
        self.v=v;self.m=m;self.t=t
        self.θ=θ;self.β=β;self.ρ=ρ;self.μ=μ
        self.G=G;self.g0=g0
        self.A=A;self.H=H
        self.w0=w0;self.w1=w1
        self.e0=e0;self.ratio0=ratio0

#%%
#class VarX(object):
#    def __init__(self,
#                 X=0,y=0,w=0,gJ=0,J=0,
#                 ŋ=0,ŋ_a=0,tol=0,ε=0,
#                 λ=0,α=0,γ=0,
#                 β1=0,β2=0,
#                 Convf=0,
#                 ):
#        self.X=X;self.y=y;self.w=w;self.n_y = len(y)
#        self.J=J;self.gJ=gJ;
#        self.ŋ=ŋ;self.ŋ_a=ŋ_a;self.tol=tol;self.ε=ε
#        self.β1=β1;self.β2=β2;self.λ=λ;
#        self.α=α;self.γ=γ
#%%
class VarSetX(object):
    def __init__(self,var):
        v = var if isinstance(var,dict) else var.__dict__
        for k,v in v.items():
            if k=='y':
                self.n_y = len(v)
            setattr(self, k, v)
    def set(self,var):
        v = var if isinstance(var,dict) else var.__dict__
        for k,v in v.items():
            if k=='y':
                self.n_y = len(v)
            setattr(self, k, v)

#class VarSetU(object):
#    def __init__(self,var):
#        for k,v in var.__dict__.items():
#            setattr(self, k, v)
#%%

#var0 = VarX(
#            X=X.copy(),y=y.copy(),w=w.copy(),
#            ŋ=1e-1,ŋ_a=1,tol=0.05,ε=.9,
#            n_b=20,isγ=0,Convf=1,
#            λ=1,α=.5,γ=0.9,
#            β1=.9,β2=.999,
#            #  ŋ=4e-3,ŋ_a=3,
#            )
#
#cc = VarSet(var0)
#cc.__dict__

