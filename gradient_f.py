# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:11:23 2019

@author: autol
"""

#%%
from depends import *

#%%

def gradient_descent_f(w,X,y,n_iters = 10,
                       func='',n_batch = 1,
                       **kwargs):
    
    isbegin = kwargs.get('isbegin',False)
    isγ = kwargs.get('isγ',False)
    isarmijo = kwargs.get('isarmijo',False)

    doplot = True
    n = len(y)
    w_h,Δh = [],[]
    
    # Shuffle X,y 
    np.random.seed(seed=43)
    r = np.random.permutation(n)
    X = X[r,:]
    y = y[r]
    
    n_b = n_batch
    
    def armijo(w):
        ŋ = 1;β = .8;α=.5
        g = gJb(w)
        while Jb(w-ŋ*g) > Jb(w) + ŋ*α*g.dot(-g):
            ŋ *= β
        return ŋ
    def B_(): # Hessian
        return 2./n_b * X_b.T.dot(X_b)
    def H_(B): # invert H
        return np.linalg.inv(B)
    def J_(w): # OLS
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y)
    def Jb(w):
        return 1./n_b*(X_b.dot(w)-y_b).dot(X_b.dot(w)-y_b)
    def gJ(w): # gradient of OLS
        return 2./n*X.T.dot(X.dot(w)-y)
    def gJb(w):
        return 2./n_b*X_b.T.dot(X_b.dot(w)-y_b)
    def J_L0(w): # Best subset selection
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y) + λ*np.linalg.norm(w,ord=0)
    def J_L1(w): # Lasso
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y) + λ*np.linalg.norm(w,ord=1)
    def J_L2(w): # Ridge
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y) + λ*w.dot(w)
    def gJ_L2(w):
        return 2./n_b*X_b.T.dot(X_b.dot(w)-y_b)+2*λ*w
    def J_GL1(w): # Group Lasso
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y) + λ*np.linalg.norm(w,ord=2)
    def J_L1L2(w): # elnet
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y) + \
                    λ*1/2*((1-α)*w.dot(w)+α*np.linalg.norm(w,ord=1))
    def group_w(w,n):
        return np.array_split(np.random.permutation(len(w)),n)
    def Sλ(x,λ):
        if x > λ:
            return x - λ
        elif x < -λ:
            return x + λ
        else:
            return 0.
    def gJj(w,j):
        Xj = X_b[:,j]
        return 2./n_b*(X_b.dot(w)-y_b).dot(Xj)
    def Wj(w,j):
        z_j = X_b[:,j].dot(X_b[:,j])
        ρ_j = X_b[:,j].dot(y_b-X_b.dot(w))
        return ρ_j/z_j+w[j]
    def Wj_L1(w,j,λ):
        c = 2./n_b
        zj = X_b[:,j].dot(X_b[:,j])
        ρj = X_b[:,j].dot(y_b-X_b.dot(w) + X_b[:,j]*w[j])
        return Sλ(ρj,λ/c)/zj
    def Wj_L1L2(w,j,λ,α):
        c = 2./n_b
        zj = X_b[:,j].dot(X_b[:,j]) + λ*(1-α)
        ρj = X_b[:,j].dot(y_b-X_b.dot(w) + X_b[:,j]*w[j])
        return Sλ(ρj,λ/c)/zj
    
    λ = 1e-2
    α = 1.
    if 'mm51' == func: J_ = J_L1
    if 'mm52' == func: J_ = J_L1
    if 'mm53' == func: J_ = J_L2
    if 'mm54' == func: J_ = J_L1L2
    
    if isbegin:Δh.append(np.hstack([0,J_(w),w.copy()]))
    
    θ = w1 = w.copy()
    B = H = np.eye(len(w))
    fname = G = m = v = u = μ = 0
    β1,β2=.9,.999
    s = 1.
    γ = 0.9
    ε = 1e-2
    ŋ = ρ = 1e-1
    
    paras = '''β1,β2=.9,.999,γ=%s
                ρ=1e-1,λ=1e-2,α=1.
                ε=1e-2 ŋ=%s '''\
            %('t/(t+3)' if isγ else γ,
              'armijo' if isarmijo else ŋ)
    for i in range(n_iters):
        t = i+1
        if isγ: γ = t/(t+3)
        for k in range(0,n,n_b):
            X_b = X[k:k + n_b]
            y_b = y[k:k + n_b]
            if isarmijo: ŋ = armijo(w)
            if 'mm51' == func:
                fname = 'Ridge'
                w += -ŋ*gJ_L2(w)
            elif 'mm53' == func:
                fname = 'elnet'
                j = np.mod(i,len(w)) # loop over each weight
                w[j] = Wj_L1L2(w,j,λ,α)
            elif 'mm10' == func: # Origin
                fname = 'SGD'
                w += -ŋ*gJb(w)
            elif 'mm52' == func:
                fname = 'Lasso'
                j = np.mod(i,len(w)) # loop over each weight
                w[j] = Wj_L1(w,j,λ)
            elif 'mm54' == func:
                fname = 'ADMM'
                β = np.linalg.inv(X_b.T.dot(X_b) + ρ*np.eye(len(w))).dot(X.T.dot(y)+ρ*θ-μ)
                θ = Sλ(np.average(β+μ/ρ),λ/ρ/n)
                μ = μ + ρ*(β-θ)
                w = β.copy()
            elif 'mm21' == func:
                fname = 'Polyak-M-H' # Polyak’s Momentum Head
                v = γ*v + ŋ*gJb(w)
                w += -v
            elif 'mm22' == func: # Polyak’s Momentum Back
                fname = 'Polyak-M-B'
                w2 = w1 - ŋ*gJb(w1) + γ*(w1 - w)
                w = w1
                w1 = w2
            elif 'mm23' == func: # NAG Head 1
                fname = 'NAG-H'
                v = γ*v + ŋ*gJb(w-γ*v)
                w += -v
            elif 'mm24' == func:
                fname = 'NAG-B' # NAG Back
                v0 = v
                v = γ*v - ŋ*gJb(w) 
                w += v + γ*(v-v0)
            elif 'mm25' == func:
                fname = 'NAG-H2' #  NAG Head 2
                w1 = w + γ * v 
                v = γ*v - ŋ*gJb(w1)
                w += v
            elif 'mm26' == func: # a proximal gradient version of Nesterov’s 1983 method
                fname = 'FISTA'
                s1 = .5*(1 + np.sqrt(1 + 4*s**2))
                w1 = θ - ŋ*gJb(θ)
                θ1 = w1 + (s-1.)/s*(w1-w)
                # if np.linalg.norm(w1-w) <= ε*np.linalg.norm(w):
                #     break
                # if np.dot(θ1-w1,w1-w) > 0:
                #     θ1 = w1.copy()
                #     t1 = 1.
                w,θ,s = w1,θ1,s1
            elif 'mm30' == func:
                fname = 'Newton'
                w += -H_(B_()).dot(gJb(w))
            elif 'mm31' == func:
                fname = 'Minimizing Along a Line'
                g = gJb(w)
                p = -g
                ŋ = -g.dot(p) / p.dot(B_()).dot(p)
                w += -ŋ*gJb(w)
            elif 'mm32' == func:
                fname = 'Conjugate Gradient'
                g = gJb(w)
                p = -g
                if k == 0:
                    β = 0
                else:
                    β = (g@g) / (g0@g0)
                p = -g + β*p
                w += ŋ*p
                g0 = g
            elif 'mm33' == func:
                fname = 'Quasi-Newton(Broyden)'
                Δw = -ŋ*H@gJb(w)
                w1 = w + Δw
                Δg = gJb(w1) - gJb(w)
                B = B + np.outer(Δg-B@Δw/Δw.dot(Δw), Δw) 
                H = H + np.outer(Δw-H@Δg, Δw.T@H) / Δw.dot(H).dot(Δg)
                w = w1
            elif 'mm40' == func:
                fname = 'Adagrad'
                g = gJb(w)
                G = G + g*g
                w += -ŋ/np.sqrt(G+ε)*g
            elif 'mm41' == func:
                fname = 'RMSProp'
                g = gJb(w)
                G = γ*G + (1-γ)*g*g
                w += -ŋ/np.sqrt(G+ε)*g
            elif 'mm42' == func:
                fname = 'Adadelta'
                g = gJb(w)
                G = γ*G + (1-γ)*g*g
                ŋ = np.sqrt(v+ε)/(np.sqrt(G+ε))
                u = - ŋ*g
                v = γ*v + (1-γ)*u*u
                w += u
            elif 'mm43' == func:
                fname = 'Adam'
                g = gJb(w)
                m = β1*m+(1-β1)*g
                v = β2*v+(1-β2)*g*g
                mh = m/(1-β1**(i+1))
                vh = v/(1-β2**(i+1))
                w += - ŋ/(np.sqrt(vh+ε))*mh
            elif 'mm44' == func:
                fname = 'AdaMax'
                g = gJb(w)
                m = β1*m+(1-β1)*g
                u = np.maximum(β2*u,abs(g))
                mh = m/(1-β1**(i+1))
                w += -ŋ/(u+ε)*mh
            elif 'mm45' == func:
                fname = 'Nadam'
                g = gJb(w)
                m = β1*m+(1-β1)*g
                v = β2*v+(1-β2)*g*g
                gh = g/(1-β1**(i+1))
                mh = m/(1-β1**(i+1))
                vh = v/(1-β2**(i+1))
                mhh = β1*mh+(1 - β1)*gh
                w += -ŋ/(np.sqrt(vh+ε))*mhh
            elif 'mm46' == func:
                fname = 'AMSGrad'
                g = gJb(w)
                m = β1*m+(1-β1)*g
                vh = v/(1-β2**(i+1))
                v = β2*v+(1-β2)*g*g
                vh = np.maximum(vh,v)
                w += - ŋ/(np.sqrt(vh+ε))*m
            elif 'mm91' == func:
                fname = 'R-Cord Wj(w)1 each w[j]'
                j = np.mod(i,len(w))
                w[j] = Wj(w,j)
            elif 'mm92' == func:
                fname = 'R-Cord Wj(w)2 each w[j]'
                j = np.mod(i,len(w))
                w[j] = Wj(w,j)
            elif 'mm93' == func:
                fname = 'S-Cord random w[j]'
                j = np.random.randint(0,len(w))
                w[j] = Wj(w,j)
            elif 'mm90' == func:
                fname = 'R-Cord gFj step descent'
                j = np.mod(i,len(w))
                w[j] += -ŋ*gJj(w,j)
            elif 'mm94' == func:
                group_n = 2
                group = group_w(w,group_n)
                fname = 'S-R-Block-Cord blocks=%s'%group_n
                j = group[np.mod(i,group_n)]
                w[j] += -ŋ*gJj(w,j)
            elif 'mm95' == func:
                group_n = 2
                group = group_w(w,group_n)
                fname = 'S-S-Block-Cord blocks=%s'%group_n
                j = group[np.random.randint(0,group_n)]
                w[j] += -ŋ*gJj(w,j)
            elif 'mm11' == func: # 
                fname = 'SGD + Armijo rules'
                ŋ = armijo(w)
                w += -ŋ*gJb(w)
            else:
                doplot = False
            Δh.append(np.hstack([i+k/n,J_(w),w.copy()]))
    w_h = pd.DataFrame(Δh,columns=['iters','ΔJ'] + w_col)
#    print('==GD=w==\n',w_h)
    return w_h,fname,doplot,paras

def iters_gd_plot(n_batch =1,
                  n_iters = 10,
                  islogy=False,
                  ff=[],**kwargs):
    fig, ax = plt.subplots(figsize = (8,8))
    mk = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    aplog = ax.semilogy if islogy else ax.plot
    lastJ = np.Infinity
    for f in ff:
        w_h,fname,doplot,paras = gradient_descent_f(w.copy(),X,y,
                                           func=f,
                                           n_iters = n_iters,
                                           n_batch = n_batch,
                                           **kwargs)
        if doplot:
            aplog(w_h['iters'],w_h['ΔJ'],
                        '-'+random.choice(mk),
                        label='%s'%fname)
            lastJ = np.append(lastJ,w_h['ΔJ'][-1:].values)
    print('lowestJ=====',lastJ)
    minJ = min(lastJ)
    ax.axhline(y = minJ,color='r')
    ax.text(0,minJ,'min(J) %.2f'%minJ,
            va='top', ha="right",color='r',
            transform = ax.get_yaxis_transform())
    ax.set_xlabel('iters')
    ax.set_ylabel("J(w)")
    ax.set_title("n=%s,batch=%s,iters=%s,\n %s"%(len(y),n_batch,n_iters,paras))
    ax.legend(loc=0)
    ax.grid(True)

#%%
    
n = 100
x = np.linspace(0,1,n)
noise = 3*np.random.uniform(size = n)
X = np.vstack([2*x,x**3,np.exp(x),np.sin(x),np.tanh(x)]).T
y = np.sin(x+noise)
y_noise = y + noise
y = y_noise - y_noise.mean()
w = np.ones(X.shape[1])
w_col = ['w'+str(s) for s in range(len(w))]

#ff = [
##      'mm54',
##      'mm52',
#      'mm521','mm522',
##      'mm53',
##      'mm10'
#      ]

rr = list(range(50,60))
ff = ['mm'+str(s) for s in rr]
iters_gd_plot(ff=ff,
              n_batch=20,
              n_iters=30,
              islogy=True,
              isbegin=False,
              isγ=True,
              isarmijo=True,
              )