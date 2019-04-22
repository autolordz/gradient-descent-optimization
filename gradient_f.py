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
    print_range = np.round(np.linspace(1,n_iters,10))
    n = len(y)
    w_h,Δh = [],[]
    
    np.random.seed(seed=43)
    r = np.random.permutation(n)
    X = X[r,:]
    y = y[r]
    
    def J_(w):
        return 1./n*(X.dot(w)-y).dot(X.dot(w)-y)
    def gJ(w):
        return 2./n*X.T.dot(X.dot(w)-y)
    def F(w):
        return 1./n_batch*(X_b.dot(w)-y_b).dot(X_b.dot(w)-y_b)
    def gF(w):
        return 2./n_batch*X_b.T.dot(X_b.dot(w)-y_b)
    def gFj(w,j):
        Xj = X_b[:,j]
        if type(j) == list:
            Xj = Xj.T # d>1 X transpose
        return 2./n_batch*Xj.dot(X_b.dot(w)-y_b)
    def gFj1(w,j):
        g = X_b[:,j].dot(y_b-X_b.dot(w))/ X_b[:,j].dot(X_b[:,j])
        g = 2./n_batch*(g+w[j])
        return g
    def gFj2(w,j):
        Xjwj = np.delete(X_b,j,1).dot(np.delete(w,j,0))
        g = 2./n_batch*X_b[:,j].dot(y_b-Xjwj) / X_b[:,j].dot(X_b[:,j])
        return g
    def armijo(w):
        ŋ = 1;β = .8;α=.5
        g = gF(w)
        while F(w-ŋ*g) > F(w) + ŋ*α*g.dot(-g):
            ŋ *= β
        return ŋ
    def B_(): # Hessian
        return 2./n_batch * X_b.T.dot(X_b)
    def H_(B): # invert H
        return np.linalg.inv(B)
    
    if isbegin:Δh.append(np.hstack([0,J_(w),w.copy()]))
    
    θ = w1 = w.copy()
    B = H = np.eye(len(w))
    fname = G = m = v = u = 0
    β1,β2=.9,.999
    s = 1.
    γ = 0.9
    ε = 1e-2
    ŋ = 1e-1
    
    paras = 'β1,β2=.9,.999 γ = %s ε = 1e-2 ŋ = %s'%('t/(t+3)' if isγ else γ,
                                                    'armijo' if isarmijo else ŋ)
    
    for i in range(n_iters):
        t = i+1
        if isγ: γ = t/(t+3)
        for k in range(0,n,n_batch):
            X_b = X[k:k + n_batch]
            y_b = y[k:k + n_batch]
            if isarmijo: ŋ = armijo(w)
            if 'mm21' == func:
                fname = 'Polyak-M-H' # Polyak’s Momentum Head
                v = γ*v + ŋ*gF(w)
                w += -v
            elif 'mm22' == func: # Polyak’s Momentum Back
                fname = 'Polyak-M-B'
                w2 = w1 - ŋ*gF(w1) + γ*(w1 - w)
                w = w1
                w1 = w2
            elif 'mm23' == func: # NAG Head 1
                fname = 'NAG-H'
                v = γ*v + ŋ*gF(w-γ*v)
                w += -v
            elif 'mm24' == func:
                fname = 'NAG-B' # NAG Back
                v0 = v
                v = γ*v - ŋ*gF(w) 
                w += v + γ*(v-v0)
            elif 'mm25' == func:
                fname = 'NAG-H2' #  NAG Head 2
                w1 = w + γ * v 
                v = γ*v - ŋ*gF(w1)
                w += v
            elif 'mm26' == func: # a proximal gradient version of Nesterov’s 1983 method
                fname = 'FISTA'
                s1 = .5*(1 + np.sqrt(1 + 4*s**2))
                w1 = θ - ŋ*gF(θ)
                θ1 = w1 + (s-1.)/s*(w1-w)
                # if np.linalg.norm(w1-w) <= ε*np.linalg.norm(w):
                #     break
                # if np.dot(θ1-w1,w1-w) > 0:
                #     θ1 = w1.copy()
                #     t1 = 1.
                w,θ,s = w1,θ1,s1
            elif 'mm30' == func:
                fname = 'Newton'
                w += -H_(B_()).dot(gF(w))
            elif 'mm31' == func:
                fname = 'Minimizing Along a Line'
                g = gF(w)
                p = -g
                ŋ = -g.dot(p) / p.dot(B_()).dot(p)
                w += -ŋ*gF(w)
            elif 'mm32' == func:
                fname = 'Conjugate Gradient'
                g = gF(w)
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
                Δw = -ŋ*H@gF(w)
                w1 = w + Δw
                Δg = gF(w1) - gF(w)
                B = B + np.outer(Δg-B@Δw/Δw.dot(Δw), Δw) 
                H = H + np.outer(Δw-H@Δg, Δw.T@H) / Δw.dot(H).dot(Δg)
                w = w1
            elif 'mm40' == func:
                fname = 'Adagrad'
                g = gF(w)
                G = G + g*g
                w += -ŋ/np.sqrt(G+ε)*g
            elif 'mm41' == func:
                fname = 'RMSProp'
                g = gF(w)
                G = γ*G + (1-γ)*g*g
                w += -ŋ/np.sqrt(G+ε)*g
            elif 'mm42' == func:
                fname = 'Adadelta'
                g = gF(w)
                G = γ*G + (1-γ)*g*g
                ŋ = np.sqrt(v+ε)/(np.sqrt(G+ε))
                u = - ŋ*g
                v = γ*v + (1-γ)*u*u
                w += u
            elif 'mm43' == func:
                fname = 'Adam'
                g = gF(w)
                m = β1*m+(1-β1)*g
                v = β2*v+(1-β2)*g*g
                mh = m/(1-β1**(i+1))
                vh = v/(1-β2**(i+1))
                w += - ŋ/(np.sqrt(vh+ε))*mh
            elif 'mm44' == func:
                fname = 'AdaMax'
                g = gF(w)
                m = β1*m+(1-β1)*g
                u = np.maximum(β2*u,abs(g))
                mh = m/(1-β1**(i+1))
                w += -ŋ/(u+ε)*mh
            elif 'mm45' == func:
                fname = 'Nadam'
                g = gF(w)
                m = β1*m+(1-β1)*g
                v = β2*v+(1-β2)*g*g
                gh = g/(1-β1**(i+1))
                mh = m/(1-β1**(i+1))
                vh = v/(1-β2**(i+1))
                mhh = β1*mh+(1 - β1)*gh
                w += -ŋ/(np.sqrt(vh+ε))*mhh
            elif 'mm46' == func:
                fname = 'AMSGrad'
                g = gF(w)
                m = β1*m+(1-β1)*g
                vh = v/(1-β2**(i+1))
                v = β2*v+(1-β2)*g*g
                vh = np.maximum(vh,v)
                w += - ŋ/(np.sqrt(vh+ε))*m
            elif 'mm90' == func:
                fname = 'R-Cord gFj step descent'
                j = np.mod(i,len(w))
                w[j] += -ŋ*gFj(w,j)
            elif 'mm91' == func:
                fname = 'R-Cord gFj1 each w[j]'
                j = np.mod(i,len(w))
                w[j] = gFj1(w,j)
            elif 'mm92' == func:
                fname = 'R-Cord gFj2 each w[j]'
                j = np.mod(i,len(w))
                w[j] = gFj2(w,j)
            elif 'mm93' == func:
                fname = 'S-Cord random w[j]'
                j = np.random.randint(0,len(w))
                w[j] = gFj2(w,j)
            elif 'mm94' == func:
                n_b = 2
                fname = 'SB-Cord batch=%s'%n_b
                j = np.random.randint(0,len(w),n_b).tolist()
                w[j] += -ŋ*gFj(w,j)
            elif 'mm10' == func: # Origin
                fname = 'SGD'
                w += -ŋ*gF(w)
            elif 'mm11' == func: # 
                fname = 'SGD + Armijo rules'
                ŋ = armijo(w)
                w += -ŋ*gF(w)
            else:
                doplot = False
            Δh.append(np.hstack([i+k/n,J_(w),w.copy()]))
    w_h = pd.DataFrame(Δh,columns=['iters','ΔJ'] + w_col)
#    print('==GD=w==\n',w_h)
    return w_h,fname,doplot,paras

def iters_gd_plot(n_batch =1,n_iters = 10,ff=[],
                 islogy=False,**kwargs):
    fig, ax = plt.subplots(figsize = (10,10))
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
##      'mm90',
##      'mm41',
#      'mm94'
#      ]

rr = list(range(30,100))
ff = ['mm'+str(s) for s in rr]
iters_gd_plot(ff=ff,
              n_batch=100,
              n_iters=30,
              islogy=True,
              isbegin=False,
              isγ=True,
              isarmijo=False,
              )