# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:28:52 2019

@author: autol
"""

#%%
#mk = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
mk = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1']


import random
import matplotlib.pyplot as plt
import numpy as np
#from itertools import cycle
#from sklearn.utils import shuffle

def plot_gd_xy(
               X,y,
               ret=[],
               n_batch=1):
    if len(ret) ==0: return

    wh = ret.get('wh',0)
    finals = ret.get('finals',0)
    method = fname_dic.get(ret.get('method',''),'')
    paras = ret.get('paras','paras')

    fig, ax = plt.subplots(figsize = (8,8))
#    ax.set_yscale('symlog',linthreshy=1e-3)
    ax.set_xlabel('iters')
    #ax.set_ylabel('ratio')
    ax.set_ylabel('obj')

    xn = wh[:,0]
#    yn = wh[:,3]
    yn = wh[:,2]
    #ax.set_ylim(min(yn)-np.std(yn),max(yn)+np.std(yn))

    islogy=1
    axplot = ax.semilogy if islogy else ax.plot
    axplot(
            xn,yn,
            '-'+random.choice(mk),
            label=method)
#    minJ = yn.iloc[-1]
#    minJ = yn[-1]
#    ax.axhline(y = minJ,color='r')
#    ax.text(0,minJ,'final(obj) %.4e'%minJ,
#            va='top', ha="right",color='r',
#            transform = ax.get_yaxis_transform())
    iters =  xn[-1]
    if isinstance(y, np.ndarray):
        ax.set_title("y.size=%s,batch=%s,iters=%s,\n %s"%(y.shape[0],n_batch,iters,paras))
    else:
        ax.set_title('X.size=%s,iters=%s'%(X.shape,iters))
    ax.legend(loc=0)
    ax.grid(True)

#%%
#@accepts(A=np.ndarray,pgrid=list)
def iters_gd_plot(rets,var,pgrid,paras=0,
                  n_iters=1,onplot=1,
                  **kwargs):
    y=var.y
    poparas = dict(iters=0,w=1,obj=2,ratio=3)
    poparas = dict(zip(poparas.values(), poparas.keys()))
    lastJ = [] ;  pfinals = [] ;   w_finals = []
    if not onplot:return None
    if not isinstance(pgrid,list) or len(pgrid) == 0:
        print('There\'s no method to plot')
        return None
    # prepare to plot
    fig, ax = plt.subplots(figsize = (8,8))
    mk = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    islogy=1  ;  axplot = ax.semilogy if islogy else ax.plot
#        mkpool = cycle(shuffle(mk))
#        cmap = plt.get_cmap('Set1')
#        colors = cmap(np.random.rand(len(pgrid)))
    for ret,pg in zip(rets,pgrid): # for loop to plot lines
        wh = ret.get('wh',0) ; finals = ret.get('finals',{}) ;
        w_finals.append(finals.get('w',0))
        pfinals.append(finals)
        idx = 2;
        xn,yn = wh[:,0],wh[:,idx]
        lastJ.append(finals.get('e1',0))
        axplot(xn,yn,'-'+random.choice(mk),label=pg)

    if onplot:
        minJ = min(lastJ)
        ax.set_xlabel(poparas[0])
        ax.set_ylabel(poparas[idx])
        ax.axhline(y = minJ,color='r')
        ax.text(0,minJ,'min() %.4e'%minJ,
                va='top', ha="right",color='r',
                transform = ax.get_yaxis_transform())
        if isinstance(y, np.ndarray):
            ax.set_title('paras %s'%paras)
#            ax.set_title('y.size=%s,batch=%s,min(iters)=%s,\n %s'%(y.shape[0],n_b,iters,paras))
        else:
            pass
#            ax.set_title('X.size=%s min(iters)=%s'%(X.shape,iters))
        ax.legend(loc=0)
        ax.grid(True)

#    return pfinals,w_finals


def plot_gd_contour(J,wws,ess,pgrid,skwargs,B):
    wstart,wend = wws[0][0],wws[0][-1]
    rs = wstart-2
    re = wstart+2*np.abs(wend-wstart)

    r_n = 10 # 等高线光滑
    arx = np.linspace(rs[0],re[0],r_n);arx
    ary = np.linspace(rs[1],re[1],r_n);ary

#    arx = np.linspace(-1,1,r_n);arx
#    ary = np.linspace(-1,1,r_n);ary

    P,Q = np.meshgrid(arx,ary);P
    T = np.hstack([P.ravel().reshape(-1,1),
                   Q.ravel().reshape(-1,1)]);T
    V = np.array([J(t) for t in T]);V
    Z = V.reshape(r_n,-1);Z

    ##########

    from matplotlib import cm,ticker
    import matplotlib.pyplot as plt
    plt.figure(figsize = (10,10))
    cs = plt.contour(P,Q,Z,
                    #locator=ticker.LogLocator(),
                     levels=15,cmap=cm.jet)
    plt.clabel(cs, inline=1, fmt = '%.2E',fontsize=10)
    plt.colorbar()

    #########

    import matplotlib.colors as mcolors

    #colors=list(mcolors.BASE_COLORS.keys())
    colors= \
    list(mcolors.BASE_COLORS.keys())+ \
    list(mcolors.TABLEAU_COLORS.keys())
    colors.remove('w')

    for j,(ww,es,pg) in enumerate(zip(wws,ess,pgrid)):
        color = colors[np.mod(np.random.randint(j,100),len(colors))]
        w0,w1 = ww[:,0],ww[:,1]

        pg['method'] = fname_dic.get(pg['method'],'')

        # add 箭头
        for i in range(len(ww)):
            if i == 0:continue
            plt.annotate('', xy=(ww[i]),xytext=(ww[i-1]),
                        arrowprops= dict(
    #                            width=1,
    #                            arrowstyle='fancy',
                                arrowstyle='->',
                                         color=color),
                        va='center', ha='center')

        plt.plot(w0,w1,'--',color=color,linewidth=2,label=pg)# 线段用于legend区分
        plt.plot(w0[0],w1[0],'o',color=color,markersize=10)
        plt.plot(w0[-1],w1[-1],'o',color=color,markersize=10)

    plt.xlabel('w0');plt.ylabel('w1')

    # add 终点
    plt.annotate('Root', xy=(B),xytext=(B+.5),
                        arrowprops= dict(arrowstyle='fancy', color='r'),
                        va='center', ha='center')


    #plt.title('Gradient descent: Root at %.3f %.3f'%(w0[-1],w1[-1]))
    plt.title('Gradient Descent of J(w)\n paras %s'%skwargs)
    plt.legend()
    plt.show()

fname_dic = {
        'mm10' : 'GD',
        'mm51' : 'Ridge',
        'mm52' : 'Lasso',
        'mm53' : 'elnet',
        'mm54' : 'ADMM F1',
        'mm55' : 'FISTA F1',
        'mm91' : 'R-Cord Wj(w) each w[j]',
        'mm92' : 'S-Cord random w[j]',
        'mm90' : 'R-Cord gFj step descent',
        'mm93' : 'S-R-Block-Cord',
        'mm94' : 'S-S-Block-Cord',
        'mm21' : 'Polyak-M-H', # Polyak’s Momentum Head
        'mm22' : 'Polyak-M-B',
        'mm23' : 'NAG-H',
        'mm24' : 'NAG-B',
        'mm25' : 'NAG-H2',
        'mm30' : 'Newton',
        'mm31' : 'Minimizing Along a Line',
        'mm32' : 'Conjugate Gradient',
        'mm33' : 'Quasi-Newton(Broyden)',
        'mm40' : 'Adagrad',
        'mm41' : 'RMSProp',
        'mm42' : 'Adadelta',
        'mm43' : 'Adam',
        'mm44' : 'AdaMax',
        'mm45' : 'Nadam',
        'mm46' : 'AMSGrad',
    }