# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:12:20 2019

@author: autol
"""
#%%
from depends import *

#%% Matrix 1
n,m=5,3
A = sparse.random(n,m, density=.8,
                  data_rvs=stats.randint(1,6).rvs).toarray()
print(A)
#%% Matrix 2
n,m = (18,15)
dnan = round(.7*n*m)
A = np.random.randint(1,5,(n,m)).astype(np.float)
A.ravel()[np.random.choice(A.size,dnan,replace=False)] = np.nan
print(A)
#%% Matrix 3
# [soft impute referense](https://cran.r-project.org/web/packages/softImpute/vignettes/softImpute.html)
A = np.array([[0.8654889,0.01565179,0.1747903,np.nan, np.nan],
        [-0.6004172,np.nan,-0.2119090,np.nan,np.nan],
        [-0.7169292,np.nan, np.nan,0.06437356,-0.09754133],
        [0.6965558,-0.50331812,0.5584839 ,1.54375663 ,np.nan],
        [1.2311610,-0.34232368,-0.8102688 ,-0.82006429 ,-0.13256942],
        [0.2664415,0.14486388,np.nan,np.nan, -2.24087863]
        ])
print(A)

#%%
@accepts(func=str)
def SoftImpute(
                method='svda',
                n_iters=1,
                **kwargs):

    Δh = [] # save iters records
    X = kwargs.get('X',0).copy()
    k = kwargs.get('k',2)
    λ = kwargs.get('λ',0)
    isscale = kwargs.get('isscale',0)
    trace = kwargs.get('trace',1)
    final_svd = kwargs.get('final_svd',1)
    svdw = kwargs.get('svdw',0)

    def P(A,fill='zero'):
        # impute: "zero", "mean", "median", "min", "random"
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

    def Pc(X):
        return X-P(X)

    def Sλ(X,λ):
        return np.maximum(X-λ,0) # np.sign(X)

    def Frob1(U0,D0,V0,U,D,V): # from github
        denom = (D0 ** 2).sum()
        utu = D * (U.T.dot(U0))
        vtv = D0 * (V0.T.dot(V))
        uvprod = utu.dot(vtv).diagonal().sum()
        num = denom + (D**2).sum() - 2*uvprod
        return num/max(denom, 1e-9)

    def Frob2(Z,Z1): # from textbook
        E = Z-Z1
        a = np.trace(E.dot(E.T)) # (E**2).sum()
        b = np.trace(Z.dot(Z.T))
        return a/max(b, 1e-9)

    n,m = X.shape
    xnas = np.isnan(X)
    nz = n*m-np.sum(xnas)
    k = np.minimum(k,min(X.shape)-1)
    paras = ''

    def svdk(Z,k): # k sparse
         return sparse.linalg.svds(Z,k=k)
    def svd(Z): # full
         return np.linalg.svd(Z,full_matrices=0)
    def obj1(Z,Z1): # origin
        E = (Z-Z1)[~xnas]
        return 1./2*(E**2).sum()/nz
    def obj_(Z,Z1,D): # penalty
        E = (Z-Z1)[~xnas]
        return (1./2*(E**2).sum()+λ*D.sum())/nz

    def trace_print():
        if trace:
            print(i,":",'obj=',obj.round(5),"ratio=",ratio)
        Δh.append(np.hstack([i,obj,ratio]))

    ismean = kwargs.get('ismean',0)
    isstd = kwargs.get('isstd',0)


    X0 = X.copy()
    sc = ScaleX(X,isscale=isscale,
                ismean=ismean,
                isstd=isstd)
    X = sc.trans(X)

#    if isscale:
#        Z_mean = np.nanmean(X, axis=1)[:,np.newaxis]
#        X -= Z_mean

    if 'svda' == method:
        r = k
        paras = 'r='%r
        print('SVD approximate',paras)
#        rr = np.minimum(k,min(X.shape))
#        for r in range(rr+1):

        i = 0;ratio = 1
        Z = P(X.copy())

        Z0 = Z.copy()
        Z1 = 0
        u,s,vt = np.linalg.svd(Z,full_matrices=0)
        for i in range(r):
            Z1 += s[i] * u[:,i][:,np.newaxis].dot(vt[i][np.newaxis])
            Z[xnas] = Z1[xnas]
            ratio = Frob2(Z0,Z1)
            obj = obj1(Z,Z1)
            Z0 = Z1.copy()
            if trace:
                aa = 100*(i+1)/(r*100+12.)
                print(" %0.2f %%" %aa)
            trace_print()

    elif 'svds' == method:
        paras = 'k=%s,λ=%s'%(k,λ)
        print('Soft Impute SVD',paras)

        i = 0;ratio = 1
        Z = P(X.copy())
        Z0 = Z.copy()
        svdZ = svdk(Z,k)
        while ratio>1e-05 and i<n_iters:
            i += 1
            svdZ0 = svdZ
            D = Sλ(svdZ[1],λ)
            Z1 = svdZ[0].dot(np.diag(D)).dot(svdZ[2])
            Z[xnas] = Z1[xnas]
            svdZ = svdk(Z,k)
#                if flag:
            ratio = Frob1(svdZ0[0],D,svdZ0[2].T,
                          svdZ[0],Sλ(svdZ[1],λ),svdZ[2].T)
#                else:
#                    ratio = Frob2(Z0,Z1)
#                        Z0 = Z1.copy()
            obj = obj_(Z,Z1,D)
            trace_print()
    elif 'als' == method:
        paras = 'k=%s,λ=%s'%(k,λ)
        print('Soft Impute ALS',paras)

        i = 0;ratio = 1
        if svdw: # warm start
            Z = X.copy()
            #must have u,d and v components
#            J = min(sum(svdw[1]>0)+1,k)
            J = k
            D = svdw[1]
            JD = sum(D>0)
            print('JD=',JD,'J=',J)
            if JD >= J:
                U = svdw[0][:,:J]
                V = (svdw[2].T)[:,:J]
                Dsq = D[:J][:,np.newaxis]
            else:
                c = np.column_stack
                fill = np.repeat(D[JD-1],J-JD) # impute zeros with last value of D matrix
                Dsq = np.append(D,fill)[:,np.newaxis]
                Ja = J-JD
                U = svdw[0]
                Ua = np.random.normal(size=n*Ja).reshape(n,Ja)
                Ua = Ua - U @ U.T @ Ua
                Ua = svd(Ua)[0]
                U = c((U,Ua))
                V = c((svdw[2].T,np.repeat(0,m*Ja).reshape(m,Ja)))
            Z1 = U @ (Dsq*V.T)
            Z[xnas]=Z1[xnas]
#            print('Z=',Z.shape,'Z1=',Z1.shape)
        else: # cool start
            Z = P(X.copy())
#            k = min(sum(svd(Z)[1]>0)+1,k)
            V = np.zeros((m,k))
            U = np.random.normal(size=n*k).reshape(n,k)
            U = svd(U)[0]
            Dsq = np.repeat(1,k)[:,np.newaxis]
#            print('Z=',Z.shape,'u=',U.shape,'dsq=',Dsq.shape,'vt=',V.T.shape)
        while ratio>1e-05 and i<n_iters:
            i += 1
            U0,Dsq0,V0=U,Dsq,V
            # U step
            B = U.T @ Z
            if λ>0:
                B = B*(Dsq/(Dsq+λ))
            Bsvd = svd(B.T)
            V = Bsvd[0]
            Dsq = Bsvd[1][:,np.newaxis]
            U = U @ Bsvd[2].T
            Z1 = U @ (Dsq*V.T)
            Z[xnas] = Z1[xnas]
            obj = obj_(Z,Z1,Dsq)
            # V step
            A = (Z @ V).T
            if λ>0:
                A = A *(Dsq/(Dsq+λ))
            Asvd = svd(A.T)
            U = Asvd[0]
            Dsq = Asvd[1][:,np.newaxis]
            V = V @ Asvd[2]
            Z1 = U @ (Dsq*V.T)
            Z[xnas] = Z1[xnas]
            ratio = Frob1(U0,Dsq0,V0,
                          U,Dsq,V)
            trace_print()
        if i==n_iters:
            print("Convergence not achieved by",n_iters,"iterations")
        if λ>0 and final_svd:
            U = Z @ V
            sU = svd(U)
            U = sU[0]
            Dsq = sU[1]
            V = V @ sU[2].T
            Dsq = Sλ(Dsq,λ)
            if trace:
                Z1 = U @ (Dsq*V.T)
                print("final SVD:", "obj=",obj_(Z,Z1,Dsq).round(5),"\n")

#    if isscale: # centering
#        Z += Z_mean
    Z = sc.trans(Z,revert=1)

    finals = {'final':i,'obj':obj,'ratio':ratio}
    print(finals)
#    print('X0= \n',X0)
#    print('λ=',λ,', X.filed= \n',Z)
    w_h = pd.DataFrame(Δh,columns=['iters','objF','ratio'])
    return {'w_h':w_h,'Z':Z,'finals':finals,'paras':paras}


#%%

pgrid = list(ParameterGrid({'method': [
#                'svda',
                'svds',
#                'als',
                    ],
                 'isscale': [1,2],
                 'ismean': [0,1],
                 'isstd': [0,1],
                 'k':[2],
                 }))

pgrid.sort(key=itemgetter('method','isscale'),reverse=1)

finalss = iters_gd_plot(
        X=A.copy(),
        objf='ratio',#'ratio'
        pgrid = pgrid,
        func = SoftImpute,
        trace = 0,
        n_iters=100,
        islogy=1,
        doplot=1,
        )

finalss.sort(key=itemgetter('final'),reverse=1)
finalss

#%%
clf = SoftImpute(n_iters=100,
           trace=0,
           isscale=0,
           method='svds',
           X=A.copy(),
           )


#%%

ff=[
    'svda',
    'svds',
    'als',
    ]


iters_gd_plot(
        X=A.copy(),
        k=2,
        func = SoftImpute,
        ff=ff,
        trace=1,
        n_iters=100,
        islogy=True,
        )

#%%
#%timeit -n1 -r1

SoftImpute(A.copy(),k=5,λs=[1],n_iters=100,
           svdw=np.linalg.svd(P(A),full_matrices=0),
           flag=1,trace=True,func='als',final_svd=False)


#%%
SoftImpute(A.copy(),k=7,λs=[1.9],n_iters=100,svdw=0,
           flag=1,trace=1,func='als',final_svd=0)


#%%
SoftImpute(A.copy(),λs=[0],n_iters=100,
           flag=1,k=2,trace=1,func='svda')


SoftImpute(A.copy(),λs=[0],n_iters=100,
           flag=0,k=2,trace=1,func='svda')



#%%
SoftImpute(A.copy(),λs=[0],n_iters=100,
           flag=1,trace=1,func='svds',final_svd=0)




#%%

print('''
推荐系统测试:
Movelens 1m dataset(ml-1m)
''')

#%%

def SVDs(R,k=6):
    R_Umean = np.mean(R, axis = 1)[:,np.newaxis]
    R = R - R_Umean
    U, S, Vt = sparse.linalg.svds(R,k)
    R = U.dot(np.diag(S)).dot(Vt)
    return R + R_Umean

def R_df_(R_df,k):
    R,cols = R_df.values.astype(np.float),R_df.columns
    R = SVDs(R,k) if k>0 else R
    return pd.DataFrame(R, columns = cols).set_index(R_df.index)

#%%
def Recommend_movies(R_df, userID, movies_df,
                     ratings_df, n_movies=5,
                     hide_ranks = False):

    u_ranks = R_df.iloc[userID-1].sort_values(ascending=False)
    u_ranks = u_ranks[u_ranks!=0]

    u_ratings = ratings_df[ratings_df.UserID == (userID)]

    m_in_lib = (u_ratings.merge(movies_df, how = 'inner',on = 'MovieID').
      sort_values(['Rating'], ascending=False))

    a = movies_df[~movies_df['MovieID'].isin(m_in_lib['MovieID'])] if hide_ranks else movies_df

    b = pd.DataFrame(u_ranks).reset_index().apply(pd.to_numeric).rename(columns = {userID: 'Scores'})


    print('用户ID=%s的R矩阵有%s部,已经评分电影%s部,评了在库电影%s\
    部,库剩余%s部电影,打算推荐电影%s部'%(
                                userID,
                                b.shape[0],
                                u_ratings.shape[0],
                                m_in_lib.shape[0],
                                a.shape[0],
                                n_movies))

    df = (a.merge(b,how = 'left',on='MovieID').
          sort_values('Scores',ascending = False).reset_index(drop=True))

    #    df = (
    #            movies_df[~movies_df['MovieID'].isin(u_ratings_m['MovieID'])].
    #            merge(pd.DataFrame(u_ranks).reset_index().apply(pd.to_numeric),
    #                  how = 'left',left_on = 'MovieID',right_on = 'MovieID').
    #            rename(columns = {user_row_number: 'Scores'}).
    #            sort_values('Scores', ascending = False).
    #            iloc[:n_movies,]#:-1
    #        )
    return df.head(n_movies)
#%%
ratings_list = [i.strip().split("::") for i in open('./ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('./ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('./ml-1m/movies.txt', 'r',
               encoding='utf-8').readlines() if i.strip()]

#%%
ratings_df = pd.DataFrame(ratings_list,
                          columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']).astype(np.int)

movies_df = pd.DataFrame(movies_list,
                         columns = ['MovieID', 'Title', 'Genres']).drop(['Genres'],axis=1)
movies_df['MovieID'] = movies_df['MovieID'].astype(np.int)

df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
n,p = df.shape
scale = 20
R_df0 = df.iloc[:n//scale,:p//scale]
R_df0.shape

#%%

ks = np.array([0,10])
#ks = np.arange(0,50,10)
#ks = np.array([0,1,25,50])

for k in ks:
    R_df = R_df_(R_df0,k)
    userID = R_df0.shape[0]
    n_movies=20
    df_10 = Recommend_movies(R_df,userID, movies_df,
                             ratings_df,n_movies,
                             hide_ranks = False)
    print(df_10)
#R_df=R_df_(R_df0,0)


#%%

