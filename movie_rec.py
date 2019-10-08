# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:47:30 2019

@author: autol
"""

#%%

print('''
      基于用户评分的推荐系统测试
      Movelens 1m dataset(ml-1m)
''')

import pandas as pd
import numpy as np
from matrix_fun import svdk,Prox,obj2,Frob1
import time

#%% 读取数据文件
'''
- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
'''
ratings_list = [i.strip().split("::") for i in open('./ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('./ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('./ml-1m/movies.txt', 'r',
               encoding='utf-8').readlines() if i.strip()]
#%% 转换为DataFrame

# 仅仅基于评分，没使用用户信息
ratings_df = pd.DataFrame(ratings_list,
                          columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']).astype(np.int)
movies_df = pd.DataFrame(movies_list,
                         columns = ['MovieID', 'Title', 'Genres']).drop(['Genres'],axis=1)
movies_df['MovieID'] = movies_df['MovieID'].astype(np.int)

df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(np.nan)  # .fillna(0)

# 评分矩阵大小 users * movies 6040 * 3706 , 电影维度不一样原因是，电影库可能没收录
df.shape

#%%

# 截取大小 使用抽样方法
ui_size = 1000
rstat = 232
dfs = df.sample(n=ui_size, replace=0, random_state=rstat) \
        .sample(n=ui_size, axis=1, random_state=rstat) # frac=1./10,
dfs.shape

# 使用全部数据
#dfs = df.copy()

def toR_df(R,dfs):
    df = pd.DataFrame(R, columns=dfs.columns).set_index(dfs.index)
    return df

#%% 方法1

k = 2
R = dfs.copy()
time1 = time.time()
Z = R.fillna(R.mean(axis=0)).fillna(0)
Z = Z.values;Z
X = R.values
xnas = np.isnan(X)
U, d, Vt = np.linalg.svd(Z,full_matrices=0)
U, d, Vt = U[:,:k],d[:k],Vt[:k,:]
(U, d, Vt)
Z = U @ np.diag(d) @ Vt;Z
#X[xnas] = Z[xnas]
time2 = time.time()
print('All Running time: %s Seconds'%(time2-time1))
R_df = toR_df(Z,dfs)
R_df.shape


#%% 方法2

λ = 1
k = 2
R = dfs.copy()
X = R.values
Z = R.fillna(R.mean(axis=0)).fillna(0)
Z = Z.values;Z

xnas = np.isnan(X) ; n,m = X.shape
nz = n*m-xnas.sum()

time1 = time.time()
svdZ = svdk(Z,k)
for i in range(20):
    svdZ0 = svdZ
    d = Prox(svdZ[1],λ) # 近端投影软阈值
    Z1 = svdZ[0] @ np.diag(d) @ svdZ[2]
    Z[xnas] = Z1[xnas]
    svdZ = svdk(Z,k)
    d1 = Prox(svdZ[1],λ)
    obj = obj2(Z,Z1,xnas,nz,d1,λ)
    ratio = Frob1(svdZ0[0],d,svdZ0[2].T, svdZ[0],d1,svdZ[2].T)
    dicts = dict(ik=i,obj='%.3e'%obj,ratio='%.3e'%ratio)
    print(dicts)

time2 = time.time()
print('All Running time: %s Seconds'%(time2-time1))
R_df = toR_df(Z,dfs)
R_df.shape

#%%

#%%
userID = int(dfs.sample(n=1).index.values) # ,random_state=435
print('用户',userID) # 随便抽一个用户
rm,mse = recommend_movies(R_df, userID, movies_df,ratings_df)

#%%

def recommend_movies(R_df, userID, movies_df,ratings_df,
                     n_movies=20,  hide_ranks = 0):

    u_ranks = R_df.loc[userID].sort_values(ascending=False) # one user ranks
    u_ranks = u_ranks[u_ranks!=0]
    u_ranks.shape
    u_ratings = ratings_df[ratings_df.UserID == userID] # find this user records in all ratings
    u_ratings.shape
    u_m_ratings = u_ratings.merge(movies_df, how = 'inner',on = 'MovieID').sort_values(['Rating'], ascending=False)
    u_m_ratings.shape

#    print(u_m_ratings[['Title', 'Rating']][:3])
    M = movies_df
#    if hide_ranks:
#        M = movies_df[~movies_df['MovieID'].isin(u_m_ratings['MovieID'])] # new except movies_df

    # 指定用户的电影评分重新索引
    u_ranks = pd.DataFrame(u_ranks).reset_index().apply(pd.to_numeric).rename(columns = {userID: 'Ranks'})
    u_ranks.shape

    u_m_ranks = (M.merge(u_ranks,how = 'left',on='MovieID').
          sort_values('Ranks',ascending = False).reset_index(drop=True)).dropna()
    u_m_ranks.shape

    u_m_rranks = u_m_ratings[['MovieID','Rating']].merge(u_m_ranks,how = 'outer',on='MovieID')[['MovieID','Title','Rating', 'Ranks']] #[:20]
    u_m_rranks = u_m_rranks.dropna(subset=['Ranks'])

    u_m_rranks.shape

    RR_old = u_m_rranks[u_m_rranks['MovieID'].isin(u_m_ratings['MovieID'])]
    # 用于过滤已评分的电影
    RR_new = u_m_rranks[~u_m_rranks['MovieID'].isin(u_m_ratings['MovieID'])]

    D = RR_old[['Rating','Ranks']].values

    E = RR_new[['Title','Ranks']].values[:n_movies]
    print('ratings: \n ',E)

    a,b = D[:,0],D[:,1]
    mse = np.linalg.norm(a-b)**2/len(a)
    print('与旧评分对比的 MSE:',mse)
    return E,mse

#%%


#%% 读写取评分矩阵
#  R_df.to_csv('svdk.csv',header=1, index=1)

R_df = pd.read_csv('svdk.csv',index_col=0)
R_df.columns.set_names('MovieID',inplace=True)
R_df.index
R_df.columns
