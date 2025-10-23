import numpy as np
import pandas as pd
import matplotlib as plt

df1 = pd.read_csv('./test data/problem1.csv')
df1.head(2)
# 1
df1.shape
# 2 
df1.loc[df1['퇴거여부']=='미퇴거'].shape
# 3
apt_gen = df1.groupby(['아파트 이름','성별'])['보증금(원)'].mean().reset_index()
pivot_apt = apt_gen.pivot_table(
    index = '아파트 이름',
    columns='성별',
    values='보증금(원)'
)
(pivot_apt['남']-pivot_apt['여']).abs().sort_values(ascending=False)
# 4
df1.loc[df1['월세(원)'].idxmax(),['아파트 이름','월세(원)']].head(2)
# 5
df1.groupby('층')['거주자 수'].mean().sort_values(ascending=False)
# 6
good = df1.loc[df1['계약구분']=='유효']
good5 = good.loc[good['재계약횟수']>=5]
good5.head(2)
good5.groupby('평형대')['나이'].mean()
df1['평형대'].unique()
# 7
df1.groupby('계약자고유번호')['거주연도'].max()
# 8
df1[['아파트 평점','계약구분']].dropna()
# 9
df1.loc[df1['퇴거여부']=='미퇴거',['퇴거연도']].dropna()
# 10
df1['재계약횟수'].median() # 8
df1.loc[df1['재계약횟수']>=8,['거주개월']].mean()
df1.loc[df1['재계약횟수']<8,['거주개월']].mean()
# 11
df1.loc[df1['재계약횟수']>=8,['나이']].median()
df1.loc[df1['재계약횟수']<8,['나이']].median()
# 12
high = df1.loc[df1['재계약횟수']>=8,['성별','나이']].groupby('성별').count()
low = df1.loc[df1['재계약횟수']<8,['성별','나이']].groupby('성별').count()
high.iloc[1,0]/(high.iloc[0,0]+high.iloc[1,0])
low.iloc[1,0]/(low.iloc[0,0]+low.iloc[1,0])


df2 = pd.read_csv('./test data/problem2.csv')
# 13
df2.dropna().to_csv('./1.csv')


df2_1 = pd.read_csv('./test data/problem2_1.csv')
# 14
df2_1.iloc[:,-9:].mean(axis=0).sort_values(ascending=False)
# 15
first = df2_1.iloc[:,1:-9:2].reset_index()
second = df2_1.iloc[:,2:-9:2].reset_index()
first.rename(columns= {'a1_1':'a1_2','a2_1':'a2_2','a3_1':'a3_2','a4_1':'a4_2','a5_1':'a5_2','a6_1':'a6_2','a7_1':'a7_2','a8_1':'a8_2','a9_1':'a9_2'})
new1 = pd.DataFrame()
for i in range(1,10):
    new1[i] = (first.iloc[:,i] - second.iloc[:,i]).abs()
new1.mean(axis=0).mean()

# 16 못품

df2_2 = pd.read_csv('./test data/problem2_2.csv')
# 17
run = df2_2.loc[(df2_2['ining1_move']==1)|(df2_2['ining1_move']==2)|(df2_2['ining1_move']==3)|(df2_2['ining1_move']==6)|(df2_2['ining1_move']==8),:]
run.loc[run['ining2_move']!=4,:]

df2_3 = pd.read_csv('./test data/problem2_3.csv')
# 18
yes = df2_3.loc[df2_3["score"]!=0,['ining2_move']].mean()
no = df2_3.loc[df2_3["score"]==0,['ining2_move']].mean()
# 19
clssi = df2_3.groupby(['ining1_move','ining2_move'])['score'].max().reset_index()
(clssi['score']==9).sum()
df2_3.loc[(df2_3['ining1_move']==6)&(df2_3['ining2_move']==7),:]
# 20
(df2_3.groupby(['ining1_move','ining2_move'])
 ['score'].mean().reset_index().sort_values(by='score').tail())
a = df2_3.loc[(df2_3['ining1_move']==3)&(df2_3['ining2_move']==8),:]
b = df2_3.loc[(df2_3['ining1_move']==6)&(df2_3['ining2_move']==8),:]
c = df2_3.loc[(df2_3['ining1_move']==3)&(df2_3['ining2_move']==6),:]
d = df2_3.loc[(df2_3['ining1_move']==3)&(df2_3['ining2_move']==1),:]
e = df2_3.loc[(df2_3['ining1_move']==2)&(df2_3['ining2_move']==2),:]
len(a)+len(b)+len(c)+len(d)+len(e)