import numpy as np
import pandas as pd
import matplotlib as plt

# 1 ##
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_1.csv')
df['합'] = df['금액1'] + df['금액2']
df_1 = df.groupby(['gender','지역코드'])['합'].sum().reset_index()
x = df_1.pivot_table(index='지역코드',
                        columns='gender',
                        values='합').fillna(0)
x['차'] = (x[0]-x[1]).abs().sort_values()
x['차'].idxmax()

df.loc[df['지역코드']==143,['gender','금액1','금액2']]

# 2
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv')
long_df = df.melt(id_vars=['연도','구분'],
                 var_name='유형',
                 value_name='건수'
                 )
wide_df = long_df.pivot_table(index='연도',
                              columns=['구분','유형'],
                              values='건수'
                              )
for i in range(10):
    if i == 1:
        wide_df['검거율10'] = wide_df['검거건수'].iloc[:,i]/wide_df['발생건수'].iloc[:,i]
    elif i == 0:
        wide_df['검거율1'] = wide_df['검거건수'].iloc[:,i]/wide_df['발생건수'].iloc[:,i]
    else:
        wide_df['검거율'+str(i)] = wide_df['검거건수'].iloc[:,i]/wide_df['발생건수'].iloc[:,i]
wide_df.head()
n = 0
for j in range(10):
    n += wide_df['검거건수'].loc[wide_df['검거율'+str(j+1)] == 1, '범죄유형'+str(j+1)].sum()

# 2-2
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv')
df1 = df.iloc[::2,:].reset_index()
df2 = df.iloc[1::2,:].reset_index()
df3 = df2.iloc[:,3:]/df1.iloc[:,3:]
df4 = df2.iloc[:,3:]
df4[df3 == 1].sum().sum()

# 3 
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_3.csv')
df['평균만족도'].fillna(df['평균만족도'].mean(), inplace=True)
df['근속연수'] = df['근속연수'].fillna(
    df.groupby(['부서', '등급'])['근속연수'].transform('mean').astype('int64'))
df.loc[(df['부서']=='HR')&(df['등급']=='A'),'근속연수'].mean()
df.loc[(df['부서']=='Sales')&(df['등급']=='B'),'교육참가횟수'].mean()

# 4
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_1.csv')
df.groupby('대륙')['맥주소비량'].mean()
df.loc[df['대륙']=='SA',['국가','맥주소비량']].groupby('국가')['맥주소비량'].sum().sort_values()
df.loc[df['국가']=='Venezuela','맥주소비량'].mean()

# 5
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_2.csv')
df['합계']=df.sum(axis=1, numeric_only=True)
a = df.groupby('국가',as_index=False)['합계'].sum()
b = df.groupby('국가',as_index=False)['관광'].sum()
c = df.groupby('국가',as_index=False)['공무'].mean()

d = pd.merge(a,b,on='국가')
d['관광율'] = d['관광']/d['합계']
d.sort_values(by='관광율',ascending=False).head(3)
e = pd.merge(c,d,on='국가')
e.sort_values('관광',ascending=False).head(3)

# 6
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_3.csv')
df.head()
df['CO scale'] = (df['CO(GT)']-df["CO(GT)"].min())/(df['CO(GT)'].max()-df["CO(GT)"].min())
df['N scale'] = (df['NMHC(GT)']-df["NMHC(GT)"].min())/(df['NMHC(GT)'].max()-df["NMHC(GT)"].min())
df['CO scale'].std()
df['N scale'].std()


# 7 ## 초단위?
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_1.csv')
df.head()
df['신고일시'] = pd.to_datetime(df['신고일시'])
df['처리일시'] = pd.to_datetime(df['처리일시'])
df['처리시간'] = (df['처리일시'] - df['신고일시']).dt.total_seconds()  ###
df.groupby('공장명')['처리시간'].mean()

# 8 
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_2.csv')
df.head()
df['구'] = df['STATION_ADDR1'].str.extract(r'([가-힣]+구)')
df_dist = df.loc[(df['구']=='마포구')|(df['구']=='성동구'),['구','dist']]
df_dist.groupby('구').mean()

# 9 ##
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_3.csv')
df['합계']=df.sum(axis=1,numeric_only=True)
x = []
for i in range(8):
    x.append(df.iloc[3*i:3*(i+1),6].mean(axis=0))
