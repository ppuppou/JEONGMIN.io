# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins

import numpy as np

penguins['body_mass_g']
vec_m = np.array(penguins['body_mass_g'])
vec_m.shape
vec_m.max()
vec_m.min()
vec_m.argmax() # 최대값이 어디에 있는가
vec_m.mean()
vec_m < 4200
sum(vec_m<4200)
sum(vec_m>=3000)

#day 5
import pandas as pd
import numpy as np
df = pd.read_csv('./data/penguins.csv')
df

# Q1 결측치가 하나라도 있는 행은 몇개인가요
a = df.iloc[:,2].isna()
b = df.iloc[:,3].isna()
c = df.iloc[:,4].isna()
d = df.iloc[:,5].isna()
a | b | c | d 
df.loc[a | b | c | d ,:].shape[0]

#다른 답변
df1 = df.iloc[:,2:6]
sum(np.sum(df1.isna(),axis = 1) >= 1)

(df.iloc[:,2:6].isna().any(axis=1)).sum()
# axis=1 은 열(수직), 0은 행(수평)

df.iloc[:,2:6].dropna().shape[0]

# Q2 몸무게가 4000 이상, 5000 이하인 펭귄은 몇마리인가요
mass = df.loc[(4000<=df['body_mass_g'])&(df['body_mass_g']<=5000),'body_mass_g'].shape[0]
f'몸무게가 4000이상, 5000이하인 펭귄은 총{mass}마리 입니다'
# and나 or는 숫자끼리만 되고  &, |는 다 되고

# 다른 답변
df['body_mass_g'].between(4000,5000).sum()

# Q3 펭귄 종 별로 평균 부리 길이가 어떻게 되나요
df['species'].unique()
Adelie = df.loc[df['species']=='Adelie','bill_length_mm']
Adelie.mean()
Chinstrap = df.loc[df['species']=='Chinstrap','bill_length_mm']
Chinstrap.mean()
Gentoo = df.loc[df['species']=='Gentoo','bill_length_mm']
Gentoo.mean()

# 다른 답변
df.groupby('species')['bill_length_mm'].mean()

# Q4 성별이 결측치가 아닌 데이터 중 성별 비율은 어떻게 되나요
non_na = df.loc[:,'sex'].dropna()
Male = df.loc[df['sex']=='Male','sex'].shape[0]
Female = df.loc[df['sex']=='Female','sex'].shape[0]
Male100 =(Male / (Male + Female))*100 
Female100 = (Female / (Male + Female))*100
f'성별이 남성인 펭귄은 {np.round(Male100, 3)}%, 여성인 펭귄은 {Female100}% 입니다'
# np.round(a, n) : a의 소숫점 이하를 n개로 자를 수 있다

# Q5 섬별로 평균 날개 길이가 가장 긴 섬은 어디인가요?
df['island'].unique()
Torgersen = df.loc[df['island']=='Torgersen','flipper_length_mm']
Biscoe = df.loc[df['island']=='Biscoe','flipper_length_mm']
Dream = df.loc[df['island']=='Dream','flipper_length_mm']
Torgersen.mean()
Biscoe.mean()
Dream.mean()
max([Torgersen.mean(),Biscoe.mean(),Dream.mean()])
means = {
    Torgersen.mean() : 'Torgersen',
    Biscoe.mean() : 'Biscoe',
    Dream.mean() : 'Dream'
}
max_means = max(means)
max_index = means[max_means]
f'평균 날개 길이가 가장 긴 섬은 {max_index}입니다'

df.describe()

# 다른 답변
mean_vec = df.groupby('island')['flipper_length_mm'].mean()
mean_vec.argmax() # 최대값의 위치
mean_vec.index[mean_vec.argmax()] 