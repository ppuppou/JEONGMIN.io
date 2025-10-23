import pandas as pd
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)

df_melted = pd.melt(df, 
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='측정요소', 
                    value_name='측정값')
print(df_melted.head(6))

# 원래 형식으로 변환
df_pivoted = df_melted.pivot_table(index='Date', 
                             columns='측정요소', 
                             values='측정값').reset_index()
df_pivoted.columns.name = None
# 2024-07-03는 평균값이 나옴
# pivot은 index 값이 중복되면 에러생겨서 요즘은 잘 안쓴대요
df_pivoted = (
    df_melted.pivot_table(index='Date', 
                          columns='측정요소', 
                          values='측정값',
                          aggfunc='sum') # 중복이 있으면 합치겠다
                          ).reset_index()
df_pivoted.columns.name = None
# 중복된 날짜의 값을 평균이 아닌 다른 함수로 변경도 가능

df = pd.read_csv('../data/dat.csv')
df.columns
df.rename(columns = {'Dalc' : 'dalc', 'Walc' : 'walc'},inplace=True)
df.info()
# astype({}) : 데이터의 type을 변경
print(df.loc[:, ['famrel', 'dalc']]
      .astype({'famrel' : 'object', 'dalc' : 'float64'}).info())
# assign() : 새로운 칼럼을 생성하거나 특정 칼럼 값을 변경
def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    elif famrel <= 4:
        return 'Medium'
    else:
        return 'High'
df = df.assign(famrel = df['famrel'].apply(classify_famrel))
# famrel 컬럼에 classify 함수를 적용해라?
# 앞에 famrel에 다른 이름을 넣으면 새로운 컬럼이 추가됨

# select_dtypes() : 원하는 타입의 데이터만 추출
df.select_dtypes('int64')
import numpy as np
def standardize(x):
    return((x-np.nanmean(x))/np.std(x)) # nanmean : 결측치를 제외한 평균
                                        # std : 표준편차
vec_a = np.arange(5)
vec_a
standardize(vec_a)
df_std = df.select_dtypes('number').apply(standardize)
# 데이터 타입이 숫자인 것만 뽑아서 위 함수를 적용해달라
df_std.mean(axis=0) 
df_std.std(axis=0) 
## standardize 함수는 평균은 0, 표준편차는 1로 만들어주는 함수

# .str.startwith('a') : 'a'로 시작하는 값은 T, 아닌 값은 F
df.columns.str.startswith('f')
df.loc[:,df.columns.str.startswith('f')]
# endwith도 가능
# str.contains('a') : 'a'가 포함된 값은 T, 아니면 F

# csv로 저장하기
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
})
df.to_csv("data.csv", index=False) # csv로 저장, False는 인덱스를 제외

# 팔머펭귄 연습문제
# 팔머펭귄 데이터 불러오기
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins()
# 1 펭귄 종별 평균 부리길이 구하기
penguins.groupby('species')['bill_length_mm']
spe_bill = penguins.pivot_table(
    index= 'species',
    values= 'bill_length_mm',
    aggfunc='mean'
).reset_index()

# 2 섬별 몸무게 중앙값 구하기
penguins.groupby('island')['body_mass_g'].median()
is_body = penguins.pivot_table(
    index= 'island',
    values= 'body_mass_g',
    aggfunc='median'
).reset_index()

# 3 성별에 따른 부리길이와 몸무게 평균 구하기
penguins.groupby('sex')[['bill_length_mm','body_mass_g']].mean()
sex_bill_mass = penguins.pivot_table(
    index= 'sex',
    values= ['bill_length_mm','body_mass_g'],
    aggfunc='mean'
).reset_index()

# 4 종과 섬에 따른 평균 지느러미길이 구하기
penguins.groupby(['species','island'])['flipper_length_mm'].mean()
spe_is_flip = penguins.pivot_table(
    index= ['species','island'],
    values= 'flipper_length_mm',
    aggfunc='mean',
    dropna=False # 결측치도 포함하겠다
).reset_index()
# 4 -2 옆으로 나열 
spe_is_flip2 = penguins.pivot_table(
    index= 'species',
    columns= 'island',
    values= 'flipper_length_mm',
    aggfunc='mean',
    dropna=False,
    fill_value='없음' # NA를 없음으로 채우겠다
).reset_index()
spe_is_flip2.columns.name = None

# 5 종과 성별에 따른 부리 깊이 합계 구하기
penguins.groupby(['species','sex'])['bill_depth_mm'].sum()
spe_sex_bill = penguins.pivot_table(
    index= ['species','sex'],
    values= 'bill_depth_mm',
    aggfunc='sum'
).reset_index()

# 6 종별 몸무게의 변동 범위(Range) 구하기 # 다시보기
penguins.groupby('species')['body_mass_g'].max()-penguins.groupby('species')['body_mass_g'].min()
def Range(x):
    return x.max() -x.min()
spe_mass = penguins.pivot_table(
    index= 'species',
    values= 'body_mass_g',
    aggfunc=Range # 내가 만든 함수의 경우 따옴표를 생략
).reset_index() 
