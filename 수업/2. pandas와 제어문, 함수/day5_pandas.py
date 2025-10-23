import pandas as pd
import numpy as np
 
# data frame
df = pd.DataFrame({
    'col1': ['one', 'two', 'three', 'four', 'five'],
    'col2': [6, 7, 8, 9, 10]
})
print(df)
df['col1']
# key는 칼럼의 이름, value는 칼럼의 값
df.shape
# data frame은 값의 type이 달라도 괜찮음
df['col1']
# key를 이용해 value를 찾아내는 것이 마치 dictionary type과 유사함

# series : 1차원구조. data frame이 여러 series의 조합이라고 볼 수 있다
data = [10, 20, 30]
df_s = pd.Series(data, index=['one', 'two', 'three'], 
                 name = 'count')
print(df_s)

# 데이터 프레임 채우면서 만들기
my_df = pd.DataFrame({
    'name': ['issac', 'bomi'],
    'birthmonth': [5, 4]
})
print(my_df)
my_df.info()

# dataframe에서의 indexing
mydata = pd.read_csv("https://bit.ly/examscore-csv")
print(mydata.head())
mydata.shape
mydata.columns
mydata['gender'].head(10)
mydata['gender'].tail() # 기본은 5개
mydata[['gender','midterm']].head() # 리스트로 여러개도 가능
mydata[['midterm','final']].head()
mydata[mydata['midterm']>15].shape
# iloc[] : 숫자를 사용하는 indexing
mydata.iloc[:,1].head()
mydata.iloc[1:5,2]
mydata.iloc[1:4,0:3]
mydata.iloc[:,1].head() # iloc으로 행을 뽑으면 series로 출력됨
mydata.iloc[:,[1,0,1]].head()
mydata.iloc[1:3,[1]].squeeze() # DF를 series로 변환
# loc[] : 숫자가 아닌 문자로 indexing (라벨 indexing) * 행은 숫자 or bool
mydata.loc[:,'midterm']
mydata.loc[1:4,'midterm']  # 얘는 end값을 포함함 ;;
mydata.loc[mydata['midterm']<=15,['gender','student_id']]
# iloc은 무조건 숫자가 들어가야 해서 filtering이 불가능함. loc만 가능
# .isin([a]) : a 라는 내용이 DF 안에 있는지 찾아내는 mathod
mydata['midterm'].isin([28,38,52]) # True,False series로 나옴
mydata.loc[mydata['midterm'].isin([28,38,52]),['gender','final']]
check_inx = np.where(mydata['midterm'].isin([28,38,52]))[0]
# [0]은, np.where의 결과가 tuple로 나오기 때문
mydata.iloc[check_inx, [3,1]] # 굳이 iloc을 쓰겠다하면 이런식으로

# 일부 데이터를 NA로 설정
mydata.iloc[1, 1] = np.nan
mydata.iloc[4, 0] = np.nan
mydata.head()
mydata['gender'].isna().sum() # NA의 개수 세기
mydata.dropna() # NA가 포함된 행을 제거한 완전한 DF 얻어내기

# NA가 있는 행을 filtering하기
# 1 
mydata['student_id'].isna()
mydata['gender'].isna()
# 2
vec_2 = ~mydata['gender'].isna()
vec_3 = ~mydata['student_id'].isna()
# 3
vec_2 & vec_3
mydata[vec_2 & vec_3]
# 위 과정이 dropna(). 헛짓거리인듯

# 변경 및 추가
mydata['total'] = mydata['midterm'] + mydata['final'] # 인덱싱하듯
mydata.head()
mydata['average'] = (mydata['total']/2).rename('average')
mydata['ave^2'] = (mydata['average']**2)
del mydata['ave^2']
mydata.rename(columns={'student_id':'std_id'},inplace=True)
# inplace = True를 입력해야 원 데이터에 반영이 됨
# 이게 어려우면 새로운 데이터로 덮어씌우기
mydata

# pd.concat([]) : 여러 DF나 series를 합치는 함수
df1 = pd.DataFrame({
'A': ['A0', 'A1', 'A2'],
'B': ['B0', 'B1', 'B2']
})
df2 = pd.DataFrame({
'C': ['C4', 'C4', 'C5'],
'B': ['B3', 'B4', 'B5']
})
pd.concat([df1, df2])
pd.concat([df1, df2], axis=1) # axis=1 이면 수평으로 합침
pd.concat([df1, df2], ignore_index=True) # index를 하나로 정리

df4 = pd.DataFrame({
'A': ['A2', 'A3', 'A4'],
'B': ['B2', 'B3', 'B4'],
'C': ['C2', 'C3', 'C4']
})
pd.concat([df1, df4], join='inner') # 공통열만 포함해 결합
pd.concat([df1, df4], join='outer') # 모든 열을 결합


df = pd.read_csv('./data/penguins.csv')
df
# .describe() : 평균, 최대값 등 통계가 가능한 데이터의 요약을 반환
df.describe()

# .sort_values (by='열 이름') : 열 기준으로 오름차순 정렬
# .sort_values (by='열 이름', ascending=False) : 내림차순 정렬
df.sort_values(by='sex')
# 열을 리스트로 넣으면 두개를 기준으로 정렬 가능

# groupby : 특정 열을 기준으로 데이터 프레임 그룹화
df.groupby('species') # 종에 따라 dataframe 그룹화
df.groupby('species')['bill_length_mm'] # 판다스에 가상의 series 3개가 생김
df.groupby('species')['bill_length_mm'].mean()
df.groupby('species').mean(numeric_only=True) # 숫자만 계산한다는 뜻
df.mean(numeric_only=True)                    # 문자열 있는데 안쓰면 안되네요
