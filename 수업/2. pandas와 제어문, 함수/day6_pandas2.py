import numpy as np
import pandas as pd

df = pd.read_csv('./data/penguins.csv')
df.info()

# 복습
df.describe()
df.sort_values('bill_length_mm')
result = df.groupby('species')['flipper_length_mm'].mean()
# df['flipper_length_mm']와 다른점은, index정보가 달라지는 것
result.index
result.values
result.values.argmax()
# argmax()는 numpy에서 사용하는, 최대값의 위치를 찾아주는 mathod,
# result.values는 numpy 벡터라 argmax() 사용
result.index[result.values.argmax()]

# .idxmax()/.idxmin()
# 시리즈에서 최대 or 최소값을 가지는 첫번째 index를 반환
result.idxmax() # result 자체는 pandas series

# pd.merge() : 두 데이터프레임을 병합, 기본적으로 공통 열을 기준으로 병합함
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
merged_df = pd.merge(df1,df2,on='key',how='inner')
# on= : 병합할 때 기준이 되는 열을 지정. 여기서는 key 열을 기준으로 병합
# how='' : 병합 방법을 지정. inner는 교집합, outer는 합집합. 기본은 inner
         # left를 입력하면 첫번째 데이터 프레임의 모든 키를 기준으로 병합
# 활용
mid_df1 = pd.DataFrame({'std_id': [1,2,4,5], 'score': [10, 20, 30, 40]})
final_df2 = pd.DataFrame({'student_id': [1,2,3,6], 'score': [4, 5, 7, 2]})
all_df = pd.merge(mid_df1,final_df2,on='std_id',how='outer')
# 만약 합치고싶은 데이터의 컬럼 이름이 다를 때
all_df = pd.merge(mid_df1,final_df2,
                  left_on='std_id', right_on='student_id',
                  how='outer') 
pd.merge(mid_df1,final_df2,on='std_id',how='left')

# 데이터 재구조화
# pd.melt() : 데이터를 wide form에서 long form으로 변환
wide_df = pd.DataFrame({
    '학생' : ['철수','영희','민수'],
    '수학' : [90,80,70],
    '영어' : [85,95,75]
})
long_df = wide_df.melt(id_vars='학생', # 변하지 않는 칼럼 설정
                       var_name='과목', # variable한 값들의 이름을 설정
                       value_name='점수') # 값들의 이름을 설정
w_df = pd.DataFrame({
    '반' : ['A','B','C'],
    '1월' : [20,18,22],
    '2월' : [19,20,21],
    '3월' : [21,17,23]
})
w_df.melt(id_vars='반',var_name='월',value_name='출석일수')

w_df1 = pd.DataFrame({
    '학년' : [1,1,2],
    '반' : ['A','B','C'],
    '1월' : [20,18,22],
    '2월' : [19,20,21],
    '3월' : [21,17,23]
})
w_df1.melt(id_vars=['반','학년'], # 두개를 변하지 않게 하겠다는 의미
           var_name='월',
           value_name='출석일수')
# submelt()
w_df2 = pd.DataFrame({
    '학생' : ['철수','영희','민수'],
    '국어' : [90,80,85],
    '수학' : [70,90,75],
    '영어' : [88,92,79],
    '학급' : ['1반','2반','3반']
})
w_df2.melt(id_vars=['학급','학생'],
var_name='언어과목',
value_vars=['국어','영어'], # 행중에 국어, 영어만 뽑겠다는 의미
value_name='성적')

# pivot_table() : long term을 wide term으로 변경
result_df =long_df.pivot_table(
    index='학생',
    columns='과목',
    values='점수'
).reset_index() # index를 행으로 변경
result_df.columns.name = None # column 명 지우기

pivot_result = long_df.pivot(
    index='학생',
    columns='과목',
    values='점수'
).reset_index() 
pivot_result.columns.name = None # 컬럼명 삭제
