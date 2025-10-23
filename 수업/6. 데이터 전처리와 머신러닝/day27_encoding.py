import pandas as pd 
import numpy as np
dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')
y = dat.grade
X = dat.drop(['grade'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

#from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
train_X6 = train_X.copy()
test_X6 = test_X.copy()
train_X6_cat = train_X6.select_dtypes('object')
test_X6_cat = test_X6.select_dtypes('object')
# 라벨인코딩은 1차원 배열 혹은 series 형식을 받고
# ordinal 인코딩은 2차원 배열 혹은 DF 형식을 받음

ordinalencoder = OrdinalEncoder().set_output(transform = 'pandas')
train_X6_cat = ordinalencoder.fit_transform(train_X6_cat)
test_X6_cat = ordinalencoder.transform(test_X6_cat)
print(train_X6_cat.head(2))

# 옵션 (특정 변수가 test에만 있을 때)
# 훈련 데이터
train_data = pd.DataFrame({
    'job': ['Doctor', 'Engineer', 'Teacher', 'Nurse']
})
# 테스트 데이터
test_data = pd.DataFrame({
    'job': ['Doctor', 'Lawyer', 'Teacher', 'Scientist']
})
# OrdinalEncoder 설정
oe = OrdinalEncoder(handle_unknown='use_encoded_value', # 학습되지 않은 카테고리를 처리할 때 오류 발생 X
                    unknown_value=-1) # 이 값으로 반환

# 훈련 데이터로 인코더 학습
oe.fit(train_data[['job']])
# 훈련 데이터 변환
train_data['job_encoded'] = oe.transform(train_data[['job']])
# 테스트 데이터 변환 (훈련 데이터에 없는 직업은 -1로 인코딩됨)
test_data['job_encoded'] = oe.transform(test_data[['job']])