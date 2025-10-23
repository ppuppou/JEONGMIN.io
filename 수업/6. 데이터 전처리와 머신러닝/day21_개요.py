import pandas as pd 
import numpy as np

dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')

# 단순 랜덤샘플링
y = dat.grade
X = dat.drop(['grade'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X, y, test_size = 0.2, # test 데이터의 비율
                   random_state = 0, # 결과 재현을 위한 임의의 고정값
                   shuffle = True, # 데이터 분할 전 데이터를 섞을지 여부(default : True)
                   stratify = None # 층화 샘플링 여부(default : None)
                   )

import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout(); 
plt.show();

# 층화샘플링
train_X, test_X, train_y, test_y = train_test_split(
                   X, y, test_size = 0.2, stratify= X['school'],
                   random_state = 0
                   )
X['school'].value_counts()
fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout(); 
plt.show();


# 결측치 채우기 
from sklearn.impute import SimpleImputer
train_X3 = train_X.copy()
test_X3 = test_X.copy()
imputer_mode = SimpleImputer(strategy = 'most_frequent') # 최빈값
train_X3['goout'] = imputer_mode.fit_transform(train_X3[['goout']])
test_X3['goout'] = imputer_mode.transform(test_X3[['goout']])
print('학습 데이터 goout 변수 결측치 확인 :', train_X3['goout'].isna().sum())

# KNN
train_X5 = train_X.copy()
test_X5 = test_X.copy()
train_X5_num = train_X5.select_dtypes('number')
test_X5_num = test_X5.select_dtypes('number')
train_X5_cat = train_X5.select_dtypes('object')
test_X5_cat = test_X5.select_dtypes('object')
from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5)
train_X5_num_imputed = knnimputer.fit_transform(train_X5_num)
test_X5_num_imputed = knnimputer.transform(test_X5_num)
                       
train_X5_num_imputed = pd.DataFrame(train_X5_num_imputed, 
                                    columns=train_X5_num.columns, 
                                    index = train_X5.index)
test_X5_num_imputed = pd.DataFrame(test_X5_num_imputed, 
                                   columns=test_X5_num.columns, 
                                   index = test_X5.index)
train_X5 = pd.concat([train_X5_cat, train_X5_num_imputed], axis = 1)
test_X5 = pd.concat([test_X5_cat, test_X5_num_imputed], axis = 1)
print('학습 데이터 goout 변수 결측치 확인 :', train_X5['goout'].isna().sum())
knnimputer2 = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')
train_X5_num_imputed2 = knnimputer2.fit_transform(train_X5_num)
test_X5_num_imputed2 = knnimputer2.transform(test_X5_num)
# 판다스 데이터프레임 출력 
print(train_X5_num_imputed2.head())


# 정규화 (분포변화)
from sklearn.preprocessing import PowerTransformer
import warnings
np.warnings = warnings
bike_data = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv")
box_tr = PowerTransformer()
bike_data['count_boxcox'] = box_tr.fit_transform(bike_data[['count']])
print('lambda : ', box_tr.lambdas_)
bike_data['count_log'] = np.log1p(bike_data[['count']])
bike_data['count_sqrt'] = np.sqrt(bike_data[['count']])
bike_data[['count', 'count_boxcox', 'count_log', 'count_sqrt']].hist();
plt.show();