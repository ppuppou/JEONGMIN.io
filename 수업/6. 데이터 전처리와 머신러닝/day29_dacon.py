import pandas as pd
import numpy as np

train = pd.read_csv('../data/mission/train.csv')
test = pd.read_csv('../data/mission/test.csv')
sub = pd.read_csv('../data/mission/sample_submission.csv')
train = train.drop(columns='ID')
test = test.drop(columns='ID')

import matplotlib.pyplot as plt
import seaborn as sns

# 1. 상관계수 행렬 계산
correlation_matrix = train.corr()
# 2. 히트맵 시각화 (변수명 df -> train으로 변경)
plt.figure(figsize=(18, 15))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of the Dataset', fontsize=16)
plt.show() # 히트맵을 화면에 표시합니다.
# 3. 상관계수가 높은 순으로 정렬하여 출력
# 중복(A-B, B-A)과 자기 자신과의 상관(A-A)을 피하기 위한 처리입니다.

# 상관계수 행렬의 절대값을 취하고, 상삼각행렬(upper triangle)만 선택
sol = correlation_matrix.abs().unstack()
# 내림차순으로 정렬
sol = sol.sort_values(kind="quicksort", ascending=False)

print("상관계수가 높은 변수 쌍 (내림차순):")
# 자기 자신과의 상관관계(1.0)를 제외하고 상위 20개만 출력
print(sol[sol < 1].head(20))

X = train.drop(columns='target')
y = train.target
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler().set_output(transform='pandas')
X_scaled = scaler.fit_transform(X)
# PCA를 적용하여 차원을 축소합니다.
# n_components는 생성할 주성분의 수를 의미합니다.
# 0.95로 설정하면 원본 데이터 분산의 95%를 설명하는 주성분 개수를 자동으로 선택합니다.
pca = PCA(n_components=0.95).set_output(transform='pandas')
X_pca = pca.fit_transform(X_scaled)
# 결과 확인
print(f'원본 데이터의 피처 개수: {X_scaled.shape[1]}')
print(f'PCA로 축소된 피처 개수: {X_pca.shape[1]}')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_params = {"n_neighbors": np.arange(1, 19)}
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
# 그리드서치 
grid_knn = GridSearchCV(estimator=knn, param_grid=knn_params, cv = cv, 
                              scoring='neg_mean_squared_error')
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier()
dct_params = {'max_depth' : np.arange(1, 10),
              'ccp_alpha': np.linspace(0, 0.05, 10)}
grid_dct = GridSearchCV(estimator=dct, param_grid=dct_params, cv = cv, 
                              scoring='neg_mean_squared_error')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_params = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 0.5],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rand_rf = RandomizedSearchCV(estimator=rf, 
                             param_distributions=rf_params, 
                             n_iter=50, 
                             cv=cv, 
                             scoring='neg_mean_squared_error', 
                             random_state=2025, 
                             n_jobs=-1)
grid_dct.fit(X_pca,y)
grid_knn.fit(X_pca,y)
rand_rf.fit(X_pca,y)
rand_rf.best_score_
grid_dct.best_score_
grid_knn.best_score_

test_scaled = scaler.transform(test)
test_pca = pca.transform(test_scaled)
result = rand_rf.predict(test_pca)

sub.target = result.astype(int)
sub.to_csv('../data/mission/result.csv',index=False)



import pandas as pd
from lightgbm import LGBMClassifier

# ID 컬럼은 예측에 사용하지 않으므로 제외합니다.
train_X = train.drop(['ID', 'target'], axis=1)
train_y = train['target']
test_X = test.drop('ID', axis=1)
# 3. LightGBM 모델 정의 및 학습
# 높은 성능을 위해 일반적으로 사용되는 파라미터들을 설정합니다.
# n_estimators를 늘리면 성능이 오를 수 있지만 학습 시간이 길어집니다.
lgbm = LGBMClassifier(objective='multiclass', 
                      metric='multi_logloss',
                      n_estimators=500,
                      learning_rate=0.05,
                      num_leaves=31,
                      max_depth=-1,
                      random_state=2025,
                      n_jobs=-1,
                      colsample_bytree=0.8,
                      subsample=0.8)
lgbm.fit(train_X, train_y)

# 4. 테스트 데이터 예측
predictions = lgbm.predict(test_X)
# 5. 제출 파일 생성
sub['target'] = predictions

sub.to_csv('../data/mission/result.csv', index=False)
