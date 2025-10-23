# 부트스트랩 예제 코드
import numpy as np

# 원 데이터
x = np.array([2, 4, 5, 7, 9])

n, B = len(x), 1000

# 부트스트랩 샘플링 (B x n 행렬)
samples = np.random.choice(x, size=(B, n), replace=True)

# 각 샘플의 평균
means = samples.mean(axis=1)
means

print("원 평균:", x.mean())
print("부트스트랩 평균의 평균:", means.mean())
print("부트스트랩 표준오차(SE):", means.std(ddof=1))
print("95% 신뢰구간:", np.percentile(means, [2.5, 97.5]))

# 시각화 코드
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.hist(means, bins=20, color="skyblue", edgecolor="k", alpha=0.7)
plt.axvline(x.mean(), color="red", linestyle="--", label="원 평균")
plt.xlabel("부트스트랩 평균")
plt.ylabel("빈도")
plt.title("Bootstrap distribution of the mean")
plt.legend()
plt.show()


import pandas as pd

train_df = pd.read_csv('../data/house/train.csv')
test_df = pd.read_csv('../data/house/test.csv')
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("SalePrice")
cat_columns = train_df.select_dtypes(include=['object']).columns
# 결측치 채우기 (간단히 처리)
# train_df = train_df.dropna()
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')
train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")
train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])
train_df_all = pd.concat([train_df_cat,
                          train_df_num], axis = 1)
X_train = train_df_all
y_train = train_df['SalePrice']

# import numpy as np
# nums = np.random.choice(np.arange(1460), 
#                         size=1460, 
#                         replace=True)
# X_train.iloc[nums, :]
# y_train[nums]

from sklearn.linear_model import ElasticNet
import numpy as np
elastic = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
# 파라미터 확인 
ElasticNet().get_params()
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)
# 그리드서치
elastic_search = GridSearchCV(estimator=elastic, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
elastic_search.fit(X_train, y_train)
pd.DataFrame(elastic_search.cv_results_)
# best prameter
print(elastic_search.best_params_)
print(-elastic_search.best_score_)

# elastic = ElasticNet(alpha=0.1, 
#                      l1_ratio=0.75)
# elastic.fit(X_train, y_train)
# elastic.coef_

from sklearn.ensemble import BaggingRegressor
base_model=elastic_search.best_estimator_
# 배깅
bagging = BaggingRegressor(
    estimator=base_model,
    n_estimators=100,    # 가상 데이터 100개
    bootstrap=True,      # 부트스트랩 샘플링
    random_state=42
)
bagging.fit(X_train, y_train)
# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])
test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])
test_df_all = pd.concat([test_df_cat,
                         test_df_num], axis = 1)
# 예측
# y_pred = elastic.predict(test_df)
y_pred = bagging.predict(test_df_all)
submit = pd.read_csv('../data/house/sample_submission.csv')
submit["SalePrice"]=np.expm1(y_pred)
