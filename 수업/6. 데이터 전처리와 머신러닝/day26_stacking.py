import pandas as pd
import numpy as np

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('../data/house/train.csv')
test_df = pd.read_csv('../data/house/test.csv')

num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop("SalePrice")
cat_columns = train_df.select_dtypes(include=['object']).columns

from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])
train_df_all = pd.concat([train_df_cat,train_df_num], axis = 1)

# 독립변수(X)와 종속변수(y) 분리

X_train = train_df_all
y_train = np.log1p(train_df['SalePrice'])

from sklearn.linear_model import ElasticNet
elastic = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor()
dct_params = {'max_depth' : np.arange(1, 7),
              'ccp_alpha': np.linspace(0, 1, 10)}

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn_params = {"n_neighbors": np.arange(1, 6)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)

# 그리드서치 - ElasticNet
elastic_search = GridSearchCV(estimator=elastic, param_grid=elastic_params, cv = cv, 
                              scoring='neg_mean_squared_error')
elastic_search.fit(X_train, y_train)

# 그리드서치 - DCT
dct_search = GridSearchCV(estimator=dct, param_grid=dct_params, cv = cv, 
                          scoring='neg_mean_squared_error')
dct_search.fit(X_train, y_train)


# 그리드서치 - KNN 회귀
knn_search = GridSearchCV(estimator=knn, param_grid=knn_params, cv = cv, 
                          scoring='neg_mean_squared_error')
knn_search.fit(X_train, y_train)
# best prameter

print(elastic_search.best_params_)
print(dct_search.best_params_)
print(knn_search.best_params_)

test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])
test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])
test_df_all = pd.concat([test_df_cat,
                         test_df_num], axis = 1)

# 예측
y_pred1 = elastic_search.predict(test_df_all)
y_pred2 = dct_search.predict(test_df_all)
y_pred3 = knn_search.predict(test_df_all)
# 블렌딩: 단순하게 각 모델 출력값을 합침 (평균)
y_pred = (y_pred1 + y_pred2 + y_pred3) / 3

# 스택킹: 새로운 모델(메타모델)을 사용해서 합침(선형회귀)
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=False)
y_pred_el = elastic_search.predict(train_df_all)
y_pred_dct = dct_search.predict(train_df_all)
y_pred_knn = knn_search.predict(train_df_all)
X_meta = pd.DataFrame({
                      'elastic':y_pred_el,
                      'dct':y_pred_dct,
                      'knn':y_pred_knn,
                      })
lr.fit(X_meta,y_train)
lr.coef_
lr.intercept_

y_pred1 = elastic_search.predict(test_df_all)
y_pred2 = dct_search.predict(test_df_all)
y_pred3 = knn_search.predict(test_df_all)
X_meta_test = pd.DataFrame({
                      'elastic':y_pred1,
                      'dct':y_pred2,
                      'knn':y_pred3,
                      })
y_pred = lr.predict(X_meta_test)

submit = pd.read_csv('../data/house/sample_submission.csv')
submit["SalePrice"]=np.expm1(y_pred)

# =========================
# 기본 모델 정의 (디시전 트리와 라쏘)
base_models = [
    ('elastic', elastic_search.best_estimator_),
    ('decision_tree', dct_search.best_estimator_),
    ('knn', knn_search.best_estimator_)
]

# 메타 모델 정의 (선형 회귀)
meta_model = LinearRegression()

# 스태킹 리그레서 설정
from sklearn.ensemble import StackingRegressor
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=cv  # 크로스밸리데이션 폴드 수
)

# 모델 학습
stacking_regressor.fit(X_train, y_train)

# 테스트 세트로 예측
y_pred = stacking_regressor.predict(test_df_all)


submit["SalePrice"]=np.expm1(y_pred)

# CSV로 저장
submit.to_csv('../data/house/second.csv', index=False)