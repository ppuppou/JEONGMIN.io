import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train 셋
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data_for_learning = pd.DataFrame({'x': x, 'y': y})

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(data_for_learning[['x']],data_for_learning['y'])

data_for_learning['x^2'] = data_for_learning['x']**2
data_for_learning['x^3'] = data_for_learning['x']**3
tr_X = data_for_learning.drop(columns='y')
tr_y = data_for_learning.y
lr.fit(tr_X,tr_y)
lr.coef_
lr.intercept_
y_hat = lr.predict(tr_X)

# 산점도
plt.figure(figsize=(8,5))
plt.scatter(data_for_learning['x'],data_for_learning['y'],
            color='blue', alpha=0.7, label='Training data')

plt.scatter(tr_X['x'],y_hat,
            color='red', alpha=0.7, label='regression line')
plt.show()


# train 셋 나누기 -> train, valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)

# print(train.shape)
# print(valid.shape)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
i=20    # i = 1에서 변동시키면서 MSE 체크 할 것
k = np.linspace(0, 1, 100)
sin_k = np.sin(2 * np.pi * k)
poly1 = PolynomialFeatures(degree=i, include_bias=True)
train_X = poly1.fit_transform(train[['x']])
model1 = LinearRegression().fit(train_X, train['y'])
model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

# 예측값 계산
train_y_pred = model1.predict(poly1.transform(train[['x']]))
valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

# MSE 계산
mse_train = mean_squared_error(train['y'], train_y_pred)
mse_valid = mean_squared_error(valid['y'], valid_y_pred)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1행 2열 서브플롯

# 왼쪽: 학습 데이터와 모델 피팅 결과
axes[0].scatter(train['x'], train['y'], color='black', label='Train Observed')
axes[0].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[0].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')
axes[0].set_ylim((-2.0, 2.0))
axes[0].legend()
axes[0].grid(True)
# 오른쪽: 검증 데이터
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')
axes[1].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[1].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')
axes[1].set_ylim((-2.0, 2.0))
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.show()

mse = {}
for i in range(1,21):
    k = np.linspace(0, 1, 100)
    sin_k = np.sin(2 * np.pi * k)
    poly1 = PolynomialFeatures(degree=i, include_bias=True)
    train_X = poly1.fit_transform(train[['x']])
    model1 = LinearRegression().fit(train_X, train['y'])
    model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

    # 예측값 계산
    train_y_pred = model1.predict(poly1.transform(train[['x']]))
    valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

    # MSE 계산
    mse_train = mean_squared_error(train['y'], train_y_pred)
    mse_valid = mean_squared_error(valid['y'], valid_y_pred)
    mse[i] = [mse_train,mse_valid]
mse_df = pd.DataFrame(mse, index=['Train MSE', 'Valid MSE']).round(4)
mse_df

plt.figure(figsize=(8,5))
plt.plot(mse_df.columns, mse_df.loc['Train MSE'], marker='o', label='Train MSE')
plt.plot(mse_df.columns, mse_df.loc['Valid MSE'], marker='o', label='Valid MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE (log scale)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.xticks(range(1, 21))
plt.title('Train vs Validation MSE (Log Scale)')
plt.show()


# 라쏘 · 릿지 회귀
import pandas as pd

train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(['number'])
train_df.dropna(inplace=True)
X = train_df.drop(columns='SalePrice')
y = train_df.SalePrice
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=100000)
lasso.fit(X,y)
lasso.coef_
lasso.intercept_

# 최적의 알파값 찾기
import numpy as np
from sklearn.linear_model import LassoCV # 최적의 값을 찾아주는 매서드
from sklearn.metrics import root_mean_squared_error

alphas = np.linspace(0.0001,20,200)
lasso_cv = LassoCV(alphas=alphas, cv=5,  # 데이터를 5개로 분할
                   random_state=42)
lasso_cv.fit(X,y)

best_alpha = lasso_cv.alpha_
best_alpha

# 릿지 (피타고라스)
from sklearn.linear_model import Ridge
train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(['number'])
train_df.dropna(inplace=True)
X = train_df.drop(columns='SalePrice')
y = train_df.SalePrice
ridge = Ridge(alpha=0.01)
ridge.fit(X,y)
ridge.coef_

# elastic (l1 + l2)
from sklearn.linear_model import ElasticNet
train_df = pd.read_csv('../data/house/train.csv')
test_df = pd.read_csv('../data/house/test.csv')
train_df = train_df.select_dtypes(['number'])
train_df.dropna(inplace=True)
X = train_df.drop(columns='SalePrice')
y = train_df.SalePrice
test_df = test_df.select_dtypes(['number'])
test_df = test_df.fillna(train_df.mean())

elastic = ElasticNet(alpha=0.1, l1_ratio=0.6)
elastic.fit(X,y)
y_pred = elastic.predict(test_df)

sample_df = pd.read_csv('../data/house/sample_submission.csv')
sample_df.SalePrice = y_pred

from sklearn.linear_model import ElasticNet
import numpy as np
elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
# 파라미터 확인 
ElasticNet().get_params()
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
# 그리드서치
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
elastic_search.fit(X, y)
pd.DataFrame(elastic_search.cv_results_).head()
# best prameter
print(elastic_search.best_params_)
# 교차검증 best score 
print(-elastic_search.best_score_)

elastic_search.predict(test_df)



train_df = pd.read_csv('../data/house/train.csv')
num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop(['SalePrice'])
cat_columns = train_df.select_dtypes(include=['object']).columns
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')
train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
train_df_cat = onehot.fit_transform(train_df[cat_columns])
std_scaler = StandardScaler().set_output(transform='pandas')
train_df_num = std_scaler.fit_transform(train_df[num_columns])
train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)

X_train = train_df_all
y_train = train_df.SalePrice


from sklearn.linear_model import ElasticNet
import numpy as np
elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
ElasticNet().get_params()
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
elastic_search = GridSearchCV(estimator=elasticnet, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
elastic_search.fit(X_train, y_train)
pd.DataFrame(elastic_search.cv_results_).head()
print(elastic_search.best_params_)
print(-elastic_search.best_score_)

test_df = pd.read_csv('../data/house/test.csv')
num_columns = test_df.select_dtypes(include=['number']).columns
cat_columns = test_df.select_dtypes(include=['object']).columns
test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])
test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])
test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)
result = elastic_search.predict(test_df_all)
sample_df.SalePrice = result
sample_df.to_csv('../data/house/second.csv', index=False)

# np.log1p(y_train).hist() # log를 씌워 정규성을 띄게 하면 회귀모델의 성능이 좀 더 좋아짐




# knn regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn_params = {'n_neighbors' : np.arange(1, 100, 1)}
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
knn_search = GridSearchCV(estimator=knn, 
                              param_grid=knn_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
knn_search.fit(X_train, y_train)
pd.DataFrame(knn_search.cv_results_).head()
print(knn_search.best_params_)
print(-knn_search.best_score_)
result2 = knn_search.predict(test_df_all)
sample_df.SalePrice = (sample_df.SalePrice + result2) / 2
sample_df.to_csv('../data/house/second.csv', index=False)