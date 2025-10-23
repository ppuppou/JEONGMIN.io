import pandas as pd 
import numpy as np 
df = pd.read_csv('../data/problem1.csv')

df.info()
df.head()

# 2. 전처리
# Chol에 결측치 존재
from sklearn.impute import KNNImputer
knn_i = KNNImputer().set_output(transform='pandas')
knn_i.get_params()
df = knn_i.fit_transform(df)
X = df.drop(columns='DBP')
y = df.DBP

# 3. 7:3 나누기
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  y, test_size = 0.3, random_state = 0
                   )
train_X.shape[0]

# 4 
X1 = X.drop(columns=['Chol','FPG','BMI','LDL'])
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif['features'] = X1.columns
vif

# 6
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
knn = KNeighborsRegressor(n_neighbors=5)
elastic = ElasticNet()
dct = DecisionTreeRegressor()
knn_params = {'n_neighbors' : np.arange(1, 100, 1)}
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
dct_params = {'max_depth' : np.arange(1,7),
              'ccp_alpha' : np.linspace(0,1,10)}

# 교차검증 knn
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
knn_search = GridSearchCV(estimator=knn, 
                              param_grid=knn_params, 
                              cv = cv, 
                              scoring='neg_root_mean_squared_error')
knn_search.fit(X1, y)
print(knn_search.best_params_)
print(-knn_search.best_score_)

# 교차검증 elastic
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
elastic_search = GridSearchCV(estimator=elastic, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_root_mean_squared_error')
elastic_search.fit(X1, y)
print(elastic_search.best_params_)
print(-elastic_search.best_score_)

# 교차검증 dct
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_root_mean_squared_error')
dct_search.fit(X1, y)
print(dct_search.best_params_)
print(-dct_search.best_score_)
