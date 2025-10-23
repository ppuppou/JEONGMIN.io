# KNN
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv') 
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')

train_X = train.drop(['grade'], axis = 1)
train_y = train['grade']
test_X = test.drop(['grade'], axis = 1)
test_y = test['grade']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()
cat_preprocess = make_pipeline(
    #SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)
num_preprocess = make_pipeline(
    SimpleImputer(strategy="mean"), 
    StandardScaler()
)
preprocess = ColumnTransformer(
    [("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)]
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
full_pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("regressor", KNeighborsRegressor())
    ]
)

import numpy as np
knn_param = {'regressor__n_neighbors': np.arange(5, 10, 1)}
from sklearn.model_selection import GridSearchCV
knn_search = GridSearchCV(full_pipe, 
                          param_grid = knn_param, 
                          cv = 3, scoring = 'neg_mean_squared_error')
knn_search.fit(train_X, train_y)

print('best 파라미터 조합 :', knn_search.best_params_)
print('교차검증 MSE :', -knn_search.best_score_)

from sklearn.metrics import mean_squared_error
knn_pred = knn_search.predict(test_X)
print('테스트 MSE :', mean_squared_error(test_y, knn_pred))




# DCT 의사결정나무
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.dropna(inplace=True)
penguins_a = penguins.loc[penguins['species']=='Adelie']

X = penguins_a[['bill_length_mm']]
y = penguins_a.bill_depth_mm

from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor()
dct.get_params()
dct_params = {'max_depth' : np.arange(1,7),
              'ccp_alpha' : np.linspace(0,1,10)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
dct_search.fit(X, y)
# best prameter
print(dct_search.best_params_)
# 교차검증 best score 
print(-dct_search.best_score_)

dct.fit(X,y)
pred_y = dct_search.predict(X)

