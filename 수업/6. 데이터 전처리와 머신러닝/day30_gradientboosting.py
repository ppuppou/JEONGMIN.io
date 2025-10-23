from matplotlib import pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins
import pandas as pd
df = load_penguins()
penguins=df.dropna()

# x와 y 설정
x = penguins[["bill_length_mm"]]
y = penguins["bill_depth_mm"]

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

model1.fit(x, y)
model1.coef_
model1.intercept_

sns.scatterplot(data=penguins, x = 'bill_depth_mm',
                y = 'bill_length_mm', palette='deep', edgecolor='w', s=50)

y_pred1 = model1.predict(x)
y - y_pred1

sns.scatterplot(x = penguins["bill_length_mm"],
                y = y - y_pred1, palette='deep', edgecolor='w', s=50)




import pandas as pd
import numpy as np
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
from sklearn.ensemble import GradientBoostingRegressor
full_pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("regressor", GradientBoostingRegressor())
    ]
)
GradientBoostingRegressor().get_params()
GradientBoosting_param = {'regressor__learning_rate': np.arange(0.1, 0.3, 0.05),
                         'regressor__random_state' : [0]}
from sklearn.model_selection import GridSearchCV
GradientBoosting_search = GridSearchCV(estimator = full_pipe, 
                      param_grid = GradientBoosting_param, 
                      cv = 5,
                      scoring = 'neg_mean_squared_error')
GradientBoosting_search.fit(train_X, train_y)
print('best 파라미터 조합 :', GradientBoosting_search.best_params_)
print('교차검증 RMSE score :', -GradientBoosting_search.best_score_)
from sklearn.metrics import mean_squared_error
gb_pred = GradientBoosting_search.predict(test_X)
print('테스트 MSE :', mean_squared_error(test_y, gb_pred))