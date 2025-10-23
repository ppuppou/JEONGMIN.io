import pandas as pd

train_df = pd.read_csv('../data/house/train.csv')
test_df = pd.read_csv('../data/house/test.csv')
sample_df = pd.read_csv('../data/house/sample_submission.csv')
train_df
test_df
sample_df

#  단순 수치형 선형회귀
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df = train_df.dropna()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
model.fit(X,y)
model.coef_
model.intercept_

test_df = test_df.select_dtypes(include=['int64','float64'])
test_df = test_df.fillna(0)

result = model.predict(test_df)
sample_df['SalePrice'] = result
sample_df

# sample_df.to_csv('../data/house/first.csv')


# ID 제거 + NA값 평균치로 치환
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df = train_df.drop('Id', axis=1)
train_df = train_df.dropna(subset=['MasVnrArea'])
train_df = train_df.fillna(train_df.mean())

X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
model.fit(X,y)
model.coef_
model.intercept_

test_df = test_df.select_dtypes(include=['int64','float64'])
test_df = test_df.fillna(test_df.mean())
test_df.drop('Id',axis=1,inplace=True)

result = model.predict(test_df)
sample_df['SalePrice'] = result
sample_df

sample_df.to_csv('../data/house/second.csv')


# 분할과 전처리 후 학습 
train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df = train_df.dropna(subset=['MasVnrArea'])
train_df = train_df.drop('Id', axis=1)
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  y, test_size = 0.3, random_state = 0
                   )
from sklearn.impute import SimpleImputer
train_X1 = train_X.copy()
test_X1 = test_X.copy()
# imputer_mean = SimpleImputer(strategy = 'most_frequent')
# train_X1['LotFrontage'] = imputer_mean.fit_transform(train_X1[['LotFrontage']])
# test_X1['LotFrontage'] = imputer_mean.transform(test_X1[['LotFrontage']])
# train_X1['GarageYrBlt'] = imputer_mean.fit_transform(train_X1[['GarageYrBlt']])
# test_X1['GarageYrBlt'] = imputer_mean.transform(test_X1[['GarageYrBlt']])

from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')
train_X1 = knnimputer.fit_transform(train_X1)
test_X1 = knnimputer.transform(test_X1)
train_X1.isna().sum().sum()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X1,train_y)

result = model.predict(test_X1)
from sklearn.metrics import root_mean_squared_error
root_mean_squared_error(test_y, result)

train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df = train_df.dropna(subset=['MasVnrArea'])
train_df = train_df.drop('Id', axis=1)
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
X1 = X.copy()
X1 = knnimputer.fit_transform(X1)
model.fit(X1,y)

test_df = test_df.select_dtypes(include=['int64','float64'])
test_df = knnimputer.fit_transform(test_df)
test_df.drop('Id',axis=1,inplace=True)

result = model.predict(test_df)
sample_df['SalePrice'] = result
sample_df
sample_df.to_csv('../data/house/second.csv',index=False)


# 한번에
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df = train_df.dropna(subset=['MasVnrArea'])
train_df = train_df.drop('Id', axis=1)
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
imputer = KNNImputer()
num_columns = X.select_dtypes('number').columns
mc_transformer = make_column_transformer(
    (imputer, num_columns), (stdscaler, num_columns), remainder='passthrough'
    ).set_output(transform = 'pandas')
X = mc_transformer.fit_transform(X)
model.fit(X,y)



