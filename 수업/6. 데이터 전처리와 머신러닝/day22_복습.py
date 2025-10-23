import pandas as pd
from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')

train_df = pd.read_csv('../data/house/train.csv')
train_df = train_df.select_dtypes(include=['int64','float64'])
train_df.shape
shuffled_tr = train_df.sample(frac=1).reset_index(drop=True)
n = shuffled_tr.shape[0]
k1 = int(n*0.33)
k2 = int(n*0.66)

df1_Xy = shuffled_tr.iloc[:k1,:]
df2_Xy = shuffled_tr.iloc[k1:k2,:]
df3_Xy = shuffled_tr.iloc[k2:,:]

tr1 = pd.concat([df1_Xy,df2_Xy],axis=0)
X = tr1.drop('SalePrice',axis=1)
y = tr1.SalePrice
valid1 = df3_Xy.copy()
X = knnimputer.fit_transform(X)
model.fit(X,y)
valid1_X = valid1.drop('SalePrice',axis=1)
valid1_y = valid1.SalePrice
valid1_X = knnimputer.transform(valid1_X)
result = model.predict(valid1_X)
from sklearn.metrics import root_mean_squared_error
root_mean_squared_error(valid1_y, result) # 41738.457547015714

tr2 = pd.concat([df1_Xy,df3_Xy],axis=0)
X = tr2.drop('SalePrice',axis=1)
y = tr2.SalePrice
valid2 = df2_Xy.copy()
X = knnimputer.fit_transform(X)
model.fit(X,y)
valid2_X = valid2.drop('SalePrice',axis=1)
valid2_y = valid2.SalePrice
valid2_X = knnimputer.transform(valid2_X)
result = model.predict(valid2_X)
root_mean_squared_error(valid2_y, result) # 39960.09285908851

tr3 = pd.concat([df2_Xy,df3_Xy],axis=0)
X = tr3.drop('SalePrice',axis=1)
y = tr3.SalePrice
valid3 = df1_Xy.copy()
X = knnimputer.fit_transform(X)
model.fit(X,y)
valid3_X = valid3.drop('SalePrice',axis=1)
valid3_y = valid3.SalePrice
valid3_X = knnimputer.transform(valid3_X)
result = model.predict(valid3_X)
root_mean_squared_error(valid3_y, result) # 38253.521183964775
