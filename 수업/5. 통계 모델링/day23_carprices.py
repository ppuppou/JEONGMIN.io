import pandas as pd

sample_df = pd.read_csv('../data/car/sample_submission.csv')
train_df = pd.read_csv('../data/car/train.csv')
test_df = pd.read_csv('../data/car/test.csv')
train_df = train_df.drop(columns='id')
test_df = test_df.drop(columns='id')
train_df.info()
train_df.head()
test_df.info()
len(train_df.clean_title.unique())
train_df['fuel_type'].unique()

import pandas as pd
import numpy as np
import car_price_func

# 데이터 로드
sample_df = pd.read_csv('../data/car/sample_submission.csv')
train_df = pd.read_csv('../data/car/train.csv')
test_df = pd.read_csv('../data/car/test.csv')
train_df = train_df.drop(columns='id')
test_df = test_df.drop(columns='id')

# 결측치 처리
train_df.isna().sum()

# 데이터 살펴보기 (clean_title vs accident)
df = train_df.copy()
df["clean_title"] = df["clean_title"].fillna("MISSING")
df["accident"] = df["accident"].fillna("UNKNOWN")

count_tab = pd.crosstab(df["clean_title"], df["accident"])
count_tab

'''
clena_title, accident 결측치
'''
# clean_title가 Yes이면서 accident가 결측(UNKNOWN)인 것 (14건)
train_df["clean_title"] = train_df["clean_title"].fillna("MISSING")
train_df["accident"] = train_df["accident"].fillna("unknown")

mask = (train_df["accident"].eq("UNKNOWN")) & (train_df["clean_title"].eq("Yes"))
train_df.loc[mask, "accident"] = "None reported"

count_tab = pd.crosstab(train_df["clean_title"], train_df["accident"])
count_tab

'''
fuel_type 결측치
'''
mask_missing = train_df["fuel_type"].isna()
train_df.loc[mask_missing, "fuel_type"] = train_df.loc[mask_missing, "engine"].apply(car_price_func.infer_fuel_type)

# engine이 'Electric'이면서 fuel_type이 결측치인 경우
mask = (train_df['engine'] == 'Electric') & (train_df['fuel_type'].isna())

# fuel_type 채우기
train_df.loc[mask, 'fuel_type'] = 'electric'

# 전기차 엔진 패턴 리스트
electric_engines = ['111.2Ah / FR 70kW / RR 160kW (697V)', '120 AH']

# 조건: fuel_type 결측 & engine이 패턴에 포함
mask = (train_df['fuel_type'].isna()) & (train_df['engine'].isin(electric_engines))

# fuel_type 채우기
train_df.loc[mask, 'fuel_type'] = 'electric'

train_df.isna()

'''
'-' 또는 'not supported'를 최빈값으로 대체
'''
cols_to_fix = ['fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']
for col in cols_to_fix:
    # '-' 또는 'not supported' 위치
    mask = train_df[col].isin(['-', 'not supported'])
    # 최빈값 계산 (exclude '-'/'not supported')
    mode_val = train_df.loc[~mask, col].mode()[0]
    # 해당 위치를 최빈값으로 채우기
    train_df.loc[mask, col] = mode_val

num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop(['price'])
cat_columns = train_df.select_dtypes(include=['object']).columns
from sklearn.impute import SimpleImputer
# freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')
# train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
# train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
train_df_cat = onehot.fit_transform(train_df[cat_columns])
std_scaler = StandardScaler().set_output(transform='pandas')
train_df_num = std_scaler.fit_transform(train_df[num_columns])
train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)

X_train = train_df_all
y_train = np.log1p(train_df.price)


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

num_columns = test_df.select_dtypes(include=['number']).columns
cat_columns = test_df.select_dtypes(include=['object']).columns

'''
clena_title, accident 결측치
'''
# clean_title가 Yes이면서 accident가 결측(UNKNOWN)인 것 (14건)
test_df["clean_title"] = test_df["clean_title"].fillna("MISSING")
test_df["accident"] = test_df["accident"].fillna("unknown")

mask = (test_df["accident"].eq("UNKNOWN")) & (test_df["clean_title"].eq("Yes"))
test_df.loc[mask, "accident"] = "None reported"

count_tab = pd.crosstab(test_df["clean_title"], test_df["accident"])
count_tab

'''
fuel_type 결측치
'''
mask_missing = test_df["fuel_type"].isna()
test_df.loc[mask_missing, "fuel_type"] = test_df.loc[mask_missing, "engine"].apply(car_price_func.infer_fuel_type)

# engine이 'Electric'이면서 fuel_type이 결측치인 경우
mask = (test_df['engine'] == 'Electric') & (test_df['fuel_type'].isna())

# fuel_type 채우기
test_df.loc[mask, 'fuel_type'] = 'electric'

# 전기차 엔진 패턴 리스트
electric_engines = ['111.2Ah / FR 70kW / RR 160kW (697V)', '120 AH']

# 조건: fuel_type 결측 & engine이 패턴에 포함
mask = (test_df['fuel_type'].isna()) & (test_df['engine'].isin(electric_engines))

# fuel_type 채우기
test_df.loc[mask, 'fuel_type'] = 'electric'

test_df.isna()

'''
'-' 또는 'not supported'를 최빈값으로 대체
'''
cols_to_fix = ['fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']
for col in cols_to_fix:
    # '-' 또는 'not supported' 위치
    mask = test_df[col].isin(['-', 'not supported'])
    # 최빈값 계산 (exclude '-'/'not supported')
    mode_val = test_df.loc[~mask, col].mode()[0]
    # 해당 위치를 최빈값으로 채우기
    test_df.loc[mask, col] = mode_val
test_df[num_columns] = mean_impute.transform(test_df[num_columns])
test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])
test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)
result = elastic_search.predict(test_df_all)
sample_df.SalePrice = result
sample_df.to_csv('../data/house/second.csv', index=False)
