import pandas as pd
import numpy as np 
train = pd.read_csv('../data/titanic/train.csv')

from sklearn.feature_extraction.text import CountVectorizer
name_list = train["Name"].astype(str)
vectorizer = CountVectorizer(
    stop_words="english",  # 불용어 처리
    max_features=7,  # top3 단어만
)
X_name = vectorizer.fit_transform(name_list)
print(type(X_name))

name_features = pd.DataFrame(
    X_name.toarray(), columns=vectorizer.get_feature_names_out()
)
print(name_features.head())
name_features.shape


train_df = pd.read_csv('../data/titanic/train.csv')
test_df = pd.read_csv('../data/titanic/test.csv')
sub = pd.read_csv('../data/titanic/gender_submission.csv')


num_columns = train_df.select_dtypes(include=['number']).columns
num_columns = num_columns.drop(["Survived",'PassengerId'])
cat_columns = train_df.select_dtypes(include=['object']).columns
cat_columns = cat_columns.drop(["Name",'Ticket','Cabin'])

from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')
train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
# freq_impute.statistics_
# mean_impute.statistics_
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")
train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])
# train_df[num_columns].mean(axis=0)
# train_df[num_columns].std(axis=0, ddof=1)
# train_df_num.mean(axis=0)
# std_scaler.mean_
train_df_all = pd.concat([train_df_cat,
                          train_df_num], axis = 1)
train_df_all = pd.concat([train_df_all,
                          name_features], axis = 1)

X_train = train_df_all
y_train = train_df['Survived']

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='gini')
dct.get_params()
dct_params = {'max_depth' : np.arange(1, 8),
              'ccp_alpha': np.linspace(0, 1, 5)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)
# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                          param_grid=dct_params, 
                          cv = cv, 
                          scoring='accuracy')
dct_search.fit(X_train, y_train)
dct_search.best_params_

test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])
test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])
name_list_test = test_df["Name"].astype(str)
X_test_name = vectorizer.transform(name_list_test)
print(type(X_test_name))
name_features_test = pd.DataFrame(
    X_test_name.toarray(), columns=vectorizer.get_feature_names_out()
)
print(name_features_test.head())
name_features_test.shape

test_df_all = pd.concat([test_df_cat, test_df_num], axis = 1)
test_df_all = pd.concat([test_df_all, name_features_test], axis = 1)

result = dct_search.predict(test_df_all)
sub['Survived'] = result

sub.to_csv('../data/titanic/first.csv', index=False)



# 이산화
X = np.array([[0, 1, 1, 2, 5, 10, 11, 14, 18]]).T
from sklearn.preprocessing import KBinsDiscretizer

# 균일하게
kbd = KBinsDiscretizer(n_bins = 3, strategy = 'uniform') # 구간의 길이가 동일
X_bin = kbd.fit_transform(X).toarray()
print(kbd.bin_edges_)

# 4분위수 기준
kbd2 = KBinsDiscretizer(n_bins = 4, 
                        strategy = 'quantile') # 사분위수를 기준으로 이산화
X_bin2 = kbd2.fit_transform(X).toarray()
print(kbd2.bin_edges_)

# 임의설정
bins = [0, 4, 7, 11, 18]
labels = ['A', 'B', 'C', 'D']
X_bin3 = pd.cut(X.reshape(-1), 
                bins = bins, 
                labels = labels)
print(X_bin3)

# 범주형변수 축소(기타처리)
train_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv')
test_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_test.csv')
print(train_bike.head(2))
freq = train_bike['weather'].value_counts(normalize = True) # 상대비율로 표현
freq
prob_columns = train_bike['weather'].map(freq)
train_bike['weather'] = train_bike['weather'].mask(prob_columns < 0.1, 'other')
test_bike['weather'] = np.where(test_bike['weather'].isin([4]), 'other', test_bike['weather'])
