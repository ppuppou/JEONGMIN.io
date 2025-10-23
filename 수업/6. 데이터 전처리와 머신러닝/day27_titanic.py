import pandas as pd
import numpy as np

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
test_df_all = pd.concat([test_df_cat, test_df_num], axis = 1)
result = dct_search.predict(test_df_all)
sub['Survived'] = result

sub.to_csv('../data/titanic/first.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# -------------------
# 데이터 로드
# -------------------
train_df = pd.read_csv('../data/titanic/train.csv')
test_df = pd.read_csv('../data/titanic/test.csv')
sub = pd.read_csv('../data/titanic/gender_submission.csv')

# -------------------
# Feature Engineering
# -------------------

# 1) Title 추출
for df in [train_df, test_df]:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev',
         'Sir','Jonkheer','Dona'], 'Rare')

# 2) 가족 크기 + 혼자인지 여부
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize']==1, 'IsAlone'] = 1

# 3) Age 결측치 처리 + 구간화
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(train_df['Age'].median())
    df['AgeBand'] = pd.cut(df['Age'], 5, labels=False)  # 0~4

# 4) Fare 결측치 처리 + 구간화
for df in [train_df, test_df]:
    df['Fare'] = df['Fare'].fillna(train_df['Fare'].median())
    df['FareBand'] = pd.qcut(df['Fare'], 4, labels=False)

# 5) Embarked 결측치 처리
for df in [train_df, test_df]:
    df['Embarked'] = df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# -------------------
# 최종 feature 선택
# -------------------
features = ['Pclass','Sex','AgeBand','FareBand','Embarked','Title','FamilySize','IsAlone']

X = pd.get_dummies(train_df[features])
y = train_df['Survived']
X_test = pd.get_dummies(test_df[features])

# train과 test 열 맞추기
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# -------------------
# 모델 학습
# -------------------
model = GradientBoostingClassifier(
    n_estimators=300, 
    max_depth=3, 
    learning_rate=0.05,
    random_state=2025
)

model.fit(X, y)

# -------------------
# 예측 + 제출 파일
# -------------------
pred = model.predict(X_test)
sub['Survived'] = pred
sub.to_csv('../data/titanic/first.csv', index=False)

print("✅ 제출 파일 저장 완료: ../data/titanic/first.csv")
