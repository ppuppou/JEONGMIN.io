import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency, anderson
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV

df1 = pd.read_csv('./quiz5/problem2.csv')
df2 = pd.read_csv('./quiz5/problem4_33.csv')
df3 = pd.read_csv('./quiz5/problem15.csv')
df4 = pd.read_csv('./quiz5/problem19_test.csv')
df5 = pd.read_csv('./quiz5/problem19.csv')
df6 = pd.read_csv('./quiz5/datasetSalaries.csv')

# 1
df6 = pd.read_csv('./quiz5/datasetSalaries.csv')
male = df6.loc[df6['sex']=='Male']
female = df6.loc[df6['sex']=='Female']
ttest_ind(male['salary'], female['salary'], 
          equal_var=False, alternative='two-sided')

# 2
df6['rank'].unique()
pro = df6.loc[df6['rank']=='Professor', 'salary']
ant = df6.loc[df6['rank']=='Assistant Professor', 'salary']
ate = df6.loc[df6['rank']=='Associate Professor', 'salary']
f_stat, p_value = f_oneway(pro, ant, ate)
print(f"F-statistic: {f_stat}, P-value: {p_value}")

# 3
import scipy.stats as sp
w, p_value = sp.shapiro(df6['salary'])
w, p_value = sp.shapiro(ant)
w, p_value = sp.shapiro(ate)

# 4
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
                          endog=df6['salary'],
                          groups=df6['rank'],
                          alpha=0.05)
print(tukey)

# 5
sample = [1.95, 1.80, 2.10, 1.82, 1.75, 2.01, 1.83, 1.90]
from scipy.stats import t
s = np.std(sample, ddof=1)
mean = np.mean(sample)
x = t.ppf(0.05, df=7)
mean - x * (s/np.sqrt(7))

# 6
from scipy.stats import chisquare, poisson
x = poisson(0.8)
x.cdf(0) # 0.449
x.cdf(1)-x.cdf(0) #0.359
x.cdf(2)-x.cdf(1) # 0.144
1-x.cdf(2) # 0.047
ob = np.array([22, 15, 9, 4])
ex = np.array([22.46,17.96,7.22,2.36])
statistic, p_value = chisquare(ob, f_exp=ex)

# 8
sample = [96, 95, 103.2, 101.0, 100.7, 99.9, 98.6, 100.1, 97.3, 98.4, 99.5, 100.2, 101.4, 100.9, 102.0, 96.8, 99.1]
from scipy.stats import norm
n = norm(100,5)
zero = 17 * (n.cdf(97))
first = 17 * (n.cdf(99)-n.cdf(97))
second = 17 * (n.cdf(101)-n.cdf(99))
third = 17 * (1-n.cdf(101))
ex = [zero,first,second,third]
ob = [3,3,8,3]
statistic, p_value = chisquare(ob, f_exp=ex)

# 9
old_env = [72, 68, 74, 70, 65, 69, 71, 73, 67, 66]
new_env = [78, 70, 76, 74, 69, 72, 75, 77, 70, 72]
ttest_rel(old_env, new_env, alternative='greater')

# 10
a = [11,5,5]
b = [10,10,1]
c = [9,8,1]
d = [7,6,0]
table = np.array([a,
                  b,
                  c,
                  d])
chi2, p, df, expected = chi2_contingency(table, correction=False)

# 11
df2
df7 = pd.DataFrame()
df7['delay_under_10'] = df2['delay_0_5'] + df2['delay_5_10']
df7['delay_10_20'] = df2['delay_10_15'] + df2['delay_15_20']
df7['delay_undelay_over_20 der_10'] = df2['delay_20_25'] + df2['delay_25_30']
df7
df7['delay_under_10'] = df7['delay_under_10'].map({0: 0,
                                                    1: "Under10",
                                                    2: "Under10",
                                                    3: "Under10",
                                                    4: "Under10",
                                                    5: "Under10",
                                                    6: "Under10",
                                                    7: "Under10",
                                                    8: "Under10",
                                                    9: "Under10",
                                                    10: "Under10",
                                                    11: "Under10",
                                                    12: "Under10"
                                                    })
df7['delay_10_20'] = df7['delay_10_20'].map({0: 0,
                                             1: "10to20",
                                             2: "10to20",
                                             3: "10to20",
                                             4: "10to20",
                                             5: "10to20",
                                             6: "10to20",
                                             7: "10to20",
                                             8: "10to20"
                                             })
df7['delay_undelay_over_20 der_10'] =  df7['delay_undelay_over_20 der_10'].map({0: 0,
                                                                                1: "Over20",
                                                                                2: "Over20",
                                                                                3: "Over20",
                                                                                4: "Over20",
                                                                                5: "Over20"
                                                                                })
                                                                                
df7


# 12

# 14
from sklearn.datasets import fetch_openml
data = fetch_openml(name="energy_efficiency", version=1, as_frame=True)
X, y = data.data, data.target
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  y, test_size = 0.2, random_state = 123
                   )
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X, train_y)
y_pred = lr.predict(test_X)
root_mean_squared_error(y_true=test_y, y_pred=y_pred)
r2_score(y_true=test_y, y_pred=y_pred)

# 15
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.get_params()
lasso_params = {'alpha' : np.arange(0.1, 1, 0.1)}
lasso_search = GridSearchCV(estimator=lasso, 
                              param_grid=lasso_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
lasso_search.fit(train_X, train_y)
lasso_search.best_params_

# 16
df3
from scipy.stats import pearsonr
corr, x = pearsonr(df3['fedu'],df3['medu'])

# 17
df3.dropna(subset='absences')
X = df3.drop('absences',axis=1)
y = df3.absences
train_X, test_X, train_y, test_y = train_test_split(
                   X,  y, test_size = 0.3, random_state = 0
                   )
from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')
train_X[['medu']] = knnimputer.fit_transform(train_X[['medu']])
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
cat = onehot.fit_transform(train_X[['sex','pstatus','guardian','traveltime',
                                    'studytime','freetime','famrel']])
train_X = pd.concat([cat,train_X[['age','medu','fedu','failures']]],axis=1)
train_X['sex_F'].mean()
y
import joblib

# 19
cv = KFold(n_splits=5, shuffle=True, random_state=0)
dct_params = {'ccp_alpha' : np.linspace(0.1,0.9,9)}
from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor()
X = df5.drop('absences',axis=1)
y = df5.absences
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_mean_absolute_error')
dct_search.fit(X,y)
dct_search.best_score_
dct_search.best_params_
lasso_params = {'alpha' : np.arange(0.1, 1, 0.1)}
lasso_search = GridSearchCV(estimator=lasso, 
                              param_grid=lasso_params, 
                              cv = cv, 
                              scoring='neg_mean_absolute_error')
lasso_search.fit(X,y)
lasso_search.best_score_
lasso_search.best_params_


# 20
# 모델 불러오기 
import joblib
lasso_search = joblib.load("./quiz5/lasso_model.pkl")
all_dat = pd.concat([df5, df4], axis=0)
N = 1000
corr = pd.DataFrame()
for i in range(1, N):
       sub_dat = all_dat.sample(frac=0.6, random_state=i)
       X = sub_dat.drop('absences',axis=1)
       y = sub_dat.absences
       lasso_search.fit(X,y)
       

# 23
dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_3_2.csv')
dat.groupby('Phone_Service').count()
193, 307
dat.groupby('Phone_Service').sum()
a = (171/307)/(136/307)
b = (80/193)/(113/193)
a/b


# 25
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
model = ols(
    "bill_length_mm ~ bill_depth_mm + species + bill_depth_mm:species", data=train_data
).fit()
print(model.summary())

