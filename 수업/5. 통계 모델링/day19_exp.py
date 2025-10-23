import numpy as np
import scipy.stats as stats
import seaborn as sns

# 1
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 13])
corr, p_value = stats.pearsonr(x,y)
corr
p_value

# 2
x = np.array([1, 2, 3, 4, 10, 11, 12])
y = np.array([2, 4, 6, 8, 100, 200, -100])
corr, p_value = stats.pearsonr(x,y)

# 3
import statsmodels.api as sm
import statsmodels.formula.api as smf
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])
# y = 3x

# 4
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
stats.pearsonr(x,y)
y.std(ddof=1)/x.std(ddof=1)

# 5
import pandas as pd
from sklearn.datasets import fetch_california_housing
cal = fetch_california_housing(as_frame=True)
df = cal.frame
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
print(model.summary())
model.params

# 6
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
print(model.summary())
model.tvalues
model.pvalues

# 7 
df['IncomeLevel'] = pd.qcut(df['MedInc'], q=3, labels=['Low', 'Mid', 'High'])
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup + C(IncomeLevel)', data=df).fit()


# 8 
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model.resid)
# 2에 가까우면 자기상관이 없다. 
# 0에 가까울수록 양의 상관성

# 9
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, model.model.exog)
# 등분산성이 위반되면 t검정, f검정 결과가 왜곡됨

# 11
from sklearn.datasets import load_diabetes
# 데이터 불러오기 및 DataFrame 변환
diabetes = load_diabetes(as_frame=True)
df2 = diabetes.frame
model2 = smf.ols('target ~ bmi + bp + s1', data=df2).fit()
print(model2.summary())

# 12

# 13

# 14
model2 = smf.ols("target ~ bmi + bp + s1 + s2", data=df2).fit()
print(model2.summary())

# 15
model3 = smf.ols('target ~ bmi + bp + s1 + s2 + s3', data=df2).fit()
print(model3.summary())

# 16
penguins = sns.load_dataset("penguins").dropna()
model_penguins = smf.ols('body_mass_g ~ bill_length_mm + flipper_length_mm',
                          data=penguins).fit()
model_penguins.rsquared
print(model_penguins.summary())

# 17
model_penguins2 = smf.ols('body_mass_g ~ bill_length_mm + flipper_length_mm + C(species)',
                          data=penguins).fit()                                #더비변수는 C 
print(model_penguins2.summary())
# 더미변수의 유의미성은 pvalue를 보고 더 낮은지를 판단
# 18
model_penguins3 = smf.ols('body_mass_g ~ bill_length_mm + flipper_length_mm + C(species) + C(sex)',
                          data=penguins).fit()
print(model_penguins3.summary())
# pvalue가 0.00이므로 통계적으로 유의미한 변수다
# >> 회귀식에 해당 변수를 포함하는 것이 적절하다

# 19
from statsmodels.stats.outliers_influence import variance_inflation_factor
x = penguins[['bill_length_mm','flipper_length_mm','bill_depth_mm']]
x_const = sm.add_constant(x)

vif_df = pd.DataFrame()
vif_df['variable'] = x_const.columns
vif_df['VIF'] = [
    variance_inflation_factor(x_const.values,i) for i in range(x_const.shape[1])
]
print(vif_df)


# 20
from scipy.stats import shapiro
resid = model_penguins3.resid # 잔차에 대해서 수행
stat, p_value = shapiro(resid)
p_value


# 21
import pandas as pd
import numpy as np
# 예제 데이터 생성
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(n_samples)
df = pd.DataFrame(X, columns=['var1', 'var2', 'var3', 'var4', 'var5'])
df['target'] = y
# 데이터 확인
print(df.head())
# 상관계수 측정
corrs = df.corr()['target'].drop('target')
# 다중회귀모형으로 타겟변수 예측할 때 모델의 결졍계수 변환
model21 = smf.ols('target ~ var1 + var2 + var3 + var4 + var5', data=df).fit()
model21.rsquared
# 유의확률이 가장 큰 변수와 그때의 pvalue
model21.pvalues



# 22
from sklearn.datasets import make_regression
import statsmodels.api as sm
# 예제 데이터 생성
X, y = make_regression(n_samples=100, n_features=3, 
                       noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'var{i}' for i in range(3)])
df['target'] = y
# 데이터 확인
print(df.head())
model = smf.ols('target ~ var0 + var1 + var2', data=df).fit()
model.pvalues
print(model.summary()) # 75.0508
# 1
newdata = pd.DataFrame({
    'var0' : [0.5],
    "var1" : [1.2],
    "var2" : [0.3]
})
predicted = model.predict(newdata)
predicted # 109.50207