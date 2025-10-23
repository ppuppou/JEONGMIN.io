import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform
import scipy.stats as sp
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, kstest, anderson
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1
dat = pd.read_csv('./problem5_32.csv')
dat.head()
male = dat[dat['Gender']=='Male']['Salary'].dropna()
female = dat[dat['Gender']=='Female']['Salary'].dropna()
genders = dat['Gender'].unique()
for gender in genders:
    plt.figure(figsize=(6, 4))
    subset = dat[dat['Gender'] == gender]['Salary']
    sp.probplot(subset, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Salary ({gender})")
    plt.grid(True)
    plt.show()
loc = np.mean(male)
scale = np.std(male, ddof=1)
# 정규분포를 기준으로 K-S 검정 수행
result = kstest(male, 'norm', args=(loc, scale))
print("검정통계량:", result.statistic)
print('p-value:', result.pvalue)
loc2 = np.mean(female)
scale2 = np.std(female, ddof=1)
result2 = kstest(female, 'norm', args=(loc2, scale2))
print("검정통계량:", result2.statistic)
print('p-value:', result2.pvalue)
# 남성과 여성의 급여 모두 정규분포를 따름

# 2
dat2 = pd.read_csv('./heart_disease.csv')
yes = dat2[dat2['target']=='yes']['chol'].dropna()
no = dat2[dat2['target']=='no']['chol'].dropna()
# Q-Q Plot: target = yes
plt.figure(figsize=(6, 4))
sp.probplot(yes, dist="norm", plot=plt)
plt.title("Q-Q Plot of 'chol' for target = yes")
plt.grid(True)
plt.show()
# Q-Q Plot: target = no
plt.figure(figsize=(6, 4))
sp.probplot(no, dist="norm", plot=plt)
plt.title("Q-Q Plot of 'chol' for target = no")
plt.grid(True)
plt.show()
w, p_value = sp.shapiro(yes)
print("W:", w, "p-value:", p_value)
w2, p_value2 = sp.shapiro(no)
print("W:", w2, "p-value:", p_value2)
# 심장질환이 없으면 정규분포, 있으면 X

# 3
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat3 = pd.read_csv(url, header=None, names=col_names)
dat3.head()
dangnyo = dat3[dat3['Outcome']==1]['BMI'].dropna()
no_dang = dat3[dat3['Outcome']==0]['BMI'].dropna()
# Q-Q Plot: 당뇨 O
plt.figure(figsize=(6, 4))
sp.probplot(dangnyo, dist="norm", plot=plt)
plt.title("당뇨가 있는 사람의 BMI QQplot")
plt.grid(True)
plt.show()
# Q-Q Plot: 당뇨 X
plt.figure(figsize=(6, 4))
sp.probplot(no_dang, dist="norm", plot=plt)
plt.title("당뇨가 없는 사람의 BMI QQplot")
plt.grid(True)
plt.show()
w3, p_value3 = sp.shapiro(dangnyo)
print("W:", w3, "p-value:", p_value3)
w4, p_value4 = sp.shapiro(no_dang)
print("W:", w4, "p-value:", p_value4)
# 당뇨 여부에 따른 BMI 수치는 정규분포를 따르지 않음

# 4
dat4 = pd.read_csv('./problem5_44.csv')
data = dat4.iloc[:, 0].values
data.mean()
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(data)
x = np.linspace(min(data), max(data),1000)
y = ecdf(x)
plt.plot(x,y,marker='.', linestyle='none')
plt.title("Estimated CDF")
plt.xlabel("X-axis")
plt.ylabel("ECDF")
# 이론적 누적분포함수
plt.plot(x, expon.cdf(x, scale=np.mean(data)), color='red')
plt.show()
from scipy.stats import anderson
result = sp.anderson(data, dist='expon') # Anderson-Darling 검정 수행
print('검정통계량',result[0], '\n', 
      '임계값:',result[1], '\n',
      '유의수준:',result[2])
# 지수분포를 따르지 않음