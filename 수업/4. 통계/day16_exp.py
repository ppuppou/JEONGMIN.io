import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform 
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 1
dat = pd.read_csv('./problem5_27.csv')
up = dat['up']
down = dat['down']
a,b = ttest_rel(up, down, alternative='two-sided')
b
x = down-up
a,b = ttest_1samp(x, popmean=0, alternative='two-sided')
# 2
dat2 = pd.read_csv('./problem5_32.csv')
man = dat2[dat2['Gender']=='Male']['Salary'].dropna()
girl = dat2[dat2['Gender']=='Female']['Salary'].dropna()
plt.figure(figsize=(12, 6))
sns.histplot(data=dat2, x="Salary", hue="Gender", kde=True, palette="Set2", bins=20)
plt.title('성별에 따른 급여 히스토그램')
plt.xlabel('급여')
plt.ylabel('빈도')
plt.show()

np.var(man)
np.var(girl)
c, d = ttest_ind(man,girl,equal_var=False, alternative='two-sided')
d

# 3
dat3 = pd.read_csv('./heart_disease.csv')
yes = dat3[dat3['target']=='yes']['chol'].dropna()
no = dat3[dat3['target']=='no']['chol'].dropna()

plt.figure(figsize=(12, 6))
sns.histplot(yes, color='red', label='Heart Disease (yes)', kde=True, stat="density", bins=30)
sns.histplot(no, color='blue', label='No Heart Disease (no)', kde=True, stat="density", bins=30)
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.xlabel('Cholesterol Level')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

yes.var()
no.var()
e, f = ttest_ind(yes,no,equal_var=True, alternative='two-sided')
f

# 4
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()
yes1 = dat[dat['Outcome']==1]['BMI'].dropna()
no1 = dat[dat['Outcome']==0]['BMI'].dropna()

plt.figure(figsize=(12, 6))
sns.histplot(yes1, color='orangered', label='Diabetes (Outcome = 1)', kde=True, stat='density', bins=30)
sns.histplot(no1, color='steelblue', label='No Diabetes (Outcome = 0)', kde=True, stat='density', bins=30)
plt.title('BMI Distribution by Diabetes Outcome')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

yes1.var()
no1.var()
g, h = ttest_ind(yes1,no1,equal_var=True, alternative='two-sided')
h