import pandas as pd
from scipy.stats import chi2_contingency
# 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 임신 유무 파생변수 생성
dat['Pregnancy_status'] = (dat['Pregnancies'] > 0).astype(int)

dat
# 1 
# 1-1
# 귀무가설 : 당뇨병과 임신유무는 독립이다

# 1-2
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Pregnancy_status', hue='Outcome', data=dat, palette='viridis')
plt.title('Relationship between Pregnancy Status and Outcome', fontsize=15)
plt.xlabel('Pregnancy Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
# x축 눈금 레이블 변경 (0 -> Never Pregnant, 1 -> Has Been Pregnant)
ax.set_xticklabels(['Never Pregnant', 'Has Been Pregnant'])
# 범례(legend) 제목 및 레이블 변경
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=['Negative', 'Positive'], title='Outcome')
plt.show()

# 1-3
from scipy.stats import chi2_contingency
observed_table = pd.crosstab(dat['Pregnancy_status'], dat['Outcome'])
observed_table
chi2, p, dof, expected = chi2_contingency(observed_table, correction=False)
chi2 # 0.02499920980068189
p # 0.8743690301893354
dof # 1
expected

# 1-5 기각할만한 근거가 부족하다

# 2
# 2-1
# 귀무가설 : 연령대에 따른 당뇨병 여부는 동일하다

# 2-2 
bins = [0,40,120]
labels = ['Under40', 'Over40']
dat['Age_group'] = pd.cut(dat['Age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Age_group', hue='Outcome', data=dat)
# 그래프 제목 및 라벨 설정
plt.title('Relationship between Age Group and Outcome', fontsize=15)
plt.ylabel('Count')
plt.xlabel('Age Group')
plt.legend(title='Outcome', labels=['Negative (0)', 'Positive (1)'])

# 2-3
observed_table = pd.crosstab(dat['Age_group'], dat['Outcome'])
chi2, p, dof, expected = chi2_contingency(observed_table, correction=False)
chi2 # 37.23722358215896
p # 1.0459803299776332e-09
expected

# 2-5 기각한다

# 3
# 3-1
import numpy as np
table1 = np.array([[30,70],
                   [50,50],
                   [70,30]])
# 귀무가설 : 운동빈도와 건강도는 서로 독립이다

# 3-2
chi2, p, dof, expected = chi2_contingency(table1, correction=False)

# 3-3 
chi2 # 32
p # 1.1253517471925916e-07
dof
# 기각한다

# 4
# 4-1
table2 = np.array([[1,4,2,8,4,6,15,20]]).reshape(4,-1)
table2
# 귀무가설 : 식습관과 건강은 서로 독립이다

# 4-2
chi2, p, dof, expected = chi2_contingency(table2, correction=False)
p
expected

# 4-3
table3 = np.array([[7,18,15,20]]).reshape(2,-1)
chi2, p, dof, expected = chi2_contingency(table3, correction=False)
expected
p