import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform
import scipy.stats as sp
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, kstest, anderson
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 분산분석 ANOVA(ANalysis Of VAriance)
# 일원분산분석 : 하나의 요인에 따라 여러 집단간 평균차이를 검정하는 기법
# H0 : 모든 집단의 평균이 동일하다
# Ha : 평균이 다른 집단이 적어도 하나 존재한다
odors = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10, 11]
minutes_rosemary = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_peppermint = [18, 17, 18, 16, 17, 19, 18, 17, 18, 19]
anova_data = pd.DataFrame({
    'Odor': np.repeat(odors, 10),
    'Minutes': minutes_lavender + minutes_rosemary + minutes_peppermint
    })
anova_data
anova_data.groupby(['Odor']).describe()
from scipy.stats import f_oneway
# 각 그룹의 데이터를 추출
lavender = anova_data[anova_data['Odor'] == 'Lavender']['Minutes']
rosemary = anova_data[anova_data['Odor'] == 'Rosemary']['Minutes']
peppermint = anova_data[anova_data['Odor'] == 'Peppermint']['Minutes']
# 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')

# 가정 체크
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Minutes ~ C(Odor)',
data=anova_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
anova_results
model.resid # 잔차 뽑기
# 잔차 분포 확인(0을 중심으로) / 잔차 퍼짐정도 확인
plt.scatter(model.fittedvalues, model.resid)
plt.show()
# 정규성 검정 (샤피로 and qqplot)
import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')
# bartlett 검정 : 등분산검정
from scipy.stats import bartlett
groups = ['Lavender', 'Rosemary', 'Peppermint']
grouped_residuals = [model.resid[anova_data['Odor'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
print(f"검정통계량: {test_statistic}, p-value: {p_value}")



# 사후검정 : 분산분석 후 검정결과가 유의미할 경우, 세부적으로 어느 
#           그룹간 평균 차이가 유의미한지 추가검정할 때 사용
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
                          endog=anova_data['Minutes'],
                          groups=anova_data['Odor'],
                          alpha=0.05) # 전체의 유의수준
# 이렇게 돌리면 각각 5/3 = 1.667 으로 알아서 계산함
print(tukey)
# reject가 True다 -> 평균에 유의미한 차이가 있다 (귀무가설을 기각한다)