import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform
import scipy.stats as sp
import seaborn as sns
from scipy.stats import t, binom, ttest_1samp, ttest_ind, ttest_rel, kstest, anderson, shapiro
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from statsmodels.distributions.empirical_distribution import ECDF

# 1
8*7*6
# 2
from scipy.special import comb
comb(12,5)
# 3
1-1/12-1/6-1/2
# 4
0.5*0.8 + 0.3*0.5 + 0.2*0.9
# 5
3*4 - 2*2 + 7
9*5+4*3
# 6
(0.95*0.01)/(0.95*0.01+0.99*0.05)
# 7
# 평균 > 중앙값 > 최빈값
# 8
# 3번
# 9
# 2
# 10
0.75**2 - 0.25**2
# 11
((1.5**2)/4)-(1/4)
# 12
# 0.3
# 13
0.2+1+0.9
0.2+2.0+2.7
4.9-2.1**2
# 14
1-0.5**4
# 15
error = 1-binom.cdf(1,20,0.02)
binom.pmf(1,3,error)
# 16
uniform(0,4).var()
# 17
expon(scale = 1/3).mean()
expon(scale = 1/3).var()
# 18
norm.cdf(4.95,5,0.05)
# 19
norm.cdf(6,8,2)
# 20
norm.ppf(0.9,32,6)
# 21
#  obesity type iii 그룹의 경우 우측으로 긴꼬리를 갖는 형태이다.
# 22
# 20명
# 23
data = np.array([21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26])
len(data)
data.sort()
np.quantile(data,[0.25,0.5,0.75])
np.median(data) # 21.5
np.median(data[data<21.5]) # 17.5
np.median(data[data>21.5]) # 25.5
data[data <(17.5-8)]
data[data >(25.5+8)]
# 24
x = norm(3,2)
x.ppf(0.75) - x.ppf(0.25)
# 25
# 교수들의 평균 연봉은 $50,221과 같다.
# 26
# 표본 평균과 기준값 간 차이를 비교하는 일표본 t-검정
# 27
dat = pd.read_csv('./test data/datasetSalaries.csv')
dat['salary']
n = len(dat['salary'])
mean = dat['salary'].mean()
se = dat['salary'].std(ddof=1)/np.sqrt(n)
ttest_1samp(dat['salary'],popmean=50221,alternative='two-sided')
# 28
line_a = [2011, 2005, 1998, 2003, 2008, 2001, 2006]
line_b = [1985, 1991, 1988, 1992, 1986, 1990, 1987]
line_c = [2020, 2024, 2019, 2026, 2023, 2025, 2022]
s, p = shapiro(line_a) # 0.99
s1, p1 = shapiro(line_b) # 0.74
s2, p2 = shapiro(line_c) # 0.83
print(p,p1,p2)
# 29
drug_a = [142.9, 140.6, 144.7, 144.0, 142.4, 146.0, 149.1, 150.4]
drug_b = [139.1, 136.4, 147.3, 139.4, 143.0, 142.2, 142.2, 147.9]
np.var(drug_a)*1.5
np.var(drug_b)
ttest_rel(drug_b,drug_a,alternative='two-sided')
# 30
call = [1.2, 0.9, 1.5, 2.1, 0.7, 0.8, 1.8, 2.2, 1.0, 1.3, 2.5, 2.0, 1.1, 1.6, 0.6]
result = anderson(call, dist='expon')
print('검정통계량',result[0], '\n', 
      '임계값:',result[1], '\n',
      '유의수준:',result[2])