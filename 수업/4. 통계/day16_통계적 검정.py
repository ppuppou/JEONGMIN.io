import numpy as np
import pandas as pd
from scipy.stats import norm, uniform, t, expon, binom
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 복습
data = [4.3, 4.1, 5.2, 4.9, 5.0, 4.5, 4.7, 4.8, 5.2, 4.6]
mean = np.mean(data)
n = len(data)
se = np.std(data, ddof=1) / np.sqrt(len(data))
mean - t.ppf(0.975, n-1) * se
mean + t.ppf(0.975, n-1) * se
# interval
ci = t.interval(0.95, loc=mean, scale=se, df=n-1)
ci

# 통계적 검정
x = norm(17,2.21)
x.cdf(14)
# p-value : 귀무가설이 참이라는 전제 하에 
#           나의 결과보다 극단적 결과가 나올 확률 ( ~ cdf)

x = norm(80,5/np.sqrt(30))
1 - x.cdf(83)

# 문제 1
x = norm(500,5)
1 - x.cdf(510)

# 문제 2
x = binom(20, 0.05)
x.pmf(2)
x.cdf(2)
1-x.cdf(2)

# 문제 3
1 - norm.cdf(85, loc=75, scale=8)
norm.cdf(80, loc=75, scale=8) - norm.cdf(70, loc=75, scale=8)
norm.ppf(0.9,loc=75,scale=8)

# 문제 4
data = [72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1]
mean = np.mean(data)
# H0 = 75, Ha != 75
se = np.std(data, ddof=1) / np.sqrt(len(data))
(mean-75)/se
(t.cdf((mean-75)/se, df=len(data)-1)) * 2
# 귀무가설을 기각하지 못한다

# 문제 5
# H0 = 50, Ha != 50
mean = 53
se = 8/np.sqrt(40)
(1 - norm.cdf((mean - 50)/se, loc=0,scale=1))*2
# 귀무가설을 기각한다



# t 검정
# 단일표본 t검정
sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
from scipy.stats import ttest_1samp
ttest_1samp(sample, popmean=10, # 귀무가설
            alternative='two-sided') # 양측검정
                                     # 단측검정은 greater or less

# 독립2표본 t검정
sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["Female"]*7 + ["Male"]*5
my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
my_tab2
from scipy.stats import ttest_ind
male = my_tab2[my_tab2['gender'] == 'Male']
female = my_tab2[my_tab2['gender'] == 'Female']
ttest_ind(male['score'], female['score'], 
          equal_var=True, 
          alternative='greater') # 단측 검정 (앞쪽이 크냐는 뜻)

# 대응표본 t검정
before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])
from scipy.stats import ttest_rel
ttest_rel(after, before, alternative='greater')
# 혹은 아래처럼 단일표본 t검정으로도 가능
x = after - before
ttest_1samp(x, popmean=0, alternative='greater')
