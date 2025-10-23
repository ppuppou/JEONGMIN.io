import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform
import scipy.stats as sp
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 펭귄 종별 부리길이의 1,2,3사분위수
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins['species'].unique()
penguins.dropna(inplace=True)
Adelie = penguins[penguins['species']=='Adelie']['bill_length_mm'].sort_values()
Gentoo = penguins[penguins['species']=='Gentoo']['bill_length_mm'].sort_values()
Chinstrap = penguins[penguins['species']=='Chinstrap']['bill_length_mm'].sort_values()
adelie_m = Adelie.median()
gentoo_m = Gentoo.median()
chinstrap_m = Chinstrap.median()
result = {}
for name, series in [('Adelie', Adelie), ('Gentoo', Gentoo), ('Chinstrap', Chinstrap)]:
    result[name + ' 1분위'] = series[series < series.median()].median()
    result[name + '중앙값'] = round(series.median(),1)
    result[name + ' 3분위'] = series[series > series.median()].median()
result

# 이상치 판별법 : Q1, 3을 기준으로 양옆으로 1.5IQR 내에 없으면 이상치로 판별
# >> 박스플롯이라고 함
scores = np.array([88,92,95,91,87,89,94,90,92,100,43])
median = np.median(scores)
Q1 = np.median(scores[scores<91])
Q3 = np.median(scores[scores>91])


data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
sorted_data = np.sort(data)
n = len(data)
np.quantile(data, np.arange(0.01,1,0.01))
np.percentile(data, [25,50,75])
# 이 데이터가 정규분포를 따르는가?
x = norm(loc=data.mean(),scale=data.std(ddof=1))
x.ppf(0.25)
x.ppf(0.5)
x.ppf(0.75)
norm_q = x.ppf(np.arange(0.01,1,0.01))
data_q = np.quantile(data, np.arange(0.01,1,0.01))
# 1. 산점도로 비교 (qqplot)
plt.figure(figsize=(8, 6))
plt.scatter(norm_q, data_q, color='blue', label='Data Quantiles vs Normal Quantiles')
plt.plot(norm_q, norm_q, color='red', linestyle='--', label='y = x (Ideal if Normal)') 
plt.xlabel('Normal Quantiles')
plt.ylabel('Data Quantiles')
plt.title('Q-Q Plot: Data vs Normal Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# or
sp.probplot(data,dist='norm',plot= plt)
# 2. 검정으로 비교 (Shapiro-Wilk 검정, 데이터가 50개 이하일 때 특화
                   # 사실 현재는 5000개 정도까지도 사용 가능
                   # H0 = 데이터가 정규분포를 따른다)
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55,
3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9])
w, p_value = sp.shapiro(data_x)
print("W:", w, "p-value:", p_value)

# 내가 만약 t검정을 하고싶다!
# 1>> 정규분포를 따르는지 QQ or SW 를 이용해 검정
# 2>> 표본이 두개일 시 분산이 같은지 F검정 or 분산 계산 을 통해 검정
# 3>> 알맞은 t 검정 진행


# 데이터를 이용한 누적분포함수 그리기
from statsmodels.distributions.empirical_distribution import ECDF
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
                  11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
ecdf = ECDF(data_x)
x = np.linspace(min(data_x), max(data_x))
y = ecdf(x)
plt.plot(x,y,marker='o', linestyle='none')
plt.title("Estimated CDF")
plt.xlabel("X-axis")
plt.ylabel("ECDF")
# 이론적 누적분포함수
k = np.arange(min(data_x), max(data_x), 0.1)
plt.plot(k, norm.cdf(k, loc=np.mean(data_x),
         scale=np.std(data_x, ddof=1)), color='red')
plt.show()

# K-S test : 이론값과의 차이 중 최대값으로 정규성을 판단
from scipy.stats import kstest
loc = np.mean(data_x)          # 표본 평균과 표준편차로 정규분포 생성
scale = np.std(data_x, ddof=1)
# 정규분포를 기준으로 K-S 검정 수행
result = kstest(data_x, 'norm', args=(loc, scale))
print("검정통계량:", result.statistic)
print('p-value:', result.pvalue)

# A-D test : 이론값과의 차이를 제곱하고(넓이) 더해서 정규성을 판단
from scipy.stats import anderson
result = sp.anderson(data_x, dist='norm') # Anderson-Darling 검정 수행
print('검정통계량',result[0], '\n', 
      '임계값:',result[1], '\n',
      '유의수준:',result[2])