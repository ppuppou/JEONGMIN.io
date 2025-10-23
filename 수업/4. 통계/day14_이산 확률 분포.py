import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import uniform

# 베르누이 분포 X ~ B(p). p는 성공 확률(0~1)
# 베르누이 확률변수는 0과 1만 가실 수 있음.
from scipy.stats import bernoulli
rv = bernoulli(0.7) # p는 성공확률
# 확률 질량 함수(PMF)
rv.pmf(1)
rv.pmf(0)
# 랜덤 샘플 생성
samples = rv.rvs(size=10)
samples
rv.mean() # 베르누이 분포의 평균은 p
rv.var() # 분산은 pq


# 이항분포 확률변수 X ~B(n,p). n은 시행횟수, p는 성공 확률(0~1)
from scipy.stats import binom
x = binom(5,0.3)
x.mean() # 이항분포의 평균 = np
x.var()  # 이항분포의 분산 = npq

x = np.arange(0, 6)  # 가능한 값: 0부터 n까지
pmf = binom.pmf(x, 5, 0.3) # PMF 계산
# 시각화
plt.figure(figsize=(8, 5))
plt.stem(x, pmf, basefmt=" ")
plt.xlabel('Number of Successes (k)')
plt.ylabel('P(X = k)')
plt.title('Binomial PMF (n=5, p=0.3)')
plt.grid(True)
plt.show()

#### 조합 계산 코드
from scipy.special import comb
comb(5,3) * 0.3**3 * 0.7**2

# 베르누이 분포와 이항분포의 상관관계
Y = bernoulli(0.3)
sum(Y.rvs(5)) # == B(5,0.3)

# 포아송 분포 X ~ Pois(mu) mu는 단위시간/공간 당 평균 이벤트 발생 횟수
from scipy.stats import poisson
x = poisson(2)
x.pmf(3) # 이벤트가 3(정수)번 발생할 확률
x.mean() # 기대값은 mu와 같음
x.var()  # 분산도 mu와 같음