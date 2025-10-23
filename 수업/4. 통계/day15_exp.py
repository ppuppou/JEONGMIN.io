import numpy as np
from scipy.stats import norm, uniform, t, expon
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 정규분포와 t분포
# 1
x = t(9)
se = 4/np.sqrt(10)
x.cdf(1.812)

# 2
x = t(14)
x.ppf(0.95)

# 3
x = t(11)
1 - x.cdf(2.18)

# 4
x = norm(12,1.5)
1 - x.cdf(10)

# 5
x = norm(8,0.8)
1 - x.cdf(9)

# 6-1
data = [84.3, 85.7, 83.9, 86.1, 84.5, 85.2, 85.8, 86.3, 84.7, 85.5]
mean = np.mean(data)
std = np.std(data, ddof=1)
se = std/np.sqrt(10)
x = t(9)
x.ppf(0.975)
mean - x.ppf(0.975) * se
mean + x.ppf(0.975) * se

# 균일분포와 지수분포
# 1
expon.cdf(2, scale = 2)

# 2
1 - expon.cdf(1, scale = 0.5)

# 3
x = expon(1/3)
x.mean() ; x.var()

# 4
uniform.cdf(4, loc=2, scale=3) - uniform.cdf(3, loc=2, scale=3) 

# 5
uniform(loc=0, scale=8).mean()
uniform(loc=0, scale=8).var()

# 6
1 - expon.cdf(5, scale=10)

# 7 
expon.cdf(3, scale=6)

# 8
1 - expon.cdf(10, scale=12)

# 9 
expon.cdf(2, scale=5)

# 10
expon.cdf(3, scale=2)

# 11
# region
import matplotlib.pyplot as plt
lambda_val = 2
scale_val = 1 / lambda_val  # scale = 1/λ
# x 값 범위 설정
x = np.linspace(0, 3, 300)
# 지수분포 PDF 계산
pdf = expon(scale=scale_val).pdf(x)
# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label=r'$\lambda = 2$', color='blue')
plt.title('지수분포 (Exponential Distribution) PDF', fontsize=14)
plt.xlabel('x')
plt.ylabel('확률 밀도 (PDF)')
plt.grid(True)
plt.legend()
plt.show()
# endregion

# 12
# region
a = 2  # 최소값
b = 6  # 최대값
loc = a
scale = b - a  # 범위
# x 값 범위 설정
x = np.linspace(0, 8, 400)
pdf = uniform(loc=loc, scale=scale).pdf(x)
# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label=r'$U(2,6)$', color='green')
plt.title('균일분포 (Uniform Distribution) PDF', fontsize=14)
plt.xlabel('x')
plt.ylabel('확률 밀도 (PDF)')
plt.grid(True)
plt.axvline(2, color='gray', linestyle='--', alpha=0.7)
plt.axvline(6, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
# endregion

# 13
1 - uniform.cdf(7, loc=0, scale=10)

# 14
1 - expon.cdf(6, scale=4)
