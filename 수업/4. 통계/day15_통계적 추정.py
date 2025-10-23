import numpy as np
from scipy.stats import norm
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

samples = norm.rvs(loc=5, scale=3, size=10)
print("표본 10개:", samples)
samples.mean()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# 설정
dist = uniform(3, 4)  # U(3, 7)
mu = dist.mean()
n = 20
n_trials = 1000
confidence = 0.95
alpha = 1 - confidence

# 정규분포 임계값
z_crit = norm.ppf(1 - alpha / 2)

# 신뢰구간 및 포함 여부 저장
contains_mu = []
ci_bounds = []  

for _ in range(n_trials):
    sample = dist.rvs(size=n)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=0)  # 모표준편차처럼 계산

    margin = z_crit * (sample_std / np.sqrt(n))
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin

    contains_mu.append(ci_lower <= mu <= ci_upper)
    ci_bounds.append((ci_lower, ci_upper))  # 신뢰구간 저장

coverage_rate = np.mean(contains_mu)
print(f"모평균 포함 비율 (정규근사): {coverage_rate:.3f}")
#region 시각화 
# 시각화
plt.figure(figsize=(10, 6))
for i, ((low, high), covered) in enumerate(zip(ci_bounds[:100], contains_mu[:100])):
    color = 'green' if covered else 'red'
    plt.plot([low, high], [i, i], color=color)
    plt.plot([mu], [i], 'ko', markersize=2)  # 모평균 점
plt.axvline(mu, color='black', linestyle='--', label='모평균 μ')
plt.title(f"100개의 신뢰구간 시각화 (모평균 포함: {coverage_rate:.3f})")
plt.xlabel("신뢰구간")
plt.ylabel("표본 번호")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#endregion


dist = uniform(3, 4)  # 균등분포 U(3, 7)
mu = dist.mean()      # 모평균
sigma = dist.std()    # 모표준편차
n = 20                # 표본 크기
# 이론적 표본평균의 분포는 정규분포로 근사 (중심극한정리)
mean_sampling_dist = mu
std_sampling_dist = sigma / np.sqrt(n)
# 이론적 분포 PDF
x = np.linspace(mu - 4 * std_sampling_dist, mu + 4 * std_sampling_dist, 500)
pdf = norm.pdf(x, loc=mean_sampling_dist, scale=std_sampling_dist)
# 표본 추출
sample = dist.rvs(size=n)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=0)
# 신뢰구간 계산 (정규 근사)
z_crit = norm.ppf(0.975)  # 95% 신뢰수준
margin = z_crit * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin
#region 시각화
# 시각화
plt.figure(figsize=(10, 6))
# 이론적 표본평균의 분포
plt.plot(x, pdf, label='이론적 표본평균 분포 (정규근사)', color='blue')
# 모평균 위치 표시
plt.axvline(mu, color='red', linestyle='-', linewidth=2, label='모평균 (빨간선)')
# 표본 데이터: 녹색 점
plt.plot(sample, np.zeros_like(sample), 'go', label='표본 데이터 (녹색 점)')
# 신뢰구간: 녹색 수직선 2개
plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label='95% 신뢰구간')
plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2)
# 꾸미기
plt.title("표본평균의 분포와 신뢰구간")
plt.xlabel("값")
plt.ylabel("확률 밀도 (PDF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#endregion

# 신뢰구간 안에 포함된 표본평균 개수 계산
within_count = sum(ci_lower <= mean <= ci_upper for mean in sample_mean)
within_percent = within_count / n_trials * 100

# 평균 신뢰구간 (optional 참고용)
avg_mean = np.mean(sample_mean)
avg_margin = np.mean(margin)
ci_lower = avg_mean - avg_margin
ci_upper = avg_mean + avg_margin
#region 시각화
# 시각화
plt.figure(figsize=(10, 6))
# 이론적 분포
plt.plot(x, pdf, label='이론적 표본평균 분포 (정규근사)', color='blue')
# 모평균 표시
plt.axvline(mu, color='red', linestyle='-', linewidth=2, label='모평균')
# 모든 표본평균을 녹색 점으로 표시
plt.plot(sample_mean, np.zeros_like(sample_mean), 'go', label='1000개 표본평균')
# 평균 신뢰구간 표시 (시각적 참고)
plt.axvline(ci_lower, color='green', linestyle='--', linewidth=2, label='평균 신뢰구간 (95%)')
plt.axvline(ci_upper, color='green', linestyle='--', linewidth=2)
# 각주(주석) 추가
plt.annotate(f'{within_count}개 ({within_percent:.1f}%)의 표본평균이 95% 신뢰구간에 포함됨',
             xy=(0.5, -0.15), xycoords='axes fraction',
             ha='center', fontsize=11)
# 꾸미기
plt.title("1000개의 표본평균 분포 + 모평균 + 신뢰구간")
plt.xlabel("표본평균")
plt.ylabel("확률 밀도 (PDF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#endregion


# 신뢰구간 구하는 방법 1
x = norm(x.mean(),sigma)
x.ppf(0.05)
x.ppf(0.95)
# 신뢰구간 구하는 방법 2
z_05 = norm.ppf(0.05,loc=0,scale=1)
x.mean() + z_05 * (3/np.sqrt(n))
x.mean() - z_05 * (3/np.sqrt(n))

# 모든 확률변수는 평균을 빼고 표준편차로 나누면 표준정규분포를 따름

# t분포
from scipy.stats import t
df = 5 # 자유도 설정
x = np.linspace(-5, 5, 1000) # x 값 범위 설정
pdf = t.pdf(x, df) # t 분포 pdf 계산
norm_pdf = norm.pdf(x, 0, 1) # 표준 정규분포 pdf 계산
# region 시각화
plt.plot(x, pdf, label=f't-distribution (df={df})', color='blue')
plt.plot(x, norm_pdf, label='Standard Normal Distribution', color='red', linestyle='--')
plt.title('t-Distribution vs Standard Normal Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
# endregion