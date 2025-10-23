import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

values = np.array([0, 1, 2])
probabilities = np.array([0.36, 0.48, 0.16])

# 확률변수 x 생성
X = np.random.choice(values, size=333, p=probabilities)
X
# 예시: 분포 확인
unique, counts = np.unique(X, return_counts=True)
empirical_distribution = dict(zip(unique, counts / 10))
print("샘플된 x의 경험적 분포:", empirical_distribution)

# 확률변수의 기대값 = 확률 분포의 무게중심 (가운데 지점)

# 이산확률변수의 기대값
exp_X = (values * probabilities).sum()
exp_X
# 표본평균 
X.mean()

values = np.array([1, 2, 3, 4])
probabilities = np.array([0.1, 0.3, 0.2, 0.4])
exp_X = np.sum(values * probabilities).round(3)
exp_X
X = np.random.choice(values, size= 1000, p=probabilities)
X.mean()
# 히스토그램화
plt.hist(X, bins=np.arange(0.5, 5.5, 1), 
         density=True, # count가 아닌 확률로 (0~1)
         edgecolor='black', rwidth=0.8, alpha=0.6)
plt.xticks(values)
plt.title('히스토그램: 확률변수 X의 분포')
plt.xlabel('값')
plt.ylabel('상대도수')
plt.axvline(exp_X, color='red', linestyle='dashed', linewidth=2,
            label=f'E[X] = {exp_X}')
# 이론적 확률 막대 (보라색 선) 추가
for val, prob in zip(values, probabilities):
    plt.vlines(val, 0, prob, colors='purple',
               linewidth=3, label='이론 확률' if val == values[0] else "")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 표본분산과 모분산
values = np.array([1, 2, 3, 4])
probabilities = np.array([0.1, 0.3, 0.2, 0.4])
exp_X = np.sum(values * probabilities).round(3)
exp_X
Var_X = (((values - exp_X)**2) * probabilities).sum()
Var_X
X = np.random.choice(values, size= 500, p=probabilities)
X.var(ddof=1) # ddof=1 : n-1로 나누기 (불편추정량 표본분산(모분산 추정))
# np.sum((X-X.mean())**2) / (300-1) 
X.var() # n으로 나누기 (표본의 분산)

n_iterations = 1000
# 결과 저장용 리스트
sample_vars = []       # 표본 분산 (ddof=1)
population_vars = []   # 모분산 (ddof=0)
# 반복 수행
for _ in range(n_iterations):
    X = np.random.choice(values, size=30, p=probabilities)
    sample_vars.append(X.var(ddof=1))
    population_vars.append(X.var())
# 이론 분산값
theoretical_var = 1.09
# 히스토그램 그리기
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# 불편추정 표본분산 히스토그램
axs[0].hist(sample_vars, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axs[0].axvline(np.mean(sample_vars), color='green', linestyle='solid', linewidth=2, label=f'표본평균: {np.mean(sample_vars):.3f}')
axs[0].axvline(theoretical_var, color='red', linestyle='solid', linewidth=2, label='이론 분산: 1.09')
axs[0].set_title('불편추정 표본분산 (ddof=1)')
axs[0].set_xlabel('분산 값')
axs[0].set_ylabel('빈도')
axs[0].legend()
axs[0].grid(True)
# 표본분산 히스토그램
axs[1].hist(population_vars, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
axs[1].axvline(np.mean(population_vars), color='green', linestyle='solid', linewidth=2, label=f'표본평균: {np.mean(population_vars):.3f}')
axs[1].axvline(theoretical_var, color='red', linestyle='solid', linewidth=2, label='이론 분산: 1.09')
axs[1].set_title('표본분산 (ddof=0)')
axs[1].set_xlabel('분산 값')
axs[1].set_ylabel('빈도')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
plt.show()

# 결과 저장
sample_vars = []       # 불편추정 표본분산 (ddof=1)
population_vars = []   # 표본분산 (ddof=0)
# 반복 수행
for _ in range(n_iterations):
    X = np.random.choice(values, size=500, p=probabilities)
    sample_vars.append(X.var(ddof=1))
    population_vars.append(X.var())
# 이론 분산
theoretical_var = 1.09
# 오차 계산
sample_errors = np.abs(np.array(sample_vars) - theoretical_var)
population_errors = np.abs(np.array(population_vars) - theoretical_var)
# =======================
# 1. 오차 히스토그램
# =======================
plt.figure(figsize=(12, 5))
plt.hist(sample_errors, bins=30, color='blue', alpha=0.6, label='표본 분산 오차 |표본 - 이론|')
plt.hist(population_errors, bins=30, color='orange', alpha=0.6, label='모분산 오차 |모 - 이론|')
plt.axvline(np.mean(sample_errors), color='green', linestyle='--', linewidth=2, label=f'표본 오차 평균: {np.mean(sample_errors):.4f}')
plt.axvline(np.mean(population_errors), color='red', linestyle='--', linewidth=2, label=f'모 오차 평균: {np.mean(population_errors):.4f}')
plt.title('이론 분산(1.09)으로부터의 오차 분포')
plt.xlabel('오차 크기')
plt.ylabel('빈도')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# =======================
# 2. 시행별 오차 꺾은선 그래프
# =======================
plt.figure(figsize=(14, 5))
plt.plot(sample_errors, label='불편추청 표 분산 오차', alpha=0.7)
plt.plot(population_errors, label='표본분산 오차', alpha=0.7)
plt.axhline(y=np.mean(sample_errors), color='green', linestyle='--', label=f'표본 오차 평균: {np.mean(sample_errors):.4f}')
plt.axhline(y=np.mean(population_errors), color='red', linestyle='--', label=f'모 오차 평균: {np.mean(population_errors):.4f}')
plt.title('각 시행에서 이론 분산(1.09)과의 오차 (절댓값)')
plt.xlabel('시행 번호')
plt.ylabel('오차 크기')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 균일분포 x ~ unif(a,b) a : 시작점 , b : 끝점
x = np.random.uniform(0, 1)
print(x)
# 균일분포의 평균 = a+b/2
# 균일분포의 분산 = (b-a)^2/12
from scipy.stats import uniform
a = 2
b = 4
x = uniform(loc=a, scale= b-a)
x.mean()
x.var()
x=x.rvs(size=100) # random variates (랜덤 샘플)
x.cdf(3.5) # = P(x<=3.5) 라는 뜻
x.cdf(3.2)-x.cdf(2.1) # P(2.1 < x <= 3.2)
# cdf : CumulativeDistribution Function, 확률누적분포

# 균일분포 pdf화 (probability density function, 확률밀도함수)
# x값 범위 설정
x_vals = np.linspace(a - 0.5, b + 0.5, 500)
pdf_vals = x.pdf(x_vals)
# CDF 값 계산
cdf_val = x.cdf(3.5)  # P(X <= 3.5)
# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(x_vals, pdf_vals, label='PDF of U(2, 4)', color='blue')
# 누적확률 영역 채우기: P(X ≤ 3.5)
x_fill = np.linspace(a, 3.5, 300)
plt.fill_between(x_fill, x.pdf(x_fill), color='orange', alpha=0.5, label=f'P(X ≤ 3.5) = {cdf_val:.2f}')
# 그래프 꾸미기
plt.title('균일분포 U(2, 4)의 확률밀도함수 및 P(X ≤ 3.5)')
plt.xlabel('x')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 지수분포
from scipy.stats import expon
# 지수분포 정의: theta = 0.5 → scale = 0.5
x = expon(scale=0.5)
# 이론 평균과 분산 확인
print("이론 평균:", x.mean())      # = 0.5 # 평균은 theta 값과 동일
print("이론 분산:", x.var())       # = 0.25 # 분산은 theta 값의 제곱
# 누적확률 예시: P(X ≤ 1)
print("P(X ≤ 1):", x.cdf(1))
# 표본 10개 추출

# x값 범위 설정
x_vals = np.linspace(-1, 5, 500)
pdf_vals = x.pdf(x_vals)
cdf_vals = x.cdf(x_vals)
# 그래프 그리기
plt.figure(figsize=(12, 5))
# PDF (확률밀도함수)
plt.subplot(1, 2, 1)
plt.plot(x_vals, pdf_vals, label='PDF', color='blue')
plt.title('지수분포 PDF (θ = 0.5)')
plt.xlabel('x')
plt.ylabel('밀도')
plt.grid(True)
plt.legend()
# CDF (누적분포함수)
plt.subplot(1, 2, 2)
plt.plot(x_vals, cdf_vals, label='CDF', color='green')
plt.title('지수분포 CDF (θ = 0.5)')
plt.xlabel('x')
plt.ylabel('누적 확률')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

x.ppf(0.2) # cdf가 0.2가 되는 x값을 찾아주는 퀀타일 함수


# 표준정규분포 N ~ (0,1)
from scipy.stats import norm
x = norm(loc=0, scale=1)
# 이론 평균과 분산 출력
print("이론 평균:", x.mean())    # 0
print("이론 분산:", x.var())     # 1
print('이론 표준편차:', x.std())
x_vals = np.linspace(-4, 4, 500)
pdf_vals = x.pdf(x_vals)
plt.figure(figsize=(8, 4))
plt.plot(x_vals, pdf_vals, label='PDF of N(0,1)', color='blue')
plt.title('정규분포 N(0, 1)의 확률밀도함수')
plt.xlabel('x')
plt.ylabel('밀도')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

x.cdf(1)-x.cdf(-1)


# 정규분포 N ~ (μ,σ^2)
from scipy.stats import norm
x = norm(loc=2, scale=3) # loc = μ, scale = σ
# 이론 평균과 분산 출력
print("이론 평균:", x.mean())  
print("이론 분산:", x.var())    
samples = x.rvs(size=300)
# PDF 위한 x값
x_vals = np.linspace(-8, 12, 500)
pdf_vals = x.pdf(x_vals)
# 히스토그램 + PDF 겹치기
plt.figure(figsize=(8, 4))
plt.hist(samples, bins=40,   # ✅ bin 수 늘려서 간격 좁게
    density=True, alpha=0.6, color='skyblue',
    edgecolor='black',       # ✅ 테두리 추가
    label='표본 히스토그램'
)
plt.plot(x_vals, pdf_vals, color='red', lw=2, label='이론 PDF (N(0,1))')
plt.title('정규분포 표본 히스토그램과 이론 PDF')
plt.xlabel('x')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
x_shade = np.linspace(x.ppf(0.9),12,200)
plt.fill_between(x_shade, x.pdf(x_shade), color='hotpink', 
                 alpha=0.9, label='상위 10% 영역')
plt.tight_layout()
plt.show()

x.cdf(5)-x.cdf(-1)
x.ppf(0.9)


