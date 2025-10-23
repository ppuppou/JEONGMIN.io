import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

numbers = np.random.choice(np.arange(1, 11),
                            size=5,
                            replace=False) # 비복원추출
print(numbers)

from scipy.stats import binom, bernoulli, poisson, expon, norm, uniform

# Var(x) = E(X**2) - (E(X))**2

# Q1
good = 0.75
bad = 0.25
2 * bad/(good + (2 * bad)) # 3번

# Q2
y2022 = 0.16 * 0.05
y2021 = 0.18 * 0.02
y2020 = 0.20 * 0.03
y2022/(y2022 + y2021 + y2020) # 2번

year = np.array([0.16,0.18,0.20])
bug = np.array([0.05,0.02,0.03])
total = (year * bug).sum()
posterior = year * bug / total

# 균일분포, 베르누이, 이항분포, 정규분포, 포아송, 지수  확률변수 6가지

x = np.array([0,1,2])
y = np.array([1/4,1/2,1/4])
plt.scatter(x,y)
plt.ylim(0, 1)  # y축 범위를 0에서 1로 설정
plt.xticks([0, 1, 2])
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Probability Distribution")
plt.grid(axis='y',linestyle='--',alpha=0.5)
plt.show()

x = np.array([0,1,2])
y = np.array([0.36,0.48,0.16])
plt.scatter(x,y)
plt.ylim(0, 1)  # y축 범위를 0에서 1로 설정
plt.xticks([0, 1, 2])
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Probability Distribution")
plt.grid(axis='y',linestyle='--',alpha=0.5)
plt.show()
