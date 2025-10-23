import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 비교할 자유도 목록
dfs = [3]

# x 구간: 가장 큰 df 기준으로 99.9% 분위까지
x_max = chi2.ppf(0.999, max(dfs))
x = np.linspace(0, x_max, 1000)

plt.figure(figsize=(7, 5))
for df in dfs:
    pdf = chi2.pdf(x, df=df)
    plt.plot(x, pdf, label=f"df={df}")

plt.title("Chi-square PDF (scipy.stats.chi2)")
plt.xlabel("x")
plt.ylabel("density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


from scipy.stats import chi2
from scipy.stats import norm

X = chi2(df=3)
1-X.cdf(8)

Y = norm(loc=3, scale=2)
data_set = Y.rvs(500 * 15).reshape(500, -1)
s_2=data_set.var(ddof=1, axis=1)
statistics = s_2 * (15 - 1) / 2**2
statistics

# 5. 히스토그램 + 이론적 카이제곱 PDF
x = np.linspace(0, max(statistics), 500)
pdf = chi2.pdf(x, df=14)  # 자유도 = n-1 = 14

plt.figure(figsize=(7,5))
plt.hist(statistics, bins=30, density=True, alpha=0.6, label="Simulated")
plt.plot(x, pdf, "r-", lw=2, label="Chi2 PDF (df=14)")
plt.title("Sample variance → Chi-square distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()