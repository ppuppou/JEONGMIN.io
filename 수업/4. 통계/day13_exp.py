import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform, binom, t, ttest_1samp

# Q1
X = np.array([1,2,3])
Prob = np.array([0.2,0.5,0.3])
exp_X = (X * Prob).sum().round(3)
exp_X

# Q2
var_X = (((X - exp_X)**2) * Prob).sum()
var_X

# Q3
Y = 2 * X + 3
exp_Y = (Y * Prob).sum().round(3)
exp_Y

# Q4
var_Y = (((Y - exp_Y)**2) * Prob).sum()
var_Y

# Q5
X = np.array([0,1,2,3])
prob = np.array([0.1,0.3,0.4,0.2])
exp_X = (X * prob).sum().round(3)
exp_X
var_X = (((X - exp_X)**2) * prob).sum()
var_X

# Q8
2 * 5 - 3 + 4

# Q9
# 기대값 : aμ + b , 분산 : a^2 * σ^2

# Q10
# p = 0.3
X = np.array([1,2,3])
Prob = np.array([0.3,0.3,0.4])
exp_X = (X * Prob).sum().round(3)
exp_X

# Q11
X = np.array([1,2,4])
Prob = np.array([0.2,0.5,0.3])
exp_X = (X * Prob).sum().round(3)
exp_X
exp_X_square = (X**2 * Prob).sum().round(3)
exp_X_square
var_X = (((X - exp_X)**2) * Prob).sum()
var_X  ## var(X) = E(X²)-(E(x))²

# [연습문제] 확률질량함수, 확률밀도함수, 누적분포함수
# 1. F(2) = 0.8
# 2. p(2) = 0.4, p(x>2) = 0.2
# 3. c = 3
# 4.
0.5**3 - 0.2**3
# 5. F(x) = {(1/3)x (0<=x<=3),
#             1     (3<x)}
# 6. E(x) = 2.1, F(2) = 0.7
# 7. f(x) = 1/2 (단, 1<=x<=3)
# 8. 3/5
# 9. 0.7
# 10. 0.4
