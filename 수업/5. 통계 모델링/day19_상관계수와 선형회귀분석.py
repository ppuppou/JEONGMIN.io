import numpy as np
import pandas as pd
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt

# 1. Iris 데이터 로드
df_iris = load_iris()
# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시
# 3. 타겟(클래스) 추가
iris["Species"] = df_iris.target
# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["Species"] = iris["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
iris.info()
iris['Species'].value_counts()

import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols("Petal_Length ~ Petal_Width + C(Species)", data = iris).fit()
print(model.summary())

new_data = pd.DataFrame({
    'Petal_Width' : [0.5],
    "Species" : ['virginica']
})
predictions = model.predict(new_data)


sm.stats.anova_lm(model)

model.params

intercept = model.params['Intercept']
coef_pw = model.params['Petal_Width']
coef_sl = model.params['Sepal_Length']

# 3. 3D 산점도
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# 품종별 색상 지정
colors = {"setosa":"red", "versicolor":"green", "virginica":"blue"}
for species, group in iris.groupby("Species"):
    ax.scatter(group["Petal_Width"], group["Sepal_Length"], group["Petal_Length"], 
               color=colors[species], label=species, alpha=0.6)

# 4. 회귀평면 그리기
x_surf, y_surf = np.meshgrid(
    np.linspace(iris['Petal_Width'].min(), iris['Petal_Width'].max(), 30),
    np.linspace(iris['Sepal_Length'].min(), iris['Sepal_Length'].max(), 30)
)
z_surf = intercept + coef_pw * x_surf + coef_sl * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, color='yellow', alpha=0.4, rstride=100, cstride=100)

# 5. 시각화 옵션
ax.set_xlabel("Petal Width")
ax.set_ylabel("Sepal Length")
ax.set_zlabel("Petal Length")
ax.set_title("3D Regression Plane: Petal_Length ~ Petal_Width + Sepal_Length")
ax.legend()
plt.show()


# 연습
import pandas as pd
import numpy as np
import scipy.stats as stats
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())

np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
train_data = penguins.iloc[train_index].dropna()
train_data

plt.figure(figsize=(7,5))
plt.scatter(train_data["bill_depth_mm"], train_data["bill_length_mm"], 
            alpha=0.7, color="steelblue", edgecolor="k")
plt.xlabel("Bill Depth (mm)")
plt.ylabel("Bill Length (mm)")
plt.title("Penguin Bill Length vs Depth (Train Data)")
plt.show()

x = train_data["bill_depth_mm"]
y = train_data["bill_length_mm"]
corr, p_value = stats.pearsonr(x,y)

model = smf.ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
print(model.summary())

plt.figure(figsize=(7,5))
plt.scatter(train_data["bill_depth_mm"], train_data["bill_length_mm"], 
            alpha=0.7, color="steelblue", edgecolor="k", label="Data")

x_vals = np.linspace(train_data["bill_depth_mm"].min(), train_data["bill_depth_mm"].max(), 100)
y_vals = model.params["Intercept"] + model.params["bill_depth_mm"] * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Bill Depth (mm)")
plt.ylabel("Bill Length (mm)")
plt.title("Penguin Bill Length vs Depth (with Regression Line)")
plt.legend()

plt.show()

model.params


import seaborn as sns
model2 = smf.ols('bill_length_mm ~ bill_depth_mm + species', data=train_data).fit()
print(model2.summary())
# model 2가 더 좋은 모델이라는 증거
table = sm.stats.anova_lm(model, model2)
table

model2.params

print(model2.summary())  # 회귀 결과 확인
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=train_data,
    x='bill_depth_mm',
    y='bill_length_mm',
    hue='sex',        # 성별에 따라 점 색깔
    style='species'   # 종(species)에 따라 점 모양
)
sns.lmplot(
    data=train_data,
    x='bill_depth_mm',
    y='bill_length_mm',
    hue='species',    # 종별 회귀 직선
    col='sex',        # 성별별로 나눠서 그리기 (선택사항)
    height=5,
    aspect=1
)
plt.show()