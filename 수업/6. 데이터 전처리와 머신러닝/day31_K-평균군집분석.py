import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head())

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), 
                        columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 4, 
                random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)


# 팔머 펭귄 데이터를 bill_length, bill_depth 변수사용
# kmeans 알고리즘으로 3개 그룹으로 분류해보세요.

from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
df=penguins[["bill_length_mm", "bill_depth_mm"]]

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), 
                        columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 3, 
                random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)

penguins["labels"] = labels
penguins

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=penguins,
                x='bill_depth_mm', y='bill_length_mm',
                hue='labels', palette='deep',
                edgecolor='w', s=50)

sns.scatterplot(data=penguins,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep',
                edgecolor='w', s=50)



# k-means
import numpy as np

x1 = np.array([2, 2, 3, 5, 6, 6])
x2 = np.array([2, 4, 3, 5, 4, 5])

# 1) 랜덤하게 그룹의 중심을 선택함.
gr_1 = np.array([3, 6], dtype=float)
gr_2 = np.array([5, 2], dtype=float)


# 2) 각 데이터 포인트에서 각 그룹 중심까지 거리 계산
# gr_1까지 거리
gr_1_dist=np.sqrt((x1 - gr_1[0])**2 + (x2 - gr_1[1])**2)
gr_1_dist

# gr_2까지 거리
gr_2_dist=np.sqrt((x1 - gr_2[0])**2 + (x2 - gr_2[1])**2)
gr_2_dist

# 그룹 할당
labels = (gr_1_dist > gr_2_dist) + 1

# 3) 그룹 중심점 업데이트
# gr_1 중심점 업데이트
gr_1[0] = x1[labels == 1].mean()
gr_1[1] = x2[labels == 1].mean()

# gr_2 중심점 업데이트
gr_2[0] = x1[labels == 2].mean()
gr_2[1] = x2[labels == 2].mean()

# 4) 2, 3번 반복

# 2) 각 데이터 포인트에서 각 그룹 중심까지 거리 계산
# gr_1까지 거리
gr_1_dist=np.sqrt((x1 - gr_1[0])**2 + (x2 - gr_1[1])**2)
gr_1_dist

# gr_2까지 거리
gr_2_dist=np.sqrt((x1 - gr_2[0])**2 + (x2 - gr_2[1])**2)
gr_2_dist

# 그룹 할당
labels = (gr_1_dist > gr_2_dist) + 1

# 3) 그룹 중심점 업데이트
# gr_1 중심점 업데이트
gr_1[0] = x1[labels == 1].mean()
gr_1[1] = x2[labels == 1].mean()

# gr_2 중심점 업데이트
gr_2[0] = x1[labels == 2].mean()
gr_2[1] = x2[labels == 2].mean()


import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
penguins = load_penguins()
penguins = penguins.dropna()
df = penguins[["bill_length_mm", "bill_depth_mm"]]
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), 
                        columns=numeric_data.columns)

# 실루엣 계수 계산
scores = []
K_range = range(2, 6)   # silhouette_score는 k=1에서 계산 불가능 (클러스터가 1개면 분리도 정의 불가)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=1)
    labels = kmeans.fit_predict(df_trans)
    score = silhouette_score(df_trans, labels)
    scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

plt.plot(list(K_range), scores, marker='o')
plt.title("실루엣 계수에 따른 최적 군집 수 탐색")
plt.xlabel("군집 수 (k)")
plt.ylabel("평균 실루엣 계수")
plt.xticks(list(K_range))
plt.grid(True)
plt.show()