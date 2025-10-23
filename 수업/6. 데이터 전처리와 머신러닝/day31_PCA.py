import pandas as pd
from palmerpenguins import load_penguins

# 1. 데이터 불러오기 및 전처리
penguins = load_penguins()
penguins = penguins.dropna()
penguins = penguins[penguins["species"] == "Adelie"]
df = penguins[["bill_length_mm", 
               "bill_depth_mm", 
               "body_mass_g"]]
df

import seaborn as sns
sns.pairplot(df, kind="scatter",
             diag_kind="hist");

df.corr()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data = pd.DataFrame(scaled_data, 
                           columns=df.columns)

scaled_data


from sklearn.decomposition import PCA
pca = PCA(n_components=3)

pca_array = pca.fit_transform(scaled_data)
my_pca = pd.DataFrame(pca_array,
                      index = scaled_data.index,
                      columns=["pc1", "pc2", "pc3"])
my_pca.shape
my_pca.corr()
my_pca.cov()

my_pca["pc1"].var(ddof=1)
my_pca["pc2"].var(ddof=1)
my_pca["pc3"].var(ddof=1)

pca.explained_variance_.round(3)
pca.components_

# 어떻게 PC들을 만들었나?
# 1) 스케일된 데이터의 공분산행렬 계산
# 2) 행렬 분해(아이겐벨류 디컴포지션) 적용
#  => 아이겐벨류, 아이겐벡터 두개 결과값이 나옴
scaled_data.cov(ddof=1)

from numpy import linalg
import numpy as np
eig_values, eig_vectors = linalg.eig(scaled_data.cov(ddof=1))
np.sqrt(eig_values[0] / eig_values[2])
eig_vectors