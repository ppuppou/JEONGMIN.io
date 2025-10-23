from palmerpenguins import load_penguins
penguins = load_penguins()
penguins = penguins.dropna()
penguins_a = penguins.loc[penguins['species'] != 'Adelie']
X = penguins_a[['bill_length_mm','bill_depth_mm']]
y = penguins_a['species']

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=penguins_a,
    x="bill_length_mm", 
    y="bill_depth_mm", 
    hue="species",      # 색상 구분
    style="species",    # 모양 구분
    markers={"Gentoo": "s", "Chinstrap": "o"},  # 사각형 vs 원형
    palette={"Gentoo": "steelblue", "Chinstrap": "darkorange"},
    s=100
)
plt.title("Penguin Bill Measurements (Gentoo vs Chinstrap)")
plt.show()

left_X = X.loc[X['bill_depth_mm']<16.5,:]
right_X = X.loc[X['bill_depth_mm']>=16.5,:]
left_y = y.loc[X['bill_depth_mm']<16.5]
right_y = y.loc[X['bill_depth_mm']>=16.5]

n = len(left_y)
m = len(left_y.loc[left_y == 'Chinstrap'])
GI_under = 1 - (m/n)**2 - ((n-m)/n)**2 # 0.017697704081632626
o = len(right_y)
p = len(right_y.loc[right_y == 'Chinstrap'])
GI_over = 1 - (p/o)**2 - ((o-p)/o)**2 # 0.19057777777777776

GI_total = (GI_under * n + GI_over * o)/(n+o)




from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.dropna(inplace=True)
penguins_a = penguins.loc[penguins['species']!='Adelie']
X = penguins_a[['bill_length_mm']]
y = penguins_a.species
import numpy as np
from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier(criterion='gini')
dct.get_params()
dct_params = {'max_depth' : np.arange(1,8),
              'ccp_alpha' : np.linspace(0,1,5)}
# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=2025)
# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='accuracy')
dct_search.fit(X, y)
dct_search.best_params_
dct_search.predict(X)
dct.predict_proba(X)