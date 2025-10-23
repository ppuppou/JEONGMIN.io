import seaborn as sns
import pandas as pd
import numpy as np
df = sns.load_dataset('titanic')
df = df.dropna(ignore_index=True)

df.head()
df.shape
df.groupby('sex')['age'].sum()

test2 = df.loc[df['sex']=='male',['age','fare']]
test3 = test2.loc[(test2['age']>=40) & (test2['age']<50),:]
test3['fare'].mean()

x = np.array([2,4,1,7,7,8]).reshape(3,2)
y = np.array([10,5,15]).reshape(3,1)
((x @ np.linalg.inv(x.T @ x)) @ x.T) @ y

np.random.seed(2025)
array_2d = np.random.randint(1, 13, 200).reshape((50, 4))
array_2d[:4,:]
mean_test = array_2d.mean(axis=1).max()

dffe = array_2d.max(axis=1)-array_2d.min(axis=1)
dffe.sum()